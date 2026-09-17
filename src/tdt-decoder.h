#pragma once
// tdt-decoder.h: Parakeet TDT prediction network, joint and greedy loop
//
// The transducer keeps two tiny graphs next to the encoder:
//
//   predict   embedding lookup, two LSTM layers of 640, output projection.
//             It advances only when a real token is emitted, so a blank
//             costs nothing on this side.
//   joint     relu of the encoder frame plus the prediction, then one
//             matrix to vocabulary plus durations. The two argmax land on
//             the device and come back as eight bytes.
//
// Token duration decoding is what makes the model fast: every step reads
// one encoder frame and predicts how many frames to skip next, so a long
// silence costs one step instead of one step per frame. A blank is forced
// to advance at least one frame, which is the only guard the reference
// keeps; max_symbols_per_step bounds the emissions at a single frame.
//
// Both graphs have fixed shapes, so they are built once into their arena.
// The LSTM state lives on the host between steps: 5 KB of copies per step,
// far below the cost of a graph rebuild.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "parakeet.h"
#include "sp-detok.h"

#include <string>
#include <vector>

struct ParakeetLstmLayer {
    struct ggml_tensor * w = nullptr;  // [d, 4d]
    struct ggml_tensor * r = nullptr;  // [d, 4d]
    struct ggml_tensor * b = nullptr;  // [4d]
};

struct ParakeetDecoderWeights {
    struct ggml_tensor *           embedding = nullptr;  // [d, vocab]
    std::vector<ParakeetLstmLayer> lstm;
    struct ggml_tensor *           proj_w = nullptr;     // [d, d]
    struct ggml_tensor *           proj_b = nullptr;     // [d]

    struct ggml_tensor * joint_w = nullptr;              // [d, vocab + n_durations]
    struct ggml_tensor * joint_b = nullptr;              // [vocab + n_durations]

    int              vocab_size  = 0;
    int              blank_id    = 0;
    int              d_decoder   = 0;
    int              n_layers    = 0;
    int              max_symbols = 10;
    std::vector<int> durations;
};

static void parakeet_load_decoder(ParakeetDecoderWeights * w, WeightCtx * wctx, const GGUFModel & gf) {
    w->vocab_size  = (int) gf_get_u32(gf, "asr.vocab_size");
    w->blank_id    = (int) gf_get_u32(gf, "asr.blank_id");
    w->d_decoder   = (int) gf_get_u32(gf, "asr.d_decoder");
    w->n_layers    = (int) gf_get_u32(gf, "asr.decoder_layers");
    w->max_symbols = (int) gf_get_u32(gf, "asr.max_symbols_per_step");

    const int64_t index = gguf_find_key(gf.gguf, "asr.durations");
    if (index < 0) {
        s2s_throw("[TDT] Key 'asr.durations' not found");
    }
    const int64_t   n_durations = gguf_get_arr_n(gf.gguf, index);
    const int32_t * values      = (const int32_t *) gguf_get_arr_data(gf.gguf, index);
    w->durations.assign(values, values + n_durations);

    w->embedding = gf_load_tensor(wctx, gf, "dec.embedding.weight");
    w->lstm.resize((size_t) w->n_layers);
    for (int i = 0; i < w->n_layers; i++) {
        const std::string p   = "dec.lstm." + std::to_string(i) + ".";
        w->lstm[(size_t) i].w = gf_load_tensor(wctx, gf, p + "w");
        w->lstm[(size_t) i].r = gf_load_tensor(wctx, gf, p + "r");
        w->lstm[(size_t) i].b = gf_load_tensor_f32(wctx, gf, p + "b");
    }
    w->proj_w  = gf_load_tensor(wctx, gf, "dec.proj.weight");
    w->proj_b  = gf_load_tensor_f32(wctx, gf, "dec.proj.bias");
    w->joint_w = gf_load_tensor(wctx, gf, "joint.weight");
    w->joint_b = gf_load_tensor_f32(wctx, gf, "joint.bias");
}

// Graph inputs and outputs of one prediction step.
struct ParakeetPredictGraph {
    struct ggml_cgraph * graph = nullptr;

    struct ggml_tensor * token = nullptr;  // i32 [1]
    struct ggml_tensor * h_in  = nullptr;  // [d, n_layers]
    struct ggml_tensor * c_in  = nullptr;  // [d, n_layers]

    struct ggml_tensor * h_out = nullptr;  // [d, n_layers]
    struct ggml_tensor * c_out = nullptr;  // [d, n_layers]
    struct ggml_tensor * state = nullptr;  // [d], the projected prediction
};

// torch packs the LSTM gates in input, forget, cell, output order.
static ParakeetPredictGraph parakeet_predict_build(struct ggml_context *          ctx,
                                                   const ParakeetDecoderWeights & w,
                                                   int                            max_nodes) {
    ParakeetPredictGraph g;
    g.graph = ggml_new_graph_custom(ctx, (size_t) max_nodes, false);

    const int d = w.d_decoder;

    g.token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    g.h_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, w.n_layers);
    g.c_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, w.n_layers);
    ggml_set_name(g.token, "token");
    ggml_set_name(g.h_in, "h_in");
    ggml_set_name(g.c_in, "c_in");
    ggml_set_input(g.token);
    ggml_set_input(g.h_in);
    ggml_set_input(g.c_in);

    struct ggml_tensor * x = ggml_reshape_1d(ctx, ggml_get_rows(ctx, w.embedding, g.token), d);

    std::vector<struct ggml_tensor *> h_layers;
    std::vector<struct ggml_tensor *> c_layers;

    for (int i = 0; i < w.n_layers; i++) {
        const ParakeetLstmLayer & layer = w.lstm[(size_t) i];
        const size_t              row   = (size_t) i * (size_t) d * sizeof(float);

        struct ggml_tensor * h = ggml_view_1d(ctx, g.h_in, d, row);
        struct ggml_tensor * c = ggml_view_1d(ctx, g.c_in, d, row);

        struct ggml_tensor * z =
            ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, layer.w, x), ggml_mul_mat(ctx, layer.r, h)), layer.b);

        const size_t gate = (size_t) d * sizeof(float);

        struct ggml_tensor * gate_i = ggml_sigmoid(ctx, ggml_view_1d(ctx, z, d, 0));
        struct ggml_tensor * gate_f = ggml_sigmoid(ctx, ggml_view_1d(ctx, z, d, gate));
        struct ggml_tensor * gate_g = ggml_tanh(ctx, ggml_view_1d(ctx, z, d, 2 * gate));
        struct ggml_tensor * gate_o = ggml_sigmoid(ctx, ggml_view_1d(ctx, z, d, 3 * gate));

        struct ggml_tensor * c_new = ggml_add(ctx, ggml_mul(ctx, gate_f, c), ggml_mul(ctx, gate_i, gate_g));
        struct ggml_tensor * h_new = ggml_mul(ctx, gate_o, ggml_tanh(ctx, c_new));

        h_layers.push_back(h_new);
        c_layers.push_back(c_new);
        x = h_new;
    }

    g.h_out = h_layers[0];
    g.c_out = c_layers[0];
    for (size_t i = 1; i < h_layers.size(); i++) {
        g.h_out = ggml_concat(ctx, g.h_out, h_layers[i], 0);
        g.c_out = ggml_concat(ctx, g.c_out, c_layers[i], 0);
    }

    g.state = ggml_add(ctx, ggml_mul_mat(ctx, w.proj_w, x), w.proj_b);

    ggml_set_name(g.state, "prediction");
    ggml_set_output(g.h_out);
    ggml_set_output(g.c_out);
    ggml_set_output(g.state);
    ggml_build_forward_expand(g.graph, g.h_out);
    ggml_build_forward_expand(g.graph, g.c_out);
    ggml_build_forward_expand(g.graph, g.state);
    return g;
}

// Graph inputs and outputs of one joint step.
struct ParakeetJointGraph {
    struct ggml_cgraph * graph = nullptr;

    struct ggml_tensor * frame = nullptr;     // [d], one projected encoder frame
    struct ggml_tensor * state = nullptr;     // [d], the projected prediction

    struct ggml_tensor * logits   = nullptr;  // [vocab + n_durations]
    struct ggml_tensor * token    = nullptr;  // i32 [1]
    struct ggml_tensor * duration = nullptr;  // i32 [1]
};

static ParakeetJointGraph parakeet_joint_build(struct ggml_context *          ctx,
                                               const ParakeetDecoderWeights & w,
                                               int                            max_nodes) {
    ParakeetJointGraph g;
    g.graph = ggml_new_graph_custom(ctx, (size_t) max_nodes, false);

    const int d = w.d_decoder;

    g.frame = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
    g.state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
    ggml_set_name(g.frame, "frame");
    ggml_set_name(g.state, "state");
    ggml_set_input(g.frame);
    ggml_set_input(g.state);

    struct ggml_tensor * joined = ggml_relu(ctx, ggml_add(ctx, g.frame, g.state));
    struct ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, w.joint_w, joined), w.joint_b);
    g.logits                    = logits;

    const int n_durations = (int) w.durations.size();

    struct ggml_tensor * vocab_logits = ggml_view_1d(ctx, logits, w.vocab_size, 0);
    struct ggml_tensor * duration_logits =
        ggml_view_1d(ctx, logits, n_durations, (size_t) w.vocab_size * sizeof(float));

    g.token    = ggml_argmax(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, vocab_logits), w.vocab_size, 1));
    g.duration = ggml_argmax(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, duration_logits), n_durations, 1));

    ggml_set_name(g.logits, "logits");
    ggml_set_name(g.token, "token");
    ggml_set_name(g.duration, "duration");
    ggml_set_output(g.logits);
    ggml_set_output(g.token);
    ggml_set_output(g.duration);
    ggml_build_forward_expand(g.graph, g.logits);
    ggml_build_forward_expand(g.graph, g.token);
    ggml_build_forward_expand(g.graph, g.duration);
    return g;
}

// What one greedy pass cost, for the perf line.
struct TdtStats {
    int n_tokens = 0;  // emitted pieces
    int n_steps  = 0;  // joint evaluations, blanks included

    // First step of the pass, which the parity harness compares: the
    // prediction the start of sequence produces and the joint logits it gives
    // on the first encoder frame.
    std::vector<float> first_state;
    std::vector<float> first_logits;
};

// Greedy transducer loop over one utterance.
//
// frames holds the projected encoder output on the host, n_frames rows of
// d_decoder floats. The loop alternates joint and prediction steps: the joint
// reads the current frame, the prediction advances only on a real token. Each
// emission is handed to on_token as soon as it is detokenized, and a callback
// returning false stops the loop with PK_STATUS_CANCELLED.
static pk_status parakeet_tdt_decode(ggml_backend_t                 backend,
                                     const ParakeetDecoderWeights & w,
                                     const SpDetok &                sp,
                                     const ParakeetPredictGraph &   predict,
                                     const ParakeetJointGraph &     joint,
                                     const float *                  frames,
                                     int                            n_frames,
                                     pk_token_cb                    on_token,
                                     void *                         user,
                                     std::string &                  text,
                                     TdtStats &                     stats) {
    const int          d = w.d_decoder;
    std::vector<float> h((size_t) d * (size_t) w.n_layers, 0.0f);
    std::vector<float> c(h.size(), 0.0f);
    std::vector<float> state((size_t) d, 0.0f);

    auto run_predict = [&](int token_id) {
        ggml_backend_tensor_set(predict.token, &token_id, 0, sizeof(int32_t));
        ggml_backend_tensor_set(predict.h_in, h.data(), 0, h.size() * sizeof(float));
        ggml_backend_tensor_set(predict.c_in, c.data(), 0, c.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, predict.graph) != GGML_STATUS_SUCCESS) {
            s2s_throw("[TDT] Prediction graph compute failed");
        }
        ggml_backend_tensor_get(predict.h_out, h.data(), 0, h.size() * sizeof(float));
        ggml_backend_tensor_get(predict.c_out, c.data(), 0, c.size() * sizeof(float));
        ggml_backend_tensor_get(predict.state, state.data(), 0, state.size() * sizeof(float));
    };

    // The reference starts the prediction network on the blank id, with a
    // zero LSTM state.
    run_predict(w.blank_id);
    stats.first_state = state;

    int frame   = 0;
    int symbols = 0;

    while (frame < n_frames) {
        stats.n_steps++;
        ggml_backend_tensor_set(joint.frame, frames + (size_t) frame * (size_t) d, 0, (size_t) d * sizeof(float));
        ggml_backend_tensor_set(joint.state, state.data(), 0, state.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, joint.graph) != GGML_STATUS_SUCCESS) {
            s2s_throw("[TDT] Joint graph compute failed");
        }

        if (stats.first_logits.empty()) {
            stats.first_logits.resize((size_t) ggml_nelements(joint.logits));
            ggml_backend_tensor_get(joint.logits, stats.first_logits.data(), 0,
                                    stats.first_logits.size() * sizeof(float));
        }

        int32_t token_id    = 0;
        int32_t duration_id = 0;
        ggml_backend_tensor_get(joint.token, &token_id, 0, sizeof(int32_t));
        ggml_backend_tensor_get(joint.duration, &duration_id, 0, sizeof(int32_t));

        int duration = w.durations[(size_t) duration_id];

        if (token_id == w.blank_id) {
            // A blank always moves forward, otherwise the loop would stall on
            // the same frame with the same prediction.
            frame += duration > 0 ? duration : 1;
            symbols = 0;
            continue;
        }

        const size_t written = text.size();
        sp_detok_append(sp, token_id, text);
        if (on_token && !on_token(text.c_str() + written, user)) {
            return PK_STATUS_CANCELLED;
        }

        run_predict(token_id);
        stats.n_tokens++;
        symbols++;

        frame += duration;
        if (duration == 0 && symbols >= w.max_symbols) {
            frame++;
            symbols = 0;
        }
    }

    return PK_STATUS_OK;
}
