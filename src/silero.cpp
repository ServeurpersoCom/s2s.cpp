// silero.cpp: Silero VAD v5 forward pass on GGML
//
// One window of 512 samples at 16 kHz, prefixed by the 64 samples of left
// context kept from the previous call, reflect padded on the right to 640
// samples, then:
//
//   conv1d(basis, k=256, stride=128)     -> [4 frames, 258 channels]
//   magnitude over the 129 bin split     -> [4, 129]
//   conv1d + relu, strides 1, 2, 2, 1    -> [1, 128]
//   LSTM 128 wide, gates in i, o, f, c order
//   relu -> linear -> sigmoid            -> speech probability
//
// The graph has fixed shapes, so it is built once into a persistent arena
// and reallocated never. Weights, graph and scratch live in the context;
// the LSTM state and the left context live in sv_state, one per stream.
// A mutex around the compute lets several streams share one context.

#include "silero.h"

#include "backend.h"
#include "conv-f32.h"
#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "graph-arena.h"
#include "s2s-error.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

#define SV_N_ENCODER 4
#define SV_MAX_NODES 256

struct sv_context {
    ggml_backend_t backend = nullptr;

    struct ggml_tensor * basis               = nullptr;  // [256, 1, 258]
    struct ggml_tensor * enc_w[SV_N_ENCODER] = {};       // [3, in, out]
    struct ggml_tensor * enc_b[SV_N_ENCODER] = {};       // [out]
    struct ggml_tensor * lstm_w              = nullptr;  // [128, 512]
    struct ggml_tensor * lstm_r              = nullptr;  // [128, 512]
    struct ggml_tensor * lstm_b              = nullptr;  // [512]
    struct ggml_tensor * dec_w               = nullptr;  // [128]
    struct ggml_tensor * dec_b               = nullptr;  // [1]

    WeightCtx      wctx  = {};
    GraphArena     arena = {};
    ggml_gallocr_t alloc = nullptr;

    struct ggml_cgraph * graph = nullptr;

    struct ggml_tensor * in_pcm = nullptr;  // [640]
    struct ggml_tensor * in_h   = nullptr;  // [128]
    struct ggml_tensor * in_c   = nullptr;  // [128]
    struct ggml_tensor * out_p  = nullptr;  // [1]
    struct ggml_tensor * out_h  = nullptr;  // [128]
    struct ggml_tensor * out_c  = nullptr;  // [128]

    int sample_rate = 0;
    int window      = 0;
    int context     = 0;
    int n_fft       = 0;
    int hop         = 0;
    int n_bins      = 0;
    int hidden      = 0;
    int padded      = 0;  // context + window + n_fft / 2

    std::mutex mutex;
};

struct sv_state {
    sv_context *       ctx = nullptr;
    std::vector<float> h;        // [hidden]
    std::vector<float> c;        // [hidden]
    std::vector<float> left;     // [context]
    std::vector<float> scratch;  // [padded]
};

// The two level enums share their values, so routing the library log to the
// caller sink only costs a cast.
static sv_log_cb g_sv_log_cb = nullptr;

static void sv_log_trampoline(enum s2s_log_level level, const char * text, void * user_data) {
    g_sv_log_cb((enum sv_log_level) level, text, user_data);
}

void sv_log_set(sv_log_cb cb, void * user_data) {
    g_sv_log_cb = cb;
    s2s_log_set(cb ? sv_log_trampoline : nullptr, user_data);
}

// Builds the forward graph with the shapes the model fixes: one window in,
// one probability and one LSTM state out.
static void sv_build_graph(sv_context * ctx) {
    struct ggml_context * gctx = graph_arena_begin(&ctx->arena);

    ctx->graph = ggml_new_graph_custom(gctx, SV_MAX_NODES, false);

    ctx->in_pcm = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, ctx->padded, 1);
    ctx->in_h   = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, ctx->hidden);
    ctx->in_c   = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, ctx->hidden);
    ggml_set_name(ctx->in_pcm, "pcm");
    ggml_set_name(ctx->in_h, "h");
    ggml_set_name(ctx->in_c, "c");
    ggml_set_input(ctx->in_pcm);
    ggml_set_input(ctx->in_h);
    ggml_set_input(ctx->in_c);

    // STFT as a strided convolution: the basis holds the cos rows first and
    // the sin rows second, hence the split at n_bins.
    struct ggml_tensor * spec = conv_1d_f32(gctx, ctx->basis, ctx->in_pcm, ctx->hop, 0);

    const int64_t n_frames = spec->ne[0];

    struct ggml_tensor * re = ggml_cont(gctx, ggml_view_2d(gctx, spec, n_frames, ctx->n_bins, spec->nb[1], 0));
    struct ggml_tensor * im = ggml_cont(
        gctx, ggml_view_2d(gctx, spec, n_frames, ctx->n_bins, spec->nb[1], (size_t) ctx->n_bins * spec->nb[1]));

    struct ggml_tensor * x = ggml_sqrt(gctx, ggml_add(gctx, ggml_sqr(gctx, re), ggml_sqr(gctx, im)));

    const int strides[SV_N_ENCODER] = { 1, 2, 2, 1 };
    for (int i = 0; i < SV_N_ENCODER; i++) {
        x = conv_1d_f32(gctx, ctx->enc_w[i], x, strides[i], 1);
        x = ggml_add(gctx, x, ggml_reshape_2d(gctx, ctx->enc_b[i], 1, ctx->enc_b[i]->ne[0]));
        x = ggml_relu(gctx, x);
    }

    // The encoder ends on a single frame, so the LSTM input is a plain vector.
    struct ggml_tensor * xv = ggml_reshape_1d(gctx, x, ctx->hidden);

    struct ggml_tensor * z =
        ggml_add(gctx, ggml_mul_mat(gctx, ctx->lstm_w, xv), ggml_mul_mat(gctx, ctx->lstm_r, ctx->in_h));
    z = ggml_add(gctx, z, ctx->lstm_b);

    const size_t gate = (size_t) ctx->hidden * sizeof(float);

    struct ggml_tensor * gate_i = ggml_sigmoid(gctx, ggml_view_1d(gctx, z, ctx->hidden, 0));
    struct ggml_tensor * gate_o = ggml_sigmoid(gctx, ggml_view_1d(gctx, z, ctx->hidden, gate));
    struct ggml_tensor * gate_f = ggml_sigmoid(gctx, ggml_view_1d(gctx, z, ctx->hidden, 2 * gate));
    struct ggml_tensor * gate_g = ggml_tanh(gctx, ggml_view_1d(gctx, z, ctx->hidden, 3 * gate));

    ctx->out_c = ggml_add(gctx, ggml_mul(gctx, gate_f, ctx->in_c), ggml_mul(gctx, gate_i, gate_g));
    ctx->out_h = ggml_mul(gctx, gate_o, ggml_tanh(gctx, ctx->out_c));

    struct ggml_tensor * head =
        ggml_mul_mat(gctx, ggml_reshape_2d(gctx, ctx->dec_w, ctx->hidden, 1), ggml_relu(gctx, ctx->out_h));
    ctx->out_p = ggml_sigmoid(gctx, ggml_add(gctx, head, ctx->dec_b));

    ggml_set_name(ctx->out_p, "prob");
    ggml_set_name(ctx->out_h, "h_out");
    ggml_set_name(ctx->out_c, "c_out");
    ggml_set_output(ctx->out_p);
    ggml_set_output(ctx->out_h);
    ggml_set_output(ctx->out_c);

    ggml_build_forward_expand(ctx->graph, ctx->out_p);
    ggml_build_forward_expand(ctx->graph, ctx->out_h);
    ggml_build_forward_expand(ctx->graph, ctx->out_c);
}

sv_context * sv_init(const char * gguf_path) {
    if (!gguf_path) {
        s2s_set_error("[Silero] Gguf_path is NULL");
        return nullptr;
    }

    sv_context * ctx = new sv_context();

    try {
        GGUFModel gf = {};
        if (!gf_load(&gf, gguf_path)) {
            s2s_set_error("[Silero] Failed to open %s", gguf_path);
            delete ctx;
            return nullptr;
        }

        ctx->sample_rate = (int) gf_get_u32(gf, "vad.sample_rate");
        ctx->window      = (int) gf_get_u32(gf, "vad.window");
        ctx->context     = (int) gf_get_u32(gf, "vad.context");
        ctx->n_fft       = (int) gf_get_u32(gf, "vad.n_fft");
        ctx->hop         = (int) gf_get_u32(gf, "vad.hop");
        ctx->n_bins      = (int) gf_get_u32(gf, "vad.n_bins");
        ctx->hidden      = (int) gf_get_u32(gf, "vad.hidden");
        ctx->padded      = ctx->context + ctx->window + ctx->n_fft / 2;

        ctx->backend = backend_init_cpu("Silero", 1).backend;
        if (!ctx->backend) {
            s2s_set_error("[Silero] Failed to init the CPU backend");
            gf_close(&gf);
            delete ctx;
            return nullptr;
        }

        wctx_init(&ctx->wctx, 2 * SV_N_ENCODER + 6);
        ctx->basis = gf_load_tensor_f32(&ctx->wctx, gf, "stft.basis");
        for (int i = 0; i < SV_N_ENCODER; i++) {
            ctx->enc_w[i] = gf_load_tensor_f32(&ctx->wctx, gf, "enc." + std::to_string(i) + ".weight");
            ctx->enc_b[i] = gf_load_tensor_f32(&ctx->wctx, gf, "enc." + std::to_string(i) + ".bias");
        }
        ctx->lstm_w = gf_load_tensor_f32(&ctx->wctx, gf, "lstm.w");
        ctx->lstm_r = gf_load_tensor_f32(&ctx->wctx, gf, "lstm.r");
        ctx->lstm_b = gf_load_tensor_f32(&ctx->wctx, gf, "lstm.b");
        ctx->dec_w  = gf_load_tensor_f32(&ctx->wctx, gf, "dec.weight");
        ctx->dec_b  = gf_load_tensor_f32(&ctx->wctx, gf, "dec.bias");

        const bool loaded = wctx_alloc(&ctx->wctx, ctx->backend);
        gf_close(&gf);
        if (!loaded) {
            s2s_set_error("[Silero] Failed to upload the weights");
            sv_free(ctx);
            return nullptr;
        }

        if (!graph_arena_init(&ctx->arena, SV_MAX_NODES)) {
            s2s_set_error("[Silero] Failed to allocate the graph arena");
            sv_free(ctx);
            return nullptr;
        }

        sv_build_graph(ctx);

        ctx->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ctx->alloc || !ggml_gallocr_alloc_graph(ctx->alloc, ctx->graph)) {
            s2s_set_error("[Silero] Failed to allocate the compute graph");
            sv_free(ctx);
            return nullptr;
        }

        s2s_log(S2S_LOG_INFO, "[Silero] %d Hz, window %d, context %d, hidden %d", ctx->sample_rate, ctx->window,
                ctx->context, ctx->hidden);
        return ctx;
    } catch (const std::exception & e) {
        s2s_set_error("%s", e.what());
        sv_free(ctx);
        return nullptr;
    }
}

void sv_free(sv_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->alloc) {
        ggml_gallocr_free(ctx->alloc);
    }
    graph_arena_free(&ctx->arena);
    wctx_free(&ctx->wctx);
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    delete ctx;
}

int sv_window(const sv_context * ctx) {
    return ctx ? ctx->window : 0;
}

int sv_sample_rate(const sv_context * ctx) {
    return ctx ? ctx->sample_rate : 0;
}

sv_state * sv_state_new(sv_context * ctx) {
    if (!ctx) {
        s2s_set_error("[Silero] Context is NULL");
        return nullptr;
    }
    sv_state * state = new sv_state();
    state->ctx       = ctx;
    state->h.assign((size_t) ctx->hidden, 0.0f);
    state->c.assign((size_t) ctx->hidden, 0.0f);
    state->left.assign((size_t) ctx->context, 0.0f);
    state->scratch.assign((size_t) ctx->padded, 0.0f);
    return state;
}

void sv_state_reset(sv_state * state) {
    if (!state) {
        return;
    }
    std::fill(state->h.begin(), state->h.end(), 0.0f);
    std::fill(state->c.begin(), state->c.end(), 0.0f);
    std::fill(state->left.begin(), state->left.end(), 0.0f);
}

void sv_state_free(sv_state * state) {
    delete state;
}

float sv_prob(sv_state * state, const float * pcm, int n_samples) {
    if (!state || !pcm) {
        s2s_set_error("[Silero] State or pcm is NULL");
        return -1.0f;
    }

    sv_context * ctx = state->ctx;
    if (n_samples != ctx->window) {
        s2s_set_error("[Silero] Expected %d samples, got %d", ctx->window, n_samples);
        return -1.0f;
    }

    // Left context, the window itself, then the reflect pad the STFT needs
    // on the right: padded[n + k] = frame[n - 2 - k].
    float *   frame  = state->scratch.data();
    const int n_head = ctx->context + ctx->window;
    memcpy(frame, state->left.data(), (size_t) ctx->context * sizeof(float));
    memcpy(frame + ctx->context, pcm, (size_t) ctx->window * sizeof(float));
    for (int k = 0; k < ctx->padded - n_head; k++) {
        frame[n_head + k] = frame[n_head - 2 - k];
    }

    float prob = 0.0f;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);

        ggml_backend_tensor_set(ctx->in_pcm, frame, 0, (size_t) ctx->padded * sizeof(float));
        ggml_backend_tensor_set(ctx->in_h, state->h.data(), 0, state->h.size() * sizeof(float));
        ggml_backend_tensor_set(ctx->in_c, state->c.data(), 0, state->c.size() * sizeof(float));

        if (ggml_backend_graph_compute(ctx->backend, ctx->graph) != GGML_STATUS_SUCCESS) {
            s2s_set_error("[Silero] Graph compute failed");
            return -1.0f;
        }

        ggml_backend_tensor_get(ctx->out_p, &prob, 0, sizeof(float));
        ggml_backend_tensor_get(ctx->out_h, state->h.data(), 0, state->h.size() * sizeof(float));
        ggml_backend_tensor_get(ctx->out_c, state->c.data(), 0, state->c.size() * sizeof(float));
    }

    memcpy(state->left.data(), pcm + ctx->window - ctx->context, (size_t) ctx->context * sizeof(float));
    return prob;
}

const char * sv_last_error(void) {
    return s2s_last_error();
}
