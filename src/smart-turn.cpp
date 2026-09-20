// smart-turn.cpp: Smart Turn v3.2 forward pass on GGML
//
// Whisper tiny encoder plus an attention pooling head:
//
//   mel      80 bins, 800 frames over 8 seconds
//   conv1d   80 -> 384, k=3, stride 1, gelu
//   conv1d   384 -> 384, k=3, stride 2, gelu  -> 400 frames
//   + learned positions [400, 384]
//   4 pre norm layers, 6 heads of 64, ffn 1536, gelu
//   final layer norm
//   pooling  tanh(Wx + b) -> score -> softmax over time -> weighted sum
//   head     linear 384 -> 256, layer norm, gelu, 256 -> 64, gelu, 64 -> 1
//   sigmoid  -> probability that the turn is complete
//
// Two graphs, both with fixed shapes and built once into their arena. The
// split sits where the Whisper normalization needs a global maximum over
// the spectrogram, which GGML cannot reduce in graph: the mel graph runs,
// the host normalizes, the encoder graph runs.
//
// One worker thread runs every forward pass. A caller hands it the audio and
// waits, so the CPU backend always computes from the same thread and a
// single OpenMP team of encoder threads serves the whole process.

#include "smart-turn.h"

#include "audio-mel.h"
#include "backend.h"
#include "conv-f32.h"
#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "graph-arena.h"
#include "s2s-error.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define ST_NORM_EPS 1e-5f

struct STLayer {
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * q_w         = nullptr;
    struct ggml_tensor * q_b         = nullptr;
    struct ggml_tensor * k_w         = nullptr;
    struct ggml_tensor * v_w         = nullptr;
    struct ggml_tensor * v_b         = nullptr;
    struct ggml_tensor * o_w         = nullptr;
    struct ggml_tensor * o_b         = nullptr;
    struct ggml_tensor * ffn_norm_w  = nullptr;
    struct ggml_tensor * ffn_norm_b  = nullptr;
    struct ggml_tensor * fc1_w       = nullptr;
    struct ggml_tensor * fc1_b       = nullptr;
    struct ggml_tensor * fc2_w       = nullptr;
    struct ggml_tensor * fc2_b       = nullptr;
};

struct st_context {
    ggml_backend_t backend = nullptr;

    struct ggml_tensor * conv1_w = nullptr;
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * conv2_w = nullptr;
    struct ggml_tensor * conv2_b = nullptr;
    struct ggml_tensor * pos     = nullptr;
    struct ggml_tensor * norm_w  = nullptr;
    struct ggml_tensor * norm_b  = nullptr;

    std::vector<STLayer> layers;

    struct ggml_tensor * pool_hidden_w = nullptr;
    struct ggml_tensor * pool_hidden_b = nullptr;
    struct ggml_tensor * pool_score_w  = nullptr;
    struct ggml_tensor * pool_score_b  = nullptr;

    struct ggml_tensor * cls0_w     = nullptr;
    struct ggml_tensor * cls0_b     = nullptr;
    struct ggml_tensor * cls_norm_w = nullptr;
    struct ggml_tensor * cls_norm_b = nullptr;
    struct ggml_tensor * cls1_w     = nullptr;
    struct ggml_tensor * cls1_b     = nullptr;
    struct ggml_tensor * cls2_w     = nullptr;
    struct ggml_tensor * cls2_b     = nullptr;

    // Mel constants: Hann window, the two DFT matrices and the filterbank.
    struct ggml_tensor * hann      = nullptr;
    struct ggml_tensor * dft_real  = nullptr;
    struct ggml_tensor * dft_imag  = nullptr;
    struct ggml_tensor * mel_basis = nullptr;

    WeightCtx wctx     = {};
    WeightCtx mel_wctx = {};

    GraphArena           mel_arena = {};
    ggml_gallocr_t       mel_alloc = nullptr;
    struct ggml_cgraph * mel_graph = nullptr;
    struct ggml_tensor * mel_in    = nullptr;  // [window + n_fft]
    struct ggml_tensor * mel_out   = nullptr;  // [n_frames + 1, n_mels]

    GraphArena           enc_arena = {};
    ggml_gallocr_t       enc_alloc = nullptr;
    struct ggml_cgraph * enc_graph = nullptr;
    struct ggml_tensor * enc_in    = nullptr;  // [n_frames, n_mels]
    struct ggml_tensor * enc_out   = nullptr;  // [1]

    AudioMelConfig    mel_cfg = {};
    AudioMelConstants mel_c   = {};

    int sample_rate = 0;
    int window      = 0;
    int n_frames    = 0;
    int n_layers    = 0;
    int n_heads     = 0;
    int d_model     = 0;
    int d_ffn       = 0;

    std::vector<float> audio;    // [window + n_fft]
    std::vector<float> log_mel;  // [(n_frames + 1) * n_mels]
    std::vector<float> mel;      // [n_frames * n_mels]

    // call_mutex holds one caller for its whole request, which serializes
    // the streams. The job fields change hands under job_mutex.
    std::thread             worker;
    std::mutex              call_mutex;
    std::mutex              job_mutex;
    std::condition_variable job_cv;
    const float *           job_pcm     = nullptr;
    int                     job_samples = 0;
    bool                    job_ready   = false;
    bool                    job_done    = false;
    bool                    stop        = false;
    float                   job_prob    = 0.0f;
    std::string             job_error;
};

static st_log_cb g_st_log_cb = nullptr;

static void st_log_trampoline(enum s2s_log_level level, const char * text, void * user_data) {
    g_st_log_cb((enum st_log_level) level, text, user_data);
}

void st_log_set(st_log_cb cb, void * user_data) {
    g_st_log_cb = cb;
    s2s_log_set(cb ? st_log_trampoline : nullptr, user_data);
}

// LayerNormalization: normalize over the model dimension, then scale and
// shift with the learned vectors.
static struct ggml_tensor * st_layer_norm(struct ggml_context * ctx,
                                          struct ggml_tensor *  x,
                                          struct ggml_tensor *  weight,
                                          struct ggml_tensor *  bias) {
    x = ggml_norm(ctx, x, ST_NORM_EPS);
    x = ggml_mul(ctx, x, weight);
    return ggml_add(ctx, x, bias);
}

static struct ggml_tensor * st_linear(struct ggml_context * ctx,
                                      struct ggml_tensor *  x,
                                      struct ggml_tensor *  weight,
                                      struct ggml_tensor *  bias) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    return bias ? ggml_add(ctx, y, bias) : y;
}

// Builds the mel graph: reflect padded audio in, log10 mel out.
static void st_build_mel_graph(st_context * ctx) {
    struct ggml_context * gctx = graph_arena_begin(&ctx->mel_arena);

    ctx->mel_graph = ggml_new_graph_custom(gctx, 512, false);

    ctx->mel_in = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, ctx->window + ctx->mel_cfg.n_fft);
    ggml_set_name(ctx->mel_in, "audio");
    ggml_set_input(ctx->mel_in);

    ctx->mel_out =
        audio_mel_build_graph(gctx, ctx->mel_in, ctx->hann, ctx->dft_real, ctx->dft_imag, ctx->mel_basis, ctx->mel_cfg);
    ggml_set_output(ctx->mel_out);
    ggml_build_forward_expand(ctx->mel_graph, ctx->mel_out);
}

// Builds the encoder graph: normalized mel in, completion probability out.
static void st_build_encoder_graph(st_context * ctx) {
    struct ggml_context * gctx = graph_arena_begin(&ctx->enc_arena);

    ctx->enc_graph = ggml_new_graph_custom(gctx, 2048, false);

    ctx->enc_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, ctx->n_frames, ctx->mel_cfg.n_mels);
    ggml_set_name(ctx->enc_in, "mel");
    ggml_set_input(ctx->enc_in);

    struct ggml_tensor * x = conv_1d_f32(gctx, ctx->conv1_w, ctx->enc_in, 1, 1);
    x                      = ggml_add(gctx, x, ggml_reshape_2d(gctx, ctx->conv1_b, 1, ctx->d_model));
    x                      = ggml_gelu_erf(gctx, x);

    x = conv_1d_f32(gctx, ctx->conv2_w, x, 2, 1);
    x = ggml_add(gctx, x, ggml_reshape_2d(gctx, ctx->conv2_b, 1, ctx->d_model));
    x = ggml_gelu_erf(gctx, x);

    // conv output is [n_tokens, d_model], the transformer wants the model
    // dimension inner.
    const int64_t n_tokens = x->ne[0];
    x                      = ggml_cont(gctx, ggml_transpose(gctx, x));
    x                      = ggml_add(gctx, x, ctx->pos);

    const int   d_head = ctx->d_model / ctx->n_heads;
    const float scale  = 1.0f / sqrtf((float) d_head);

    for (const STLayer & layer : ctx->layers) {
        struct ggml_tensor * residual = x;
        struct ggml_tensor * h        = st_layer_norm(gctx, x, layer.attn_norm_w, layer.attn_norm_b);

        struct ggml_tensor * q = st_linear(gctx, h, layer.q_w, layer.q_b);
        struct ggml_tensor * k = st_linear(gctx, h, layer.k_w, nullptr);
        struct ggml_tensor * v = st_linear(gctx, h, layer.v_w, layer.v_b);

        q = ggml_permute(gctx, ggml_reshape_3d(gctx, q, d_head, ctx->n_heads, n_tokens), 0, 2, 1, 3);
        k = ggml_permute(gctx, ggml_reshape_3d(gctx, k, d_head, ctx->n_heads, n_tokens), 0, 2, 1, 3);
        v = ggml_permute(gctx, ggml_reshape_3d(gctx, v, d_head, ctx->n_heads, n_tokens), 0, 2, 1, 3);

        struct ggml_tensor * kq = ggml_mul_mat(gctx, ggml_cont(gctx, k), ggml_cont(gctx, q));
        kq                      = ggml_soft_max_ext(gctx, kq, nullptr, scale, 0.0f);

        struct ggml_tensor * vt  = ggml_cont(gctx, ggml_permute(gctx, v, 1, 0, 2, 3));
        struct ggml_tensor * kqv = ggml_mul_mat(gctx, vt, kq);

        kqv = ggml_cont(gctx, ggml_permute(gctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(gctx, kqv, ctx->d_model, n_tokens);

        x = ggml_add(gctx, residual, st_linear(gctx, kqv, layer.o_w, layer.o_b));

        residual = x;
        h        = st_layer_norm(gctx, x, layer.ffn_norm_w, layer.ffn_norm_b);
        h        = ggml_gelu_erf(gctx, st_linear(gctx, h, layer.fc1_w, layer.fc1_b));
        h        = st_linear(gctx, h, layer.fc2_w, layer.fc2_b);
        x        = ggml_add(gctx, residual, h);
    }

    x = st_layer_norm(gctx, x, ctx->norm_w, ctx->norm_b);

    // Attention pooling: one score per frame, softmax over time, weighted sum.
    struct ggml_tensor * score = ggml_tanh(gctx, st_linear(gctx, x, ctx->pool_hidden_w, ctx->pool_hidden_b));
    score                      = st_linear(gctx, score, ctx->pool_score_w, ctx->pool_score_b);
    score                      = ggml_cont(gctx, ggml_transpose(gctx, score));
    struct ggml_tensor * attn  = ggml_soft_max(gctx, score);

    struct ggml_tensor * pooled = ggml_mul_mat(gctx, ggml_cont(gctx, ggml_transpose(gctx, x)), attn);
    pooled                      = ggml_reshape_1d(gctx, pooled, ctx->d_model);

    struct ggml_tensor * head = st_linear(gctx, pooled, ctx->cls0_w, ctx->cls0_b);
    head                      = st_layer_norm(gctx, head, ctx->cls_norm_w, ctx->cls_norm_b);
    head                      = ggml_gelu_erf(gctx, head);
    head                      = ggml_gelu_erf(gctx, st_linear(gctx, head, ctx->cls1_w, ctx->cls1_b));
    head                      = st_linear(gctx, head, ctx->cls2_w, ctx->cls2_b);

    ctx->enc_out = ggml_sigmoid(gctx, head);
    ggml_set_name(ctx->enc_out, "completion");
    ggml_set_output(ctx->enc_out);
    ggml_build_forward_expand(ctx->enc_graph, ctx->enc_out);
}

// Uploads the Hann window, the DFT matrices and the filterbank as weights.
static bool st_load_mel_constants(st_context * ctx) {
    audio_mel_compute_constants(ctx->mel_cfg, ctx->mel_c);

    wctx_init(&ctx->mel_wctx, 4);
    const int n_fft  = ctx->mel_cfg.n_fft;
    const int n_freq = ctx->mel_c.n_freq;
    const int n_mels = ctx->mel_cfg.n_mels;

    ctx->hann      = ggml_new_tensor_1d(ctx->mel_wctx.ctx, GGML_TYPE_F32, n_fft);
    ctx->dft_real  = ggml_new_tensor_2d(ctx->mel_wctx.ctx, GGML_TYPE_F32, n_fft, n_freq);
    ctx->dft_imag  = ggml_new_tensor_2d(ctx->mel_wctx.ctx, GGML_TYPE_F32, n_fft, n_freq);
    ctx->mel_basis = ggml_new_tensor_2d(ctx->mel_wctx.ctx, GGML_TYPE_F32, n_freq, n_mels);

    if (!wctx_alloc(&ctx->mel_wctx, ctx->backend)) {
        return false;
    }

    ggml_backend_tensor_set(ctx->hann, ctx->mel_c.hann.data(), 0, ctx->mel_c.hann.size() * sizeof(float));
    ggml_backend_tensor_set(ctx->dft_real, ctx->mel_c.dft_real.data(), 0, ctx->mel_c.dft_real.size() * sizeof(float));
    ggml_backend_tensor_set(ctx->dft_imag, ctx->mel_c.dft_imag.data(), 0, ctx->mel_c.dft_imag.size() * sizeof(float));
    ggml_backend_tensor_set(ctx->mel_basis, ctx->mel_c.mel_basis.data(), 0,
                            ctx->mel_c.mel_basis.size() * sizeof(float));
    return true;
}

// The forward pass, on the worker thread only.
static float st_compute(st_context * ctx, const float * pcm, int n_samples) {
    const int half = ctx->mel_cfg.n_fft / 2;
    const int kept = std::min(n_samples, ctx->window);
    const int skip = n_samples - kept;

    // Center padding for the STFT, zero padding on the right for a turn
    // shorter than the window: both match the upstream feature extractor.
    std::fill(ctx->audio.begin(), ctx->audio.end(), 0.0f);
    memcpy(ctx->audio.data() + half, pcm + skip, (size_t) kept * sizeof(float));
    for (int i = 0; i < half; i++) {
        ctx->audio[(size_t) half - 1 - (size_t) i] = ctx->audio[(size_t) half + 1 + (size_t) i];
        ctx->audio[(size_t) ctx->window + (size_t) half + (size_t) i] =
            ctx->audio[(size_t) ctx->window + (size_t) half - 2 - (size_t) i];
    }

    ggml_backend_tensor_set(ctx->mel_in, ctx->audio.data(), 0, ctx->audio.size() * sizeof(float));
    if (ggml_backend_graph_compute(ctx->backend, ctx->mel_graph) != GGML_STATUS_SUCCESS) {
        s2s_set_error("[SmartTurn] Mel graph compute failed");
        return -1.0f;
    }
    ggml_backend_tensor_get(ctx->mel_out, ctx->log_mel.data(), 0, ctx->log_mel.size() * sizeof(float));

    audio_mel_normalize(ctx->log_mel, ctx->mel_cfg.n_mels, (size_t) ctx->n_frames + 1, ctx->mel);

    ggml_backend_tensor_set(ctx->enc_in, ctx->mel.data(), 0, ctx->mel.size() * sizeof(float));
    if (ggml_backend_graph_compute(ctx->backend, ctx->enc_graph) != GGML_STATUS_SUCCESS) {
        s2s_set_error("[SmartTurn] Encoder graph compute failed");
        return -1.0f;
    }

    float prob = 0.0f;
    ggml_backend_tensor_get(ctx->enc_out, &prob, 0, sizeof(float));
    return prob;
}

// Runs one job at a time until st_free raises stop. The diagnostic of a
// failed pass is recorded on this thread, so it travels back with the result.
static void st_worker(st_context * ctx) {
    s2s_log_thread("SmartTurn");

    std::unique_lock<std::mutex> lock(ctx->job_mutex);
    for (;;) {
        ctx->job_cv.wait(lock, [ctx] { return ctx->job_ready || ctx->stop; });
        if (ctx->stop) {
            return;
        }
        ctx->job_ready = false;
        lock.unlock();

        const float prob = st_compute(ctx, ctx->job_pcm, ctx->job_samples);

        lock.lock();
        ctx->job_prob  = prob;
        ctx->job_error = prob < 0.0f ? s2s_last_error() : "";
        ctx->job_done  = true;
        ctx->job_cv.notify_all();
    }
}

st_context * st_init(const char * gguf_path, int n_threads) {
    if (!gguf_path) {
        s2s_set_error("[SmartTurn] Gguf_path is NULL");
        return nullptr;
    }

    st_context * ctx = new st_context();

    try {
        GGUFModel gf = {};
        if (!gf_load(&gf, gguf_path)) {
            s2s_set_error("[SmartTurn] Failed to open %s", gguf_path);
            delete ctx;
            return nullptr;
        }

        ctx->sample_rate         = (int) gf_get_u32(gf, "turn.sample_rate");
        ctx->window              = (int) gf_get_u32(gf, "turn.window");
        ctx->n_frames            = (int) gf_get_u32(gf, "turn.n_frames");
        ctx->n_layers            = (int) gf_get_u32(gf, "turn.n_layers");
        ctx->n_heads             = (int) gf_get_u32(gf, "turn.n_heads");
        ctx->d_model             = (int) gf_get_u32(gf, "turn.d_model");
        ctx->d_ffn               = (int) gf_get_u32(gf, "turn.d_ffn");
        ctx->mel_cfg.sample_rate = ctx->sample_rate;
        ctx->mel_cfg.n_fft       = (int) gf_get_u32(gf, "turn.n_fft");
        ctx->mel_cfg.hop         = (int) gf_get_u32(gf, "turn.hop");
        ctx->mel_cfg.n_mels      = (int) gf_get_u32(gf, "turn.n_mels");
        ctx->mel_cfg.fmin        = 0.0f;
        ctx->mel_cfg.fmax        = (float) ctx->sample_rate * 0.5f;

        ctx->backend = backend_init_cpu("SmartTurn", n_threads).backend;
        if (!ctx->backend) {
            s2s_set_error("[SmartTurn] Failed to init the CPU backend");
            gf_close(&gf);
            delete ctx;
            return nullptr;
        }

        wctx_init(&ctx->wctx, 15 * ctx->n_layers + 24);
        ctx->conv1_w = gf_load_tensor_f32(&ctx->wctx, gf, "enc.conv1.weight");
        ctx->conv1_b = gf_load_tensor_f32(&ctx->wctx, gf, "enc.conv1.bias");
        ctx->conv2_w = gf_load_tensor_f32(&ctx->wctx, gf, "enc.conv2.weight");
        ctx->conv2_b = gf_load_tensor_f32(&ctx->wctx, gf, "enc.conv2.bias");
        ctx->pos     = gf_load_tensor_f32(&ctx->wctx, gf, "enc.pos");

        ctx->layers.resize((size_t) ctx->n_layers);
        for (int i = 0; i < ctx->n_layers; i++) {
            STLayer &         l = ctx->layers[(size_t) i];
            const std::string p = "enc." + std::to_string(i) + ".";
            l.attn_norm_w       = gf_load_tensor_f32(&ctx->wctx, gf, p + "attn_norm.weight");
            l.attn_norm_b       = gf_load_tensor_f32(&ctx->wctx, gf, p + "attn_norm.bias");
            l.q_w               = gf_load_tensor_f32(&ctx->wctx, gf, p + "q.weight");
            l.q_b               = gf_load_tensor_f32(&ctx->wctx, gf, p + "q.bias");
            l.k_w               = gf_load_tensor_f32(&ctx->wctx, gf, p + "k.weight");
            l.v_w               = gf_load_tensor_f32(&ctx->wctx, gf, p + "v.weight");
            l.v_b               = gf_load_tensor_f32(&ctx->wctx, gf, p + "v.bias");
            l.o_w               = gf_load_tensor_f32(&ctx->wctx, gf, p + "o.weight");
            l.o_b               = gf_load_tensor_f32(&ctx->wctx, gf, p + "o.bias");
            l.ffn_norm_w        = gf_load_tensor_f32(&ctx->wctx, gf, p + "ffn_norm.weight");
            l.ffn_norm_b        = gf_load_tensor_f32(&ctx->wctx, gf, p + "ffn_norm.bias");
            l.fc1_w             = gf_load_tensor_f32(&ctx->wctx, gf, p + "fc1.weight");
            l.fc1_b             = gf_load_tensor_f32(&ctx->wctx, gf, p + "fc1.bias");
            l.fc2_w             = gf_load_tensor_f32(&ctx->wctx, gf, p + "fc2.weight");
            l.fc2_b             = gf_load_tensor_f32(&ctx->wctx, gf, p + "fc2.bias");
        }

        ctx->norm_w = gf_load_tensor_f32(&ctx->wctx, gf, "enc.norm.weight");
        ctx->norm_b = gf_load_tensor_f32(&ctx->wctx, gf, "enc.norm.bias");

        ctx->pool_hidden_w = gf_load_tensor_f32(&ctx->wctx, gf, "pool.hidden.weight");
        ctx->pool_hidden_b = gf_load_tensor_f32(&ctx->wctx, gf, "pool.hidden.bias");
        ctx->pool_score_w  = gf_load_tensor_f32(&ctx->wctx, gf, "pool.score.weight");
        ctx->pool_score_b  = gf_load_tensor_f32(&ctx->wctx, gf, "pool.score.bias");

        ctx->cls0_w     = gf_load_tensor_f32(&ctx->wctx, gf, "cls.0.weight");
        ctx->cls0_b     = gf_load_tensor_f32(&ctx->wctx, gf, "cls.0.bias");
        ctx->cls_norm_w = gf_load_tensor_f32(&ctx->wctx, gf, "cls.norm.weight");
        ctx->cls_norm_b = gf_load_tensor_f32(&ctx->wctx, gf, "cls.norm.bias");
        ctx->cls1_w     = gf_load_tensor_f32(&ctx->wctx, gf, "cls.1.weight");
        ctx->cls1_b     = gf_load_tensor_f32(&ctx->wctx, gf, "cls.1.bias");
        ctx->cls2_w     = gf_load_tensor_f32(&ctx->wctx, gf, "cls.2.weight");
        ctx->cls2_b     = gf_load_tensor_f32(&ctx->wctx, gf, "cls.2.bias");

        const bool loaded = wctx_alloc(&ctx->wctx, ctx->backend);
        gf_close(&gf);
        if (!loaded) {
            s2s_set_error("[SmartTurn] Failed to upload the weights");
            st_free(ctx);
            return nullptr;
        }

        if (!st_load_mel_constants(ctx)) {
            s2s_set_error("[SmartTurn] Failed to upload the mel constants");
            st_free(ctx);
            return nullptr;
        }

        if (!graph_arena_init(&ctx->mel_arena, 512) || !graph_arena_init(&ctx->enc_arena, 2048)) {
            s2s_set_error("[SmartTurn] Failed to allocate the graph arenas");
            st_free(ctx);
            return nullptr;
        }

        st_build_mel_graph(ctx);
        st_build_encoder_graph(ctx);

        ctx->mel_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        ctx->enc_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ctx->mel_alloc || !ggml_gallocr_alloc_graph(ctx->mel_alloc, ctx->mel_graph) || !ctx->enc_alloc ||
            !ggml_gallocr_alloc_graph(ctx->enc_alloc, ctx->enc_graph)) {
            s2s_set_error("[SmartTurn] Failed to allocate the compute graphs");
            st_free(ctx);
            return nullptr;
        }

        ctx->audio.assign((size_t) ctx->window + (size_t) ctx->mel_cfg.n_fft, 0.0f);
        ctx->log_mel.assign((size_t) (ctx->n_frames + 1) * (size_t) ctx->mel_cfg.n_mels, 0.0f);

        ctx->worker = std::thread(st_worker, ctx);

        s2s_log(S2S_LOG_INFO, "[SmartTurn] %d Hz, %.1fs window, %d frames, %d layers, %d heads, d_model %d",
                ctx->sample_rate, (double) ctx->window / ctx->sample_rate, ctx->n_frames, ctx->n_layers, ctx->n_heads,
                ctx->d_model);
        return ctx;
    } catch (const std::exception & e) {
        s2s_set_error("%s", e.what());
        st_free(ctx);
        return nullptr;
    }
}

void st_free(st_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(ctx->job_mutex);
            ctx->stop = true;
        }
        ctx->job_cv.notify_all();
        ctx->worker.join();
    }
    if (ctx->mel_alloc) {
        ggml_gallocr_free(ctx->mel_alloc);
    }
    if (ctx->enc_alloc) {
        ggml_gallocr_free(ctx->enc_alloc);
    }
    graph_arena_free(&ctx->mel_arena);
    graph_arena_free(&ctx->enc_arena);
    wctx_free(&ctx->wctx);
    wctx_free(&ctx->mel_wctx);
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    delete ctx;
}

int st_sample_rate(const st_context * ctx) {
    return ctx ? ctx->sample_rate : 0;
}

int st_window(const st_context * ctx) {
    return ctx ? ctx->window : 0;
}

float st_predict(st_context * ctx, const float * pcm, int n_samples) {
    if (!ctx || !pcm) {
        s2s_set_error("[SmartTurn] Context or pcm is NULL");
        return -1.0f;
    }
    if (n_samples <= 0) {
        s2s_set_error("[SmartTurn] Empty audio");
        return -1.0f;
    }

    std::lock_guard<std::mutex>  call(ctx->call_mutex);
    std::unique_lock<std::mutex> lock(ctx->job_mutex);

    ctx->job_pcm     = pcm;
    ctx->job_samples = n_samples;
    ctx->job_done    = false;
    ctx->job_ready   = true;
    ctx->job_cv.notify_all();
    ctx->job_cv.wait(lock, [ctx] { return ctx->job_done; });

    if (ctx->job_prob < 0.0f) {
        s2s_set_error("%s", ctx->job_error.c_str());
    }
    return ctx->job_prob;
}

const char * st_last_error(void) {
    return s2s_last_error();
}
