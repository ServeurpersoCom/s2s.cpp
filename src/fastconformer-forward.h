#pragma once
// fastconformer-forward.h: Parakeet TDT encoder, weights and graph
//
// The stack, from 16 kHz audio to encoder states:
//
//   mel      preemphasis 0.97, stft n_fft 512 win 400 hop 160 centered on
//            zeros, power, slaney filterbank 128, log(x + 2^-24), then a per
//            feature normalization over the utterance
//   stem     conv2d 1 -> 256 stride 2, then two depthwise plus pointwise
//            pairs at stride 2, relu between, so time and frequency both
//            shrink 8x, then a linear over channels times frequency -> 1024
//   blocks   24 macaron conformer blocks: half feed forward, relative
//            attention, convolution, half feed forward, output norm
//   project  1024 -> 640, the width the joint network consumes
//
// The attention follows Transformer-XL: the content term uses q + bias_u
// against the keys, the position term uses q + bias_v against the projected
// sinusoids and is realigned by the relative shift. Both terms share one
// scaling and are summed before the softmax.
//
// The graph depends on the number of frames, so it is rebuilt per utterance
// instead of living in a fixed arena. Building 24 blocks costs a few
// microseconds next to the compute they describe.

#include "audio-mel.h"
#include "conv-f32.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <string>
#include <vector>

#define PARAKEET_NORM_EPS  1e-5f
#define PARAKEET_LOG_GUARD 5.9604644775390625e-08f  // 2^-24

struct ParakeetHParams {
    int   sample_rate = 16000;
    int   n_fft       = 512;
    int   win_length  = 400;
    int   hop         = 160;
    int   n_mels      = 128;
    float preemphasis = 0.97f;
    int   n_layers    = 24;
    int   n_heads     = 8;
    int   d_model     = 1024;
    int   conv_kernel = 9;
    int   sub_factor  = 8;
    int   d_decoder   = 640;
};

struct ParakeetBlockWeights {
    struct ggml_tensor * norm_ff1_w  = nullptr;
    struct ggml_tensor * norm_ff1_b  = nullptr;
    struct ggml_tensor * ff1_linear1 = nullptr;
    struct ggml_tensor * ff1_linear2 = nullptr;
    struct ggml_tensor * norm_attn_w = nullptr;
    struct ggml_tensor * norm_attn_b = nullptr;
    struct ggml_tensor * q_w         = nullptr;
    struct ggml_tensor * k_w         = nullptr;
    struct ggml_tensor * v_w         = nullptr;
    struct ggml_tensor * o_w         = nullptr;
    struct ggml_tensor * rel_k_w     = nullptr;
    struct ggml_tensor * bias_u      = nullptr;
    struct ggml_tensor * bias_v      = nullptr;
    struct ggml_tensor * norm_conv_w = nullptr;
    struct ggml_tensor * norm_conv_b = nullptr;
    struct ggml_tensor * conv_pw1    = nullptr;
    struct ggml_tensor * conv_dw_w   = nullptr;
    struct ggml_tensor * conv_dw_b   = nullptr;
    struct ggml_tensor * conv_pw2    = nullptr;
    struct ggml_tensor * norm_ff2_w  = nullptr;
    struct ggml_tensor * norm_ff2_b  = nullptr;
    struct ggml_tensor * ff2_linear1 = nullptr;
    struct ggml_tensor * ff2_linear2 = nullptr;
    struct ggml_tensor * norm_out_w  = nullptr;
    struct ggml_tensor * norm_out_b  = nullptr;
};

struct ParakeetEncoderWeights {
    struct ggml_tensor * sub0_w       = nullptr;
    struct ggml_tensor * sub0_b       = nullptr;
    struct ggml_tensor * sub_dw_w[2]  = {};
    struct ggml_tensor * sub_dw_b[2]  = {};
    struct ggml_tensor * sub_pw_w[2]  = {};
    struct ggml_tensor * sub_pw_b[2]  = {};
    struct ggml_tensor * sub_linear_w = nullptr;
    struct ggml_tensor * sub_linear_b = nullptr;

    std::vector<ParakeetBlockWeights> blocks;

    struct ggml_tensor * proj_w = nullptr;
    struct ggml_tensor * proj_b = nullptr;
};

static void parakeet_read_hparams(const GGUFModel & gf, ParakeetHParams & hp) {
    hp.sample_rate = (int) gf_get_u32(gf, "asr.sample_rate");
    hp.n_fft       = (int) gf_get_u32(gf, "asr.n_fft");
    hp.win_length  = (int) gf_get_u32(gf, "asr.win_length");
    hp.hop         = (int) gf_get_u32(gf, "asr.hop");
    hp.n_mels      = (int) gf_get_u32(gf, "asr.n_mels");
    hp.preemphasis = gf_get_f32(gf, "asr.preemphasis");
    hp.n_layers    = (int) gf_get_u32(gf, "asr.n_layers");
    hp.n_heads     = (int) gf_get_u32(gf, "asr.n_heads");
    hp.d_model     = (int) gf_get_u32(gf, "asr.d_model");
    hp.conv_kernel = (int) gf_get_u32(gf, "asr.conv_kernel");
    hp.sub_factor  = (int) gf_get_u32(gf, "asr.subsampling_factor");
    hp.d_decoder   = (int) gf_get_u32(gf, "asr.d_decoder");
}

static AudioMelConfig parakeet_mel_config(const ParakeetHParams & hp) {
    AudioMelConfig cfg;
    cfg.sample_rate = hp.sample_rate;
    cfg.n_fft       = hp.n_fft;
    cfg.win_length  = hp.win_length;
    cfg.hop         = hp.hop;
    cfg.n_mels      = hp.n_mels;
    cfg.fmin        = 0.0f;
    cfg.fmax        = (float) hp.sample_rate * 0.5f;
    cfg.periodic    = false;
    return cfg;
}

// Matrices that feed ggml_mul_mat keep the type they were stored with, so a
// quantized file stays quantized. Norms, biases and the convolution kernels
// always land as f32: they are small, and the conv path has no quantized
// kernel.
static void parakeet_load_encoder(ParakeetEncoderWeights * w,
                                  WeightCtx *              wctx,
                                  const GGUFModel &        gf,
                                  const ParakeetHParams &  hp) {
    w->sub0_w = gf_load_tensor_f32(wctx, gf, "sub.0.weight");
    w->sub0_b = gf_load_tensor_f32(wctx, gf, "sub.0.bias");
    for (int i = 0; i < 2; i++) {
        const std::string p = "sub." + std::to_string(i + 1) + ".";
        w->sub_dw_w[i]      = gf_load_tensor_f32(wctx, gf, p + "dw.weight");
        w->sub_dw_b[i]      = gf_load_tensor_f32(wctx, gf, p + "dw.bias");
        w->sub_pw_w[i]      = gf_load_tensor_f32(wctx, gf, p + "pw.weight");
        w->sub_pw_b[i]      = gf_load_tensor_f32(wctx, gf, p + "pw.bias");
    }
    w->sub_linear_w = gf_load_tensor(wctx, gf, "sub.linear.weight");
    w->sub_linear_b = gf_load_tensor_f32(wctx, gf, "sub.linear.bias");

    w->blocks.resize((size_t) hp.n_layers);
    for (int i = 0; i < hp.n_layers; i++) {
        ParakeetBlockWeights & b = w->blocks[(size_t) i];
        const std::string      p = "enc." + std::to_string(i) + ".";
        b.norm_ff1_w             = gf_load_tensor_f32(wctx, gf, p + "norm_ff1.weight");
        b.norm_ff1_b             = gf_load_tensor_f32(wctx, gf, p + "norm_ff1.bias");
        b.ff1_linear1            = gf_load_tensor(wctx, gf, p + "ff1.linear1.weight");
        b.ff1_linear2            = gf_load_tensor(wctx, gf, p + "ff1.linear2.weight");
        b.norm_attn_w            = gf_load_tensor_f32(wctx, gf, p + "norm_attn.weight");
        b.norm_attn_b            = gf_load_tensor_f32(wctx, gf, p + "norm_attn.bias");
        b.q_w                    = gf_load_tensor(wctx, gf, p + "attn.q.weight");
        b.k_w                    = gf_load_tensor(wctx, gf, p + "attn.k.weight");
        b.v_w                    = gf_load_tensor(wctx, gf, p + "attn.v.weight");
        b.o_w                    = gf_load_tensor(wctx, gf, p + "attn.o.weight");
        b.rel_k_w                = gf_load_tensor(wctx, gf, p + "attn.rel_k.weight");
        b.bias_u                 = gf_load_tensor_f32(wctx, gf, p + "attn.bias_u");
        b.bias_v                 = gf_load_tensor_f32(wctx, gf, p + "attn.bias_v");
        b.norm_conv_w            = gf_load_tensor_f32(wctx, gf, p + "norm_conv.weight");
        b.norm_conv_b            = gf_load_tensor_f32(wctx, gf, p + "norm_conv.bias");
        b.conv_pw1               = gf_load_tensor(wctx, gf, p + "conv.pw1.weight");
        b.conv_dw_w              = gf_load_tensor_f32(wctx, gf, p + "conv.dw.weight");
        b.conv_dw_b              = gf_load_tensor_f32(wctx, gf, p + "conv.dw.bias");
        b.conv_pw2               = gf_load_tensor(wctx, gf, p + "conv.pw2.weight");
        b.norm_ff2_w             = gf_load_tensor_f32(wctx, gf, p + "norm_ff2.weight");
        b.norm_ff2_b             = gf_load_tensor_f32(wctx, gf, p + "norm_ff2.bias");
        b.ff2_linear1            = gf_load_tensor(wctx, gf, p + "ff2.linear1.weight");
        b.ff2_linear2            = gf_load_tensor(wctx, gf, p + "ff2.linear2.weight");
        b.norm_out_w             = gf_load_tensor_f32(wctx, gf, p + "norm_out.weight");
        b.norm_out_b             = gf_load_tensor_f32(wctx, gf, p + "norm_out.bias");
    }

    w->proj_w = gf_load_tensor(wctx, gf, "enc.proj.weight");
    w->proj_b = gf_load_tensor_f32(wctx, gf, "enc.proj.bias");
}

// Sinusoids for the relative positions T-1 down to -(T-1), sin and cos
// interleaved over the model dimension. Host side because the table is pure
// geometry: it depends on the frame count, never on the audio.
static void parakeet_positions(int n_tokens, int d_model, std::vector<float> & out) {
    const int n_pos = 2 * n_tokens - 1;
    out.assign((size_t) n_pos * (size_t) d_model, 0.0f);
    for (int p = 0; p < n_pos; p++) {
        const double position = (double) (n_tokens - 1 - p);
        for (int i = 0; i < d_model / 2; i++) {
            const double inv_freq = std::pow(10000.0, -(double) (2 * i) / (double) d_model);
            const double angle    = position * inv_freq;
            out[(size_t) p * (size_t) d_model + (size_t) (2 * i)]     = (float) std::sin(angle);
            out[(size_t) p * (size_t) d_model + (size_t) (2 * i + 1)] = (float) std::cos(angle);
        }
    }
}

static struct ggml_tensor * parakeet_layer_norm(struct ggml_context * ctx,
                                                struct ggml_tensor *  x,
                                                struct ggml_tensor *  weight,
                                                struct ggml_tensor *  bias) {
    x = ggml_norm(ctx, x, PARAKEET_NORM_EPS);
    x = ggml_mul(ctx, x, weight);
    return ggml_add(ctx, x, bias);
}

// Moves the channel axis of a [W, H, C] image to the innermost position so a
// 1x1 convolution becomes a plain matmul, and back again.
static struct ggml_tensor * parakeet_to_channel_first(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
}

static struct ggml_tensor * parakeet_to_channel_last(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
}

// Relative shift of the position scores. Prepending one zero per row and
// reading the result with a row stride one element shorter realigns every
// query onto its own position window, which is the shift of appendix B.
static struct ggml_tensor * parakeet_rel_shift(struct ggml_context * ctx, struct ggml_tensor * bd, int n_tokens) {
    const int64_t n_pos   = bd->ne[0];
    const int64_t n_heads = bd->ne[2];

    struct ggml_tensor * zeros =
        ggml_scale(ctx, ggml_cont(ctx, ggml_view_3d(ctx, bd, 1, n_tokens, n_heads, bd->nb[1], bd->nb[2], 0)), 0.0f);
    struct ggml_tensor * padded = ggml_cont(ctx, ggml_concat(ctx, zeros, bd, 0));

    return ggml_cont(ctx,
                     ggml_view_3d(ctx, padded, n_tokens, n_tokens, n_heads, n_pos * sizeof(float),
                                  (size_t) n_tokens * (n_pos + 1) * sizeof(float), (size_t) n_tokens * sizeof(float)));
}

// Mel in, encoder states out. out_stem and out_block0 expose the two stages
// the parity harness compares before the states: the subsampling output and
// the first conformer block. mel is the normalized spectrogram with
// ne = [n_mels, n_frames], positions is the sinusoid table [d_model, 2T-1]
// with T the token count after subsampling.
static struct ggml_tensor * parakeet_encoder_build(struct ggml_context *          ctx,
                                                   const ParakeetEncoderWeights & w,
                                                   const ParakeetHParams &        hp,
                                                   struct ggml_tensor *           mel,
                                                   struct ggml_tensor *           positions,
                                                   struct ggml_tensor **          out_stem   = nullptr,
                                                   struct ggml_tensor **          out_block0 = nullptr) {
    struct ggml_tensor * x = ggml_reshape_4d(ctx, mel, mel->ne[0], mel->ne[1], 1, 1);

    x = conv_2d_f32(ctx, w.sub0_w, x, 2, 1);
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, w.sub0_b, 1, 1, w.sub0_b->ne[0], 1));
    x = ggml_relu(ctx, x);

    for (int i = 0; i < 2; i++) {
        x = conv_2d_dw_f32(ctx, w.sub_dw_w[i], x, 2, 1);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, w.sub_dw_b[i], 1, 1, w.sub_dw_b[i]->ne[0], 1));

        x = parakeet_to_channel_first(ctx, x);
        x = ggml_mul_mat(ctx, w.sub_pw_w[i], x);
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, w.sub_pw_b[i], w.sub_pw_b[i]->ne[0], 1, 1));
        x = parakeet_to_channel_last(ctx, x);

        x = ggml_relu(ctx, x);
    }

    // [freq, tokens, channels] -> [freq, channels, tokens] so one token is a
    // contiguous run of channel outer, frequency inner, the order the linear
    // was trained on.
    const int64_t n_tokens = x->ne[1];
    x                      = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x                      = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], n_tokens);
    x                      = ggml_add(ctx, ggml_mul_mat(ctx, w.sub_linear_w, x), w.sub_linear_b);

    if (out_stem) {
        *out_stem = x;
    }

    const int   d_head = hp.d_model / hp.n_heads;
    const float scale  = 1.0f / sqrtf((float) d_head);

    for (const ParakeetBlockWeights & b : w.blocks) {
        struct ggml_tensor * residual = x;
        struct ggml_tensor * h        = parakeet_layer_norm(ctx, x, b.norm_ff1_w, b.norm_ff1_b);
        h                             = ggml_silu(ctx, ggml_mul_mat(ctx, b.ff1_linear1, h));
        h                             = ggml_mul_mat(ctx, b.ff1_linear2, h);
        x                             = ggml_add(ctx, residual, ggml_scale(ctx, h, 0.5f));

        residual = x;
        h        = parakeet_layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);

        struct ggml_tensor * q = ggml_mul_mat(ctx, b.q_w, h);
        struct ggml_tensor * k = ggml_mul_mat(ctx, b.k_w, h);
        struct ggml_tensor * v = ggml_mul_mat(ctx, b.v_w, h);

        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, d_head, hp.n_heads, n_tokens), 0, 2, 1, 3);
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, d_head, hp.n_heads, n_tokens), 0, 2, 1, 3));
        v = ggml_permute(ctx, ggml_reshape_3d(ctx, v, d_head, hp.n_heads, n_tokens), 0, 2, 1, 3);

        struct ggml_tensor * bias_u = ggml_reshape_3d(ctx, b.bias_u, d_head, 1, hp.n_heads);
        struct ggml_tensor * bias_v = ggml_reshape_3d(ctx, b.bias_v, d_head, 1, hp.n_heads);

        struct ggml_tensor * q_u = ggml_cont(ctx, ggml_add(ctx, q, bias_u));
        struct ggml_tensor * q_v = ggml_cont(ctx, ggml_add(ctx, q, bias_v));

        struct ggml_tensor * rel_k = ggml_mul_mat(ctx, b.rel_k_w, positions);
        rel_k                      = ggml_cont(
            ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, rel_k, d_head, hp.n_heads, positions->ne[1]), 0, 2, 1, 3));

        struct ggml_tensor * ac = ggml_mul_mat(ctx, k, q_u);
        struct ggml_tensor * bd = parakeet_rel_shift(ctx, ggml_mul_mat(ctx, rel_k, q_v), (int) n_tokens);

        struct ggml_tensor * scores = ggml_soft_max(ctx, ggml_scale(ctx, ggml_add(ctx, ac, bd), scale));

        struct ggml_tensor * vt  = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, vt, scores);

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx, kqv, hp.d_model, n_tokens);

        x = ggml_add(ctx, residual, ggml_mul_mat(ctx, b.o_w, kqv));

        residual = x;
        h        = parakeet_layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);

        struct ggml_tensor * gated = ggml_mul_mat(ctx, b.conv_pw1, h);
        struct ggml_tensor * lhs   = ggml_cont(ctx, ggml_view_2d(ctx, gated, hp.d_model, n_tokens, gated->nb[1], 0));
        struct ggml_tensor * rhs   = ggml_cont(
            ctx, ggml_view_2d(ctx, gated, hp.d_model, n_tokens, gated->nb[1], (size_t) hp.d_model * sizeof(float)));
        h = ggml_mul(ctx, lhs, ggml_sigmoid(ctx, rhs));

        h = ggml_cont(ctx, ggml_transpose(ctx, h));
        h = conv_1d_dw_f32(ctx, b.conv_dw_w, h, 1, (hp.conv_kernel - 1) / 2);
        h = ggml_cont(ctx, ggml_transpose(ctx, h));
        h = ggml_silu(ctx, ggml_add(ctx, h, b.conv_dw_b));
        h = ggml_mul_mat(ctx, b.conv_pw2, h);

        x = ggml_add(ctx, residual, h);

        residual = x;
        h        = parakeet_layer_norm(ctx, x, b.norm_ff2_w, b.norm_ff2_b);
        h        = ggml_silu(ctx, ggml_mul_mat(ctx, b.ff2_linear1, h));
        h        = ggml_mul_mat(ctx, b.ff2_linear2, h);
        x        = ggml_add(ctx, residual, ggml_scale(ctx, h, 0.5f));

        x = parakeet_layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);

        if (out_block0 && &b == &w.blocks[0]) {
            *out_block0 = x;
        }
    }

    return x;
}

// Encoder states to the width the joint network reads.
static struct ggml_tensor * parakeet_project(struct ggml_context *          ctx,
                                             const ParakeetEncoderWeights & w,
                                             struct ggml_tensor *           states) {
    return ggml_add(ctx, ggml_mul_mat(ctx, w.proj_w, states), w.proj_b);
}
