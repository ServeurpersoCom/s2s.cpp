// localvqe.cpp: LocalVQE v1.3 streaming forward pass on GGML
//
// One hop of 256 samples at 16 kHz per call, microphone and far end
// reference side by side. Each goes through a 512 point analysis over the
// previous hop and the current one, a sqrt-Hann window folded into the
// weights, which yields 256 bins of two channels per frame. Then:
//
//   power law compression                  -> [256, 1, 2]
//   mic encoders 1-2, far encoders 1-2     -> [64, 1, C] each
//   soft delay attention over dmax frames  -> far end aligned on the mic
//   mic encoders 3-5 on the concat         -> [8, 1, C]
//   diagonal state space bottleneck
//   subpixel decoders 5-1 with skips       -> [256, 1, 27] mask
//   3x3 complex convolving mask on the mic analysis
//   synthesis to a 512 sample frame, overlap-added by the caller
//
// Every layer is causal. A conv reads the KT - 1 frames it saw before plus
// the current one, the alignment reads dmax frames of the far end, the mask
// reads two past frames, the bottleneck carries its complex state: all of it
// enters the graph as slots of lv_state and leaves as their next value, so
// the graph has fixed shapes and is built once.

#include "localvqe.h"

#include "backend.h"
#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "graph-arena.h"
#include "s2s-error.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define LV_MAX_NODES   4096
#define LV_KT          4  // time taps of every encoder and decoder conv
#define LV_ALIGN_KT    5  // time taps of the alignment smoothing conv
#define LV_MASK_KT     3  // time taps of the complex mask
#define LV_N_ENCODER   5
#define LV_N_FAR       2
#define LV_N_DECODER   5
#define LV_MAG_EPS     1e-12f
#define LV_FREQ_PAD_LO 1  // the (1, 2) frequency pad of a width 4 kernel
#define LV_FREQ_PAD_HI 2

struct LvNorm {
    struct ggml_tensor * w = nullptr;  // [C]
    struct ggml_tensor * b = nullptr;  // [C]
};

struct LvConv {
    struct ggml_tensor * w = nullptr;  // [KW, KH, IC, OC]
    struct ggml_tensor * b = nullptr;  // [OC]
};

struct LvPoint {
    struct ggml_tensor * w = nullptr;  // [IC, OC]
    struct ggml_tensor * b = nullptr;  // [OC]
};

struct LvResidual {
    LvNorm norm;
    LvConv conv;
};

struct LvEncoder {
    LvNorm     norm;
    LvConv     conv;
    LvResidual res;
};

struct LvDecoder {
    LvNorm     skip_norm;
    LvPoint    skip;
    LvResidual res;
    LvNorm     norm;
    LvConv     conv;
};

// Per stream memory the graph reads from `in` and writes back through `out`.
struct LvSlot {
    struct ggml_tensor * in     = nullptr;
    struct ggml_tensor * out    = nullptr;
    size_t               offset = 0;  // in floats, inside lv_state::slots
};

struct lv_context {
    BackendPair bp = {};

    struct ggml_tensor * analysis  = nullptr;  // [512, 512]
    struct ggml_tensor * synthesis = nullptr;  // [512, 512]

    LvEncoder mic[LV_N_ENCODER];
    LvEncoder far[LV_N_FAR];
    LvDecoder dec[LV_N_DECODER];

    LvPoint              align_q;
    LvPoint              align_k;
    LvConv               align_smooth;  // [3, 5, H, 1]
    struct ggml_tensor * bn_in_w  = nullptr;
    struct ggml_tensor * bn_in_b  = nullptr;
    struct ggml_tensor * bn_out_w = nullptr;
    struct ggml_tensor * bn_out_b = nullptr;
    struct ggml_tensor * bn_a_re  = nullptr;
    struct ggml_tensor * bn_a_im  = nullptr;
    struct ggml_tensor * bn_b_re  = nullptr;
    struct ggml_tensor * bn_b_im  = nullptr;
    struct ggml_tensor * bn_c_re  = nullptr;
    struct ggml_tensor * bn_c_im  = nullptr;
    struct ggml_tensor * bn_d     = nullptr;
    struct ggml_tensor * mask_re  = nullptr;  // [3]
    struct ggml_tensor * mask_im  = nullptr;  // [3]

    WeightCtx      wctx  = {};
    GraphArena     arena = {};
    ggml_gallocr_t alloc = nullptr;

    struct ggml_cgraph * graph   = nullptr;
    struct ggml_tensor * in_mic  = nullptr;  // [n_fft]
    struct ggml_tensor * in_ref  = nullptr;  // [n_fft]
    struct ggml_tensor * out_pcm = nullptr;  // [n_fft]

    std::vector<LvSlot> slots;
    size_t              state_size = 0;

    int   sample_rate = 0;
    int   n_fft       = 0;
    int   hop         = 0;
    int   dmax        = 0;
    float power_law   = 0.0f;
    float eps         = 0.0f;

    std::mutex mutex;
};

struct lv_state {
    lv_context *       ctx = nullptr;
    std::vector<float> slots;     // every graph slot, back to back
    std::vector<float> mic_prev;  // [hop] the half of the analysis window kept
    std::vector<float> ref_prev;  // [hop]
    std::vector<float> mic_win;   // [n_fft]
    std::vector<float> ref_win;   // [n_fft]
    std::vector<float> frame;     // [n_fft]
    std::vector<float> ola;       // [n_fft]
};

// A slot of shape [ne0, ne1, ne2], zero at the start of a stream. Returns
// its index in ctx->slots.
static size_t lv_slot(lv_context * ctx, struct ggml_context * g, int64_t ne0, int64_t ne1, int64_t ne2) {
    LvSlot slot;
    slot.in     = ggml_new_tensor_3d(g, GGML_TYPE_F32, ne0, ne1, ne2);
    slot.offset = ctx->state_size;
    ggml_set_input(slot.in);
    ctx->state_size += (size_t) ggml_nelements(slot.in);
    ctx->slots.push_back(slot);
    return ctx->slots.size() - 1;
}

// Appends the current frame to the frames a slot holds along dim 1, and
// stores the most recent ones back as the next value of the slot.
static struct ggml_tensor * lv_window(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * x, int n_past) {
    LvSlot &             slot = ctx->slots[lv_slot(ctx, g, x->ne[0], n_past, x->ne[2])];
    struct ggml_tensor * win  = ggml_concat(g, slot.in, x, 1);
    slot.out = ggml_cont(g, ggml_view_3d(g, win, win->ne[0], n_past, win->ne[2], win->nb[1], win->nb[2], win->nb[1]));
    ggml_set_output(slot.out);
    return win;
}

static struct ggml_tensor * lv_norm(lv_context *          ctx,
                                    struct ggml_context * g,
                                    const LvNorm &        n,
                                    struct ggml_tensor *  x) {
    // One frame at a time: the statistics run over channels and frequency.
    struct ggml_tensor * y = ggml_norm(g, ggml_reshape_1d(g, ggml_cont(g, x), ggml_nelements(x)), ctx->eps);
    y                      = ggml_reshape_3d(g, y, x->ne[0], x->ne[1], x->ne[2]);
    y                      = ggml_mul(g, y, ggml_reshape_3d(g, n.w, 1, 1, n.w->ne[0]));
    return ggml_add(g, y, ggml_reshape_3d(g, n.b, 1, 1, n.b->ne[0]));
}

// Causal conv, KT frames on time, (1, 2) padded on frequency, stride on
// frequency only. The im2col stays in f32. Returns one frame [OF, 1, OC].
static struct ggml_tensor * lv_conv(lv_context *          ctx,
                                    struct ggml_context * g,
                                    const LvConv &        c,
                                    struct ggml_tensor *  x,
                                    int                   stride) {
    struct ggml_tensor * win = lv_window(ctx, g, x, LV_KT - 1);
    win                      = ggml_pad_ext(g, win, LV_FREQ_PAD_LO, LV_FREQ_PAD_HI, 0, 0, 0, 0, 0, 0);

    struct ggml_tensor * col = ggml_im2col(g, c.w, win, stride, 1, 0, 0, 1, 1, true, GGML_TYPE_F32);
    struct ggml_tensor * y   = ggml_mul_mat(g, ggml_reshape_2d(g, col, col->ne[0], col->ne[1] * col->ne[2]),
                                            ggml_reshape_2d(g, c.w, c.w->ne[0] * c.w->ne[1] * c.w->ne[2], c.w->ne[3]));
    y                        = ggml_reshape_3d(g, y, col->ne[1], 1, c.w->ne[3]);
    return ggml_add(g, y, ggml_reshape_3d(g, c.b, 1, 1, c.b->ne[0]));
}

// 1x1 conv on one frame [F, 1, IC] -> [F, 1, OC].
static struct ggml_tensor * lv_point(struct ggml_context * g, const LvPoint & p, struct ggml_tensor * x) {
    struct ggml_tensor * xt = ggml_cont(g, ggml_transpose(g, ggml_reshape_2d(g, ggml_cont(g, x), x->ne[0], x->ne[2])));
    struct ggml_tensor * y  = ggml_add(g, ggml_mul_mat(g, xt, p.w), ggml_reshape_2d(g, p.b, 1, p.b->ne[0]));
    return ggml_reshape_3d(g, y, x->ne[0], 1, p.w->ne[1]);
}

static struct ggml_tensor * lv_residual(lv_context *          ctx,
                                        struct ggml_context * g,
                                        const LvResidual &    r,
                                        struct ggml_tensor *  x) {
    return ggml_add(g, ggml_silu(g, lv_conv(ctx, g, r.conv, lv_norm(ctx, g, r.norm, x), 1)), x);
}

static struct ggml_tensor * lv_encoder(lv_context *          ctx,
                                       struct ggml_context * g,
                                       const LvEncoder &     e,
                                       struct ggml_tensor *  x) {
    x = ggml_silu(g, lv_conv(ctx, g, e.conv, lv_norm(ctx, g, e.norm, x), 2));
    return lv_residual(ctx, g, e.res, x);
}

// The subpixel conv doubles the frequency: output channel r * OC + c lands
// on bin r * F + f, then the result is trimmed to the width of the skip.
static struct ggml_tensor * lv_decoder(lv_context *          ctx,
                                       struct ggml_context * g,
                                       const LvDecoder &     d,
                                       struct ggml_tensor *  x,
                                       struct ggml_tensor *  skip,
                                       int64_t               n_freq,
                                       bool                  last) {
    struct ggml_tensor * y = ggml_add(g, x, lv_point(g, d.skip, lv_norm(ctx, g, d.skip_norm, skip)));
    y                      = lv_residual(ctx, g, d.res, y);
    y                      = lv_conv(ctx, g, d.conv, lv_norm(ctx, g, d.norm, y), 1);

    const int64_t        oc = y->ne[2] / 2;
    struct ggml_tensor * lo = ggml_cont(g, ggml_view_3d(g, y, y->ne[0], 1, oc, y->nb[1], y->nb[2], 0));
    struct ggml_tensor * hi =
        ggml_cont(g, ggml_view_3d(g, y, y->ne[0], 1, oc, y->nb[1], y->nb[2], (size_t) oc * y->nb[2]));
    y = ggml_concat(g, lo, hi, 0);
    if (y->ne[0] != n_freq) {
        y = ggml_cont(g, ggml_view_3d(g, y, n_freq, 1, oc, y->nb[1], y->nb[2], 0));
    }
    return last ? y : ggml_silu(g, y);
}

// Cross-attention between the current mic frame and the last dmax far end
// frames, a softmax over the delay, and the far end frames weighted by it.
static struct ggml_tensor * lv_align(lv_context *          ctx,
                                     struct ggml_context * g,
                                     struct ggml_tensor *  mic,
                                     struct ggml_tensor *  far) {
    const int64_t n_freq = mic->ne[0];

    struct ggml_tensor * q = lv_point(g, ctx->align_q, mic);
    struct ggml_tensor * k = lv_window(ctx, g, lv_point(g, ctx->align_k, far), ctx->dmax - 1);  // [F, dmax, H]

    struct ggml_tensor * v = ggml_sum_rows(g, ggml_mul(g, k, q));                               // [1, dmax, H]
    v                      = ggml_scale(g, ggml_reshape_3d(g, v, ctx->dmax, 1, v->ne[2]), 1.0f / sqrtf((float) n_freq));

    struct ggml_tensor * vwin = lv_window(ctx, g, v, LV_ALIGN_KT - 1);  // [dmax, 5, H]
    vwin                      = ggml_pad_ext(g, vwin, 1, 1, 0, 0, 0, 0, 0, 0);

    struct ggml_tensor * col =
        ggml_im2col(g, ctx->align_smooth.w, vwin, 1, 1, 0, 0, 1, 1, true, GGML_TYPE_F32);  // [3 * 5 * H, dmax, 1]
    struct ggml_tensor * score =
        ggml_mul_mat(g, ggml_reshape_2d(g, col, col->ne[0], col->ne[1]),
                     ggml_reshape_2d(g, ctx->align_smooth.w, ggml_nelements(ctx->align_smooth.w), 1));  // [dmax, 1]
    score                        = ggml_add(g, score, ctx->align_smooth.b);
    struct ggml_tensor * weights = ggml_soft_max(g, score);

    struct ggml_tensor * refs = lv_window(ctx, g, far, ctx->dmax - 1);            // [F, dmax, C]
    refs                      = ggml_cont(g, ggml_permute(g, refs, 1, 0, 2, 3));  // [dmax, F, C]
    struct ggml_tensor * aligned =
        ggml_mul_mat(g, ggml_reshape_2d(g, refs, ctx->dmax, refs->ne[1] * refs->ne[2]), weights);
    return ggml_reshape_3d(g, aligned, n_freq, 1, far->ne[2]);
}

// Diagonal state space step: h = a h + b v on complex numbers, y = Re(c h).
static struct ggml_tensor * lv_bottleneck(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * x) {
    struct ggml_tensor * u = ggml_reshape_1d(g, ggml_cont(g, x), ggml_nelements(x));
    struct ggml_tensor * v = ggml_add(g, ggml_mul_mat(g, ctx->bn_in_w, u), ctx->bn_in_b);

    const int64_t        n_state = v->ne[0];
    const size_t         index   = lv_slot(ctx, g, n_state, 2, 1);
    LvSlot &             slot    = ctx->slots[index];
    struct ggml_tensor * h_re    = ggml_view_1d(g, slot.in, n_state, 0);
    struct ggml_tensor * h_im    = ggml_view_1d(g, slot.in, n_state, (size_t) n_state * sizeof(float));

    struct ggml_tensor * re = ggml_sub(g, ggml_mul(g, ctx->bn_a_re, h_re), ggml_mul(g, ctx->bn_a_im, h_im));
    re                      = ggml_add(g, re, ggml_mul(g, ctx->bn_b_re, v));
    struct ggml_tensor * im = ggml_add(g, ggml_mul(g, ctx->bn_a_re, h_im), ggml_mul(g, ctx->bn_a_im, h_re));
    im                      = ggml_add(g, im, ggml_mul(g, ctx->bn_b_im, v));

    slot.out = ggml_reshape_3d(g, ggml_concat(g, re, im, 0), n_state, 2, 1);
    ggml_set_output(slot.out);

    struct ggml_tensor * y   = ggml_sub(g, ggml_mul(g, ctx->bn_c_re, re), ggml_mul(g, ctx->bn_c_im, im));
    struct ggml_tensor * out = ggml_add(g, ggml_mul_mat(g, ctx->bn_out_w, y), ctx->bn_out_b);
    out                      = ggml_add(g, out, ggml_mul(g, ctx->bn_d, u));
    return ggml_reshape_3d(g, out, x->ne[0], 1, x->ne[2]);
}

// Complex convolving mask: 27 channels are 3 cube root of unity components
// times a 3x3 kernel over (time t-2..t, frequency f-1..f+1), applied to the
// mic analysis. Returns the enhanced frame as [2, F], real and imaginary
// interleaved per bin, the layout the synthesis reads.
static struct ggml_tensor * lv_mask(lv_context *          ctx,
                                    struct ggml_context * g,
                                    struct ggml_tensor *  m,
                                    struct ggml_tensor *  spec) {
    const int64_t n_freq = spec->ne[0];

    struct ggml_tensor * mr = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, m, n_freq, 9, 3), 1, 2, 0, 3));
    mr                      = ggml_reshape_2d(g, mr, 3, n_freq * 9);  // [3, F * 9]
    struct ggml_tensor * hre =
        ggml_reshape_2d(g, ggml_mul_mat(g, mr, ggml_reshape_2d(g, ctx->mask_re, 3, 1)), n_freq, 9);
    struct ggml_tensor * him =
        ggml_reshape_2d(g, ggml_mul_mat(g, mr, ggml_reshape_2d(g, ctx->mask_im, 3, 1)), n_freq, 9);

    struct ggml_tensor * win = lv_window(ctx, g, spec, LV_MASK_KT - 1);  // [F, 3, 2]
    win                      = ggml_cont(g, ggml_pad_ext(g, win, 1, 1, 0, 0, 0, 0, 0, 0));

    const int64_t        width = n_freq + 2;
    struct ggml_tensor * taps[2];
    for (int c = 0; c < 2; c++) {
        struct ggml_tensor * t = nullptr;
        for (int k = 0; k < 9; k++) {
            const size_t offset =
                ((size_t) (k % 3) + (size_t) (k / 3) * width + (size_t) c * width * LV_MASK_KT) * sizeof(float);
            struct ggml_tensor * tap = ggml_view_2d(g, win, n_freq, 1, win->nb[1], offset);
            t                        = t ? ggml_concat(g, t, tap, 1) : ggml_cont(g, tap);
        }
        taps[c] = t;  // [F, 9]
    }

    struct ggml_tensor * re = ggml_sub(g, ggml_mul(g, hre, taps[0]), ggml_mul(g, him, taps[1]));
    struct ggml_tensor * im = ggml_add(g, ggml_mul(g, hre, taps[1]), ggml_mul(g, him, taps[0]));
    re                      = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, re)));  // [1, F]
    im                      = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, im)));
    return ggml_concat(g, re, im, 0);                                                 // [2, F]
}

// Analysis of one 512 sample window into [F, 1, 2], real then imaginary.
static struct ggml_tensor * lv_analysis(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * pcm) {
    struct ggml_tensor * v = ggml_mul_mat(g, ctx->analysis, pcm);  // bins interleaved as re, im
    v                      = ggml_cont(g, ggml_transpose(g, ggml_reshape_2d(g, v, 2, ctx->n_fft / 2)));
    return ggml_reshape_3d(g, v, ctx->n_fft / 2, 1, 2);
}

// Power law compression of the magnitude, the phase kept.
static struct ggml_tensor * lv_compress(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * spec) {
    const int64_t        n_freq = spec->ne[0];
    struct ggml_tensor * re     = ggml_view_1d(g, spec, n_freq, 0);
    struct ggml_tensor * im     = ggml_view_1d(g, spec, n_freq, (size_t) n_freq * sizeof(float));
    struct ggml_tensor * mag =
        ggml_sqrt(g, ggml_scale_bias(g, ggml_add(g, ggml_sqr(g, re), ggml_sqr(g, im)), 1.0f, LV_MAG_EPS));
    struct ggml_tensor * div =
        ggml_scale_bias(g, ggml_exp(g, ggml_scale(g, ggml_log(g, mag), 1.0f - ctx->power_law)), 1.0f, LV_MAG_EPS);
    return ggml_div(g, spec, ggml_reshape_3d(g, div, n_freq, 1, 1));
}

static void lv_build_graph(lv_context * ctx) {
    struct ggml_context * g = graph_arena_begin(&ctx->arena);
    ctx->graph              = ggml_new_graph_custom(g, LV_MAX_NODES, false);

    ctx->in_mic = ggml_new_tensor_1d(g, GGML_TYPE_F32, ctx->n_fft);
    ctx->in_ref = ggml_new_tensor_1d(g, GGML_TYPE_F32, ctx->n_fft);
    ggml_set_input(ctx->in_mic);
    ggml_set_input(ctx->in_ref);

    struct ggml_tensor * mic_spec = lv_analysis(ctx, g, ctx->in_mic);
    struct ggml_tensor * ref_spec = lv_analysis(ctx, g, ctx->in_ref);

    struct ggml_tensor * skips[LV_N_ENCODER];
    struct ggml_tensor * x = lv_compress(ctx, g, mic_spec);
    for (int i = 0; i < 2; i++) {
        x        = lv_encoder(ctx, g, ctx->mic[i], x);
        skips[i] = x;
    }

    struct ggml_tensor * far = lv_compress(ctx, g, ref_spec);
    for (int i = 0; i < LV_N_FAR; i++) {
        far = lv_encoder(ctx, g, ctx->far[i], far);
    }

    x = ggml_concat(g, x, lv_align(ctx, g, x, far), 2);
    for (int i = 2; i < LV_N_ENCODER; i++) {
        x        = lv_encoder(ctx, g, ctx->mic[i], x);
        skips[i] = x;
    }

    x = lv_bottleneck(ctx, g, x);
    for (int i = LV_N_ENCODER - 1; i >= 0; i--) {
        const int64_t n_freq = i > 0 ? skips[i - 1]->ne[0] : mic_spec->ne[0];
        x                    = lv_decoder(ctx, g, ctx->dec[i], x, skips[i], n_freq, i == 0);
    }

    struct ggml_tensor * enhanced = lv_mask(ctx, g, x, mic_spec);
    ctx->out_pcm = ggml_mul_mat(g, ctx->synthesis, ggml_reshape_1d(g, enhanced, ggml_nelements(enhanced)));
    ggml_set_output(ctx->out_pcm);

    ggml_build_forward_expand(ctx->graph, ctx->out_pcm);
    for (const LvSlot & slot : ctx->slots) {
        ggml_build_forward_expand(ctx->graph, slot.out);
    }
}

static void lv_load_norm(lv_context * ctx, const GGUFModel & gf, LvNorm & n, const std::string & name) {
    n.w = gf_load_tensor_f32(&ctx->wctx, gf, name + ".weight");
    n.b = gf_load_tensor_f32(&ctx->wctx, gf, name + ".bias");
}

static void lv_load_conv(lv_context * ctx, const GGUFModel & gf, LvConv & c, const std::string & name) {
    c.w = gf_load_tensor_f32(&ctx->wctx, gf, name + ".weight");
    c.b = gf_load_tensor_f32(&ctx->wctx, gf, name + ".bias");
}

static void lv_load_point(lv_context * ctx, const GGUFModel & gf, LvPoint & p, const std::string & name) {
    p.w = gf_load_tensor_f32(&ctx->wctx, gf, name + ".weight");
    p.b = gf_load_tensor_f32(&ctx->wctx, gf, name + ".bias");
}

static void lv_load_encoder(lv_context * ctx, const GGUFModel & gf, LvEncoder & e, const std::string & name) {
    lv_load_norm(ctx, gf, e.norm, name + ".norm");
    lv_load_conv(ctx, gf, e.conv, name + ".conv");
    lv_load_norm(ctx, gf, e.res.norm, name + ".resblock.norm");
    lv_load_conv(ctx, gf, e.res.conv, name + ".resblock.conv");
}

static void lv_load_decoder(lv_context * ctx, const GGUFModel & gf, LvDecoder & d, const std::string & name) {
    lv_load_norm(ctx, gf, d.skip_norm, name + ".skip_norm");
    lv_load_point(ctx, gf, d.skip, name + ".skip_conv");
    lv_load_norm(ctx, gf, d.res.norm, name + ".resblock.norm");
    lv_load_conv(ctx, gf, d.res.conv, name + ".resblock.conv");
    lv_load_norm(ctx, gf, d.norm, name + ".deconv.norm");
    lv_load_conv(ctx, gf, d.conv, name + ".deconv.conv");
}

lv_context * lv_init(const char * gguf_path, int use_gpu, int n_threads) {
    if (!gguf_path) {
        s2s_set_error("[LocalVQE] Gguf_path is NULL");
        return nullptr;
    }

    lv_context * ctx = new lv_context();

    try {
        GGUFModel gf = {};
        if (!gf_load(&gf, gguf_path)) {
            s2s_set_error("[LocalVQE] Failed to open %s", gguf_path);
            delete ctx;
            return nullptr;
        }

        ctx->sample_rate = (int) gf_get_u32(gf, "lv.sample_rate");
        ctx->n_fft       = (int) gf_get_u32(gf, "lv.n_fft");
        ctx->hop         = (int) gf_get_u32(gf, "lv.hop");
        ctx->dmax        = (int) gf_get_u32(gf, "lv.dmax");
        ctx->power_law   = gf_get_f32(gf, "lv.power_law_c");
        ctx->eps         = gf_get_f32(gf, "lv.norm_eps");

        ctx->bp = use_gpu ? backend_init("LocalVQE") : backend_init_cpu("LocalVQE", n_threads);
        if (!ctx->bp.backend) {
            s2s_set_error("[LocalVQE] Failed to init the backend");
            gf_close(&gf);
            delete ctx;
            return nullptr;
        }

        wctx_init(&ctx->wctx, (int) gguf_get_n_tensors(gf.gguf));
        ctx->analysis  = gf_load_tensor_f32(&ctx->wctx, gf, "encoder.conv.weight");
        ctx->synthesis = gf_load_tensor_f32(&ctx->wctx, gf, "decoder.linear.weight");
        for (int i = 0; i < LV_N_ENCODER; i++) {
            lv_load_encoder(ctx, gf, ctx->mic[i], "mic_enc" + std::to_string(i + 1));
            lv_load_decoder(ctx, gf, ctx->dec[i], "dec" + std::to_string(i + 1));
        }
        for (int i = 0; i < LV_N_FAR; i++) {
            lv_load_encoder(ctx, gf, ctx->far[i], "far_enc" + std::to_string(i + 1));
        }
        lv_load_point(ctx, gf, ctx->align_q, "align.pconv_mic");
        lv_load_point(ctx, gf, ctx->align_k, "align.pconv_ref");
        lv_load_conv(ctx, gf, ctx->align_smooth, "align.conv.1");
        ctx->bn_in_w  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.input_proj.weight");
        ctx->bn_in_b  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.input_proj.bias");
        ctx->bn_out_w = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.output_proj.weight");
        ctx->bn_out_b = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.output_proj.bias");
        ctx->bn_a_re  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.a_real");
        ctx->bn_a_im  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.a_imag");
        ctx->bn_b_re  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.B_real");
        ctx->bn_b_im  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.B_imag");
        ctx->bn_c_re  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.C_real");
        ctx->bn_c_im  = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.C_imag");
        ctx->bn_d     = gf_load_tensor_f32(&ctx->wctx, gf, "bottleneck.D");
        ctx->mask_re  = gf_load_tensor_f32(&ctx->wctx, gf, "mask.v_real");
        ctx->mask_im  = gf_load_tensor_f32(&ctx->wctx, gf, "mask.v_imag");

        const bool loaded = wctx_alloc(&ctx->wctx, ctx->bp.backend);
        gf_close(&gf);
        if (!loaded) {
            s2s_set_error("[LocalVQE] Failed to upload the weights");
            lv_free(ctx);
            return nullptr;
        }

        if (!graph_arena_init(&ctx->arena, LV_MAX_NODES)) {
            s2s_set_error("[LocalVQE] Failed to allocate the graph arena");
            lv_free(ctx);
            return nullptr;
        }

        lv_build_graph(ctx);

        ctx->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->bp.backend));
        if (!ctx->alloc || !ggml_gallocr_alloc_graph(ctx->alloc, ctx->graph)) {
            s2s_set_error("[LocalVQE] Failed to allocate the compute graph");
            lv_free(ctx);
            return nullptr;
        }

        s2s_log(S2S_LOG_INFO, "[LocalVQE] %d Hz, hop %d, dmax %d, %zu state slots of %.1f KB, %d graph nodes",
                ctx->sample_rate, ctx->hop, ctx->dmax, ctx->slots.size(),
                (double) ctx->state_size * sizeof(float) / 1024.0, ggml_graph_n_nodes(ctx->graph));
        return ctx;
    } catch (const std::exception & e) {
        s2s_set_error("%s", e.what());
        lv_free(ctx);
        return nullptr;
    }
}

void lv_free(lv_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->alloc) {
        ggml_gallocr_free(ctx->alloc);
    }
    graph_arena_free(&ctx->arena);
    wctx_free(&ctx->wctx);
    backend_release(ctx->bp.backend, ctx->bp.cpu_backend);
    delete ctx;
}

int lv_hop(const lv_context * ctx) {
    return ctx ? ctx->hop : 0;
}

int lv_sample_rate(const lv_context * ctx) {
    return ctx ? ctx->sample_rate : 0;
}

lv_state * lv_state_new(lv_context * ctx) {
    if (!ctx) {
        s2s_set_error("[LocalVQE] Context is NULL");
        return nullptr;
    }
    lv_state * state = new lv_state();
    state->ctx       = ctx;
    state->slots.assign(ctx->state_size, 0.0f);
    state->mic_prev.assign((size_t) ctx->hop, 0.0f);
    state->ref_prev.assign((size_t) ctx->hop, 0.0f);
    state->mic_win.assign((size_t) ctx->n_fft, 0.0f);
    state->ref_win.assign((size_t) ctx->n_fft, 0.0f);
    state->frame.assign((size_t) ctx->n_fft, 0.0f);
    state->ola.assign((size_t) ctx->n_fft, 0.0f);
    return state;
}

void lv_state_reset(lv_state * state) {
    if (!state) {
        return;
    }
    std::fill(state->slots.begin(), state->slots.end(), 0.0f);
    std::fill(state->mic_prev.begin(), state->mic_prev.end(), 0.0f);
    std::fill(state->ref_prev.begin(), state->ref_prev.end(), 0.0f);
    std::fill(state->ola.begin(), state->ola.end(), 0.0f);
}

void lv_state_free(lv_state * state) {
    delete state;
}

int lv_process(lv_state * state, const float * mic, const float * ref, float * out) {
    if (!state || !mic || !ref || !out) {
        s2s_set_error("[LocalVQE] State or buffer is NULL");
        return -1;
    }

    lv_context * ctx  = state->ctx;
    const size_t hop  = (size_t) ctx->hop;
    const size_t half = hop * sizeof(float);

    // The analysis window is the previous hop followed by the current one.
    memcpy(state->mic_win.data(), state->mic_prev.data(), half);
    memcpy(state->mic_win.data() + hop, mic, half);
    memcpy(state->ref_win.data(), state->ref_prev.data(), half);
    memcpy(state->ref_win.data() + hop, ref, half);
    memcpy(state->mic_prev.data(), mic, half);
    memcpy(state->ref_prev.data(), ref, half);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);

        ggml_backend_tensor_set(ctx->in_mic, state->mic_win.data(), 0, state->mic_win.size() * sizeof(float));
        ggml_backend_tensor_set(ctx->in_ref, state->ref_win.data(), 0, state->ref_win.size() * sizeof(float));
        for (const LvSlot & slot : ctx->slots) {
            ggml_backend_tensor_set(slot.in, state->slots.data() + slot.offset, 0, ggml_nbytes(slot.in));
        }

        if (ggml_backend_graph_compute(ctx->bp.backend, ctx->graph) != GGML_STATUS_SUCCESS) {
            s2s_set_error("[LocalVQE] Graph compute failed");
            return -1;
        }

        ggml_backend_tensor_get(ctx->out_pcm, state->frame.data(), 0, state->frame.size() * sizeof(float));
        for (const LvSlot & slot : ctx->slots) {
            ggml_backend_tensor_get(slot.out, state->slots.data() + slot.offset, 0, ggml_nbytes(slot.out));
        }
    }

    // The sqrt-Hann pair sums to one at half overlap: plain overlap-add.
    for (size_t i = 0; i < state->ola.size(); i++) {
        state->ola[i] += state->frame[i];
    }
    memcpy(out, state->ola.data(), half);
    memmove(state->ola.data(), state->ola.data() + hop, state->ola.size() * sizeof(float) - half);
    memset(state->ola.data() + state->ola.size() - hop, 0, half);
    return 0;
}

const char * lv_last_error(void) {
    return s2s_last_error();
}
