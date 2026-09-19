// localvqe.cpp: LocalVQE v1.3 streaming forward pass on GGML, batched
//
// One hop of 256 samples at 16 kHz per stream, microphone and far end
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
// Every per stream tensor carries the streams on its last dim, so one graph
// serves a whole batch of them: the weights are read once and the kernels
// launched once, whatever the number of streams.
//
// Every layer is causal. A conv reads the KT - 1 frames it saw before plus
// the current one, the alignment reads dmax frames of the far end, the mask
// reads two past frames, the bottleneck carries its complex state. All of it
// lives on the device, one slot per stream in tensors outside the compute
// allocator: the graph reads a state, computes its next value and writes it
// back in place, so a hop moves only the audio. A stream without a hop in a
// run keeps its state: the write back is masked per stream.
//
// A worker owns the device. Callers queue their hop and wait; the worker
// takes every hop waiting, runs them in one pass of the graph and wakes their
// callers. Under load the hops pile up during a pass, so the batch fills by
// itself.

#include "localvqe.h"

#include "backend.h"
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

#define LV_MAX_NODES   4096
#define LV_MAX_STATES  64  // state tensors the graph may declare
#define LV_KT          4   // time taps of every encoder and decoder conv
#define LV_ALIGN_KT    5   // time taps of the alignment smoothing conv
#define LV_MASK_KT     3   // time taps of the complex mask
#define LV_MASK_KF     3   // frequency taps of the complex mask
#define LV_MASK_TAPS   (LV_MASK_KT * LV_MASK_KF)
#define LV_MASK_ROOTS  3   // cube roots of unity the mask channels combine
#define LV_N_ENCODER   5
#define LV_N_BEFORE    2   // mic encoders before the alignment
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

struct lv_context {
    BackendPair bp = {};

    struct ggml_tensor * analysis  = nullptr;  // [512, 512]
    struct ggml_tensor * synthesis = nullptr;  // [512, 512]

    LvEncoder mic[LV_N_ENCODER];
    LvEncoder ref[LV_N_FAR];
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

    // The graph for `capacity` streams, and its per stream inputs and output.
    int                  capacity = 0;
    struct ggml_cgraph * graph    = nullptr;
    struct ggml_tensor * in_mic   = nullptr;  // [n_fft, capacity]
    struct ggml_tensor * in_ref   = nullptr;  // [n_fft, capacity]
    struct ggml_tensor * in_valid = nullptr;  // [1, 1, 1, capacity] 1 where a stream has a hop
    struct ggml_tensor * in_hold  = nullptr;  // [1, 1, 1, capacity] 1 - valid
    struct ggml_tensor * out_pcm  = nullptr;  // [n_fft, capacity]

    // Host side of the per run inputs. A slot at rest keeps the last window
    // it sent, finite audio, so its masked lane stays finite as well.
    std::vector<float> mic_host;  // [n_fft * capacity]
    std::vector<float> ref_host;  // [n_fft * capacity]

    // Layer histories, one slot per stream on the last dim, in their own
    // buffer: they outlive every run of the graph.
    struct ggml_context *             state_ctx = nullptr;
    ggml_backend_buffer_t             state_buf = nullptr;
    std::vector<struct ggml_tensor *> states;
    std::vector<bool>                 used;   // [capacity] slots held by a stream
    std::vector<float>                valid;  // [capacity] host side of in_valid
    std::vector<float>                hold;   // [capacity] host side of in_hold

    int   sample_rate = 0;
    int   n_fft       = 0;
    int   hop         = 0;
    int   dmax        = 0;
    float power_law   = 0.0f;
    float eps         = 0.0f;

    // The worker and its queue. computing is set while a pass runs off the
    // lock: a growth or a slot reset waits for it to drop.
    std::mutex              mutex;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    std::vector<lv_state *> queue;
    bool                    computing = false;
    bool                    stop      = false;
    std::thread             worker;
    std::vector<float>      out_host;  // [n_fft * capacity]
    std::string             error;     // why the last pass failed
};

struct lv_state {
    lv_context *       ctx  = nullptr;
    int                slot = 0;  // its index on the stream dim of every per stream tensor
    std::vector<float> mic_prev;  // [hop] the half of the analysis window kept
    std::vector<float> ref_prev;  // [hop]
    std::vector<float> mic_win;   // [n_fft]
    std::vector<float> ref_win;   // [n_fft]
    std::vector<float> frame;     // [n_fft]
    std::vector<float> ola;       // [n_fft]

    // The hop in flight: queued by the caller, answered by the worker.
    bool done   = false;
    int  status = 0;
};

// A state of shape [ne0, ne1, ne2] per stream, in the state buffer.
static struct ggml_tensor * lv_state_tensor(lv_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2) {
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx->state_ctx, GGML_TYPE_F32, ne0, ne1, ne2, ctx->capacity);
    ctx->states.push_back(state);
    return state;
}

// Writes next back into state for the streams that had a hop; the others
// keep their history. Both lanes are weighted by an exact 0 or 1, so each
// slot gets one of the two values bit for bit. The write comes after the
// only read of the state, which next depends on.
static void lv_state_commit(lv_context *          ctx,
                            struct ggml_context * g,
                            struct ggml_tensor *  state,
                            struct ggml_tensor *  next) {
    struct ggml_tensor * kept = ggml_add(g, ggml_mul(g, next, ctx->in_valid), ggml_mul(g, state, ctx->in_hold));
    ggml_build_forward_expand(ctx->graph, ggml_cpy(g, kept, state));
}

// Appends the current frame to the frames a state holds along dim 1, and
// commits the most recent ones as its next value.
static struct ggml_tensor * lv_window(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * x, int n_past) {
    struct ggml_tensor * state = lv_state_tensor(ctx, x->ne[0], n_past, x->ne[2]);
    struct ggml_tensor * win   = ggml_concat(g, state, x, 1);
    lv_state_commit(ctx, g, state,
                    ggml_cont(g, ggml_view_4d(g, win, win->ne[0], n_past, win->ne[2], win->ne[3], win->nb[1],
                                              win->nb[2], win->nb[3], win->nb[1])));
    return win;
}

static struct ggml_tensor * lv_norm(lv_context *          ctx,
                                    struct ggml_context * g,
                                    const LvNorm &        n,
                                    struct ggml_tensor *  x) {
    // One frame of one stream at a time: the statistics run over channels
    // and frequency.
    struct ggml_tensor * y =
        ggml_norm(g, ggml_reshape_2d(g, ggml_cont(g, x), x->ne[0] * x->ne[1] * x->ne[2], x->ne[3]), ctx->eps);
    y = ggml_reshape_4d(g, y, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    y = ggml_mul(g, y, ggml_reshape_4d(g, n.w, 1, 1, n.w->ne[0], 1));
    return ggml_add(g, y, ggml_reshape_4d(g, n.b, 1, 1, n.b->ne[0], 1));
}

// Causal conv, KT frames on time, (1, 2) padded on frequency, stride on
// frequency only. The im2col stays in f32. Returns one frame [OF, 1, OC, B].
static struct ggml_tensor * lv_conv(lv_context *          ctx,
                                    struct ggml_context * g,
                                    const LvConv &        c,
                                    struct ggml_tensor *  x,
                                    int                   stride) {
    struct ggml_tensor * win = lv_window(ctx, g, x, LV_KT - 1);
    win                      = ggml_pad_ext(g, win, LV_FREQ_PAD_LO, LV_FREQ_PAD_HI, 0, 0, 0, 0, 0, 0);

    // [K, OF, 1, B]: one output frame per stream, the frames side by side
    struct ggml_tensor * col = ggml_im2col(g, c.w, win, stride, 1, 0, 0, 1, 1, true, GGML_TYPE_F32);
    const int64_t        of  = col->ne[1];
    const int64_t        oc  = c.w->ne[3];
    struct ggml_tensor * y   = ggml_mul_mat(g, ggml_reshape_2d(g, col, col->ne[0], of * col->ne[3]),
                                            ggml_reshape_2d(g, c.w, c.w->ne[0] * c.w->ne[1] * c.w->ne[2], oc));
    y = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, y, of, col->ne[3], oc, 1), 0, 3, 2, 1));  // [OF, 1, OC, B]
    return ggml_add(g, y, ggml_reshape_4d(g, c.b, 1, 1, oc, 1));
}

// 1x1 conv on one frame [F, 1, IC, B] -> [F, 1, OC, B].
static struct ggml_tensor * lv_point(struct ggml_context * g, const LvPoint & p, struct ggml_tensor * x) {
    struct ggml_tensor * xt = ggml_cont(g, ggml_transpose(g, ggml_reshape_3d(g, ggml_cont(g, x), x->ne[0], x->ne[2],
                                                                             x->ne[3])));  // [IC, F, B]
    struct ggml_tensor * y  = ggml_add(g, ggml_mul_mat(g, p.w, xt), ggml_reshape_3d(g, p.b, p.b->ne[0], 1, 1));
    y                       = ggml_cont(g, ggml_transpose(g, y));                          // [F, OC, B]
    return ggml_reshape_4d(g, y, x->ne[0], 1, p.w->ne[1], x->ne[3]);
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
    struct ggml_tensor * lo =
        ggml_cont(g, ggml_view_4d(g, y, y->ne[0], 1, oc, y->ne[3], y->nb[1], y->nb[2], y->nb[3], 0));
    struct ggml_tensor * hi = ggml_cont(
        g, ggml_view_4d(g, y, y->ne[0], 1, oc, y->ne[3], y->nb[1], y->nb[2], y->nb[3], (size_t) oc * y->nb[2]));
    y = ggml_concat(g, lo, hi, 0);
    if (y->ne[0] != n_freq) {
        y = ggml_cont(g, ggml_view_4d(g, y, n_freq, 1, oc, y->ne[3], y->nb[1], y->nb[2], y->nb[3], 0));
    }
    return last ? y : ggml_silu(g, y);
}

// Cross-attention between the current mic frame and the last dmax far end
// frames, a softmax over the delay, and the far end frames weighted by it.
static struct ggml_tensor * lv_align(lv_context *          ctx,
                                     struct ggml_context * g,
                                     struct ggml_tensor *  mic,
                                     struct ggml_tensor *  ref) {
    const int64_t n_freq = mic->ne[0];
    const int64_t n_b    = mic->ne[3];

    struct ggml_tensor * q = lv_point(g, ctx->align_q, mic);
    struct ggml_tensor * k = lv_window(ctx, g, lv_point(g, ctx->align_k, ref), ctx->dmax - 1);  // [F, dmax, H, B]

    struct ggml_tensor * v = ggml_sum_rows(g, ggml_mul(g, k, q));                               // [1, dmax, H, B]
    v = ggml_scale(g, ggml_reshape_4d(g, v, ctx->dmax, 1, v->ne[2], n_b), 1.0f / sqrtf((float) n_freq));

    struct ggml_tensor * vwin = lv_window(ctx, g, v, LV_ALIGN_KT - 1);  // [dmax, 5, H, B]
    vwin                      = ggml_pad_ext(g, vwin, 1, 1, 0, 0, 0, 0, 0, 0);

    struct ggml_tensor * col =
        ggml_im2col(g, ctx->align_smooth.w, vwin, 1, 1, 0, 0, 1, 1, true, GGML_TYPE_F32);  // [3 * 5 * H, dmax, 1, B]
    struct ggml_tensor * score =
        ggml_mul_mat(g, ggml_reshape_2d(g, col, col->ne[0], col->ne[1] * n_b),
                     ggml_reshape_2d(g, ctx->align_smooth.w, ggml_nelements(ctx->align_smooth.w), 1));  // [dmax * B]
    score                        = ggml_add(g, ggml_reshape_2d(g, score, ctx->dmax, n_b), ctx->align_smooth.b);
    struct ggml_tensor * weights = ggml_reshape_3d(g, ggml_soft_max(g, score), ctx->dmax, 1, n_b);

    struct ggml_tensor * refs = lv_window(ctx, g, ref, ctx->dmax - 1);            // [F, dmax, C, B]
    refs                      = ggml_cont(g, ggml_permute(g, refs, 1, 0, 2, 3));  // [dmax, F, C, B]
    struct ggml_tensor * aligned =
        ggml_mul_mat(g, ggml_reshape_3d(g, refs, ctx->dmax, refs->ne[1] * refs->ne[2], n_b), weights);
    return ggml_reshape_4d(g, aligned, n_freq, 1, ref->ne[2], n_b);
}

// Diagonal state space step: h = a h + b v on complex numbers, y = Re(c h).
static struct ggml_tensor * lv_bottleneck(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * x) {
    const int64_t        n_b = x->ne[3];
    struct ggml_tensor * u   = ggml_reshape_2d(g, ggml_cont(g, x), x->ne[0] * x->ne[1] * x->ne[2], n_b);
    struct ggml_tensor * v   = ggml_add(g, ggml_mul_mat(g, ctx->bn_in_w, u), ctx->bn_in_b);  // [S, B]

    const int64_t        n_state = v->ne[0];
    struct ggml_tensor * state   = lv_state_tensor(ctx, n_state, 2, 1);
    struct ggml_tensor * h_re    = ggml_cont(g, ggml_view_2d(g, state, n_state, n_b, state->nb[3], 0));
    struct ggml_tensor * h_im    = ggml_cont(g, ggml_view_2d(g, state, n_state, n_b, state->nb[3], state->nb[1]));

    struct ggml_tensor * re = ggml_sub(g, ggml_mul(g, h_re, ctx->bn_a_re), ggml_mul(g, h_im, ctx->bn_a_im));
    re                      = ggml_add(g, re, ggml_mul(g, v, ctx->bn_b_re));
    struct ggml_tensor * im = ggml_add(g, ggml_mul(g, h_im, ctx->bn_a_re), ggml_mul(g, h_re, ctx->bn_a_im));
    im                      = ggml_add(g, im, ggml_mul(g, v, ctx->bn_b_im));

    lv_state_commit(
        ctx, g, state,
        ggml_reshape_4d(
            g, ggml_concat(g, ggml_reshape_3d(g, re, n_state, 1, n_b), ggml_reshape_3d(g, im, n_state, 1, n_b), 1),
            n_state, 2, 1, n_b));

    struct ggml_tensor * y   = ggml_sub(g, ggml_mul(g, re, ctx->bn_c_re), ggml_mul(g, im, ctx->bn_c_im));
    struct ggml_tensor * out = ggml_add(g, ggml_mul_mat(g, ctx->bn_out_w, y), ctx->bn_out_b);
    out                      = ggml_add(g, out, ggml_mul(g, u, ctx->bn_d));
    return ggml_reshape_4d(g, out, x->ne[0], 1, x->ne[2], n_b);
}

// Complex convolving mask: 27 channels are 3 cube root of unity components
// times a 3x3 kernel over (time t-2..t, frequency f-1..f+1), applied to the
// mic analysis. Returns the enhanced frames as [2, F, B], real and imaginary
// interleaved per bin, the layout the synthesis reads.
static struct ggml_tensor * lv_mask(lv_context *          ctx,
                                    struct ggml_context * g,
                                    struct ggml_tensor *  m,
                                    struct ggml_tensor *  spec) {
    const int64_t n_freq = spec->ne[0];
    const int64_t n_b    = spec->ne[3];

    struct ggml_tensor * mr =
        ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, m, n_freq, LV_MASK_TAPS, LV_MASK_ROOTS, n_b), 1, 2, 0, 3));
    mr                       = ggml_reshape_2d(g, mr, LV_MASK_ROOTS, n_freq * LV_MASK_TAPS * n_b);
    struct ggml_tensor * hre = ggml_reshape_3d(
        g, ggml_mul_mat(g, mr, ggml_reshape_2d(g, ctx->mask_re, LV_MASK_ROOTS, 1)), n_freq, LV_MASK_TAPS, n_b);
    struct ggml_tensor * him = ggml_reshape_3d(
        g, ggml_mul_mat(g, mr, ggml_reshape_2d(g, ctx->mask_im, LV_MASK_ROOTS, 1)), n_freq, LV_MASK_TAPS, n_b);

    struct ggml_tensor * win = lv_window(ctx, g, spec, LV_MASK_KT - 1);  // [F, 3, 2, B]
    win                      = ggml_cont(g, ggml_pad_ext(g, win, 1, 1, 0, 0, 0, 0, 0, 0));

    const int64_t        width = n_freq + 2;
    struct ggml_tensor * taps[2];
    for (int c = 0; c < 2; c++) {
        struct ggml_tensor * t = nullptr;
        for (int k = 0; k < LV_MASK_TAPS; k++) {
            const size_t offset =
                ((size_t) (k % LV_MASK_KF) + (size_t) (k / LV_MASK_KF) * width + (size_t) c * width * LV_MASK_KT) *
                sizeof(float);
            struct ggml_tensor * tap = ggml_view_3d(g, win, n_freq, 1, n_b, win->nb[1], win->nb[3], offset);
            t                        = t ? ggml_concat(g, t, tap, 1) : ggml_cont(g, tap);
        }
        taps[c] = t;  // [F, LV_MASK_TAPS, B]
    }

    struct ggml_tensor * re = ggml_sub(g, ggml_mul(g, hre, taps[0]), ggml_mul(g, him, taps[1]));
    struct ggml_tensor * im = ggml_add(g, ggml_mul(g, hre, taps[1]), ggml_mul(g, him, taps[0]));
    re                      = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, re)));  // [1, F, B]
    im                      = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, im)));
    return ggml_concat(g, re, im, 0);                                                 // [2, F, B]
}

// Analysis of one 512 sample window per stream into [F, 1, 2, B], real then
// imaginary.
static struct ggml_tensor * lv_analysis(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * pcm) {
    const int64_t        n_b = pcm->ne[1];
    struct ggml_tensor * v   = ggml_mul_mat(g, ctx->analysis, pcm);  // bins interleaved as re, im
    v                        = ggml_cont(g, ggml_transpose(g, ggml_reshape_3d(g, v, 2, ctx->n_fft / 2, n_b)));
    return ggml_reshape_4d(g, v, ctx->n_fft / 2, 1, 2, n_b);
}

// Power law compression of the magnitude, the phase kept.
static struct ggml_tensor * lv_compress(lv_context * ctx, struct ggml_context * g, struct ggml_tensor * spec) {
    const int64_t        n_freq = spec->ne[0];
    const int64_t        n_b    = spec->ne[3];
    struct ggml_tensor * re =
        ggml_cont(g, ggml_view_4d(g, spec, n_freq, 1, 1, n_b, spec->nb[1], spec->nb[2], spec->nb[3], 0));
    struct ggml_tensor * im =
        ggml_cont(g, ggml_view_4d(g, spec, n_freq, 1, 1, n_b, spec->nb[1], spec->nb[2], spec->nb[3], spec->nb[2]));
    struct ggml_tensor * mag =
        ggml_sqrt(g, ggml_scale_bias(g, ggml_add(g, ggml_sqr(g, re), ggml_sqr(g, im)), 1.0f, LV_MAG_EPS));
    struct ggml_tensor * div =
        ggml_scale_bias(g, ggml_exp(g, ggml_scale(g, ggml_log(g, mag), 1.0f - ctx->power_law)), 1.0f, LV_MAG_EPS);
    return ggml_div(g, spec, div);
}

// Builds the graph for ctx->capacity streams, with a fresh set of zeroed
// states.
static bool lv_build_graph(lv_context * ctx) {
    struct ggml_init_params sp = { ggml_tensor_overhead() * LV_MAX_STATES, nullptr, true };
    ctx->state_ctx             = ggml_init(sp);
    if (!ctx->state_ctx) {
        return false;
    }
    ctx->states.clear();

    const int64_t         n_b = ctx->capacity;
    struct ggml_context * g   = graph_arena_begin(&ctx->arena);
    ctx->graph                = ggml_new_graph_custom(g, LV_MAX_NODES, false);

    ctx->in_mic   = ggml_new_tensor_2d(g, GGML_TYPE_F32, ctx->n_fft, n_b);
    ctx->in_ref   = ggml_new_tensor_2d(g, GGML_TYPE_F32, ctx->n_fft, n_b);
    ctx->in_valid = ggml_new_tensor_4d(g, GGML_TYPE_F32, 1, 1, 1, n_b);
    ctx->in_hold  = ggml_new_tensor_4d(g, GGML_TYPE_F32, 1, 1, 1, n_b);
    ggml_set_input(ctx->in_mic);
    ggml_set_input(ctx->in_ref);
    ggml_set_input(ctx->in_valid);
    ggml_set_input(ctx->in_hold);

    struct ggml_tensor * mic_spec = lv_analysis(ctx, g, ctx->in_mic);
    struct ggml_tensor * ref_spec = lv_analysis(ctx, g, ctx->in_ref);

    struct ggml_tensor * skips[LV_N_ENCODER];
    struct ggml_tensor * x = lv_compress(ctx, g, mic_spec);
    for (int i = 0; i < LV_N_BEFORE; i++) {
        x        = lv_encoder(ctx, g, ctx->mic[i], x);
        skips[i] = x;
    }

    struct ggml_tensor * ref = lv_compress(ctx, g, ref_spec);
    for (int i = 0; i < LV_N_FAR; i++) {
        ref = lv_encoder(ctx, g, ctx->ref[i], ref);
    }

    x = ggml_concat(g, x, lv_align(ctx, g, x, ref), 2);
    for (int i = LV_N_BEFORE; i < LV_N_ENCODER; i++) {
        x        = lv_encoder(ctx, g, ctx->mic[i], x);
        skips[i] = x;
    }

    x = lv_bottleneck(ctx, g, x);
    for (int i = LV_N_ENCODER - 1; i >= 0; i--) {
        const int64_t n_freq = i > 0 ? skips[i - 1]->ne[0] : mic_spec->ne[0];
        x                    = lv_decoder(ctx, g, ctx->dec[i], x, skips[i], n_freq, i == 0);
    }

    struct ggml_tensor * enhanced = lv_mask(ctx, g, x, mic_spec);
    ctx->out_pcm = ggml_mul_mat(g, ctx->synthesis, ggml_reshape_2d(g, enhanced, 2 * enhanced->ne[1], n_b));
    ggml_set_output(ctx->out_pcm);
    ggml_build_forward_expand(ctx->graph, ctx->out_pcm);

    ctx->state_buf = ggml_backend_alloc_ctx_tensors(ctx->state_ctx, ctx->bp.backend);
    if (!ctx->state_buf) {
        return false;
    }
    ggml_backend_buffer_clear(ctx->state_buf, 0);

    if (!ggml_gallocr_alloc_graph(ctx->alloc, ctx->graph)) {
        return false;
    }
    ctx->used.resize((size_t) n_b, false);
    ctx->valid.assign((size_t) n_b, 0.0f);
    ctx->hold.assign((size_t) n_b, 1.0f);
    ctx->mic_host.resize((size_t) (ctx->n_fft * n_b), 0.0f);
    ctx->ref_host.resize((size_t) (ctx->n_fft * n_b), 0.0f);
    ctx->out_host.resize((size_t) (ctx->n_fft * n_b), 0.0f);
    return true;
}

// Bytes of history one stream holds, over every state.
static size_t lv_stream_bytes(const lv_context * ctx) {
    size_t bytes = 0;
    for (const struct ggml_tensor * state : ctx->states) {
        bytes += state->nb[3];
    }
    return bytes;
}

// One more slot: the graph is rebuilt for one more stream, and the states of
// the streams already running move into the new buffer. The stream dim is
// the outermost one, so the old states are the head of the new ones. The
// capacity follows the peak of concurrent streams exactly: every lane a pass
// computes is a lane some stream may use, which matters where compute, not
// launches, bounds a pass.
static bool lv_grow(lv_context * ctx) {
    struct ggml_context *             old_ctx    = ctx->state_ctx;
    ggml_backend_buffer_t             old_buf    = ctx->state_buf;
    std::vector<struct ggml_tensor *> old_states = ctx->states;

    ctx->capacity += 1;
    const bool built = lv_build_graph(ctx);
    if (built) {
        std::vector<uint8_t> host;
        for (size_t i = 0; i < old_states.size(); i++) {
            host.resize(ggml_nbytes(old_states[i]));
            ggml_backend_tensor_get(old_states[i], host.data(), 0, host.size());
            ggml_backend_tensor_set(ctx->states[i], host.data(), 0, host.size());
        }
        s2s_log(S2S_LOG_INFO, "[LocalVQE] Capacity %d streams", ctx->capacity);
    }
    ggml_backend_buffer_free(old_buf);
    ggml_free(old_ctx);
    return built;
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

// One pass over every hop waiting: the batch is laid out under the lock, the
// device runs off it, and the results go back under it.
static void lv_worker(lv_context * ctx) {
    s2s_log_thread("AEC");
    const size_t            n_fft = (size_t) ctx->n_fft;
    std::vector<lv_state *> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            ctx->work_cv.wait(lock, [ctx]() { return ctx->stop || !ctx->queue.empty(); });
            if (ctx->stop) {
                return;
            }
            batch.swap(ctx->queue);
            for (const lv_state * state : batch) {
                const size_t slot = (size_t) state->slot;
                std::copy(state->mic_win.begin(), state->mic_win.end(),
                          ctx->mic_host.begin() + (ptrdiff_t) (slot * n_fft));
                std::copy(state->ref_win.begin(), state->ref_win.end(),
                          ctx->ref_host.begin() + (ptrdiff_t) (slot * n_fft));
                ctx->valid[slot] = 1.0f;
                ctx->hold[slot]  = 0.0f;
            }
            ctx->computing = true;
        }

        // The inputs go whole: the compute buffer reuses their memory once
        // they are read. A slot at rest sends the last window it had.
        ggml_backend_tensor_set(ctx->in_mic, ctx->mic_host.data(), 0, ctx->mic_host.size() * sizeof(float));
        ggml_backend_tensor_set(ctx->in_ref, ctx->ref_host.data(), 0, ctx->ref_host.size() * sizeof(float));
        ggml_backend_tensor_set(ctx->in_valid, ctx->valid.data(), 0, ctx->valid.size() * sizeof(float));
        ggml_backend_tensor_set(ctx->in_hold, ctx->hold.data(), 0, ctx->hold.size() * sizeof(float));
        const bool ok = ggml_backend_graph_compute(ctx->bp.backend, ctx->graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(ctx->out_pcm, ctx->out_host.data(), 0, ctx->out_host.size() * sizeof(float));
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            for (lv_state * state : batch) {
                const size_t slot = (size_t) state->slot;
                std::copy(ctx->out_host.begin() + (ptrdiff_t) (slot * n_fft),
                          ctx->out_host.begin() + (ptrdiff_t) ((slot + 1) * n_fft), state->frame.begin());
                ctx->valid[slot] = 0.0f;
                ctx->hold[slot]  = 1.0f;
                state->status    = ok ? 0 : -1;
                state->done      = true;
            }
            if (!ok) {
                ctx->error = "[LocalVQE] Graph compute failed";
            }
            ctx->computing = false;
            batch.clear();
        }
        ctx->done_cv.notify_all();
    }
}

lv_context * lv_init(const char * gguf_path, int use_gpu, int n_threads) {
    if (!gguf_path) {
        s2s_set_error("[LocalVQE] Gguf path is NULL");
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
            lv_load_encoder(ctx, gf, ctx->ref[i], "far_enc" + std::to_string(i + 1));
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

        ctx->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->bp.backend));
        if (!graph_arena_init(&ctx->arena, LV_MAX_NODES) || !ctx->alloc) {
            s2s_set_error("[LocalVQE] Failed to allocate the graph arena");
            lv_free(ctx);
            return nullptr;
        }

        ctx->capacity = 1;
        if (!lv_build_graph(ctx)) {
            s2s_set_error("[LocalVQE] Failed to allocate the compute graph");
            lv_free(ctx);
            return nullptr;
        }

        ctx->worker = std::thread(lv_worker, ctx);

        s2s_log(S2S_LOG_INFO, "[LocalVQE] %d Hz, hop %d, dmax %d, %zu states of %.1f KB per stream, %d graph nodes",
                ctx->sample_rate, ctx->hop, ctx->dmax, ctx->states.size(), (double) lv_stream_bytes(ctx) / 1024.0,
                ggml_graph_n_nodes(ctx->graph));
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
    if (ctx->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->stop = true;
        }
        ctx->work_cv.notify_one();
        ctx->worker.join();
    }
    if (ctx->alloc) {
        ggml_gallocr_free(ctx->alloc);
    }
    if (ctx->state_buf) {
        ggml_backend_buffer_free(ctx->state_buf);
    }
    if (ctx->state_ctx) {
        ggml_free(ctx->state_ctx);
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

// Zeroes the history of one slot, on the device.
static void lv_clear_slot(lv_context * ctx, int slot) {
    for (struct ggml_tensor * state : ctx->states) {
        ggml_backend_tensor_memset(state, 0, (size_t) slot * state->nb[3], state->nb[3]);
    }
}

lv_state * lv_state_new(lv_context * ctx) {
    if (!ctx) {
        s2s_set_error("[LocalVQE] Context is NULL");
        return nullptr;
    }

    // A growth rebuilds the graph: no pass may be running.
    std::unique_lock<std::mutex> lock(ctx->mutex);
    ctx->done_cv.wait(lock, [ctx]() { return !ctx->computing; });

    int slot = 0;
    while (slot < ctx->capacity && ctx->used[(size_t) slot]) {
        slot++;
    }
    if (slot == ctx->capacity && !lv_grow(ctx)) {
        s2s_set_error("[LocalVQE] Failed to grow to %d streams", ctx->capacity);
        return nullptr;
    }
    ctx->used[(size_t) slot] = true;
    lv_clear_slot(ctx, slot);

    lv_state * state = new lv_state();
    state->ctx       = ctx;
    state->slot      = slot;
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
    {
        lv_context *                 ctx = state->ctx;
        std::unique_lock<std::mutex> lock(ctx->mutex);
        ctx->done_cv.wait(lock, [ctx]() { return !ctx->computing; });
        lv_clear_slot(ctx, state->slot);
    }
    std::fill(state->mic_prev.begin(), state->mic_prev.end(), 0.0f);
    std::fill(state->ref_prev.begin(), state->ref_prev.end(), 0.0f);
    std::fill(state->ola.begin(), state->ola.end(), 0.0f);
}

void lv_state_free(lv_state * state) {
    if (!state) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state->ctx->mutex);
        state->ctx->used[(size_t) state->slot] = false;
    }
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
        std::unique_lock<std::mutex> lock(ctx->mutex);
        state->done = false;
        ctx->queue.push_back(state);
        ctx->work_cv.notify_one();
        ctx->done_cv.wait(lock, [state]() { return state->done; });
        if (state->status != 0) {
            s2s_set_error("%s", ctx->error.c_str());
            return -1;
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
