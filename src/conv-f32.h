#pragma once
// conv-f32.h: convolutions with an f32 im2col
//
// ggml_conv_1d stages its im2col matrix in f16, which costs about 1e-3 of
// absolute error on the outputs. The models here are small enough that the
// f32 staging is free, and the parity harnesses land at 1e-6 with it.
// Batch is always 1, which also removes the permute the generic helper
// needs to restore [OL, OC, N].
//
//   conv_1d_f32     kernel [K, IC, OC]      data [L, IC]        result [OL, OC]
//   conv_2d_f32     kernel [KW, KH, IC, OC] data [W, H, IC, 1]  result [OW, OH, OC, 1]
//   conv_2d_dw_f32  kernel [KW, KH, 1, C]   data [W, H, C, 1]   result [OW, OH, C, 1]
//   conv_1d_dw_f32  kernel [K, 1, C]        data [L, C]         result [OL, C]

#include "ggml.h"

static struct ggml_tensor * conv_1d_f32(struct ggml_context * ctx,
                                        struct ggml_tensor *  kernel,
                                        struct ggml_tensor *  data,
                                        int                   stride,
                                        int                   pad) {
    struct ggml_tensor * im2col = ggml_im2col(ctx, kernel, data, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);

    struct ggml_tensor * result =
        ggml_mul_mat(ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]),
                     ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));

    return ggml_reshape_2d(ctx, result, im2col->ne[1], kernel->ne[2]);
}

static struct ggml_tensor * conv_2d_f32(struct ggml_context * ctx,
                                        struct ggml_tensor *  kernel,
                                        struct ggml_tensor *  data,
                                        int                   stride,
                                        int                   pad) {
    struct ggml_tensor * im2col = ggml_im2col(ctx, kernel, data, stride, stride, pad, pad, 1, 1, true, GGML_TYPE_F32);

    struct ggml_tensor * result =
        ggml_mul_mat(ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                     ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]));

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], 1, kernel->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
}

static struct ggml_tensor * conv_2d_dw_f32(struct ggml_context * ctx,
                                           struct ggml_tensor *  kernel,
                                           struct ggml_tensor *  data,
                                           int                   stride,
                                           int                   pad) {
    struct ggml_tensor * flat_kernel =
        ggml_reshape_4d(ctx, kernel, kernel->ne[0], kernel->ne[1], 1, kernel->ne[2] * kernel->ne[3]);
    struct ggml_tensor * im2col =
        ggml_im2col(ctx, flat_kernel, ggml_reshape_4d(ctx, data, data->ne[0], data->ne[1], 1, data->ne[2]), stride,
                    stride, pad, pad, 1, 1, true, GGML_TYPE_F32);

    struct ggml_tensor * rows =
        ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], data->ne[2], 1);
    struct ggml_tensor * taps =
        ggml_reshape_4d(ctx, flat_kernel, flat_kernel->ne[0] * flat_kernel->ne[1], 1, flat_kernel->ne[3], 1);

    struct ggml_tensor * result = ggml_mul_mat(ctx, taps, rows);
    return ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], data->ne[2], 1);
}

static struct ggml_tensor * conv_1d_dw_f32(struct ggml_context * ctx,
                                           struct ggml_tensor *  kernel,
                                           struct ggml_tensor *  data,
                                           int                   stride,
                                           int                   pad) {
    struct ggml_tensor * rows   = ggml_reshape_4d(ctx, data, data->ne[0], 1, data->ne[1], data->ne[2]);
    struct ggml_tensor * im2col = ggml_im2col(ctx, kernel, rows, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);

    struct ggml_tensor * result = ggml_mul_mat(ctx, im2col, kernel);
    return ggml_reshape_2d(ctx, result, result->ne[0], result->ne[2]);
}
