#pragma once
// gguf-weights.h: load model weights from GGUF files
//
// The one loader of every model of the project, over the GGUF files
// convert.py and quantize write: the file is mapped, the tensors are
// described in a weight context, and wctx_alloc uploads them in one go.
//
// Usage:
//   GGUFModel gf;
//   if (!gf_load(&gf, "model.gguf")) { error; }
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = gf_load_tensor(&wctx, gf, "layer.0.weight");
//   wctx_alloc(&wctx, backend);
//   gf_close(&gf);   // safe after wctx_alloc copied data to GPU

#include "gguf.h"
#include "s2s-error.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    include "utf8.h"
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

struct GGUFModel {
    struct gguf_context * gguf;         // parsed header (KV + tensor metadata)
    struct ggml_context * meta;         // tensor descriptors (no data)
    uint8_t *             mapping;      // mmapped file
    size_t                file_size;
    size_t                data_offset;  // gguf_get_data_offset(gguf)
#ifdef _WIN32
    HANDLE fh;
    HANDLE mh;
#else
    int fd;
#endif
};

static void gf_close(GGUFModel * gf) {
    if (gf->gguf) {
        gguf_free(gf->gguf);
    }
    if (gf->meta) {
        ggml_free(gf->meta);
    }
#ifdef _WIN32
    if (gf->mapping) {
        UnmapViewOfFile(gf->mapping);
    }
    if (gf->mh) {
        CloseHandle(gf->mh);
    }
    if (gf->fh && gf->fh != INVALID_HANDLE_VALUE) {
        CloseHandle(gf->fh);
    }
#else
    if (gf->mapping) {
        munmap(gf->mapping, gf->file_size);
    }
    if (gf->fd >= 0) {
        close(gf->fd);
    }
#endif
    *gf = {};
}

static bool gf_load(GGUFModel * gf, const char * path) {
    *gf = {};

    // mmap the file
#ifdef _WIN32
    std::wstring wpath = utf8_to_wide(path);
    gf->fh =
        CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (gf->fh == INVALID_HANDLE_VALUE) {
        s2s_log(S2S_LOG_ERROR, "[GGUF] Cannot open %s", path);
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(gf->fh, &li);
    gf->file_size = (size_t) li.QuadPart;
    gf->mh        = CreateFileMappingW(gf->fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!gf->mh) {
        CloseHandle(gf->fh);
        s2s_log(S2S_LOG_ERROR, "[GGUF] CreateFileMapping failed %s", path);
        return false;
    }
    gf->mapping = (uint8_t *) MapViewOfFile(gf->mh, FILE_MAP_READ, 0, 0, 0);
    if (!gf->mapping) {
        CloseHandle(gf->mh);
        CloseHandle(gf->fh);
        s2s_log(S2S_LOG_ERROR, "[GGUF] MapViewOfFile failed %s", path);
        return false;
    }
#else
    gf->fd = open(path, O_RDONLY);
    if (gf->fd < 0) {
        s2s_log(S2S_LOG_ERROR, "[GGUF] Cannot open %s", path);
        return false;
    }
    struct stat sb;
    fstat(gf->fd, &sb);
    gf->file_size = (size_t) sb.st_size;
    gf->mapping   = (uint8_t *) mmap(NULL, gf->file_size, PROT_READ, MAP_PRIVATE, gf->fd, 0);
    if (gf->mapping == MAP_FAILED) {
        close(gf->fd);
        gf->mapping = NULL;
        s2s_log(S2S_LOG_ERROR, "[GGUF] Mmap failed %s", path);
        return false;
    }
#endif

    // Parse GGUF header, create tensor metadata context
    struct ggml_context *   meta   = NULL;
    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&meta };
    gf->gguf                       = gguf_init_from_file(path, params);
    if (!gf->gguf) {
        s2s_log(S2S_LOG_ERROR, "[GGUF] Failed to parse %s", path);
        gf_close(gf);
        return false;
    }
    gf->meta        = meta;
    gf->data_offset = gguf_get_data_offset(gf->gguf);

    int64_t n = gguf_get_n_tensors(gf->gguf);

    // Verify every tensor fits inside the mapped file. Catches truncated
    // downloads early with a clear message instead of a segfault deep in
    // cuMemcpyHtoDAsync when the backend reads past the mmap.
    for (int64_t i = 0; i < n; i++) {
        const char *         tname = gguf_get_tensor_name(gf->gguf, i);
        struct ggml_tensor * t     = ggml_get_tensor(gf->meta, tname);
        size_t               toff  = gguf_get_tensor_offset(gf->gguf, i);
        size_t               tsize = ggml_nbytes(t);
        size_t               end   = gf->data_offset + toff + tsize;
        if (end > gf->file_size) {
            s2s_log(S2S_LOG_ERROR,
                    "[GGUF] FATAL: '%s' is truncated or corrupt.\n"
                    "       tensor '%s' needs bytes [%zu..%zu) but file is only %zu "
                    "bytes.\n"
                    "       Re-download the file and verify its size or checksum.",
                    path, tname, gf->data_offset + toff, end, gf->file_size);
            gf_close(gf);
            return false;
        }
    }

    s2s_log(S2S_LOG_INFO, "[GGUF] %s: %lld tensors, data at offset %zu", path, (long long) n, gf->data_offset);
    return true;
}

// Load a tensor from GGUF into the weight context.
// Returns ggml_tensor (not yet backed by memory; call wctx_alloc after all
// loads). Tensor shapes are already in ggml order (ne[0]=innermost).
static struct ggml_tensor * gf_load_tensor(WeightCtx *         wctx,
                                           const GGUFModel &   gf,
                                           const std::string & name,
                                           const int64_t *     shape_override  = nullptr,
                                           int                 n_dims_override = 0) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        s2s_throw("[GGUF] Tensor '%s' not found", name.c_str());
    }

    // Get metadata from the context populated by gguf_init_from_file
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        s2s_throw("[GGUF] Tensor '%s' not in meta context", name.c_str());
    }

    int     n_dims;
    int64_t ne[4] = { 1, 1, 1, 1 };

    if (shape_override && n_dims_override > 0) {
        n_dims = n_dims_override;
        for (int i = 0; i < n_dims; i++) {
            ne[i] = shape_override[i];
        }
    } else {
        n_dims = ggml_n_dims(src);
        for (int i = 0; i < n_dims; i++) {
            ne[i] = src->ne[i];
        }
    }

    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, src->type, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    size_t       offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * data   = gf.mapping + gf.data_offset + offset;
    size_t       nbytes = ggml_nbytes(src);

    wctx->pending.push_back({ tensor, data, nbytes, 0 });
    return tensor;
}

// Loads a tensor as F32 when it is stored as BF16 or F16, so no cast node runs
// at compute time: norms, biases and the kernels of the convolution path. A
// quantized or F32 tensor loads as it is.
static struct ggml_tensor * gf_load_tensor_f32(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        s2s_throw("[GGUF] Tensor '%s' not found (f32 load)", name.c_str());
    }
    struct ggml_tensor * src    = ggml_get_tensor(gf.meta, name.c_str());
    int                  n_dims = ggml_n_dims(src);
    int64_t              ne[4]  = { 1, 1, 1, 1 };
    for (int i = 0; i < n_dims; i++) {
        ne[i] = src->ne[i];
    }

    // If already F32, just load normally
    if (src->type == GGML_TYPE_F32) {
        return gf_load_tensor(wctx, gf, name);
    }

    // Quantized types load natively, GGML dequantizes them at compute time.
    if (src->type != GGML_TYPE_BF16 && src->type != GGML_TYPE_F16) {
        return gf_load_tensor(wctx, gf, name);
    }

    // Create F32 tensor
    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    // Convert data into staging buffer. unique_ptr keeps .get() stable even
    // when wctx->staging grows on subsequent calls.
    size_t  n    = ggml_nelements(src);
    auto    buf  = std::make_unique<float[]>(n);
    float * data = buf.get();

    size_t       offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * raw    = gf.mapping + gf.data_offset + offset;

    if (src->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) raw;
        for (size_t i = 0; i < n; i++) {
            data[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &p[i]);
        }
    } else {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, data, (int) n);
    }

    wctx->pending.push_back({ tensor, data, n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return tensor;
}

// Read a uint32 KV value (returns 0 if not found)
static uint32_t gf_get_u32(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        return 0;
    }
    return gguf_get_val_u32(gf.gguf, idx);
}

// Read a float32 KV value (returns 0 if not found)
static float gf_get_f32(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        return 0.0f;
    }
    return gguf_get_val_f32(gf.gguf, idx);
}
