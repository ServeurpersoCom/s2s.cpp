// parakeet.cpp: the public ABI entries
//
// Thin layer over pipeline-asr: parameter validation, the exception to status
// conversion at the boundary, and the string ownership contract. Nothing about
// the model lives here.

#include "parakeet.h"

#include "pipeline-asr.h"
#include "s2s-error.h"
#include "version.h"

#include <cstdlib>
#include <cstring>
#include <string>

struct pk_context {
    pipeline_asr * pipeline = nullptr;
};

static pk_log_cb g_pk_log_cb = nullptr;

// The two level enums share their values, so routing the library log to the
// caller sink only costs a cast.
static void pk_log_trampoline(enum s2s_log_level level, const char * text, void * user) {
    g_pk_log_cb((enum pk_log_level) level, text, user);
}

const char * pk_version(void) {
    return S2S_VERSION;
}

const char * pk_last_error(void) {
    return s2s_last_error();
}

void pk_log_set(pk_log_cb cb, void * user) {
    g_pk_log_cb = cb;
    s2s_log_set(cb ? pk_log_trampoline : nullptr, user);
}

struct pk_init_params pk_init_default_params(void) {
    struct pk_init_params params;
    params.abi_version = PK_ABI_VERSION;
    params.model_path  = nullptr;
    return params;
}

struct pk_transcribe_params pk_transcribe_default_params(void) {
    struct pk_transcribe_params params;
    params.abi_version = PK_ABI_VERSION;
    params.on_token    = nullptr;
    params.user        = nullptr;
    return params;
}

pk_context * pk_init(const struct pk_init_params * params) {
    if (!params || params->abi_version > PK_ABI_VERSION) {
        s2s_set_error("[ASR] Unsupported ABI version");
        return nullptr;
    }
    if (!params->model_path) {
        s2s_set_error("[ASR] Model_path is NULL");
        return nullptr;
    }

    pipeline_asr_params pipeline_params;
    pipeline_params.model_path = params->model_path;

    pipeline_asr * pipeline = pipeline_asr_load(pipeline_params);
    if (!pipeline) {
        return nullptr;
    }

    pk_context * ctx = new pk_context();
    ctx->pipeline    = pipeline;
    return ctx;
}

void pk_free(pk_context * ctx) {
    if (!ctx) {
        return;
    }
    pipeline_asr_free(ctx->pipeline);
    delete ctx;
}

int pk_sample_rate(const pk_context * ctx) {
    return ctx ? pipeline_asr_sample_rate(ctx->pipeline) : 0;
}

enum pk_status pk_transcribe(pk_context *                        ctx,
                             const float *                       samples,
                             size_t                              n_samples,
                             int                                 sample_rate,
                             const struct pk_transcribe_params * params,
                             char **                             out_text) {
    if (!ctx || !samples || !out_text || sample_rate <= 0) {
        s2s_set_error("[ASR] Invalid transcribe arguments");
        return PK_STATUS_INVALID_PARAMS;
    }
    if (params && params->abi_version > PK_ABI_VERSION) {
        s2s_set_error("[ASR] Unsupported ABI version");
        return PK_STATUS_INVALID_PARAMS;
    }

    *out_text = nullptr;

    std::string     text;
    const pk_status status = pipeline_asr_run(ctx->pipeline, samples, n_samples, sample_rate, params, text);
    if (status != PK_STATUS_OK) {
        return status;
    }

    char * copy = (char *) malloc(text.size() + 1);
    if (!copy) {
        s2s_set_error("[ASR] Out of memory copying the transcript");
        return PK_STATUS_DECODE_FAILED;
    }
    memcpy(copy, text.c_str(), text.size() + 1);
    *out_text = copy;
    return PK_STATUS_OK;
}

void pk_free_text(char * text) {
    free(text);
}
