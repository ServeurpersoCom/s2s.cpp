#pragma once
// parakeet.h: public ABI for the Parakeet TDT recognizer.
//
// Single-header public API. Pure C99, consumable from C and C++ alike.
// Bindings (Python ctypes, Rust bindgen, Go cgo) parse this file directly.
// Style follows whisper.h / llama.h: extern "C" linkage on every entry,
// POD structs only, const char * UTF-8 strings, pk_status enum returns.
//
// The opaque pk_context aggregates everything the recognition path needs
// (mel frontend, conv2d stem, FastConformer encoder, TDT prediction network
// and joint, the piece table, the GGML backend pair). One init, one free,
// one transcribe call covers the full pcm -> text path. The lower level
// pipeline_asr_* entries in pipeline-asr.h stay available for the debug
// paths that dump stage tensors, but they are not part of this ABI.
//
// The model is stateless across calls, so one context serves any number of
// streams. Concurrent transcribe calls are serialized internally.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(PARAKEET_STATIC)
#        define PK_API
#    elif defined(PARAKEET_BUILD)
#        define PK_API __declspec(dllexport)
#    else
#        define PK_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define PK_API __attribute__((visibility("default")))
#else
#    define PK_API
#endif

// Struct ABI version. Incremented when a public POD struct grows a field at
// its tail. Callers set .abi_version = PK_ABI_VERSION or let the
// pk_*_default_params helpers fill it. Entries reject inputs whose
// abi_version exceeds the build-time constant.
#define PK_ABI_VERSION 1

// Returns "<git-hash> (<date>)" for the exact commit this binary came from.
PK_API const char * pk_version(void);

enum pk_status {
    PK_STATUS_OK             = 0,
    PK_STATUS_INVALID_PARAMS = -1,
    PK_STATUS_LOAD_FAILED    = -2,
    PK_STATUS_DECODE_FAILED  = -3,
    PK_STATUS_CANCELLED      = -4,
};

// Thread-local errno-style message, only meaningful right after a failure.
PK_API const char * pk_last_error(void);

// Log level for the redirectable diagnostic stream.
enum pk_log_level {
    PK_LOG_ERROR = 0,
    PK_LOG_WARN,
    PK_LOG_INFO,
    PK_LOG_DEBUG,
};

// Installs a callback receiving every internal diagnostic. NULL routes to
// stderr. user carries caller state.
typedef void (*pk_log_cb)(enum pk_log_level level, const char * msg, void * user);
PK_API void pk_log_set(pk_log_cb cb, void * user);

typedef struct pk_context pk_context;

// Init params. model_path points at the GGUF holding the encoder, the
// transducer and the piece table. The model runs on the best device, or on
// the one GGML_BACKEND names.
struct pk_init_params {
    int          abi_version;
    const char * model_path;
};

PK_API struct pk_init_params pk_init_default_params(void);
PK_API pk_context *          pk_init(const struct pk_init_params * params);
PK_API void                  pk_free(pk_context * ctx);

// Streaming token callback. Fires once per emitted piece, already detokenized.
// Returning false requests cancellation and the call unwinds with
// PK_STATUS_CANCELLED. user carries caller state.
typedef bool (*pk_token_cb)(const char * utf8_chunk, void * user);

struct pk_transcribe_params {
    int         abi_version;
    pk_token_cb on_token;
    void *      user;
};

PK_API struct pk_transcribe_params pk_transcribe_default_params(void);

// Transcribe mono PCM. samples are f32 in [-1, 1], any sample_rate (resampled
// to the model rate internally). On success out_text receives a freshly
// allocated NUL terminated UTF-8 string the caller releases with pk_free_text.
PK_API enum pk_status pk_transcribe(pk_context *                        ctx,
                                    const float *                       samples,
                                    size_t                              n_samples,
                                    int                                 sample_rate,
                                    const struct pk_transcribe_params * params,
                                    char **                             out_text);

PK_API void pk_free_text(char * text);

// Sample rate the model works at. Audio at any other rate is resampled.
PK_API int pk_sample_rate(const pk_context * ctx);

#ifdef __cplusplus
}
#endif
