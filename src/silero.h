#pragma once
// silero.h: Silero VAD v5, public C ABI
//
// Speech probability for one 512 sample window of 16 kHz mono audio. The
// model carries two kinds of memory across calls: the LSTM state and the
// last 64 samples of the previous window, which the STFT needs as left
// context. Both live in sv_state, one per audio stream.
//
// The model runs on the CPU with a single thread: 2 MB of weights per
// 32 ms window is far below the point where a GPU dispatch pays off, and
// keeping it off the device leaves the whole GPU to the ASR and the TTS.
//
// One sv_context holds the weights and the compute graph, and serializes
// concurrent sv_prob calls internally, so several streams share one
// instance of the model.
//
//   sv_context * vad   = sv_init("models/silero-vad-F32.gguf");
//   sv_state *   state = sv_state_new(vad);
//   float        p     = sv_prob(state, pcm, 512);
//
// Errors: sv_init and sv_state_new return NULL, sv_prob returns a
// negative value. sv_last_error() describes the failure on the calling
// thread.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sv_context sv_context;
typedef struct sv_state   sv_state;

enum sv_log_level {
    SV_LOG_ERROR = 0,
    SV_LOG_WARN  = 1,
    SV_LOG_INFO  = 2,
    SV_LOG_DEBUG = 3,
};

typedef void (*sv_log_cb)(enum sv_log_level level, const char * text, void * user_data);

// Routes library diagnostics to a caller owned sink. Without a callback
// the messages go to stderr.
void sv_log_set(sv_log_cb cb, void * user_data);

// Loads the GGUF and builds the compute graph, on the CPU with one thread:
// on a 32 ms window a GPU dispatch would cost more than the work.
sv_context * sv_init(const char * gguf_path);
void         sv_free(sv_context * ctx);

// Window size in samples and sample rate the model expects.
int sv_window(const sv_context * ctx);
int sv_sample_rate(const sv_context * ctx);

// Per stream memory. Reset clears the LSTM state and the left context,
// which is what a new turn or a stream restart needs.
sv_state * sv_state_new(sv_context * ctx);
void       sv_state_reset(sv_state * state);
void       sv_state_free(sv_state * state);

// Speech probability in [0, 1] for one window. n_samples must equal
// sv_window(). Returns a negative value on failure.
float sv_prob(sv_state * state, const float * pcm, int n_samples);

// Diagnostic recorded by the last failing call on this thread.
const char * sv_last_error(void);

#ifdef __cplusplus
}
#endif
