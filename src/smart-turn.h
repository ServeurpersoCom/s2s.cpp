#pragma once
// smart-turn.h: Smart Turn v3.2 end of turn classifier, public C ABI
//
// Answers one question: did the speaker finish their turn, or did they
// just pause. The model reads the last 8 seconds of the audio it is given
// as 16 kHz mono and returns the probability that the turn is complete.
//
// It runs on a speech to silence boundary, a few times per turn, never on
// the 32 ms window loop, so it sits on the CPU with the encoder threads
// it needs and leaves the GPU to the models that do run there.
//
// The model is stateless: every call re-reads the audio it is given, so
// one st_context serves any number of streams. Concurrent calls are
// serialized onto one internal worker thread, which runs every forward pass.
//
//   st_context * turn = st_init("models/smart-turn-v3.2-F32.gguf");
//   float        p    = st_predict(turn, pcm, n_samples);
//
// Errors: st_init returns NULL, st_predict returns a negative value.
// st_last_error() describes the failure on the calling thread.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_context st_context;

enum st_log_level {
    ST_LOG_ERROR = 0,
    ST_LOG_WARN  = 1,
    ST_LOG_INFO  = 2,
    ST_LOG_DEBUG = 3,
};

typedef void (*st_log_cb)(enum st_log_level level, const char * text, void * user_data);

// Routes library diagnostics to a caller owned sink. Without a callback
// the messages go to stderr.
void st_log_set(st_log_cb cb, void * user_data);

// Loads the GGUF and builds the two compute graphs, mel then encoder, on
// the CPU with one thread per physical core.
st_context * st_init(const char * gguf_path);
void         st_free(st_context * ctx);

// Audio the model consumes: sample rate and the 8 second window in samples.
int st_sample_rate(const st_context * ctx);
int st_window(const st_context * ctx);

// Probability in [0, 1] that the turn is complete. Audio longer than the
// window keeps its last st_window() samples, shorter audio is zero padded
// on the right, which is what the upstream feature extractor does.
// Returns a negative value on failure.
float st_predict(st_context * ctx, const float * pcm, int n_samples);

// Diagnostic recorded by the last failing call on this thread.
const char * st_last_error(void);

#ifdef __cplusplus
}
#endif
