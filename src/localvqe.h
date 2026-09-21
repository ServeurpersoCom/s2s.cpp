#pragma once
// localvqe.h: LocalVQE v1.3 echo canceller, public C ABI
//
// Removes from the microphone what the loudspeaker played, the far end
// reference, and cleans noise and reverberation in the same pass. The model
// is a DeepVQE derivative that estimates the echo delay itself, over a window
// of about one second, so the reference only has to be roughly in step with
// the microphone.
//
// It runs one hop at a time: 256 samples of microphone and reference at
// 16 kHz in, 256 cleaned samples out, one hop late. Weights, graph and the
// history of every causal layer live in the context, on the device, one slot
// per stream: an lv_state is a stream holding its slot, and the context grows
// its batch when every slot is taken. The context runs a worker: a call to
// lv_process queues its hop and returns once the worker has run it, in one
// pass with every other hop waiting at that moment.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lv_context lv_context;
typedef struct lv_state   lv_state;

// Loads the GGUF and builds the compute graph, on the best device or on the
// one GGML_BACKEND names.
lv_context * lv_init(const char * gguf_path);
void         lv_free(lv_context * ctx);

// Hop size in samples and sample rate the model expects.
int lv_hop(const lv_context * ctx);
int lv_sample_rate(const lv_context * ctx);

// Per stream memory. Reset forgets the echo path and every layer history,
// which is what a stream restart needs.
lv_state * lv_state_new(lv_context * ctx);
void       lv_state_reset(lv_state * state);
void       lv_state_free(lv_state * state);

// Cleans one hop. mic, ref and out hold lv_hop() samples; out is the hop
// that ended one hop before this call. Returns 0 on success.
int lv_process(lv_state * state, const float * mic, const float * ref, float * out);

// Diagnostic recorded by the last failing call on this thread.
const char * lv_last_error(void);

#ifdef __cplusplus
}
#endif
