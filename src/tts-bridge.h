#pragma once
// tts-bridge.h: the speaking half of the loop, over the qwentts.cpp ABI
//
// One bridge holds one qt_context and speaks one unit at a time. The
// synthesis streams: chunks reach the caller as the codec decodes them,
// starting at a single 12.5 Hz frame so the first audio leaves early, and
// the caller writes them straight to the client.
//
// Cancellation is what makes a barge-in feel instant. The bridge polls an
// atomic flag the caller raises, and qwentts checks it at the top of every
// decode step, so the floor is released within about one frame.
//
// The voice is a reference loaded once at startup from the voices directory.
// The talker is a Base one, and every unit is conditioned on the same speaker
// embedding, plus the same reference codes and transcript when the voice
// carries them, so every sentence continues the same recording.
//
// Sampling is a passthrough: the defaults come from the submodule through
// qt_tts_default_params, never from a copy kept here, so a bump of qwentts
// moves them without touching this project.
//
// Callbacks run on the qwentts compute worker, never on the calling thread.
// They must not call back into the bridge.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct tts_bridge;

// The generation knobs of the two stacks. Every field is only sent when it
// is set: a negative value leaves the submodule default in place.
struct tts_sampling {
    float   temperature           = -1.0f;
    int     top_k                 = -1;
    float   top_p                 = -1.0f;
    float   repetition_penalty    = -1.0f;
    float   subtalker_temperature = -1.0f;
    int     subtalker_top_k       = -1;
    float   subtalker_top_p       = -1.0f;
    int     max_new_tokens        = -1;
    int64_t seed                  = -1;  // negative draws a hardware seed, logged with the sentence
};

// The two guards that bound a synthesis. They are settings and not buried
// constants: a voice that gets cut short, or a recognizer that names noises,
// is tuned from the surface.
struct tts_guards {
    int   min_chars        = 3;      // below this a loopback transcript is not spoken back
    float chars_per_second = 15.0f;  // speech rate used to budget the frames
    float margin_seconds   = 2.0f;   // added to the budget, and its floor
};

// Engine knobs of the submodule, the ones a GPU sometimes needs. They belong
// to the process and not to a session, so they arrive from the command line.
struct tts_engine {
    int   max_batch       = 1;      // concurrent syntheses coalesced on the GPU
    bool  use_fa          = true;   // flash attention, off for a driver that misbehaves
    bool  clamp_fp16      = false;  // clamp the hidden states to the FP16 range
    float codec_chunk_sec = 0.0f;   // 0 keeps the submodule default
};

struct tts_bridge_params {
    std::string  talker_path;
    std::string  codec_path;
    std::string  voices_dir;  // one voice per <name>.spk, with <name>.rvq and <name>.txt as reference speech
    tts_sampling sampling;
    tts_guards   guards;
    tts_engine   engine;
};

// What one synthesis needs beyond its text. It travels per call because the
// handle is shared: two sessions speak with two voices at the same time, and
// a setting posted by one must never reach the other.
struct tts_request {
    std::string              voice;     // one of the labels tts_bridge_voices lists, empty keeps the default
    std::string              language;  // auto or one of tts_bridge_languages, empty keeps the default
    // the languages of the browser, by preference: under auto the first one
    // the talker knows gives the id, English stays when it knows none
    std::vector<std::string> browser_languages;
    tts_sampling             sampling;
    tts_guards               guards;
};

// Receives mono float PCM at the codec rate. Returning false cancels the
// synthesis in flight.
typedef bool (*tts_chunk_cb)(const float * pcm, size_t n_samples, void * user);

tts_bridge * tts_bridge_load(const tts_bridge_params & params);
void         tts_bridge_free(tts_bridge * b);

int tts_bridge_sample_rate(const tts_bridge * b);

// The languages of the talker, read from the model. auto is not among them:
// it is the absence of a language id, the default.
const std::vector<std::string> & tts_bridge_languages(const tts_bridge * b);

// Every way to speak with the loaded voices, one label each, voices sorted
// by name: freeman.{spk,rvq,txt} reference speech, then freeman.spk speaker
// embedding only. The first label is the default.
const std::vector<std::string> & tts_bridge_voices(const tts_bridge * b);

// The submodule defaults, so a caller can publish them instead of guessing.
const tts_sampling & tts_bridge_defaults(const tts_bridge * b);

// The request the bridge falls back on: what the server was started with,
// resolved against the model and the submodule defaults.
const tts_request & tts_bridge_defaults_request(const tts_bridge * b);

// Speaks one unit. cancel is polled by the synthesis and by the chunk
// callback, so raising it from any thread stops the audio. Returns false on
// failure or cancellation, with the reason in tts_bridge_last_error().
bool tts_bridge_speak(tts_bridge *              b,
                      const std::string &       text,
                      const tts_request &       request,
                      tts_chunk_cb              cb,
                      void *                    user,
                      const std::atomic<bool> * cancel);

const char * tts_bridge_last_error(void);
