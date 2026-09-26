#pragma once
// s2s-session.h: the turn state machine of one conversation
//
// Owns the listening half of the loop: it consumes 16 kHz mono audio window
// by window, runs the VAD on every window and the turn classifier on every
// speech to silence boundary, and reports what it decided through events.
//
//   IDLE          --speech >= min_speech_ms-->            USER_SPEAKING
//   IDLE, speaking --speech >= barge_in_ms-->             barge-in, USER_SPEAKING
//   USER_SPEAKING --silence >= min_silence_ms-->          PENDING_END
//   PENDING_END   --turn complete-->                      committed, IDLE
//   PENDING_END   --incomplete, then incomplete_delay_ms--> committed, IDLE
//   PENDING_END   --speech >= min_speech_continuation_ms--> USER_SPEAKING
//   committed     --speech >= min_speech_continuation_ms--> USER_SPEAKING, same turn
//   committed     --grace over and released-->           final
//
// A turn keeps its identity across a reopening: the revision counter grows
// and the audio keeps accumulating, so the recognizer sees the whole
// utterance as one piece.
//
// A committed turn turns final once two things hold: its grace has run out,
// and the caller released it, which it does when the answer is ready to be
// heard or over. Until then the turn stays open, its audio still
// accumulating silence included: a speaker who goes on resumes it, and the
// next commit hands the whole utterance over again under a new revision. A
// user who has heard nothing of the answer is still in the same turn.
//
// The grace keeps the answer silent, counted on the audio, so a breath
// inside a sentence never lets the assistant cut in: reopen_grace_ms after a
// commit the classifier judged complete. A turn it judged unfinished waits
// incomplete_delay_ms with nothing running, the pauses of a breath, then
// commits with the rest of turn_max_wait_ms as its grace: the answer is
// computed while the silence lasts and heard at turn_max_wait_ms at the
// latest.
//
// The committed audio ends speech_pad_ms after the last speech: the silence
// a turn keeps while it waits is not handed over, since a short word drowned
// in seconds of it comes back from the recognizer empty.
//
// The session never transcribes and never speaks. It hands the committed
// audio to the caller, which owns the recognizer worker, and it is told when
// the assistant holds the floor so it can arm the barge-in. That keeps this
// file free of any model beyond the two that run on every window.

#include "silero.h"
#include "smart-turn.h"

#include <cstddef>
#include <string>
#include <vector>

enum s2s_session_event {
    S2S_EVENT_SPEECH_STARTED = 0,  // a turn opened
    S2S_EVENT_SPEECH_STOPPED,      // the speech to silence boundary
    S2S_EVENT_TURN_REOPENED,       // the classifier said the turn was not over
    S2S_EVENT_TURN_COMMITTED,      // the audio is ready for the recognizer
    S2S_EVENT_TURN_FINAL,          // the committed turn stands, its answer may be heard
    S2S_EVENT_TURN_RESUMED,        // the speaker went on before the turn was final: same turn, next revision
    S2S_EVENT_BARGE_IN,            // the user spoke while the assistant held the floor
};

struct s2s_session_params {
    // In the order a turn goes through them. Speech starts at vad_threshold
    // and lasts while the probability stays at or above vad_neg_threshold.
    float vad_neg_threshold          = 0.45f;
    float vad_threshold              = 0.6f;
    int   min_speech_ms              = 192;   // opens a turn while the assistant is silent
    int   barge_in_ms                = 384;   // opens one over the assistant, and cuts it
    int   min_silence_ms             = 64;    // ends the speech
    int   min_speech_continuation_ms = 192;   // resumes it
    int   speech_pad_ms              = 500;   // audio kept before the onset and after the end
    float turn_threshold             = 0.5f;
    int   incomplete_delay_ms        = 600;   // nothing runs on an unfinished turn until then
    int   turn_max_wait_ms           = 2000;  // the answer to it is heard by then
    int   reopen_grace_ms            = 800;   // silence kept on the answer to a finished one
};

// One event, with the audio attached when the turn is committed.
struct s2s_session_report {
    s2s_session_event event;
    int               turn_id;
    int               revision;
    double            time_sec;    // position in the stream where the event fired
    float             turn_score;  // classifier output, only on a boundary
    const float *     pcm;         // committed turn audio, only on a commit
    size_t            n_samples;   // up to speech_pad_ms after the last speech
    size_t            n_held;      // what the turn holds, the silence it waited through included
    double            grace_sec;   // silence kept on the answer, only on a commit
};

typedef void (*s2s_session_cb)(const s2s_session_report * report, void * user);

struct s2s_session;

// vad and turn stay owned by the caller: one instance of each model serves
// every session.
s2s_session * s2s_session_new(sv_context *               vad,
                              st_context *               turn,
                              const s2s_session_params & params,
                              s2s_session_cb             cb,
                              void *                     user);
void          s2s_session_free(s2s_session * s);

// Applies new thresholds to a running session. The turn in flight, the model
// states and the turn numbering are kept: a threshold moved in the middle of
// a sentence does not cut it.
void s2s_session_set_params(s2s_session * s, const s2s_session_params & params);

// Feeds mono 16 kHz audio. Any length works: the session buffers what does
// not fill a window.
void s2s_session_push(s2s_session * s, const float * pcm, size_t n_samples);

// Tells the session whether the assistant holds the floor, which is what
// turns a speech start into a barge-in, held to barge_in_ms: a word said over
// the assistant has to outlast what the echo canceller leaves of it. The
// session holds no lock: a caller that shares it between threads serializes
// every call, this one included.
void s2s_session_set_speaking(s2s_session * s, bool speaking);

// Releases the committed turn: its answer is ready to be heard, or over. The
// turn is final at once if its grace has run out, at the end of the grace
// otherwise. A release that names another turn or an older revision does
// nothing, so the caller may repeat it freely.
void s2s_session_release(s2s_session * s, int turn_id, int revision);
