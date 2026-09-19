#pragma once
// s2s-session.h: the turn state machine of one conversation
//
// Owns the listening half of the loop: it consumes 16 kHz mono audio window
// by window, runs the VAD on every window and the turn classifier on every
// speech to silence boundary, and reports what it decided through events.
//
//   IDLE          --speech >= min_speech_ms-->            USER_SPEAKING
//   USER_SPEAKING --silence >= min_silence_ms-->          PENDING_END
//   PENDING_END   --turn complete-->                      committed, IDLE
//   PENDING_END   --incomplete, then turn_max_wait_ms-->  committed, IDLE
//   PENDING_END   --speech >= min_speech_continuation_ms--> USER_SPEAKING
//   grace         --speech >= min_speech_continuation_ms--> USER_SPEAKING, same turn
//
// A turn keeps its identity across a reopening: the revision counter grows
// and the audio keeps accumulating, so the recognizer sees the whole
// utterance as one piece.
//
// A commit the classifier judged complete keeps its answer silent for
// reopen_grace_ms, counted on the audio, then the session reports the turn
// final: a breath inside a sentence never lets the assistant cut in. The
// turn stays open during the grace, its audio still accumulating silence
// included: a speaker who goes on resumes it, and the next commit hands the
// whole utterance over again under a new revision. A commit forced by
// turn_max_wait_ms or by the caller has waited already and is final at once.
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

enum s2s_session_state {
    S2S_SESSION_IDLE = 0,
    S2S_SESSION_USER_SPEAKING,
    S2S_SESSION_PENDING_END,
};

enum s2s_session_event {
    S2S_EVENT_SPEECH_STARTED = 0,  // a turn opened
    S2S_EVENT_SPEECH_STOPPED,      // the speech to silence boundary
    S2S_EVENT_TURN_REOPENED,       // the classifier said the turn was not over
    S2S_EVENT_TURN_COMMITTED,      // the audio is ready for the recognizer
    S2S_EVENT_TURN_FINAL,          // the committed turn stands, its answer may be heard
    S2S_EVENT_TURN_RESUMED,        // the speaker went on during the grace: same turn, next revision
    S2S_EVENT_BARGE_IN,            // the user spoke while the assistant held the floor
};

struct s2s_session_params {
    float vad_threshold              = 0.6f;   // speech starts at this probability
    float vad_neg_threshold          = 0.45f;  // and lasts while it stays at or above this one
    int   min_speech_ms              = 384;
    int   min_speech_continuation_ms = 192;
    int   min_silence_ms             = 64;
    int   speech_pad_ms              = 500;
    float turn_threshold             = 0.5f;
    int   turn_max_wait_ms           = 2000;
    int   reopen_grace_ms            = 800;
};

// One event, with the audio attached when the turn is committed.
struct s2s_session_report {
    s2s_session_event event;
    int               turn_id;
    int               revision;
    double            time_sec;    // position in the stream where the event fired
    float             turn_score;  // classifier output, only on a boundary
    const float *     pcm;         // committed turn audio, only on a commit
    size_t            n_samples;
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
// turns a speech start into a barge-in. Like every other call, it belongs to
// the thread that pushes the audio.
void s2s_session_set_speaking(s2s_session * s, bool speaking);

// Commits the turn in flight right now, whatever the classifier thinks. This
// is what a push to talk button and the end of a stream need. No turn open
// means nothing happens.
void s2s_session_commit_now(s2s_session * s);

// Drops the current turn and the model states, for a new conversation.
void s2s_session_reset(s2s_session * s);

s2s_session_state s2s_session_get_state(const s2s_session * s);
