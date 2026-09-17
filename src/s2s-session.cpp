// s2s-session.cpp: the turn state machine
//
// One window of 512 samples is the unit of time here: 32 ms at 16 kHz. Every
// threshold is converted to a window count once, so the loop only compares
// integers.
//
// The hysteresis is what makes the loop usable in a room: opening a turn
// needs min_speech_ms of speech, so a chair or a breath never starts one,
// while reopening a turn that the classifier judged unfinished needs only
// min_speech_continuation_ms, because the speaker is already talking and the
// first syllable must not be lost.

#include "s2s-session.h"

#include "s2s-error.h"

#include <algorithm>
#include <cstring>

// Defined below, next to the state machine it belongs to.
static void s2s_session_commit(s2s_session * s, float score);

struct s2s_session {
    sv_context * vad   = nullptr;
    st_context * turn  = nullptr;
    sv_state *   state = nullptr;

    s2s_session_params params;
    s2s_session_cb     cb   = nullptr;
    void *             user = nullptr;

    int window         = 0;  // samples per VAD window
    int sample_rate    = 0;
    int open_windows   = 0;  // windows of speech needed to open a turn
    int reopen_windows = 0;
    int close_windows  = 0;  // windows of silence that end the speech
    int pad_windows    = 0;
    int wait_windows   = 0;  // windows to wait on an incomplete turn

    s2s_session_state phase = S2S_SESSION_IDLE;

    std::vector<float> partial;   // samples that did not fill a window
    std::vector<float> turn_pcm;
    std::vector<float> lookback;  // recent windows, the speech pad reads from here
    std::vector<float> stream;    // the classifier window, independent of the turn

    int    speech_run     = 0;    // consecutive windows above the threshold
    int    silence_run    = 0;
    int    pending_run    = 0;    // windows spent in PENDING_END
    size_t stream_samples = 0;    // what the classifier wants, in samples

    int  turn_id  = 0;
    int  revision = 0;
    bool speaking = false;

    size_t n_windows = 0;  // windows consumed since the start of the stream
};

static int s2s_session_windows(int ms, int window, int sample_rate) {
    const double window_ms = 1000.0 * (double) window / (double) sample_rate;
    const int    n         = (int) ((double) ms / window_ms + 0.5);
    return n > 0 ? n : 1;
}

s2s_session * s2s_session_new(sv_context *               vad,
                              st_context *               turn,
                              const s2s_session_params & params,
                              s2s_session_cb             cb,
                              void *                     user) {
    if (!vad || !turn) {
        s2s_set_error("[Session] Vad or turn context is NULL");
        return nullptr;
    }

    s2s_session * s = new s2s_session();
    s->vad          = vad;
    s->turn         = turn;
    s->params       = params;
    s->cb           = cb;
    s->user         = user;

    s->state = sv_state_new(vad);
    if (!s->state) {
        delete s;
        return nullptr;
    }

    s->window      = sv_window(vad);
    s->sample_rate = sv_sample_rate(vad);

    // The turn classifier reads a fixed window of the stream, whatever the
    // turn holds: a short burst judged on its own, surrounded by silence,
    // looks like a finished sentence to it. Prosody needs its context.
    s->stream_samples = (size_t) st_window(turn);

    s->open_windows   = s2s_session_windows(params.min_speech_ms, s->window, s->sample_rate);
    s->reopen_windows = s2s_session_windows(params.min_speech_continuation_ms, s->window, s->sample_rate);
    s->close_windows  = s2s_session_windows(params.min_silence_ms, s->window, s->sample_rate);
    s->pad_windows    = s2s_session_windows(params.speech_pad_ms, s->window, s->sample_rate);
    s->wait_windows   = s2s_session_windows(params.turn_max_wait_ms, s->window, s->sample_rate);

    s2s_log(S2S_LOG_INFO, "[Session] Window %d samples, open %d, reopen %d, close %d, wait %d windows", s->window,
            s->open_windows, s->reopen_windows, s->close_windows, s->wait_windows);
    return s;
}

void s2s_session_free(s2s_session * s) {
    if (!s) {
        return;
    }
    sv_state_free(s->state);
    delete s;
}

void s2s_session_set_speaking(s2s_session * s, bool speaking) {
    if (s) {
        s->speaking = speaking;
    }
}

s2s_session_state s2s_session_get_state(const s2s_session * s) {
    return s ? s->phase : S2S_SESSION_IDLE;
}

void s2s_session_commit_now(s2s_session * s) {
    if (!s || s->phase == S2S_SESSION_IDLE || s->turn_pcm.empty()) {
        return;
    }
    s2s_session_commit(s, 0.0f);
}

void s2s_session_reset(s2s_session * s) {
    if (!s) {
        return;
    }
    sv_state_reset(s->state);
    s->partial.clear();
    s->turn_pcm.clear();
    s->lookback.clear();
    s->stream.clear();
    s->phase       = S2S_SESSION_IDLE;
    s->speech_run  = 0;
    s->silence_run = 0;
    s->pending_run = 0;
    s->revision    = 0;
    s->speaking    = false;
}

static void s2s_session_emit(s2s_session * s, s2s_session_event event, float score) {
    if (!s->cb) {
        return;
    }
    s2s_session_report report = {};
    report.event              = event;
    report.turn_id            = s->turn_id;
    report.revision           = s->revision;
    report.time_sec           = (double) (s->n_windows * (size_t) s->window) / (double) s->sample_rate;
    report.turn_score         = score;
    if (event == S2S_EVENT_TURN_COMMITTED) {
        report.pcm       = s->turn_pcm.data();
        report.n_samples = s->turn_pcm.size();
    }
    s->cb(&report, s->user);
}

// Opens a turn on the window that crossed the threshold, and prepends the
// lookback so the first consonant is not clipped.
static void s2s_session_open_turn(s2s_session * s) {
    s->turn_id++;
    s->revision = 0;
    s->turn_pcm.assign(s->lookback.begin(), s->lookback.end());
    s->phase = S2S_SESSION_USER_SPEAKING;
    s2s_session_emit(s, S2S_EVENT_SPEECH_STARTED, 0.0f);
}

static void s2s_session_commit(s2s_session * s, float score) {
    s2s_session_emit(s, S2S_EVENT_TURN_COMMITTED, score);
    s->turn_pcm.clear();
    s->phase       = S2S_SESSION_IDLE;
    s->pending_run = 0;
    s->revision    = 0;
}

// One 512 sample window: probability, then the state machine.
static void s2s_session_window(s2s_session * s, const float * window) {
    const float prob      = sv_prob(s->state, window, s->window);
    const bool  is_speech = prob >= s->params.vad_threshold;

    s->speech_run  = is_speech ? s->speech_run + 1 : 0;
    s->silence_run = is_speech ? 0 : s->silence_run + 1;

    // The lookback holds the windows a turn needs to open plus the speech pad,
    // so the audio starts before the first syllable that crossed the
    // threshold instead of in the middle of it.
    s->lookback.insert(s->lookback.end(), window, window + s->window);
    const size_t lookback_max = (size_t) (s->open_windows + s->pad_windows) * (size_t) s->window;
    if (s->lookback.size() > lookback_max) {
        s->lookback.erase(s->lookback.begin(), s->lookback.end() - (ptrdiff_t) lookback_max);
    }

    // The classifier window follows the stream, not the turn, so a boundary is
    // judged with everything that led to it.
    s->stream.insert(s->stream.end(), window, window + s->window);
    if (s->stream.size() > s->stream_samples) {
        s->stream.erase(s->stream.begin(), s->stream.end() - (ptrdiff_t) s->stream_samples);
    }

    if (s->phase != S2S_SESSION_IDLE) {
        s->turn_pcm.insert(s->turn_pcm.end(), window, window + s->window);
    }

    switch (s->phase) {
        case S2S_SESSION_IDLE:
            {
                if (s->speech_run >= s->open_windows) {
                    if (s->speaking) {
                        s2s_session_emit(s, S2S_EVENT_BARGE_IN, 0.0f);
                    }
                    s2s_session_open_turn(s);
                }
                break;
            }

        case S2S_SESSION_USER_SPEAKING:
            {
                if (s->silence_run >= s->close_windows) {
                    s->phase       = S2S_SESSION_PENDING_END;
                    s->pending_run = 0;
                    s2s_session_emit(s, S2S_EVENT_SPEECH_STOPPED, 0.0f);

                    const float score = st_predict(s->turn, s->stream.data(), (int) s->stream.size());
                    s2s_log(S2S_LOG_INFO, "[Session] Turn classifier on %.2fs of stream, completion %.3f",
                            (double) s->stream.size() / s->sample_rate, (double) score);
                    if (score >= s->params.turn_threshold) {
                        s2s_session_commit(s, score);
                    } else {
                        s2s_session_emit(s, S2S_EVENT_TURN_REOPENED, score);
                    }
                }
                break;
            }

        case S2S_SESSION_PENDING_END:
            {
                s->pending_run++;
                if (s->speech_run >= s->reopen_windows) {
                    s->revision++;
                    s->phase = S2S_SESSION_USER_SPEAKING;
                    s2s_session_emit(s, S2S_EVENT_SPEECH_STARTED, 0.0f);
                } else if (s->pending_run >= s->wait_windows) {
                    // The classifier judged the turn unfinished and the
                    // speaker never came back: the floor goes to the assistant
                    // anyway, otherwise the conversation stalls.
                    s2s_session_commit(s, 0.0f);
                }
                break;
            }
    }

    s->n_windows++;
}

void s2s_session_push(s2s_session * s, const float * pcm, size_t n_samples) {
    if (!s || !pcm) {
        return;
    }

    size_t offset = 0;

    if (!s->partial.empty()) {
        const size_t missing = (size_t) s->window - s->partial.size();
        const size_t take    = std::min(missing, n_samples);
        s->partial.insert(s->partial.end(), pcm, pcm + take);
        offset = take;
        if (s->partial.size() < (size_t) s->window) {
            return;
        }
        s2s_session_window(s, s->partial.data());
        s->partial.clear();
    }

    while (offset + (size_t) s->window <= n_samples) {
        s2s_session_window(s, pcm + offset);
        offset += (size_t) s->window;
    }

    s->partial.assign(pcm + offset, pcm + n_samples);
}
