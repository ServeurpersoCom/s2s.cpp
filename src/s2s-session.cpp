// s2s-session.cpp: the turn state machine
//
// One window of 512 samples is the unit of time here: 32 ms at 16 kHz. Every
// threshold is converted to a window count once, so the loop only compares
// integers.
//
// Two hystereses make the loop usable in a room. On the probability, speech
// starts at vad_threshold and lasts down to vad_neg_threshold, so a dip in
// the middle of a word does not break it. On time, opening a turn needs
// min_speech_ms of speech, so a chair or a breath never starts one, and
// barge_in_ms while the assistant speaks, since cutting it costs more than a
// turn opened on a noise, and the echo canceller leaves a residue. Reopening
// a turn that the classifier judged unfinished needs only
// min_speech_continuation_ms, because the speaker is already talking and the
// first syllable must not be lost.

#include "s2s-session.h"

#include "s2s-error.h"

#include <algorithm>
#include <cstring>

// Defined below, next to the state machine it belongs to.
static void s2s_session_commit(s2s_session * s, float score, int grace);

// The most recent samples of the stream, up to a fixed capacity: a push
// overwrites the oldest ones, and a read copies what is held out, oldest
// first. Nothing moves in memory on a push.
struct SampleRing {
    std::vector<float> data;
    size_t             head = 0;  // where the next sample goes
    size_t             size = 0;  // samples held, up to the capacity

    // A new capacity keeps the most recent samples that still fit.
    void resize(size_t capacity) {
        std::vector<float> held;
        read(held);
        data.assign(capacity, 0.0f);
        head              = 0;
        size              = 0;
        const size_t keep = std::min(held.size(), capacity);
        push(held.data() + held.size() - keep, keep);
    }

    void clear() {
        head = 0;
        size = 0;
    }

    void push(const float * samples, size_t n) {
        const size_t capacity = data.size();
        for (size_t i = 0; i < n && capacity > 0; i++) {
            data[head] = samples[i];
            head       = head + 1 == capacity ? 0 : head + 1;
        }
        size = std::min(size + n, capacity);
    }

    void read(std::vector<float> & out) const {
        const size_t capacity = data.size();
        out.resize(size);
        size_t at = (head + capacity - size) % (capacity ? capacity : 1);
        for (size_t i = 0; i < size; i++) {
            out[i] = data[at];
            at     = at + 1 == capacity ? 0 : at + 1;
        }
    }
};

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
    int barge_windows  = 0;  // the same while the assistant speaks
    int reopen_windows = 0;
    int close_windows  = 0;  // windows of silence that end the speech
    int pad_windows    = 0;
    int delay_windows  = 0;  // windows an incomplete turn waits before its commit
    int wait_windows   = 0;  // windows before the answer to an incomplete turn is heard
    int grace_windows  = 0;  // windows a complete commit keeps its answer silent

    s2s_session_state phase = S2S_SESSION_IDLE;

    std::vector<float> partial;         // samples that did not fill a window
    std::vector<float> turn_pcm;
    size_t             speech_end = 0;  // samples of turn_pcm up to the last speech
    SampleRing         lookback;        // recent windows, the speech pad reads from here
    SampleRing         stream;          // the classifier window, independent of the turn
    std::vector<float> scratch;         // the classifier window, laid out for the model

    int speech_run  = 0;                // consecutive windows above the threshold
    int silence_run = 0;
    int pending_run = 0;                // windows spent in PENDING_END

    int  grace_left  = 0;               // windows of grace left to the committed turn
    int  grace_given = 0;               // windows of grace the commit was given
    bool committed   = false;           // a committed turn not final yet, resumable until it is
    bool released    = false;           // its answer is ready to be heard, or over
    int  turn_id     = 0;
    int  revision    = 0;
    bool speaking    = false;
    bool in_speech   = false;  // the VAD side of the hysteresis

    size_t n_windows = 0;      // windows consumed since the start of the stream
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
    s->stream.resize((size_t) st_window(turn));

    s2s_session_set_params(s, params);
    return s;
}

void s2s_session_set_params(s2s_session * s, const s2s_session_params & params) {
    if (!s) {
        return;
    }
    s->params         = params;
    s->open_windows   = s2s_session_windows(params.min_speech_ms, s->window, s->sample_rate);
    s->barge_windows  = s2s_session_windows(params.barge_in_ms, s->window, s->sample_rate);
    s->reopen_windows = s2s_session_windows(params.min_speech_continuation_ms, s->window, s->sample_rate);
    s->close_windows  = s2s_session_windows(params.min_silence_ms, s->window, s->sample_rate);
    s->pad_windows    = s2s_session_windows(params.speech_pad_ms, s->window, s->sample_rate);
    s->wait_windows   = s2s_session_windows(params.turn_max_wait_ms, s->window, s->sample_rate);
    s->delay_windows =
        std::min(s->wait_windows, s2s_session_windows(params.incomplete_delay_ms, s->window, s->sample_rate));
    s->grace_windows =
        params.reopen_grace_ms > 0 ? s2s_session_windows(params.reopen_grace_ms, s->window, s->sample_rate) : 0;

    // The lookback holds the windows a turn needs to open plus the speech pad,
    // so the audio starts before the first syllable that crossed the
    // threshold instead of in the middle of it.
    s->lookback.resize((size_t) (std::max(s->open_windows, s->barge_windows) + s->pad_windows) * (size_t) s->window);

    s2s_log(
        S2S_LOG_INFO,
        "[Session] Window %d samples, open %d, barge-in %d, reopen %d, close %d, delay %d, wait %d, grace %d windows",
        s->window, s->open_windows, s->barge_windows, s->reopen_windows, s->close_windows, s->delay_windows,
        s->wait_windows, s->grace_windows);
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
    s2s_session_commit(s, 0.0f, 0);
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
    s->grace_left  = 0;
    s->committed   = false;
    s->released    = false;
    s->speaking    = false;
    s->in_speech   = false;
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
        const size_t pad = (size_t) s->pad_windows * (size_t) s->window;
        report.pcm       = s->turn_pcm.data();
        report.n_samples = std::min(s->turn_pcm.size(), s->speech_end + pad);
        report.n_held    = s->turn_pcm.size();
        report.grace_sec = (double) (s->grace_given * (size_t) s->window) / (double) s->sample_rate;
    }
    s->cb(&report, s->user);
}

// The committed turn stands once its grace ran out and its answer is ready:
// the answer may be heard and the audio goes.
static void s2s_session_try_final(s2s_session * s) {
    if (s->committed && s->released && s->grace_left == 0) {
        s->committed = false;
        s->turn_pcm.clear();
        s2s_session_emit(s, S2S_EVENT_TURN_FINAL, 0.0f);
    }
}

void s2s_session_release(s2s_session * s, int turn_id, int revision) {
    if (!s || !s->committed || turn_id != s->turn_id || revision != s->revision) {
        return;
    }
    s->released = true;
    s2s_session_try_final(s);
}

// Opens a turn on the window that crossed the threshold, and prepends the
// lookback so the first consonant is not clipped.
static void s2s_session_open_turn(s2s_session * s) {
    s->turn_id++;
    s->revision   = 0;
    s->grace_left = 0;
    s->committed  = false;
    s->lookback.read(s->turn_pcm);
    s->speech_end = s->turn_pcm.size();
    s->phase      = S2S_SESSION_USER_SPEAKING;
    s2s_session_emit(s, S2S_EVENT_SPEECH_STARTED, 0.0f);
}

// Hands the turn over with grace windows of silence on its answer, and keeps
// its audio, which a resumption continues, until the turn is final. A
// resumption takes speech that starts after the commit: a commit made mid
// word does not resume on the rest of that word.
static void s2s_session_commit(s2s_session * s, float score, int grace) {
    s->phase       = S2S_SESSION_IDLE;
    s->pending_run = 0;
    s->speech_run  = 0;
    s->committed   = true;
    s->released    = false;
    s->grace_left  = grace;
    s->grace_given = grace;
    s2s_session_emit(s, S2S_EVENT_TURN_COMMITTED, score);
}

// One 512 sample window: the grace, the probability, then the state machine.
// The grace counts the windows after the one that committed, so it lasts
// exactly what the commit was given.
static void s2s_session_window(s2s_session * s, const float * window) {
    if (s->grace_left > 0 && --s->grace_left == 0) {
        s2s_session_try_final(s);
    }

    // Two thresholds, the Silero hysteresis: speech starts at vad_threshold
    // and lasts while the probability stays at or above vad_neg_threshold,
    // so a dip inside a word, or a voice the echo canceller left fainter,
    // does not cut it. A lower bound set above the upper one is the upper.
    const float prob     = sv_prob(s->state, window, s->window);
    const float stay     = std::min(s->params.vad_neg_threshold, s->params.vad_threshold);
    s->in_speech         = prob >= (s->in_speech ? stay : s->params.vad_threshold);
    const bool is_speech = s->in_speech;

    s->speech_run  = is_speech ? s->speech_run + 1 : 0;
    s->silence_run = is_speech ? 0 : s->silence_run + 1;

    // Both rings follow the stream, not the turn: the classifier judges a
    // boundary with everything that led to it.
    s->lookback.push(window, (size_t) s->window);
    s->stream.push(window, (size_t) s->window);

    if (s->phase != S2S_SESSION_IDLE || s->committed) {
        s->turn_pcm.insert(s->turn_pcm.end(), window, window + s->window);
        if (is_speech) {
            s->speech_end = s->turn_pcm.size();
        }
    }

    switch (s->phase) {
        case S2S_SESSION_IDLE:
            {
                // The speaker goes on before the turn is final: the same
                // turn, one revision further, its audio continuous.
                if (s->committed && s->speech_run >= s->reopen_windows) {
                    s->grace_left = 0;
                    s->committed  = false;
                    s->revision++;
                    s->phase = S2S_SESSION_USER_SPEAKING;
                    s2s_session_emit(s, S2S_EVENT_TURN_RESUMED, 0.0f);
                } else if (s->speech_run >= (s->speaking ? s->barge_windows : s->open_windows)) {
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

                    s->stream.read(s->scratch);
                    const float score = st_predict(s->turn, s->scratch.data(), (int) s->scratch.size());
                    s2s_log(S2S_LOG_INFO, "[Session] Turn classifier on %.2fs of stream, completion %.3f",
                            (double) s->scratch.size() / s->sample_rate, (double) score);
                    if (score >= s->params.turn_threshold) {
                        s2s_session_commit(s, score, s->grace_windows);
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
                } else if (s->pending_run >= s->delay_windows) {
                    // The classifier judged the turn unfinished and the
                    // speaker did not come back within the delay: the answer
                    // starts computing now and stays silent until the wait
                    // is over, so the floor goes to the assistant by then
                    // and the conversation never stalls.
                    s2s_session_commit(s, 0.0f, s->wait_windows - s->delay_windows);
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
