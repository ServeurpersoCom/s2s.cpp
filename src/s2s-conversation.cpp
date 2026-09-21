// s2s-conversation.cpp: the voice loop of one conversation
//
// One process, any number of conversations. Every model loads once and is
// shared: the VAD, the turn classifier and the echo canceller keep their per
// stream state in the session or the connection, while the recognizer and
// the voice serialize internally, so a second client queues rather than
// doubling the VRAM.
//
// Each connection runs four threads. The reader owns the incoming frames and
// the listening half: it decodes them, feeds the session, and answers the
// events the state machine raises. The recognizer transcribes every
// committed turn and revision as soon as it is committed, and hands it to
// the responder, which asks the endpoint and speaks the answer unit by unit.
// Splitting them is what makes the barge-in work, since the reader keeps
// consuming audio while an answer runs, and what keeps the words of the user
// live whatever the endpoint does: a request stuck on the network holds the
// answer, never the next transcript. The writer sends every outgoing frame,
// so a client that reads slowly never holds the synthesis worker the
// connections share.
//
// Modes:
//   conversation  recognize, ask the LLM, speak the answer
//   loopback      recognize and speak the transcript back, no endpoint in the
//                 path, which is how the microphone, the turn detection, the
//                 recognizer and the voice get tested on their own

#include "s2s-conversation.h"

#include "llm-agent.h"
#include "s2s-error.h"
#include "s2s-session.h"
#include "sentence-split.h"
#include "timer.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

std::string url_host(const std::string & url) {
    const size_t scheme = url.find("://");
    const size_t start  = scheme == std::string::npos ? 0 : scheme + 3;
    const size_t end    = url.find('/', start);
    return url.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

bool host_allowed(const std::vector<std::string> & hosts, const std::string & url) {
    if (hosts.empty()) {
        return true;
    }
    const std::string host = url_host(url);
    for (const std::string & entry : hosts) {
        if (host == url_host(entry)) {
            return true;
        }
    }
    return false;
}

// A committed turn, with the identity the session gave it.
struct TurnAudio {
    std::vector<float> pcm;
    int                turn_id  = 0;
    int                revision = 0;
};

// A recognized turn, waiting for its answer.
struct AnswerJob {
    std::string transcript;
    int         turn_id  = 0;
    int         revision = 0;
};

struct Connection {
    // What every conversation of the process shares, and how a frame reaches
    // the client: the transport belongs to whoever opened the conversation.
    const ConversationSetup * setup     = nullptr;
    conn_send_fn              send      = nullptr;
    void *                    send_user = nullptr;

    // Numbers the connection in the log: its three threads write as
    // Reader-N, Responder-N and Writer-N, so one grep follows one client.
    int id = 0;

    // The turn state machine. The reader feeds it and the responder releases
    // its turns, each call under session_mutex. Its events run under that
    // lock and take the turn, queue and output locks after it; a session
    // update takes it under client_mutex. Never the reverse.
    std::mutex         session_mutex;
    s2s_session *      session = nullptr;
    s2s_session_params params;

    // Only server runs a canceller here: native and off leave the microphone
    // as the client captured it. The default method is the component's, which
    // picks its microphone constraints before the server is even reached.
    std::string echo = "native";

    // Written by the reader, copied whole by the responder when it answers a
    // turn: a change lands on the next answer, never under a running one.
    std::mutex     client_mutex;
    ClientSettings client;

    // Frames to the client. Every thread queues, one writer sends: a client
    // that reads slowly stalls its own writer, never the synthesis worker
    // every connection shares.
    std::mutex              out_mutex;
    std::condition_variable out_cv;
    std::queue<std::string> out;
    bool                    out_stop = false;

    // The endpoint client, the responder's alone. It lives across turns so its
    // connection stays open: a turn pays the TCP and TLS handshakes only when
    // the endpoint host changes.
    llm_client * llm = nullptr;

    // Committed turns waiting for the recognizer, and recognized ones waiting
    // for the responder.
    std::mutex              queue_mutex;
    std::condition_variable queue_cv;
    std::queue<TurnAudio>   queue;
    std::condition_variable answers_cv;
    std::queue<AnswerJob>   answers;
    std::atomic<bool>       stop{ false };

    // Raised against the answer in flight; the responder lowers it as it
    // starts the next one.
    std::atomic<bool> cancel{ false };

    // The first sample of an answer waits for its turn to be final. The
    // session decides on the reader thread: the latest turn declared final,
    // and the latest turn and revision that opened. An answer to anything
    // older than what opened last is void.
    std::mutex              turn_mutex;
    std::condition_variable turn_cv;
    int                     final_turn    = 0;
    int                     open_turn     = 0;
    int                     open_revision = 0;

    // The answer in flight: its turn and revision, written by the responder
    // under turn_mutex and read there by the reader, and whether its first
    // unit has gone past the wait, the responder's alone.
    int  answer_turn     = 0;
    int  answer_revision = 0;
    bool answer_released = false;

    // Whether the assistant holds the floor: set by the responder, handed to
    // the session by the reader with every push.
    std::atomic<bool> speaking{ false };

    // The protocol carries 24 kHz, the models want 16 kHz: the microphone and
    // the reference go through the same Hann-windowed sinc, the one of
    // torchaudio, so they stay aligned sample for sample.
    AudioResampleStream mic_resample;
    AudioResampleStream ref_resample;

    // Server side echo cancellation, alive while the client asks for it. The
    // canceller works in hops, so what does not fill one waits here.
    lv_state *         aec = nullptr;
    std::vector<float> aec_mic;
    std::vector<float> aec_ref;

    // Energy in and out of the canceller over one run of playback, and the
    // compute of every hop since the last report, for the log lines that
    // close the run. The output is one hop late, so the energy of the
    // microphone hop waits one call for the output it produced.
    double aec_in         = 0.0;
    double aec_out        = 0.0;
    size_t aec_played     = 0;
    int    aec_quiet      = 0;
    size_t aec_hops       = 0;
    double aec_ms         = 0.0;
    double aec_peak       = 0.0;
    double aec_pending_in = 0.0;
    bool   aec_pending    = false;
    bool   aec_failing    = false;

    // The threads of the talking half, and what the reader keeps between two
    // frames: whether the microphone was announced, and its buffers.
    std::thread        writer;
    std::thread        recognizer;
    std::thread        responder;
    bool               receiving = false;
    std::vector<float> resampled;
    std::vector<float> reference;
    std::vector<float> silence;
};

// Frames without playback that close a run of it, 0.5 s of client frames: a
// shorter gap is a pause between two sentences of the same answer.
#define AEC_QUIET_FRAMES 25

static void conn_send(Connection * conn, std::string frame) {
    std::lock_guard<std::mutex> lock(conn->out_mutex);
    if (!conn->out_stop) {
        conn->out.push(std::move(frame));
        conn->out_cv.notify_one();
    }
}

// A failure goes to the client and to the log alike: the client sees what
// went wrong, and the log keeps it. The messages carry the tag of the module
// that failed.
static void conn_error(Connection * conn, const char * message) {
    s2s_log(S2S_LOG_WARN, "%s", message);
    conn_send(conn, rt_event_error(message));
}

// Runs the canceller over every complete hop and hands back in pcm what is
// clean so far. The canceller is one hop late and its hops of 256 samples do
// not line up with the client frames, so the cleaned stream trails the raw
// one by one to two hops, about 16 to 32 ms.
static void conn_cancel_echo(Connection * conn, std::vector<float> & pcm, const std::vector<float> & ref, bool played) {
    conn->aec_mic.insert(conn->aec_mic.end(), pcm.begin(), pcm.end());
    conn->aec_ref.insert(conn->aec_ref.end(), ref.begin(), ref.end());

    const size_t hop  = (size_t) lv_hop(conn->setup->models.aec);
    size_t       done = 0;
    pcm.clear();
    for (; done + hop <= conn->aec_mic.size(); done += hop) {
        const size_t base = pcm.size();
        pcm.resize(base + hop);
        Timer      t_hop;
        const bool failed =
            lv_process(conn->aec, conn->aec_mic.data() + done, conn->aec_ref.data() + done, pcm.data() + base) != 0;
        if (failed && !conn->aec_failing) {
            s2s_log(S2S_LOG_ERROR, "[AEC] %s", lv_last_error());
            conn_send(conn, rt_event_error(lv_last_error()));
        }
        conn->aec_failing = failed;

        const double ms = t_hop.ms();
        conn->aec_ms += ms;
        conn->aec_peak = ms > conn->aec_peak ? ms : conn->aec_peak;
        conn->aec_hops++;

        // pcm holds the output of the previous hop: it is weighed against the
        // microphone energy kept from the previous call, when that hop played.
        if (conn->aec_pending) {
            conn->aec_in += conn->aec_pending_in;
            for (size_t i = 0; i < hop; i++) {
                conn->aec_out += (double) pcm[base + i] * pcm[base + i];
            }
            conn->aec_played += hop;
        }
        conn->aec_pending    = false;
        conn->aec_pending_in = 0.0;
        for (size_t i = 0; i < hop; i++) {
            conn->aec_pending = conn->aec_pending || conn->aec_ref[done + i] != 0.0f;
            conn->aec_pending_in += (double) conn->aec_mic[done + i] * conn->aec_mic[done + i];
        }
    }
    conn->aec_mic.erase(conn->aec_mic.begin(), conn->aec_mic.begin() + (ptrdiff_t) done);
    conn->aec_ref.erase(conn->aec_ref.begin(), conn->aec_ref.begin() + (ptrdiff_t) done);

    conn->aec_quiet = played ? 0 : conn->aec_quiet + 1;
    if (conn->aec_played > 0 && conn->aec_quiet == AEC_QUIET_FRAMES) {
        s2s_log(S2S_LOG_INFO, "[AEC] Microphone %.1f dB above the cleaned signal over %.2fs of playback",
                10.0 * log10((conn->aec_in + 1e-12) / (conn->aec_out + 1e-12)),
                (double) conn->aec_played / S2S_MODEL_RATE);
        const double hop_ms = (double) hop * 1000.0 / S2S_MODEL_RATE;
        const double mean   = conn->aec_ms / (double) conn->aec_hops;
        s2s_log(S2S_LOG_INFO, "[Perf] AEC %zu hops, %.2f ms per hop, peak %.2f ms, %.1fx real time", conn->aec_hops,
                mean, conn->aec_peak, hop_ms / mean);
        conn->aec_in     = 0.0;
        conn->aec_out    = 0.0;
        conn->aec_played = 0;
        conn->aec_hops   = 0;
        conn->aec_ms     = 0.0;
        conn->aec_peak   = 0.0;
    }
}

// Wakes whatever waits on the turn: a final turn, a new one, a cancel or the
// end of the connection.
static void conn_turn_notify(Connection * conn) {
    std::lock_guard<std::mutex> lock(conn->turn_mutex);
    conn->turn_cv.notify_all();
}

// Ends the talking half: the answer in flight stops and the queued turns are
// dropped, nobody is left to hear them.
static void conn_stop(Connection * conn) {
    conn->cancel.store(true);
    {
        std::lock_guard<std::mutex> lock(conn->queue_mutex);
        conn->stop = true;
        conn->queue_cv.notify_one();
        conn->answers_cv.notify_one();
    }
    conn_turn_notify(conn);
}

// Sends the queued frames in order. A failed send means the client is gone,
// or stopped reading for longer than the write timeout: the answer in flight
// stops, nothing more is queued for it, and the connection is over. The
// reader closes it: it is the one thread that reads the socket, closing
// handshake included. It sees the stop with the next frame the client sends,
// or, from a client gone silent, when its read times out.
static void conn_writer(Connection * conn) {
    s2s_log_thread(("Writer-" + std::to_string(conn->id)).c_str());
    for (;;) {
        std::string frame;
        {
            std::unique_lock<std::mutex> lock(conn->out_mutex);
            conn->out_cv.wait(lock, [conn]() { return conn->out_stop || !conn->out.empty(); });
            if (conn->out_stop) {
                return;
            }
            frame = std::move(conn->out.front());
            conn->out.pop();
        }
        if (!conn->send(frame, conn->send_user)) {
            s2s_log(S2S_LOG_WARN, "[Server] Send failed, the client is gone or too slow");
            {
                std::lock_guard<std::mutex> lock(conn->out_mutex);
                conn->out_stop = true;
                std::queue<std::string>().swap(conn->out);
            }
            conn_stop(conn);
            return;
        }
    }
}

// Whether something opened after this turn and revision: a later turn, or
// the same turn resumed. The user spoke again, so an answer to it that
// nobody heard yet has lost its point. Called with turn_mutex held.
static bool conn_outdated(const Connection * conn, int turn_id, int revision) {
    return conn->open_turn > turn_id || (conn->open_turn == turn_id && conn->open_revision > revision);
}

// Whether the same turn went on past this revision: its audio comes again,
// whole, in a later commit.
static bool conn_revised(Connection * conn, int turn_id, int revision) {
    std::lock_guard<std::mutex> lock(conn->turn_mutex);
    return conn->open_turn == turn_id && conn->open_revision > revision;
}

// Tells the session that the answer to this turn and revision is ready to be
// heard, or over: a turn stays resumable until the user can hear something.
// Called with no other lock held, like every session call.
static void conn_release(Connection * conn, int turn_id, int revision) {
    std::lock_guard<std::mutex> lock(conn->session_mutex);
    s2s_session_release(conn->session, turn_id, revision);
}

// Holds the first unit of the answer until its turn is final. Returns false
// when the floor went back meanwhile, or when something opened after it.
static bool conn_wait_final(Connection * conn) {
    std::unique_lock<std::mutex> lock(conn->turn_mutex);
    conn->turn_cv.wait(lock, [conn]() {
        return conn->stop || conn->cancel.load() || conn->final_turn >= conn->answer_turn ||
               conn_outdated(conn, conn->answer_turn, conn->answer_revision);
    });
    if (!conn->cancel.load() && conn_outdated(conn, conn->answer_turn, conn->answer_revision)) {
        s2s_log(S2S_LOG_INFO, "[Turn] Answer to turn %d rev %d voided by turn %d rev %d before its first unit",
                conn->answer_turn, conn->answer_revision, conn->open_turn, conn->open_revision);
        conn->cancel.store(true);
    }
    return !conn->stop && !conn->cancel.load();
}

// Speaks one unit and streams it to the client. The transcript of the unit
// goes out right before its first audio chunk, so the client only ever sees
// text that has sound behind it, and places it in the audio stream. Returns
// false when the floor was taken back.
static bool conn_speak(Connection * conn, const SentenceUnit & unit, const tts_request & tts) {
    if (!conn->answer_released) {
        conn_release(conn, conn->answer_turn, conn->answer_revision);
        if (!conn_wait_final(conn)) {
            return false;
        }
        conn->answer_released = true;
    }

    struct SpeakTap {
        Connection *         conn    = nullptr;
        const SentenceUnit * unit    = nullptr;
        size_t               samples = 0;
    } speak_tap;

    speak_tap.conn = conn;
    speak_tap.unit = &unit;

    const bool spoke = tts_bridge_speak(
        conn->setup->models.tts, unit.text, tts,
        [](const float * pcm, size_t n_samples, void * user) {
            SpeakTap * self = (SpeakTap *) user;
            if (self->samples == 0 && n_samples > 0) {
                conn_send(self->conn, rt_event_spoken(self->unit->text, self->unit->end));
            }
            self->samples += n_samples;
            conn_send(self->conn, rt_event_audio(pcm, n_samples));
            return !self->conn->cancel.load();
        },
        &speak_tap, &conn->cancel);

    if (!spoke && !conn->cancel.load()) {
        s2s_log(S2S_LOG_WARN, "[TTS] %s", tts_bridge_last_error());
        conn_send(conn, rt_event_error(tts_bridge_last_error()));
    }

    return spoke;
}

// The endpoint client of the connection, created on its first turn and
// pointed at the settings of every turn after.
static llm_client * conn_llm(Connection * conn, const llm_client_params & params) {
    if (!conn->llm) {
        conn->llm = llm_client_new(params);
        return conn->llm;
    }
    return llm_client_set_params(conn->llm, params) ? conn->llm : nullptr;
}

// Appends a message, joining two user messages in a row into one: a turn
// whose answer was never heard is followed by the next one, and some chat
// templates refuse two user messages in a row.
static void llm_push(std::vector<llm_message> & messages, const llm_message & message) {
    if (message.role == "user" && !messages.empty() && messages.back().role == "user") {
        messages.back().content += " " + message.content;
    } else {
        messages.push_back(message);
    }
}

// Turn audio in, transcript out, and on to the responder. A turn this ends
// without a transcript is released here, like one whose answer is over.
static void conn_recognize(Connection * conn, const TurnAudio & turn) {
    const std::vector<float> & pcm = turn.pcm;

    // A later revision of the same turn carries this audio and more: only the
    // last one is worth recognizing.
    if (conn_revised(conn, turn.turn_id, turn.revision)) {
        s2s_log(S2S_LOG_INFO, "[Turn] Turn %d rev %d resumed before its recognition", turn.turn_id, turn.revision);
        conn_release(conn, turn.turn_id, turn.revision);
        return;
    }

    Timer t_asr;

    pk_transcribe_params asr_params = pk_transcribe_default_params();
    char *               text       = nullptr;
    const pk_status      status =
        pk_transcribe(conn->setup->models.asr, pcm.data(), pcm.size(), S2S_MODEL_RATE, &asr_params, &text);
    if (status != PK_STATUS_OK) {
        conn_error(conn, pk_last_error());
        conn_release(conn, turn.turn_id, turn.revision);
        return;
    }

    const std::string transcript = text;
    pk_free_text(text);

    s2s_log(S2S_LOG_INFO, "[Perf] Recognize %.1f ms (%.2fs of audio)", t_asr.ms(),
            (double) pcm.size() / S2S_MODEL_RATE);

    if (transcript.empty()) {
        s2s_log(S2S_LOG_INFO, "[Turn] Empty transcript, nothing to answer");
        conn_release(conn, turn.turn_id, turn.revision);
        return;
    }

    s2s_log(S2S_LOG_INFO, "[Turn] Heard %zu characters in %.2fs of audio (turn %d rev %d)", transcript.size(),
            (double) pcm.size() / S2S_MODEL_RATE, turn.turn_id, turn.revision);
    conn_send(conn, rt_event_transcript("turn_" + std::to_string(turn.turn_id), transcript));

    AnswerJob job;
    job.transcript = transcript;
    job.turn_id    = turn.turn_id;
    job.revision   = turn.revision;

    std::lock_guard<std::mutex> lock(conn->queue_mutex);
    conn->answers.push(std::move(job));
    conn->answers_cv.notify_one();
}

// Transcript in, answer spoken out.
static void conn_answer(Connection * conn, const AnswerJob & job) {
    const std::string & transcript = job.transcript;
    const std::string   item       = "turn_" + std::to_string(job.turn_id);

    // Whatever way the answer ends here, the responder is done with it.
    struct TurnRelease {
        Connection * conn;
        int          turn_id;
        int          revision;

        ~TurnRelease() { conn_release(conn, turn_id, revision); }
    } release = { conn, job.turn_id, job.revision };

    if (conn->stop) {
        return;
    }

    // A cancel raised from here on belongs to this answer.
    conn->cancel.store(false);

    // The settings and the list as they stand now: a change that arrives
    // later belongs to the next turn, not to this one.
    ClientSettings client;
    {
        std::lock_guard<std::mutex> lock(conn->client_mutex);
        client = conn->client;
    }

    // The answer in flight, published with the floor under the turn lock: from
    // here on a turn or a revision that opens voids it at once. If the user
    // spoke again during the recognition, it is void before it starts, and
    // still opens and closes like any other, so the client files the turn in
    // its conversation all the same.
    {
        std::lock_guard<std::mutex> lock(conn->turn_mutex);
        conn->answer_turn     = job.turn_id;
        conn->answer_revision = job.revision;
        conn->answer_released = false;
        conn->speaking.store(true);
        if (conn_outdated(conn, job.turn_id, job.revision)) {
            conn->cancel.store(true);
        }
    }
    conn_send(conn, rt_event("response.created"));

    std::string answer;

    if (conn->cancel.load()) {
        s2s_log(S2S_LOG_INFO, "[Turn] Turn %d superseded or cancelled before its answer", job.turn_id);
    } else if (client.mode == "loopback") {
        // No endpoint in the path: the recognized text is the answer.
        answer = transcript;

        // A couple of characters is not speech, it is the tail of a noise the
        // recognizer had to name: spoken back, it sends the talker off its
        // distribution. Characters, not bytes: an accented letter is one.
        int n_chars = 0;
        for (const char c : answer) {
            n_chars += ((unsigned char) c & 0xC0) != 0x80;
        }
        if (n_chars < client.tts.guards.min_chars) {
            s2s_log(S2S_LOG_INFO, "[TTS] Skipped %d characters, fewer than %d", n_chars, client.tts.guards.min_chars);
        } else {
            SentenceUnit unit;
            unit.text = answer;
            unit.end  = sentence_utf16_len(answer);
            conn_speak(conn, unit, client.tts);
        }
    } else {
        std::vector<llm_message> messages;
        if (!client.system_prompt.empty()) {
            messages.push_back({ "system", client.system_prompt });
        }
        // An earlier revision of this turn may sit in the list, pushed when
        // its answer closed unheard: the transcript below replaces it.
        for (const rt_message & message : client.history) {
            if (message.role != "user" || message.item != item) {
                llm_push(messages, { message.role, message.content });
            }
        }
        llm_push(messages, { "user", transcript });

        if (client.llm.base_url.empty()) {
            conn_error(conn, "[Realtime] No endpoint: name one in the session, or start the server with --llm-url");
        } else if (llm_client * llm = conn_llm(conn, client.llm); !llm) {
            conn_error(conn, llm_client_last_error());
        } else {
            struct StreamTap {
                Connection *        conn = nullptr;
                const tts_request * tts  = nullptr;
                SentenceSplitter    splitter;
                Timer               timer;
                double              first_unit_ms = -1.0;
            } stream_tap;

            stream_tap.conn = conn;
            stream_tap.tts  = &client.tts;

            // What the model writes, as it writes it. The transcript event
            // that follows says what is really spoken, one unit later.
            llm_delta_cb on_delta = [](const char * delta, void * user) {
                StreamTap * self = (StreamTap *) user;

                conn_send(self->conn, rt_event_text("response.output_text.delta", "delta", delta));

                for (const SentenceUnit & unit : sentence_split_push(&self->splitter, delta)) {
                    if (self->first_unit_ms < 0.0) {
                        self->first_unit_ms = self->timer.ms();
                    }
                    if (!conn_speak(self->conn, unit, *self->tts)) {
                        return false;
                    }
                }
                return !self->conn->cancel.load();
            };

            Timer      t_llm;
            const bool streamed = client.mode == "agentic" ?
                                      llm_agent_run(llm, client.llm, client.tools, client.max_rounds, messages,
                                                    on_delta, &stream_tap, &conn->cancel, answer) :
                                      llm_client_stream(llm, messages, on_delta, &stream_tap, &conn->cancel, answer);

            // The tail is a unit like the others: a one sentence answer has
            // no other, and its time is the time to the first unit.
            const SentenceUnit tail = sentence_split_flush(&stream_tap.splitter);
            if (streamed && !tail.text.empty()) {
                if (stream_tap.first_unit_ms < 0.0) {
                    stream_tap.first_unit_ms = stream_tap.timer.ms();
                }
                conn_speak(conn, tail, client.tts);
            }

            if (stream_tap.first_unit_ms < 0.0) {
                s2s_log(S2S_LOG_INFO, "[Perf] Respond %.1f ms (no unit, %zu characters)", t_llm.ms(), answer.size());
            } else {
                s2s_log(S2S_LOG_INFO, "[Perf] Respond %.1f ms (first unit %.1f ms, %zu characters)", t_llm.ms(),
                        stream_tap.first_unit_ms, answer.size());
            }

            if (!streamed && !conn->cancel.load()) {
                conn_error(conn, llm_client_last_error());
            }
        }
    }

    s2s_log(S2S_LOG_INFO, "[Turn] Answered %zu characters%s", answer.size(), conn->cancel.load() ? ", cut short" : "");

    // Every response.created ends here, with exactly one of the two.
    conn->speaking.store(false);
    conn_send(conn, rt_event(conn->cancel.load() ? "response.cancelled" : "response.done"));
}

static void conn_recognizer(Connection * conn) {
    s2s_log_thread(("Recognizer-" + std::to_string(conn->id)).c_str());
    for (;;) {
        TurnAudio turn;
        {
            std::unique_lock<std::mutex> lock(conn->queue_mutex);
            conn->queue_cv.wait(lock, [conn]() { return conn->stop || !conn->queue.empty(); });
            if (conn->stop) {
                return;
            }
            turn = std::move(conn->queue.front());
            conn->queue.pop();
        }
        conn_recognize(conn, turn);
    }
}

static void conn_responder(Connection * conn) {
    s2s_log_thread(("Responder-" + std::to_string(conn->id)).c_str());
    for (;;) {
        AnswerJob job;
        {
            std::unique_lock<std::mutex> lock(conn->queue_mutex);
            conn->answers_cv.wait(lock, [conn]() { return conn->stop || !conn->answers.empty(); });
            if (conn->stop) {
                return;
            }
            job = std::move(conn->answers.front());
            conn->answers.pop();
        }
        conn_answer(conn, job);
    }
}

// The latest turn and revision that opened. An answer in flight to anything
// older is void from this moment: its synthesis and its endpoint request
// stop now instead of at their first unit, and the responder is free for
// what just opened.
static void conn_opened(Connection * conn, int turn_id, int revision) {
    std::lock_guard<std::mutex> lock(conn->turn_mutex);
    if (turn_id > conn->open_turn || (turn_id == conn->open_turn && revision > conn->open_revision)) {
        conn->open_turn     = turn_id;
        conn->open_revision = revision;
    }
    if (conn->speaking.load() && !conn->cancel.load() &&
        conn_outdated(conn, conn->answer_turn, conn->answer_revision)) {
        s2s_log(S2S_LOG_INFO,
                "[Turn] Answer to turn %d rev %d voided by turn %d rev %d, its synthesis and endpoint request stop",
                conn->answer_turn, conn->answer_revision, turn_id, revision);
        conn->cancel.store(true);
    }
    conn->turn_cv.notify_all();
}

// Events the state machine raises, on the reader thread.
static void conn_on_session_event(const s2s_session_report * report, void * user) {
    Connection * conn = (Connection *) user;

    switch (report->event) {
        case S2S_EVENT_SPEECH_STARTED:
            {
                s2s_log(S2S_LOG_INFO, "[Session] Speech started at %.2fs (turn %d rev %d)", report->time_sec,
                        report->turn_id, report->revision);
                conn_send(conn, rt_event("input_audio_buffer.speech_started"));
                conn_opened(conn, report->turn_id, report->revision);
                break;
            }

        case S2S_EVENT_TURN_RESUMED:
            {
                // The client sees the user talking again, like any start; the
                // answer to the previous revision is void.
                s2s_log(S2S_LOG_INFO, "[Session] Turn resumed at %.2fs (turn %d rev %d), its audio goes on",
                        report->time_sec, report->turn_id, report->revision);
                conn_send(conn, rt_event("input_audio_buffer.speech_started"));
                conn_opened(conn, report->turn_id, report->revision);
                break;
            }

        case S2S_EVENT_TURN_FINAL:
            {
                s2s_log(S2S_LOG_INFO, "[Session] Turn final at %.2fs (turn %d rev %d), its answer may be heard",
                        report->time_sec, report->turn_id, report->revision);
                std::lock_guard<std::mutex> lock(conn->turn_mutex);
                conn->final_turn = report->turn_id > conn->final_turn ? report->turn_id : conn->final_turn;
                conn->turn_cv.notify_all();
                break;
            }

        case S2S_EVENT_SPEECH_STOPPED:
            s2s_log(S2S_LOG_INFO, "[Session] Speech stopped at %.2fs (turn %d rev %d)", report->time_sec,
                    report->turn_id, report->revision);
            conn_send(conn, rt_event("input_audio_buffer.speech_stopped"));
            break;

        case S2S_EVENT_TURN_REOPENED:
            s2s_log(S2S_LOG_INFO, "[Session] Turn reopened at %.2fs (turn %d rev %d), completion %.3f",
                    report->time_sec, report->turn_id, report->revision, (double) report->turn_score);
            break;

        case S2S_EVENT_BARGE_IN:
            // The floor goes back to the user: the synthesis and the endpoint
            // request stop, the client flushes its playback on the
            // speech_started that follows, and the responder closes the
            // response.
            s2s_log(S2S_LOG_INFO, "[Session] Barge-in at %.2fs, the user took the floor", report->time_sec);
            conn->cancel.store(true);
            conn_turn_notify(conn);
            break;

        case S2S_EVENT_TURN_COMMITTED:
            {
                s2s_log(
                    S2S_LOG_INFO,
                    "[Session] Turn committed at %.2fs (turn %d rev %d), %.2fs of audio kept of %.2fs, completion %.3f",
                    report->time_sec, report->turn_id, report->revision, (double) report->n_samples / S2S_MODEL_RATE,
                    (double) report->n_held / S2S_MODEL_RATE, (double) report->turn_score);

                TurnAudio turn;
                turn.pcm.assign(report->pcm, report->pcm + report->n_samples);
                turn.turn_id  = report->turn_id;
                turn.revision = report->revision;

                std::lock_guard<std::mutex> lock(conn->queue_mutex);
                conn->queue.push(std::move(turn));
                conn->queue_cv.notify_one();
                break;
            }
    }
}

// Every session.update describes the whole session: a field it leaves out
// takes the server default, so clearing a field on the page brings the
// default back. The conversation has its own event and stays, and the echo
// canceller, a state with a learned path, only changes when a method is named.
static void conn_apply_patch(Connection * conn, const rt_session_patch & patch) {
    std::lock_guard<std::mutex> lock(conn->client_mutex);

    std::vector<rt_message> history = std::move(conn->client.history);
    conn->client                    = conn->setup->defaults;
    conn->client.history            = std::move(history);

    const s2s_session_params listening = conn->params;
    conn->params                       = s2s_session_params();

    if (!patch.mode.empty()) {
        conn->client.mode = patch.mode;
    }
    conn->client.tools = patch.tools;
    if (patch.max_rounds > 0) {
        conn->client.max_rounds = patch.max_rounds;
    }
    if (patch.tool_timeout_sec > 0) {
        conn->client.llm.tool_timeout_sec = patch.tool_timeout_sec;
    }
    if (!patch.echo.empty() && patch.echo != conn->echo) {
        // A fresh canceller learns the echo path of the new setup from
        // nothing; the previous one would start from a stale path.
        conn->echo = patch.echo;
        if (conn->aec) {
            s2s_log(S2S_LOG_INFO, "[AEC] Canceller off");
        }
        lv_state_free(conn->aec);
        conn->aec = conn->echo == "server" ? lv_state_new(conn->setup->models.aec) : nullptr;
        if (conn->aec) {
            s2s_log(S2S_LOG_INFO, "[AEC] Canceller on, fresh echo path");
        }
        // The microphone and the reference restart their resampling together:
        // two streams of equal length in give two of equal length out, hop
        // for hop.
        conn->aec_mic.clear();
        conn->aec_ref.clear();
        audio_resample_stream_init(&conn->mic_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
        audio_resample_stream_init(&conn->ref_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
        conn->aec_in         = 0.0;
        conn->aec_out        = 0.0;
        conn->aec_played     = 0;
        conn->aec_quiet      = 0;
        conn->aec_hops       = 0;
        conn->aec_ms         = 0.0;
        conn->aec_peak       = 0.0;
        conn->aec_pending_in = 0.0;
        conn->aec_pending    = false;
        conn->aec_failing    = false;
    }
    if (conn->setup->llm_fixed) {
        if (!patch.llm_url.empty() || !patch.llm_model.empty() || !patch.llm_key.empty()) {
            conn_error(conn, "[Realtime] The endpoint is set by the server");
        }
    } else {
        // A key travels with its URL: the URL of a patch takes the key of the
        // same patch, none when it carries none, so a key never reaches a
        // host it was not given for.
        if (!patch.llm_url.empty()) {
            if (host_allowed(conn->setup->llm_hosts, patch.llm_url)) {
                conn->client.llm.base_url = patch.llm_url;
                conn->client.llm.api_key  = patch.llm_key;
            } else {
                conn_error(conn, ("[Realtime] Endpoint host " + url_host(patch.llm_url) + " is not allowed").c_str());
            }
        }
        if (!patch.llm_model.empty()) {
            conn->client.llm.model = patch.llm_model;
        }
    }
    if (!patch.system_prompt.empty()) {
        conn->client.system_prompt = patch.system_prompt;
    }
    if (patch.temperature >= 0.0f) {
        conn->client.llm.sampling.temperature = patch.temperature;
    }
    if (patch.top_p >= 0.0f) {
        conn->client.llm.sampling.top_p = patch.top_p;
    }
    if (patch.top_k >= 0) {
        conn->client.llm.sampling.top_k = patch.top_k;
    }
    if (patch.min_p >= 0.0f) {
        conn->client.llm.sampling.min_p = patch.min_p;
    }
    if (patch.max_tokens >= 0) {
        conn->client.llm.sampling.max_tokens = patch.max_tokens;
    }
    if (patch.presence_penalty > -3.0f) {
        conn->client.llm.sampling.presence_penalty = patch.presence_penalty;
    }
    if (patch.frequency_penalty > -3.0f) {
        conn->client.llm.sampling.frequency_penalty = patch.frequency_penalty;
    }
    if (patch.seed >= 0) {
        conn->client.llm.sampling.seed = (int) patch.seed;
    }
    if (!patch.reasoning_effort.empty()) {
        conn->client.llm.sampling.reasoning_effort = patch.reasoning_effort;
    }
    if (!patch.tts_voice.empty()) {
        conn->client.tts.voice = patch.tts_voice;
    }
    if (!patch.tts_language.empty()) {
        conn->client.tts.language = patch.tts_language;
    }
    if (!patch.tts_browser_languages.empty()) {
        conn->client.tts.browser_languages = patch.tts_browser_languages;
    }
    if (patch.tts_min_chars >= 0) {
        conn->client.tts.guards.min_chars = patch.tts_min_chars;
    }
    if (patch.tts_chars_per_second > 0.0f) {
        conn->client.tts.guards.chars_per_second = patch.tts_chars_per_second;
    }
    if (patch.tts_margin_seconds >= 0.0f) {
        conn->client.tts.guards.margin_seconds = patch.tts_margin_seconds;
    }

    if (patch.llm_timeout_sec > 0) {
        conn->client.llm.timeout_sec = patch.llm_timeout_sec;
    }

    if (patch.tts_temperature >= 0.0f) {
        conn->client.tts.sampling.temperature = patch.tts_temperature;
    }
    if (patch.tts_top_k >= 0) {
        conn->client.tts.sampling.top_k = patch.tts_top_k;
    }
    if (patch.tts_top_p >= 0.0f) {
        conn->client.tts.sampling.top_p = patch.tts_top_p;
    }
    if (patch.tts_repetition_penalty >= 0.0f) {
        conn->client.tts.sampling.repetition_penalty = patch.tts_repetition_penalty;
    }
    if (patch.tts_subtalker_temperature >= 0.0f) {
        conn->client.tts.sampling.subtalker_temperature = patch.tts_subtalker_temperature;
    }
    if (patch.tts_subtalker_top_k >= 0) {
        conn->client.tts.sampling.subtalker_top_k = patch.tts_subtalker_top_k;
    }
    if (patch.tts_subtalker_top_p >= 0.0f) {
        conn->client.tts.sampling.subtalker_top_p = patch.tts_subtalker_top_p;
    }
    if (patch.tts_max_new_tokens >= 0) {
        conn->client.tts.sampling.max_new_tokens = patch.tts_max_new_tokens;
    }
    if (patch.tts_seed >= 0) {
        conn->client.tts.sampling.seed = patch.tts_seed;
    }

    if (patch.vad_neg_threshold >= 0.0f) {
        conn->params.vad_neg_threshold = patch.vad_neg_threshold;
    }
    if (patch.vad_threshold >= 0.0f) {
        conn->params.vad_threshold = patch.vad_threshold;
    }
    if (patch.min_speech_ms >= 0) {
        conn->params.min_speech_ms = patch.min_speech_ms;
    }
    if (patch.barge_in_ms >= 0) {
        conn->params.barge_in_ms = patch.barge_in_ms;
    }
    if (patch.min_silence_ms >= 0) {
        conn->params.min_silence_ms = patch.min_silence_ms;
    }
    if (patch.min_speech_continuation_ms >= 0) {
        conn->params.min_speech_continuation_ms = patch.min_speech_continuation_ms;
    }
    if (patch.speech_pad_ms >= 0) {
        conn->params.speech_pad_ms = patch.speech_pad_ms;
    }
    if (patch.turn_threshold >= 0.0f) {
        conn->params.turn_threshold = patch.turn_threshold;
    }
    if (patch.turn_max_wait_ms >= 0) {
        conn->params.turn_max_wait_ms = patch.turn_max_wait_ms;
    }
    if (patch.reopen_grace_ms >= 0) {
        conn->params.reopen_grace_ms = patch.reopen_grace_ms;
    }

    // Only a change of the listening thresholds reaches the session, which
    // keeps the turn it is in the middle of hearing.
    if (memcmp(&listening, &conn->params, sizeof(listening)) != 0) {
        std::lock_guard<std::mutex> session_lock(conn->session_mutex);
        s2s_session_set_params(conn->session, conn->params);
    }
}

Connection * conn_open(const ConversationSetup * setup, int id, conn_send_fn send, void * send_user) {
    Connection * conn = new Connection();
    conn->setup       = setup;
    conn->send        = send;
    conn->send_user   = send_user;
    conn->id          = id;
    audio_resample_stream_init(&conn->mic_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
    audio_resample_stream_init(&conn->ref_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
    conn->client  = setup->defaults;
    conn->session = s2s_session_new(setup->models.vad, setup->models.turn, conn->params, conn_on_session_event, conn);
    if (!conn->session) {
        s2s_log(S2S_LOG_WARN, "%s", sv_last_error());
        send(rt_event_error(sv_last_error()), send_user);
        delete conn;
        return nullptr;
    }

    conn->writer     = std::thread(conn_writer, conn);
    conn->recognizer = std::thread(conn_recognizer, conn);
    conn->responder  = std::thread(conn_responder, conn);
    conn_send(conn, rt_event("session.created"));
    return conn;
}

void conn_frame(Connection * conn, const std::string & frame) {
    const rt_client_message message = rt_parse(frame);
    switch (message.type) {
        case RT_CLIENT_SESSION_UPDATE:
            conn_apply_patch(conn, message.patch);
            if (!message.patch.invalid.empty()) {
                conn_error(conn, ("[Realtime] Session update: " + message.patch.invalid +
                                  " is not a number in its range, it keeps the server default")
                                     .c_str());
            }
            // The log is streamed to every page on /logs: it says
            // whether an endpoint is set, never which one.
            s2s_log(S2S_LOG_INFO, "[Realtime] Session update: mode %s, echo %s, endpoint %s, voice %s, language %s",
                    conn->client.mode.c_str(), conn->echo.c_str(), conn->client.llm.base_url.empty() ? "none" : "set",
                    conn->client.tts.voice.empty() ? "default" : conn->client.tts.voice.c_str(),
                    conn->client.tts.language.empty() ? "default" : conn->client.tts.language.c_str());
            conn_send(conn, rt_event("session.updated"));
            break;

        case RT_CLIENT_AUDIO_APPEND:
            if (!conn->receiving) {
                conn->receiving = true;
                s2s_log(S2S_LOG_INFO, "[Realtime] Microphone streaming, %zu samples per frame", message.audio.size());
            }
            audio_resample_stream_push(&conn->mic_resample, message.audio.data(), message.audio.size(),
                                       conn->resampled);
            if (conn->aec) {
                // a frame without reference is a frame where nothing played
                const bool played = message.reference.size() == message.audio.size();
                if (!played) {
                    conn->silence.assign(message.audio.size(), 0.0f);
                }
                const std::vector<float> & played_pcm = played ? message.reference : conn->silence;
                audio_resample_stream_push(&conn->ref_resample, played_pcm.data(), played_pcm.size(), conn->reference);
                conn_cancel_echo(conn, conn->resampled, conn->reference, played);
            }
            {
                std::lock_guard<std::mutex> lock(conn->session_mutex);
                s2s_session_set_speaking(conn->session, conn->speaking.load());
                s2s_session_push(conn->session, conn->resampled.data(), conn->resampled.size());
            }
            break;

        case RT_CLIENT_AUDIO_COMMIT:
            s2s_log(S2S_LOG_INFO, "[Realtime] Audio buffer commit");
            {
                std::lock_guard<std::mutex> lock(conn->session_mutex);
                s2s_session_commit_now(conn->session);
            }
            break;

        case RT_CLIENT_RESPONSE_CANCEL:
            s2s_log(S2S_LOG_INFO, "[Realtime] Response cancel");
            conn->cancel.store(true);
            conn_turn_notify(conn);
            break;

        case RT_CLIENT_HISTORY:
            {
                std::lock_guard<std::mutex> lock(conn->client_mutex);
                conn->client.history = message.messages;
                s2s_log(S2S_LOG_INFO, "[Realtime] History: %zu messages", conn->client.history.size());
            }
            break;

        case RT_CLIENT_UNKNOWN:
            s2s_log(S2S_LOG_WARN, "[Realtime] Unsupported event");
            conn_send(conn, rt_event_error("unsupported event"));
            break;
    }
}

bool conn_stopped(const Connection * conn) {
    return conn->stop;
}

void conn_close(Connection * conn) {
    conn_stop(conn);
    conn->recognizer.join();
    conn->responder.join();
    llm_client_free(conn->llm);
    {
        std::lock_guard<std::mutex> lock(conn->out_mutex);
        conn->out_stop = true;
        conn->out_cv.notify_one();
    }
    conn->writer.join();
    s2s_session_free(conn->session);
    lv_state_free(conn->aec);
    s2s_log(S2S_LOG_INFO, "[Server] Connection closed");
    delete conn;
}
