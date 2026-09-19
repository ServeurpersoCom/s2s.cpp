// s2s-server.cpp: the voice loop behind a WebSocket
//
// One process, four models, any number of conversations. The models load once
// and are shared: the VAD and the turn classifier are stateless per window and
// keep their per stream state in the session, while the recognizer and the
// voice serialize internally, so a second client queues rather than doubling
// the VRAM.
//
// Each connection runs two threads. The reader owns the socket and the
// listening half: it decodes frames, feeds the session, and answers the
// events the state machine raises. The responder owns the talking half: it
// recognizes a committed turn, asks the endpoint, and speaks the answer unit
// by unit. Splitting them is what makes the barge-in work, since the reader
// keeps consuming audio while the responder is busy.
//
// Modes:
//   conversation  recognize, ask the LLM, speak the answer
//   loopback      recognize and speak the transcript back, no endpoint in the
//                 path, which is how the microphone, the turn detection, the
//                 recognizer and the voice get tested on their own

#include "httplib.h"
#include "index.html.gz.hpp"
#include "llm-client.h"
#include "localvqe.h"
#include "parakeet.h"
#include "realtime-proto.h"
#include "s2s-error.h"
#include "s2s-session.h"
#include "s2s.js.gz.hpp"
#include "sentence-split.h"
#include "silero.h"
#include "smart-turn.h"
#include "timer.h"
#include "tts-bridge.h"
#include "version.h"

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#    include <windows.h>
#    ifndef STDERR_FILENO
#        define STDERR_FILENO 2
#    endif
#else
#    include <unistd.h>
#endif

// portable fd wrappers. avoids macros that collide with C++ method names
// (e.g. sink.write() in httplib would be eaten by a write() macro).
#ifdef _WIN32
static int fd_pipe(int fd[2]) {
    return _pipe(fd, 4096, _O_BINARY);
}

static int fd_dup(int fd) {
    return _dup(fd);
}

static int fd_dup2(int src, int dst) {
    return _dup2(src, dst);
}

static int fd_read(int fd, void * buf, size_t n) {
    return _read(fd, buf, (unsigned) n);
}

static int fd_write(int fd, const void * buf, size_t n) {
    return _write(fd, buf, (unsigned) n);
}

static void fd_close(int fd) {
    _close(fd);
}
#else
static int fd_pipe(int fd[2]) {
    return pipe(fd);
}

static int fd_dup(int fd) {
    return dup(fd);
}

static int fd_dup2(int src, int dst) {
    return dup2(src, dst);
}

static int fd_read(int fd, void * buf, size_t n) {
    return (int) read(fd, buf, n);
}

static int fd_write(int fd, const void * buf, size_t n) {
    return (int) write(fd, buf, n);
}

static void fd_close(int fd) {
    close(fd);
}
#endif

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define S2S_INPUT_RATE SAMPLE_RATE_24K  // the rate the Realtime protocol carries

// Frame ceiling of one synthesis, the value the Python reference settles on.
// The per utterance budget of the bridge sits well below it; this only bounds
// the worst case.
#define S2S_TTS_MAX_NEW_TOKENS 1536
#define S2S_MODEL_RATE         16000  // the rate the VAD and the recognizer work at

// log capture: intercept stderr via pipe, forward to terminal + ring buffer.
// SSE clients connect to /logs and receive lines in real time.
#define LOG_RING_BITS 9
#define LOG_RING_SIZE (1 << LOG_RING_BITS)
#define LOG_RING_MASK (LOG_RING_SIZE - 1)

static std::mutex              mtx_log;
static std::condition_variable cv_log;
static std::string             log_ring[LOG_RING_SIZE];
static uint64_t                log_seq = 0;

static int         g_real_stderr_fd = -1;
static int         g_pipe_read_fd   = -1;
static std::thread g_log_reader;

// reader thread: drain pipe, forward to real stderr, push lines to ring.
// exits when the write end of the pipe is closed (fd_dup2 restores real stderr).
static void log_reader_main() {
    char        buf[4096];
    std::string partial;
    for (;;) {
        int n = fd_read(g_pipe_read_fd, buf, sizeof(buf));
        if (n <= 0) {
            break;
        }
        fd_write(g_real_stderr_fd, buf, (size_t) n);
        partial.append(buf, (size_t) n);
        size_t pos;
        while ((pos = partial.find('\n')) != std::string::npos) {
            std::lock_guard<std::mutex> lock(mtx_log);
            log_ring[log_seq & LOG_RING_MASK] = partial.substr(0, pos);
            log_seq++;
            cv_log.notify_all();
            partial.erase(0, pos + 1);
        }
    }
    if (!partial.empty()) {
        std::lock_guard<std::mutex> lock(mtx_log);
        log_ring[log_seq & LOG_RING_MASK] = std::move(partial);
        log_seq++;
        cv_log.notify_all();
    }
    fd_close(g_pipe_read_fd);
}

// Restore stderr and drain the reader before the pipe dies with the process.
// Idempotent: the destructor and the exit hook both land here, either order.
static void log_capture_stop() {
    if (g_real_stderr_fd < 0) {
        return;
    }
    fflush(stderr);
    // the restore drops the last write end, so the reader reads EOF and returns
    fd_dup2(g_real_stderr_fd, STDERR_FILENO);
    cv_log.notify_all();
    if (g_log_reader.joinable()) {
        g_log_reader.join();
    }
    fd_close(g_real_stderr_fd);
    g_real_stderr_fd = -1;
}

// A crash kills the process with its last words still in the pipe. This
// names the crash through the pipe, so /logs carries it too, then drops the
// last write end so the reader drains everything to the real stderr before
// the process goes. A crash on the reader itself writes straight out.
static void log_capture_crash(const char * what) {
    char      line[512];
    const int n = snprintf(line, sizeof(line), "[Server] FATAL: %s\n", what);
    if (g_real_stderr_fd < 0) {
        fd_write(STDERR_FILENO, line, (size_t) n);
        return;
    }
    if (std::this_thread::get_id() == g_log_reader.get_id()) {
        fd_write(g_real_stderr_fd, line, (size_t) n);
        return;
    }
    fd_write(STDERR_FILENO, line, (size_t) n);
    fd_dup2(g_real_stderr_fd, STDERR_FILENO);
    if (g_log_reader.joinable()) {
        g_log_reader.join();
    }
}

// Once the log is out, the crash goes on the way it would have.
static void on_crash(int sig) {
    log_capture_crash(sig == SIGABRT ? "abort" :
                      sig == SIGSEGV ? "invalid memory access" :
                      sig == SIGFPE  ? "arithmetic fault" :
                      sig == SIGILL  ? "illegal instruction" :
                                       "crash");
    signal(sig, SIG_DFL);
    raise(sig);
}

#ifdef _WIN32
// Windows raises no signal for an access violation outside the thread that
// installed it: the process wide filter catches every thread.
static LONG WINAPI on_exception(EXCEPTION_POINTERS * info) {
    char what[64];
    snprintf(what, sizeof(what), "exception 0x%08lx at %p", (unsigned long) info->ExceptionRecord->ExceptionCode,
             info->ExceptionRecord->ExceptionAddress);
    log_capture_crash(what);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static void setup_log_capture() {
    g_real_stderr_fd = fd_dup(STDERR_FILENO);
    int pipefd[2];
    if (fd_pipe(pipefd) != 0) {
        fd_close(g_real_stderr_fd);
        g_real_stderr_fd = -1;
        return;
    }
    g_pipe_read_fd = pipefd[0];
    fd_dup2(pipefd[1], STDERR_FILENO);
    fd_close(pipefd[1]);
    // A loader aborts the process with exit() on a fatal error, which skips
    // every destructor: the hook still drains the pipe, so the message that
    // explains the failure reaches the terminal.
    atexit(log_capture_stop);
    g_log_reader = std::thread(log_reader_main);

    signal(SIGABRT, on_crash);
    signal(SIGSEGV, on_crash);
    signal(SIGFPE, on_crash);
    signal(SIGILL, on_crash);
#ifdef _WIN32
    SetUnhandledExceptionFilter(on_exception);
#endif
}

// RAII: captures stderr on construction, restores and drains on destruction.
struct LogCapture {
    LogCapture() { setup_log_capture(); }

    ~LogCapture() { log_capture_stop(); }
};

// GET /logs: SSE stream of stderr lines.
// sends backlog (up to LOG_RING_SIZE) then streams new lines in real time.
static void handle_logs(const httplib::Request &, httplib::Response & res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream", [cursor = uint64_t(0), init = false](size_t, httplib::DataSink & sink) mutable -> bool {
            std::unique_lock<std::mutex> lock(mtx_log);
            if (!init) {
                uint64_t avail = log_seq < LOG_RING_SIZE ? log_seq : (uint64_t) LOG_RING_SIZE;
                cursor         = log_seq - avail;
                while (cursor < log_seq) {
                    std::string ev = "data: " + log_ring[cursor & LOG_RING_MASK] + "\n\n";
                    cursor++;
                    lock.unlock();
                    if (!sink.write(ev.c_str(), ev.size())) {
                        return false;
                    }
                    lock.lock();
                }
                init = true;
            }
            cv_log.wait_for(lock, std::chrono::seconds(2));
            while (cursor < log_seq) {
                std::string ev = "data: " + log_ring[cursor & LOG_RING_MASK] + "\n\n";
                cursor++;
                lock.unlock();
                if (!sink.write(ev.c_str(), ev.size())) {
                    return false;
                }
                lock.lock();
            }
            return true;
        });
}

struct ServerModels {
    sv_context * vad  = nullptr;
    st_context * turn = nullptr;
    pk_context * asr  = nullptr;
    tts_bridge * tts  = nullptr;
    lv_context * aec  = nullptr;
};

static ServerModels      g_models;
static httplib::Server * g_server = nullptr;

// Host of an endpoint URL, with its port when it carries one: what an
// allowlist entry is compared against.
static std::string url_host(const std::string & url) {
    const size_t scheme = url.find("://");
    const size_t start  = scheme == std::string::npos ? 0 : scheme + 3;
    const size_t end    = url.find('/', start);
    return url.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

// An endpoint the client names is fetched by this process, so an empty
// allowlist means the server can be asked to reach anything it can route to:
// a private network, a metadata service. Naming the hosts closes that.
static bool host_allowed(const std::vector<std::string> & hosts, const std::string & url) {
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

// Origin is a browser header: it keeps a third party page out, and it is
// worth nothing against a script that sets it by hand. It is one layer, not
// the protection.
static bool origin_allowed(const std::vector<std::string> & origins, const httplib::Request & req) {
    const std::string origin = req.get_header_value("Origin");
    if (origins.empty() || origin.empty()) {
        return true;
    }
    return std::find(origins.begin(), origins.end(), origin) != origins.end();
}

static void on_signal(int) {
    if (g_server) {
        g_server->stop();
    }
}

struct Connection {
    httplib::ws::WebSocket * ws = nullptr;

    s2s_session *      session = nullptr;
    s2s_session_params params;

    std::string mode = "conversation";
    std::string echo = "native";
    tts_request tts;

    llm_client_params llm;
    std::string       system_prompt;

    // The conversation belongs to the client: it pushes its list when it
    // changes, the responder copies it when it picks a turn up, and nothing
    // survives here between two turns.
    std::mutex               history_mutex;
    std::vector<llm_message> history;

    // Committed turns waiting for the responder.
    std::mutex                     queue_mutex;
    std::condition_variable        queue_cv;
    std::queue<std::vector<float>> queue;
    bool                           stop = false;

    std::atomic<bool> cancel{ false };

    // Whether the assistant holds the floor: set by the responder, handed to
    // the session by the reader, the only thread that touches the session.
    std::atomic<bool> speaking{ false };

    // Input resampling state: the protocol carries 24 kHz, the models want
    // 16 kHz, and the ratio is 3 to 2, so the decimation carries a remainder
    // across frames. The reference follows the microphone sample for sample.
    std::vector<float> input_tail;
    std::vector<float> reference_tail;

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
};

// Frames without playback that close a run of it, 0.5 s of client frames: a
// shorter gap is a pause between two sentences of the same answer.
#define AEC_QUIET_FRAMES 25

static void conn_send(Connection * conn, const std::string & frame) {
    if (conn->ws && conn->ws->is_open()) {
        conn->ws->send(frame);
    }
}

// A failure goes to the client and to the log alike: the client sees what
// went wrong, and the log keeps it. The messages carry the tag of the module
// that failed.
static void conn_error(Connection * conn, const char * message) {
    s2s_log(S2S_LOG_WARN, "%s", message);
    conn_send(conn, rt_event_error(message));
}

// 24 kHz to 16 kHz: three samples in, two out, with a short averaging window
// instead of a full polyphase bank. The VAD and the recognizer both tolerate
// it, and it costs nothing on the reader thread.
static void resample_in(std::vector<float> & tail, const std::vector<float> & input, std::vector<float> & out) {
    std::vector<float> buffer = tail;
    buffer.insert(buffer.end(), input.begin(), input.end());

    const size_t groups = buffer.size() / 3;
    out.resize(groups * 2);
    for (size_t i = 0; i < groups; i++) {
        const float a  = buffer[3 * i];
        const float b  = buffer[3 * i + 1];
        const float c  = buffer[3 * i + 2];
        out[2 * i]     = a * 0.5f + b * 0.5f;
        out[2 * i + 1] = b * 0.25f + c * 0.75f;
    }

    tail.assign(buffer.begin() + (ptrdiff_t) (groups * 3), buffer.end());
}

// Runs the canceller over every complete hop and hands back in pcm what is
// clean so far. The canceller is one hop late and its hops of 256 samples do
// not line up with the client frames, so the cleaned stream trails the raw
// one by one to two hops, about 16 to 32 ms.
static void conn_cancel_echo(Connection * conn, std::vector<float> & pcm, const std::vector<float> & ref, bool played) {
    conn->aec_mic.insert(conn->aec_mic.end(), pcm.begin(), pcm.end());
    conn->aec_ref.insert(conn->aec_ref.end(), ref.begin(), ref.end());

    const size_t hop  = (size_t) lv_hop(g_models.aec);
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

// Speaks one unit and streams it to the client. The transcript of the unit
// goes out right before its first audio chunk, so the client only ever sees
// text that has sound behind it, and places it in the audio stream. Returns
// false when the floor was taken back.
static bool conn_speak(Connection * conn, const std::string & unit) {
    struct SpeakTap {
        Connection *        conn    = nullptr;
        const std::string * unit    = nullptr;
        size_t              samples = 0;
    } speak_tap;

    speak_tap.conn = conn;
    speak_tap.unit = &unit;

    const bool spoke = tts_bridge_speak(
        g_models.tts, unit, conn->tts,
        [](const float * pcm, size_t n_samples, void * user) {
            SpeakTap * self = (SpeakTap *) user;
            if (self->samples == 0 && n_samples > 0) {
                conn_send(self->conn, rt_event_text("response.output_audio_transcript.delta", "delta", *self->unit));
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

// Turn audio in, answer spoken out.
static void conn_respond(Connection * conn, const std::vector<float> & pcm) {
    Timer t_asr;

    pk_transcribe_params asr_params = pk_transcribe_default_params();
    char *               text       = nullptr;
    const pk_status status = pk_transcribe(g_models.asr, pcm.data(), pcm.size(), S2S_MODEL_RATE, &asr_params, &text);
    if (status != PK_STATUS_OK) {
        conn_error(conn, pk_last_error());
        return;
    }

    const std::string transcript = text;
    pk_free_text(text);

    s2s_log(S2S_LOG_INFO, "[Perf] Recognize %.1f ms (%.2fs of audio)", t_asr.ms(),
            (double) pcm.size() / S2S_MODEL_RATE);

    if (transcript.empty()) {
        s2s_log(S2S_LOG_INFO, "[Turn] Empty transcript, nothing to answer");
        return;
    }

    s2s_log(S2S_LOG_INFO, "[Turn] Heard %zu characters in %.2fs of audio", transcript.size(),
            (double) pcm.size() / S2S_MODEL_RATE);
    conn_send(conn, rt_event_text("conversation.item.input_audio_transcription.completed", "transcript", transcript));

    conn->cancel.store(false);
    conn->speaking.store(true);
    conn_send(conn, rt_event("response.created"));

    std::string answer;

    if (conn->mode == "loopback") {
        // No endpoint in the path: the recognized text is the answer.
        answer = transcript;
        conn_speak(conn, answer);
    } else {
        // The list as it stood when this turn was picked up: a push that
        // arrives later belongs to the next turn, not to this one.
        std::vector<llm_message> messages;
        if (!conn->system_prompt.empty()) {
            messages.push_back({ "system", conn->system_prompt });
        }
        {
            std::lock_guard<std::mutex> lock(conn->history_mutex);
            messages.insert(messages.end(), conn->history.begin(), conn->history.end());
        }
        messages.push_back({ "user", transcript });

        llm_client * client = llm_client_new(conn->llm);
        if (!client) {
            conn_error(conn, llm_client_last_error());
        } else {
            struct StreamTap {
                Connection *     conn = nullptr;
                SentenceSplitter splitter;
                Timer            timer;
                double           first_unit_ms = -1.0;
            } stream_tap;

            stream_tap.conn = conn;

            Timer      t_llm;
            const bool streamed = llm_client_stream(
                client, messages,
                [](const char * delta, void * user) {
                    StreamTap * self = (StreamTap *) user;

                    // What the model writes, as it writes it. The transcript event
                    // that follows says what is really spoken, one unit later.
                    conn_send(self->conn, rt_event_text("response.output_text.delta", "delta", delta));

                    for (const std::string & unit : sentence_split_push(&self->splitter, delta)) {
                        if (self->first_unit_ms < 0.0) {
                            self->first_unit_ms = self->timer.ms();
                        }
                        if (!conn_speak(self->conn, unit)) {
                            return false;
                        }
                    }
                    return !self->conn->cancel.load();
                },
                &stream_tap, &conn->cancel, answer);

            const std::string tail = sentence_split_flush(&stream_tap.splitter);
            if (streamed && !tail.empty()) {
                conn_speak(conn, tail);
            }

            s2s_log(S2S_LOG_INFO, "[Perf] Respond %.1f ms (first unit %.1f ms, %zu characters)", t_llm.ms(),
                    stream_tap.first_unit_ms, answer.size());

            if (!streamed && !conn->cancel.load()) {
                conn_error(conn, llm_client_last_error());
            }
            llm_client_free(client);
        }
    }

    s2s_log(S2S_LOG_INFO, "[Turn] Answered %zu characters%s", answer.size(), conn->cancel.load() ? ", cut short" : "");

    // Every response.created ends here, with exactly one of the two.
    conn->speaking.store(false);
    conn_send(conn, rt_event(conn->cancel.load() ? "response.cancelled" : "response.done"));
}

static void conn_responder(Connection * conn) {
    for (;;) {
        std::vector<float> pcm;
        {
            std::unique_lock<std::mutex> lock(conn->queue_mutex);
            conn->queue_cv.wait(lock, [conn]() { return conn->stop || !conn->queue.empty(); });
            if (conn->stop && conn->queue.empty()) {
                return;
            }
            pcm = std::move(conn->queue.front());
            conn->queue.pop();
        }
        conn_respond(conn, pcm);
    }
}

// Events the state machine raises, on the reader thread.
static void conn_on_session_event(const s2s_session_report * report, void * user) {
    Connection * conn = (Connection *) user;

    switch (report->event) {
        case S2S_EVENT_SPEECH_STARTED:
            s2s_log(S2S_LOG_INFO, "[Session] Speech started at %.2fs (turn %d rev %d)", report->time_sec,
                    report->turn_id, report->revision);
            conn_send(conn, rt_event("input_audio_buffer.speech_started"));
            break;

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
            break;

        case S2S_EVENT_TURN_COMMITTED:
            {
                s2s_log(S2S_LOG_INFO, "[Session] Turn committed at %.2fs (turn %d), %.2fs of audio, completion %.3f",
                        report->time_sec, report->turn_id, (double) report->n_samples / S2S_MODEL_RATE,
                        (double) report->turn_score);

                std::lock_guard<std::mutex> lock(conn->queue_mutex);
                conn->queue.emplace(report->pcm, report->pcm + report->n_samples);
                conn->queue_cv.notify_one();
                break;
            }
    }
}

static std::vector<std::string> g_llm_hosts;

static void conn_apply_patch(Connection * conn, const rt_session_patch & patch) {
    if (!patch.mode.empty()) {
        conn->mode = patch.mode;
    }
    if (!patch.echo.empty() && patch.echo != conn->echo) {
        // A fresh canceller learns the echo path of the new setup from
        // nothing; the previous one would start from a stale path.
        conn->echo = patch.echo;
        if (conn->aec) {
            s2s_log(S2S_LOG_INFO, "[AEC] Canceller off");
        }
        lv_state_free(conn->aec);
        conn->aec = conn->echo == "server" ? lv_state_new(g_models.aec) : nullptr;
        if (conn->aec) {
            s2s_log(S2S_LOG_INFO, "[AEC] Canceller on, fresh echo path");
        }
        conn->aec_mic.clear();
        conn->aec_ref.clear();
        conn->reference_tail.clear();
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
    if (!patch.llm_url.empty()) {
        if (host_allowed(g_llm_hosts, patch.llm_url)) {
            conn->llm.base_url = patch.llm_url;
        } else {
            s2s_log(S2S_LOG_WARN, "[Realtime] Endpoint host %s is not allowed", url_host(patch.llm_url).c_str());
            conn_send(conn, rt_event_error("endpoint host not allowed"));
        }
    }
    if (!patch.llm_model.empty()) {
        conn->llm.model = patch.llm_model;
    }
    if (!patch.llm_key.empty()) {
        conn->llm.api_key = patch.llm_key;
    }
    if (!patch.system_prompt.empty()) {
        conn->system_prompt = patch.system_prompt;
    }
    if (patch.temperature >= 0.0f) {
        conn->llm.sampling.temperature = patch.temperature;
    }
    if (patch.top_p >= 0.0f) {
        conn->llm.sampling.top_p = patch.top_p;
    }
    if (patch.top_k >= 0.0f) {
        conn->llm.sampling.top_k = (int) patch.top_k;
    }
    if (patch.min_p >= 0.0f) {
        conn->llm.sampling.min_p = patch.min_p;
    }
    if (patch.max_tokens >= 0.0f) {
        conn->llm.sampling.max_tokens = (int) patch.max_tokens;
    }
    if (patch.presence_penalty > -3.0f) {
        conn->llm.sampling.presence_penalty = patch.presence_penalty;
    }
    if (patch.frequency_penalty > -3.0f) {
        conn->llm.sampling.frequency_penalty = patch.frequency_penalty;
    }
    if (patch.seed >= 0.0f) {
        conn->llm.sampling.seed = (int) patch.seed;
    }
    if (!patch.tts_speaker.empty()) {
        conn->tts.speaker = patch.tts_speaker;
    }
    if (!patch.tts_language.empty()) {
        conn->tts.language = patch.tts_language;
    }
    if (patch.tts_min_chars >= 0.0f) {
        conn->tts.guards.min_chars = (int) patch.tts_min_chars;
    }
    if (patch.tts_chars_per_second > 0.0f) {
        conn->tts.guards.chars_per_second = patch.tts_chars_per_second;
    }
    if (patch.tts_margin_seconds >= 0.0f) {
        conn->tts.guards.margin_seconds = patch.tts_margin_seconds;
    }

    if (patch.llm_timeout_sec > 0.0f) {
        conn->llm.timeout_sec = (int) patch.llm_timeout_sec;
    }

    if (patch.tts_temperature >= 0.0f) {
        conn->tts.sampling.temperature = patch.tts_temperature;
    }
    if (patch.tts_top_k >= 0.0f) {
        conn->tts.sampling.top_k = (int) patch.tts_top_k;
    }
    if (patch.tts_top_p >= 0.0f) {
        conn->tts.sampling.top_p = patch.tts_top_p;
    }
    if (patch.tts_repetition_penalty >= 0.0f) {
        conn->tts.sampling.repetition_penalty = patch.tts_repetition_penalty;
    }
    if (patch.tts_subtalker_temperature >= 0.0f) {
        conn->tts.sampling.subtalker_temperature = patch.tts_subtalker_temperature;
    }
    if (patch.tts_subtalker_top_k >= 0.0f) {
        conn->tts.sampling.subtalker_top_k = (int) patch.tts_subtalker_top_k;
    }
    if (patch.tts_subtalker_top_p >= 0.0f) {
        conn->tts.sampling.subtalker_top_p = patch.tts_subtalker_top_p;
    }
    if (patch.tts_max_new_tokens >= 0.0f) {
        conn->tts.sampling.max_new_tokens = (int) patch.tts_max_new_tokens;
    }
    if (patch.tts_seed >= 0.0f) {
        conn->tts.sampling.seed = (int64_t) patch.tts_seed;
    }

    // Only the listening thresholds need the session rebuilt: everything else
    // is read at the next turn, so a voice or a sampling change must not drop
    // what the microphone is in the middle of hearing.
    bool listening_changed = false;

    if (patch.vad_threshold >= 0.0f) {
        conn->params.vad_threshold = patch.vad_threshold;
        listening_changed          = true;
    }
    if (patch.min_speech_ms >= 0) {
        conn->params.min_speech_ms = patch.min_speech_ms;
        listening_changed          = true;
    }
    if (patch.min_speech_continuation_ms >= 0) {
        conn->params.min_speech_continuation_ms = patch.min_speech_continuation_ms;
        listening_changed                       = true;
    }
    if (patch.min_silence_ms >= 0) {
        conn->params.min_silence_ms = patch.min_silence_ms;
        listening_changed           = true;
    }
    if (patch.speech_pad_ms >= 0) {
        conn->params.speech_pad_ms = patch.speech_pad_ms;
        listening_changed          = true;
    }
    if (patch.turn_threshold >= 0.0f) {
        conn->params.turn_threshold = patch.turn_threshold;
        listening_changed           = true;
    }
    if (patch.turn_max_wait_ms >= 0) {
        conn->params.turn_max_wait_ms = patch.turn_max_wait_ms;
        listening_changed             = true;
    }

    if (listening_changed) {
        s2s_session_free(conn->session);
        conn->session = s2s_session_new(g_models.vad, g_models.turn, conn->params, conn_on_session_event, conn);
        s2s_session_set_speaking(conn->session, conn->speaking.load());
    }
}

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Models:\n"
            "  --models <dir>         Directory holding the GGUF files (default: ./models)\n"
            "\n"
            "Server:\n"
            "  --host <addr>          Bind address (default: 127.0.0.1)\n"
            "  --port <N>             Bind port (default: 8088)\n"
            "\n"
            "Security:\n"
            "  --origin <url>         Allowed browser origin, repeatable. Rejects a WebSocket\n"
            "                         or an HTTP route called from another page. Empty allows\n"
            "                         every origin. A script can forge this header, so it only\n"
            "                         keeps third party pages out.\n"
            "  --llm-host <host>      Allowed endpoint host, repeatable, host[:port]. The server\n"
            "                         fetches the endpoint a client names, so an empty list lets\n"
            "                         it reach anything it can route to. Naming the hosts closes\n"
            "                         that door.\n"
            "\n"
            "Engine:\n"
            "  --max-batch <N>        Concurrent syntheses batched on the GPU (default: 1)\n"
            "  --no-fa                Disable flash attention in the TTS\n"
            "  --clamp-fp16           Clamp hidden states to the FP16 range in the TTS\n"
            "  --codec-chunk-dur <s>  Codec decode chunk, bounds the peak decode memory\n"
            "\n"
            "Everything else belongs to the client: mode, endpoint, prompt, voice,\n"
            "sampling and turn detection travel in session.update, and their\n"
            "defaults are published on /props.\n",
            prog);
}

// Parameter count in billions read from a "-1.7b-" tag, 0 when the name has none.
static float model_size(const std::string & name) {
    for (size_t i = 0; i < name.size(); i++) {
        if (name[i] != '-') {
            continue;
        }
        const char * start = name.c_str() + i + 1;
        char *       end   = nullptr;
        const float  size  = strtof(start, &end);
        if (end != start && end[0] == 'b' && end[1] == '-') {
            return size;
        }
    }
    return 0.0f;
}

// Quant preference, best first and clamped at Q8_0: a wider file costs memory
// for no audible gain, so F32 and BF16 rank last.
static int quant_rank(const std::string & name) {
    static const char * quants[] = { "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M" };
    const int           n_quants = (int) (sizeof(quants) / sizeof(quants[0]));
    for (int i = 0; i < n_quants; i++) {
        if (name.find(quants[i]) != std::string::npos) {
            return i;
        }
    }
    return n_quants;
}

// Picks among the GGUF files starting with the prefix and holding the variant:
// the largest model first, then the best quant, then the name for a stable
// choice. The pick is logged at load.
static std::string find_model(const std::string & dir, const char * prefix, const char * variant) {
    std::string best;
    std::string best_name;

    std::error_code error;
    for (const auto & entry : std::filesystem::directory_iterator(dir, error)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0 || name.find(variant) == std::string::npos || name.size() <= 5 ||
            name.compare(name.size() - 5, 5, ".gguf") != 0) {
            continue;
        }
        if (!best.empty()) {
            const float size      = model_size(name);
            const float best_size = model_size(best_name);
            if (size < best_size) {
                continue;
            }
            if (size == best_size) {
                const int rank      = quant_rank(name);
                const int best_rank = quant_rank(best_name);
                if (rank > best_rank || (rank == best_rank && name > best_name)) {
                    continue;
                }
            }
        }
        best      = entry.path().string();
        best_name = name;
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string models_dir = "models";
    std::string host       = "127.0.0.1";
    int         port       = 8088;

    std::vector<std::string> origins;
    std::vector<std::string> llm_hosts;

    tts_engine engine;

    // Session defaults, published on /props and overridable per client.
    const llm_client_params llm_defaults;
    const std::string       system_prompt = "You are a voice assistant. Answer in one or two short spoken sentences.";
    const std::string       voice;
    const std::string       language = "auto";
    const std::string       mode     = "loopback";

    for (int i = 1; i < argc; i++) {
        const std::string arg       = argv[i];
        const bool        has_value = i + 1 < argc;

        if (arg == "--models" && has_value) {
            models_dir = argv[++i];
        } else if (arg == "--host" && has_value) {
            host = argv[++i];
        } else if (arg == "--port" && has_value) {
            port = atoi(argv[++i]);
        } else if (arg == "--origin" && has_value) {
            origins.push_back(argv[++i]);
        } else if (arg == "--llm-host" && has_value) {
            llm_hosts.push_back(argv[++i]);
        } else if (arg == "--max-batch" && has_value) {
            engine.max_batch = atoi(argv[++i]);
        } else if (arg == "--no-fa") {
            engine.use_fa = false;
        } else if (arg == "--clamp-fp16") {
            engine.clamp_fp16 = true;
        } else if (arg == "--codec-chunk-dur" && has_value) {
            engine.codec_chunk_sec = (float) atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::string vad_path    = find_model(models_dir, "silero-vad", "");
    const std::string turn_path   = find_model(models_dir, "smart-turn", "");
    const std::string asr_path    = find_model(models_dir, "parakeet", "");
    const std::string talker_path = find_model(models_dir, "qwen-talker", "-customvoice-");
    const std::string codec_path  = find_model(models_dir, "qwen-tokenizer", "");
    const std::string aec_path    = find_model(models_dir, "localvqe", "");

    if (vad_path.empty() || turn_path.empty() || asr_path.empty() || talker_path.empty() || codec_path.empty() ||
        aec_path.empty()) {
        fprintf(stderr, "[Server] FATAL: missing models in %s, run ./models.sh\n", models_dir.c_str());
        return 1;
    }

    s2s_log(S2S_LOG_INFO, "[Load] VAD %s", vad_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] Turn %s", turn_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] ASR %s", asr_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] TTS %s + %s", talker_path.c_str(), codec_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] AEC %s", aec_path.c_str());

    g_models.vad = sv_init(vad_path.c_str(), 1);
    if (!g_models.vad) {
        fprintf(stderr, "[Server] FATAL: %s\n", sv_last_error());
        return 1;
    }

    g_models.turn = st_init(turn_path.c_str(), 0);
    if (!g_models.turn) {
        fprintf(stderr, "[Server] FATAL: %s\n", st_last_error());
        return 1;
    }

    pk_init_params asr_init = pk_init_default_params();
    asr_init.model_path     = asr_path.c_str();

    g_models.asr = pk_init(&asr_init);
    if (!g_models.asr) {
        fprintf(stderr, "[Server] FATAL: %s\n", pk_last_error());
        return 1;
    }

    tts_bridge_params tts_init;
    tts_init.talker_path             = talker_path;
    tts_init.codec_path              = codec_path;
    tts_init.speaker                 = voice;
    tts_init.language                = language;
    tts_init.sampling.max_new_tokens = S2S_TTS_MAX_NEW_TOKENS;
    tts_init.engine                  = engine;

    g_models.tts = tts_bridge_load(tts_init);
    if (!g_models.tts) {
        fprintf(stderr, "[Server] FATAL: %s\n", tts_bridge_last_error());
        return 1;
    }

    g_models.aec = lv_init(aec_path.c_str(), 1, 0);
    if (!g_models.aec) {
        fprintf(stderr, "[Server] FATAL: %s\n", lv_last_error());
        return 1;
    }

    // stderr is mirrored into a ring buffer and streamed to the UI, so a
    // browser with no sound and no microphone still shows every stage the
    // server went through.
    LogCapture log_capture;

    g_llm_hosts = llm_hosts;

    httplib::Server server;
    g_server = &server;

    // SO_REUSEADDR lets us rebind a port still in TIME_WAIT after a restart.
    // SO_REUSEPORT is deliberately not set: a second instance on the same port
    // then fails with EADDRINUSE instead of silently sharing the socket and
    // splitting traffic between two daemons.
    server.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    server.Get("/logs", [&](const httplib::Request & req, httplib::Response & res) {
        // The stream carries everything the server did, so it is not public.
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }
        handle_logs(req, res);
    });

    server.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Single source of truth for the session defaults: the UI leaves a field
    // empty to mean "whatever the server was started with", and shows this
    // value as the placeholder. The engine setup is not here: it belongs to
    // the command line and to the startup log, like in the sibling projects. Sampling is absent on purpose: an empty sampling field
    // leaves the endpoint to its own defaults, which is the only sane answer
    // when the endpoint can be llama-server, Ollama or a cloud API.
    server.Get("/props", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }
        const s2s_session_params turn;

        std::string body = "{";
        body += "\"version\":\"" + rt_escape(S2S_VERSION) + "\",";
        body += "\"sample_rate\":" + std::to_string(S2S_INPUT_RATE) + ",";
        // Which files are actually loaded: the directory can hold several
        // quants of the same model and nobody should have to guess.
        // File names only: the page shows what is loaded, not where the
        // server keeps it, and the separator is the OS's business.
        const auto file = [](const std::string & path) {
            return rt_escape(std::filesystem::path(path).filename().string());
        };
        body += "\"models\":{";
        body += "\"vad\":\"" + file(vad_path) + "\",";
        body += "\"turn\":\"" + file(turn_path) + "\",";
        body += "\"asr\":\"" + file(asr_path) + "\",";
        body += "\"talker\":\"" + file(talker_path) + "\",";
        body += "\"codec\":\"" + file(codec_path) + "\",";
        body += "\"aec\":\"" + file(aec_path) + "\"";
        body += "},";
        body += "\"defaults\":{";
        body += "\"mode\":\"" + rt_escape(mode) + "\",";
        body += "\"llm_url\":\"" + rt_escape(llm_defaults.base_url) + "\",";
        body += "\"llm_model\":\"" + rt_escape(llm_defaults.model) + "\",";
        body += "\"instructions\":\"" + rt_escape(system_prompt) + "\",";
        body += "\"voice\":\"" + rt_escape(tts_bridge_speaker(g_models.tts)) + "\",";
        body += "\"language\":\"" + rt_escape(language) + "\",";
        const tts_sampling & tts = tts_bridge_defaults(g_models.tts);

        body += "\"tts_speakers\":[";
        const std::vector<std::string> & speakers = tts_bridge_speakers(g_models.tts);
        for (size_t i = 0; i < speakers.size(); i++) {
            body += std::string(i ? "," : "") + "\"" + rt_escape(speakers[i]) + "\"";
        }
        body += "],";

        body += "\"tts_languages\":[";
        const std::vector<std::string> & languages = tts_bridge_languages(g_models.tts);
        for (size_t i = 0; i < languages.size(); i++) {
            body += std::string(i ? "," : "") + "\"" + rt_escape(languages[i]) + "\"";
        }
        body += "],";

        const tts_guards & guards = tts_bridge_defaults_request(g_models.tts).guards;
        body += "\"tts_min_chars\":" + std::to_string(guards.min_chars) + ",";
        body += "\"tts_chars_per_second\":" + std::to_string(guards.chars_per_second) + ",";
        body += "\"tts_margin_seconds\":" + std::to_string(guards.margin_seconds) + ",";
        body += "\"llm_timeout_sec\":" + std::to_string(llm_defaults.timeout_sec) + ",";
        body += "\"tts_temperature\":" + std::to_string(tts.temperature) + ",";
        body += "\"tts_top_k\":" + std::to_string(tts.top_k) + ",";
        body += "\"tts_top_p\":" + std::to_string(tts.top_p) + ",";
        body += "\"tts_repetition_penalty\":" + std::to_string(tts.repetition_penalty) + ",";
        body += "\"tts_subtalker_temperature\":" + std::to_string(tts.subtalker_temperature) + ",";
        body += "\"tts_subtalker_top_k\":" + std::to_string(tts.subtalker_top_k) + ",";
        body += "\"tts_subtalker_top_p\":" + std::to_string(tts.subtalker_top_p) + ",";
        // The ceiling in force, not the submodule maximum: this server caps
        // it lower, and the placeholder must say what actually applies.
        body += "\"tts_max_new_tokens\":" + std::to_string(S2S_TTS_MAX_NEW_TOKENS) + ",";
        body += "\"vad_threshold\":" + std::to_string(turn.vad_threshold) + ",";
        body += "\"min_speech_ms\":" + std::to_string(turn.min_speech_ms) + ",";
        body += "\"min_speech_continuation_ms\":" + std::to_string(turn.min_speech_continuation_ms) + ",";
        body += "\"min_silence_ms\":" + std::to_string(turn.min_silence_ms) + ",";
        body += "\"speech_pad_ms\":" + std::to_string(turn.speech_pad_ms) + ",";
        body += "\"turn_threshold\":" + std::to_string(turn.turn_threshold) + ",";
        body += "\"turn_max_wait_ms\":" + std::to_string(turn.turn_max_wait_ms);
        body += "}}";
        res.set_content(body, "application/json");
    });

    // Embedded webui: gzipped single page app built by tools/webui.
    // The browser decompresses it through Content-Encoding.
    if (index_html_gz_len > 0) {
        server.Get("/", [](const httplib::Request & req, httplib::Response & res) {
            if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                res.set_content("Error: gzip is not supported by this browser", "text/plain");
            } else {
                res.set_header("Content-Encoding", "gzip");
                res.set_content(reinterpret_cast<const char *>(index_html_gz), index_html_gz_len,
                                "text/html; charset=utf-8");
            }
        });
    }

    // Browser log: the page posts what happens on its side, so the microphone
    // permission, the socket and the audio context show up in the same stream
    // as the server stages instead of dying in a console nobody reads.
    server.Post("/log", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }
        std::string line = req.body.substr(0, 512);
        for (char & c : line) {
            if (c == '\n' || c == '\r') {
                c = ' ';
            }
        }
        s2s_log(S2S_LOG_INFO, "[Browser] %s", line.c_str());
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Model list proxy: the browser asks s2s-server, s2s-server asks the
    // endpoint. No CORS to negotiate, no endpoint reachable from the client
    // side, and the API key stays on this machine. The body overrides the
    // defaults so the panel can probe a URL before the session opens.
    server.Post("/v1/models", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }

        llm_client_params params = llm_defaults;

        yyjson_doc * doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (doc) {
            yyjson_val *      root = yyjson_doc_get_root(doc);
            const std::string url  = rt_json_str(root, "url");
            const std::string key  = rt_json_str(root, "key");
            if (!url.empty()) {
                params.base_url = url;
            }
            if (!key.empty()) {
                params.api_key = key;
            }
            yyjson_doc_free(doc);
        }

        if (!host_allowed(llm_hosts, params.base_url)) {
            s2s_log(S2S_LOG_WARN, "[HTTP] Endpoint host %s is not allowed", url_host(params.base_url).c_str());
            res.status = 403;
            res.set_content("{\"error\":\"endpoint host not allowed\"}", "application/json");
            return;
        }

        s2s_log(S2S_LOG_INFO, "[HTTP] Model list from %s", params.base_url.c_str());

        std::vector<std::string> models;
        if (!llm_client_models(params, models)) {
            s2s_log(S2S_LOG_WARN, "[LLM] Model list failed: %s", llm_client_last_error());
            res.status = 502;
            res.set_content(std::string("{\"error\":\"") + rt_escape(llm_client_last_error()) + "\"}",
                            "application/json");
            return;
        }
        s2s_log(S2S_LOG_INFO, "[LLM] %zu models", models.size());

        std::string body = "{\"data\":[";
        for (size_t i = 0; i < models.size(); i++) {
            body += std::string(i ? "," : "") + "{\"id\":\"" + rt_escape(models[i]) + "\"}";
        }
        body += "]}";
        res.set_content(body, "application/json");
    });

    // s2s.js: the same client, standalone, for a page hosted elsewhere.
    if (s2s_js_gz_len > 0) {
        server.Get("/s2s.js", [](const httplib::Request & req, httplib::Response & res) {
            if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                res.status = 406;
                res.set_content("gzip required", "text/plain");
                return;
            }
            res.set_header("Content-Encoding", "gzip");
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_content(reinterpret_cast<const char *>(s2s_js_gz), s2s_js_gz_len, "text/javascript; charset=utf-8");
        });
    }

    // A plain GET on the realtime path means the upgrade never happened: a
    // reverse proxy that does not tunnel WebSocket answers this instead of
    // handing the socket over. Saying so beats a silent 404.
    server.Get("/v1/realtime", [](const httplib::Request & req, httplib::Response & res) {
        s2s_log(S2S_LOG_WARN, "[HTTP] GET /v1/realtime without an upgrade, Connection: '%s', Upgrade: '%s'",
                req.get_header_value("Connection").c_str(), req.get_header_value("Upgrade").c_str());
        res.status = 426;
        res.set_content("{\"error\":\"websocket upgrade required\"}", "application/json");
    });

    server.WebSocket("/v1/realtime", [&](const httplib::Request & req, httplib::ws::WebSocket & ws) {
        // WebSocket handshakes bypass CORS, so the origin is checked here.
        const std::string origin = req.get_header_value("Origin");
        if (!origins.empty() && !origin.empty() && std::find(origins.begin(), origins.end(), origin) == origins.end()) {
            s2s_log(S2S_LOG_WARN, "[Server] Rejected origin %s", origin.c_str());
            ws.close(httplib::ws::CloseStatus::PolicyViolation, "origin not allowed");
            return;
        }

        s2s_log(S2S_LOG_INFO, "[Server] Connection from %s", origin.empty() ? "unknown origin" : origin.c_str());

        Connection conn;
        conn.ws            = &ws;
        conn.mode          = mode;
        conn.tts           = tts_bridge_defaults_request(g_models.tts);
        conn.llm           = llm_defaults;
        conn.system_prompt = system_prompt;
        conn.session       = s2s_session_new(g_models.vad, g_models.turn, conn.params, conn_on_session_event, &conn);
        if (!conn.session) {
            s2s_log(S2S_LOG_WARN, "%s", sv_last_error());
            ws.send(rt_event_error(sv_last_error()));
            return;
        }

        std::thread responder(conn_responder, &conn);
        conn_send(&conn, rt_event("session.created"));

        std::string        frame;
        std::vector<float> resampled;
        std::vector<float> reference;
        std::vector<float> silence;
        bool               receiving = false;

        for (;;) {
            const httplib::ws::ReadResult result = ws.read(frame);
            if (result == httplib::ws::ReadResult::Fail) {
                break;
            }
            if (result != httplib::ws::ReadResult::Text) {
                continue;
            }

            const rt_client_message message = rt_parse(frame);
            switch (message.type) {
                case RT_CLIENT_SESSION_UPDATE:
                    conn_apply_patch(&conn, message.patch);
                    s2s_log(S2S_LOG_INFO,
                            "[Realtime] Session update: mode %s, echo %s, endpoint %s, model %s, voice %s, language %s",
                            conn.mode.c_str(), conn.echo.c_str(), conn.llm.base_url.c_str(), conn.llm.model.c_str(),
                            conn.tts.speaker.empty() ? "default" : conn.tts.speaker.c_str(),
                            conn.tts.language.empty() ? "default" : conn.tts.language.c_str());
                    conn_send(&conn, rt_event("session.updated"));
                    break;

                case RT_CLIENT_AUDIO_APPEND:
                    if (!receiving) {
                        receiving = true;
                        s2s_log(S2S_LOG_INFO, "[Realtime] Microphone streaming, %zu samples per frame",
                                message.audio.size());
                    }
                    resample_in(conn.input_tail, message.audio, resampled);
                    if (conn.aec) {
                        // a frame without reference is a frame where nothing played
                        const bool played = message.reference.size() == message.audio.size();
                        if (!played) {
                            silence.assign(message.audio.size(), 0.0f);
                        }
                        resample_in(conn.reference_tail, played ? message.reference : silence, reference);
                        conn_cancel_echo(&conn, resampled, reference, played);
                    }
                    s2s_session_set_speaking(conn.session, conn.speaking.load());
                    s2s_session_push(conn.session, resampled.data(), resampled.size());
                    break;

                case RT_CLIENT_AUDIO_COMMIT:
                    s2s_log(S2S_LOG_INFO, "[Realtime] Audio buffer commit");
                    s2s_session_commit_now(conn.session);
                    break;

                case RT_CLIENT_RESPONSE_CANCEL:
                    s2s_log(S2S_LOG_INFO, "[Realtime] Response cancel");
                    conn.cancel.store(true);
                    break;

                case RT_CLIENT_HISTORY:
                    {
                        std::lock_guard<std::mutex> lock(conn.history_mutex);
                        conn.history.clear();
                        for (const rt_message & item : message.messages) {
                            conn.history.push_back({ item.role, item.content });
                        }
                        s2s_log(S2S_LOG_INFO, "[Realtime] History: %zu messages", conn.history.size());
                    }
                    break;

                case RT_CLIENT_UNKNOWN:
                    s2s_log(S2S_LOG_WARN, "[Realtime] Unsupported event");
                    conn_send(&conn, rt_event_error("unsupported event"));
                    break;
            }
        }

        conn.cancel.store(true);
        {
            std::lock_guard<std::mutex> lock(conn.queue_mutex);
            conn.stop = true;
            conn.queue_cv.notify_one();
        }
        responder.join();
        s2s_session_free(conn.session);
        lv_state_free(conn.aec);
        s2s_log(S2S_LOG_INFO, "[Server] Connection closed");
    });

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    fprintf(stderr, "[Server] s2s-server %s\n", S2S_VERSION);
    fprintf(stderr, "[Server] Mode: %s, endpoint %s\n", mode.c_str(), llm_defaults.base_url.c_str());
    fprintf(stderr, "[Server] Listening on http://%s:%d\n", host.c_str(), port);

    if (!server.listen(host, port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", host.c_str(), port);
        return 1;
    }

    lv_free(g_models.aec);
    tts_bridge_free(g_models.tts);
    pk_free(g_models.asr);
    st_free(g_models.turn);
    sv_free(g_models.vad);
    return 0;
}
