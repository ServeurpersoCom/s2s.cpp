#pragma once
// log-capture.h: stderr captured into a ring and streamed on /logs
//
// Every line the process writes to stderr reaches the terminal as before and
// a ring of the latest lines, which GET /logs streams as server sent events:
// a browser with no sound and no microphone still shows every stage the
// server went through. A crash names itself in the stream before the
// process goes.

#include "httplib.h"
#include "s2s-error.h"

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

// Portable fd wrappers, functions rather than macros: a write() macro would
// eat a method of the same name, sink.write() of httplib for instance.
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
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

// The ring of the latest lines, which /logs replays to a new client before it
// streams the lines that follow.
#define LOG_RING_BITS 9
#define LOG_RING_SIZE (1 << LOG_RING_BITS)
#define LOG_RING_MASK (LOG_RING_SIZE - 1)

static std::mutex              mtx_log;
static std::condition_variable cv_log;
static std::string             log_ring[LOG_RING_SIZE];
static uint64_t                log_seq = 0;

static int               g_real_stderr_fd = -1;
static int               g_pipe_read_fd   = -1;
static std::thread       g_log_reader;
static std::atomic<bool> g_log_drained{ false };  // the reader forwarded everything and returned

// How long a crash waits for the reader to drain the pipe before it goes on.
#define LOG_CRASH_DRAIN_MS 1000

// Drains the pipe into the real stderr and the ring, until the write end of
// the pipe closes when the capture stops.
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
    g_log_drained.store(true);
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
// the process goes. A crash on the reader itself writes straight out. The
// wait is bounded, never a join: the crashed thread may hold a lock the
// reader needs, the log ring or the heap, and a crash must end the process,
// not hang it.
static void log_capture_crash(const char * what) {
    char      line[512];
    const int n = snprintf(line, sizeof(line), "%s\n", s2s_log_named(std::string("[Server] FATAL: ") + what).c_str());
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
    for (int ms = 0; ms < LOG_CRASH_DRAIN_MS && !g_log_drained.load(); ms++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    // exit() skips the destructors of the stack, this capture's included: the
    // hook still drains the pipe, so the last message reaches the terminal.
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

// GET /logs: the ring as server sent events, then every new line as it comes.
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
