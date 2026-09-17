#pragma once
// s2s-error.h: diagnostics shared by every lib in the project
//
// Header only so the vad, turn and asr libs pull the same helpers without
// a translation unit of their own. Three entry points:
//
//   s2s_log       routes a formatted message to the installed callback, or
//                 to stderr when none is set. A wrapper (Python logging,
//                 a systemd journal, the server access log) installs its
//                 own sink once and every module follows.
//   s2s_set_error records a diagnostic on the calling thread. Storage is
//                 thread_local so concurrent sessions never race on each
//                 other's message. Passing NULL clears the slot.
//   s2s_throw     load path counterpart: the GGUF reader and the weight
//                 chain cannot return false up hundreds of call sites, so
//                 they throw and the ABI boundary converts the exception
//                 into s2s_set_error plus a negative status. Exceptions
//                 never cross an extern "C" boundary.
//
// printf semantics everywhere. Messages longer than the internal buffer
// are truncated, never split.

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>

enum s2s_log_level {
    S2S_LOG_ERROR = 0,
    S2S_LOG_WARN  = 1,
    S2S_LOG_INFO  = 2,
    S2S_LOG_DEBUG = 3,
};

typedef void (*s2s_log_cb)(enum s2s_log_level level, const char * text, void * user_data);

struct S2SLogSink {
    s2s_log_cb cb        = nullptr;
    void *     user_data = nullptr;
};

static S2SLogSink & s2s_log_sink(void) {
    static S2SLogSink sink;
    return sink;
}

static void s2s_log_set(s2s_log_cb cb, void * user_data) {
    s2s_log_sink() = { cb, user_data };
}

static std::string s2s_format_v(const char * fmt, va_list ap) {
    char    buf[1024];
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap2);
    va_end(ap2);
    if (n < 0) {
        return std::string();
    }
    return std::string(buf, (size_t) n < sizeof(buf) ? (size_t) n : sizeof(buf) - 1);
}

static void s2s_log(enum s2s_log_level level, const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

static void s2s_log(enum s2s_log_level level, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const std::string msg = s2s_format_v(fmt, ap);
    va_end(ap);

    const S2SLogSink sink = s2s_log_sink();
    if (sink.cb) {
        sink.cb(level, msg.c_str(), sink.user_data);
        return;
    }
    fprintf(stderr, "%s\n", msg.c_str());
    fflush(stderr);
}

static std::string & s2s_error_slot(void) {
    static thread_local std::string slot;
    return slot;
}

static void s2s_set_error(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

static void s2s_set_error(const char * fmt, ...) {
    if (!fmt) {
        s2s_error_slot().clear();
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    s2s_error_slot() = s2s_format_v(fmt, ap);
    va_end(ap);
}

// Returns the diagnostic recorded by the last failing call on this thread,
// or an empty string when the thread has not failed yet.
static const char * s2s_last_error(void) {
    return s2s_error_slot().c_str();
}

[[noreturn]] static void s2s_throw(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

[[noreturn]] static void s2s_throw(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const std::string msg = s2s_format_v(fmt, ap);
    va_end(ap);
    throw std::runtime_error(msg);
}
