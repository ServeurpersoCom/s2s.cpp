#pragma once
// backend.h: GGML backend initialization
//
// A model on the device takes the backend of the device its caller names,
// or the best one; a model that never leaves the host takes the CPU backend on the
// threads it asks for. Each call returns a fresh backend with its own device
// context and memory pool, so two models never share allocator state and can
// run concurrently.

#include "ggml-backend.h"
#include "s2s-error.h"

#include <cstring>
#include <mutex>
#include <string>
#include <thread>

// Physical core count heuristic (logical / 2 for HT/SMT): GEMM shares the
// SIMD units across hyperthreads, so one thread per physical core is optimal.
static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
    }
    return cpu;
}

// ggml may hand a line over in pieces: the pieces gather until the line
// ends, and each whole line goes out once through s2s_log, tagged, with a
// run of identical lines collapsed into a count (the CUDA graph capture
// repeats itself).
static void s2s_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    static std::mutex  mutex;
    static std::string pending;
    static std::string last;
    static int         count = 0;

    std::lock_guard<std::mutex> lock(mutex);
    pending += text;
    size_t end;
    while ((end = pending.find('\n')) != std::string::npos) {
        const std::string line = pending.substr(0, end);
        pending.erase(0, end + 1);
        if (count > 0 && line == last) {
            count++;
            continue;
        }
        if (count > 1) {
            s2s_log(S2S_LOG_INFO, "[Dedup] Previous line repeated %d times total", count);
        }
        s2s_log(level == GGML_LOG_LEVEL_ERROR ? S2S_LOG_ERROR :
                level == GGML_LOG_LEVEL_WARN  ? S2S_LOG_WARN :
                                                S2S_LOG_INFO,
                "[GGML] %s", line.c_str());
        last  = line;
        count = 1;
    }
}

// The log callback and the dynamic backend loading. Magic static: they
// happen exactly once, safe under concurrent init calls.
static void backend_load(void) {
    static const bool loaded = [] {
        ggml_log_set(s2s_ggml_log, nullptr);
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;
}

// True when name matches a device (CUDA0, Vulkan0, MTL0, CPU...), the check
// of a --device argument. Otherwise logs the available devices on one line,
// so the log reads it whole, and returns false.
static bool backend_check_device(const char * name) {
    backend_load();
    if (ggml_backend_dev_by_name(name)) {
        return true;
    }
    std::string msg = "--device ";
    msg += name;
    msg += " not found. Available:";
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        msg += ' ';
        msg += ggml_backend_dev_name(ggml_backend_dev_get(i));
    }
    s2s_log(S2S_LOG_ERROR, "%s", msg.c_str());
    return false;
}

// The backend of the device named by device, or the best one when device is
// NULL, on one thread per physical core when it is the CPU. label opens the
// log line. NULL when no backend can be initialized.
static ggml_backend_t backend_init(const char * label, const char * device) {
    backend_load();

    ggml_backend_t backend = device ? ggml_backend_init_by_name(device, nullptr) : ggml_backend_init_best();
    if (!backend) {
        if (device) {
            s2s_log(S2S_LOG_ERROR, "[Load] Device %s not available", device);
        } else {
            s2s_log(S2S_LOG_ERROR, "[Load] No backend available");
        }
        return nullptr;
    }
    const int n_threads = backend_cpu_n_threads();
    if (strcmp(ggml_backend_name(backend), "CPU") == 0) {
        ggml_backend_free(backend);
        backend = cpu_backend_new(n_threads);
        if (!backend) {
            s2s_log(S2S_LOG_ERROR, "[Load] Failed to init the CPU backend");
            return nullptr;
        }
        s2s_log(S2S_LOG_INFO, "[Load] %s backend: CPU (threads: %d)", label, n_threads);
        return backend;
    }
    s2s_log(S2S_LOG_INFO, "[Load] %s backend: %s", label, ggml_backend_name(backend));
    return backend;
}

// The CPU backend on n_threads threads, for the models that never leave the
// host: the VAD window loop and the turn classifier. NULL on failure.
static ggml_backend_t backend_init_cpu(const char * label, int n_threads) {
    // In DL builds the CPU device only appears in the registry once the
    // backends are loaded, so a context that never asks for a GPU still has to
    // go through the loader.
    backend_load();

    ggml_backend_t backend = cpu_backend_new(n_threads);
    if (!backend) {
        s2s_log(S2S_LOG_ERROR, "[Load] Failed to init the CPU backend");
        return nullptr;
    }
    s2s_log(S2S_LOG_INFO, "[Load] %s backend: CPU (threads: %d)", label, n_threads);
    return backend;
}
