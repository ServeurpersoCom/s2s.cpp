// test-localvqe.cpp: echo canceller output for the parity harness
//
// Reads a microphone and a far end reference as raw float32 at 16 kHz, runs
// them hop by hop through the streaming canceller exactly like the server
// does, and writes the cleaned signal as raw float32. The output is one hop
// late, which test-localvqe.py accounts for. The device follows GGML_BACKEND.
//
// With a stream count, the stream runs among others fed with other audio,
// each on its own thread like the connections of the server: their hops meet
// in the same passes of the batch. Halfway through they all leave and twice
// as many join, so its slot also lives through a growth of the batch. Its
// output must not change.

#include "localvqe.h"
#include "timer.h"
#include "version.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <gguf> <mic.f32> <ref.f32> <out.f32> [streams]\n"
            "\n"
            "Cleans the microphone of the reference echo, hop by hop, and writes\n"
            "the result as raw float32, one hop late. streams runs it among\n"
            "streams - 1 others that share the batch.\n",
            prog);
}

static std::vector<float> read_f32(const char * path) {
    std::vector<float> data;
    FILE *             fp = fopen(path, "rb");
    if (!fp) {
        return data;
    }
    fseek(fp, 0, SEEK_END);
    data.resize((size_t) ftell(fp) / sizeof(float));
    fseek(fp, 0, SEEK_SET);
    if (fread(data.data(), sizeof(float), data.size(), fp) != data.size()) {
        data.clear();
    }
    fclose(fp);
    return data;
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 6) {
        print_usage(argv[0]);
        return 1;
    }
    const int n_streams = argc == 6 ? atoi(argv[5]) : 1;
    if (n_streams < 1) {
        print_usage(argv[0]);
        return 1;
    }

    const std::vector<float> mic = read_f32(argv[2]);
    const std::vector<float> ref = read_f32(argv[3]);
    if (mic.empty() || mic.size() != ref.size()) {
        fprintf(stderr, "[LocalVQE] FATAL: %s and %s must hold the same number of samples\n", argv[2], argv[3]);
        return 1;
    }

    lv_context * ctx = lv_init(argv[1], 1, 0);
    if (!ctx) {
        fprintf(stderr, "[LocalVQE] FATAL: %s\n", lv_last_error());
        return 1;
    }
    // The others hear the same call played backwards: other audio, as loud.
    const std::vector<float> mic_other(mic.rbegin(), mic.rend());
    const std::vector<float> ref_other(ref.rbegin(), ref.rend());

    std::vector<lv_state *> others;
    for (int k = 1; k < n_streams; k++) {
        others.push_back(lv_state_new(ctx));
    }
    lv_state * state = lv_state_new(ctx);

    const size_t       hop    = (size_t) lv_hop(ctx);
    const size_t       n_hops = mic.size() / hop;
    std::vector<float> out(n_hops * hop, 0.0f);

    // Runs hops [from, to) of every stream, each on its own thread.
    std::atomic<bool> failed(false);
    const auto        run = [&](size_t from, size_t to) {
        std::vector<std::thread> threads;
        for (lv_state * other : others) {
            threads.emplace_back([&, other]() {
                std::vector<float> scratch(hop);
                for (size_t i = from; i < to && !failed; i++) {
                    if (lv_process(other, mic_other.data() + i * hop, ref_other.data() + i * hop, scratch.data()) !=
                        0) {
                        failed = true;
                    }
                }
            });
        }
        for (size_t i = from; i < to && !failed; i++) {
            if (lv_process(state, mic.data() + i * hop, ref.data() + i * hop, out.data() + i * hop) != 0) {
                failed = true;
            }
        }
        for (std::thread & thread : threads) {
            thread.join();
        }
    };

    Timer timer;
    run(0, n_hops / 2);
    if (n_streams > 1) {
        for (lv_state * other : others) {
            lv_state_free(other);
        }
        others.clear();
        for (int k = 0; k < 2 * n_streams; k++) {
            others.push_back(lv_state_new(ctx));
        }
    }
    run(n_hops / 2, n_hops);
    if (failed) {
        fprintf(stderr, "[LocalVQE] FATAL: %s\n", lv_last_error());
        return 1;
    }
    const double ms = timer.ms();
    fprintf(stderr, "[LocalVQE] %zu hops of %d streams in %.1f ms, %.3f ms per hop, %.1fx real time\n", n_hops,
            n_streams, ms, ms / (double) n_hops, (double) (n_hops * hop) / lv_sample_rate(ctx) * 1000.0 / ms);

    FILE * fp = fopen(argv[4], "wb");
    if (!fp) {
        fprintf(stderr, "[LocalVQE] FATAL: cannot write %s\n", argv[4]);
        return 1;
    }
    fwrite(out.data(), sizeof(float), out.size(), fp);
    fclose(fp);

    for (lv_state * other : others) {
        lv_state_free(other);
    }
    lv_state_free(state);
    lv_free(ctx);
    return 0;
}
