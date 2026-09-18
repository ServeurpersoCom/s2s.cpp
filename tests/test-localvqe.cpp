// test-localvqe.cpp: echo canceller output for the parity harness
//
// Reads a microphone and a far end reference as raw float32 at 16 kHz, runs
// them hop by hop through the streaming canceller exactly like the server
// does, and writes the cleaned signal as raw float32. The output is one hop
// late, which test-localvqe.py accounts for. The device follows GGML_BACKEND.

#include "localvqe.h"
#include "timer.h"
#include "version.h"

#include <cstdio>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <gguf> <mic.f32> <ref.f32> <out.f32>\n"
            "\n"
            "Cleans the microphone of the reference echo, hop by hop, and writes\n"
            "the result as raw float32, one hop late.\n",
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
    if (argc != 5) {
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
    lv_state * state = lv_state_new(ctx);

    const size_t       hop    = (size_t) lv_hop(ctx);
    const size_t       n_hops = mic.size() / hop;
    std::vector<float> out(n_hops * hop, 0.0f);

    Timer timer;
    for (size_t i = 0; i < n_hops; i++) {
        if (lv_process(state, mic.data() + i * hop, ref.data() + i * hop, out.data() + i * hop) != 0) {
            fprintf(stderr, "[LocalVQE] FATAL: %s\n", lv_last_error());
            return 1;
        }
    }
    const double ms = timer.ms();
    fprintf(stderr, "[LocalVQE] %zu hops in %.1f ms, %.3f ms per hop, %.1fx real time\n", n_hops, ms,
            ms / (double) n_hops, (double) (n_hops * hop) / lv_sample_rate(ctx) * 1000.0 / ms);

    FILE * fp = fopen(argv[4], "wb");
    if (!fp) {
        fprintf(stderr, "[LocalVQE] FATAL: cannot write %s\n", argv[4]);
        return 1;
    }
    fwrite(out.data(), sizeof(float), out.size(), fp);
    fclose(fp);

    lv_state_free(state);
    lv_free(ctx);
    return 0;
}
