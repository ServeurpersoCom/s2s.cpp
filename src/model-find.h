#pragma once
// model-find.h: the GGUF files the server loads, found by name in a directory

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

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
