#pragma once
// sp-detok.h: SentencePiece detokenization
//
// The transducer only ever emits ids, so the runtime needs the piece table
// and nothing else: no merges, no scores, no encoder. Decoding concatenates
// the pieces, turns the U+2581 word marker back into a space, and drops the
// leading space the Metaspace scheme always prepends.

#include "gguf-weights.h"

#include <string>
#include <vector>

#define SP_SPACE_MARKER "\xe2\x96\x81"  // U+2581 LOWER ONE EIGHTH BLOCK

struct SpDetok {
    std::vector<std::string> pieces;
};

static void sp_detok_load(SpDetok * sp, const GGUFModel & gf, const char * key) {
    const int64_t index = gguf_find_key(gf.gguf, key);
    if (index < 0) {
        s2s_throw("[SP] Key '%s' not found", key);
    }
    const int64_t n = gguf_get_arr_n(gf.gguf, index);
    sp->pieces.resize((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        sp->pieces[(size_t) i] = gguf_get_arr_str(gf.gguf, index, i);
    }
}

// Appends one piece to a transcript being built.
static void sp_detok_append(const SpDetok & sp, int id, std::string & text) {
    if (id < 0 || (size_t) id >= sp.pieces.size()) {
        s2s_throw("[SP] Id %d outside the %zu piece table", id, sp.pieces.size());
    }
    const std::string & piece = sp.pieces[(size_t) id];

    const size_t marker = sizeof(SP_SPACE_MARKER) - 1;
    for (size_t i = 0; i < piece.size();) {
        if (piece.compare(i, marker, SP_SPACE_MARKER) == 0) {
            if (!text.empty()) {
                text += ' ';
            }
            i += marker;
            continue;
        }
        text += piece[i];
        i++;
    }
}
