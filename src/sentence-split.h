#pragma once
// sentence-split.h: turns an LLM text stream into synthesis units
//
// The sentence is the unit, and nothing else. A unit leaves as soon as a
// terminator is confirmed by the character after it, which keeps decimals and
// abbreviations whole, or as soon as a line break arrives. What the stream
// ends with leaves on the flush.
//
// There is no length cap on purpose: cutting a sentence on a character count
// lands mid syntagm and the voice pauses where no reader would. If a model
// ever writes without punctuation, that deserves a parameter on the surface,
// not a number buried here.
//
// Markdown emphasis, list markers and emoji are dropped on the way out: they
// are written for the eye, and a talker given nothing to say never finds its
// end of speech. For the same reason a unit left without a single letter or
// digit is not emitted at all.

#include "emoji.h"

#include <cstdint>
#include <string>
#include <vector>

struct SentenceSplitter {
    std::string pending;
};

static bool sentence_is_terminator(char c) {
    return c == '.' || c == '!' || c == '?' || c == ':' || c == ';';
}

// Decodes the UTF-8 code point at text[i] and stores its byte length. A
// malformed sequence reads as U+FFFD, one byte long.
static uint32_t sentence_decode(const std::string & text, size_t i, size_t * len) {
    const unsigned char lead = (unsigned char) text[i];
    const size_t n = lead < 0x80 ? 1 : (lead >> 5) == 0x06 ? 2 : (lead >> 4) == 0x0E ? 3 : (lead >> 3) == 0x1E ? 4 : 0;

    *len = 1;
    if (n == 0 || i + n > text.size()) {
        return 0xFFFD;
    }
    uint32_t cp = n == 1 ? lead : n == 2 ? lead & 0x1F : n == 3 ? lead & 0x0F : lead & 0x07;
    for (size_t k = 1; k < n; k++) {
        const unsigned char next = (unsigned char) text[i + k];
        if ((next & 0xC0) != 0x80) {
            return 0xFFFD;
        }
        cp = (cp << 6) | (next & 0x3F);
    }
    *len = n;
    return cp;
}

// A code point a voice can say: an ASCII letter or digit, or a non ASCII code
// point outside the Latin-1 symbols (U+0080 to U+00BF), the punctuation and
// symbol blocks (U+2000 to U+2BFF) and the CJK punctuation (U+3000 to U+303F).
// Accented letters and every script count.
static bool sentence_is_spoken(uint32_t cp) {
    if (cp < 0x80) {
        return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    }
    return cp >= 0xC0 && !(cp >= 0x2000 && cp <= 0x2BFF) && !(cp >= 0x3000 && cp <= 0x303F);
}

// Strips what is read with the eyes and keeps the words. Returns an empty
// string when nothing is left to say.
static std::string sentence_clean(const std::string & text) {
    std::string out;
    out.reserve(text.size());

    bool spoken = false;
    for (size_t i = 0, len = 1; i < text.size(); i += len) {
        const uint32_t cp = sentence_decode(text, i, &len);
        if (cp == '*' || cp == '_' || cp == '`' || cp == '#') {
            continue;
        }
        if (cp == '\n' || cp == '\r' || cp == '\t' || emoji_is(cp)) {
            if (!out.empty() && out.back() != ' ') {
                out += ' ';
            }
            continue;
        }
        if (cp == ' ' && !out.empty() && out.back() == ' ') {
            continue;
        }
        spoken = spoken || sentence_is_spoken(cp);
        out.append(text, i, len);
    }
    if (!spoken) {
        return std::string();
    }

    while (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

// Length of the leading sentence of pending, or 0 while none is complete.
static size_t sentence_cut_point(const SentenceSplitter & s) {
    for (size_t i = 0; i < s.pending.size(); i++) {
        const char c = s.pending[i];

        if (c == '\n') {
            return i + 1;
        }

        if (sentence_is_terminator(c)) {
            // The next character confirms the terminator. At the end of the
            // buffer there is no next character yet, so the decision waits for
            // the following delta.
            if (i + 1 == s.pending.size()) {
                continue;
            }
            const char next = s.pending[i + 1];
            if (next == ' ' || next == '\n' || next == '"' || next == '\'') {
                return i + 1;
            }
        }
    }
    return 0;
}

// Feeds a stream delta and returns the units that became complete.
static std::vector<std::string> sentence_split_push(SentenceSplitter * s, const std::string & delta) {
    std::vector<std::string> units;
    s->pending += delta;

    for (;;) {
        const size_t cut = sentence_cut_point(*s);
        if (cut == 0) {
            break;
        }
        const std::string unit = sentence_clean(s->pending.substr(0, cut));
        s->pending.erase(0, cut);
        if (!unit.empty()) {
            units.push_back(unit);
        }
    }
    return units;
}

// Flushes whatever is left when the stream ends.
static std::string sentence_split_flush(SentenceSplitter * s) {
    const std::string unit = sentence_clean(s->pending);
    s->pending.clear();
    return unit;
}
