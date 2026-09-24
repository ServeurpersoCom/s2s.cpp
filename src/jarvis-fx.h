#pragma once
// jarvis-fx.h: the Jarvis voice effect, streaming
//
// The synthetic voice of the assistant in Iron Man (2008), measured on the VF
// 5.1 master: the dry voice sits alone on the center channel and the effect
// lives in the four surrounds. Folded to mono, the effect is three copies of
// the voice added to it: a faint bright leak, an echo 50.39 ms late, and two
// chorus voices read through delays swept around 34 ms, which give the
// phasing and the brightness. It works with any dry voice.
//
// Everything is causal and runs sample by sample, so a stream goes through
// in chunks of any size. One state per stream: the delay lines and the LFOs
// carry over from one chunk to the next.

#include <cstddef>

struct jarvis_fx;

jarvis_fx * jarvis_fx_new(int sample_rate);
void        jarvis_fx_free(jarvis_fx * fx);

// Silence in the delay lines and the LFOs back to their start: what a new
// stream needs, so nothing of the previous one echoes into it.
void jarvis_fx_reset(jarvis_fx * fx);

// Runs n samples in place: pcm comes back as the voice plus its effect.
void jarvis_fx_process(jarvis_fx * fx, float * pcm, size_t n);

// Samples the effect still rings for once the voice stops: the longest delay.
// Running that much silence through jarvis_fx_process gives the tail.
size_t jarvis_fx_tail(const jarvis_fx * fx);
