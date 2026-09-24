// jarvis-fx.cpp: the Jarvis voice effect, streaming
//
// out = x + leak + echo + chorus, driven by the dry voice x:
//   leak    x through a low shelf and a high shelf
//   echo    x delayed by 50.39 ms through a high shelf
//   chorus  two copies of x read through sine swept delays, summed, through a
//           low shelf and a high shelf
// Every delay reads one ring buffer, the LFOs are complex rotators and the
// filters are RBJ shelves, so a sample costs a few dozen multiplies.

#include "jarvis-fx.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#define JARVIS_ECHO_MS 50.39f

struct jarvis_shelf {
    float f0;
    float gain_db;
    bool  high;
};

struct jarvis_path_params {
    float        gain_db;
    jarvis_shelf lo;
    jarvis_shelf hi;
};

// Gains and shelves relative to the dry voice, measured on the VF surrounds
// against the center channel. The echo low shelf is flat.
static const jarvis_path_params JARVIS_LEAK = {
    -13.10f,
    { 426.0f,  -1.81f, false },
    { 5417.0f, +6.97f, true  }
};
static const jarvis_path_params JARVIS_ECHO = {
    -1.60f,
    { 100.0f,   0.00f,  false },
    { 10391.0f, -3.49f, true  }
};
static const jarvis_path_params JARVIS_CHORUS = {
    -0.57f,
    { 217.0f,  -4.12f, false },
    { 3617.0f, +2.94f, true  }
};

struct jarvis_voice_params {
    float share;      // part of the chorus power
    float center_ms;  // delay around which the LFO sweeps
    float depth_ms;   // sweep amplitude
    float rate_hz;    // sweep rate
    float phase;      // LFO phase at the start of a stream, radians
};

// One voice per family of surrounds: FL with SR, FR with SL.
static const jarvis_voice_params JARVIS_VOICES[2] = {
    { 0.33f, 35.4f, 1.0f, 0.55f, 0.0f       },
    { 0.67f, 33.2f, 0.9f, 0.90f, 1.5707963f },
};

struct jarvis_biquad {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    float z1 = 0.0f, z2 = 0.0f;

    // transposed direct form II
    float run(float x) {
        const float y = b0 * x + z1;
        z1            = b1 * x - a1 * y + z2;
        z2            = b2 * x - a2 * y;
        return y;
    }
};

// RBJ cookbook shelf with a slope of 1.
static jarvis_biquad jarvis_shelf_init(const jarvis_shelf & p, float sample_rate) {
    const double A  = pow(10.0, p.gain_db / 40.0);
    const double w  = 2.0 * M_PI * p.f0 / sample_rate;
    const double c  = cos(w);
    const double sq = sqrt(2.0 * A) * sin(w);
    const double s  = p.high ? 1.0 : -1.0;

    const double b0 = A * ((A + 1) + s * (A - 1) * c + sq);
    const double b1 = -2.0 * s * A * ((A - 1) + s * (A + 1) * c);
    const double b2 = A * ((A + 1) + s * (A - 1) * c - sq);
    const double a0 = (A + 1) - s * (A - 1) * c + sq;
    const double a1 = 2.0 * s * ((A - 1) - s * (A + 1) * c);
    const double a2 = (A + 1) - s * (A - 1) * c - sq;

    jarvis_biquad q;
    q.b0 = (float) (b0 / a0);
    q.b1 = (float) (b1 / a0);
    q.b2 = (float) (b2 / a0);
    q.a1 = (float) (a1 / a0);
    q.a2 = (float) (a2 / a0);
    return q;
}

struct jarvis_path {
    float         gain = 1.0f;
    jarvis_biquad lo, hi;

    float run(float x) { return gain * hi.run(lo.run(x)); }
};

static jarvis_path jarvis_path_init(const jarvis_path_params & p, float sample_rate) {
    jarvis_path path;
    path.gain = powf(10.0f, p.gain_db / 20.0f);
    path.lo   = jarvis_shelf_init(p.lo, sample_rate);
    path.hi   = jarvis_shelf_init(p.hi, sample_rate);
    return path;
}

struct jarvis_voice {
    float amp    = 0.0f;
    float center = 0.0f;         // samples
    float depth  = 0.0f;         // samples
    float c = 1.0f, s = 0.0f;    // LFO phasor
    float wc = 1.0f, ws = 0.0f;  // its rotation per sample
};

struct jarvis_fx {
    float              sample_rate = 0.0f;
    std::vector<float> ring;            // a power of two past the echo delay and the interpolation taps
    uint32_t           mask       = 0;
    uint32_t           pos        = 0;  // next write slot
    float              echo_delay = 0.0f;
    jarvis_path        leak, echo, chorus;
    jarvis_voice       voices[2];
};

jarvis_fx * jarvis_fx_new(int sample_rate) {
    jarvis_fx * fx  = new jarvis_fx();
    fx->sample_rate = (float) sample_rate;
    fx->echo_delay  = JARVIS_ECHO_MS * fx->sample_rate / 1000.0f;
    uint32_t size   = 1;
    while (size < (uint32_t) fx->echo_delay + 8) {
        size <<= 1;
    }
    fx->ring.resize(size);
    fx->mask = size - 1;
    jarvis_fx_reset(fx);
    return fx;
}

void jarvis_fx_free(jarvis_fx * fx) {
    delete fx;
}

void jarvis_fx_reset(jarvis_fx * fx) {
    const float ms = fx->sample_rate / 1000.0f;
    std::fill(fx->ring.begin(), fx->ring.end(), 0.0f);
    fx->pos    = 0;
    fx->leak   = jarvis_path_init(JARVIS_LEAK, fx->sample_rate);
    fx->echo   = jarvis_path_init(JARVIS_ECHO, fx->sample_rate);
    fx->chorus = jarvis_path_init(JARVIS_CHORUS, fx->sample_rate);
    for (int i = 0; i < 2; i++) {
        const jarvis_voice_params & p = JARVIS_VOICES[i];
        const double                w = 2.0 * M_PI * p.rate_hz / fx->sample_rate;
        jarvis_voice &              v = fx->voices[i];
        v.amp                         = sqrtf(p.share);
        v.center                      = p.center_ms * ms;
        v.depth                       = p.depth_ms * ms;
        v.c                           = cosf(p.phase);
        v.s                           = sinf(p.phase);
        v.wc                          = (float) cos(w);
        v.ws                          = (float) sin(w);
    }
}

// 4 point cubic Hermite read of the sample written delay samples before the
// last one. The integer part stays apart from the fraction, so the index is
// exact however long the stream runs.
static inline float jarvis_read(const jarvis_fx * fx, float delay) {
    const float    fd  = floorf(delay);
    const float    t   = 1.0f - (delay - fd);
    const uint32_t i   = fx->pos - (uint32_t) fd - 2;
    const float    xm1 = fx->ring[(i - 1) & fx->mask];
    const float    x0  = fx->ring[i & fx->mask];
    const float    x1  = fx->ring[(i + 1) & fx->mask];
    const float    x2  = fx->ring[(i + 2) & fx->mask];
    const float    c1  = 0.5f * (x1 - xm1);
    const float    c2  = xm1 - 2.5f * x0 + 2.0f * x1 - 0.5f * x2;
    const float    c3  = 0.5f * (x2 - xm1) + 1.5f * (x0 - x1);
    return ((c3 * t + c2) * t + c1) * t + x0;
}

void jarvis_fx_process(jarvis_fx * fx, float * pcm, size_t n) {
    for (size_t k = 0; k < n; k++) {
        const float x                = pcm[k];
        // the shortest read is 32 ms back, far behind the sample written here
        fx->ring[fx->pos & fx->mask] = x;
        fx->pos++;

        float ch = 0.0f;
        for (jarvis_voice & v : fx->voices) {
            ch += v.amp * jarvis_read(fx, v.center + v.depth * v.s);
            const float c = v.c * v.wc - v.s * v.ws;
            v.s           = v.s * v.wc + v.c * v.ws;
            v.c           = c;
        }
        pcm[k] = x + fx->leak.run(x) + fx->echo.run(jarvis_read(fx, fx->echo_delay)) + fx->chorus.run(ch);
    }
    // the phasors drift slowly in float: back on the unit circle once per call
    for (jarvis_voice & v : fx->voices) {
        const float r = 1.0f / sqrtf(v.c * v.c + v.s * v.s);
        v.c *= r;
        v.s *= r;
    }
}

size_t jarvis_fx_tail(const jarvis_fx * fx) {
    return (size_t) ceilf(fx->echo_delay) + 2;
}
