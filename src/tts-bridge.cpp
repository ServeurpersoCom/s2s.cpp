// tts-bridge.cpp: qwentts.cpp driven from the session loop
//
// The submodule owns its worker, its KV cache sets and its device context, so
// this file is only the translation between the two vocabularies: a unit of
// text in, PCM chunks out, one atomic flag for the barge-in.
//
// Two guards live here, both learned from a talker that ran to its frame cap
// on a degenerate input: a text shorter than the floor is not spoken at all,
// and the frame budget of a synthesis is derived from the length of the text
// instead of being left at the model maximum. Both are settings published in
// /props and editable from the surface, never hidden numbers.

#include "tts-bridge.h"

#include "qwen.h"
#include "s2s-error.h"

#include <mutex>
#include <string>

struct tts_bridge {
    qt_context * ctx = nullptr;

    std::vector<std::string> speakers;
    std::vector<std::string> languages;

    // What a request falls back on, and what the caller publishes: the model
    // table, the submodule sampling and the guards of this build.
    tts_request  defaults;
    tts_sampling sampling_defaults;
    tts_engine   engine;

    int sample_rate = 0;
};

// Both callbacks run on the qwentts worker. They carry the same state so a
// cancellation raised between two chunks is seen by whichever fires first.
struct TtsCall {
    tts_chunk_cb              cb     = nullptr;
    void *                    user   = nullptr;
    const std::atomic<bool> * cancel = nullptr;
};

static bool tts_bridge_cancelled(void * user_data) {
    const TtsCall * call = (const TtsCall *) user_data;
    return call->cancel && call->cancel->load();
}

static bool tts_bridge_chunk(const float * samples, int n_samples, void * user_data) {
    const TtsCall * call = (const TtsCall *) user_data;
    if (call->cancel && call->cancel->load()) {
        return false;
    }
    if (!call->cb) {
        return true;
    }
    return call->cb(samples, (size_t) n_samples, call->user);
}

tts_bridge * tts_bridge_load(const tts_bridge_params & params) {
    if (params.talker_path.empty() || params.codec_path.empty()) {
        s2s_set_error("[TTS] Talker_path or codec_path is empty");
        return nullptr;
    }

    qt_init_params init;
    qt_init_default_params(&init);
    init.talker_path = params.talker_path.c_str();
    init.codec_path  = params.codec_path.c_str();
    init.max_batch   = params.engine.max_batch > 0 ? params.engine.max_batch : 1;
    init.use_fa      = params.engine.use_fa;
    init.clamp_fp16  = params.engine.clamp_fp16;
    if (params.engine.codec_chunk_sec > 0.0f) {
        init.codec_chunk_sec = params.engine.codec_chunk_sec;
    }

    qt_context * ctx = qt_init(&init);
    if (!ctx) {
        s2s_set_error("[TTS] %s", qt_last_error());
        return nullptr;
    }

    tts_bridge * b            = new tts_bridge();
    b->ctx                    = ctx;
    b->defaults.speaker       = params.speaker;
    b->defaults.language      = params.language;
    b->defaults.sampling      = params.sampling;
    b->defaults.guards        = params.guards;
    b->engine                 = params.engine;
    b->engine.max_batch       = init.max_batch;
    b->engine.codec_chunk_sec = init.codec_chunk_sec;
    b->sample_rate            = 24000;

    for (int i = 0; i < qt_n_speakers(ctx); i++) {
        b->speakers.emplace_back(qt_speaker_name(ctx, i));
    }
    for (int i = 0; i < qt_n_languages(ctx); i++) {
        b->languages.emplace_back(qt_language_name(ctx, i));
    }

    // The defaults belong to the submodule: read them once, publish them,
    // never copy them into this project.
    qt_tts_params reference;
    qt_tts_default_params(&reference);
    b->sampling_defaults.temperature           = reference.temperature;
    b->sampling_defaults.top_k                 = reference.top_k;
    b->sampling_defaults.top_p                 = reference.top_p;
    b->sampling_defaults.repetition_penalty    = reference.repetition_penalty;
    b->sampling_defaults.subtalker_temperature = reference.subtalker_temperature;
    b->sampling_defaults.subtalker_top_k       = reference.subtalker_top_k;
    b->sampling_defaults.subtalker_top_p       = reference.subtalker_top_p;
    b->sampling_defaults.max_new_tokens        = reference.max_new_tokens;
    b->sampling_defaults.seed                  = reference.seed;

    // A custom voice model refuses to speak without a speaker, so the table it
    // carries decides rather than a name written here.
    if (b->defaults.speaker.empty() && !b->speakers.empty()) {
        b->defaults.speaker = b->speakers.front();
    }

    s2s_log(S2S_LOG_INFO, "[TTS] %s, %zu speakers, %zu languages, %d Hz", qt_version(), b->speakers.size(),
            b->languages.size(), b->sample_rate);
    s2s_log(S2S_LOG_INFO, "[TTS] Speaker %s, language %s",
            b->defaults.speaker.empty() ? "none" : b->defaults.speaker.c_str(), b->defaults.language.c_str());
    s2s_log(S2S_LOG_INFO, "[TTS] Engine: batch %d, flash attention %s, clamp fp16 %s, codec chunk %.1f s",
            b->engine.max_batch, b->engine.use_fa ? "on" : "off", b->engine.clamp_fp16 ? "on" : "off",
            (double) b->engine.codec_chunk_sec);
    return b;
}

void tts_bridge_free(tts_bridge * b) {
    if (!b) {
        return;
    }
    qt_free(b->ctx);
    delete b;
}

int tts_bridge_sample_rate(const tts_bridge * b) {
    return b ? b->sample_rate : 0;
}

const std::vector<std::string> & tts_bridge_speakers(const tts_bridge * b) {
    static const std::vector<std::string> empty;
    return b ? b->speakers : empty;
}

const std::vector<std::string> & tts_bridge_languages(const tts_bridge * b) {
    static const std::vector<std::string> empty;
    return b ? b->languages : empty;
}

const std::string & tts_bridge_speaker(const tts_bridge * b) {
    static const std::string empty;
    return b ? b->defaults.speaker : empty;
}

const tts_sampling & tts_bridge_defaults(const tts_bridge * b) {
    static const tts_sampling empty;
    return b ? b->sampling_defaults : empty;
}

const tts_request & tts_bridge_defaults_request(const tts_bridge * b) {
    static const tts_request empty;
    return b ? b->defaults : empty;
}

// Frames this text deserves at the codec rate, bounded by the model maximum.
static int tts_bridge_budget(const tts_bridge * b, const tts_request & request, size_t n_chars) {
    const float seconds = (float) n_chars / request.guards.chars_per_second + request.guards.margin_seconds;
    const int   frames  = qt_duration_sec_to_tokens(b->ctx, seconds);
    const int   maximum =
        request.sampling.max_new_tokens > 0 ? request.sampling.max_new_tokens : b->sampling_defaults.max_new_tokens;
    return frames < maximum ? frames : maximum;
}

bool tts_bridge_speak(tts_bridge *              b,
                      const std::string &       text,
                      const tts_request &       request,
                      tts_chunk_cb              cb,
                      void *                    user,
                      const std::atomic<bool> * cancel) {
    if (!b || text.empty()) {
        s2s_set_error("[TTS] Bridge is NULL or text is empty");
        return false;
    }
    // Characters, not bytes: an accented letter is one character.
    int n_chars = 0;
    for (const char c : text) {
        n_chars += ((unsigned char) c & 0xC0) != 0x80;
    }
    if (n_chars < request.guards.min_chars) {
        // A couple of characters is not speech, it is the tail of a noise the
        // recognizer had to name. Saying it aloud is what sends the talker
        // off its distribution.
        s2s_log(S2S_LOG_INFO, "[TTS] Skipped %d characters, fewer than %d", n_chars, request.guards.min_chars);
        return true;
    }
    if (cancel && cancel->load()) {
        return false;
    }

    TtsCall call;
    call.cb     = cb;
    call.user   = user;
    call.cancel = cancel;

    qt_tts_params params;
    qt_tts_default_params(&params);

    {
        const std::string & speaker  = request.speaker.empty() ? b->defaults.speaker : request.speaker;
        const std::string & language = request.language.empty() ? b->defaults.language : request.language;

        params.text    = text.c_str();
        params.lang    = language.empty() ? nullptr : language.c_str();
        params.speaker = speaker.empty() ? nullptr : speaker.c_str();

        if (request.sampling.temperature >= 0.0f) {
            params.temperature = request.sampling.temperature;
        }
        if (request.sampling.top_k >= 0) {
            params.top_k = request.sampling.top_k;
        }
        if (request.sampling.top_p >= 0.0f) {
            params.top_p = request.sampling.top_p;
        }
        if (request.sampling.repetition_penalty >= 0.0f) {
            params.repetition_penalty = request.sampling.repetition_penalty;
        }
        if (request.sampling.subtalker_temperature >= 0.0f) {
            params.subtalker_temperature = request.sampling.subtalker_temperature;
        }
        if (request.sampling.subtalker_top_k >= 0) {
            params.subtalker_top_k = request.sampling.subtalker_top_k;
        }
        if (request.sampling.subtalker_top_p >= 0.0f) {
            params.subtalker_top_p = request.sampling.subtalker_top_p;
        }
        if (request.sampling.seed >= 0) {
            params.seed = request.sampling.seed;
        }

        params.max_new_tokens = tts_bridge_budget(b, request, (size_t) n_chars);
    }

    params.cancel             = tts_bridge_cancelled;
    params.cancel_user_data   = &call;
    params.on_chunk           = tts_bridge_chunk;
    params.on_chunk_user_data = &call;

    s2s_log(S2S_LOG_INFO, "[TTS] Speaking %d characters, budget %d frames", n_chars, params.max_new_tokens);

    qt_audio        audio  = {};
    const qt_status status = qt_synthesize(b->ctx, &params, &audio);
    qt_audio_free(&audio);

    if (status == QT_STATUS_CANCELLED) {
        return false;
    }
    if (status != QT_STATUS_OK) {
        s2s_set_error("[TTS] %s", qt_last_error());
        return false;
    }

    return true;
}

const char * tts_bridge_last_error(void) {
    return s2s_last_error();
}
