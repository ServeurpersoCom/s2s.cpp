// tts-bridge.cpp: qwentts.cpp driven from the session loop
//
// The submodule owns its worker, its KV cache sets and its device context, so
// this file is only the translation between the two vocabularies: a unit of
// text in, PCM chunks out, one atomic flag for the barge-in.
//
// The voices are latent references read once from the voices directory: a
// <name>.spk speaker embedding alone conditions the timbre, and a <name>.rvq
// with its <name>.txt transcript next to it is reference speech, which the
// talker continues on every unit. A voice with all three files is offered
// both ways, the reference speech first:
//
//   freeman.{spk,rvq,txt} reference speech
//   freeman.spk speaker embedding only
//
// Two guards live here, both learned from a talker that ran to its frame cap
// on a degenerate input: a text shorter than the floor is not spoken at all,
// and the frame budget of a synthesis is derived from the length of the text
// instead of being left at the model maximum. Both are settings published in
// /props and editable from the surface, never hidden numbers.

#include "tts-bridge.h"

#include "qwen.h"
#include "rvq-file.h"
#include "s2s-error.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>

// Bits per packed code in a .rvq file: the 2048 entry codebooks of the 12 Hz
// codec.
#define TTS_RVQ_CODE_BITS 11

// One voice as its files hold it. codes and text are empty for a voice that
// only carries its speaker embedding.
struct TtsVoice {
    std::string          name;
    std::vector<float>   spk;
    std::vector<int32_t> codes;
    int                  n_frames = 0;
    std::string          text;
};

// One way to speak with a voice, under the label a session names it by.
struct TtsVoiceEntry {
    std::string label;
    size_t      voice     = 0;
    bool        reference = false;  // the reference speech on top of the embedding
};

struct tts_bridge {
    qt_context * ctx = nullptr;

    std::vector<TtsVoice>      voices;
    std::vector<TtsVoiceEntry> entries;
    std::vector<std::string>   labels;
    std::vector<std::string>   languages;

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

// qwentts reports through its own log: its lines join the log of the
// project, tagged by thread like every other.
static void tts_bridge_log(enum qt_log_level level, const char * msg, void * user_data) {
    (void) user_data;
    s2s_log(level == QT_LOG_ERROR ? S2S_LOG_ERROR :
            level == QT_LOG_WARN  ? S2S_LOG_WARN :
            level == QT_LOG_DEBUG ? S2S_LOG_DEBUG :
                                    S2S_LOG_INFO,
            "%s", msg);
}

// Whole file into bytes, false when it cannot be read or is empty.
static bool tts_bridge_read(const std::filesystem::path & path, std::string & out) {
    FILE * f = utf8_fopen(path.string().c_str(), "rb");
    if (!f) {
        return false;
    }
    out.clear();
    char   buf[4096];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return !out.empty();
}

static bool tts_bridge_load_voice(const tts_bridge * b, const std::filesystem::path & spk_path, TtsVoice & voice) {
    voice.name = spk_path.stem().string();

    std::string bytes;
    if (!tts_bridge_read(spk_path, bytes) || bytes.size() % sizeof(float) != 0) {
        s2s_set_error("[TTS] Voice %s is not a float embedding", voice.name.c_str());
        return false;
    }
    voice.spk.resize(bytes.size() / sizeof(float));
    memcpy(voice.spk.data(), bytes.data(), bytes.size());

    std::filesystem::path rvq_path = spk_path;
    std::filesystem::path txt_path = spk_path;
    rvq_path.replace_extension(".rvq");
    txt_path.replace_extension(".txt");

    const bool has_rvq = std::filesystem::exists(rvq_path);
    const bool has_txt = std::filesystem::exists(txt_path);
    if (has_rvq != has_txt) {
        s2s_set_error("[TTS] Voice %s needs both its .rvq and its .txt, or neither", voice.name.c_str());
        return false;
    }
    if (!has_rvq) {
        return true;
    }

    if (!rvq_read_file(rvq_path.string().c_str(), qt_num_codebooks(b->ctx), TTS_RVQ_CODE_BITS, voice.codes,
                       &voice.n_frames)) {
        s2s_set_error("[TTS] Voice %s has an unreadable .rvq", voice.name.c_str());
        return false;
    }
    if (!tts_bridge_read(txt_path, voice.text)) {
        s2s_set_error("[TTS] Voice %s has an empty .txt", voice.name.c_str());
        return false;
    }
    while (!voice.text.empty() && (unsigned char) voice.text.back() <= ' ') {
        voice.text.pop_back();
    }
    return true;
}

// Every <name>.spk of the directory, sorted by name, each voice with its
// entries: the default is the first voice, with its reference speech when it
// has one.
static bool tts_bridge_load_voices(tts_bridge * b, const std::string & dir) {
    std::vector<std::filesystem::path> paths;
    std::error_code                    error;
    for (const auto & entry : std::filesystem::directory_iterator(dir, error)) {
        if (entry.path().extension() == ".spk") {
            paths.push_back(entry.path());
        }
    }
    if (paths.empty()) {
        s2s_set_error("[TTS] No .spk voice in %s", dir.c_str());
        return false;
    }
    std::sort(paths.begin(), paths.end());

    for (const std::filesystem::path & path : paths) {
        TtsVoice voice;
        if (!tts_bridge_load_voice(b, path, voice)) {
            return false;
        }
        const size_t index = b->voices.size();
        if (voice.codes.empty()) {
            s2s_log(S2S_LOG_INFO, "[TTS] Voice %s.spk: speaker embedding of %zu values", voice.name.c_str(),
                    voice.spk.size());
        } else {
            s2s_log(S2S_LOG_INFO,
                    "[TTS] Voice %s.{spk,rvq,txt}: speaker embedding of %zu values, reference speech of %d frames",
                    voice.name.c_str(), voice.spk.size(), voice.n_frames);
            b->entries.push_back({ voice.name + ".{spk,rvq,txt} reference speech", index, true });
        }
        b->entries.push_back({ voice.name + ".spk speaker embedding only", index, false });
        b->voices.push_back(std::move(voice));
    }
    for (const TtsVoiceEntry & entry : b->entries) {
        b->labels.push_back(entry.label);
    }
    return true;
}

static const TtsVoiceEntry * tts_bridge_find_voice(const tts_bridge * b, const std::string & label) {
    for (const TtsVoiceEntry & entry : b->entries) {
        if (entry.label == label) {
            return &entry;
        }
    }
    return nullptr;
}

tts_bridge * tts_bridge_load(const tts_bridge_params & params) {
    if (params.talker_path.empty() || params.codec_path.empty()) {
        s2s_set_error("[TTS] Talker or codec path is empty");
        return nullptr;
    }

    qt_log_set(tts_bridge_log, nullptr);

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

    // Voices are references, and only a Base talker takes a reference.
    if (strcmp(qt_model_type(ctx), "base") != 0) {
        s2s_set_error("[TTS] The talker is %s, the voices need a base one", qt_model_type(ctx));
        qt_free(ctx);
        return nullptr;
    }

    tts_bridge * b            = new tts_bridge();
    b->ctx                    = ctx;
    b->defaults.sampling      = params.sampling;
    b->defaults.guards        = params.guards;
    b->engine                 = params.engine;
    b->engine.max_batch       = init.max_batch;
    b->engine.codec_chunk_sec = init.codec_chunk_sec;
    b->sample_rate            = 24000;

    if (!tts_bridge_load_voices(b, params.voices_dir)) {
        tts_bridge_free(b);
        return nullptr;
    }
    b->defaults.voice    = b->labels.front();
    b->defaults.language = "auto";
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

    s2s_log(S2S_LOG_INFO, "[TTS] %s, %zu voices, default %s, %zu languages, %d Hz", qt_version(), b->voices.size(),
            b->defaults.voice.c_str(), b->languages.size(), b->sample_rate);
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

const std::vector<std::string> & tts_bridge_languages(const tts_bridge * b) {
    static const std::vector<std::string> empty;
    return b ? b->languages : empty;
}

const std::vector<std::string> & tts_bridge_voices(const tts_bridge * b) {
    static const std::vector<std::string> empty;
    return b ? b->labels : empty;
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
    if (cancel && cancel->load()) {
        return false;
    }

    const std::string &   name  = request.voice.empty() ? b->defaults.voice : request.voice;
    const TtsVoiceEntry * entry = tts_bridge_find_voice(b, name);
    if (!entry) {
        s2s_set_error("[TTS] Unknown voice %s", name.c_str());
        return false;
    }

    TtsCall call;
    call.cb     = cb;
    call.user   = user;
    call.cancel = cancel;

    qt_tts_params params;
    qt_tts_default_params(&params);

    {
        // The language id comes first in the prompt, whatever the voice: auto
        // leaves it out and the model reads the text as it is written. With
        // the embedding only it sets the pronunciation; reference speech
        // carries its own, which another language pulls against.
        const std::string & language = request.language.empty() ? b->defaults.language : request.language;
        params.text                  = text.c_str();
        params.lang                  = language == "auto" ? nullptr : language.c_str();
        const TtsVoice & voice       = b->voices[entry->voice];
        params.ref_spk_emb           = voice.spk.data();
        params.ref_spk_dim           = (int) voice.spk.size();
        if (entry->reference) {
            params.ref_codes = voice.codes.data();
            params.ref_T     = voice.n_frames;
            params.ref_text  = voice.text.c_str();
        }

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
        // The seed is drawn here rather than in the submodule, so the log
        // names the one every sentence used and any take can be replayed.
        // 53 bits: a JavaScript number holds it exactly, so the seed copied
        // from the log into the page is the one that reaches the talker.
        if (request.sampling.seed >= 0) {
            params.seed = request.sampling.seed;
        } else {
            std::random_device rd;
            params.seed = (int64_t) ((((uint64_t) rd() << 32) ^ (uint64_t) rd()) >> 11);
        }

        params.max_new_tokens = tts_bridge_budget(b, request, (size_t) n_chars);
    }

    params.cancel             = tts_bridge_cancelled;
    params.cancel_user_data   = &call;
    params.on_chunk           = tts_bridge_chunk;
    params.on_chunk_user_data = &call;

    s2s_log(S2S_LOG_INFO, "[TTS] Speaking %d characters, budget %d frames, seed %lld", n_chars, params.max_new_tokens,
            (long long) params.seed);

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
