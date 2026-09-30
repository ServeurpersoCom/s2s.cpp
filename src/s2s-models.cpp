// s2s-models.cpp: the models of the voice loop, found and loaded once

#include "s2s-models.h"

#include "model-find.h"
#include "s2s-error.h"

bool models_load(const std::string & models_dir,
                 const std::string & voices_dir,
                 const tts_engine &  engine,
                 ModelFiles &        files,
                 ConversationSetup & setup) {
    files.vad  = find_model(models_dir, "silero-vad", "");
    files.turn = find_model(models_dir, "smart-turn", "");
    files.asr  = find_model(models_dir, "parakeet-ultra", "");
    if (files.asr.empty()) {
        files.asr = find_model(models_dir, "parakeet-tdt", "");
    }
    files.talker = find_model(models_dir, "qwen-talker", "-base-");
    files.codec  = find_model(models_dir, "qwen-tokenizer", "");
    files.aec    = find_model(models_dir, "localvqe", "");

    if (files.vad.empty() || files.turn.empty() || files.asr.empty() || files.talker.empty() || files.codec.empty() ||
        files.aec.empty()) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: missing models in %s, run ./models.sh", models_dir.c_str());
        return false;
    }

    s2s_log(S2S_LOG_INFO, "[Load] VAD %s", files.vad.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] Turn %s", files.turn.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] ASR %s", files.asr.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] TTS %s + %s, voices from %s", files.talker.c_str(), files.codec.c_str(),
            voices_dir.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] AEC %s", files.aec.c_str());

    ServerModels & models = setup.models;

    models.vad = sv_init(files.vad.c_str());
    if (!models.vad) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", sv_last_error());
        return false;
    }

    models.turn = st_init(files.turn.c_str());
    if (!models.turn) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", st_last_error());
        return false;
    }

    pk_init_params asr_init = pk_init_default_params();
    asr_init.model_path     = files.asr.c_str();

    models.asr = pk_init(&asr_init);
    if (!models.asr) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", pk_last_error());
        return false;
    }

    tts_bridge_params tts_init;
    tts_init.talker_path             = files.talker;
    tts_init.codec_path              = files.codec;
    tts_init.voices_dir              = voices_dir;
    tts_init.sampling.max_new_tokens = S2S_TTS_MAX_NEW_TOKENS;
    tts_init.engine                  = engine;

    models.tts = tts_bridge_load(tts_init);
    if (!models.tts) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", tts_bridge_last_error());
        return false;
    }

    models.aec = lv_init(files.aec.c_str());
    if (!models.aec) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", lv_last_error());
        return false;
    }

    setup.defaults.tts = tts_bridge_defaults_request(models.tts);
    return true;
}

void models_free(ServerModels & models) {
    lv_free(models.aec);
    tts_bridge_free(models.tts);
    pk_free(models.asr);
    st_free(models.turn);
    sv_free(models.vad);
    models = ServerModels();
}
