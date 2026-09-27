#pragma once
// s2s-models.h: the models of the voice loop, found in one directory and
// loaded once for every conversation of the process
//
// Shared by the hosts: s2s-server lends them to its connections, s2s-cli to
// the one conversation it runs on the sound card.

#include "s2s-conversation.h"

#include <string>

// Frame ceiling of one synthesis, the value the Python reference settles on.
// The per utterance budget of the bridge sits well below it; this only bounds
// the worst case.
#define S2S_TTS_MAX_NEW_TOKENS 1536

// The GGUF files picked in the models directory, one per stage.
struct ModelFiles {
    std::string vad;
    std::string turn;
    std::string asr;
    std::string talker;
    std::string codec;
    std::string aec;
};

// Picks every model in models_dir, loads them into setup.models with the
// voices of voices_dir, and sets the voice request of setup.defaults. Returns
// false once the reason is logged.
bool models_load(const std::string & models_dir,
                 const std::string & voices_dir,
                 const tts_engine &  engine,
                 ModelFiles &        files,
                 ConversationSetup & setup);

void models_free(ServerModels & models);
