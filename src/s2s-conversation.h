#pragma once
// s2s-conversation.h: one conversation, from the frames a client sends to the
// frames it receives
//
// The engine knows the Realtime events and nothing of the transport: whoever
// opens a conversation hands it every text frame the client sends, and a
// function that delivers the frames going back. The models and the session
// defaults are shared by every conversation of the process.

#include "audio-resample.h"
#include "llm-agent.h"
#include "localvqe.h"
#include "parakeet.h"
#include "realtime-proto.h"
#include "silero.h"
#include "smart-turn.h"
#include "tts-bridge.h"

#include <string>
#include <vector>

#define S2S_INPUT_RATE SAMPLE_RATE_24K  // the rate the Realtime protocol carries
#define S2S_MODEL_RATE 16000            // the rate the VAD and the recognizer work at

struct ServerModels {
    sv_context * vad  = nullptr;
    st_context * turn = nullptr;
    pk_context * asr  = nullptr;
    tts_bridge * tts  = nullptr;
    lv_context * aec  = nullptr;
};

// The wake phrase gate of the conversation and agentic modes. off answers
// every turn. alone and anywhere answer none until a turn wakes the
// assistant, one made of a wake phrase alone or one that holds one
// anywhere; a turn that holds a sleep phrase anywhere puts it back to sleep
// after its answer.
struct WakeSettings {
    std::string              mode          = "off";
    std::vector<std::string> phrases       = { "jarvis" };
    std::vector<std::string> sleep_phrases = { "shut up" };
};

// What the client sets for its answers, conversation included: the list is
// the client's, pushed when it changes, and nothing survives here between two
// turns.
struct ClientSettings {
    std::string             mode = "loopback";
    tts_request             tts;
    llm_client_params       llm;
    std::string             system_prompt = "You are a voice assistant. Answer in one or two short spoken sentences.";
    std::vector<rt_message> history;
    WakeSettings            wake;

    // Tools the model is offered in the agentic mode, by name, the MCP
    // servers that run some of them, and the rounds of calls one turn may
    // take.
    std::vector<std::string>       tools;
    std::vector<mcp_server_params> mcp;
    int                            max_rounds = LLM_AGENT_MAX_ROUNDS;
};

// What every conversation of the process shares, set once at startup.
struct ConversationSetup {
    ServerModels models;

    // What a session runs with before its first session.update, and what
    // every field an update leaves out goes back to: the values /props
    // publishes.
    ClientSettings defaults;

    // The endpoint belongs to the server once it names one on the command
    // line, and so do the MCP servers once it names any: sessions neither see
    // them nor change them. Otherwise the server has none, and each session
    // names its own, from the allowed hosts.
    bool                     llm_fixed = false;
    bool                     mcp_fixed = false;
    std::vector<std::string> llm_hosts;
};

// Delivers one frame to the client. Returns false once the client is gone.
typedef bool (*conn_send_fn)(const std::string & frame, void * user);

struct Connection;

// Opens conversation id and starts its talking half. Returns nullptr when its
// session cannot start, after telling the client why.
Connection * conn_open(const ConversationSetup * setup, int id, conn_send_fn send, void * send_user);

// Handles one text frame the client sent, on the thread that reads them.
void conn_frame(Connection * conn, const std::string & frame);

// Whether the conversation is over on its side: the client is gone, or too
// slow to take what is sent.
bool conn_stopped(const Connection * conn);

// Stops the talking half, frees the conversation, and logs its end.
void conn_close(Connection * conn);

// The effects a voice can run through, off first: the default.
const std::vector<std::string> & conn_effects();

// The modes of the wake phrase gate, off first: the default.
const std::vector<std::string> & conn_wake_modes();

// The built-in tools are defined once for the whole process: they sit at the
// head of every prompt, so a word that changed there would make the endpoint
// prefill the whole conversation again.

// The definition of sleep, the built-in tool that lets the model stop
// answering until a wake phrase is heard again.
llm_tool conn_sleep_tool();

// The definition of set_voice, the built-in tool that lets the model change
// the voice it speaks with and the effect over it: every label the bridge
// lists and every effect.
llm_tool conn_voice_tool(const tts_bridge * tts);

// Host of an endpoint URL, with its port when it carries one: what an
// allowlist entry is compared against.
std::string url_host(const std::string & url);

// An endpoint the client names is fetched by this process, so an empty
// allowlist means the server can be asked to reach anything it can route to:
// a private network, a metadata service. Naming the hosts closes that.
bool host_allowed(const std::vector<std::string> & hosts, const std::string & url);
