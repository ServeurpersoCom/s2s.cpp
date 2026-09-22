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

// What the client sets for its answers, conversation included: the list is
// the client's, pushed when it changes, and nothing survives here between two
// turns.
struct ClientSettings {
    std::string             mode = "conversation";
    tts_request             tts;
    llm_client_params       llm;
    std::string             system_prompt;
    std::vector<rt_message> history;

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

// Opens conversation id and starts its talking half, then greets the client
// with session.created. Returns nullptr when its session cannot start, after
// telling the client why.
Connection * conn_open(const ConversationSetup * setup, int id, conn_send_fn send, void * send_user);

// Handles one text frame the client sent, on the thread that reads them.
void conn_frame(Connection * conn, const std::string & frame);

// Whether the conversation is over on its side: the client is gone, or too
// slow to take what is sent.
bool conn_stopped(const Connection * conn);

// Stops the talking half, frees the conversation, and logs its end.
void conn_close(Connection * conn);

// Host of an endpoint URL, with its port when it carries one: what an
// allowlist entry is compared against.
std::string url_host(const std::string & url);

// An endpoint the client names is fetched by this process, so an empty
// allowlist means the server can be asked to reach anything it can route to:
// a private network, a metadata service. Naming the hosts closes that.
bool host_allowed(const std::vector<std::string> & hosts, const std::string & url);
