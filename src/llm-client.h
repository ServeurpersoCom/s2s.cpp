#pragma once
// llm-client.h: the reasoning half of the loop, over any OpenAI compatible
// chat completions endpoint
//
// This is the one piece of the pipeline that stays outside the binary: the
// user points the server at llama-server, Ollama, LM Studio or a cloud API
// and keeps control of the model. The client speaks the smallest dialect
// that works everywhere: POST /chat/completions with stream true, then the
// content deltas of choices[0].
//
// The tool verbs are the exception: they speak the llama.cpp routes, and a
// conversation only reaches them in the agentic mode.
//
// Fields other engines add are ignored on purpose, reasoning traces first:
// a reasoning model must not have its thinking read out loud.
//
// Cancellation is the barge-in path. The atomic flag is polled on every
// chunk the socket delivers, and the request is aborted by returning from
// the receiver, which closes the connection and tells the endpoint to stop
// generating.

#include <atomic>
#include <string>
#include <vector>

struct llm_client;

struct llm_message {
    std::string role;  // system, user, assistant, tool
    std::string content;

    // The calls an assistant message asks for, as the JSON array the endpoint
    // sent, and the call a tool message answers. Both empty on a plain turn.
    std::string tool_calls   = {};
    std::string tool_call_id = {};
};

// One tool the endpoint offers: the name a call names, and its OpenAI
// compatible definition, kept as the JSON object the endpoint published.
struct llm_tool {
    std::string name;
    std::string definition;
};

// Sampling knobs the OpenAI dialect carries, plus the ones llama.cpp adds.
// Each one is only sent when it is set, so an endpoint that ignores a field
// never sees it and its own default applies. The sentinels sit outside the
// valid range of each field.
struct llm_sampling {
    float temperature       = -1.0f;    // 0 to 2
    float top_p             = -1.0f;    // 0 to 1
    int   top_k             = -1;       // 0 disables
    float min_p             = -1.0f;    // 0 to 1
    int   max_tokens        = -1;       // completion length cap
    float presence_penalty  = -100.0f;  // -2 to 2
    float frequency_penalty = -100.0f;  // -2 to 2
    int   seed              = -1;       // negative draws a random one

    // How long a reasoning model thinks before it answers, sent as the OpenAI
    // compatible reasoning_effort. none turns the thinking off, which suits a
    // conversation: the voice waits for the thinking, never speaks it. Other
    // values depend on the model: its chat template, or the provider's docs.
    std::string reasoning_effort = "none";
};

// No endpoint and no model by default: they come from whoever runs the
// client. An empty model leaves the choice to the endpoint.
struct llm_client_params {
    std::string  base_url;
    std::string  model;
    std::string  api_key;
    llm_sampling sampling;
    int          timeout_sec = 120;

    // The tools every request offers the model, as a JSON array of OpenAI
    // compatible definitions. Empty leaves the request to the plain dialect.
    std::string tools;
};

// Receives text deltas as they arrive. Returning false cancels the request.
typedef bool (*llm_delta_cb)(const char * text, void * user);

// Returns NULL, with the reason in llm_client_last_error(), on a URL the
// client cannot reach: no scheme, a bad port, a scheme it does not speak,
// https on a build without TLS.
llm_client * llm_client_new(const llm_client_params & params);
void         llm_client_free(llm_client * c);

// Points a client at new settings. The connection to the endpoint stays open
// from one request to the next while the host stays the same, so a request
// skips the TCP and TLS handshakes. Returns false, the client unchanged, on
// a URL it cannot reach.
bool llm_client_set_params(llm_client * c, const llm_client_params & params);

// Streams one completion. text receives the full answer, deltas included.
// Returns false on a transport error, on an error the endpoint reports, as a
// status or inside the stream, or on cancellation, with the reason in
// llm_client_last_error().
bool llm_client_stream(llm_client *                     c,
                       const std::vector<llm_message> & messages,
                       llm_delta_cb                     cb,
                       void *                           user,
                       const std::atomic<bool> *        cancel,
                       std::string &                    text);

// Lists the models the endpoint exposes, through GET {base}/models. The
// server proxies this for the web UI: the browser only ever talks to
// s2s-server, so an endpoint on a loopback address or without CORS headers
// still fills the selector, and the API key never leaves the machine.
bool llm_client_models(const llm_client_params & params, std::vector<std::string> & models);

// The calls the last stream ended on, as the JSON array a request takes back
// in an assistant message, empty when the model answered with words alone.
const char * llm_client_tool_calls(const llm_client * c);

// Lists the tools the endpoint runs, through GET {host}/tools, the llama.cpp
// route that serves its built-in tools and those of its MCP servers alike.
// An endpoint without that route fails here, which is what tells the user
// that the tools of this mode need a llama.cpp server.
bool llm_client_tools(const llm_client_params & params, std::vector<llm_tool> & tools);

// Runs one tool, through POST {host}/tools, and returns what goes into the
// content of the tool message: the plain text of a text answer, the JSON
// object itself otherwise. arguments is the JSON object the model wrote.
bool llm_client_tool_call(llm_client *              c,
                          const std::string &       name,
                          const std::string &       arguments,
                          const std::atomic<bool> * cancel,
                          std::string &             result);

const char * llm_client_last_error(void);
