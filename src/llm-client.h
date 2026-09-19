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
    std::string role;  // system, user, assistant
    std::string content;
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

    // How long a reasoning model thinks before it answers, in the words the
    // endpoint takes (none, minimal, low, medium, high). The thinking is never
    // spoken, but the voice waits for it.
    std::string reasoning_effort;
};

// No endpoint and no model by default: they come from whoever runs the
// client. An empty model leaves the choice to the endpoint.
struct llm_client_params {
    std::string  base_url;
    std::string  model;
    std::string  api_key;
    llm_sampling sampling;
    int          timeout_sec = 120;
};

// Receives text deltas as they arrive. Returning false cancels the request.
typedef bool (*llm_delta_cb)(const char * text, void * user);

llm_client * llm_client_new(const llm_client_params & params);
void         llm_client_free(llm_client * c);

// Points a client at new settings. The connection to the endpoint stays open
// from one request to the next while the host stays the same, so a request
// skips the TCP and TLS handshakes. Returns false, the client unchanged, on
// a URL without a scheme.
bool llm_client_set_params(llm_client * c, const llm_client_params & params);

// Streams one completion. text receives the full answer, deltas included.
// Returns false on a transport error or on cancellation, with the reason in
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

const char * llm_client_last_error(void);
