#pragma once
// mcp-client.h: a Model Context Protocol client over Streamable HTTP
//
// One MCP server, reached at one URL by JSON-RPC 2.0 over POST, the transport
// the specification calls Streamable HTTP and the one the official SDK
// speaks. The client does what a voice needs of a server and nothing else:
// the initialize handshake, the list of tools, and their calls. Resources,
// prompts, sampling and the standalone event stream a server may offer are
// left where they are.
//
// The conversation with the server, as the SDK conducts it:
//
//   initialize                -> protocol version, server name, session id
//   notifications/initialized -> 202, nothing to read
//   tools/list                -> tools, page by page
//   tools/call                -> the content the tool produced
//   DELETE                    -> the session ends, 405 when the server keeps none
//
// Every request carries the session id the server gave and the protocol
// version both agreed on. A 404 on a session the server forgot starts a new
// one and repeats the request once. An answer comes back as JSON or as a
// server sent event stream; both are read to the response of the request.

#include "llm-client.h"

#include <atomic>
#include <string>
#include <vector>

// The version the client asks for, the newest the SDK of the reference lists.
#define MCP_PROTOCOL_VERSION "2025-11-25"

struct mcp_client;

struct mcp_server_params {
    std::string url;      // the MCP endpoint, scheme and path included
    std::string api_key;  // sent as a bearer token when set

    // Longest a request may take, connection and answer alike: a tool at
    // work is bounded like the tools of the endpoint.
    int timeout_sec = 10;

    bool operator==(const mcp_server_params & other) const {
        return url == other.url && api_key == other.api_key && timeout_sec == other.timeout_sec;
    }

    bool operator!=(const mcp_server_params & other) const { return !(*this == other); }
};

// Opens the connection without talking to the server. NULL, with the reason
// in mcp_client_last_error(), on a URL the client cannot reach.
mcp_client * mcp_client_new(const mcp_server_params & params);

// Ends the session with the server, when it holds one, then frees the client.
void mcp_client_free(mcp_client * c);

// Runs the handshake and keeps what it gives: the session id and the protocol
// version every later request carries. A client already open does nothing.
bool mcp_client_open(mcp_client * c, const std::atomic<bool> * cancel);

// The name the server gave itself, empty before the handshake, and the URL
// it was reached at.
const char * mcp_client_server_name(const mcp_client * c);
const char * mcp_client_url(const mcp_client * c);

// The tools the server offers, every page of them, as the OpenAI compatible
// definitions a request to the model carries: the input schema of a tool is
// the parameters of a function. Opens the client if needed.
bool mcp_client_tools(mcp_client * c, const std::atomic<bool> * cancel, std::vector<llm_tool> & tools);

// Runs one tool and returns what goes into the content of the tool message:
// the text items of its content joined by newlines, or its structured
// content as JSON when it has no text. A tool that reports an error returns
// true with what it said, so the model reads it. arguments is the JSON object
// the model wrote, an empty object when it wrote none.
bool mcp_client_call(mcp_client *              c,
                     const std::string &       name,
                     const std::string &       arguments,
                     const std::atomic<bool> * cancel,
                     std::string &             result);

const char * mcp_client_last_error(void);
