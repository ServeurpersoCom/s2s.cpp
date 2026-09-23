#pragma once
// llm-agent.h: the loop that lets the model use tools
//
// The tools come from three places: the built-in tools the process runs
// itself, the endpoint, when it is a llama.cpp server with its /tools routes,
// and any number of MCP servers the session names, which the agent holds
// open for as long as it lives. The client drives the rounds: the loop offers
// the tools the session checked, streams an answer, runs the calls it asks
// for, hands the results back, and starts again until an answer comes
// without a call. The voice hears every round, so a model that speaks before
// it calls is spoken as it writes.
//
// With MCP servers the endpoint can be any OpenAI compatible server; the
// /tools routes are the one thing that needs llama.cpp.

#include "llm-client.h"
#include "mcp-client.h"

#define LLM_AGENT_MAX_ROUNDS 10          // rounds a turn takes when the session names none
#define LLM_AGENT_BUILTIN    "built-in"  // the group of the tools the process runs itself

// Runs a built-in tool on the arguments the model wrote. Returns false with
// the reason in s2s_set_error, which the model reads as the result.
typedef bool (*llm_agent_fn)(const std::string & arguments, void * user, std::string & result);

// A tool the process runs itself. A list only shows it: fn stays NULL there.
struct llm_agent_builtin {
    llm_tool     tool;
    llm_agent_fn fn   = nullptr;
    void *       user = nullptr;
};

struct llm_agent;

// The tool servers of one session: the MCP servers, not yet reached. The
// endpoint of every call is named by its params.
llm_agent * llm_agent_new(const std::vector<mcp_server_params> & servers);
void        llm_agent_free(llm_agent * agent);

// The tools of one server: an MCP server by its URL and the name it gave
// itself, the endpoint, with an empty URL and "llama.cpp" for a name, or the
// process, with an empty URL and LLM_AGENT_BUILTIN.
struct llm_agent_group {
    std::string              url;
    std::string              name;
    std::vector<std::string> tools;
};

// The tools the session may check, server by server, the built-in ones
// first, the MCP servers after them in the order they were named, the
// endpoint's own last, a name listed once. A server that cannot be reached is
// left out. An endpoint without /tools contributes none; that is the failure
// only when nothing else offers a tool either.
bool llm_agent_tools(llm_agent *                            agent,
                     const std::vector<llm_agent_builtin> & builtins,
                     const llm_client_params &              params,
                     const std::atomic<bool> *              cancel,
                     std::vector<llm_agent_group> &         groups);

// Answers one turn, growing messages with the calls and their results. Text
// receives the words of every round, deltas included. A call that fails, a
// timeout included, is answered with its error as the tool result, and the
// model goes on from there. Returns false on a transport error of the
// endpoint, on cancellation, or when the rounds run out, with the reason in
// llm_client_last_error().
bool llm_agent_run(llm_agent *                            agent,
                   const std::vector<llm_agent_builtin> & builtins,
                   llm_client *                           c,
                   const llm_client_params &              params,
                   const std::vector<std::string> &       enabled,
                   int                                    max_rounds,
                   std::vector<llm_message> &             messages,
                   llm_delta_cb                           cb,
                   void *                                 user,
                   const std::atomic<bool> *              cancel,
                   std::string &                          text);
