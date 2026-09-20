#pragma once
// llm-agent.h: the loop that lets the model use the tools of a llama.cpp
// server
//
// The endpoint runs the tools and the client drives the rounds: the loop
// offers the tools the session checked, streams an answer, runs the calls it
// asks for, hands the results back, and starts again until an answer comes
// without a call. The voice hears every round, so a model that speaks before
// it calls is spoken as it writes.
//
// This is the one mode that is not OpenAI compatible: the routes it uses
// belong to llama.cpp.

#include "llm-client.h"

#define LLM_AGENT_MAX_ROUNDS 10  // rounds a turn takes when the session names none

// Names of the tools the endpoint runs, built-in and MCP alike, in the order
// it lists them: what the page turns into checkboxes.
bool llm_agent_tools(const llm_client_params & params, std::vector<std::string> & names);

// Answers one turn, growing messages with the calls and their results. Text
// receives the words of every round, deltas included. Returns false on a
// transport error, on a tool that fails, on cancellation, or when the rounds
// run out, with the reason in llm_client_last_error().
bool llm_agent_run(llm_client *                     c,
                   const llm_client_params &        params,
                   const std::vector<std::string> & enabled,
                   int                              max_rounds,
                   std::vector<llm_message> &       messages,
                   llm_delta_cb                     cb,
                   void *                           user,
                   const std::atomic<bool> *        cancel,
                   std::string &                    text);
