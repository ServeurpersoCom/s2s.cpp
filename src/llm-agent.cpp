// llm-agent.cpp: one turn, as many rounds as the model asks for
//
// A round is a stream plus the calls it ends on. The endpoint keeps no state
// between rounds, so the whole conversation travels again each time, the
// assistant message carrying its calls and one tool message carrying each
// result.

#include "llm-agent.h"

#include "s2s-error.h"
#include "timer.h"
#include "yyjson.h"

#include <algorithm>

// A call that fails this close to the tool timeout ran into it: the socket
// wait may wake a clock tick before its deadline.
#define LLM_AGENT_TIMER_SLACK_MS 50

// One tool the session may check, and the server that runs it: an MCP
// client, or the endpoint when there is none.
struct LlmAgentTool {
    llm_tool     tool;
    mcp_client * mcp = nullptr;
};

struct llm_agent {
    std::vector<mcp_client *> clients;  // one per server, in its order, NULL for one that cannot be reached
};

llm_agent * llm_agent_new(const std::vector<mcp_server_params> & servers) {
    llm_agent * agent = new llm_agent();
    for (const mcp_server_params & server : servers) {
        mcp_client * client = mcp_client_new(server);
        if (!client) {
            s2s_log(S2S_LOG_WARN, "[Agent] MCP server %s: %s", server.url.c_str(), mcp_client_last_error());
        }
        agent->clients.push_back(client);
    }
    return agent;
}

void llm_agent_free(llm_agent * agent) {
    if (!agent) {
        return;
    }
    for (mcp_client * client : agent->clients) {
        mcp_client_free(client);
    }
    delete agent;
}

// Every tool the session may check, each with its server. A name that comes
// up twice keeps its first server.
static bool llm_agent_gather(llm_agent *                 agent,
                             const llm_client_params &   params,
                             const std::atomic<bool> *   cancel,
                             std::vector<LlmAgentTool> & tools) {
    tools.clear();
    const auto add = [&tools](const llm_tool & tool, mcp_client * mcp) {
        for (const LlmAgentTool & known : tools) {
            if (known.tool.name == tool.name) {
                s2s_log(S2S_LOG_WARN, "[Agent] Tool %s is offered twice, the first server keeps it", tool.name.c_str());
                return;
            }
        }
        tools.push_back({ tool, mcp });
    };
    for (mcp_client * client : agent->clients) {
        std::vector<llm_tool> listed;
        if (!client || !mcp_client_tools(client, cancel, listed)) {
            if (cancel && cancel->load()) {
                return false;
            }
            s2s_log(S2S_LOG_WARN, "[Agent] MCP tools unavailable: %s", mcp_client_last_error());
            continue;
        }
        for (const llm_tool & tool : listed) {
            add(tool, client);
        }
    }
    std::vector<llm_tool> own;
    if (llm_client_tools(params, own)) {
        for (const llm_tool & tool : own) {
            add(tool, nullptr);
        }
    } else if (agent->clients.empty()) {
        return false;
    } else {
        s2s_log(S2S_LOG_INFO, "[Agent] The endpoint runs no tools of its own: %s", llm_client_last_error());
    }
    return true;
}

// The definitions of the checked tools, as the JSON array a request carries.
static std::string llm_agent_definitions(const std::vector<LlmAgentTool> & tools,
                                         const std::vector<std::string> &  enabled,
                                         size_t &                          count) {
    std::string array = "[";
    count             = 0;
    for (const LlmAgentTool & entry : tools) {
        if (std::find(enabled.begin(), enabled.end(), entry.tool.name) == enabled.end()) {
            continue;
        }
        array += count++ ? "," : "";
        array += entry.tool.definition;
    }
    array += "]";
    return array;
}

// Runs one call on the server of its tool.
static bool llm_agent_call(const std::vector<LlmAgentTool> & tools,
                           llm_client *                      c,
                           const std::string &               name,
                           const std::string &               arguments,
                           const std::atomic<bool> *         cancel,
                           std::string &                     result) {
    for (const LlmAgentTool & entry : tools) {
        if (entry.tool.name == name) {
            return entry.mcp ? mcp_client_call(entry.mcp, name, arguments, cancel, result) :
                               llm_client_tool_call(c, name, arguments, cancel, result);
        }
    }
    s2s_set_error("[Agent] The model called %s, a tool nobody offered", name.c_str());
    return false;
}

// One call of an answer: what to run, with which arguments, and the id the
// result answers.
struct LlmAgentCall {
    std::string id;
    std::string name;
    std::string arguments;
};

static std::vector<LlmAgentCall> llm_agent_calls(const std::string & json) {
    std::vector<LlmAgentCall> calls;

    yyjson_doc * doc = yyjson_read(json.c_str(), json.size(), 0);
    if (!doc) {
        return calls;
    }

    const auto text = [](yyjson_val * value) {
        return value && yyjson_is_str(value) ? std::string(yyjson_get_str(value), yyjson_get_len(value)) :
                                               std::string();
    };

    yyjson_val * root  = yyjson_doc_get_root(doc);
    size_t       index = 0;
    size_t       max   = 0;
    yyjson_val * item  = nullptr;
    yyjson_arr_foreach(root, index, max, item) {
        yyjson_val * function = yyjson_obj_get(item, "function");
        LlmAgentCall call;
        call.id        = text(yyjson_obj_get(item, "id"));
        call.name      = function ? text(yyjson_obj_get(function, "name")) : "";
        call.arguments = function ? text(yyjson_obj_get(function, "arguments")) : "";
        calls.push_back(call);
    }

    yyjson_doc_free(doc);
    return calls;
}

bool llm_agent_tools(llm_agent *                    agent,
                     const llm_client_params &      params,
                     const std::atomic<bool> *      cancel,
                     std::vector<llm_agent_group> & groups) {
    std::vector<LlmAgentTool> tools;
    if (!llm_agent_gather(agent, params, cancel, tools)) {
        return false;
    }

    // The servers in the order gather listed them, one group each: the tools
    // of a server come in a row, the endpoint's last.
    groups.clear();
    for (const LlmAgentTool & entry : tools) {
        const std::string url = entry.mcp ? mcp_client_url(entry.mcp) : "";
        if (groups.empty() || groups.back().url != url) {
            groups.push_back({ url, entry.mcp ? mcp_client_server_name(entry.mcp) : "llama.cpp", {} });
        }
        groups.back().tools.push_back(entry.tool.name);
    }
    return true;
}

bool llm_agent_run(llm_agent *                      agent,
                   llm_client *                     c,
                   const llm_client_params &        params,
                   const std::vector<std::string> & enabled,
                   int                              max_rounds,
                   std::vector<llm_message> &       messages,
                   llm_delta_cb                     cb,
                   void *                           user,
                   const std::atomic<bool> *        cancel,
                   std::string &                    text) {
    std::vector<LlmAgentTool> available;
    if (!llm_agent_gather(agent, params, cancel, available)) {
        return false;
    }

    size_t            count       = 0;
    const std::string definitions = llm_agent_definitions(available, enabled, count);
    s2s_log(S2S_LOG_INFO, "[Agent] %zu tools offered of %zu available", count, available.size());

    // Offering nothing is a conversation, and the request says so rather than
    // carrying an empty list.
    llm_client_params tooled = params;
    tooled.tools             = count ? definitions : "";
    if (!llm_client_set_params(c, tooled)) {
        return false;
    }

    text.clear();

    for (int round = 0; round < max_rounds; round++) {
        std::string answer;
        if (!llm_client_stream(c, messages, cb, user, cancel, answer)) {
            return false;
        }
        text += answer;

        const std::vector<LlmAgentCall> calls = llm_agent_calls(llm_client_tool_calls(c));
        if (calls.empty()) {
            return true;
        }

        messages.push_back({ "assistant", answer, llm_client_tool_calls(c), "" });

        for (const LlmAgentCall & call : calls) {
            std::string result;
            const Timer timer;
            if (!llm_agent_call(available, c, call.name, call.arguments, cancel, result)) {
                if (cancel && cancel->load()) {
                    return false;
                }
                // The model reads why its call failed and decides what comes
                // next: another call, another tool, or words.
                const double ms = timer.ms();
                s2s_log(S2S_LOG_WARN, "[Agent] Round %d, %s failed after %.1f s: %s", round + 1, call.name.c_str(),
                        ms / 1000.0, s2s_last_error());
                const bool timed_out = ms + LLM_AGENT_TIMER_SLACK_MS >= params.tool_timeout_sec * 1000.0;
                result               = timed_out ? "Error: " + call.name + " did not answer within " +
                                         std::to_string(params.tool_timeout_sec) + " s" :
                                                   std::string("Error: ") + s2s_last_error();
                messages.push_back({ "tool", result, "", call.id });
                continue;
            }
            s2s_log(S2S_LOG_INFO, "[Agent] Round %d, %s returned %zu bytes", round + 1, call.name.c_str(),
                    result.size());
            messages.push_back({ "tool", result, "", call.id });
        }
    }

    s2s_set_error("[Agent] The tools still had work to do after %d rounds", max_rounds);
    return false;
}
