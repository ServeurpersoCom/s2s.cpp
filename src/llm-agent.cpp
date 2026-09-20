// llm-agent.cpp: one turn, as many rounds as the model asks for
//
// A round is a stream plus the calls it ends on. The endpoint keeps no state
// between rounds, so the whole conversation travels again each time, the
// assistant message carrying its calls and one tool message carrying each
// result.

#include "llm-agent.h"

#include "s2s-error.h"
#include "yyjson.h"

#include <algorithm>

// The definitions of the checked tools, as the JSON array a request carries.
static std::string llm_agent_definitions(const std::vector<llm_tool> &    tools,
                                         const std::vector<std::string> & enabled,
                                         size_t &                         count) {
    std::string array = "[";
    count             = 0;
    for (const llm_tool & tool : tools) {
        if (std::find(enabled.begin(), enabled.end(), tool.name) == enabled.end()) {
            continue;
        }
        array += count++ ? "," : "";
        array += tool.definition;
    }
    array += "]";
    return array;
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

bool llm_agent_tools(const llm_client_params & params, std::vector<std::string> & names) {
    std::vector<llm_tool> tools;
    if (!llm_client_tools(params, tools)) {
        return false;
    }

    names.clear();
    for (const llm_tool & tool : tools) {
        names.push_back(tool.name);
    }
    return true;
}

bool llm_agent_run(llm_client *                     c,
                   const llm_client_params &        params,
                   const std::vector<std::string> & enabled,
                   int                              max_rounds,
                   std::vector<llm_message> &       messages,
                   llm_delta_cb                     cb,
                   void *                           user,
                   const std::atomic<bool> *        cancel,
                   std::string &                    text) {
    std::vector<llm_tool> available;
    if (!llm_client_tools(params, available)) {
        return false;
    }

    size_t            count       = 0;
    const std::string definitions = llm_agent_definitions(available, enabled, count);
    s2s_log(S2S_LOG_INFO, "[Agent] %zu tools offered of %zu the endpoint runs", count, available.size());

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
            if (!llm_client_tool_call(c, call.name, call.arguments, cancel, result)) {
                return false;
            }
            s2s_log(S2S_LOG_INFO, "[Agent] Round %d, %s returned %zu bytes", round + 1, call.name.c_str(),
                    result.size());
            messages.push_back({ "tool", result, "", call.id });
        }
    }

    s2s_set_error("[Agent] The tools still had work to do after %d rounds", max_rounds);
    return false;
}
