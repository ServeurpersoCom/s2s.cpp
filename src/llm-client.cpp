// llm-client.cpp: chat completions over server sent events
//
// The endpoint answers a stream of lines. Only two shapes matter:
//
//   data: {"choices":[{"delta":{"content":"..."}}]}
//   data: [DONE]
//
// Everything else, comments, keep alives, usage frames, is skipped. Chunks
// arrive split anywhere, including in the middle of a line and in the middle
// of a UTF-8 sequence, so the parser accumulates and only consumes complete
// lines.

#include "llm-client.h"

#include "httplib.h"
#include "s2s-error.h"
#include "yyjson.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#define LLM_REASON_MAX     200  // characters of an endpoint's error reason kept in the message
#define LLM_CANCEL_POLL_MS 10   // how often a request that receives nothing looks at the cancel flag

struct llm_client {
    llm_client_params params;

    std::string                      host;  // scheme, host and port, what httplib::Client takes
    std::string                      path;  // prefix of the endpoint, /v1 by default
    std::unique_ptr<httplib::Client> http;  // kept alive across requests to the same host
};

// Splits "http://host:port/v1" into the part httplib connects to and the
// prefix every request hangs off.
static bool llm_client_split_url(const std::string & url, std::string & host, std::string & path) {
    const size_t scheme = url.find("://");
    if (scheme == std::string::npos) {
        return false;
    }
    const size_t slash = url.find('/', scheme + 3);
    if (slash == std::string::npos) {
        host = url;
        path = "";
        return true;
    }
    host = url.substr(0, slash);
    path = url.substr(slash);
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    return true;
}

llm_client * llm_client_new(const llm_client_params & params) {
    llm_client * c = new llm_client();
    if (!llm_client_set_params(c, params)) {
        delete c;
        return nullptr;
    }
    return c;
}

bool llm_client_set_params(llm_client * c, const llm_client_params & params) {
    std::string host;
    std::string path;
    if (!llm_client_split_url(params.base_url, host, path)) {
        s2s_set_error("[LLM] Base URL '%s' has no scheme", params.base_url.c_str());
        return false;
    }
    if (!c->http || host != c->host) {
        c->http = std::make_unique<httplib::Client>(host);
        c->http->set_keep_alive(true);
    }
    c->params = params;
    c->host   = host;
    c->path   = path;
    return true;
}

void llm_client_free(llm_client * c) {
    delete c;
}

// Request body: the conversation plus the streaming switch.
static std::string llm_client_body(const llm_client * c, const std::vector<llm_message> & messages) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    if (!c->params.model.empty()) {
        yyjson_mut_obj_add_str(doc, root, "model", c->params.model.c_str());
    }
    yyjson_mut_obj_add_bool(doc, root, "stream", true);

    const llm_sampling & s = c->params.sampling;
    if (s.temperature >= 0.0f) {
        yyjson_mut_obj_add_real(doc, root, "temperature", s.temperature);
    }
    if (s.top_p >= 0.0f) {
        yyjson_mut_obj_add_real(doc, root, "top_p", s.top_p);
    }
    if (s.top_k >= 0) {
        yyjson_mut_obj_add_int(doc, root, "top_k", s.top_k);
    }
    if (s.min_p >= 0.0f) {
        yyjson_mut_obj_add_real(doc, root, "min_p", s.min_p);
    }
    if (s.max_tokens > 0) {
        yyjson_mut_obj_add_int(doc, root, "max_tokens", s.max_tokens);
    }
    if (s.presence_penalty > -3.0f) {
        yyjson_mut_obj_add_real(doc, root, "presence_penalty", s.presence_penalty);
    }
    if (s.frequency_penalty > -3.0f) {
        yyjson_mut_obj_add_real(doc, root, "frequency_penalty", s.frequency_penalty);
    }
    if (s.seed >= 0) {
        yyjson_mut_obj_add_int(doc, root, "seed", s.seed);
    }
    if (!s.reasoning_effort.empty()) {
        yyjson_mut_obj_add_str(doc, root, "reasoning_effort", s.reasoning_effort.c_str());
    }

    yyjson_mut_val * array = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "messages", array);
    for (const llm_message & message : messages) {
        yyjson_mut_val * item = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, item, "role", message.role.c_str());
        yyjson_mut_obj_add_strcpy(doc, item, "content", message.content.c_str());
        yyjson_mut_arr_add_val(array, item);
    }

    char *            json = yyjson_mut_write(doc, 0, nullptr);
    const std::string body = json ? json : "";
    free(json);
    yyjson_mut_doc_free(doc);
    return body;
}

// The reason an endpoint gives with an error status: the error message of an
// OpenAI style body, or the first line of the body, cut to a readable length.
static std::string llm_client_reason(const std::string & body) {
    std::string  reason;
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (doc) {
        yyjson_val * error   = yyjson_obj_get(yyjson_doc_get_root(doc), "error");
        yyjson_val * message = error && yyjson_is_obj(error) ? yyjson_obj_get(error, "message") : error;
        if (message && yyjson_is_str(message)) {
            reason.assign(yyjson_get_str(message), yyjson_get_len(message));
        }
        yyjson_doc_free(doc);
    }
    if (reason.empty()) {
        reason = body.substr(0, body.find('\n'));
    }
    if (reason.size() > LLM_REASON_MAX) {
        reason.resize(LLM_REASON_MAX);
    }
    return reason;
}

// Pulls the content delta out of one data frame. Returns false when the
// frame carries nothing to say, which covers role only frames, usage frames
// and reasoning traces.
static bool llm_client_delta(const char * json, size_t size, std::string & delta) {
    yyjson_doc * doc = yyjson_read(json, size, 0);
    if (!doc) {
        return false;
    }

    bool found = false;

    yyjson_val * choices = yyjson_obj_get(yyjson_doc_get_root(doc), "choices");
    yyjson_val * choice  = choices ? yyjson_arr_get_first(choices) : nullptr;
    yyjson_val * message = choice ? yyjson_obj_get(choice, "delta") : nullptr;
    yyjson_val * content = message ? yyjson_obj_get(message, "content") : nullptr;

    if (content && yyjson_is_str(content)) {
        delta.assign(yyjson_get_str(content), yyjson_get_len(content));
        found = !delta.empty();
    }

    yyjson_doc_free(doc);
    return found;
}

bool llm_client_stream(llm_client *                     c,
                       const std::vector<llm_message> & messages,
                       llm_delta_cb                     cb,
                       void *                           user,
                       const std::atomic<bool> *        cancel,
                       std::string &                    text) {
    if (!c || messages.empty()) {
        s2s_set_error("[LLM] Client is NULL or the conversation is empty");
        return false;
    }

    httplib::Client & client = *c->http;
    if (!client.is_valid()) {
        s2s_set_error("[LLM] Cannot reach %s, https needs an OpenSSL build", c->host.c_str());
        return false;
    }
    client.set_read_timeout(c->params.timeout_sec, 0);
    client.set_write_timeout(c->params.timeout_sec, 0);

    httplib::Headers headers = {
        { "Accept", "text/event-stream" }
    };
    if (!c->params.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + c->params.api_key);
    }

    const std::string request = llm_client_body(c, messages);
    const std::string url     = c->path + "/chat/completions";

    text.clear();

    std::string body;  // everything received, the reason of an error status
    std::string pending;
    bool        cancelled = false;
    bool        done      = false;

    auto receiver = [&](const char * data, size_t size) {
        if (cancel && cancel->load()) {
            cancelled = true;
            return false;
        }

        body.append(data, size);
        pending.append(data, size);

        for (;;) {
            const size_t newline = pending.find('\n');
            if (newline == std::string::npos) {
                break;
            }

            std::string line = pending.substr(0, newline);
            pending.erase(0, newline + 1);
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                line.pop_back();
            }

            if (line.compare(0, 5, "data:") != 0) {
                continue;
            }
            size_t start = 5;
            while (start < line.size() && line[start] == ' ') {
                start++;
            }

            if (line.compare(start, std::string::npos, "[DONE]") == 0) {
                done = true;
                return false;
            }

            std::string delta;
            if (!llm_client_delta(line.c_str() + start, line.size() - start, delta)) {
                continue;
            }

            text += delta;
            if (cb && !cb(delta.c_str(), user)) {
                cancelled = true;
                return false;
            }
        }
        return true;
    };

    // The receiver only runs when bytes arrive: while the endpoint is silent,
    // during a prefill or before its headers, the watcher is what sees the
    // cancel flag, and it closes the socket under the request. It keeps
    // closing until the request returns, so a socket opened after the flag
    // rose is closed too.
    std::atomic<bool> finished(false);
    std::thread       watcher;
    if (cancel) {
        watcher = std::thread([&]() {
            while (!finished.load()) {
                if (cancel->load()) {
                    client.stop();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(LLM_CANCEL_POLL_MS));
            }
        });
    }

    httplib::Result result = client.Post(url.c_str(), headers, request, "application/json", receiver);

    finished.store(true);
    if (watcher.joinable()) {
        watcher.join();
    }

    if (cancelled || (cancel && cancel->load())) {
        s2s_set_error("[LLM] Cancelled");
        return false;
    }
    if (done) {
        return true;
    }
    if (!result) {
        s2s_set_error("[LLM] %s", httplib::to_string(result.error()).c_str());
        return false;
    }
    if (result->status < 200 || result->status >= 300) {
        // An error status carries no event stream: the body says why.
        s2s_set_error("[LLM] HTTP %d, %s", result->status, llm_client_reason(body).c_str());
        return false;
    }
    return true;
}

bool llm_client_models(const llm_client_params & params, std::vector<std::string> & models) {
    std::string host;
    std::string path;
    if (!llm_client_split_url(params.base_url, host, path)) {
        s2s_set_error("[LLM] Base URL '%s' has no scheme", params.base_url.c_str());
        return false;
    }

    httplib::Client client(host.c_str());
    if (!client.is_valid()) {
        s2s_set_error("[LLM] Cannot reach %s, https needs an OpenSSL build", host.c_str());
        return false;
    }
    client.set_read_timeout(params.timeout_sec, 0);

    httplib::Headers headers;
    if (!params.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + params.api_key);
    }

    httplib::Result result = client.Get((path + "/models").c_str(), headers);
    if (!result) {
        s2s_set_error("[LLM] %s", httplib::to_string(result.error()).c_str());
        return false;
    }
    if (result->status < 200 || result->status >= 300) {
        s2s_set_error("[LLM] HTTP %d, %s", result->status, llm_client_reason(result->body).c_str());
        return false;
    }

    yyjson_doc * doc = yyjson_read(result->body.c_str(), result->body.size(), 0);
    if (!doc) {
        s2s_set_error("[LLM] Model list is not JSON");
        return false;
    }

    models.clear();

    yyjson_val * data = yyjson_obj_get(yyjson_doc_get_root(doc), "data");
    if (data && yyjson_is_arr(data)) {
        size_t       index = 0;
        size_t       max   = 0;
        yyjson_val * item  = nullptr;
        yyjson_arr_foreach(data, index, max, item) {
            yyjson_val * id = yyjson_obj_get(item, "id");
            if (id && yyjson_is_str(id)) {
                models.emplace_back(yyjson_get_str(id), yyjson_get_len(id));
            }
        }
    }

    yyjson_doc_free(doc);
    return true;
}

const char * llm_client_last_error(void) {
    return s2s_last_error();
}
