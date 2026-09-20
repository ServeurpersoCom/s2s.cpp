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

// The HTTP client of one endpoint host, or none when httplib cannot make one:
// a port out of range, a scheme it does not speak, https on a build without
// TLS. It throws for some of them and hands back an empty client for the
// others; either way nothing past this point sees an unusable client.
static std::unique_ptr<httplib::Client> llm_client_http(const std::string & host) {
    std::unique_ptr<httplib::Client> http;
    try {
        http = std::make_unique<httplib::Client>(host);
    } catch (const std::exception &) {
        http.reset();
    }
    if (!http || !http->is_valid()) {
        s2s_set_error("[LLM] The endpoint URL cannot be reached: bad port, unknown scheme, or https without TLS");
        return nullptr;
    }
    return http;
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
        s2s_set_error("[LLM] The endpoint URL has no scheme");
        return false;
    }
    if (!c->http || host != c->host) {
        std::unique_ptr<httplib::Client> http = llm_client_http(host);
        if (!http) {
            return false;
        }
        http->set_keep_alive(true);
        c->http = std::move(http);
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

// What one data frame says: its content delta, the length of its reasoning
// trace, which is never spoken, whether it reports a failure, an endpoint
// can do so inside a stream it opened with a 200, and the finish_reason that
// ends the generation. Role only frames and usage frames say none of it.
struct LlmFrame {
    std::string delta;
    size_t      reasoning = 0;
    std::string finish;
    bool        failed = false;
};

static LlmFrame llm_client_frame(const char * json, size_t size) {
    LlmFrame     frame;
    yyjson_doc * doc = yyjson_read(json, size, 0);
    if (!doc) {
        return frame;
    }

    yyjson_val * root    = yyjson_doc_get_root(doc);
    yyjson_val * choices = yyjson_obj_get(root, "choices");
    yyjson_val * choice  = choices ? yyjson_arr_get_first(choices) : nullptr;
    yyjson_val * reason  = choice ? yyjson_obj_get(choice, "finish_reason") : nullptr;
    yyjson_val * message = choice ? yyjson_obj_get(choice, "delta") : nullptr;
    yyjson_val * content = message ? yyjson_obj_get(message, "content") : nullptr;

    frame.failed = yyjson_obj_get(root, "error") != nullptr;
    if (reason && yyjson_is_str(reason)) {
        frame.finish.assign(yyjson_get_str(reason), yyjson_get_len(reason));
    }
    if (content && yyjson_is_str(content)) {
        frame.delta.assign(yyjson_get_str(content), yyjson_get_len(content));
    }
    // llama.cpp and most engines name the trace reasoning_content, some
    // reasoning.
    for (const char * key : { "reasoning_content", "reasoning" }) {
        yyjson_val * trace = message ? yyjson_obj_get(message, key) : nullptr;
        if (trace && yyjson_is_str(trace)) {
            frame.reasoning += yyjson_get_len(trace);
        }
    }

    yyjson_doc_free(doc);
    return frame;
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
    // Reaching the endpoint is waiting on it too: an unreachable host fails
    // within the timeout of the session, not the minutes of the library.
    client.set_connection_timeout(c->params.timeout_sec, 0);
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

    std::string body;     // everything received, the reason of an error status
    std::string pending;
    std::string failure;  // the reason of an error frame inside the stream
    bool        cancelled = false;
    bool        done      = false;
    std::string finish;         // the finish_reason of the generation, empty until a frame gives it
    size_t      reasoning = 0;  // bytes of reasoning trace received and dropped

    // [DONE] ends the answer, not the request: what follows it is left for
    // httplib to read to the end of the body, so the connection stays open
    // for the next turn. Returning false aborts the request and closes it,
    // which only a cancellation does.
    auto receiver = [&](const char * data, size_t size) {
        if (done) {
            return true;
        }
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
                return true;
            }

            const LlmFrame frame = llm_client_frame(line.c_str() + start, line.size() - start);
            if (frame.failed) {
                failure = llm_client_reason(line.substr(start));
                return false;
            }
            if (!frame.finish.empty()) {
                finish = frame.finish;
            }
            reasoning += frame.reasoning;
            if (frame.delta.empty()) {
                continue;
            }

            text += frame.delta;
            if (cb && !cb(frame.delta.c_str(), user)) {
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

    if (!failure.empty()) {
        s2s_set_error("[LLM] %s", failure.c_str());
        return false;
    }
    if (cancelled || (cancel && cancel->load())) {
        s2s_set_error("[LLM] Cancelled");
        return false;
    }
    // Every answer that ends says what came back, an empty one included, so
    // the log explains a silence the voice alone cannot.
    auto answered = [&]() {
        s2s_log(S2S_LOG_INFO, "[LLM] Answer of %zu bytes, %zu bytes of reasoning dropped, %s%s", text.size(), reasoning,
                finish.empty() ? "no finish_reason" : "finish_reason ", finish.c_str());
        return true;
    };
    if (done) {
        return answered();
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
    // A stream that closes without [DONE] or a finish_reason was cut short:
    // what came is not the whole answer.
    if (finish.empty()) {
        s2s_set_error("[LLM] The stream ended before the answer did");
        return false;
    }
    return answered();
}

bool llm_client_models(const llm_client_params & params, std::vector<std::string> & models) {
    std::string host;
    std::string path;
    if (!llm_client_split_url(params.base_url, host, path)) {
        s2s_set_error("[LLM] The endpoint URL has no scheme");
        return false;
    }

    std::unique_ptr<httplib::Client> http = llm_client_http(host);
    if (!http) {
        return false;
    }
    httplib::Client & client = *http;
    client.set_connection_timeout(params.timeout_sec, 0);
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
