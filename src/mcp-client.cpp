// mcp-client.cpp: JSON-RPC 2.0 over Streamable HTTP
//
// Every exchange is one POST of one message. A request carries an id and
// reads its response out of the body, which the server writes either as a
// JSON message or as a server sent event stream whose events carry messages;
// a notification carries no id and gets a 202 with nothing to read. What the
// server sends besides the response, notifications of progress or requests of
// its own, is skipped: the voice waits for one thing.

#include "mcp-client.h"

#include "http-client.h"
#include "version.h"
#include "yyjson.h"

#define MCP_REASON_MAX 200  // characters of a server's error reason kept in the message

// The versions a server may answer initialize with: the list of the SDK.
static const char * const MCP_SUPPORTED_VERSIONS[] = { MCP_PROTOCOL_VERSION, "2025-06-18", "2025-03-26", "2024-11-05",
                                                       "2024-10-07" };

struct mcp_client {
    mcp_server_params params;

    std::string                      host;  // scheme, host and port, what httplib::Client takes
    std::string                      path;  // the endpoint, what every POST goes to
    std::unique_ptr<httplib::Client> http;  // kept alive across requests

    std::string session;                    // Mcp-Session-Id the server gave, empty on a server that keeps none
    std::string version;                    // protocol version agreed on, empty before the handshake
    std::string server_name;
    int64_t     next_id = 1;                // id of the next request
    bool        open    = false;
};

static std::string mcp_text(yyjson_val * value) {
    return value && yyjson_is_str(value) ? std::string(yyjson_get_str(value), yyjson_get_len(value)) : std::string();
}

// The reason a server gives with an error status: the message of a JSON-RPC
// error body, or the first line of the body, cut to a readable length.
static std::string mcp_reason(const std::string & body) {
    std::string  reason;
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (doc) {
        yyjson_val * error = yyjson_obj_get(yyjson_doc_get_root(doc), "error");
        reason             = mcp_text(error && yyjson_is_obj(error) ? yyjson_obj_get(error, "message") : error);
        yyjson_doc_free(doc);
    }
    if (reason.empty()) {
        reason = body.substr(0, body.find('\n'));
    }
    if (reason.size() > MCP_REASON_MAX) {
        reason.resize(MCP_REASON_MAX);
    }
    return reason;
}

// One message under construction: the envelope with its method, and the
// params object every field goes into. mcp_message_end writes it and frees
// the document. A request gets the id it was given, a notification none.
struct McpMessage {
    yyjson_mut_doc * doc    = nullptr;
    yyjson_mut_val * params = nullptr;
};

static McpMessage mcp_message_begin(const char * method, int64_t id) {
    McpMessage message;
    message.doc           = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(message.doc);
    yyjson_mut_doc_set_root(message.doc, root);
    yyjson_mut_obj_add_str(message.doc, root, "jsonrpc", "2.0");
    if (id > 0) {
        yyjson_mut_obj_add_sint(message.doc, root, "id", id);
    }
    yyjson_mut_obj_add_str(message.doc, root, "method", method);
    message.params = yyjson_mut_obj_add_obj(message.doc, root, "params");
    return message;
}

static std::string mcp_message_end(McpMessage & message) {
    char *      json = yyjson_mut_write(message.doc, 0, nullptr);
    std::string out  = json ? json : "{}";
    free(json);
    yyjson_mut_doc_free(message.doc);
    return out;
}

// Is this message the response to request id: an object whose id is that
// number. Responses of a server keep the type of the id they answer.
static bool mcp_answers(yyjson_val * message, int64_t id) {
    yyjson_val * value = yyjson_is_obj(message) ? yyjson_obj_get(message, "id") : nullptr;
    return value && yyjson_is_int(value) && yyjson_get_sint(value) == id;
}

// The response to request id in a JSON body: the body itself when it is that
// message, the member of a batch that is.
static bool mcp_json_response(const std::string & body, int64_t id, std::string & message) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root  = yyjson_doc_get_root(doc);
    bool         found = false;
    if (mcp_answers(root, id)) {
        message = body;
        found   = true;
    } else if (yyjson_is_arr(root)) {
        size_t       index = 0;
        size_t       max   = 0;
        yyjson_val * item  = nullptr;
        yyjson_arr_foreach(root, index, max, item) {
            if (mcp_answers(item, id)) {
                char * json = yyjson_val_write(item, 0, nullptr);
                message     = json ? json : "";
                free(json);
                found = true;
                break;
            }
        }
    }
    yyjson_doc_free(doc);
    return found;
}

// The response to request id in an event stream: the data of the message
// event that carries it. Events are blocks of lines ended by an empty line;
// the data lines of one event join with newlines; an event of another name,
// a comment, a keep alive and an id or retry field are skipped.
static bool mcp_sse_response(const std::string & body, int64_t id, std::string & message) {
    std::string data;
    std::string event;
    size_t      start = 0;
    bool        found = false;

    const auto dispatch = [&]() {
        if (!data.empty() && (event.empty() || event == "message")) {
            yyjson_doc * doc = yyjson_read(data.c_str(), data.size(), 0);
            if (doc) {
                if (mcp_answers(yyjson_doc_get_root(doc), id)) {
                    message = data;
                    found   = true;
                }
                yyjson_doc_free(doc);
            }
        }
        data.clear();
        event.clear();
    };

    while (start < body.size() && !found) {
        size_t end = body.find('\n', start);
        if (end == std::string::npos) {
            end = body.size();
        }
        std::string line = body.substr(start, end - start);
        start            = end + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            dispatch();
        } else if (line.compare(0, 5, "data:") == 0) {
            const size_t at = line.size() > 5 && line[5] == ' ' ? 6 : 5;
            data += data.empty() ? "" : "\n";
            data += line.substr(at);
        } else if (line.compare(0, 6, "event:") == 0) {
            const size_t at = line.size() > 6 && line[6] == ' ' ? 7 : 6;
            event           = line.substr(at);
        }
    }
    if (!found) {
        dispatch();
    }
    return found;
}

static bool mcp_client_open_session(mcp_client * c, const std::atomic<bool> * cancel);

// Posts one message and reads what answers it. A request, id above zero,
// ends with its response in message; a notification ends with a 2xx and
// nothing to read. Once, on a 404 for a session the server no longer knows,
// the session is opened again and the message sent again.
static bool mcp_exchange(mcp_client *              c,
                         const std::string &       request,
                         int64_t                   id,
                         const std::atomic<bool> * cancel,
                         std::string &             message,
                         bool                      retry = true) {
    httplib::Client & client = *c->http;
    client.set_connection_timeout(c->params.timeout_sec, 0);
    client.set_read_timeout(c->params.timeout_sec, 0);
    client.set_write_timeout(c->params.timeout_sec, 0);

    httplib::Headers headers = {
        { "Accept", "application/json, text/event-stream" }
    };
    if (!c->params.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + c->params.api_key);
    }
    if (!c->session.empty()) {
        headers.emplace("Mcp-Session-Id", c->session);
    }
    if (!c->version.empty()) {
        headers.emplace("MCP-Protocol-Version", c->version);
    }

    httplib::Result response;
    {
        HttpCancelWatch watch(client, cancel);
        response = client.Post(c->path.empty() ? "/" : c->path.c_str(), headers, request, "application/json");
    }

    if (cancel && cancel->load()) {
        s2s_set_error("[MCP] Cancelled");
        return false;
    }
    if (!response) {
        s2s_set_error("[MCP] %s", http_error(response).c_str());
        return false;
    }
    if (response->has_header("Mcp-Session-Id")) {
        c->session = response->get_header_value("Mcp-Session-Id");
    }
    if (response->status == 404 && !c->session.empty() && retry) {
        s2s_log(S2S_LOG_WARN, "[MCP] The server forgot the session, opening a new one");
        c->session.clear();
        c->open = false;
        if (!mcp_client_open_session(c, cancel)) {
            return false;
        }
        return mcp_exchange(c, request, id, cancel, message, false);
    }
    if (response->status < 200 || response->status >= 300) {
        s2s_set_error("[MCP] HTTP %d, %s", response->status, mcp_reason(response->body).c_str());
        return false;
    }

    message.clear();
    if (id <= 0 || response->status == 202) {
        return true;
    }
    const std::string type  = response->get_header_value("Content-Type");
    const bool        found = type.find("text/event-stream") != std::string::npos ?
                                  mcp_sse_response(response->body, id, message) :
                                  mcp_json_response(response->body, id, message);
    if (!found) {
        s2s_set_error("[MCP] The server answered without the response to request %lld", (long long) id);
        return false;
    }
    return true;
}

// Sends a request and hands back the result of its response, in a document
// the caller frees. A response that carries an error fails with its message.
static bool mcp_request(mcp_client *              c,
                        McpMessage &              request,
                        int64_t                   id,
                        const std::atomic<bool> * cancel,
                        yyjson_doc *&             doc,
                        yyjson_val *&             result) {
    const std::string body = mcp_message_end(request);
    std::string       message;
    if (!mcp_exchange(c, body, id, cancel, message)) {
        return false;
    }
    doc = yyjson_read(message.c_str(), message.size(), 0);
    if (!doc) {
        s2s_set_error("[MCP] The response is not JSON");
        return false;
    }
    yyjson_val * root  = yyjson_doc_get_root(doc);
    yyjson_val * error = yyjson_obj_get(root, "error");
    if (error) {
        const std::string reason = mcp_text(yyjson_obj_get(error, "message"));
        yyjson_val *      code   = yyjson_obj_get(error, "code");
        s2s_set_error("[MCP] %s (%lld)", reason.c_str(),
                      code && yyjson_is_int(code) ? (long long) yyjson_get_sint(code) : 0LL);
        yyjson_doc_free(doc);
        doc = nullptr;
        return false;
    }
    result = yyjson_obj_get(root, "result");
    if (!result) {
        s2s_set_error("[MCP] The response has no result");
        yyjson_doc_free(doc);
        doc = nullptr;
        return false;
    }
    return true;
}

mcp_client * mcp_client_new(const mcp_server_params & params) {
    std::string host;
    std::string path;
    if (!http_split_url(params.url, host, path)) {
        s2s_set_error("[MCP] The server URL has no scheme");
        return nullptr;
    }
    std::unique_ptr<httplib::Client> http = http_open(host, "[MCP]");
    if (!http) {
        return nullptr;
    }
    mcp_client * c = new mcp_client();
    c->params      = params;
    c->host        = host;
    c->path        = path;
    c->http        = std::move(http);
    return c;
}

void mcp_client_free(mcp_client * c) {
    if (!c) {
        return;
    }
    // A server that keeps sessions ends this one on DELETE; one that keeps
    // none answers 405, which is its way of saying there was nothing to end.
    if (!c->session.empty()) {
        httplib::Headers headers = {
            { "Mcp-Session-Id", c->session }
        };
        if (!c->params.api_key.empty()) {
            headers.emplace("Authorization", "Bearer " + c->params.api_key);
        }
        if (!c->version.empty()) {
            headers.emplace("MCP-Protocol-Version", c->version);
        }
        c->http->set_connection_timeout(c->params.timeout_sec, 0);
        c->http->set_read_timeout(c->params.timeout_sec, 0);
        c->http->Delete(c->path.empty() ? "/" : c->path.c_str(), headers);
    }
    delete c;
}

// The handshake: initialize, then the notification that the client is ready.
static bool mcp_client_open_session(mcp_client * c, const std::atomic<bool> * cancel) {
    const int64_t id      = c->next_id++;
    McpMessage    request = mcp_message_begin("initialize", id);
    yyjson_mut_obj_add_str(request.doc, request.params, "protocolVersion", MCP_PROTOCOL_VERSION);
    yyjson_mut_obj_add_obj(request.doc, request.params, "capabilities");
    yyjson_mut_val * client = yyjson_mut_obj_add_obj(request.doc, request.params, "clientInfo");
    yyjson_mut_obj_add_str(request.doc, client, "name", "s2s.cpp");
    yyjson_mut_obj_add_str(request.doc, client, "version", S2S_VERSION);

    yyjson_doc * doc    = nullptr;
    yyjson_val * result = nullptr;
    if (!mcp_request(c, request, id, cancel, doc, result)) {
        return false;
    }
    const std::string version = mcp_text(yyjson_obj_get(result, "protocolVersion"));
    yyjson_val *      info    = yyjson_obj_get(result, "serverInfo");
    c->server_name            = mcp_text(info ? yyjson_obj_get(info, "name") : nullptr);
    yyjson_doc_free(doc);

    bool supported = false;
    for (const char * candidate : MCP_SUPPORTED_VERSIONS) {
        supported = supported || version == candidate;
    }
    if (!supported) {
        s2s_set_error("[MCP] The server speaks protocol version %s, which this client does not", version.c_str());
        return false;
    }
    c->version = version;

    McpMessage        initialized = mcp_message_begin("notifications/initialized", 0);
    const std::string body        = mcp_message_end(initialized);
    std::string       none;
    if (!mcp_exchange(c, body, 0, cancel, none)) {
        return false;
    }
    c->open = true;
    s2s_log(S2S_LOG_INFO, "[MCP] Session with %s, protocol %s%s", c->server_name.c_str(), c->version.c_str(),
            c->session.empty() ? "" : ", session id kept");
    return true;
}

bool mcp_client_open(mcp_client * c, const std::atomic<bool> * cancel) {
    if (!c) {
        s2s_set_error("[MCP] Client is NULL");
        return false;
    }
    return c->open || mcp_client_open_session(c, cancel);
}

const char * mcp_client_server_name(const mcp_client * c) {
    return c ? c->server_name.c_str() : "";
}

const char * mcp_client_url(const mcp_client * c) {
    return c ? c->params.url.c_str() : "";
}

// One tool of a list, as the definition a request to the model carries.
static llm_tool mcp_tool_definition(yyjson_val * tool) {
    llm_tool out;
    out.name              = mcp_text(yyjson_obj_get(tool, "name"));
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "type", "function");
    yyjson_mut_val * function = yyjson_mut_obj_add_obj(doc, root, "function");
    yyjson_mut_obj_add_strn(doc, function, "name", out.name.c_str(), out.name.size());
    yyjson_val * description = yyjson_obj_get(tool, "description");
    if (description && yyjson_is_str(description)) {
        yyjson_mut_obj_add_strn(doc, function, "description", yyjson_get_str(description), yyjson_get_len(description));
    }
    yyjson_val * schema = yyjson_obj_get(tool, "inputSchema");
    yyjson_mut_obj_add_val(doc, function, "parameters",
                           schema ? yyjson_val_mut_copy(doc, schema) : yyjson_mut_obj(doc));
    char * json    = yyjson_mut_write(doc, 0, nullptr);
    out.definition = json ? json : "";
    free(json);
    yyjson_mut_doc_free(doc);
    return out;
}

bool mcp_client_tools(mcp_client * c, const std::atomic<bool> * cancel, std::vector<llm_tool> & tools) {
    if (!mcp_client_open(c, cancel)) {
        return false;
    }
    tools.clear();
    std::string cursor;
    do {
        const int64_t id      = c->next_id++;
        McpMessage    request = mcp_message_begin("tools/list", id);
        if (!cursor.empty()) {
            yyjson_mut_obj_add_strn(request.doc, request.params, "cursor", cursor.c_str(), cursor.size());
        }
        yyjson_doc * doc    = nullptr;
        yyjson_val * result = nullptr;
        if (!mcp_request(c, request, id, cancel, doc, result)) {
            return false;
        }
        yyjson_val * list  = yyjson_obj_get(result, "tools");
        size_t       index = 0;
        size_t       max   = 0;
        yyjson_val * tool  = nullptr;
        yyjson_arr_foreach(list, index, max, tool) {
            llm_tool definition = mcp_tool_definition(tool);
            if (!definition.name.empty()) {
                tools.push_back(definition);
            }
        }
        cursor = mcp_text(yyjson_obj_get(result, "nextCursor"));
        yyjson_doc_free(doc);
    } while (!cursor.empty());
    s2s_log(S2S_LOG_INFO, "[MCP] %zu tools from %s", tools.size(), c->server_name.c_str());
    return true;
}

bool mcp_client_call(mcp_client *              c,
                     const std::string &       name,
                     const std::string &       arguments,
                     const std::atomic<bool> * cancel,
                     std::string &             result) {
    if (!mcp_client_open(c, cancel)) {
        return false;
    }
    const int64_t id      = c->next_id++;
    McpMessage    request = mcp_message_begin("tools/call", id);
    yyjson_mut_obj_add_strn(request.doc, request.params, "name", name.c_str(), name.size());
    yyjson_mut_val * written = arguments.empty() ? nullptr : yyjson_mut_rawcpy(request.doc, arguments.c_str());
    yyjson_mut_obj_add_val(request.doc, request.params, "arguments", written ? written : yyjson_mut_obj(request.doc));

    yyjson_doc * doc   = nullptr;
    yyjson_val * value = nullptr;
    if (!mcp_request(c, request, id, cancel, doc, value)) {
        return false;
    }

    // The text of the content, item by item; a tool that speaks in structure
    // alone is handed over as that structure.
    result.clear();
    yyjson_val * content = yyjson_obj_get(value, "content");
    size_t       index   = 0;
    size_t       max     = 0;
    yyjson_val * item    = nullptr;
    yyjson_arr_foreach(content, index, max, item) {
        if (mcp_text(yyjson_obj_get(item, "type")) == "text") {
            result += result.empty() ? "" : "\n";
            result += mcp_text(yyjson_obj_get(item, "text"));
        }
    }
    yyjson_val * structured = yyjson_obj_get(value, "structuredContent");
    if (result.empty() && structured) {
        char * json = yyjson_val_write(structured, 0, nullptr);
        result      = json ? json : "";
        free(json);
    }
    yyjson_val * failed = yyjson_obj_get(value, "isError");
    if (failed && yyjson_is_true(failed)) {
        s2s_log(S2S_LOG_WARN, "[MCP] %s reported an error: %s", name.c_str(), result.substr(0, MCP_REASON_MAX).c_str());
    }
    yyjson_doc_free(doc);
    return true;
}

const char * mcp_client_last_error(void) {
    return s2s_last_error();
}
