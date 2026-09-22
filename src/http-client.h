#pragma once
// http-client.h: what every HTTP client of the server shares
//
// The endpoint client and the MCP client open the same kind of connection:
// one httplib::Client per host, kept alive across requests, that a cancelled
// request closes from a watching thread. This is that plumbing, and nothing
// about what travels over it.

#include "httplib.h"
#include "s2s-error.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#define HTTP_CANCEL_POLL_MS 10  // how often a request that receives nothing looks at the cancel flag

// Splits "http://host:port/v1" into the part httplib connects to and the
// prefix every request hangs off. False on a URL without a scheme.
static inline bool http_split_url(const std::string & url, std::string & host, std::string & path) {
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

// The HTTP client of one host, or none when httplib cannot make one: a port
// out of range, a scheme it does not speak, https on a build without TLS. It
// throws for some of them and hands back an empty client for the others;
// either way nothing past this point sees an unusable client. tag opens the
// error message.
static inline std::unique_ptr<httplib::Client> http_open(const std::string & host, const char * tag) {
    std::unique_ptr<httplib::Client> http;
    try {
        http = std::make_unique<httplib::Client>(host);
    } catch (const std::exception &) {
        http.reset();
    }
    if (!http || !http->is_valid()) {
        s2s_set_error("%s The URL cannot be reached: bad port, unknown scheme, or https without TLS", tag);
        return nullptr;
    }
    http->set_keep_alive(true);
    return http;
}

// Closes the socket under a request the caller cancels. The receiver of a
// stream only runs when bytes arrive, so while the server is silent, during
// a prefill or before its headers, this is what sees the flag. It keeps
// closing until the request returns, so a socket opened after the flag rose
// is closed too.
struct HttpCancelWatch {
    httplib::Client &         client;
    const std::atomic<bool> * cancel;
    std::atomic<bool>         finished{ false };
    std::thread               thread;

    HttpCancelWatch(httplib::Client & http, const std::atomic<bool> * flag) : client(http), cancel(flag) {
        if (!cancel) {
            return;
        }
        thread = std::thread([this]() {
            while (!finished.load()) {
                if (this->cancel->load()) {
                    this->client.stop();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(HTTP_CANCEL_POLL_MS));
            }
        });
    }

    ~HttpCancelWatch() {
        finished.store(true);
        if (thread.joinable()) {
            thread.join();
        }
    }
};
