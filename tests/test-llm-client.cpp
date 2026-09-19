// test-llm-client.cpp: streaming, sentence splitting and cancellation
//
// Runs a mock chat completions endpoint in process, so the harness measures
// the client and the splitter rather than somebody's GPU. The mock answers
// the way llama-server does: one data frame per token, a role only frame
// first, a usage frame at the end, and [DONE] to close.
//
// Four passes: a full stream cut into synthesis units and a second one on the
// same connection, a cancellation after
// a few deltas, an endpoint that answers 500 with a pretty printed body, and
// a cancellation while the endpoint says nothing yet. A pass feeds the splitter
// alone with accents and an emoji, one byte per delta, so every multibyte
// character is cut across two deltas, and another parses the endpoint
// settings of a session.update the way the server receives them.

#include "httplib.h"
#include "llm-client.h"
#include "realtime-proto.h"
#include "sentence-split.h"
#include "timer.h"
#include "version.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s [port]\n"
            "\n"
            "Starts a mock chat completions endpoint on 127.0.0.1 and streams\n"
            "through it. Default port: 18080.\n",
            prog);
}

// The answer the mock streams back, token by token, terminators included so
// the splitter has something to cut.
static const char * ANSWER =
    "Creation stories differ across cultures. Some describe a world born from water, "
    "others from a cosmic egg or from the body of a giant. The shared thread is an "
    "attempt to explain where everything came from, in the language of the time.";

#define SILENT_MS 2000  // how long the silent endpoint holds a request before answering
#define RAISE_MS  100   // when the silent request is cancelled

static std::vector<std::string> tokenize(const std::string & text) {
    std::vector<std::string> tokens;
    std::string              current;
    for (char c : text) {
        current += c;
        if (c == ' ') {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

static std::string frame(const std::string & content) {
    std::string escaped;
    for (char c : content) {
        if (c == '"' || c == '\\') {
            escaped += '\\';
        }
        escaped += c;
    }
    return "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + escaped + "\"}}]}\n\n";
}

struct Collector {
    SentenceSplitter          splitter;
    std::vector<SentenceUnit> units;
    int                       n_deltas     = 0;
    int                       cancel_after = 0;
    std::atomic<bool> *       cancel       = nullptr;
    Timer                     timer;
    double                    first_unit_ms = -1.0;
};

static bool on_delta(const char * text, void * user) {
    Collector * collector = (Collector *) user;
    collector->n_deltas++;

    for (const SentenceUnit & unit : sentence_split_push(&collector->splitter, text)) {
        if (collector->first_unit_ms < 0.0) {
            collector->first_unit_ms = collector->timer.ms();
        }
        collector->units.push_back(unit);
    }

    if (collector->cancel_after > 0 && collector->n_deltas >= collector->cancel_after) {
        collector->cancel->store(true);
    }
    return true;
}

int main(int argc, char ** argv) {
    int port = 18080;
    if (argc == 2) {
        port = atoi(argv[1]);
        if (port <= 0) {
            print_usage(argv[0]);
            return 1;
        }
    } else if (argc > 2) {
        print_usage(argv[0]);
        return 1;
    }

    httplib::Server mock;

    // The client port of every completion: one port for several requests
    // means they went over one kept alive connection.
    std::mutex    ports_mutex;
    std::set<int> ports;

    mock.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        {
            std::lock_guard<std::mutex> lock(ports_mutex);
            ports.insert(req.remote_port);
        }
        res.set_chunked_content_provider("text/event-stream", [](size_t, httplib::DataSink & sink) {
            const std::string role = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            sink.write(role.data(), role.size());

            for (const std::string & token : tokenize(ANSWER)) {
                const std::string chunk = frame(token);
                if (!sink.write(chunk.data(), chunk.size())) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            const std::string usage = "data: {\"choices\":[],\"usage\":{\"total_tokens\":64}}\n\n";
            sink.write(usage.data(), usage.size());

            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });

    // Pretty printed, the way a cloud API writes its errors: the reason sits
    // on a line of its own.
    mock.Post("/broken/chat/completions", [](const httplib::Request &, httplib::Response & res) {
        res.status = 500;
        res.set_content("{\n  \"error\": {\n    \"message\": \"model not loaded\"\n  }\n}\n", "application/json");
    });

    // Opens the stream with a 200, writes a little, then reports a failure
    // inside it, the way an endpoint does when the context overflows.
    mock.Post("/midstream/chat/completions", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(
            frame("Once upon ") + frame("a time") + "data: {\"error\":{\"message\":\"context size exceeded\"}}\n\n",
            "text/event-stream");
    });

    // Says nothing for a while, like an endpoint in a long prefill.
    mock.Post("/silent/chat/completions", [](const httplib::Request &, httplib::Response & res) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SILENT_MS));
        res.set_content("data: [DONE]\n\n", "text/event-stream");
    });

    std::thread server([&mock, port]() { mock.listen("127.0.0.1", port); });
    mock.wait_until_ready();

    const std::string base = "http://127.0.0.1:" + std::to_string(port);

    llm_client_params params;
    params.base_url = base + "/v1";
    params.model    = "mock";

    llm_client * client = llm_client_new(params);
    if (!client) {
        fprintf(stderr, "[LLM] FATAL: %s\n", llm_client_last_error());
        mock.stop();
        server.join();
        return 1;
    }

    const std::vector<llm_message> messages = {
        { "system", "You answer in two sentences."    },
        { "user",   "Tell me about creation stories." },
    };

    // Full stream.
    std::atomic<bool> cancel(false);
    Collector         collector;
    collector.cancel = &cancel;

    std::string  text;
    Timer        t_full;
    const bool   ok      = llm_client_stream(client, messages, on_delta, &collector, &cancel, text);
    const double full_ms = t_full.ms();

    if (!ok) {
        fprintf(stderr, "[LLM] FATAL: %s\n", llm_client_last_error());
        llm_client_free(client);
        mock.stop();
        server.join();
        return 1;
    }

    const SentenceUnit tail = sentence_split_flush(&collector.splitter);
    if (!tail.text.empty()) {
        collector.units.push_back(tail);
    }

    printf("[LLM] Streamed %d deltas, %zu characters, %zu units in %.1f ms\n", collector.n_deltas, text.size(),
           collector.units.size(), full_ms);
    printf("[LLM] First unit after %.1f ms\n", collector.first_unit_ms);
    printf("[LLM] Written %zu UTF-16 units, the last unit ends at %zu\n", sentence_utf16_len(text), tail.end);
    for (size_t i = 0; i < collector.units.size(); i++) {
        printf("[Unit] %zu: end %zu \"%s\"\n", i, collector.units[i].end, collector.units[i].text.c_str());
    }

    // A second completion on the same client, which reuses the connection.
    {
        std::string                 again;
        const bool                  ok_again = llm_client_stream(client, messages, nullptr, nullptr, &cancel, again);
        std::lock_guard<std::mutex> lock(ports_mutex);
        printf("[LLM] Two completions over %zu connections, returned %s\n", ports.size(), ok_again ? "true" : "false");
    }

    // Accents and an emoji, one byte at a time.
    {
        const std::string mixed = "D\xc3\xa9j\xc3\xa0 vu ? Oui \xf0\x9f\x98\x84. Fin";
        SentenceSplitter  split;
        for (const char c : mixed) {
            for (const SentenceUnit & unit : sentence_split_push(&split, std::string(1, c))) {
                printf("[Split] end %zu \"%s\"\n", unit.end, unit.text.c_str());
            }
        }
        const SentenceUnit last = sentence_split_flush(&split);
        printf("[Split] end %zu \"%s\"\n", last.end, last.text.c_str());
    }

    // The endpoint settings of a session.update, in the shape s2s.js sends.
    {
        const rt_client_message message = rt_parse(
            "{\"type\":\"session.update\",\"session\":{\"llm_timeout_sec\":30,\"sampling\":{"
            "\"temperature\":0.7,\"top_p\":0.9,\"top_k\":40,\"min_p\":0.05,\"max_tokens\":256,"
            "\"presence_penalty\":0.5,\"frequency_penalty\":0.25,\"seed\":42,\"reasoning_effort\":\"low\"}}}");
        const rt_session_patch & p = message.patch;
        printf(
            "[Parse] temperature %g top_p %g top_k %g min_p %g max_tokens %g presence %g frequency %g seed %g "
            "timeout %g reasoning %s\n",
            (double) p.temperature, (double) p.top_p, (double) p.top_k, (double) p.min_p, (double) p.max_tokens,
            (double) p.presence_penalty, (double) p.frequency_penalty, (double) p.seed, (double) p.llm_timeout_sec,
            p.reasoning_effort.c_str());
    }

    // Session updates the server refuses: a duration past its range, and a
    // number that is not one. Both read as absent, the first one met named.
    {
        const rt_client_message message = rt_parse(
            "{\"type\":\"session.update\",\"session\":{\"vad\":{\"min_speech_ms\":1e9},"
            "\"sampling\":{\"temperature\":\"hot\"}}}");
        printf("[Parse] refused %s, min_speech_ms %d, temperature %g\n", message.patch.invalid.c_str(),
               message.patch.min_speech_ms, (double) message.patch.temperature);
    }

    // URLs no client can reach: a bad port, a scheme httplib does not speak.
    // A client pointed at one keeps the endpoint it had.
    {
        int refused = 0;
        for (const char * url : { "http://127.0.0.1:65536/v1", "ftp://127.0.0.1/v1" }) {
            llm_client_params bad = params;
            bad.base_url          = url;
            refused += llm_client_new(bad) == nullptr;
            refused += !llm_client_set_params(client, bad);
        }
        std::string again;
        const bool  kept = llm_client_stream(client, messages, nullptr, nullptr, &cancel, again);
        printf("[LLM] Unreachable URLs refused %d of 4, the client still answers: %s\n", refused,
               kept ? "true" : "false");
    }

    // Cancellation after three deltas.
    cancel.store(false);
    Collector interrupted;
    interrupted.cancel       = &cancel;
    interrupted.cancel_after = 3;

    std::string partial;
    const bool  spoke = llm_client_stream(client, messages, on_delta, &interrupted, &cancel, partial);
    printf("[LLM] Cancelled after %d deltas, %zu characters, returned %s\n", interrupted.n_deltas, partial.size(),
           spoke ? "true" : "false");

    // Endpoint failure.
    llm_client_params broken_params = params;
    broken_params.base_url          = base + "/broken";
    llm_client * broken             = llm_client_new(broken_params);

    std::string nothing;
    const bool  answered = llm_client_stream(broken, messages, nullptr, nullptr, nullptr, nothing);
    printf("[LLM] Broken endpoint returned %s: %s\n", answered ? "true" : "false", llm_client_last_error());

    // A failure reported inside a stream that opened fine.
    {
        llm_client_params midstream_params = params;
        midstream_params.base_url          = base + "/midstream";
        llm_client * midstream             = llm_client_new(midstream_params);
        std::string  partial_answer;
        const bool   streamed = llm_client_stream(midstream, messages, nullptr, nullptr, nullptr, partial_answer);
        printf("[LLM] Midstream error returned %s after %zu characters: %s\n", streamed ? "true" : "false",
               partial_answer.size(), llm_client_last_error());
        llm_client_free(midstream);
    }

    // Cancellation while the endpoint says nothing yet.
    llm_client_params silent_params = params;
    silent_params.base_url          = base + "/silent";
    llm_client * silent             = llm_client_new(silent_params);

    std::atomic<bool> raised(false);
    std::thread       raiser([&raised]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(RAISE_MS));
        raised.store(true);
    });
    std::string       quiet;
    Timer             t_silent;
    const bool        heard     = llm_client_stream(silent, messages, nullptr, nullptr, &raised, quiet);
    const double      silent_ms = t_silent.ms();
    raiser.join();
    printf("[LLM] Silent endpoint cancelled after %.1f ms of %d, returned %s\n", silent_ms, SILENT_MS,
           heard ? "true" : "false");

    llm_client_free(silent);
    llm_client_free(broken);
    llm_client_free(client);
    mock.stop();
    server.join();
    return 0;
}
