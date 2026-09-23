// test-server.cpp: one scripted conversation over the Realtime WebSocket
//
// Plays a script into a running s2s-server the way a browser would: 24 kHz
// PCM16, base64, one 20 ms frame at a time on a fixed clock, so the turn
// detection sees the silences the script asks for. Every server event, every
// step of the script and every request the mock endpoint serves is printed on
// one line with its time, and test-server.py judges the timeline.
//
// The script is a list of steps:
//
//   say:A-B    the passage of the WAV from A to B seconds
//   pause:S    S seconds of silence
//   commit     input_audio_buffer.commit, the end of a push to talk
//   mute:S     S seconds without a frame, the microphone off
//   heard:MS   silence until MS ms of the answer have played, 20 s at most
//
// Once the script is done, the client settles: it streams silence while a
// response is open or its playback lasts, and until the server has been
// quiet for a moment, so every scenario ends on a closed answer whatever the
// speed of the machine.
//
// The client is a browser on a loudspeaker: what the server sends plays in
// real time, and a speech_started drops what has not played yet. With --room
// the microphone also hears that playback back through a room: delayed,
// smeared by a decaying response, added to the voice.
//
// With --llm the client runs a mock chat completions endpoint in process and
// names it in its session, so the latency of the model is part of the script.
// With --llm-url the session names an endpoint of its own instead, a real
// one or an address nothing answers, and --llm-timeout bounds the wait on it.
// Its route picks its behavior: v1 answers, broken fails with an HTTP 500,
// midstream reports a failure inside the stream, empty thinks and then ends
// the generation without a word, agent calls the clock tool of the mock MCP
// server and then speaks its result, or says the clock did not answer when
// the call comes back as an error, and voice calls the built-in set_voice
// before it speaks with the voice it picked.

#include "audio-resample.h"
#include "httplib.h"
#include "realtime-proto.h"
#include "timer.h"
#include "version.h"
#include "wav.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define FRAME_MS 20

#define ROOM_DELAY_MS  120     // loudspeaker to microphone
#define ROOM_TAIL_MS   50      // decaying room response
#define ROOM_ECHO_GAIN 0.8f    // the echo about as loud as the voice

#define HEARD_WAIT_SEC   20.0  // longest wait for the answer a heard step expects
#define SETTLE_QUIET_SEC 1.5   // server silence that ends the settling
#define SETTLE_MAX_SEC   30.0  // longest settling

// The answer of the mock endpoint, streamed word by word: several sentences,
// so the voice speaks for a while, one unit per sentence.
static const char * MOCK_ANSWER =
    "Sure, here is a short answer. It has a few sentences, so the voice speaks for a while. "
    "Each sentence is a unit of its own. That is all for now.";

// The voice the voice route asks set_voice for: the timbre only way of the
// default voice, which speaks with its reference speech.
#define MOCK_VOICE "freeman (timbre only)"

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <url> <wav> [options] <step>...\n"
            "\n"
            "Steps: say:A-B, pause:S, commit, mute:S, heard:MS\n"
            "\n"
            "Options:\n"
            "  --mode <mode>          loopback or conversation (default: loopback)\n"
            "  --echo <method>        both, server, client or off (default: off)\n"
            "  --room                 the playback reaches the microphone through a room\n"
            "  --llm <route>          runs the mock endpoint and names it: v1, broken, midstream, empty,\n"
            "                         agent, which calls the clock tool once before it answers,\n"
            "                         voice, which calls set_voice once before it answers\n"
            "  --mcp                  names the mock MCP server, which runs the clock tool\n"
            "  --mcp-delay-ms <N>     mock delay before the clock answers (default: 0)\n"
            "  --tool-timeout <s>     the session timeout of one tool call\n"
            "  --llm-first-ms <N>     mock delay before the first word (default: 100)\n"
            "  --llm-token-ms <N>     mock delay between words (default: 20)\n"
            "  --llm-url <url>        names this endpoint instead of the mock\n"
            "  --llm-timeout <s>      the session timeout of the endpoint\n"
            "  --other-endpoint       names an endpoint and a key of its own\n",
            prog);
}

// One step of the script, with its argument when it takes one.
struct Step {
    std::string name;
    double      a = 0.0;
    double      b = 0.0;
};

static bool parse_step(const char * arg, Step & step) {
    const char * colon = strchr(arg, ':');
    step.name          = colon ? std::string(arg, (size_t) (colon - arg)) : std::string(arg);
    if (step.name == "commit") {
        return colon == nullptr;
    }
    if (!colon) {
        return false;
    }
    if (step.name == "say") {
        return sscanf(colon + 1, "%lf-%lf", &step.a, &step.b) == 2 && step.b > step.a;
    }
    step.a = atof(colon + 1);
    return step.name == "pause" || step.name == "mute" || step.name == "heard";
}

// The loudspeaker: what the server sent and has not played yet, everything it
// played, which the room turns into echo, and the current run of playback.
// The reader also notes here whether a response is open and when the server
// last spoke, which the settling reads.
struct Speaker {
    std::mutex         mutex;
    std::deque<float>  queue;
    std::vector<float> played;
    int                run_ms        = 0;
    bool               open_response = false;
    double             last_event    = 0.0;
};

static std::vector<std::string> mock_words(const std::string & text) {
    std::vector<std::string> words;
    std::string              current;
    for (char c : text) {
        current += c;
        if (c == ' ') {
            words.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

static std::string mock_frame(const std::string & content) {
    yyjson_mut_doc * doc     = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root    = yyjson_mut_obj(doc);
    yyjson_mut_val * choices = yyjson_mut_obj_add_arr(doc, root, "choices");
    yyjson_mut_val * choice  = yyjson_mut_arr_add_obj(doc, choices);
    yyjson_mut_obj_add_int(doc, choice, "index", 0);
    yyjson_mut_val * delta = yyjson_mut_obj_add_obj(doc, choice, "delta");
    yyjson_mut_obj_add_strn(doc, delta, "content", content.c_str(), content.size());
    yyjson_mut_doc_set_root(doc, root);
    char *      json  = yyjson_mut_write(doc, 0, nullptr);
    std::string frame = std::string("data: ") + (json ? json : "{}") + "\n\n";
    free(json);
    yyjson_mut_doc_free(doc);
    return frame;
}

// The mock MCP server: one JSON-RPC message per POST, the way the reference
// SDK answers, a session id given at initialize and demanded after it, the
// tool list as an event stream and the call as plain JSON, a bearer key on
// every request.
static void mock_mcp(httplib::Server & mock, Timer & timer, int delay_ms) {
    mock.Post("/mcp", [&timer, delay_ms](const httplib::Request & req, httplib::Response & res) {
        yyjson_doc *  doc    = yyjson_read(req.body.c_str(), req.body.size(), 0);
        yyjson_val *  root   = doc ? yyjson_doc_get_root(doc) : nullptr;
        std::string   method = rt_json_str(root, "method");
        yyjson_val *  value  = root ? yyjson_obj_get(root, "id") : nullptr;
        const int64_t id     = value && yyjson_is_int(value) ? yyjson_get_sint(value) : 0;
        yyjson_doc_free(doc);
        printf("[Mock] %7.2fs  mcp      %s id=%lld\n", timer.ms() / 1000.0, method.c_str(), (long long) id);
        fflush(stdout);

        if (req.get_header_value("Authorization") != "Bearer mock-key") {
            res.status = 401;
            res.set_content(
                "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32000,\"message\":\"unauthorized\"},\"id\":null}",
                "application/json");
            return;
        }
        if (method == "initialize") {
            res.set_header("Mcp-Session-Id", "mock-session");
            res.set_content("{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) +
                                ",\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},"
                                "\"serverInfo\":{\"name\":\"mock-mcp\",\"version\":\"1\"}}}",
                            "application/json");
            return;
        }
        if (method == "notifications/initialized") {
            res.status = 202;
            return;
        }
        if (req.get_header_value("Mcp-Session-Id") != "mock-session") {
            res.status = 404;
            return;
        }
        if (method == "tools/list") {
            res.set_content(
                ": keep alive\n\n"
                "event: message\nid: 1\ndata: {\"jsonrpc\":\"2.0\",\"id\":" +
                    std::to_string(id) +
                    ",\"result\":{\"tools\":[{\"name\":\"clock\",\"description\":\"The time of day\","
                    "\"inputSchema\":{\"type\":\"object\",\"properties\":{}}}]}}\n\n",
                "text/event-stream");
            return;
        }
        if (method == "tools/call") {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            res.set_content(
                "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) +
                    ",\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"It is noon.\"}],\"isError\":false}}",
                "application/json");
            return;
        }
        res.set_content("{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) +
                            ",\"error\":{\"code\":-32601,\"message\":\"method not found\"}}",
                        "application/json");
    });
    mock.Delete("/mcp", [](const httplib::Request &, httplib::Response & res) { res.status = 405; });
}

// One round of the mock model: does the conversation already carry the
// result of a tool, and what it says.
static bool mock_tool_result(const std::string & body, std::string & content) {
    yyjson_doc * doc      = yyjson_read(body.c_str(), body.size(), 0);
    yyjson_val * messages = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "messages") : nullptr;
    size_t       idx, max;
    yyjson_val * message;
    bool         found = false;
    yyjson_arr_foreach(messages, idx, max, message) {
        if (rt_json_str(message, "role") == "tool") {
            found   = true;
            content = rt_json_str(message, "content");
        }
    }
    yyjson_doc_free(doc);
    return found;
}

int main(int argc, char ** argv) {
    std::string       mode = "loopback";
    std::string       echo = "off";
    std::string       route;
    std::string       named;
    int               timeout      = -1;
    int               tool_timeout = -1;
    int               mcp_delay_ms = 0;
    bool              room         = false;
    bool              other        = false;
    bool              mcp          = false;
    int               first_ms     = 100;
    int               token_ms     = 20;
    std::vector<Step> script;

    std::vector<const char *> args;
    for (int i = 1; i < argc; i++) {
        const std::string arg       = argv[i];
        const bool        has_value = i + 1 < argc;
        Step              step;
        if (arg == "--mode" && has_value) {
            mode = argv[++i];
        } else if (arg == "--echo" && has_value) {
            echo = argv[++i];
        } else if (arg == "--room") {
            room = true;
        } else if (arg == "--llm" && has_value) {
            route = argv[++i];
        } else if (arg == "--llm-first-ms" && has_value) {
            first_ms = atoi(argv[++i]);
        } else if (arg == "--llm-token-ms" && has_value) {
            token_ms = atoi(argv[++i]);
        } else if (arg == "--llm-url" && has_value) {
            named = argv[++i];
        } else if (arg == "--llm-timeout" && has_value) {
            timeout = atoi(argv[++i]);
        } else if (arg == "--other-endpoint") {
            other = true;
        } else if (arg == "--mcp") {
            mcp = true;
        } else if (arg == "--mcp-delay-ms" && has_value) {
            mcp_delay_ms = atoi(argv[++i]);
        } else if (arg == "--tool-timeout" && has_value) {
            tool_timeout = atoi(argv[++i]);
        } else if (args.size() < 2) {
            args.push_back(argv[i]);
        } else if (parse_step(argv[i], step)) {
            script.push_back(step);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (args.size() != 2 || script.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string url      = args[0];
    const char *      wav_path = args[1];

    FILE * fp = fopen(wav_path, "rb");
    if (!fp) {
        fprintf(stderr, "[Client] FATAL: cannot open %s\n", wav_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t) size);
    if (fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        fprintf(stderr, "[Client] FATAL: short read on %s\n", wav_path);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    int     n_frames = 0;
    int     rate     = 0;
    float * stereo   = read_wav_buf(raw.data(), raw.size(), &n_frames, &rate);
    if (!stereo) {
        return 1;
    }
    std::vector<float> mono((size_t) n_frames);
    for (int i = 0; i < n_frames; i++) {
        mono[(size_t) i] = 0.5f * (stereo[i * 2] + stereo[i * 2 + 1]);
    }
    free(stereo);

    // The protocol carries 24 kHz, whatever the file holds.
    if (rate != SAMPLE_RATE_24K) {
        int     n_out = 0;
        float * r     = audio_resample(mono.data(), n_frames, rate, SAMPLE_RATE_24K, 1, &n_out);
        if (!r) {
            fprintf(stderr, "[Client] FATAL: resample %d -> %d failed\n", rate, SAMPLE_RATE_24K);
            return 1;
        }
        mono.assign(r, r + n_out);
        free(r);
    }
    for (const Step & step : script) {
        if (step.name == "say" && (size_t) (step.b * SAMPLE_RATE_24K) > mono.size()) {
            fprintf(stderr, "[Client] FATAL: %s is shorter than %.2fs\n", wav_path, step.b);
            return 1;
        }
    }

    // A fixed decaying room response, normalized so the echo level only
    // depends on the gain.
    std::vector<float> room_ir((size_t) (ROOM_TAIL_MS * SAMPLE_RATE_24K / 1000));
    {
        uint32_t seed = 1;
        double   sum  = 0.0;
        for (size_t k = 0; k < room_ir.size(); k++) {
            seed = seed * 1664525u + 1013904223u;
            const double tap =
                ((double) (seed >> 8) / (double) (1u << 24) - 0.5) * std::exp(-(double) k / (0.01 * SAMPLE_RATE_24K));
            room_ir[k] = (float) tap;
            sum += std::fabs(tap);
        }
        for (float & tap : room_ir) {
            tap = (float) (tap / sum);
        }
    }
    const size_t room_delay = (size_t) (ROOM_DELAY_MS * SAMPLE_RATE_24K / 1000);

    Timer timer;

    // The mock endpoint. Every request prints how many user messages it
    // carries and the last one, so the timeline shows what the model read.
    httplib::Server mock;
    std::thread     mock_thread;
    std::string     llm_url;
    if (!route.empty()) {
        mock.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
            int          n_user = 0;
            std::string  last;
            yyjson_doc * doc      = yyjson_read(req.body.c_str(), req.body.size(), 0);
            yyjson_val * messages = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "messages") : nullptr;
            size_t       idx, max;
            yyjson_val * message;
            yyjson_arr_foreach(messages, idx, max, message) {
                if (rt_json_str(message, "role") == "user") {
                    n_user++;
                    last = rt_json_str(message, "content");
                }
            }
            yyjson_doc_free(doc);
            printf("[Mock] %7.2fs  request  %d user messages, last \"%s\"\n", timer.ms() / 1000.0, n_user,
                   last.c_str());
            fflush(stdout);

            res.set_chunked_content_provider("text/event-stream", [&](size_t, httplib::DataSink & sink) {
                std::this_thread::sleep_for(std::chrono::milliseconds(first_ms));
                for (const std::string & word : mock_words(MOCK_ANSWER)) {
                    const std::string chunk = mock_frame(word);
                    if (!sink.write(chunk.data(), chunk.size())) {
                        printf("[Mock] %7.2fs  aborted by the server\n", timer.ms() / 1000.0);
                        fflush(stdout);
                        return false;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(token_ms));
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
        });
        mock.Post("/broken/chat/completions", [](const httplib::Request &, httplib::Response & res) {
            res.status = 500;
            res.set_content("{\"error\":{\"message\":\"model not loaded\"}}", "application/json");
        });
        mock.Post("/midstream/chat/completions", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(mock_frame("Once upon a time. ") + mock_frame("There was ") +
                                "data: {\"error\":{\"message\":\"context size exceeded\"}}\n\n",
                            "text/event-stream");
        });
        // The agent: a call of the clock first, the words once its result is
        // in the conversation.
        mock.Post("/agent/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
            std::string result;
            if (mock_tool_result(req.body, result)) {
                const bool failed = result.rfind("Error:", 0) == 0;
                printf("[Mock] %7.2fs  request  with %s\n", timer.ms() / 1000.0,
                       failed ? "the error of the clock" : "the result of the clock");
                fflush(stdout);
                std::string frames;
                for (const std::string & word :
                     mock_words(failed ? "The clock did not answer, sorry. " : "It is noon, the clock said. ")) {
                    frames += mock_frame(word);
                }
                res.set_content(frames + "data: [DONE]\n\n", "text/event-stream");
                return;
            }
            printf("[Mock] %7.2fs  request  %s\n", timer.ms() / 1000.0,
                   req.body.find("\"clock\"") != std::string::npos ? "offered the clock, calling it" :
                                                                     "without the clock");
            fflush(stdout);
            res.set_content(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\","
                "\"type\":\"function\",\"function\":{\"name\":\"clock\",\"arguments\":\"{}\"}}]}}]}\n\n"
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
                "data: [DONE]\n\n",
                "text/event-stream");
        });
        // The voice: a call of set_voice first, the words once its result is
        // in the conversation.
        mock.Post("/voice/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
            std::string result;
            if (mock_tool_result(req.body, result)) {
                printf("[Mock] %7.2fs  request  with \"%s\"\n", timer.ms() / 1000.0, result.c_str());
                fflush(stdout);
                std::string frames;
                for (const std::string & word : mock_words("This is my new voice. ")) {
                    frames += mock_frame(word);
                }
                res.set_content(frames + "data: [DONE]\n\n", "text/event-stream");
                return;
            }
            printf("[Mock] %7.2fs  request  %s\n", timer.ms() / 1000.0,
                   req.body.find("\"set_voice\"") != std::string::npos ? "offered set_voice, calling it" :
                                                                         "without set_voice");
            fflush(stdout);
            res.set_content(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\","
                "\"type\":\"function\",\"function\":{\"name\":\"set_voice\","
                "\"arguments\":\"{\\\"voice\\\":\\\"" MOCK_VOICE
                "\\\"}\"}}]}}]}\n\n"
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
                "data: [DONE]\n\n",
                "text/event-stream");
        });
        mock_mcp(mock, timer, mcp_delay_ms);
        mock.Post("/empty/chat/completions", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Nothing to add.\"}}]}\n\n"
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                "data: [DONE]\n\n",
                "text/event-stream");
        });
        const int port = mock.bind_to_any_port("127.0.0.1");
        if (port < 0) {
            fprintf(stderr, "[Client] FATAL: the mock endpoint cannot bind\n");
            return 1;
        }
        mock_thread = std::thread([&mock]() { mock.listen_after_bind(); });
        mock.wait_until_ready();
        llm_url = "http://127.0.0.1:" + std::to_string(port) + "/" + route;
    }

    httplib::ws::WebSocketClient client(url);
    if (!client.is_valid() || !client.connect()) {
        fprintf(stderr, "[Client] FATAL: cannot connect to %s\n", url.c_str());
        return 1;
    }

    Speaker speaker;
    size_t  audio_bytes = 0;
    int     n_events    = 0;
    bool    running     = true;

    std::thread reader([&]() {
        std::string frame;
        bool        audio_seen = false;
        while (running) {
            if (client.read(frame) == httplib::ws::ReadResult::Fail) {
                break;
            }

            yyjson_doc * doc = yyjson_read(frame.c_str(), frame.size(), 0);
            if (!doc) {
                continue;
            }
            yyjson_val *      root = yyjson_doc_get_root(doc);
            const std::string type = rt_json_str(root, "type");
            const double      now  = timer.ms() / 1000.0;

            n_events++;
            {
                std::lock_guard<std::mutex> lock(speaker.mutex);
                speaker.last_event = now;
                if (type == "response.created") {
                    speaker.open_response = true;
                } else if (type == "response.done" || type == "response.cancelled") {
                    speaker.open_response = false;
                }
            }

            if (type == "response.output_audio.delta") {
                const std::string  delta = rt_json_str(root, "delta");
                std::vector<float> pcm;
                rt_pcm16_to_float(rt_base64_decode(delta.c_str(), delta.size()), pcm);
                std::lock_guard<std::mutex> lock(speaker.mutex);
                audio_bytes += delta.size();
                speaker.queue.insert(speaker.queue.end(), pcm.begin(), pcm.end());
                // The first chunk of each response marks when its sound starts.
                if (!audio_seen) {
                    printf("[Event] %7.2fs  %s\n", now, type.c_str());
                    audio_seen = true;
                }
            } else if (type == "response.output_text.delta") {
                // What the model writes, one line per word: the spoken units
                // below say the same, one sentence at a time.
            } else if (type == "conversation.item.input_audio_transcription.completed") {
                printf("[Event] %7.2fs  %-46s %s \"%s\"\n", now, type.c_str(), rt_json_str(root, "item_id").c_str(),
                       rt_json_str(root, "transcript").c_str());
            } else if (type == "response.output_audio_transcript.delta") {
                printf("[Event] %7.2fs  %-46s \"%s\"\n", now, type.c_str(), rt_json_str(root, "delta").c_str());
            } else if (type == "session.updated") {
                yyjson_val * session = yyjson_obj_get(root, "session");
                yyjson_val * tts     = session ? yyjson_obj_get(session, "tts") : nullptr;
                printf("[Event] %7.2fs  %-46s %s\n", now, type.c_str(), tts ? rt_json_str(tts, "voice").c_str() : "");
            } else if (type == "error") {
                yyjson_val * error = yyjson_obj_get(root, "error");
                printf("[Event] %7.2fs  %-46s %s\n", now, type.c_str(),
                       error ? rt_json_str(error, "message").c_str() : "");
            } else {
                printf("[Event] %7.2fs  %s\n", now, type.c_str());
                if (type == "response.created") {
                    audio_seen = false;
                }
                // The browser drops what it has not played yet once the user
                // takes the floor.
                if (type == "input_audio_buffer.speech_started") {
                    std::lock_guard<std::mutex> lock(speaker.mutex);
                    if (!speaker.queue.empty()) {
                        printf("[Client] %7.2fs  playback flushed\n", now);
                        speaker.queue.clear();
                    }
                }
            }
            fflush(stdout);
            yyjson_doc_free(doc);
        }
    });

    if (!named.empty()) {
        llm_url = named;
    }
    rt_frame         update  = rt_frame_begin("session.update");
    yyjson_mut_val * session = yyjson_mut_obj_add_obj(update.doc, update.root, "session");
    yyjson_mut_obj_add_strn(update.doc, session, "mode", mode.c_str(), mode.size());
    yyjson_mut_obj_add_strn(update.doc, session, "echo", echo.c_str(), echo.size());
    if (!llm_url.empty()) {
        yyjson_mut_obj_add_strn(update.doc, session, "llm_url", llm_url.c_str(), llm_url.size());
        yyjson_mut_obj_add_str(update.doc, session, "llm_model", "mock");
    }
    if (timeout > 0) {
        yyjson_mut_obj_add_int(update.doc, session, "llm_timeout_sec", timeout);
    }
    if (tool_timeout > 0) {
        yyjson_mut_obj_add_int(update.doc, session, "tool_timeout_sec", tool_timeout);
    }
    if (mcp && !route.empty()) {
        // The mock MCP server lives on the mock endpoint, and the session
        // checks its one tool.
        const std::string mcp_url = llm_url.substr(0, llm_url.rfind('/')) + "/mcp";
        yyjson_mut_val *  servers = yyjson_mut_obj_add_arr(update.doc, session, "mcp");
        yyjson_mut_val *  server  = yyjson_mut_arr_add_obj(update.doc, servers);
        yyjson_mut_obj_add_strncpy(update.doc, server, "url", mcp_url.c_str(), mcp_url.size());
        yyjson_mut_obj_add_str(update.doc, server, "key", "mock-key");
        yyjson_mut_val * tools = yyjson_mut_obj_add_arr(update.doc, session, "tools");
        yyjson_mut_arr_add_str(update.doc, tools, "clock");
    }
    if (route == "voice") {
        yyjson_mut_val * tools = yyjson_mut_obj_add_arr(update.doc, session, "tools");
        yyjson_mut_arr_add_str(update.doc, tools, "set_voice");
    }
    if (other) {
        yyjson_mut_obj_add_str(update.doc, session, "llm_url", "http://127.0.0.1:9/v1");
        yyjson_mut_obj_add_str(update.doc, session, "llm_key", "stolen");
    }
    client.send(rt_frame_end(update));

    // One frame per tick of a fixed clock: the voice of the current step, what
    // the loudspeaker plays during it, and the microphone that hears both.
    const size_t       frame = (size_t) (FRAME_MS * SAMPLE_RATE_24K / 1000);
    std::vector<float> voice(frame), playback(frame), mic(frame);

    size_t step_index = 0;
    size_t said       = 0;     // samples of the current say step sent
    double step_start = -1.0;  // seconds, when the current step began
    bool   sent       = true;  // whether the connection still takes frames
    auto   tick       = std::chrono::steady_clock::now();

    // The settling runs as one more step, silence until the server is done.
    script.push_back({ "settle", 0.0, 0.0 });

    while (sent && step_index < script.size()) {
        const double now  = timer.ms() / 1000.0;
        const Step & step = script[step_index];
        if (step_start < 0.0) {
            step_start = now;
            if (step.name == "say") {
                printf("[Client] %7.2fs  say %.2f-%.2f\n", now, step.a, step.b);
            } else if (step.name == "commit" || step.name == "settle") {
                printf("[Client] %7.2fs  %s\n", now, step.name.c_str());
            } else {
                printf("[Client] %7.2fs  %s %g\n", now, step.name.c_str(), step.a);
            }
            fflush(stdout);
        }

        std::fill(voice.begin(), voice.end(), 0.0f);
        bool done = false;
        bool mute = false;
        if (step.name == "say") {
            const size_t from = (size_t) (step.a * SAMPLE_RATE_24K) + said;
            const size_t to   = (size_t) (step.b * SAMPLE_RATE_24K);
            const size_t n    = std::min(frame, to - from);
            std::copy(mono.begin() + (ptrdiff_t) from, mono.begin() + (ptrdiff_t) (from + n), voice.begin());
            said += n;
            done = from + n >= to;
        } else if (step.name == "pause") {
            done = now - step_start >= step.a;
        } else if (step.name == "mute") {
            mute = true;
            done = now - step_start >= step.a;
        } else if (step.name == "heard") {
            std::lock_guard<std::mutex> lock(speaker.mutex);
            if (speaker.run_ms >= (int) step.a) {
                printf("[Client] %7.2fs  heard %d ms of the answer\n", now, speaker.run_ms);
                done = true;
            } else if (now - step_start >= HEARD_WAIT_SEC) {
                printf("[Client] %7.2fs  no answer heard\n", now);
                done = true;
            }
        } else if (step.name == "commit") {
            sent = client.send(std::string("{\"type\":\"input_audio_buffer.commit\"}"));
            done = true;
        } else if (step.name == "settle") {
            std::lock_guard<std::mutex> lock(speaker.mutex);
            const bool                  quiet = !speaker.open_response && speaker.queue.empty() &&
                               now - std::max(speaker.last_event, step_start) >= SETTLE_QUIET_SEC;
            done = quiet || now - step_start >= SETTLE_MAX_SEC;
        }

        bool any = false;
        {
            std::lock_guard<std::mutex> lock(speaker.mutex);
            for (size_t i = 0; i < frame; i++) {
                playback[i] = 0.0f;
                if (!speaker.queue.empty()) {
                    playback[i] = speaker.queue.front();
                    speaker.queue.pop_front();
                    any = true;
                }
            }
            speaker.run_ms = any ? speaker.run_ms + FRAME_MS : 0;
            speaker.played.insert(speaker.played.end(), playback.begin(), playback.end());

            const size_t end = speaker.played.size();
            for (size_t i = 0; i < frame; i++) {
                const size_t t     = end - frame + i;
                float        heard = 0.0f;
                for (size_t k = 0; room && k < room_ir.size() && t >= room_delay + k; k++) {
                    heard += room_ir[k] * speaker.played[t - room_delay - k];
                }
                mic[i] = voice[i] + ROOM_ECHO_GAIN * heard;
            }
        }

        if (!mute) {
            const std::string audio = rt_float_to_pcm16_base64(mic.data(), frame);
            const std::string reference =
                any && (echo == "server" || echo == "both") ? rt_float_to_pcm16_base64(playback.data(), frame) : "";
            rt_frame append = rt_frame_begin("input_audio_buffer.append");
            rt_frame_str(append, "audio", audio);
            if (!reference.empty()) {
                rt_frame_str(append, "reference", reference);
            }
            sent = client.send(rt_frame_end(append));
        }

        if (done) {
            step_index++;
            said       = 0;
            step_start = -1.0;
        }
        tick += std::chrono::milliseconds(FRAME_MS);
        std::this_thread::sleep_until(tick);
    }

    printf("[Client] %7.2fs  script done\n", timer.ms() / 1000.0);
    fflush(stdout);

    running = false;
    client.close();
    reader.join();
    if (mock_thread.joinable()) {
        mock.stop();
        mock_thread.join();
    }

    const double audio_sec = (double) audio_bytes * 3.0 / 4.0 / 2.0 / SAMPLE_RATE_24K;
    printf("[Client] %d events, %.2fs of audio received over %.2fs\n", n_events, audio_sec, timer.ms() / 1000.0);
    return 0;
}
