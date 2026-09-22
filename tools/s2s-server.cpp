// s2s-server.cpp: the voice loop behind a WebSocket
//
// One process, any number of conversations. This file is the host: the
// command line, the models loaded once and shared, the HTTP routes, and the
// WebSocket that carries each conversation. What a conversation does lives
// in s2s-conversation.cpp, which only sees Realtime frames.
//
// Modes:
//   conversation  recognize, ask the LLM, speak the answer
//   loopback      recognize and speak the transcript back, no endpoint in the
//                 path, which is how the microphone, the turn detection, the
//                 recognizer and the voice get tested on their own

#include "httplib.h"
#include "index.html.gz.hpp"
#include "llm-agent.h"
#include "llm-client.h"
#include "localvqe.h"
#include "log-capture.h"
#include "model-find.h"
#include "parakeet.h"
#include "realtime-proto.h"
#include "s2s-conversation.h"
#include "s2s-error.h"
#include "s2s-session.h"
#include "s2s.js.gz.hpp"
#include "silero.h"
#include "smart-turn.h"
#include "tts-bridge.h"
#include "version.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Frame ceiling of one synthesis, the value the Python reference settles on.
// The per utterance budget of the bridge sits well below it; this only bounds
// the worst case.
#define S2S_TTS_MAX_NEW_TOKENS 1536

static httplib::Server * g_server = nullptr;

// Connections opened since the start, which numbers the next one: a number
// is never given twice, so a reconnection reads apart in the log.
static std::atomic<int> g_connections{ 0 };

// Origin is a browser header: it keeps a third party page out, and it is
// worth nothing against a script that sets it by hand. It is one layer, not
// the protection.
static bool origin_allowed(const std::vector<std::string> & origins, const httplib::Request & req) {
    const std::string origin = req.get_header_value("Origin");
    if (origins.empty() || origin.empty()) {
        return true;
    }
    return std::find(origins.begin(), origins.end(), origin) != origins.end();
}

// Writes a document and frees it: the body of a JSON route.
static std::string json_write(yyjson_mut_doc * doc, yyjson_write_flag flags) {
    char *      json = yyjson_mut_write(doc, flags, nullptr);
    std::string out  = json ? json : "{}";
    free(json);
    yyjson_mut_doc_free(doc);
    return out;
}

// One string under one key: the shape of every error body.
static std::string json_string(const char * key, const std::string & value) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strn(doc, root, key, value.c_str(), value.size());
    return json_write(doc, 0);
}

static void on_signal(int) {
    if (g_server) {
        g_server->stop();
    }
}

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Models:\n"
            "  --models <dir>         Directory holding the GGUF files (default: ./models)\n"
            "  --voices <dir>         Directory holding the voices, <name>.spk with an optional\n"
            "                         <name>.rvq and <name>.txt pair of reference speech\n"
            "                         (default: ./voices)\n"
            "\n"
            "Server:\n"
            "  --host <addr>          Bind address (default: 127.0.0.1)\n"
            "  --port <N>             Bind port (default: 8088)\n"
            "\n"
            "Endpoint, set by the server, hidden from and fixed for every session:\n"
            "  --llm-url <url>        OpenAI compatible endpoint\n"
            "  --llm-model <name>     Model on that endpoint\n"
            "  --llm-key-file <path>  File holding its API key, read at startup\n"
            "\n"
            "Security:\n"
            "  --origin <url>         Allowed browser origin, repeatable. Rejects a WebSocket\n"
            "                         or an HTTP route called from another page. Empty allows\n"
            "                         every origin. A script can forge this header, so it only\n"
            "                         keeps third party pages out.\n"
            "  --llm-host <host>      Allowed endpoint host, repeatable, host[:port]. The server\n"
            "                         fetches the endpoint a client names, so an empty list lets\n"
            "                         it reach anything it can route to. Naming the hosts closes\n"
            "                         that door.\n"
            "\n"
            "Engine:\n"
            "  --max-batch <N>        Concurrent syntheses batched on the GPU (default: 1)\n"
            "  --no-fa                Disable flash attention in the TTS\n"
            "  --clamp-fp16           Clamp hidden states to the FP16 range in the TTS\n"
            "  --codec-chunk-dur <s>  Codec decode chunk, bounds the peak decode memory\n"
            "\n"
            "Everything else belongs to the client: mode, endpoint, prompt, voice,\n"
            "sampling and turn detection travel in session.update, and their\n"
            "defaults are published on /props.\n",
            prog);
}

int main(int argc, char ** argv) {
    // Every thread this server starts names itself; the one that logs
    // without a name is the compute worker of qwentts.
    s2s_log_thread("Main");
    s2s_log_thread_default("TTS");

    std::string models_dir = "models";
    std::string voices_dir = "voices";
    std::string host       = "127.0.0.1";
    int         port       = 8088;

    std::vector<std::string> origins;
    std::vector<std::string> llm_hosts;

    tts_engine        engine;
    ConversationSetup setup;

    // Session defaults, published on /props and overridable per client. The
    // endpoint is the exception: the server's own when the command line names
    // one, never published, never overridden.
    llm_client_params llm_defaults;
    const std::string system_prompt = "You are a voice assistant. Answer in one or two short spoken sentences.";
    const std::string mode          = "loopback";

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        const std::string arg       = argv[i];
        const bool        has_value = i + 1 < argc;

        if (arg == "--models" && has_value) {
            models_dir = argv[++i];
        } else if (arg == "--voices" && has_value) {
            voices_dir = argv[++i];
        } else if (arg == "--host" && has_value) {
            host = argv[++i];
        } else if (arg == "--port" && has_value) {
            port = atoi(argv[++i]);
        } else if (arg == "--origin" && has_value) {
            origins.push_back(argv[++i]);
        } else if (arg == "--llm-host" && has_value) {
            llm_hosts.push_back(argv[++i]);
        } else if (arg == "--llm-url" && has_value) {
            llm_defaults.base_url = argv[++i];
            setup.llm_fixed       = true;
        } else if (arg == "--llm-model" && has_value) {
            llm_defaults.model = argv[++i];
        } else if (arg == "--llm-key-file" && has_value) {
            // The key is the first line of the file.
            std::ifstream in(argv[++i]);
            if (!std::getline(in, llm_defaults.api_key) || llm_defaults.api_key.empty()) {
                s2s_log(S2S_LOG_ERROR, "[Server] FATAL: no key in %s", argv[i]);
                return 1;
            }
            setup.llm_fixed = true;
        } else if (arg == "--max-batch" && has_value) {
            engine.max_batch = atoi(argv[++i]);
        } else if (arg == "--no-fa") {
            engine.use_fa = false;
        } else if (arg == "--clamp-fp16") {
            engine.clamp_fp16 = true;
        } else if (arg == "--codec-chunk-dur" && has_value) {
            engine.codec_chunk_sec = (float) atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::string vad_path    = find_model(models_dir, "silero-vad", "");
    const std::string turn_path   = find_model(models_dir, "smart-turn", "");
    const std::string asr_path    = find_model(models_dir, "parakeet", "");
    const std::string talker_path = find_model(models_dir, "qwen-talker", "-base-");
    const std::string codec_path  = find_model(models_dir, "qwen-tokenizer", "");
    const std::string aec_path    = find_model(models_dir, "localvqe", "");

    if (vad_path.empty() || turn_path.empty() || asr_path.empty() || talker_path.empty() || codec_path.empty() ||
        aec_path.empty()) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: missing models in %s, run ./models.sh", models_dir.c_str());
        return 1;
    }

    s2s_log(S2S_LOG_INFO, "[Load] VAD %s", vad_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] Turn %s", turn_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] ASR %s", asr_path.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] TTS %s + %s, voices from %s", talker_path.c_str(), codec_path.c_str(),
            voices_dir.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] AEC %s", aec_path.c_str());

    setup.models.vad = sv_init(vad_path.c_str());
    if (!setup.models.vad) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: %s", sv_last_error());
        return 1;
    }

    setup.models.turn = st_init(turn_path.c_str());
    if (!setup.models.turn) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: %s", st_last_error());
        return 1;
    }

    pk_init_params asr_init = pk_init_default_params();
    asr_init.model_path     = asr_path.c_str();

    setup.models.asr = pk_init(&asr_init);
    if (!setup.models.asr) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: %s", pk_last_error());
        return 1;
    }

    tts_bridge_params tts_init;
    tts_init.talker_path             = talker_path;
    tts_init.codec_path              = codec_path;
    tts_init.voices_dir              = voices_dir;
    tts_init.sampling.max_new_tokens = S2S_TTS_MAX_NEW_TOKENS;
    tts_init.engine                  = engine;

    setup.models.tts = tts_bridge_load(tts_init);
    if (!setup.models.tts) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: %s", tts_bridge_last_error());
        return 1;
    }

    setup.models.aec = lv_init(aec_path.c_str());
    if (!setup.models.aec) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: %s", lv_last_error());
        return 1;
    }

    // stderr is mirrored into a ring buffer and streamed to the UI, so a
    // browser with no sound and no microphone still shows every stage the
    // server went through.
    LogCapture log_capture;

    setup.llm_hosts = llm_hosts;

    setup.defaults.mode          = mode;
    setup.defaults.tts           = tts_bridge_defaults_request(setup.models.tts);
    setup.defaults.llm           = llm_defaults;
    setup.defaults.system_prompt = system_prompt;

    httplib::Server server;
    g_server = &server;

    // The pool threads serve requests, and a WebSocket connection names its
    // reader for as long as it lasts.
    server.set_pre_routing_handler([](const httplib::Request &, httplib::Response &) {
        s2s_log_thread("HTTP");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // SO_REUSEADDR lets us rebind a port still in TIME_WAIT after a restart.
    // SO_REUSEPORT is deliberately not set: a second instance on the same port
    // then fails with EADDRINUSE instead of silently sharing the socket and
    // splitting traffic between two daemons.
    server.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    server.Get("/logs", [&](const httplib::Request & req, httplib::Response & res) {
        // The stream carries everything the server did, so it is not public.
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }
        handle_logs(req, res);
    });

    server.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Single source of truth for the session defaults: the UI leaves a field
    // empty to mean "whatever the server was started with", and shows this
    // value as the placeholder. The engine setup is not here: it belongs to
    // the command line and to the startup log. Sampling is absent on purpose:
    // an empty sampling field leaves the endpoint to its own defaults, which
    // is the only sane answer when the endpoint can be llama-server, Ollama
    // or a cloud API.
    server.Get("/props", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }
        const s2s_session_params turn;
        const tts_request &      voice  = tts_bridge_defaults_request(setup.models.tts);
        const tts_sampling &     tts    = tts_bridge_defaults(setup.models.tts);
        const tts_guards &       guards = voice.guards;

        yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_str(doc, root, "version", S2S_VERSION);
        yyjson_mut_obj_add_int(doc, root, "sample_rate", S2S_INPUT_RATE);

        // Which files are actually loaded: the directory can hold several
        // quants of the same model and nobody should have to guess.
        // File names only: the page shows what is loaded, not where the
        // server keeps it, and the separator is the OS's business.
        yyjson_mut_val * models = yyjson_mut_obj_add_obj(doc, root, "models");
        const auto       file   = [doc, models](const char * key, const std::string & path) {
            const std::string name = std::filesystem::path(path).filename().string();
            yyjson_mut_obj_add_strncpy(doc, models, key, name.c_str(), name.size());
        };
        file("vad", vad_path);
        file("turn", turn_path);
        file("asr", asr_path);
        file("talker", talker_path);
        file("codec", codec_path);
        file("aec", aec_path);

        yyjson_mut_val * defaults = yyjson_mut_obj_add_obj(doc, root, "defaults");
        const auto       str      = [doc, defaults](const char * key, const std::string & value) {
            yyjson_mut_obj_add_strn(doc, defaults, key, value.c_str(), value.size());
        };
        const auto strs = [doc, defaults](const char * key, const std::vector<std::string> & values) {
            yyjson_mut_val * array = yyjson_mut_obj_add_arr(doc, defaults, key);
            for (const std::string & value : values) {
                yyjson_mut_arr_add_strn(doc, array, value.c_str(), value.size());
            }
        };
        str("mode", mode);
        yyjson_mut_obj_add_bool(doc, defaults, "llm_fixed", setup.llm_fixed);
        str("instructions", system_prompt);
        str("voice", voice.voice);
        str("language", voice.language);
        strs("tts_voices", tts_bridge_voices(setup.models.tts));
        strs("tts_languages", tts_bridge_languages(setup.models.tts));
        yyjson_mut_obj_add_int(doc, defaults, "tts_min_chars", guards.min_chars);
        yyjson_mut_obj_add_real(doc, defaults, "tts_chars_per_second", guards.chars_per_second);
        yyjson_mut_obj_add_real(doc, defaults, "tts_margin_seconds", guards.margin_seconds);
        yyjson_mut_obj_add_int(doc, defaults, "llm_timeout_sec", llm_defaults.timeout_sec);
        yyjson_mut_obj_add_int(doc, defaults, "max_rounds", setup.defaults.max_rounds);
        yyjson_mut_obj_add_int(doc, defaults, "tool_timeout_sec", llm_defaults.tool_timeout_sec);
        str("reasoning_effort", llm_defaults.sampling.reasoning_effort);
        yyjson_mut_obj_add_real(doc, defaults, "tts_temperature", tts.temperature);
        yyjson_mut_obj_add_int(doc, defaults, "tts_top_k", tts.top_k);
        yyjson_mut_obj_add_real(doc, defaults, "tts_top_p", tts.top_p);
        yyjson_mut_obj_add_real(doc, defaults, "tts_repetition_penalty", tts.repetition_penalty);
        yyjson_mut_obj_add_real(doc, defaults, "tts_subtalker_temperature", tts.subtalker_temperature);
        yyjson_mut_obj_add_int(doc, defaults, "tts_subtalker_top_k", tts.subtalker_top_k);
        yyjson_mut_obj_add_real(doc, defaults, "tts_subtalker_top_p", tts.subtalker_top_p);
        // The ceiling in force, not the submodule maximum: this server caps
        // it lower, and the placeholder must say what actually applies.
        yyjson_mut_obj_add_int(doc, defaults, "tts_max_new_tokens", S2S_TTS_MAX_NEW_TOKENS);
        yyjson_mut_obj_add_real(doc, defaults, "vad_neg_threshold", turn.vad_neg_threshold);
        yyjson_mut_obj_add_real(doc, defaults, "vad_threshold", turn.vad_threshold);
        yyjson_mut_obj_add_int(doc, defaults, "min_speech_ms", turn.min_speech_ms);
        yyjson_mut_obj_add_int(doc, defaults, "barge_in_ms", turn.barge_in_ms);
        yyjson_mut_obj_add_int(doc, defaults, "min_silence_ms", turn.min_silence_ms);
        yyjson_mut_obj_add_int(doc, defaults, "min_speech_continuation_ms", turn.min_speech_continuation_ms);
        yyjson_mut_obj_add_int(doc, defaults, "speech_pad_ms", turn.speech_pad_ms);
        yyjson_mut_obj_add_real(doc, defaults, "turn_threshold", turn.turn_threshold);
        yyjson_mut_obj_add_int(doc, defaults, "turn_max_wait_ms", turn.turn_max_wait_ms);
        yyjson_mut_obj_add_int(doc, defaults, "reopen_grace_ms", turn.reopen_grace_ms);

        // Every real above is a float: written as the shortest text that
        // reads back to it, 0.6 and not 0.6000000238418579.
        res.set_content(json_write(doc, YYJSON_WRITE_FP_TO_FLOAT), "application/json");
    });

    // Embedded webui: gzipped single page app built by tools/webui.
    // The browser decompresses it through Content-Encoding.
    if (index_html_gz_len > 0) {
        server.Get("/", [](const httplib::Request & req, httplib::Response & res) {
            if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                res.set_content("Error: gzip is not supported by this browser", "text/plain");
            } else {
                res.set_header("Content-Encoding", "gzip");
                res.set_content(reinterpret_cast<const char *>(index_html_gz), index_html_gz_len,
                                "text/html; charset=utf-8");
            }
        });
    }

    // Browser log: the page posts what happens on its side, so the microphone
    // permission, the socket and the audio context show up in the same stream
    // as the server stages instead of dying in a console nobody reads.
    server.Post("/log", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }
        std::string line = req.body.substr(0, 512);
        for (char & c : line) {
            if (c == '\n' || c == '\r') {
                c = ' ';
            }
        }
        s2s_log(S2S_LOG_INFO, "[Browser] %s", line.c_str());
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Model list proxy: the browser asks s2s-server, s2s-server asks the
    // endpoint. No CORS to negotiate, no endpoint reachable from the client
    // side, and the API key stays on this machine. The body overrides the
    // defaults so the panel can probe a URL before the session opens.
    server.Post("/v1/models", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req) || setup.llm_fixed) {
            res.status = 403;
            return;
        }

        llm_client_params params = llm_defaults;

        yyjson_doc * doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (doc) {
            yyjson_val *      root = yyjson_doc_get_root(doc);
            const std::string url  = rt_json_str(root, "url");
            const std::string key  = rt_json_str(root, "key");
            if (!url.empty()) {
                params.base_url = url;
            }
            if (!key.empty()) {
                params.api_key = key;
            }
            yyjson_doc_free(doc);
        }

        if (!host_allowed(llm_hosts, params.base_url)) {
            s2s_log(S2S_LOG_WARN, "[HTTP] Endpoint host %s is not allowed", url_host(params.base_url).c_str());
            res.status = 403;
            res.set_content("{\"error\":\"endpoint host not allowed\"}", "application/json");
            return;
        }

        s2s_log(S2S_LOG_INFO, "[HTTP] Model list");

        std::vector<std::string> models;
        if (!llm_client_models(params, models)) {
            s2s_log(S2S_LOG_WARN, "[LLM] Model list failed: %s", llm_client_last_error());
            res.status = 502;
            res.set_content(json_string("error", llm_client_last_error()), "application/json");
            return;
        }
        s2s_log(S2S_LOG_INFO, "[LLM] %zu models", models.size());

        yyjson_mut_doc * body = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val * root = yyjson_mut_obj(body);
        yyjson_mut_doc_set_root(body, root);
        yyjson_mut_val * data = yyjson_mut_obj_add_arr(body, root, "data");
        for (const std::string & model : models) {
            yyjson_mut_val * entry = yyjson_mut_arr_add_obj(body, data);
            yyjson_mut_obj_add_strn(body, entry, "id", model.c_str(), model.size());
        }
        res.set_content(json_write(body, 0), "application/json");
    });

    // Tool list proxy: the same path as the model list, for the tools a
    // llama.cpp endpoint runs. An endpoint without that route answers with
    // its error, which is how the page says that this mode needs llama.cpp.
    server.Post("/v1/tools", [&](const httplib::Request & req, httplib::Response & res) {
        if (!origin_allowed(origins, req)) {
            res.status = 403;
            return;
        }

        llm_client_params params = llm_defaults;

        yyjson_doc * doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (doc) {
            yyjson_val *      root = yyjson_doc_get_root(doc);
            const std::string url  = rt_json_str(root, "url");
            const std::string key  = rt_json_str(root, "key");
            if (!setup.llm_fixed && !url.empty()) {
                params.base_url = url;
            }
            if (!setup.llm_fixed && !key.empty()) {
                params.api_key = key;
            }
            yyjson_doc_free(doc);
        }

        if (!host_allowed(llm_hosts, params.base_url)) {
            s2s_log(S2S_LOG_WARN, "[HTTP] Endpoint host %s is not allowed", url_host(params.base_url).c_str());
            res.status = 403;
            res.set_content("{\"error\":\"endpoint host not allowed\"}", "application/json");
            return;
        }

        s2s_log(S2S_LOG_INFO, "[HTTP] Tool list");

        std::vector<std::string> tools;
        if (!llm_agent_tools(params, tools)) {
            s2s_log(S2S_LOG_WARN, "[Agent] Tool list failed: %s", llm_client_last_error());
            res.status = 502;
            res.set_content(json_string("error", llm_client_last_error()), "application/json");
            return;
        }
        s2s_log(S2S_LOG_INFO, "[Agent] %zu tools", tools.size());

        yyjson_mut_doc * body = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val * root = yyjson_mut_obj(body);
        yyjson_mut_doc_set_root(body, root);
        yyjson_mut_val * data = yyjson_mut_obj_add_arr(body, root, "data");
        for (const std::string & tool : tools) {
            yyjson_mut_arr_add_strn(body, data, tool.c_str(), tool.size());
        }
        res.set_content(json_write(body, 0), "application/json");
    });

    // s2s.js: the same client, standalone, for a page hosted elsewhere.
    if (s2s_js_gz_len > 0) {
        server.Get("/s2s.js", [](const httplib::Request & req, httplib::Response & res) {
            if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                res.status = 406;
                res.set_content("gzip required", "text/plain");
                return;
            }
            res.set_header("Content-Encoding", "gzip");
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_content(reinterpret_cast<const char *>(s2s_js_gz), s2s_js_gz_len, "text/javascript; charset=utf-8");
        });
    }

    // A plain GET on the realtime path means the upgrade never happened: a
    // reverse proxy that does not tunnel WebSocket answers this instead of
    // handing the socket over. Saying so beats a silent 404.
    server.Get("/v1/realtime", [](const httplib::Request & req, httplib::Response & res) {
        s2s_log(S2S_LOG_WARN, "[HTTP] GET /v1/realtime without an upgrade, Connection: '%s', Upgrade: '%s'",
                req.get_header_value("Connection").c_str(), req.get_header_value("Upgrade").c_str());
        res.status = 426;
        res.set_content("{\"error\":\"websocket upgrade required\"}", "application/json");
    });

    server.WebSocket("/v1/realtime", [&](const httplib::Request & req, httplib::ws::WebSocket & ws) {
        // The pool thread is this connection's reader until it closes.
        const int    id = ++g_connections;
        S2SLogThread named("Reader-" + std::to_string(id));

        // WebSocket handshakes bypass CORS, so the origin is checked here.
        const std::string origin = req.get_header_value("Origin");
        if (!origin_allowed(origins, req)) {
            s2s_log(S2S_LOG_WARN, "[Server] Rejected origin %s", origin.c_str());
            ws.close(httplib::ws::CloseStatus::PolicyViolation, "origin not allowed");
            return;
        }

        s2s_log(S2S_LOG_INFO, "[Server] Connection from %s", origin.empty() ? "unknown origin" : origin.c_str());

        Connection * conn = conn_open(
            &setup, id,
            [](const std::string & frame, void * user) { return ((httplib::ws::WebSocket *) user)->send(frame); }, &ws);
        if (!conn) {
            return;
        }

        std::string frame;
        for (;;) {
            const httplib::ws::ReadResult result = ws.read(frame);
            if (result == httplib::ws::ReadResult::Fail || conn_stopped(conn)) {
                break;
            }
            if (result != httplib::ws::ReadResult::Text) {
                continue;
            }
            conn_frame(conn, frame);
        }
        conn_close(conn);
    });

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    s2s_log(S2S_LOG_INFO, "[Server] s2s-server %s", S2S_VERSION);
    s2s_log(S2S_LOG_INFO, "[Server] Mode: %s, endpoint %s", mode.c_str(),
            setup.llm_fixed ? "set by the command line" : "named by the session");
    s2s_log(S2S_LOG_INFO, "[Server] Listening on http://%s:%d", host.c_str(), port);

    if (!server.listen(host, port)) {
        s2s_log(S2S_LOG_ERROR, "[Server] FATAL: cannot bind %s:%d", host.c_str(), port);
        return 1;
    }

    lv_free(setup.models.aec);
    tts_bridge_free(setup.models.tts);
    pk_free(setup.models.asr);
    st_free(setup.models.turn);
    sv_free(setup.models.vad);
    return 0;
}
