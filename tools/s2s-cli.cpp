// s2s-cli.cpp: the voice loop on the microphone and the loudspeaker of this
// machine
//
// A host like s2s-server, with the sound card where the browser was: the
// models load here and one conversation runs in process. The host speaks the
// Realtime protocol to the engine, the frames s2s.js sends and receives, so
// the conversation is the one the server runs, and so is its log.
//
// One miniaudio duplex device at the protocol rate: its callback plays the
// answer and captures the microphone, and what it plays is the reference of
// the echo canceller, sample for sample. miniaudio converts to and from the
// rates of the hardware.
//
// The client keeps three things apart: the audio the callback shares, the
// context the model knows, and the display the user sees, the conversation
// on stdout while the log of the engine goes to stderr.
//
// Three threads carry the conversation. The audio callback touches nothing
// but two ring buffers and a few atomics. The main thread is the reader of
// the connection: every frame reaches the engine from it, the microphone
// and the frames of the client alike. The writer of the engine delivers the
// frames going back, and the context and the display they change sit under
// one mutex.

#include "realtime-proto.h"
#include "s2s-conversation.h"
#include "s2s-error.h"
#include "s2s-models.h"
#include "utf8.h"
#include "version.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#    include <CoreFoundation/CoreFoundation.h>
#endif

#if defined(_WIN32)
#    include <io.h>
#    define CLI_ISATTY(f) _isatty(_fileno(f))
#else
#    include <unistd.h>
#    define CLI_ISATTY(f) isatty(fileno(f))
#endif

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#include "miniaudio/miniaudio.h"

#define CLI_FRAME_SAMPLES    480  // 20 ms at the protocol rate, the frame s2s.js sends
#define CLI_PERIOD_MS        10   // period of the audio callback
#define CLI_CAPTURE_SECONDS  2    // microphone waiting for the reader
#define CLI_PLAYBACK_SECONDS 120  // answer waiting for the loudspeaker
#define CLI_POLL_MS          2    // sleep of the reader when no frame is ready

#define CLI_COLOR_USER      "\033[32m"
#define CLI_COLOR_ASSISTANT "\033[36m"
#define CLI_COLOR_TOOL      "\033[97m"
#define CLI_COLOR_FAILED    "\033[31m"
#define CLI_COLOR_RESET     "\033[0m"

// The one conversation of the process, numbered like a connection of the
// server so the log reads the same.
#define CLI_CONNECTION 1

static std::atomic<bool> g_stop{ false };

// Whether the conversation lines carry their ANSI color.
static bool g_color = false;

static void on_signal(int) {
    g_stop = true;
}

// The sound card, shared with the audio callback without a lock: the rings
// have one producer and one consumer each.
struct Audio {
    ma_pcm_rb             capture;         // two channels: the microphone and what played meanwhile
    ma_pcm_rb             playback;        // the answer, waiting for the loudspeaker
    std::atomic<bool>     flush{ false };  // raised by the client, lowered by the callback once the queue is gone
    std::atomic<uint64_t> heard{ 0 };      // samples the loudspeaker played
    std::atomic<uint64_t> consumed{ 0 };   // samples played or dropped
};

// A unit the voice speaks, with the sample of the answer where its audio
// starts.
struct Unit {
    std::string text;
    uint64_t    start = 0;
};

// What the model knows: the conversation the client owns, and the answer in
// flight, filed with what was heard of it.
struct Context {
    std::vector<rt_message> history;
    std::string             user_item;              // the item of the last user message
    std::vector<Unit>       units;
    bool                    answering     = false;  // between response.created and its close
    bool                    answer_done   = false;  // the server finished the answer
    uint64_t                queued        = 0;      // samples of the answer sent to the loudspeaker
    uint64_t                heard_base    = 0;      // heard at response.created
    uint64_t                consumed_base = 0;      // consumed at response.created
};

// What the user sees: the conversation on stdout, apart from the log, tagged
// like it and in color on a terminal. Every transcript of the user takes a
// line numbered by its revision, from 0, and every answer one line, written
// as the model writes it, token by token, with its tool calls where the
// model makes them. Loopback has no model writing, so its answers show the
// units the voice speaks.
struct Display {
    int  revision  = 0;      // of the last user line
    bool line_open = false;  // the line of the answer awaits its end
    bool written   = false;  // the answer in flight wrote something
};

struct Cli {
    bool  aec = true;  // the reference travels with the microphone
    Audio audio;

    // Changed by the frames of the engine, read by the reader.
    std::mutex               mutex;
    Context                  context;
    Display                  display;
    std::vector<std::string> outbox;  // frames waiting for the reader
};

// Colors only on a terminal: a redirected stdout stays plain text. The
// Windows console reads the ANSI sequences once asked to.
static bool console_colors() {
    if (!CLI_ISATTY(stdout)) {
        return false;
    }
#if defined(_WIN32)
    HANDLE out  = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  mode = 0;
    return GetConsoleMode(out, &mode) && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
    return true;
#endif
}

static void display_answer_start(Display & display) {
    display.written = false;
}

static void display_answer_end(Display & display) {
    if (display.line_open) {
        printf("%s\n", g_color ? CLI_COLOR_RESET : "");
        fflush(stdout);
        display.line_open = false;
    }
}

static void display_user(Display & display, const std::string & text, bool revised) {
    display_answer_end(display);
    display.revision = revised ? display.revision + 1 : 0;
    printf("%s[User-%d] %s%s\n", g_color ? CLI_COLOR_USER : "", display.revision, text.c_str(),
           g_color ? CLI_COLOR_RESET : "");
    fflush(stdout);
}

// Appends to the line of the answer, opening it on the first text.
static void display_append(Display & display, const std::string & text) {
    if (!display.line_open) {
        printf("%s[Assistant] ", g_color ? CLI_COLOR_ASSISTANT : "");
        display.line_open = true;
    }
    printf("%s", text.c_str());
    fflush(stdout);
}

static void display_text(Display & display, const std::string & delta) {
    display.written = true;
    display_append(display, delta);
}

static void display_unit(Display & display, const std::string & unit) {
    if (!display.written) {
        display_append(display, display.line_open ? " " + unit : unit);
    }
}

// A call shows where the model makes it, inserted as is between two deltas:
// its name while it runs, then the time it ran and the size of its result,
// or that it failed.
static void display_tool_start(Display & display, const std::string & name) {
    display_append(display, std::string(g_color ? CLI_COLOR_TOOL : "") + "<" + name);
}

static void display_tool_done(Display & display, bool ok, uint64_t ms, uint64_t bytes) {
    const std::string tail = ok ? " " + std::to_string(ms) + " ms " + std::to_string(bytes) + " bytes>" :
                                  std::string(g_color ? CLI_COLOR_FAILED : "") + " failed>";
    display_append(display, tail + (g_color ? CLI_COLOR_ASSISTANT : ""));
}

static void on_audio(ma_device * device, void * output, const void * input, ma_uint32 n_frames) {
    Audio *       audio = (Audio *) device->pUserData;
    float *       out   = (float *) output;
    const float * in    = (const float *) input;

    if (audio->flush.load(std::memory_order_acquire)) {
        const ma_uint32 dropped = ma_pcm_rb_available_read(&audio->playback);
        ma_pcm_rb_seek_read(&audio->playback, dropped);
        audio->consumed += dropped;
        audio->flush.store(false, std::memory_order_release);
    }

    ma_uint32 played = 0;
    while (played < n_frames) {
        ma_uint32 n     = n_frames - played;
        void *    chunk = nullptr;
        if (ma_pcm_rb_acquire_read(&audio->playback, &n, &chunk) != MA_SUCCESS || n == 0) {
            break;
        }
        memcpy(out + played, chunk, n * sizeof(float));
        ma_pcm_rb_commit_read(&audio->playback, n);
        played += n;
    }
    memset(out + played, 0, (n_frames - played) * sizeof(float));
    audio->heard += played;
    audio->consumed += played;

    ma_uint32 captured = 0;
    while (captured < n_frames) {
        ma_uint32 n     = n_frames - captured;
        void *    chunk = nullptr;
        if (ma_pcm_rb_acquire_write(&audio->capture, &n, &chunk) != MA_SUCCESS || n == 0) {
            break;
        }
        float * pair = (float *) chunk;
        for (ma_uint32 i = 0; i < n; i++) {
            pair[2 * i]     = in[captured + i];
            pair[2 * i + 1] = out[captured + i];
        }
        ma_pcm_rb_commit_write(&audio->capture, n);
        captured += n;
    }
}

// Drops what the loudspeaker has not played yet, and returns once the
// callback did.
static void audio_flush(Audio & audio) {
    audio.flush.store(true, std::memory_order_release);
    while (audio.flush.load(std::memory_order_acquire) && !g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Queues samples for the loudspeaker, and returns how many it took.
static size_t audio_play(Audio & audio, const std::vector<float> & pcm) {
    size_t done = 0;
    while (done < pcm.size()) {
        ma_uint32 n     = (ma_uint32) (pcm.size() - done);
        void *    chunk = nullptr;
        if (ma_pcm_rb_acquire_write(&audio.playback, &n, &chunk) != MA_SUCCESS || n == 0) {
            s2s_log(S2S_LOG_WARN, "[Audio] Playback queue full, %zu samples dropped", pcm.size() - done);
            break;
        }
        memcpy(chunk, pcm.data() + done, n * sizeof(float));
        ma_pcm_rb_commit_write(&audio.playback, n);
        done += n;
    }
    return done;
}

// The conversation, as the frame that hands it to the engine.
static std::string context_history(const Context & context) {
    rt_frame         frame    = rt_frame_begin("conversation.history");
    yyjson_mut_val * messages = yyjson_mut_obj_add_arr(frame.doc, frame.root, "messages");
    for (const rt_message & message : context.history) {
        yyjson_mut_val * entry = yyjson_mut_arr_add_obj(frame.doc, messages);
        yyjson_mut_obj_add_strn(frame.doc, entry, "role", message.role.c_str(), message.role.size());
        yyjson_mut_obj_add_strn(frame.doc, entry, "content", message.content.c_str(), message.content.size());
        if (!message.item.empty()) {
            yyjson_mut_obj_add_strn(frame.doc, entry, "item", message.item.c_str(), message.item.size());
        }
    }
    return rt_frame_end(frame);
}

// A later transcript of the same turn replaces its user message instead of
// adding one. Returns whether it did.
static bool context_user(Context & context, const std::string & item, const std::string & transcript) {
    const bool revised =
        !item.empty() && item == context.user_item && !context.history.empty() && context.history.back().role == "user";
    if (revised) {
        context.history.back().content = transcript;
    } else {
        context.history.push_back({ "user", transcript, item });
    }
    context.user_item = item;
    return revised;
}

static void context_open(Context & context, const Audio & audio) {
    context.answering     = true;
    context.answer_done   = false;
    context.queued        = 0;
    context.heard_base    = audio.heard;
    context.consumed_base = audio.consumed;
}

// Whether the loudspeaker played the last sample of an answer the server
// finished.
static bool context_played(const Context & context, const Audio & audio) {
    return context.answering && context.answer_done && audio.consumed - context.consumed_base >= context.queued;
}

// Closes the answer in flight with what was actually heard. A unit counts
// once its audio started playing: a barge-in keeps the sentence it cut and
// drops the ones still queued, so the model never believes it said more. An
// answer with nothing heard is not filed. Every close hands the list to the
// engine.
static void context_close(Context & context, const Audio & audio, std::vector<std::string> & outbox) {
    const uint64_t heard = audio.heard - context.heard_base;
    std::string    content;
    for (const Unit & unit : context.units) {
        if (unit.start < heard) {
            content += (content.empty() ? "" : " ") + unit.text;
        }
    }
    const size_t first = content.find_first_not_of(" \t\r\n");
    const size_t last  = content.find_last_not_of(" \t\r\n");
    content            = first == std::string::npos ? "" : content.substr(first, last - first + 1);

    context.units.clear();
    context.answering   = false;
    context.answer_done = false;
    if (!content.empty()) {
        context.history.push_back({ "assistant", content, "" });
    }
    outbox.push_back(context_history(context));
}

// Ends the answer in flight, in the context and on the display.
static void close_answer(Cli * cli) {
    if (!cli->context.answering) {
        return;
    }
    context_close(cli->context, cli->audio, cli->outbox);
    display_answer_end(cli->display);
}

// Every frame the engine sends, on its writer thread.
static bool on_event(const std::string & frame, void * user) {
    Cli * cli = (Cli *) user;

    yyjson_doc * doc = yyjson_read(frame.c_str(), frame.size(), 0);
    if (!doc) {
        return true;
    }
    yyjson_val *      root = yyjson_doc_get_root(doc);
    const std::string type = rt_json_str(root, "type");

    std::lock_guard<std::mutex> lock(cli->mutex);

    if (type == "input_audio_buffer.speech_started") {
        // The user speaks: whatever still plays is over, and the answer keeps
        // only what was heard.
        if (cli->context.answering) {
            audio_flush(cli->audio);
            close_answer(cli);
        }
    } else if (type == "conversation.item.input_audio_transcription.completed") {
        const std::string transcript = rt_json_str(root, "transcript");
        const bool        revised    = context_user(cli->context, rt_json_str(root, "item_id"), transcript);
        display_user(cli->display, transcript, revised);
    } else if (type == "response.created") {
        // An answer the client already closed takes nothing more: every delta
        // below checks it is still open.
        close_answer(cli);
        context_open(cli->context, cli->audio);
        display_answer_start(cli->display);
    } else if (type == "response.output_text.delta") {
        if (cli->context.answering) {
            display_text(cli->display, rt_json_str(root, "delta"));
        }
    } else if (type == "response.tool_call.started") {
        if (cli->context.answering) {
            display_tool_start(cli->display, rt_json_str(root, "name"));
        }
    } else if (type == "response.tool_call.done") {
        if (cli->context.answering) {
            display_tool_done(cli->display, yyjson_get_bool(yyjson_obj_get(root, "ok")),
                              yyjson_get_uint(yyjson_obj_get(root, "ms")),
                              yyjson_get_uint(yyjson_obj_get(root, "bytes")));
        }
    } else if (type == "response.output_audio_transcript.delta") {
        if (cli->context.answering) {
            const std::string delta = rt_json_str(root, "delta");
            cli->context.units.push_back({ delta, cli->context.queued });
            display_unit(cli->display, delta);
        }
    } else if (type == "response.output_audio.delta") {
        if (cli->context.answering) {
            const std::string  delta = rt_json_str(root, "delta");
            std::vector<float> pcm;
            rt_pcm16_to_float(rt_base64_decode(delta.c_str(), delta.size()), pcm);
            cli->context.queued += audio_play(cli->audio, pcm);
        }
    } else if (type == "response.cancelled") {
        audio_flush(cli->audio);
        close_answer(cli);
    } else if (type == "response.done") {
        // The server is done, the loudspeaker may not be: the reader closes
        // the answer once the last queued sample played.
        cli->context.answer_done = true;
    }

    yyjson_doc_free(doc);
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Models:\n"
            "  --models <dir>              Directory holding the GGUF files (default: ./models)\n"
            "  --voices <dir>              Directory holding the voices, <name>.spk with an optional\n"
            "                              <name>.rvq and <name>.txt pair of reference speech\n"
            "                              (default: ./voices)\n"
            "\n"
            "Session:\n"
            "  --mode <mode>               loopback, conversation or agentic (default: loopback)\n"
            "  --instructions-file <path>  File holding the system prompt of the model, read at\n"
            "                              startup\n"
            "\n"
            "Endpoint:\n"
            "  --llm-url <url>             OpenAI compatible endpoint\n"
            "  --llm-model <name>          Model on that endpoint\n"
            "  --llm-timeout <s>           Longest the endpoint may stay silent, a prefill included\n"
            "                              (default: 10)\n"
            "  --llm-key-file <path>       File holding its API key, read at startup\n"
            "  --reasoning-effort <e>      How long a reasoning model thinks before it answers:\n"
            "                              none, or a level of its chat template (default: none)\n"
            "\n"
            "Tools, for the agentic mode:\n"
            "  --mcp <url>                 MCP server, repeatable, Streamable HTTP\n"
            "  --mcp-key-file <path>       File holding the key of the --mcp named before it\n"
            "  --tool <name>               Tool offered to the model, repeatable: set_voice, a tool\n"
            "                              of an MCP server or of the endpoint\n"
            "  --tool-timeout <s>          Longest a tool may work before its call fails\n"
            "                              (default: 10)\n"
            "  --max-rounds <N>            Rounds of tool calls one turn may take (default: 10)\n"
            "\n"
            "Voice:\n"
            "  --voice <label>             One of the labels the log lists at startup\n"
            "  --effect <name>             Effect over the voice: off or jarvis (default: off)\n"
            "  --language <name>           Language of the voice (default: auto, the first language\n"
            "                              of the system the voice speaks, else English)\n"
            "\n"
            "Sound:\n"
            "  --list-devices              Lists the microphones and the loudspeakers, then exits\n"
            "  --mic <N>                   Microphone, by its number in the list (default: the\n"
            "                              system's)\n"
            "  --speaker <N>               Loudspeaker, by its number in the list (default: the\n"
            "                              system's)\n"
            "  --no-aec                    Disable echo cancellation, for headphones\n"
            "\n"
            "Engine:\n"
            "  --no-fa                     Disable flash attention in the TTS\n"
            "  --clamp-fp16                Clamp hidden states to the FP16 range in the TTS\n"
            "  --codec-chunk-dur <s>       Codec decode chunk, bounds the peak decode memory\n",
            prog);
}

// The first line of a file: a key.
static bool read_key(const char * path, std::string & key) {
    std::ifstream in(std::filesystem::u8path(path));
    if (!std::getline(in, key) || key.empty()) {
        s2s_log(S2S_LOG_ERROR, "[Main] FATAL: no key in %s", path);
        return false;
    }
    return true;
}

// A whole file: a system prompt.
static bool read_text(const char * path, std::string & text) {
    std::ifstream      in(std::filesystem::u8path(path), std::ios::binary);
    std::ostringstream content;
    content << in.rdbuf();
    text = content.str();
    if (!in || text.empty()) {
        s2s_log(S2S_LOG_ERROR, "[Main] FATAL: nothing to read in %s", path);
        return false;
    }
    return true;
}

// The languages of the system by preference, named the way the page names
// the languages of the browser, the talker's own names. Only the languages
// the talker knows are kept: under auto the first one gives the voice its
// language.
static std::vector<std::string> system_languages() {
    std::vector<std::string> tags;
#if defined(_WIN32)
    ULONG n    = 0;
    ULONG size = 0;
    if (GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, &n, nullptr, &size) && size) {
        std::vector<wchar_t> list(size);
        if (GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, &n, list.data(), &size)) {
            for (const wchar_t * tag = list.data(); *tag; tag += wcslen(tag) + 1) {
                std::string ascii;
                for (const wchar_t * c = tag; *c; c++) {
                    ascii += (char) *c;
                }
                tags.push_back(ascii);
            }
        }
    }
#elif defined(__APPLE__)
    CFArrayRef list = CFLocaleCopyPreferredLanguages();
    for (CFIndex i = 0; list && i < CFArrayGetCount(list); i++) {
        char tag[64];
        if (CFStringGetCString((CFStringRef) CFArrayGetValueAtIndex(list, i), tag, sizeof(tag),
                               kCFStringEncodingUTF8)) {
            tags.push_back(tag);
        }
    }
    if (list) {
        CFRelease(list);
    }
#else
    for (const char * variable : { "LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG" }) {
        std::stringstream value(getenv(variable) ? getenv(variable) : "");
        for (std::string tag; std::getline(value, tag, ':');) {
            tags.push_back(tag);
        }
    }
#endif
    static const char * names[][2] = {
        { "zh", "chinese"    },
        { "en", "english"    },
        { "ja", "japanese"   },
        { "ko", "korean"     },
        { "de", "german"     },
        { "fr", "french"     },
        { "ru", "russian"    },
        { "pt", "portuguese" },
        { "es", "spanish"    },
        { "it", "italian"    },
    };
    std::vector<std::string> languages;
    for (const std::string & tag : tags) {
        for (const auto & name : names) {
            if (tag.compare(0, 2, name[0]) == 0 &&
                std::find(languages.begin(), languages.end(), name[1]) == languages.end()) {
                languages.push_back(name[1]);
            }
        }
    }
    return languages;
}

static void list_devices(const ma_device_info * infos, ma_uint32 count) {
    for (ma_uint32 i = 0; i < count; i++) {
        printf("  %u  %s%s\n", i, infos[i].name, infos[i].isDefault ? " (default)" : "");
    }
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);

    // Every thread this host starts names itself; the one that logs without
    // a name is the compute worker of qwentts.
    s2s_log_thread("Main");
    s2s_log_thread_default("TTS");
    g_color = console_colors();

    std::string models_dir = "models";
    std::string voices_dir = "voices";

    std::string mode;
    std::string instructions;

    std::string llm_url;
    std::string llm_model;
    int         llm_timeout = -1;
    std::string llm_key;
    std::string reasoning_effort;

    std::vector<mcp_server_params> mcp;
    std::vector<std::string>       tools;
    int                            tool_timeout = -1;
    int                            max_rounds   = -1;

    std::string voice;
    std::string effect;
    std::string language;

    bool list    = false;
    int  mic     = -1;
    int  speaker = -1;

    tts_engine engine;
    Cli        cli;

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
        } else if (arg == "--mode" && has_value) {
            mode = argv[++i];
        } else if (arg == "--instructions-file" && has_value) {
            if (!read_text(argv[++i], instructions)) {
                return 1;
            }
        } else if (arg == "--llm-url" && has_value) {
            llm_url = argv[++i];
        } else if (arg == "--llm-model" && has_value) {
            llm_model = argv[++i];
        } else if (arg == "--llm-timeout" && has_value) {
            llm_timeout = atoi(argv[++i]);
        } else if (arg == "--llm-key-file" && has_value) {
            if (!read_key(argv[++i], llm_key)) {
                return 1;
            }
        } else if (arg == "--reasoning-effort" && has_value) {
            reasoning_effort = argv[++i];
        } else if (arg == "--mcp" && has_value) {
            mcp_server_params server;
            server.url = argv[++i];
            mcp.push_back(server);
        } else if (arg == "--mcp-key-file" && has_value) {
            if (mcp.empty()) {
                s2s_log(S2S_LOG_ERROR, "[Main] FATAL: --mcp-key-file needs an --mcp before it");
                return 1;
            }
            if (!read_key(argv[++i], mcp.back().api_key)) {
                return 1;
            }
        } else if (arg == "--tool" && has_value) {
            tools.push_back(argv[++i]);
        } else if (arg == "--tool-timeout" && has_value) {
            tool_timeout = atoi(argv[++i]);
        } else if (arg == "--max-rounds" && has_value) {
            max_rounds = atoi(argv[++i]);
        } else if (arg == "--voice" && has_value) {
            voice = argv[++i];
        } else if (arg == "--effect" && has_value) {
            effect = argv[++i];
        } else if (arg == "--language" && has_value) {
            language = argv[++i];
        } else if (arg == "--list-devices") {
            list = true;
        } else if (arg == "--mic" && has_value) {
            mic = atoi(argv[++i]);
        } else if (arg == "--speaker" && has_value) {
            speaker = atoi(argv[++i]);
        } else if (arg == "--no-aec") {
            cli.aec = false;
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

    ma_context context;
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) {
        s2s_log(S2S_LOG_ERROR, "[Audio] FATAL: no audio backend");
        return 1;
    }

    ma_device_info * speakers   = nullptr;
    ma_device_info * mics       = nullptr;
    ma_uint32        n_speakers = 0;
    ma_uint32        n_mics     = 0;
    if (ma_context_get_devices(&context, &speakers, &n_speakers, &mics, &n_mics) != MA_SUCCESS) {
        s2s_log(S2S_LOG_ERROR, "[Audio] FATAL: cannot list the devices");
        ma_context_uninit(&context);
        return 1;
    }

    if (list) {
        printf("Microphones, %s:\n", ma_get_backend_name(context.backend));
        list_devices(mics, n_mics);
        printf("Loudspeakers, %s:\n", ma_get_backend_name(context.backend));
        list_devices(speakers, n_speakers);
        ma_context_uninit(&context);
        return 0;
    }

    if (mic >= (int) n_mics || speaker >= (int) n_speakers) {
        s2s_log(S2S_LOG_ERROR, "[Audio] FATAL: no such %s, see --list-devices",
                mic >= (int) n_mics ? "microphone" : "loudspeaker");
        ma_context_uninit(&context);
        return 1;
    }

    ConversationSetup setup;
    ModelFiles        files;
    if (!models_load(models_dir, voices_dir, engine, files, setup)) {
        ma_context_uninit(&context);
        return 1;
    }

    std::string labels;
    for (const std::string & label : tts_bridge_voices(setup.models.tts)) {
        labels += (labels.empty() ? "" : ", ") + label;
    }
    s2s_log(S2S_LOG_INFO, "[Load] Voices: %s", labels.c_str());

    ma_pcm_rb_init(ma_format_f32, 2, CLI_CAPTURE_SECONDS * S2S_INPUT_RATE, nullptr, nullptr, &cli.audio.capture);
    ma_pcm_rb_init(ma_format_f32, 1, CLI_PLAYBACK_SECONDS * S2S_INPUT_RATE, nullptr, nullptr, &cli.audio.playback);

    ma_device_config config         = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate               = S2S_INPUT_RATE;
    config.periodSizeInMilliseconds = CLI_PERIOD_MS;
    config.capture.format           = ma_format_f32;
    config.capture.channels         = 1;
    config.capture.pDeviceID        = mic >= 0 ? &mics[mic].id : nullptr;
    config.playback.format          = ma_format_f32;
    config.playback.channels        = 1;
    config.playback.pDeviceID       = speaker >= 0 ? &speakers[speaker].id : nullptr;
    config.dataCallback             = on_audio;
    config.pUserData                = &cli.audio;

    ma_device device;
    if (ma_device_init(&context, &config, &device) != MA_SUCCESS) {
        s2s_log(S2S_LOG_ERROR, "[Audio] FATAL: cannot open the microphone and the loudspeaker");
        models_free(setup.models);
        ma_context_uninit(&context);
        return 1;
    }
    s2s_log(S2S_LOG_INFO, "[Audio] %s, microphone %s at %u Hz, loudspeaker %s at %u Hz, echo cancellation %s",
            ma_get_backend_name(context.backend), device.capture.name, device.capture.internalSampleRate,
            device.playback.name, device.playback.internalSampleRate, cli.aec ? "on" : "off");

    Connection * conn = conn_open(&setup, CLI_CONNECTION, on_event, &cli);
    if (!conn) {
        ma_device_uninit(&device);
        models_free(setup.models);
        ma_context_uninit(&context);
        return 1;
    }

    // The session as the command line describes it: every field left out
    // takes the default of the engine.
    rt_frame         update  = rt_frame_begin("session.update");
    yyjson_mut_val * session = yyjson_mut_obj_add_obj(update.doc, update.root, "session");
    const auto       str     = [&update](yyjson_mut_val * object, const char * key, const std::string & value) {
        if (!value.empty()) {
            yyjson_mut_obj_add_strncpy(update.doc, object, key, value.c_str(), value.size());
        }
    };
    str(session, "echo", cli.aec ? "server" : "off");
    str(session, "mode", mode);
    str(session, "instructions", instructions);
    str(session, "llm_url", llm_url);
    str(session, "llm_model", llm_model);
    str(session, "llm_key", llm_key);
    if (llm_timeout > 0) {
        yyjson_mut_obj_add_int(update.doc, session, "llm_timeout_sec", llm_timeout);
    }
    yyjson_mut_val * tool_names = yyjson_mut_obj_add_arr(update.doc, session, "tools");
    for (const std::string & tool : tools) {
        yyjson_mut_arr_add_strn(update.doc, tool_names, tool.c_str(), tool.size());
    }
    yyjson_mut_val * servers = yyjson_mut_obj_add_arr(update.doc, session, "mcp");
    for (const mcp_server_params & server : mcp) {
        yyjson_mut_val * entry = yyjson_mut_arr_add_obj(update.doc, servers);
        str(entry, "url", server.url);
        str(entry, "key", server.api_key);
    }
    yyjson_mut_val * sampling = yyjson_mut_obj_add_obj(update.doc, session, "sampling");
    str(sampling, "reasoning_effort", reasoning_effort);
    if (tool_timeout > 0) {
        yyjson_mut_obj_add_int(update.doc, session, "tool_timeout_sec", tool_timeout);
    }
    if (max_rounds > 0) {
        yyjson_mut_obj_add_int(update.doc, session, "max_rounds", max_rounds);
    }
    yyjson_mut_val * tts = yyjson_mut_obj_add_obj(update.doc, session, "tts");
    str(tts, "voice", voice);
    str(tts, "effect", effect);
    str(tts, "language", language);
    const std::vector<std::string> languages = system_languages();
    yyjson_mut_val *               spoken    = yyjson_mut_obj_add_arr(update.doc, tts, "browser_languages");
    for (const std::string & name : languages) {
        yyjson_mut_arr_add_strn(update.doc, spoken, name.c_str(), name.size());
    }

    // From here on this thread is the reader of the connection.
    s2s_log_thread(("Reader-" + std::to_string(CLI_CONNECTION)).c_str());
    conn_frame(conn, rt_frame_end(update));
    {
        std::lock_guard<std::mutex> lock(cli.mutex);
        cli.outbox.push_back(context_history(cli.context));
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#ifndef _WIN32
    // A write to a socket the endpoint or an MCP server closed fails with
    // EPIPE instead of ending the process.
    signal(SIGPIPE, SIG_IGN);
#endif

    if (ma_device_start(&device) != MA_SUCCESS) {
        s2s_log(S2S_LOG_ERROR, "[Audio] FATAL: cannot start the microphone and the loudspeaker");
        g_stop = true;
    } else {
        s2s_log(S2S_LOG_INFO, "[Audio] Listening, Ctrl+C to quit");
    }

    std::vector<float>       pair(2 * CLI_FRAME_SAMPLES);
    std::vector<float>       mic_frame(CLI_FRAME_SAMPLES);
    std::vector<float>       ref_frame(CLI_FRAME_SAMPLES);
    std::vector<std::string> outbox;

    while (!g_stop && !conn_stopped(conn)) {
        {
            std::lock_guard<std::mutex> lock(cli.mutex);
            if (context_played(cli.context, cli.audio)) {
                close_answer(&cli);
            }
            outbox.swap(cli.outbox);
        }
        for (const std::string & frame : outbox) {
            conn_frame(conn, frame);
        }
        outbox.clear();

        if (ma_pcm_rb_available_read(&cli.audio.capture) < CLI_FRAME_SAMPLES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(CLI_POLL_MS));
            continue;
        }
        ma_uint32 got = 0;
        while (got < CLI_FRAME_SAMPLES) {
            ma_uint32 n     = CLI_FRAME_SAMPLES - got;
            void *    chunk = nullptr;
            ma_pcm_rb_acquire_read(&cli.audio.capture, &n, &chunk);
            memcpy(pair.data() + 2 * got, chunk, 2 * n * sizeof(float));
            ma_pcm_rb_commit_read(&cli.audio.capture, n);
            got += n;
        }

        // The reference only travels to the canceller, and only when the
        // frame played something: silence is what its absence means.
        bool played = false;
        for (size_t i = 0; i < CLI_FRAME_SAMPLES; i++) {
            mic_frame[i] = pair[2 * i];
            ref_frame[i] = pair[2 * i + 1];
            played       = played || ref_frame[i] != 0.0f;
        }
        const std::string audio = rt_float_to_pcm16_base64(mic_frame.data(), CLI_FRAME_SAMPLES);
        const std::string reference =
            cli.aec && played ? rt_float_to_pcm16_base64(ref_frame.data(), CLI_FRAME_SAMPLES) : "";
        rt_frame append = rt_frame_begin("input_audio_buffer.append");
        rt_frame_str(append, "audio", audio);
        if (!reference.empty()) {
            rt_frame_str(append, "reference", reference);
        }
        conn_frame(conn, rt_frame_end(append));
    }

    g_stop = true;
    ma_device_uninit(&device);
    conn_close(conn);
    ma_pcm_rb_uninit(&cli.audio.playback);
    ma_pcm_rb_uninit(&cli.audio.capture);
    ma_context_uninit(&context);
    models_free(setup.models);
    return 0;
}
