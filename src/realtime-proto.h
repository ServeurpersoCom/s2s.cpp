#pragma once
// realtime-proto.h: the OpenAI Realtime subset s2s-server speaks
//
// One JSON object per WebSocket frame, each carrying a "type". Audio travels
// as base64 PCM16 at 24 kHz in both directions, which is the format the
// upstream protocol fixes and, by luck, the rate the codec already produces.
//
// Client to server:
//   session.update                   settings: mode, endpoint, voice, thresholds
//   input_audio_buffer.append        base64 PCM16 from the microphone
//   input_audio_buffer.commit        push to talk release, commits the turn
//   response.cancel                  barge-in raised by the client
//   conversation.history             the conversation, owned by the client
//
// Server to client:
//   session.created, session.updated
//   input_audio_buffer.speech_started, input_audio_buffer.speech_stopped
//   conversation.item.input_audio_transcription.completed
//   response.created
//   response.output_text.delta       what the model writes, as it writes it
//   response.output_audio.delta, response.output_audio_transcript.delta
//   response.done, response.cancelled
//   error
//
// Unknown types and unknown fields are ignored rather than rejected: a client
// written against a later revision of the protocol still works here.

#include "yyjson.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// The rate the protocol carries in both directions.
#define SAMPLE_RATE_24K 24000

enum rt_client_event {
    RT_CLIENT_UNKNOWN = 0,
    RT_CLIENT_SESSION_UPDATE,
    RT_CLIENT_AUDIO_APPEND,
    RT_CLIENT_AUDIO_COMMIT,
    RT_CLIENT_RESPONSE_CANCEL,
    RT_CLIENT_HISTORY,
};

// One line of the conversation, in the shape the endpoint takes.
struct rt_message {
    std::string role;
    std::string content;
};

// Settings a client may change mid session. Empty strings and negative
// numbers mean "leave as is", so a partial session.update is legal.
struct rt_session_patch {
    std::string mode;  // conversation or loopback
    std::string llm_url;
    std::string llm_model;
    std::string llm_key;
    std::string system_prompt;
    std::string voice;
    std::string language;
    float       temperature       = -1.0f;
    float       top_p             = -1.0f;
    float       top_k             = -1.0f;
    float       min_p             = -1.0f;
    float       max_tokens        = -1.0f;
    float       presence_penalty  = -100.0f;
    float       frequency_penalty = -100.0f;
    float       seed              = -1.0f;
    std::string tts_speaker;
    std::string tts_language;
    float       tts_temperature            = -1.0f;
    float       tts_top_k                  = -1.0f;
    float       tts_top_p                  = -1.0f;
    float       tts_repetition_penalty     = -1.0f;
    float       tts_subtalker_temperature  = -1.0f;
    float       tts_subtalker_top_k        = -1.0f;
    float       tts_subtalker_top_p        = -1.0f;
    float       tts_max_new_tokens         = -1.0f;
    float       tts_seed                   = -1.0f;
    float       tts_min_chars              = -1.0f;
    float       tts_chars_per_second       = -1.0f;
    float       tts_margin_seconds         = -1.0f;
    float       llm_timeout_sec            = -1.0f;
    float       vad_threshold              = -1.0f;
    int         min_speech_ms              = -1;
    int         min_speech_continuation_ms = -1;
    int         min_silence_ms             = -1;
    int         speech_pad_ms              = -1;
    float       turn_threshold             = -1.0f;
    int         turn_max_wait_ms           = -1;
};

struct rt_client_message {
    rt_client_event         type = RT_CLIENT_UNKNOWN;
    rt_session_patch        patch;
    std::vector<float>      audio;  // decoded from base64 PCM16
    std::vector<rt_message> messages;
};

static const char RT_BASE64_ALPHABET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string rt_base64_encode(const uint8_t * data, size_t size) {
    std::string out;
    out.reserve((size + 2) / 3 * 4);

    for (size_t i = 0; i < size; i += 3) {
        const uint32_t a      = data[i];
        const uint32_t b      = i + 1 < size ? data[i + 1] : 0;
        const uint32_t c      = i + 2 < size ? data[i + 2] : 0;
        const uint32_t triple = (a << 16) | (b << 8) | c;

        out += RT_BASE64_ALPHABET[(triple >> 18) & 0x3f];
        out += RT_BASE64_ALPHABET[(triple >> 12) & 0x3f];
        out += i + 1 < size ? RT_BASE64_ALPHABET[(triple >> 6) & 0x3f] : '=';
        out += i + 2 < size ? RT_BASE64_ALPHABET[triple & 0x3f] : '=';
    }
    return out;
}

static int rt_base64_value(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

static std::vector<uint8_t> rt_base64_decode(const char * data, size_t size) {
    std::vector<uint8_t> out;
    out.reserve(size / 4 * 3);

    uint32_t bits  = 0;
    int      count = 0;
    for (size_t i = 0; i < size; i++) {
        const int value = rt_base64_value(data[i]);
        if (value < 0) {
            continue;  // padding, whitespace, newlines inside the payload
        }
        bits = (bits << 6) | (uint32_t) value;
        count += 6;
        if (count >= 8) {
            count -= 8;
            out.push_back((uint8_t) ((bits >> count) & 0xff));
        }
    }
    return out;
}

// PCM16 little endian to float, and back. The client speaks PCM16 because
// that is what the protocol carries and what an AudioWorklet produces
// cheaply.
static void rt_pcm16_to_float(const std::vector<uint8_t> & bytes, std::vector<float> & out) {
    out.resize(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        const int16_t sample = (int16_t) ((uint16_t) bytes[2 * i] | ((uint16_t) bytes[2 * i + 1] << 8));
        out[i]               = (float) sample / 32768.0f;
    }
}

static std::string rt_float_to_pcm16_base64(const float * pcm, size_t n_samples) {
    std::vector<uint8_t> bytes(n_samples * 2);
    for (size_t i = 0; i < n_samples; i++) {
        float value          = pcm[i];
        value                = value > 1.0f ? 1.0f : (value < -1.0f ? -1.0f : value);
        const int16_t sample = (int16_t) (value * 32767.0f);
        bytes[2 * i]         = (uint8_t) ((uint16_t) sample & 0xff);
        bytes[2 * i + 1]     = (uint8_t) (((uint16_t) sample >> 8) & 0xff);
    }
    return rt_base64_encode(bytes.data(), bytes.size());
}

static std::string rt_json_str(yyjson_val * object, const char * key) {
    yyjson_val * value = yyjson_obj_get(object, key);
    return value && yyjson_is_str(value) ? std::string(yyjson_get_str(value), yyjson_get_len(value)) : std::string();
}

static float rt_json_num(yyjson_val * object, const char * key, float fallback) {
    yyjson_val * value = yyjson_obj_get(object, key);
    return value && yyjson_is_num(value) ? (float) yyjson_get_num(value) : fallback;
}

// Parses one client frame. An unparsable frame comes back as RT_CLIENT_UNKNOWN
// rather than as an error: the caller answers with an error event and keeps
// the session alive.
static rt_client_message rt_parse(const std::string & frame) {
    rt_client_message message;

    yyjson_doc * doc = yyjson_read(frame.c_str(), frame.size(), 0);
    if (!doc) {
        return message;
    }
    yyjson_val *      root = yyjson_doc_get_root(doc);
    const std::string type = rt_json_str(root, "type");

    if (type == "session.update") {
        message.type                = RT_CLIENT_SESSION_UPDATE;
        yyjson_val * session        = yyjson_obj_get(root, "session");
        yyjson_val * fields         = session ? session : root;
        message.patch.mode          = rt_json_str(fields, "mode");
        message.patch.llm_url       = rt_json_str(fields, "llm_url");
        message.patch.llm_model     = rt_json_str(fields, "llm_model");
        message.patch.llm_key       = rt_json_str(fields, "llm_key");
        message.patch.system_prompt = rt_json_str(fields, "instructions");
        message.patch.voice         = rt_json_str(fields, "voice");
        message.patch.language      = rt_json_str(fields, "language");
        message.patch.temperature   = rt_json_num(fields, "temperature", -1.0f);

        // The two models that decide when a turn ends are configured apart,
        // because they are apart: Silero scores every window, Smart Turn only
        // scores a boundary.
        yyjson_val * tts = yyjson_obj_get(fields, "tts");
        if (tts) {
            message.patch.tts_speaker          = rt_json_str(tts, "speaker");
            message.patch.tts_language         = rt_json_str(tts, "language");
            message.patch.tts_min_chars        = rt_json_num(tts, "min_chars", -1.0f);
            message.patch.tts_chars_per_second = rt_json_num(tts, "chars_per_second", -1.0f);
            message.patch.tts_margin_seconds   = rt_json_num(tts, "margin_seconds", -1.0f);

            yyjson_val * sampling = yyjson_obj_get(tts, "sampling");
            if (sampling) {
                message.patch.tts_temperature           = rt_json_num(sampling, "temperature", -1.0f);
                message.patch.tts_top_k                 = rt_json_num(sampling, "top_k", -1.0f);
                message.patch.tts_top_p                 = rt_json_num(sampling, "top_p", -1.0f);
                message.patch.tts_repetition_penalty    = rt_json_num(sampling, "repetition_penalty", -1.0f);
                message.patch.tts_subtalker_temperature = rt_json_num(sampling, "subtalker_temperature", -1.0f);
                message.patch.tts_subtalker_top_k       = rt_json_num(sampling, "subtalker_top_k", -1.0f);
                message.patch.tts_subtalker_top_p       = rt_json_num(sampling, "subtalker_top_p", -1.0f);
                message.patch.tts_max_new_tokens        = rt_json_num(sampling, "max_new_tokens", -1.0f);
                message.patch.tts_seed                  = rt_json_num(sampling, "seed", -1.0f);
            }
        }

        yyjson_val * vad = yyjson_obj_get(fields, "vad");
        if (vad) {
            message.patch.vad_threshold              = rt_json_num(vad, "threshold", -1.0f);
            message.patch.min_speech_ms              = (int) rt_json_num(vad, "min_speech_ms", -1.0f);
            message.patch.min_speech_continuation_ms = (int) rt_json_num(vad, "min_speech_continuation_ms", -1.0f);
            message.patch.min_silence_ms             = (int) rt_json_num(vad, "min_silence_ms", -1.0f);
            message.patch.speech_pad_ms              = (int) rt_json_num(vad, "speech_pad_ms", -1.0f);
        }

        yyjson_val * turn = yyjson_obj_get(fields, "turn");
        if (turn) {
            message.patch.turn_threshold   = rt_json_num(turn, "threshold", -1.0f);
            message.patch.turn_max_wait_ms = (int) rt_json_num(turn, "max_wait_ms", -1.0f);
        }
    } else if (type == "input_audio_buffer.append") {
        message.type            = RT_CLIENT_AUDIO_APPEND;
        const std::string audio = rt_json_str(root, "audio");
        rt_pcm16_to_float(rt_base64_decode(audio.c_str(), audio.size()), message.audio);
    } else if (type == "input_audio_buffer.commit") {
        message.type = RT_CLIENT_AUDIO_COMMIT;
    } else if (type == "response.cancel") {
        message.type = RT_CLIENT_RESPONSE_CANCEL;
    } else if (type == "conversation.history") {
        message.type = RT_CLIENT_HISTORY;

        yyjson_val * messages = yyjson_obj_get(root, "messages");
        if (messages && yyjson_is_arr(messages)) {
            size_t       index = 0;
            size_t       max   = 0;
            yyjson_val * item  = nullptr;
            yyjson_arr_foreach(messages, index, max, item) {
                const std::string role = rt_json_str(item, "role");
                if (!role.empty()) {
                    message.messages.push_back({ role, rt_json_str(item, "content") });
                }
            }
        }
    }

    yyjson_doc_free(doc);
    return message;
}

// Server frames. Each helper writes one complete object: the caller only ever
// hands strings and numbers, never JSON.
static std::string rt_event(const char * type) {
    return std::string("{\"type\":\"") + type + "\"}";
}

static std::string rt_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

static std::string rt_event_text(const char * type, const char * key, const std::string & value) {
    return std::string("{\"type\":\"") + type + "\",\"" + key + "\":\"" + rt_escape(value) + "\"}";
}

static std::string rt_event_audio(const float * pcm, size_t n_samples) {
    return "{\"type\":\"response.output_audio.delta\",\"delta\":\"" + rt_float_to_pcm16_base64(pcm, n_samples) + "\"}";
}

static std::string rt_event_error(const std::string & message) {
    return "{\"type\":\"error\",\"error\":{\"message\":\"" + rt_escape(message) + "\"}}";
}
