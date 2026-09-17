# s2s.cpp

Local AI voice assistant, powered by GGML. C++17 speech to speech loop:
voice activity detection, end of turn detection, speech recognition and
speech synthesis in one binary, driving any OpenAI compatible chat
completions endpoint. Runs on CUDA, Vulkan, SYCL and Metal.

## Features

- Full duplex conversation loop: listen, detect the end of the turn,
  transcribe, query the LLM, speak, and yield the floor the moment the
  user speaks again
- Silero VAD on CPU, one 32 ms window at a time, with hysteresis so a
  breath never opens a turn
- Smart Turn v3.2 end of turn classifier, called only on a speech to
  silence boundary, so a thinking pause is not mistaken for a finished
  sentence
- Parakeet TDT 0.6B v3 recognition, 25 European languages with
  punctuation and casing, non autoregressive duration prediction
- Qwen3-TTS synthesis through the qwentts.cpp C ABI, in process and
  streamed sentence by sentence
- Barge-in: the TTS is cancelled, the LLM request is aborted and the
  assistant message is truncated to what was really played
- OpenAI Realtime protocol over WebSocket, so existing clients connect
  unchanged
- Embedded web UI plus `s2s.js`, the same client as a standalone ES
  module to drop on any page

## Architecture

```
mic -> WebSocket -> Silero VAD -> Smart Turn -> Parakeet TDT -+
                                                              |
                                          OpenAI chat completions (external)
                                                              |
speaker <- WebSocket <- sentence split <- qwentts.cpp <-------+
```

Everything except the LLM runs inside `s2s-server`. A GPU is required:
the ASR and the TTS share one device context.

## Build

```
git clone https://github.com/ServeurpersoCom/s2s.cpp.git
cd s2s.cpp
git submodule update --init          # ggml and qwentts.cpp, not the ggml inside qwentts.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildsycl.sh                   # Intel GPU (SYCL)
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

## Models

```
./models.sh                      # prebuilt GGUF -> models/
./checkpoints.sh                 # upstream checkpoints -> checkpoints/
./convert.py                     # checkpoints -> GGUF
./quantize.sh                    # Q4_K_M to Q8_0 derived from the F32 base
```

## Run

```
./server.sh                      # then open http://localhost:8088
```

`--llm-url` points at any OpenAI compatible endpoint: llama-server,
Ollama, LM Studio, a cloud API. The microphone needs a secure context,
which `http://localhost` satisfies; serving the UI from a LAN address or
a domain requires HTTPS and `wss://`.

## Using the component on your own page

`s2s.js` is the same component the bundled page runs, built from the same
sources, with no dependency and no markup of its own: it owns the microphone,
the socket and the turn state, and reports through callbacks. A host draws
whatever it wants around it, or nothing at all.

```html
<script type="module">
    import { S2S } from "https://your-host/s2s.js";

    const s2s = new S2S({
        url: "wss://your-host/v1/realtime",
        mode: "conversation",          // or "loopback", no endpoint in the path
        instructions: "You are a voice assistant. Answer in one or two sentences.",
        llmUrl: "http://127.0.0.1:8080/v1",
        llmModel: "local",
        tts: { speaker: "ryan", language: "french" },
        vad: { threshold: 0.6, minSilenceMs: 64 },
        turn: { threshold: 0.5, maxWaitMs: 2000 }
    });

    s2s.on("state", (state) => console.log(state));
    s2s.on("user_text", (text) => console.log("user:", text));
    s2s.on("assistant_text", (text) => console.log("assistant:", text));

    // the conversation belongs to the page: persist it, reload it, edit it
    s2s.setHistory(JSON.parse(localStorage.getItem("chat") ?? "[]"));
    s2s.on("history", (messages) => localStorage.setItem("chat", JSON.stringify(messages)));

    // start() needs a user gesture and a secure context
    document.querySelector("button").onclick = () => s2s.start();
</script>
```

The settings panel of the bundled page edits exactly these options, nothing
more: it is the visible version of what a host writes in code.

## License

MIT
