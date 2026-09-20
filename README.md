# s2s.cpp

Local AI voice assistant, powered by GGML. C++17 speech to speech loop:
voice activity detection, end of turn detection, speech recognition and
speech synthesis in one binary, driving any OpenAI compatible chat
completions endpoint. Runs on CUDA, Vulkan, SYCL, Metal and CPU.

## Features

- Full duplex: talk over it at any time, it stops at once and remembers
  only what you heard
- Works on laptop speakers, no headphones: the server cancels the echo
  of its own voice, on every browser
- A pause to think does not hand it the floor, and going on before its
  answer starts keeps your sentence whole
- Speaks as soon as the model has written its first sentence
- Clones one reference voice for every sentence, so the voice stays the
  same from the first word to the last
- OpenAI Realtime protocol over WebSocket, the conversation owned by the
  client
- Embedded web UI plus `s2s.js`, the same client as an ES module to drop
  on any page

## Architecture

1. Microphone and played reference, PCM16 at 24 kHz over the Realtime WebSocket
2. Resampling to 16 kHz
3. LocalVQE v1.3: echo, noise and reverberation removed
4. Silero VAD: speech detected window by window
5. Smart Turn v3.2: end of turn decided
6. Parakeet TDT 0.6B v3: turn transcribed, 25 European languages
7. OpenAI chat completions endpoint (external): answer streamed
8. Sentence split and text clean: synthesis units
9. Qwen3-TTS 1.7B Base through qwentts.cpp: answer spoken, unit by unit
10. WebSocket back to the speaker

Everything except the LLM runs inside `s2s-server`. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how: the threads, the
turn state machine, the echo cancellation and the protocol.

## Build

```
git clone https://github.com/ServeurpersoCom/s2s.cpp.git
cd s2s.cpp
git submodule update --init      # ggml and qwentts.cpp, not the ggml inside qwentts.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildsycl.sh                   # Intel GPU (SYCL)
./buildcpu.sh                    # CPU with BLAS
./buildtermux.sh                 # Android, Termux
./buildall.sh                    # CPU, CUDA and Vulkan, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

`GGML_BACKEND` picks a device by name (`CUDA0`, `Vulkan1`...) when the
machine has several.

## Models

```
./models.sh                      # prebuilt GGUF -> models/, Q8_0
./checkpoints.sh                 # upstream checkpoints -> checkpoints/
./convert.py                     # checkpoints -> GGUF
./quantize.sh                    # Q4_K_M to Q8_0 derived from the F32 base
```

`--quant` and `--tts` pick the ASR and TTS quants.

## Voices

Drop a voice in `voices/` and it shows up in the web UI. A `.spk` alone
gives the timbre; with the `.rvq` and `.txt` of a recording next to it,
the voice also keeps its pace and accent, and is offered both ways:

```
freeman.{spk,rvq,txt} reference speech
freeman.spk speaker embedding only
```

`qwen-codec --talker` of the qwentts.cpp submodule makes the three files
from a clean recording. `freeman` ships as the default.

## Run

```
./server.sh                      # then open http://localhost:8088
```

The command line sets the machine, the client sets the conversation:
mode, prompt, voice, sampling and turn detection travel in
`session.update`, their defaults on `/props`. `--help` lists the rest.

The endpoint is any OpenAI compatible server: llama-server, Ollama,
LM Studio, a cloud API. The voice reads the text in the language it is
written in, so the system prompt decides the language of the answers.

A page shown to other people must not hold the endpoint key, so the
server can own the endpoint. The playground then hides its endpoint
fields, and the client code it copies opens with this command, filled in:

```
echo "YOUR_API_KEY" > llm.key
./build/s2s-server --origin https://your-site.example \
    --llm-url https://api.example.com/v1 --llm-model model-name --llm-key-file llm.key
```

The microphone needs a secure context, which `http://localhost`
satisfies; serving the UI from a LAN address or a domain requires HTTPS
and `wss://`.

## Using the component on your own page

The embedded web UI is a playground: tune the voice, the turn detection,
the echo cancellation and the endpoint, talk to it until it feels right,
then press "Copy the client code with your current settings". You get a
ready to paste snippet that loads `s2s.js` from your server with exactly
the options you changed, the base of your own Jarvis-like assistant at
home.

## License

MIT
