# s2s.cpp

Local AI voice assistant, powered by GGML. C++17 speech to speech loop:
voice activity detection, end of turn detection, speech recognition and
speech synthesis in one binary, driving any OpenAI compatible chat
completions endpoint. Runs on CUDA, Vulkan, SYCL, Metal and CPU.

## Features

- Full duplex conversation loop: listen, detect the end of the turn,
  transcribe, query the LLM, speak, and yield the floor the moment the
  user speaks again
- Universal echo cancellation: talk to it on laptop speakers, no
  headphones, on every browser. The client streams the exact audio it
  played next to the microphone, and LocalVQE v1.3, a DeepVQE derivative
  ported to GGML, removes the assistant's voice (and the noise and the
  reverberation) in the server before anything listens
- Speaks as soon as the LLM has written its first sentence: the request
  streams over SSE, and each sentence is synthesized while the model
  writes the next one
- Silero VAD on CPU, one 32 ms window at a time, with two thresholds, so
  a dip inside a word or a voice the echo canceller left fainter does not
  cut the speech, and a minimum length, so a breath never opens a turn
- Smart Turn v3.2 end of turn classifier, called only on a speech to
  silence boundary, on a sliding window of the stream, so a thinking
  pause is not mistaken for a finished sentence; the answer is computed
  at once but stays silent for a short grace, so a breath inside a
  sentence never lets the assistant cut in, and a speaker who goes on
  before hearing anything of the answer resumes the same turn,
  recognized again as one utterance
- Parakeet TDT 0.6B v3 recognition, 25 European languages with
  punctuation and casing, non autoregressive duration prediction
- Qwen3-TTS 1.7B Base synthesis through the qwentts.cpp C ABI, in
  process, streamed sentence by sentence, every sentence cloned from the
  same reference voice, and optional batching of concurrent sessions on
  the GPU
- Barge-in: the synthesis stops, the LLM request is aborted, and the
  conversation keeps only the sentences the user really heard
- OpenAI Realtime protocol over WebSocket, with the conversation owned by
  the client: the server keeps nothing between two turns
- Embedded web UI plus `s2s.js`, the same client as a standalone ES
  module to drop on any page

## Architecture

1. Microphone and played reference, PCM16 at 24 kHz over the Realtime WebSocket
2. Decimation to 16 kHz
3. LocalVQE: echo, noise and reverberation removed
4. Silero VAD: speech detected window by window
5. Smart Turn: end of turn decided
6. Parakeet TDT: turn transcribed
7. OpenAI chat completions endpoint (external): answer streamed
8. Sentence split and text clean: synthesis units
9. qwentts.cpp: answer spoken, unit by unit
10. WebSocket back to the speaker

Everything except the LLM runs inside `s2s-server`. LocalVQE, Parakeet and
Qwen3-TTS run on the best GPU found, Silero and Smart Turn on the CPU. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the threads, the turn
state machine and the protocol.

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

`--quant` and `--tts` pick the ASR and TTS quants. The server loads, for
each model, the largest one present in `models/`, at the best quant up to
Q8_0. The TTS checkpoints belong to the qwentts.cpp submodule.

## Voices

`voices/` holds the reference voices, loaded once at startup and listed
in the web UI. A voice is a `<name>.spk` speaker embedding, plus an
optional `<name>.rvq` and `<name>.txt` pair, the reference codes and
their transcript, that turns on ICL: the talker then continues the
reference recording on every sentence, timbre, pace and accent included.
The files come from `qwen-codec --talker` of the qwentts.cpp submodule
run on a clean recording, and are tied to the hidden size of the 1.7B
talker. The voice `freeman` ships as the default.

## Run

```
./server.sh                      # then open http://localhost:8088
```

The command line only holds what belongs to the host: `--models`,
`--voices`, `--host`, `--port`, the security allowlists `--origin` and `--llm-host`,
the endpoint when the server owns it, and the TTS engine options
(`--max-batch`, `--no-fa`, `--clamp-fp16`, `--codec-chunk-dur`).
Everything else belongs to the client: mode, prompt, voice, sampling,
echo cancellation and turn detection travel in `session.update`, and
their defaults are published on `/props`. The endpoint travels in
`session.update` too when the server has none of its own, and is never
published.

A page shown to other people must not hold the endpoint key, so the
server can own the endpoint. The playground then hides its endpoint
fields, and the client code it copies opens with this command, filled in:

```
echo "YOUR_API_KEY" > llm.key
./build/s2s-server --origin https://your-site.example \
    --llm-url https://api.example.com/v1 --llm-model model-name --llm-key-file llm.key
```

The endpoint is any OpenAI compatible server: llama-server, Ollama,
LM Studio, a cloud API. The voice reads the text in the language it is
written in, so the system prompt decides the language of the answers.

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
