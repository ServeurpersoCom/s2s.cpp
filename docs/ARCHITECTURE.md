# Architecture

s2s.cpp is a speech to speech loop. One binary owns the microphone
stream, decides when the user has finished speaking, transcribes the
turn, asks an external LLM, synthesizes the answer and streams it back,
while staying ready to be interrupted at any instant.

## Models

| Stage | Model | Source | Device |
| --- | --- | --- | --- |
| Voice activity | Silero VAD v5 | onnx-community/silero-vad | CPU, 1 thread |
| End of turn | Smart Turn v3.2 | pipecat-ai/smart-turn-v3 | CPU, 1 thread |
| Recognition | Parakeet TDT 0.6B v3 | nvidia/parakeet-tdt-0.6b-v3 | shared GPU |
| Synthesis | Qwen3-TTS | qwentts.cpp submodule | shared GPU |
| Reasoning | any OpenAI chat completions endpoint | external | external |

The VAD runs on every 32 ms window, so it stays on the CPU with a single
thread: the model is 2 MB and a GPU dispatch would cost more than the
work. Smart Turn runs once per speech to silence boundary, on the same
terms. Parakeet and Qwen3-TTS share one backend and one device context.

## Threading

| Thread | Owns |
| --- | --- |
| WebSocket, one per client | frame decode, PCM ring buffer |
| session, one per client | VAD window loop, Smart Turn, turn state machine, outgoing events |
| ASR worker, one process wide | the Parakeet backend and its job queue |
| TTS worker, inside libqwen | the Qwen3-TTS backend |
| LLM, one per response | SSE read loop, aborted by closing the socket |

Every GGML graph is computed by the worker that owns its backend. A
session thread never touches a device: it posts a job and waits.

## Turn state machine

```
IDLE --speech >= min_speech_ms--> USER_SPEAKING
USER_SPEAKING --silence >= min_silence_ms--> PENDING_END
PENDING_END --Smart Turn complete--> THINKING
PENDING_END --Smart Turn incomplete, then wait > turn_max_wait_ms--> THINKING
PENDING_END --speech >= min_speech_continuation_ms--> USER_SPEAKING
THINKING --first TTS chunk--> SPEAKING
SPEAKING --response done--> IDLE
THINKING, SPEAKING --speech >= min_speech_ms--> USER_SPEAKING (barge-in)
```

Defaults, inherited from the Python reference this project replaces:

| Parameter | Default | Role |
| --- | --- | --- |
| `vad_threshold` | 0.6 | Silero speech probability |
| `min_speech_ms` | 384 | opens a turn, and arms a barge-in |
| `min_speech_continuation_ms` | 192 | reopens a turn from `PENDING_END` |
| `min_silence_ms` | 64 | speech to silence boundary |
| `speech_pad_ms` | 30 | margin kept around a segment |
| `turn_threshold` | 0.5 | Smart Turn completion probability |
| `turn_max_wait_ms` | 2000 | commit an incomplete turn anyway |

Smart Turn reads the last 8 seconds of the current turn. A turn judged
incomplete keeps its identity: reopening increments a revision counter
instead of starting a new turn, so the transcript of the whole utterance
reaches the LLM as one message.

## Barge-in

Speech detected while the assistant holds the floor cancels the response
in three steps:

1. the TTS cancel flag is raised, the LLM socket is closed and the
   outgoing audio queue is dropped,
2. `response.cancelled` tells the client to flush its playback buffer,
3. the client answers with `conversation.item.truncate` and the played
   sample count, and the server trims the assistant message to the last
   sentence that was actually heard.

The server knows the sample count of every synthesized sentence, so the
trim lands on a sentence boundary and the conversation history matches
what the user experienced.

## Sentence splitting

The LLM stream is split on sentence terminators and line breaks, with a
length cap that falls back to the last comma or space, because a model
can emit 300 characters without punctuation. Markdown and emoji are
stripped before synthesis. The first unit uses a shorter threshold to
cut the time to first audio.

## Protocol

`WS /v1/realtime`, a subset of the OpenAI Realtime API. Audio is PCM16
at 24 kHz, base64 encoded, which matches the TTS output rate and needs a
single resample on the way in.

Client to server: `session.update`, `input_audio_buffer.append`,
`input_audio_buffer.commit`, `response.create`, `response.cancel`,
`conversation.item.truncate`.

Server to client: `session.created`, `session.updated`,
`input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`,
`conversation.item.input_audio_transcription.completed`,
`response.created`, `response.output_audio.delta`,
`response.output_audio_transcript.delta`, `response.done`, `error`.

WebSocket handshakes bypass CORS, so the server checks the `Origin`
header against the `--origin` allowlist.

## Module map

| File | Role |
| --- | --- |
| `src/backend.h` | backend selection, CPU thread count, log dedup |
| `src/gguf-weights.h` | GGUF reader and weight upload |
| `src/static-graph.h` | direct galloc path with scheduler fallback |
| `src/graph-arena.h` | graph context arena |
| `src/audio-resample.h` | polyphase resampler, any rate to 16 or 24 kHz |
| `src/audio-mel.h` | log mel frontends, NeMo and Whisper variants |
| `src/silero.h` | VAD lib, C ABI |
| `src/smart-turn.h` | end of turn lib, C ABI |
| `src/parakeet.h` | ASR lib, C ABI |
| `src/s2s-session.h` | turn state machine |
| `src/llm-client.h` | chat completions SSE client |
| `src/sentence-split.h` | streaming text to synthesis units |
| `src/realtime-proto.h` | Realtime event encode and decode |
| `tools/s2s-server.cpp` | the product binary |
| `tools/webui` | Svelte demo and the `s2s.js` client lib |

Each model lib is self contained: its own GGUF, its own C ABI, no
dependency on the session layer. Extracting one into a standalone
project is a move, not a rewrite.

## GGUF layout

One file per model, named `<model>-<quant>.gguf`. Every dimension is
read from the metadata, never hardcoded: the same code path loads a
future cache aware streaming encoder as long as the converter writes the
same keys.

## Validation

Every stage has a parity harness in `tests/`: the C++ target dumps
tensors, the Python script loads the reference and reports cosine
similarity.

| Stage | Reference |
| --- | --- |
| Silero VAD | onnxruntime |
| Smart Turn | onnxruntime |
| Parakeet mel, encoder, decoder | transformers `ParakeetForTDT` |
| Full transcript | transformers, exact string match |

The sweep covers CUDA, Vulkan and CPU against F32, Q8_0 and Q4_K_M.
