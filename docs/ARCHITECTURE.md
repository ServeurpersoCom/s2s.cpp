# Architecture

s2s.cpp is a speech to speech loop. One binary owns the microphone
stream, decides when the user has finished speaking, transcribes the
turn, asks an external LLM, synthesizes the answer and streams it back,
while staying ready to be interrupted at any instant.

## Models

| Stage | Model | Source | Device |
| --- | --- | --- | --- |
| Echo cancellation | LocalVQE v1.3 | LocalAI-io/LocalVQE | best GPU |
| Voice activity | Silero VAD v5 | onnx-community/silero-vad | CPU, 1 thread |
| End of turn | Smart Turn v3.2 | pipecat-ai/smart-turn-v3 | CPU |
| Recognition | Parakeet TDT 0.6B v3 | nvidia/parakeet-tdt-0.6b-v3 | best GPU |
| Synthesis | Qwen3-TTS 1.7B Base | qwentts.cpp submodule | best GPU |
| Reasoning | any OpenAI chat completions endpoint | external | external |

The VAD runs on every 32 ms window, so it stays on the CPU with a single
thread: the model is 2 MB and a GPU dispatch would cost more than the
work. Smart Turn runs once per speech to silence boundary, on the CPU as
well. LocalVQE, Parakeet and Qwen3-TTS take the best device
`ggml_backend_init_best` finds, or the one `GGML_BACKEND` names.

The models are found in `--models` by file name. For each one the server
takes the largest model present, then the best quant up to Q8_0, F32 and
BF16 last; the talker must be a `base` one. The six files really
loaded are logged at startup and published on `/props`.

## Threading

| Thread | Log name | Owns |
| --- | --- | --- |
| main | `Main` | loading, then the listening socket |
| HTTP pool | `HTTP` | the page, `/props`, `/v1/models`, `/log` |
| reader, one per connection | `Reader-N` | the socket, frame decode, 24 to 16 kHz resampling, the echo canceller, the turn session, incoming events |
| responder, one per connection | `Responder-N` | recognition, the LLM stream, the sentence splitter, synthesis |
| writer, one per connection | `Writer-N` | every outgoing frame, in order: a slow client stalls its own writer, never the TTS worker |
| TTS worker, inside qwentts.cpp | `TTS` | the Qwen3-TTS backend and its queue, up to `--max-batch` syntheses per step |
| echo canceller worker | `AEC` | every hop waiting from every connection, in one pass |
| end of turn worker | `SmartTurn` | every end of turn classification, from every connection, one at a time |
| log reader, one per process | | stderr capture, the ring behind `/logs` |

A log line opens with its thread, then its component: `[Reader-4-Session]`
is the turn session of connection 4 on its reader. N numbers the
connections from the start and is never given twice, so one grep follows
one client, reconnections apart. Component names hold no hyphen: the
component is what follows the last one.

The reader keeps consuming audio while the responder talks, which is what
makes the barge-in possible. The responder takes committed turns from a
queue, so a turn spoken during an answer waits for the next one. It copies
the session settings and the conversation when it answers a turn: a
`session.update` lands on the next answer, never under a running one. A
turn that a later one superseded during its recognition gets no answer, and
a closed connection drops the turns still queued.

Every model keeps one context for the whole process. Silero and Parakeet
serialize their compute behind a mutex, Smart Turn on its own worker
thread, which keeps a single OpenMP team of encoder threads whatever
connection asks; LocalVQE and the TTS
batch the connections together, each through its worker.

## Turn state machine

```
IDLE          --speech >= min_speech_ms-->               USER_SPEAKING
USER_SPEAKING --silence >= min_silence_ms-->             PENDING_END
PENDING_END   --turn complete-->                         committed, IDLE
PENDING_END   --incomplete, then turn_max_wait_ms-->     committed, IDLE
PENDING_END   --speech >= min_speech_continuation_ms-->  USER_SPEAKING
committed     --speech >= min_speech_continuation_ms-->  USER_SPEAKING, same turn
committed     --grace over and released-->               final
```

The session is told when the assistant holds the floor. Speech of
`min_speech_ms` during that time raises a barge-in.

Recognition and the endpoint request start at the commit, but the first
unit waits until the session reports the turn final, which takes two
things. The grace: a commit the classifier judged complete keeps its
answer silent for `reopen_grace_ms`, counted on the audio, so a breath
inside a sentence never lets the assistant cut in; a commit forced by
`turn_max_wait_ms` has waited already and has none. The release: the
responder tells the session, under the session lock it shares with the
reader, when its first unit reaches the gate or when the turn ends with
nothing to say. A client that stops streaming after its commit still
hears its answer.

Until the turn is final it stays open, its audio still accumulating, and a
speaker who goes on resumes it: a user who has heard nothing of the answer
is still in the same turn, however long the endpoint takes. The answer to
the previous revision is dropped before anyone hears it, and the next
commit hands the whole utterance to the recognizer again, under the same
turn and a new revision. A revision that a later one overtook before its
recognition is not recognized at all.

A patch applies its thresholds to the running session: the turn in
flight, the model states and the turn numbering are kept.

Defaults, published on `/props` and patchable per session:

| Parameter | Default | Role |
| --- | --- | --- |
| `vad_threshold` | 0.6 | Silero probability where speech starts |
| `vad_neg_threshold` | 0.45 | probability speech has to stay at or above to go on |
| `min_speech_ms` | 384 | opens a turn, and arms a barge-in |
| `min_speech_continuation_ms` | 192 | reopens a turn from `PENDING_END` |
| `min_silence_ms` | 64 | speech to silence boundary |
| `speech_pad_ms` | 500 | audio kept before the speech onset |
| `turn_threshold` | 0.5 | Smart Turn completion probability |
| `turn_max_wait_ms` | 2000 | commit an incomplete turn anyway |
| `reopen_grace_ms` | 800 | least silence kept on the answer to a complete commit |

Smart Turn reads a sliding window of the last 8 seconds of the stream,
fed on every VAD window and independent of the turn, so a short turn is
judged with the context that precedes it. A turn judged incomplete keeps
its identity: reopening increments a revision counter and the audio keeps
accumulating, so the whole utterance reaches the recognizer as one piece.

## Conversation and barge-in

The conversation belongs to the client. It pushes the whole list with
`conversation.history` whenever it changes; the responder copies it when
it picks a turn up and appends the new transcript. A user message names
its turn in `item`, the `item_id` of its transcript, so an earlier
revision of the turn being answered is left out of the request. The
server keeps nothing between two turns.

The transcript of each synthesis unit leaves right before its first audio
chunk, so the client only ever receives text that has sound behind it.
The client notes the sample where each unit starts in its playback
stream, and its playback worklet counts the samples really played. An
answer closes with the units whose audio started:

- on `response.done`, once the last queued sample has been played,
- on a barge-in: the server raises the cancel flag, which stops the
  synthesis and aborts the LLM request, and sends `response.cancelled`;
  the client flushes its playback and keeps what was heard.

## Sentence splitting

The LLM is queried with `stream: true` and read as SSE, and the voice
starts on the first complete sentence while the model keeps writing:
the time to first audio is the time to the first sentence, not to the
whole answer.

The stream is split into synthesis units, the sentence and nothing
else. A terminator is confirmed by the character after it, which keeps
decimals and abbreviations whole, and a line break ends a unit too. There
is no length cap: cutting on a character count lands mid syntagm.

Markdown markers and emoji (Extended_Pictographic and the emoji
components of the Unicode emoji data, `src/emoji.h`) are dropped, and a
unit left without a single letter or digit is not emitted: a talker given
nothing to say never finds its end of speech.

The TTS bridge then skips a unit shorter than `tts_min_chars` characters,
and bounds each synthesis to a frame budget derived from the text length,
`tts_chars_per_second` plus `tts_margin_seconds`. All three are session
parameters.

Every unit is a clone of one reference voice, read once at startup from
`--voices`: a `<name>.spk` speaker embedding, and when the voice has them
a `<name>.rvq` and `<name>.txt` pair that turns on ICL. The prompt carries
no language id, so the talker reads the text in its own language and the
reference sets the accent. `/props` lists the voices by name in
`tts_voices`, the first one being the default, and a session picks one
with `tts.voice`.

## Echo cancellation

The client picks it with `echo`, `server` by default.

`server` works on every browser. The client plays and captures through one
duplex AudioWorklet, so each 20 ms microphone frame leaves with the samples
played during the same render quanta: the exact echo reference, with only
the loudspeaker to microphone path left to estimate. The microphone is
taken raw, no browser echo cancellation, noise suppression or gain control
bending the path. The server resamples the reference beside the
microphone and runs LocalVQE v1.3 on hops of 256 samples before the VAD.
The model estimates the echo delay itself by a soft cross-attention over
the last 64 frames, about one second, and removes noise and reverberation
in the same pass. It costs 16 to 32 ms of latency before the VAD, and each
connection carries 2.3 MB of layer history. A log line closes every run of
playback with the level of the microphone above the cleaned signal.

`native` asks the browser for `echoCancellation: "all"`, which only
Chrome based browsers honor for a page's own playback. `off` cancels no
echo, headphones do: the browser still suppresses noise and levels the
gain.

## Protocol

`WS /v1/realtime`, a subset of the OpenAI Realtime API. Audio is PCM16
at 24 kHz, base64 encoded, in both directions: it matches the codec
output, and the input is resampled to the 16 kHz the canceller, the VAD
and the recognizer work at, with the Hann-windowed sinc of torchaudio. `input_audio_buffer.append` carries
an extra `reference` field, the audio played during the same samples,
when the echo cancellation runs on the server and something played.

Client to server: `session.update`, `input_audio_buffer.append`,
`input_audio_buffer.commit`, `response.cancel`, `conversation.history`.

Server to client: `session.created`, `session.updated`,
`input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`,
`conversation.item.input_audio_transcription.completed`,
`response.created`, `response.output_text.delta`,
`response.output_audio.delta`, `response.output_audio_transcript.delta`,
`response.done`, `response.cancelled`, `error`.

The transcript event carries the item of its turn, `turn_<id>`: a later
transcript of the same turn replaces the user message instead of adding
one.

`response.output_text.delta` is what the model writes, as it writes it;
`response.output_audio_transcript.delta` is what the voice speaks, one
unit at a time, with `text_end`: how far into the written text the spoken
part reaches, in UTF-16 code units, the index a browser slices its copy
with. What lies past it was written and not spoken, which is how the page
tells an answer cut off by a barge-in from one said to the end. Unknown
fields are ignored, an unknown type gets an `error` event.

`session.update` describes the whole session: a field it leaves out takes
the default published on `/props`, so clearing a field brings the default
back. The conversation has its own event, and the echo method only
changes when one is named. A number has to be finite and in its range, a
turn duration up to 60 s: the server refuses any other with an `error`
naming the field, which keeps its default.

HTTP routes: `/` the page, `/s2s.js` the component, `/props` the session
defaults and the loaded models, `/health`, `/logs` the server log as SSE,
`/log` where the page posts its own lines, and `POST /v1/models` which
lists the models of the endpoint a client names.

## Security

`--origin` is an allowlist of browser origins, checked on the WebSocket
handshake, which CORS does not cover, and on the HTTP routes. A request
without `Origin` passes, a foreign one gets 403. `--llm-host` is an
allowlist of endpoint hosts: the server fetches the endpoint a client
names, so without it the server reaches anything it can route to. The
log carries no conversation text, only character counts and timings, and
no endpoint: it says whether one is set, never which.

`--llm-url`, `--llm-model` and `--llm-key-file` give the server its own
endpoint, the key read from the first line of a file so it shows neither
in the process list nor in a shell history. The endpoint is then hidden
from every session and fixed for it: the model list route answers 403,
and a patch that names an endpoint, a model or a key is refused with an
error. Without these options the server has no endpoint at all, and each
session names its own. `/props` never publishes the endpoint: its
`llm_fixed` only tells the page whether it may name one.

## Module map

| File | Role |
| --- | --- |
| `src/backend.h` | backend selection, CPU thread count, log dedup |
| `src/gguf-weights.h`, `src/weight-ctx.h` | GGUF reader and weight upload |
| `src/graph-arena.h` | graph allocation |
| `src/conv-f32.h` | f32 convolutions, no f16 im2col staging |
| `src/audio-resample.h` | polyphase resampler, any rate to 16 kHz for the recognizer |
| `src/audio-mel.h`, `src/parakeet-mel.h` | log mel frontends |
| `src/localvqe.h` | echo canceller lib, C ABI |
| `src/silero.h` | VAD lib, C ABI |
| `src/smart-turn.h` | end of turn lib, C ABI |
| `src/parakeet.h` | ASR lib, C ABI |
| `src/pipeline-asr.h`, `src/fastconformer-forward.h`, `src/tdt-decoder.h`, `src/sp-detok.h` | Parakeet internals |
| `src/s2s-session.h` | turn state machine |
| `src/llm-client.h` | chat completions SSE client |
| `src/sentence-split.h`, `src/emoji.h` | streaming text to synthesis units |
| `src/tts-bridge.h` | qwentts.cpp calls, per request voice, sampling and guards |
| `src/realtime-proto.h` | Realtime event encode and decode |
| `src/s2s-error.h`, `src/timer.h`, `src/utf8.h`, `src/wav.h` | log, timing, UTF-8 argv, wav io |
| `tools/s2s-server.cpp` | the product binary |
| `tools/parakeet-transcribe.cpp` | the recognizer alone, on a wav file |
| `tools/quantize.cpp` | GGUF quantizer |
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

`tests/` holds two kinds of harness, each a C++ target or the binary
itself driven by a Python script that prints `[Parity]` or `[Check]`
lines. The parity harnesses hold every model to its reference:

| Test | Checks |
| --- | --- |
| `test-localvqe` | streamed output against the upstream PyTorch model, and the echo removed on a synthetic call |
| `test-silero` | probabilities and decisions against onnxruntime |
| `test-smart-turn` | completion probabilities and decisions against onnxruntime |
| `test-parakeet` | mel, encoder, projection and transcript against transformers `ParakeetForTDT` |

`test-server` holds the behavior of the whole server. A scripted client
plays passages of the example file, silences, commits and a microphone
cut in real time, as a browser on a loudspeaker, and runs a mock endpoint
in process whose latency the scenario sets. The checks judge its timeline
in windows relative to its own events:

| Scenario | Checks |
| --- | --- |
| loopback | the answer is the transcript, first audio once the grace ran out |
| grace | a monologue resumed during the grace, recognized whole, silent until it ends |
| before answer | speech after the grace but before a slow endpoint answered stays in the turn, one message reaches the model |
| barge-in | a talk over an answer in flight: new turn, playback, response and endpoint request stopped |
| push to talk | a commit mid sentence and a microphone cut still get an answer, with no grace |
| room | the echo canceller keeps the assistant out of what is heard through a room |
| no endpoint, broken endpoint, midstream | the error reaches the client, the response still closes |
| owned endpoint | nothing of the endpoint published or logged, another endpoint refused |

Every scenario also checks that each response closes exactly once and that
the transcripts of a turn come in a row.

The Parakeet sweep covers CUDA, Vulkan and CPU against F32, Q8_0 and
Q4_K_M. The synthesis itself has its parity harnesses in qwentts.cpp.
