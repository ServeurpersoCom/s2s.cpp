// s2s.ts: the browser half of the loop, as a dependency free module
//
// Drop it on any page and point it at an s2s-server:
//
//     import { S2S } from './s2s.js';
//
//     const s2s = new S2S({ url: 'wss://host/v1/realtime' });
//     s2s.on('state', (state) => console.log(state));
//     button.onclick = () => s2s.start();
//
// One duplex AudioWorklet carries the audio both ways. It plays the answer
// from a queue and counts the samples really played, which is what closing
// an answer on a barge-in needs, and it ships 20 ms frames of the microphone
// together with the samples it played during the same render quanta: the
// exact echo reference a server side canceller needs. It is inlined as a
// source string and loaded through a blob URL, so the module stays a single
// file.
//
// The AudioContext is created at the rate the protocol carries, 24 kHz, and
// the browser resamples the microphone into it. That removes the only
// resampler this client would otherwise need.
//
// The component draws nothing: it owns audio, a socket and a state machine,
// and reports through callbacks. A host renders whatever it wants around it,
// or nothing at all.
//
// start() must run inside a user gesture, and the page needs a secure
// context: https, or localhost during development.

// The rate the Realtime protocol carries, in both directions. Fixed by the
// protocol, not a preference: the server reads and writes PCM16 at this rate.
const SAMPLE_RATE = 24000;
const FRAME_SAMPLES = 480; // 20 ms

// Who removes the assistant voice from the microphone. native asks the
// browser to cancel everything the system plays, this page included, server
// hands the raw microphone and the played reference to s2s-server, off hands
// over the raw microphone and nothing else.
export const ECHO_MODES = ['server', 'native', 'off'] as const;
export const ECHO_DEFAULT: S2SEcho = 'server';
export type S2SEcho = (typeof ECHO_MODES)[number];

// Plain true only covers WebRTC remote tracks, which this component never
// plays through. lib.dom still types the constraint as a boolean, hence the
// cast.
const ECHO_CANCELLATION_ALL = 'all' as unknown as ConstrainBoolean;

export type S2SState = 'idle' | 'listening' | 'thinking' | 'speaking';

// One line of the conversation, in the shape the endpoint takes. The client
// owns this list: the server holds a copy for the turn it is answering and
// keeps nothing between two turns.
export interface S2SMessage {
	role: 'user' | 'assistant';
	content: string;
}

// Everything the component takes. The settings panel of the bundled page is
// nothing more than an editor for this object: a host that embeds the
// component writes the same fields in code and needs no UI at all.
// Silero scores every 32 ms window, Smart Turn only scores a speech to
// silence boundary, so the two are configured apart.
export interface S2SVad {
	threshold?: number;
	minSpeechMs?: number;
	minSpeechContinuationMs?: number;
	minSilenceMs?: number;
	speechPadMs?: number;
}

export interface S2STurn {
	threshold?: number;
	maxWaitMs?: number;
	// how long the answer to a turn judged complete stays silent, so the
	// speaker can go on without the assistant cutting in
	graceMs?: number;
}

export interface S2SSampling {
	temperature?: number;
	topP?: number;
	topK?: number;
	minP?: number;
	maxTokens?: number;
	presencePenalty?: number;
	frequencyPenalty?: number;
	seed?: number;
}

export interface S2STtsSampling {
	temperature?: number;
	topK?: number;
	topP?: number;
	repetitionPenalty?: number;
	subtalkerTemperature?: number;
	subtalkerTopK?: number;
	subtalkerTopP?: number;
	maxNewTokens?: number;
	seed?: number;
}

export interface S2STts {
	speaker?: string;
	language?: string;
	sampling?: S2STtsSampling;
	// guards: a unit shorter than minChars is not spoken, and the frame budget
	// of a synthesis is derived from the text length
	minChars?: number;
	charsPerSecond?: number;
	marginSeconds?: number;
}

export interface S2SOptions {
	// Empty, or absent, connects to v1/realtime relative to the page, which
	// works behind any reverse proxy prefix. An absolute ws:// or wss:// URL
	// points the component at a server on another host.
	url?: string;
	mode?: 'conversation' | 'loopback';
	instructions?: string;
	llmUrl?: string;
	llmModel?: string;
	llmKey?: string;
	sampling?: S2SSampling;
	tts?: S2STts;
	llmTimeoutSec?: number;
	vad?: S2SVad;
	turn?: S2STurn;
	// ECHO_DEFAULT unless set. Read at start().
	echo?: S2SEcho;
}

export interface S2SEvents {
	state: (state: S2SState) => void;
	// revised: a later transcript of the same turn, which replaces the
	// previous one along with any answer to it that was never heard
	user_text: (text: string, revised: boolean) => void;
	// what the model writes, as it writes it
	assistant_delta: (text: string) => void;
	// what the voice really speaks, one synthesis unit at a time, and how far
	// into the written text it reaches: what lies past textEnd in the
	// assistant_delta text was written and not spoken
	assistant_text: (text: string, textEnd: number) => void;
	// the conversation, every time it changes: persist it, or ignore it
	history: (messages: S2SMessage[]) => void;
	// every step of a start, a stop or a failure, so a host can show what
	// happened instead of guessing
	log: (line: string) => void;
	error: (message: string) => void;
}

const DUPLEX_WORKLET = `
class DuplexProcessor extends AudioWorkletProcessor {
	constructor() {
		super();
		this.mic = new Float32Array(${FRAME_SAMPLES});
		this.ref = new Float32Array(${FRAME_SAMPLES});
		this.filled = 0;
		this.muted = false;
		this.volume = 1;
		this.queue = [];
		this.offset = 0;
		this.played = 0;
		this.generation = 0;
		this.port.onmessage = (e) => {
			const data = e.data;
			if (data instanceof Float32Array) {
				this.queue.push(data);
			} else if (data.flush) {
				this.queue = [];
				this.offset = 0;
			} else if (data.reset !== undefined) {
				this.played = 0;
				this.generation = data.reset;
			} else if (data.muted !== undefined) {
				this.muted = data.muted;
			} else if (data.volume !== undefined) {
				this.volume = data.volume;
			}
		};
	}
	process(inputs, outputs) {
		const output = outputs[0][0];
		let written = 0;
		while (written < output.length && this.queue.length > 0) {
			const chunk = this.queue[0];
			const take = Math.min(chunk.length - this.offset, output.length - written);
			for (let i = 0; i < take; i++) {
				output[written + i] = chunk[this.offset + i] * this.volume;
			}
			this.offset += take;
			written += take;
			if (this.offset === chunk.length) {
				this.queue.shift();
				this.offset = 0;
			}
		}
		output.fill(0, written);
		if (written > 0) {
			this.played += written;
			this.port.postMessage({ played: this.played, generation: this.generation });
		}

		const input = inputs[0] && inputs[0][0];
		for (let i = 0; i < output.length; i++) {
			this.mic[this.filled] = input && !this.muted ? input[i] : 0;
			this.ref[this.filled] = output[i];
			if (++this.filled === this.mic.length) {
				this.port.postMessage({ mic: this.mic.slice(), ref: this.ref.slice() });
				this.filled = 0;
			}
		}
		return true;
	}
}
registerProcessor('s2s-duplex', DuplexProcessor);
`;

// v1/realtime next to the page, with the scheme the page was loaded with, so
// an https page gets wss and keeps its secure context.
export function realtimeUrl(url?: string): string {
	const base = new URL('v1/realtime', url ? url.replace(/\/?$/, '/') : location.href);
	base.protocol =
		base.protocol === 'https:' ? 'wss:' : base.protocol === 'http:' ? 'ws:' : base.protocol;
	return base.toString();
}

// Echo cancellation matters more than anything else here: on laptop speakers
// the assistant would otherwise hear itself and barge in on its own voice.
// The server canceller wants the raw microphone, since a browser processing in
// front of it would bend the echo path it models.
function micConstraints(echo: S2SEcho): MediaTrackConstraints {
	const raw = echo === 'server';
	return {
		echoCancellation: echo === 'native' ? ECHO_CANCELLATION_ALL : false,
		noiseSuppression: !raw,
		autoGainControl: !raw,
		channelCount: 1
	};
}

function workletUrl(source: string): string {
	return URL.createObjectURL(new Blob([source], { type: 'application/javascript' }));
}

function toBase64(pcm: Float32Array): string {
	const bytes = new Uint8Array(pcm.length * 2);
	const view = new DataView(bytes.buffer);
	for (let i = 0; i < pcm.length; i++) {
		const clamped = Math.max(-1, Math.min(1, pcm[i]));
		view.setInt16(i * 2, clamped * 32767, true);
	}
	let binary = '';
	for (let i = 0; i < bytes.length; i += 0x8000) {
		binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}
	return btoa(binary);
}

function fromBase64(data: string): Float32Array {
	const binary = atob(data);
	const bytes = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) {
		bytes[i] = binary.charCodeAt(i);
	}
	const view = new DataView(bytes.buffer);
	const pcm = new Float32Array(bytes.length / 2);
	for (let i = 0; i < pcm.length; i++) {
		pcm[i] = view.getInt16(i * 2, true) / 32768;
	}
	return pcm;
}

export class S2S {
	private options: S2SOptions;
	private handlers: Partial<S2SEvents> = {};

	private ws: WebSocket | null = null;
	private context: AudioContext | null = null;
	private stream: MediaStream | null = null;
	private duplex: AudioWorkletNode | null = null;
	private volume = 1;

	private state: S2SState = 'idle';

	// The answer in flight, measured in samples since its response.created:
	// what the server sent, what the speaker played, and each synthesis unit
	// with the sample where its audio starts. The generation tags the played
	// reports, so a report of a previous answer is ignored.
	private generation = 0;
	private queuedSamples = 0;
	private playedSamples = 0;
	private units: { text: string; start: number }[] = [];
	private answerDone = false;

	private history: S2SMessage[] = [];
	private userItem = ''; // the server item of the last user message

	constructor(options: S2SOptions) {
		this.options = options;
	}

	on<K extends keyof S2SEvents>(event: K, handler: S2SEvents[K]) {
		this.handlers[event] = handler;
	}

	getState(): S2SState {
		return this.state;
	}

	// Must be called from a user gesture: browsers refuse both the microphone
	// and an audio context without one.
	async start() {
		if (this.ws) {
			return;
		}

		try {
			await this.open();
		} catch (e) {
			const message = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
			this.log(`Start failed, ${message}`);
			this.handlers.error?.(message);
			this.stop();
			throw e;
		}
	}

	private log(line: string) {
		this.handlers.log?.(line);
	}

	private async open() {
		this.log('Start requested');

		this.context = new AudioContext({ sampleRate: SAMPLE_RATE });
		await this.context.audioWorklet.addModule(workletUrl(DUPLEX_WORKLET));

		this.log(`Audio context at ${this.context.sampleRate} Hz`);

		this.stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints(this.echo()) });
		this.log(`Microphone granted, ${this.micApplied()}`);

		this.duplex = new AudioWorkletNode(this.context, 's2s-duplex', {
			numberOfInputs: 1,
			numberOfOutputs: 1,
			outputChannelCount: [1]
		});
		this.duplex.port.postMessage({ volume: this.volume });
		this.duplex.port.onmessage = (e) => {
			if (e.data.mic) {
				this.sendAudio(e.data.mic as Float32Array, e.data.ref as Float32Array);
				return;
			}
			if (e.data.generation !== this.generation) {
				return;
			}
			this.playedSamples = e.data.played;
			if (this.answerDone && this.playedSamples >= this.queuedSamples) {
				this.closeAnswer();
				this.setState('listening');
			}
		};
		this.context.createMediaStreamSource(this.stream).connect(this.duplex);
		this.duplex.connect(this.context.destination);

		await this.connect();
		this.setState('listening');
	}

	stop() {
		this.log('Stop requested');
		this.ws?.close();
		this.ws = null;
		this.stream?.getTracks().forEach((track) => track.stop());
		this.stream = null;
		this.context?.close();
		this.context = null;
		this.duplex = null;
		this.setState('idle');
	}

	setVolume(volume: number) {
		this.volume = volume;
		this.duplex?.port.postMessage({ volume });
	}

	mute(muted: boolean) {
		this.duplex?.port.postMessage({ muted });
	}

	// Client side barge-in: the user took the floor, so playback stops now and
	// the server stops the response. The answer keeps what was actually heard
	// once the server confirms with response.cancelled.
	cancel() {
		this.send({ type: 'response.cancel' });
		this.flushPlayback();
	}

	getHistory(): S2SMessage[] {
		return this.history;
	}

	// Seeds the conversation, for a host that persisted it or that brings its
	// own context.
	setHistory(messages: S2SMessage[]) {
		this.history = messages.slice();
		this.userItem = '';
		this.pushHistory();
	}

	clearHistory() {
		this.log('History cleared');
		this.setHistory([]);
	}

	// A field left out keeps its value: an undefined arriving from a cleared
	// input must not erase what the session already runs with.
	update(options: Partial<S2SOptions>) {
		const echo = this.echo();
		for (const [key, value] of Object.entries(options)) {
			if (value !== undefined) {
				(this.options as Record<string, unknown>)[key] = value;
			}
		}
		this.sendSessionUpdate();
		if (this.stream && this.echo() !== echo) {
			this.applyEcho();
		}
	}

	// The server switches its canceller on the session.update; the microphone
	// follows here, so the browser processing always matches the method.
	private applyEcho() {
		const track = this.stream?.getAudioTracks()[0];
		track
			?.applyConstraints(micConstraints(this.echo()))
			.then(() => this.log(`Microphone switched, ${this.micApplied()}`))
			.catch((e) =>
				this.log(`Microphone switch refused, ${e instanceof Error ? e.message : String(e)}`)
			);
	}

	// What the browser really runs on the microphone, which is not always
	// what was asked.
	private micApplied(): string {
		const settings = this.stream?.getAudioTracks()[0]?.getSettings();
		return `echo ${this.echo()}, browser echo cancellation ${settings?.echoCancellation}, noise suppression ${settings?.noiseSuppression}, gain control ${settings?.autoGainControl}`;
	}

	private setState(state: S2SState) {
		if (this.state !== state) {
			this.state = state;
			this.handlers.state?.(state);
		}
	}

	private send(message: unknown) {
		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			this.ws.send(JSON.stringify(message));
		}
	}

	// The reference only travels to a server canceller, and only when the
	// frame played something: silence is what its absence means.
	private sendAudio(mic: Float32Array, ref: Float32Array) {
		const reference = this.echo() === 'server' && ref.some((sample) => sample !== 0);
		this.send({
			type: 'input_audio_buffer.append',
			audio: toBase64(mic),
			...(reference ? { reference: toBase64(ref) } : {})
		});
	}

	private echo(): S2SEcho {
		return this.options.echo ?? ECHO_DEFAULT;
	}

	private sendSessionUpdate() {
		this.send({
			type: 'session.update',
			session: {
				mode: this.options.mode,
				echo: this.echo(),
				instructions: this.options.instructions,
				llm_url: this.options.llmUrl,
				llm_model: this.options.llmModel,
				llm_key: this.options.llmKey,
				llm_timeout_sec: this.options.llmTimeoutSec,
				sampling: {
					temperature: this.options.sampling?.temperature,
					top_p: this.options.sampling?.topP,
					top_k: this.options.sampling?.topK,
					min_p: this.options.sampling?.minP,
					max_tokens: this.options.sampling?.maxTokens,
					presence_penalty: this.options.sampling?.presencePenalty,
					frequency_penalty: this.options.sampling?.frequencyPenalty,
					seed: this.options.sampling?.seed
				},
				tts: {
					speaker: this.options.tts?.speaker,
					language: this.options.tts?.language,
					min_chars: this.options.tts?.minChars,
					chars_per_second: this.options.tts?.charsPerSecond,
					margin_seconds: this.options.tts?.marginSeconds,
					sampling: {
						temperature: this.options.tts?.sampling?.temperature,
						top_k: this.options.tts?.sampling?.topK,
						top_p: this.options.tts?.sampling?.topP,
						repetition_penalty: this.options.tts?.sampling?.repetitionPenalty,
						subtalker_temperature: this.options.tts?.sampling?.subtalkerTemperature,
						subtalker_top_k: this.options.tts?.sampling?.subtalkerTopK,
						subtalker_top_p: this.options.tts?.sampling?.subtalkerTopP,
						max_new_tokens: this.options.tts?.sampling?.maxNewTokens,
						seed: this.options.tts?.sampling?.seed
					}
				},
				vad: {
					threshold: this.options.vad?.threshold,
					min_speech_ms: this.options.vad?.minSpeechMs,
					min_speech_continuation_ms: this.options.vad?.minSpeechContinuationMs,
					min_silence_ms: this.options.vad?.minSilenceMs,
					speech_pad_ms: this.options.vad?.speechPadMs
				},
				turn: {
					threshold: this.options.turn?.threshold,
					max_wait_ms: this.options.turn?.maxWaitMs,
					reopen_grace_ms: this.options.turn?.graceMs
				}
			}
		});
	}

	private flushPlayback() {
		this.duplex?.port.postMessage({ flush: true });
	}

	private pushHistory() {
		this.send({ type: 'conversation.history', messages: this.history });
		this.handlers.history?.(this.history);
	}

	// Closes the answer in flight with what was actually heard. A unit counts
	// once its audio started playing: a barge-in keeps the sentence it cut and
	// drops the ones still queued, so the model never believes it said more.
	private closeAnswer() {
		const content = this.units
			.filter((unit) => unit.start < this.playedSamples)
			.map((unit) => unit.text)
			.join(' ')
			.trim();
		this.units = [];
		this.answerDone = false;
		if (!content) {
			return;
		}
		this.history.push({ role: 'assistant', content });
		this.pushHistory();
	}

	private connect(): Promise<void> {
		return new Promise((resolve, reject) => {
			const url = realtimeUrl(this.options.url);
			const ws = new WebSocket(url);
			this.ws = ws;

			this.log(`Connecting to ${url}`);

			ws.onopen = () => {
				this.log('Connected');
				this.sendSessionUpdate();
				this.pushHistory();
				resolve();
			};
			ws.onerror = () => {
				this.log(`Connection to ${url} failed`);
				reject(new Error(`cannot reach ${url}`));
			};
			ws.onclose = (event) => {
				this.log(`Connection closed, code ${event.code}`);
				this.ws = null;
				this.setState('idle');
			};
			ws.onmessage = (event) => this.onServerEvent(event.data as string);
		});
	}

	private onServerEvent(raw: string) {
		let message: { type?: string; [key: string]: unknown };
		try {
			message = JSON.parse(raw);
		} catch {
			return;
		}

		switch (message.type) {
			case 'input_audio_buffer.speech_started':
				// The user speaks: whatever is still playing is over, and the
				// answer keeps only what was heard.
				if (this.state === 'speaking' || this.state === 'thinking') {
					this.flushPlayback();
					this.closeAnswer();
				}
				this.setState('listening');
				break;

			case 'conversation.item.input_audio_transcription.completed':
				{
					// A later transcript of the same turn replaces its user
					// message instead of adding one.
					const transcript = String(message.transcript ?? '');
					const item = String(message.item_id ?? '');
					const last = this.history[this.history.length - 1];
					const revised = item !== '' && item === this.userItem && last?.role === 'user';
					if (revised) {
						last.content = transcript;
					} else {
						this.history.push({ role: 'user', content: transcript });
					}
					this.userItem = item;
					this.handlers.user_text?.(transcript, revised);
				}
				break;

			case 'response.created':
				this.closeAnswer();
				this.generation++;
				this.queuedSamples = 0;
				this.playedSamples = 0;
				this.duplex?.port.postMessage({ reset: this.generation });
				this.setState('thinking');
				break;

			case 'response.output_text.delta':
				{
					this.handlers.assistant_delta?.(String(message.delta ?? ''));
				}
				break;

			case 'response.output_audio_transcript.delta':
				{
					const delta = String(message.delta ?? '');
					this.units.push({ text: delta, start: this.queuedSamples });
					this.handlers.assistant_text?.(delta, Number(message.text_end ?? 0));
				}
				break;

			case 'response.output_audio.delta':
				{
					const pcm = fromBase64(String(message.delta ?? ''));
					this.queuedSamples += pcm.length;
					this.duplex?.port.postMessage(pcm);
					this.setState('speaking');
				}
				break;

			case 'response.cancelled':
				this.flushPlayback();
				this.closeAnswer();
				this.setState('listening');
				break;

			case 'response.done':
				// the server is done, the speaker may not be: the answer closes
				// when the last queued sample is played
				this.answerDone = true;
				if (this.playedSamples >= this.queuedSamples) {
					this.closeAnswer();
					this.setState('listening');
				}
				break;

			case 'error':
				{
					const error = message.error as { message?: string } | undefined;
					this.handlers.error?.(error?.message ?? 'unknown error');
				}
				break;
		}
	}
}
