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
// Two AudioWorklets carry the audio. The capture worklet ships 20 ms frames
// of the microphone, the playback worklet holds a ring buffer and counts the
// frames it has really played, which is what a truncation needs. Both are
// inlined as source strings and loaded through a blob URL, so the module
// stays a single file.
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
// browser to cancel everything the system plays, this page included, off
// hands over the raw microphone.
export const ECHO_MODES = ['native', 'off'] as const;
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
	// native unless set. Read at start().
	echo?: S2SEcho;
}

export interface S2SEvents {
	state: (state: S2SState) => void;
	user_text: (text: string) => void;
	// what the model writes, as it writes it
	assistant_delta: (text: string) => void;
	// what the voice really speaks, one synthesis unit at a time
	assistant_text: (text: string) => void;
	// the conversation, every time it changes: persist it, or ignore it
	history: (messages: S2SMessage[]) => void;
	// every step of a start, a stop or a failure, so a host can show what
	// happened instead of guessing
	log: (line: string) => void;
	error: (message: string) => void;
}

const CAPTURE_WORKLET = `
class CaptureProcessor extends AudioWorkletProcessor {
	constructor() {
		super();
		this.buffer = new Float32Array(${FRAME_SAMPLES});
		this.filled = 0;
		this.muted = false;
		this.port.onmessage = (e) => { this.muted = !!e.data.muted; };
	}
	process(inputs) {
		const input = inputs[0] && inputs[0][0];
		if (!input) return true;
		for (let i = 0; i < input.length; i++) {
			this.buffer[this.filled++] = this.muted ? 0 : input[i];
			if (this.filled === this.buffer.length) {
				this.port.postMessage(this.buffer.slice());
				this.filled = 0;
			}
		}
		return true;
	}
}
registerProcessor('s2s-capture', CaptureProcessor);
`;

const PLAYBACK_WORKLET = `
class PlaybackProcessor extends AudioWorkletProcessor {
	constructor() {
		super();
		this.queue = [];
		this.offset = 0;
		this.played = 0;
		this.generation = 0;
		this.port.onmessage = (e) => {
			if (e.data.flush) {
				this.queue = [];
				this.offset = 0;
			} else if (e.data.reset !== undefined) {
				this.played = 0;
				this.generation = e.data.reset;
			} else {
				this.queue.push(e.data);
			}
		};
	}
	process(_inputs, outputs) {
		const output = outputs[0][0];
		let written = 0;
		while (written < output.length && this.queue.length > 0) {
			const chunk = this.queue[0];
			const take = Math.min(chunk.length - this.offset, output.length - written);
			output.set(chunk.subarray(this.offset, this.offset + take), written);
			this.offset += take;
			written += take;
			this.played += take;
			if (this.offset === chunk.length) {
				this.queue.shift();
				this.offset = 0;
			}
		}
		output.fill(0, written);
		if (written > 0) this.port.postMessage({ played: this.played, generation: this.generation });
		return true;
	}
}
registerProcessor('s2s-playback', PlaybackProcessor);
`;

// v1/realtime next to the page, with the scheme the page was loaded with, so
// an https page gets wss and keeps its secure context.
export function realtimeUrl(url?: string): string {
	const base = new URL('v1/realtime', url ? url.replace(/\/?$/, '/') : location.href);
	base.protocol =
		base.protocol === 'https:' ? 'wss:' : base.protocol === 'http:' ? 'ws:' : base.protocol;
	return base.toString();
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
	private capture: AudioWorkletNode | null = null;
	private playback: AudioWorkletNode | null = null;
	private gain: GainNode | null = null;
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
		await this.context.audioWorklet.addModule(workletUrl(CAPTURE_WORKLET));
		await this.context.audioWorklet.addModule(workletUrl(PLAYBACK_WORKLET));

		this.log(`Audio context at ${this.context.sampleRate} Hz`);

		// Echo cancellation matters more than anything else here: on laptop
		// speakers the assistant would otherwise hear itself and barge in on
		// its own voice.
		const echo: S2SEcho = this.options.echo ?? 'native';
		this.stream = await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: echo === 'native' ? ECHO_CANCELLATION_ALL : false,
				noiseSuppression: true,
				autoGainControl: true,
				channelCount: 1
			}
		});

		this.log('Microphone granted');

		this.playback = new AudioWorkletNode(this.context, 's2s-playback', {
			numberOfInputs: 0,
			outputChannelCount: [1]
		});
		this.playback.port.onmessage = (e) => {
			if (e.data.generation !== this.generation) {
				return;
			}
			this.playedSamples = e.data.played;
			if (this.answerDone && this.playedSamples >= this.queuedSamples) {
				this.closeAnswer();
				this.setState('listening');
			}
		};
		this.gain = this.context.createGain();
		this.gain.gain.value = this.volume;
		this.playback.connect(this.gain).connect(this.context.destination);

		this.capture = new AudioWorkletNode(this.context, 's2s-capture');
		this.context.createMediaStreamSource(this.stream).connect(this.capture);
		this.capture.port.onmessage = (e) => this.sendAudio(e.data as Float32Array);

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
		this.capture = null;
		this.playback = null;
		this.gain = null;
		this.setState('idle');
	}

	setVolume(volume: number) {
		this.volume = volume;
		if (this.gain) {
			this.gain.gain.value = volume;
		}
	}

	mute(muted: boolean) {
		this.capture?.port.postMessage({ muted });
	}

	// Client side barge-in: the user took the floor, so playback stops now and
	// the server is told how much was actually heard.
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
		this.pushHistory();
	}

	clearHistory() {
		this.log('History cleared');
		this.setHistory([]);
	}

	// A field left out keeps its value: an undefined arriving from a cleared
	// input must not erase what the session already runs with.
	update(options: Partial<S2SOptions>) {
		for (const [key, value] of Object.entries(options)) {
			if (value !== undefined) {
				(this.options as Record<string, unknown>)[key] = value;
			}
		}
		this.sendSessionUpdate();
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

	private sendAudio(pcm: Float32Array) {
		this.send({ type: 'input_audio_buffer.append', audio: toBase64(pcm) });
	}

	private sendSessionUpdate() {
		this.send({
			type: 'session.update',
			session: {
				mode: this.options.mode,
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
					max_wait_ms: this.options.turn?.maxWaitMs
				}
			}
		});
	}

	private flushPlayback() {
		this.playback?.port.postMessage({ flush: true });
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
					const transcript = String(message.transcript ?? '');
					this.history.push({ role: 'user', content: transcript });
					this.handlers.user_text?.(transcript);
				}
				break;

			case 'response.created':
				this.closeAnswer();
				this.generation++;
				this.queuedSamples = 0;
				this.playedSamples = 0;
				this.playback?.port.postMessage({ reset: this.generation });
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
					this.handlers.assistant_text?.(delta);
				}
				break;

			case 'response.output_audio.delta':
				{
					const pcm = fromBase64(String(message.delta ?? ''));
					this.queuedSamples += pcm.length;
					this.playback?.port.postMessage(pcm);
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
