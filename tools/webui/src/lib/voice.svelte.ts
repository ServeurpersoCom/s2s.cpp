import type { S2S, S2SOptions, S2SState } from './s2s.js';
import { postLog } from './api.js';
import { num } from './fields.js';
import { app, settings, toast } from './state.svelte.js';

// The one place the settings panel and the component meet. The panel edits a
// Settings object, this turns it into the options the component takes, and a
// host embedding the component elsewhere fills the same fields by hand.
export function toOptions(): S2SOptions {
	// a server that owns its endpoint takes none of it from the page
	const fixed = app.props?.defaults.llm_fixed ?? false;
	return {
		url: settings.serverUrl,
		mode: settings.mode || undefined,
		instructions: settings.systemPrompt,
		llmUrl: fixed ? undefined : settings.llmUrl,
		llmModel: fixed ? undefined : settings.llmModel,
		llmKey: fixed ? undefined : settings.llmKey,
		sampling: {
			temperature: num(settings.temperature),
			topP: num(settings.topP),
			topK: num(settings.topK),
			minP: num(settings.minP),
			maxTokens: num(settings.maxTokens),
			presencePenalty: num(settings.presencePenalty),
			frequencyPenalty: num(settings.frequencyPenalty),
			seed: num(settings.seed),
			reasoningEffort: settings.reasoningEffort
		},
		llmTimeoutSec: num(settings.llmTimeoutSec),
		tts: {
			speaker: settings.voice,
			language: settings.language,
			minChars: num(settings.ttsMinChars),
			charsPerSecond: num(settings.ttsCharsPerSecond),
			marginSeconds: num(settings.ttsMarginSeconds),
			sampling: {
				temperature: num(settings.ttsTemperature),
				topK: num(settings.ttsTopK),
				topP: num(settings.ttsTopP),
				repetitionPenalty: num(settings.ttsRepetitionPenalty),
				subtalkerTemperature: num(settings.ttsSubtalkerTemperature),
				subtalkerTopK: num(settings.ttsSubtalkerTopK),
				subtalkerTopP: num(settings.ttsSubtalkerTopP),
				maxNewTokens: num(settings.ttsMaxNewTokens),
				seed: num(settings.ttsSeed)
			}
		},
		vad: {
			threshold: num(settings.vadThreshold),
			minSpeechMs: num(settings.minSpeechMs),
			minSpeechContinuationMs: num(settings.minSpeechContinuationMs),
			minSilenceMs: num(settings.minSilenceMs),
			speechPadMs: num(settings.speechPadMs)
		},
		turn: {
			threshold: num(settings.turnThreshold),
			maxWaitMs: num(settings.turnMaxWaitMs),
			graceMs: num(settings.turnGraceMs)
		},
		echo: settings.echo || undefined
	};
}

// One line of the conversation. draft is what the model wrote, spokenEnd how
// far into it the voice went: it trails the draft by one synthesis unit, and
// stops there on a barge-in.
export interface ChatTurn {
	role: 'user' | 'assistant';
	draft: string;
	spokenEnd: number;
	done: boolean;
}

// Live view of the component, for whatever the page decides to draw. Nothing
// here touches the DOM: the component stays invisible until a host renders
// something from this state.
export const voice = $state({
	state: 'idle' as S2SState,
	error: '',
	chat: [] as ChatTurn[]
});

function lastAssistant(): ChatTurn | undefined {
	const turn = voice.chat[voice.chat.length - 1];
	return turn && turn.role === 'assistant' && !turn.done ? turn : undefined;
}

function openAssistant(): ChatTurn {
	const open = lastAssistant();
	if (open) {
		return open;
	}
	const turn: ChatTurn = { role: 'assistant', draft: '', spokenEnd: 0, done: false };
	voice.chat.push(turn);
	return turn;
}

function closeAssistant() {
	const open = lastAssistant();
	if (open) {
		open.done = true;
	}
}

let client: S2S | null = null;

export function getClient(): S2S | null {
	return client;
}

// The page loads the very module an integrator imports, from the same
// relative URL, instead of embedding a copy of the sources in its own bundle.
// The demo then proves that the published file works, not merely that the
// code compiles. @vite-ignore keeps the bundler out of this import.
type S2SModule = { S2S: new (options: S2SOptions) => S2S };

let loading: Promise<S2SModule> | null = null;

function loadModule(): Promise<S2SModule> {
	if (!loading) {
		loading = import(
			/* @vite-ignore */ new URL('s2s.js', location.href).href
		) as Promise<S2SModule>;
	}
	return loading;
}

// Creates the component and wires its callbacks into the live view. The
// microphone and the socket only open on start(), which a host must call from
// a user gesture.
export async function createVoice(): Promise<S2S> {
	const { S2S } = await loadModule();
	const s2s = new S2S(toOptions());

	s2s.on('state', (state) => {
		voice.state = state;
		if (state === 'listening') {
			voice.error = '';
			closeAssistant();
		}
	});
	s2s.on('user_text', (text, revised) => {
		closeAssistant();
		// a revision replaces the turn it revises, and the answer drafted for
		// it that nobody heard
		if (revised) {
			let at = voice.chat.length - 1;
			while (at >= 0 && voice.chat[at].role !== 'user') {
				at--;
			}
			voice.chat.splice(Math.max(at, 0));
		}
		voice.chat.push({
			role: 'user',
			draft: text,
			spokenEnd: text.length,
			done: true
		});
	});
	s2s.on('assistant_delta', (text) => {
		openAssistant().draft += text;
	});
	s2s.on('assistant_text', (text, textEnd) => {
		const turn = openAssistant();
		// loopback has no model writing ahead: what is spoken is the whole turn
		if (!turn.draft) {
			turn.draft = text;
		}
		turn.spokenEnd = textEnd;
	});
	// everything the component reports goes to the server log, so the card on
	// the right tells the whole story, and a failure also pops the toast
	s2s.on('log', (line) => {
		postLog(settings.serverUrl, line);
	});
	s2s.on('error', (message) => {
		voice.error = message;
		postLog(settings.serverUrl, `Error: ${message}`);
		toast(message);
	});

	client = s2s;
	// the slider may have been moved before the module finished loading
	s2s.setVolume(settings.volume);
	return s2s;
}

export function destroyVoice() {
	client?.stop();
	client = null;
	voice.state = 'idle';
}

// Forgets the conversation, in the component and on the page alike.
export function clearContext() {
	client?.clearHistory();
	voice.chat = [];
}

// Pushes a settings change to a running session. Everything but the server
// URL applies live; a new URL needs a reconnection.
// The options are built before the call: `client?.update(toOptions())` would
// skip the argument entirely while the component is still loading, and an
// effect that never reads the store never tracks it, so a later change would
// go unnoticed. That optional call is what made mode switches need a reload.
export function applySettings() {
	const options = toOptions();
	client?.update(options);
}

// Same reason: the read happens before the optional call, so the effect
// tracks the slider even when the component is not loaded yet.
export function applyVolume() {
	const volume = settings.volume;
	client?.setVolume(volume);
}
