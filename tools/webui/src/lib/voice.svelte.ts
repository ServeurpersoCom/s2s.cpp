import type { S2S, S2SMcpServer, S2SMessage, S2SOptions, S2SState } from './s2s.js';
import { num } from './fields.js';
import { app, settings, toast } from './state.svelte.js';

// A value the server or the browser does not serve is left out and its
// default applies: a voice or a language kept from another server, a
// microphone no longer plugged in. Before the list is known there is nothing
// to judge by, and the value goes as it is.
function served(value: string, list: string[] | undefined): string | undefined {
	return value && (!list || list.includes(value)) ? value : undefined;
}

export function sessionVoice(): string | undefined {
	return served(settings.voice, app.props?.defaults.tts_voices);
}

export function sessionEffect(): string | undefined {
	return served(settings.ttsEffect, app.props?.defaults.tts_effects);
}

// auto is not in the list, and every server takes it
export function sessionLanguage(): string | undefined {
	const list = app.props?.defaults.tts_languages;
	return served(settings.language, list && [...list, 'auto']);
}

// The one place the settings panel and the component meet. The panel edits a
// Settings object, this turns it into the options the component takes, and a
// host embedding the component elsewhere fills the same fields by hand. The
// page talks to the server that served it: no url here, the snippet adds its
// own. Only what the session applies goes out: the tools in the agentic
// mode alone, the model settings in every mode but loopback, where no model
// is in the path. The choices of the other modes stay in the settings.
// The MCP servers of the settings, one per line: the URL, then the key it
// takes after a space, when it takes one.
export function mcpServers(text: string): S2SMcpServer[] {
	return text
		.split('\n')
		.map((line) => line.trim().split(/\s+/))
		.filter((parts) => parts[0])
		.map(([url, key]) => (key ? { url, key } : { url }));
}

export function toOptions(): S2SOptions {
	const d = app.props?.defaults;
	// a server that owns its endpoint takes none of it from the page, and
	// the same for its MCP servers
	const fixed = d?.llm_fixed ?? false;
	const mcpFixed = d?.mcp_fixed ?? false;
	const mode = settings.mode || d?.mode || '';
	const agentic = mode === 'agentic';
	const model = mode !== 'loopback';
	return {
		mode: settings.mode || undefined,
		instructions: model ? settings.systemPrompt : undefined,
		llmUrl: fixed ? undefined : settings.llmUrl,
		llmModel: fixed ? undefined : settings.llmModel,
		llmKey: fixed ? undefined : settings.llmKey,
		tools: agentic && settings.tools.length ? [...settings.tools] : undefined,
		mcp: agentic && !mcpFixed && settings.mcp.trim() ? mcpServers(settings.mcp) : undefined,
		maxRounds: agentic ? num(settings.maxRounds) : undefined,
		toolTimeoutSec: agentic ? num(settings.toolTimeoutSec) : undefined,
		sampling: model
			? {
					temperature: num(settings.temperature),
					topP: num(settings.topP),
					topK: num(settings.topK),
					minP: num(settings.minP),
					maxTokens: num(settings.maxTokens),
					presencePenalty: num(settings.presencePenalty),
					frequencyPenalty: num(settings.frequencyPenalty),
					seed: num(settings.seed),
					reasoningEffort: settings.reasoningEffort
				}
			: undefined,
		llmTimeoutSec: model ? num(settings.llmTimeoutSec) : undefined,
		tts: {
			voice: sessionVoice(),
			effect: sessionEffect(),
			language: sessionLanguage(),
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
			negThreshold: num(settings.vadNegThreshold),
			threshold: num(settings.vadThreshold),
			minSpeechMs: num(settings.minSpeechMs),
			bargeInMs: num(settings.bargeInMs),
			minSilenceMs: num(settings.minSilenceMs),
			minSpeechContinuationMs: num(settings.minSpeechContinuationMs),
			speechPadMs: num(settings.speechPadMs)
		},
		turn: {
			threshold: num(settings.turnThreshold),
			incompleteDelayMs: num(settings.turnIncompleteDelayMs),
			maxWaitMs: num(settings.turnMaxWaitMs),
			graceMs: num(settings.turnGraceMs)
		},
		echo: settings.echo || undefined,
		mic: served(
			settings.mic,
			app.mics?.map((mic) => mic.id)
		)
	};
}

// A call of the model, at the place in written where it was made: while it
// runs only its name, then ok, the ms it ran and the bytes of its result.
export interface ChatTool {
	at: number;
	name: string;
	done: boolean;
	ok: boolean;
	ms: number;
	bytes: number;
}

// One line of the display. A user line holds its transcript in written and
// its revision, from 0. An answer holds what the model wrote in written, its
// calls in tools, what the voice spoke in spoken, and in spokenEnd how far
// into written the voice went: it trails by one synthesis unit and stops
// there on a barge-in. Loopback has no model writing, so its answers hold
// spoken alone.
export interface ChatEntry {
	role: 'user' | 'assistant';
	written: string;
	tools: ChatTool[];
	spoken: string;
	spokenEnd: number;
	revision: number;
	open: boolean;
}

// Each role keeps its own memory across a reload: the context is what the
// model knows, only what was heard, and the display is what the page showed,
// text written past the voice included. Their own keys: a reset of the
// settings leaves them alone.
const HISTORY_KEY = 's2s.history';
const CHAT_KEY = 's2s.chat';

function load<T>(key: string): T[] {
	try {
		const raw = localStorage.getItem(key);
		if (raw) {
			return JSON.parse(raw) as T[];
		}
	} catch {
		// corrupt or unavailable
	}
	return [];
}

function save(key: string, value: unknown) {
	try {
		localStorage.setItem(key, JSON.stringify(value));
	} catch {
		// full or unavailable
	}
}

// the turn identity dies with its connection, so only the words are kept
function saveHistory(messages: S2SMessage[]) {
	save(
		HISTORY_KEY,
		messages.map(({ role, content }) => ({ role, content }))
	);
}

// Live view of the component, for whatever the page decides to draw. Nothing
// here touches the DOM: the component stays invisible until a host renders
// something from this state. The display filed by the last visit shows
// before any start.
export const voice = $state({
	state: 'idle' as S2SState,
	chat: load<ChatEntry>(CHAT_KEY)
});

// the display is filed each time a line closes
function saveChat() {
	save(CHAT_KEY, voice.chat);
}

// The answer the component opened and has not closed yet.
function openAnswer(): ChatEntry | undefined {
	const entry = voice.chat[voice.chat.length - 1];
	return entry && entry.role === 'assistant' && entry.open ? entry : undefined;
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
	s2s.setHistory(load<S2SMessage>(HISTORY_KEY));

	s2s.on('state', (state) => {
		voice.state = state;
	});
	// A revision replaces the line it revises, and the answer drafted for it
	// that nobody heard: the component only calls it revised when the
	// context filed nothing after that line.
	s2s.on('user_text', (text, revised) => {
		let revision = 0;
		if (revised) {
			let at = voice.chat.length - 1;
			while (at >= 0 && voice.chat[at].role !== 'user') {
				at--;
			}
			revision = at >= 0 ? voice.chat[at].revision + 1 : 0;
			voice.chat.splice(Math.max(at, 0));
		}
		voice.chat.push({
			role: 'user',
			written: text,
			tools: [],
			spoken: '',
			spokenEnd: 0,
			revision,
			open: false
		});
		saveChat();
	});
	s2s.on('assistant_start', () => {
		voice.chat.push({
			role: 'assistant',
			written: '',
			tools: [],
			spoken: '',
			spokenEnd: 0,
			revision: 0,
			open: true
		});
	});
	s2s.on('assistant_delta', (text) => {
		const entry = openAnswer();
		if (entry) {
			entry.written += text;
		}
	});
	s2s.on('tool_start', (name) => {
		const entry = openAnswer();
		if (entry) {
			entry.tools.push({ at: entry.written.length, name, done: false, ok: false, ms: 0, bytes: 0 });
		}
	});
	// the calls of an answer run one after the other: the last one runs
	s2s.on('tool_end', (name, ok, ms, bytes) => {
		const tool = openAnswer()?.tools.at(-1);
		if (tool && tool.name === name && !tool.done) {
			Object.assign(tool, { done: true, ok, ms, bytes });
		}
	});
	s2s.on('assistant_text', (text, textEnd) => {
		const entry = openAnswer();
		if (entry) {
			entry.spoken += (entry.spoken ? ' ' : '') + text;
			entry.spokenEnd = textEnd;
		}
	});
	// an answer that neither wrote nor spoke leaves no line, the way the
	// context files it
	s2s.on('assistant_end', () => {
		const entry = openAnswer();
		if (!entry) {
			return;
		}
		if (!entry.written && !entry.spoken && !entry.tools.length) {
			voice.chat.pop();
		} else {
			entry.open = false;
		}
		saveChat();
	});
	s2s.on('history', saveHistory);
	// the voice and the effect the model picked are the ones the panel shows
	// and keeps
	s2s.on('voice', (label, effect) => {
		settings.voice = label;
		settings.ttsEffect = effect;
	});
	s2s.on('error', (message) => {
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
	saveHistory([]);
	saveChat();
}

// Pushes a settings change to a running session. The options are built
// before the call, so the effect reads the store even while the component is
// loading: an optional call skips its argument, and an effect that reads
// nothing never runs again.
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
