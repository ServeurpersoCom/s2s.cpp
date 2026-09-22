import type { S2S, S2SMcpServer, S2SMessage, S2SOptions, S2SState } from './s2s.js';
import { postLog } from './api.js';
import { num } from './fields.js';
import { app, settings, toast } from './state.svelte.js';

// A value the server does not serve is left out and its default applies: a
// voice or a language kept from another server. Before /props answers there
// is no list to judge by, and the value goes as it is.
function served(value: string, list: string[] | undefined): string | undefined {
	return value && (!list || list.includes(value)) ? value : undefined;
}

export function sessionVoice(): string | undefined {
	return served(settings.voice, app.props?.defaults.tts_voices);
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
		echo: settings.echo || undefined
	};
}

// One line of the conversation. draft is what the model wrote, spokenEnd how
// far into it the voice went: it trails the draft by one synthesis unit, and
// stops there on a barge-in.
interface ChatTurn {
	role: 'user' | 'assistant';
	draft: string;
	spokenEnd: number;
	done: boolean;
}

// The conversation survives a reload: the page files every version the
// component reports and seeds the next component with it. It holds what
// the model knows, so a reloaded line is what was heard, never the text
// written past it. Its own key: a reset of the settings leaves it alone.
const HISTORY_KEY = 's2s.history';

function loadHistory(): S2SMessage[] {
	try {
		const raw = localStorage.getItem(HISTORY_KEY);
		if (raw) {
			return JSON.parse(raw) as S2SMessage[];
		}
	} catch {
		// corrupt or unavailable
	}
	return [];
}

// the turn identity dies with its connection, so only the words are kept
function saveHistory(messages: S2SMessage[]) {
	try {
		const words = messages.map(({ role, content }) => ({ role, content }));
		localStorage.setItem(HISTORY_KEY, JSON.stringify(words));
	} catch {
		// full or unavailable
	}
}

// Live view of the component, for whatever the page decides to draw. Nothing
// here touches the DOM: the component stays invisible until a host renders
// something from this state. The conversation filed by the last visit shows
// before any start.
export const voice = $state({
	state: 'idle' as S2SState,
	chat: loadHistory().map(({ role, content }): ChatTurn => ({
		role,
		draft: content,
		spokenEnd: content.length,
		done: true
	}))
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
	s2s.setHistory(loadHistory());

	s2s.on('state', (state) => {
		voice.state = state;
		if (state === 'listening') {
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
	// a finished answer closes its line; one that wrote nothing has none, the
	// way the conversation files it
	s2s.on('assistant_done', () => {
		closeAssistant();
	});
	s2s.on('assistant_text', (text, textEnd) => {
		const turn = openAssistant();
		// loopback has no model writing ahead: what is spoken is the whole turn
		if (!turn.draft) {
			turn.draft = text;
		}
		turn.spokenEnd = textEnd;
	});
	s2s.on('history', saveHistory);
	// everything the component reports goes to the server log, so the card on
	// the right tells the whole story, and a failure also pops the toast
	s2s.on('log', (line) => {
		postLog(line);
	});
	s2s.on('error', (message) => {
		postLog(`Error: ${message}`);
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
