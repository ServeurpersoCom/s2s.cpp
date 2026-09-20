import type { Mode } from './config.js';
import type { S2SEcho } from './s2s.js';
import type { S2SProps } from './types.js';

const STORAGE_KEY = 's2s';

// Every field is a string, empty meaning "use the server default", which the
// placeholder shows. Numbers are parsed when the options are built, so the
// inputs stay plain text: no spinner, no min, no max.
export interface Settings {
	serverUrl: string;
	mode: Mode | '';
	llmUrl: string;
	llmModel: string;
	llmKey: string;
	systemPrompt: string;
	voice: string;
	ttsTemperature: string;
	ttsTopK: string;
	ttsTopP: string;
	ttsRepetitionPenalty: string;
	ttsSubtalkerTemperature: string;
	ttsSubtalkerTopK: string;
	ttsSubtalkerTopP: string;
	ttsMaxNewTokens: string;
	ttsSeed: string;
	ttsMinChars: string;
	ttsCharsPerSecond: string;
	ttsMarginSeconds: string;
	llmTimeoutSec: string;
	temperature: string;
	topP: string;
	topK: string;
	minP: string;
	maxTokens: string;
	presencePenalty: string;
	frequencyPenalty: string;
	seed: string;
	reasoningEffort: string;
	vadThreshold: string;
	vadNegThreshold: string;
	minSpeechMs: string;
	minSpeechContinuationMs: string;
	minSilenceMs: string;
	speechPadMs: string;
	turnThreshold: string;
	turnMaxWaitMs: string;
	turnGraceMs: string;
	echo: S2SEcho | '';
	volume: number;
	chatOpen: boolean;
	logsOpen: boolean;
	dark: boolean;
}

function defaults(): Settings {
	return {
		serverUrl: '',
		mode: '',
		llmUrl: '',
		llmModel: '',
		llmKey: '',
		systemPrompt: '',
		voice: '',
		ttsTemperature: '',
		ttsTopK: '',
		ttsTopP: '',
		ttsRepetitionPenalty: '',
		ttsSubtalkerTemperature: '',
		ttsSubtalkerTopK: '',
		ttsSubtalkerTopP: '',
		ttsMaxNewTokens: '',
		ttsSeed: '',
		ttsMinChars: '',
		ttsCharsPerSecond: '',
		ttsMarginSeconds: '',
		llmTimeoutSec: '',
		temperature: '',
		topP: '',
		topK: '',
		minP: '',
		maxTokens: '',
		presencePenalty: '',
		frequencyPenalty: '',
		seed: '',
		reasoningEffort: '',
		vadThreshold: '',
		vadNegThreshold: '',
		minSpeechMs: '',
		minSpeechContinuationMs: '',
		minSilenceMs: '',
		speechPadMs: '',
		turnThreshold: '',
		turnMaxWaitMs: '',
		turnGraceMs: '',
		echo: '',
		volume: 1,
		chatOpen: true,
		logsOpen: true,
		dark: true
	};
}

function load(): Settings {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			return { ...defaults(), ...JSON.parse(raw) };
		}
	} catch {
		// corrupt or unavailable
	}
	return defaults();
}

export const settings = $state(load());

// server defaults, null until /props answers, plus the toast the page shows
// when something the user did could not happen
export const app = $state({
	props: null as S2SProps | null,
	// true once the endpoint answered its model list: what tells the rest of
	// the UI whether a language model is in the path at all
	endpointOk: false,
	toast: '' as string,
	toastOk: false
});

let toastTimer = 0;

export function toast(msg: string, ms = 4000, ok = false) {
	clearTimeout(toastTimer);
	app.toast = msg;
	app.toastOk = ok;
	toastTimer = setTimeout(() => {
		app.toast = '';
	}, ms) as unknown as number;
}

// persist on every change
$effect.root(() => {
	$effect(() => {
		const data: Settings = {
			serverUrl: settings.serverUrl,
			mode: settings.mode,
			llmUrl: settings.llmUrl,
			llmModel: settings.llmModel,
			llmKey: settings.llmKey,
			systemPrompt: settings.systemPrompt,
			voice: settings.voice,
			ttsTemperature: settings.ttsTemperature,
			ttsTopK: settings.ttsTopK,
			ttsTopP: settings.ttsTopP,
			ttsRepetitionPenalty: settings.ttsRepetitionPenalty,
			ttsSubtalkerTemperature: settings.ttsSubtalkerTemperature,
			ttsSubtalkerTopK: settings.ttsSubtalkerTopK,
			ttsSubtalkerTopP: settings.ttsSubtalkerTopP,
			ttsMaxNewTokens: settings.ttsMaxNewTokens,
			ttsSeed: settings.ttsSeed,
			ttsMinChars: settings.ttsMinChars,
			ttsCharsPerSecond: settings.ttsCharsPerSecond,
			ttsMarginSeconds: settings.ttsMarginSeconds,
			llmTimeoutSec: settings.llmTimeoutSec,
			temperature: settings.temperature,
			topP: settings.topP,
			topK: settings.topK,
			minP: settings.minP,
			maxTokens: settings.maxTokens,
			presencePenalty: settings.presencePenalty,
			frequencyPenalty: settings.frequencyPenalty,
			seed: settings.seed,
			reasoningEffort: settings.reasoningEffort,
			vadThreshold: settings.vadThreshold,
			vadNegThreshold: settings.vadNegThreshold,
			minSpeechMs: settings.minSpeechMs,
			minSpeechContinuationMs: settings.minSpeechContinuationMs,
			minSilenceMs: settings.minSilenceMs,
			speechPadMs: settings.speechPadMs,
			turnThreshold: settings.turnThreshold,
			turnMaxWaitMs: settings.turnMaxWaitMs,
			turnGraceMs: settings.turnGraceMs,
			echo: settings.echo,
			volume: settings.volume,
			chatOpen: settings.chatOpen,
			logsOpen: settings.logsOpen,
			dark: settings.dark
		};
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	});
});
