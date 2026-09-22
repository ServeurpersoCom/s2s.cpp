<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { RefreshCw, X } from '@lucide/svelte';
	import { app, settings, toast } from '../lib/state.svelte.js';
	import { fetchModels, fetchTools, props } from '../lib/api.js';
	import { snippet } from '../lib/snippet.js';
	import { ph } from '../lib/fields.js';
	import { ENDPOINT_EXAMPLE, MODES, type Mode } from '../lib/config.js';
	import { ECHO_DEFAULT, ECHO_MODES, type S2SEcho } from '../lib/s2s.js';
	import {
		clearContext,
		getClient,
		sessionLanguage,
		sessionVoice,
		voice
	} from '../lib/voice.svelte.js';

	let models = $state<string[]>([]);
	let tools = $state<string[]>([]);

	let d = $derived(app.props?.defaults);

	// the mode in force: the field when set, the server default otherwise
	let mode = $derived(settings.mode || d?.mode || '');

	// modes are lowercase on the wire, spelled out on screen
	const MODE_LABELS: Record<string, string> = {
		loopback: 'Loopback, to test the system',
		conversation: 'Conversation, to plug your LLM',
		agentic: 'Agentic, llama.cpp built-in tools'
	};

	function label(mode: string): string {
		return MODE_LABELS[mode] ?? mode;
	}

	function onMode(e: Event) {
		settings.mode = (e.target as HTMLSelectElement).value as Mode;
	}

	// the whole configuration, as the code an integrator pastes on their page
	async function copySnippet() {
		try {
			await navigator.clipboard.writeText(snippet());
			toast('Snippet copied', 2000, true);
		} catch {
			toast('Clipboard refused, the page needs a secure context');
		}
	}

	const ECHO_LABELS: Record<S2SEcho, string> = {
		server: 'Server, every browser',
		native: 'Native, Chrome only',
		off: 'Off, use headphones'
	};

	function onEcho(e: Event) {
		settings.echo = (e.target as HTMLSelectElement).value as S2SEcho;
	}

	function onModel(e: Event) {
		settings.llmModel = (e.target as HTMLSelectElement).value;
	}

	// A select shows the value in force: the field when the server serves it,
	// the server default otherwise, never an extra entry for either.
	function onVoice(e: Event) {
		settings.voice = (e.target as HTMLSelectElement).value;
	}

	function onLanguage(e: Event) {
		settings.language = (e.target as HTMLSelectElement).value;
	}

	// an endpoint that answers fills the selector, one that does not leaves the
	// free text field. The failure itself lands in the server logs.
	// the selector is either filled by the endpoint or empty: a model name is
	// what the endpoint answered, never a placeholder invented here.
	async function loadModels() {
		if (!settings.llmUrl) {
			models = [];
			return;
		}
		try {
			models = await fetchModels(settings.llmUrl, settings.llmKey);
			if (models.length > 0 && !models.includes(settings.llmModel)) {
				settings.llmModel = models[0];
			}
		} catch (e: unknown) {
			models = [];
			toast(e instanceof Error ? e.message : String(e));
		}
	}

	// the tools the endpoint runs, which only a llama.cpp server has: an
	// endpoint without them answers with an error, the list stays empty and
	// the toast says why.
	async function loadTools() {
		try {
			tools = await fetchTools(settings.llmUrl, settings.llmKey);
		} catch (e: unknown) {
			tools = [];
			toast(e instanceof Error ? e.message : String(e));
		}
	}

	// a tool the endpoint no longer runs stays checked in the settings until
	// the user says otherwise: the list is the session's, not the endpoint's.
	function onTool(name: string, e: Event) {
		const checked = (e.target as HTMLInputElement).checked;
		settings.tools = checked
			? [...settings.tools, name]
			: settings.tools.filter((tool) => tool !== name);
	}

	// One gesture fills both lists: in the agentic mode the tools come from
	// the endpoint the models come from.
	function reload() {
		loadModels();
		if (mode === 'agentic') {
			loadTools();
		}
	}

	function loadProps() {
		props()
			.then((p) => (app.props = p))
			.catch((e: unknown) => {
				app.props = null;
				toast(e instanceof Error ? e.message : String(e));
			});
	}

	// an empty sampling field leaves the endpoint to its own default, so
	// clearing the section is how a user gets back to it
	function clearLlm() {
		settings.llmUrl = '';
		settings.llmModel = '';
		settings.llmKey = '';
		settings.systemPrompt = '';
		settings.temperature = '';
		settings.topP = '';
		settings.topK = '';
		settings.minP = '';
		settings.maxTokens = '';
		settings.presencePenalty = '';
		settings.frequencyPenalty = '';
		settings.seed = '';
		settings.reasoningEffort = '';
		settings.llmTimeoutSec = '';
	}

	function clearTts() {
		settings.voice = '';
		settings.language = '';
		settings.ttsTemperature = '';
		settings.ttsTopK = '';
		settings.ttsTopP = '';
		settings.ttsRepetitionPenalty = '';
		settings.ttsSubtalkerTemperature = '';
		settings.ttsSubtalkerTopK = '';
		settings.ttsSubtalkerTopP = '';
		settings.ttsMaxNewTokens = '';
		settings.ttsSeed = '';
		settings.ttsMinChars = '';
		settings.ttsCharsPerSecond = '';
		settings.ttsMarginSeconds = '';
	}

	function clearVad() {
		settings.vadNegThreshold = '';
		settings.vadThreshold = '';
		settings.minSpeechMs = '';
		settings.bargeInMs = '';
		settings.minSilenceMs = '';
		settings.minSpeechContinuationMs = '';
		settings.speechPadMs = '';
	}

	function clearTurn() {
		settings.turnThreshold = '';
		settings.turnIncompleteDelayMs = '';
		settings.turnMaxWaitMs = '';
		settings.turnGraceMs = '';
	}

	// start() has to run inside the click: the browser grants the microphone
	// and resumes the audio context only from a user gesture. The component
	// reports the reason on its own, so the rejection is only swallowed here.
	function start() {
		getClient()
			?.start()
			.catch(() => {});
	}

	function stop() {
		getClient()?.stop();
	}

	onMount(loadProps);

	// The page lists the models once /props says the endpoint belongs to the
	// session, and only in conversation. The effect follows this boolean alone:
	// /props landing or a field being typed does not change it, so neither
	// sends a request.
	let listsModels = $derived(!!d && mode !== 'loopback' && !d.llm_fixed);

	$effect(() => {
		if (listsModels) {
			untrack(loadModels);
		}
	});

	// The tool list follows the mode the same way, and needs no endpoint of
	// its own: a server that owns its endpoint asks it for the page.
	let listsTools = $derived(!!d && mode === 'agentic');

	$effect(() => {
		if (listsTools) {
			untrack(loadTools);
		}
	});
</script>

<div class="config">
	<details open>
		<summary>Playground session</summary>
		<div class="details-body">
			<div class="model-row">
				<span class="model-label">Server</span>
				<input
					class="model-select"
					type="text"
					placeholder={new URL('./', location.href).href}
					bind:value={settings.serverUrl}
				/>
			</div>

			<div class="model-row">
				<span class="model-label">Mode</span>
				<select class="model-select" value={mode} onchange={onMode}>
					{#each MODES as mode (mode)}
						<option value={mode}>{label(mode)}</option>
					{/each}
				</select>
			</div>

			<div class="model-row">
				<span class="model-label">Models</span>
				<span class="model-files">
					{#each Object.values(app.props?.models ?? {}) as file (file)}
						<span class="model-file">{file}</span>
					{/each}
				</span>
			</div>
		</div>
	</details>

	<details>
		<summary>Echo cancellation</summary>
		<div class="details-body">
			<div class="model-row">
				<span class="model-label">Method</span>
				<select class="model-select" value={settings.echo || ECHO_DEFAULT} onchange={onEcho}>
					{#each ECHO_MODES as echo (echo)}
						<option value={echo}>{ECHO_LABELS[echo]}</option>
					{/each}
				</select>
			</div>
		</div>
	</details>

	<details class="has-clear">
		<summary>Voice Activity Detection</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear voice activity detection"
			onclick={clearVad}
			aria-label="Clear voice activity detection"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div class="meta-grid">
				<label
					title="Speech goes on while the voice probability stays at or above this value, from 0 to 1. Lower keeps a word whole through a dip or a faint consonant; higher cuts sooner on noise."
					>Threshold min <input
						type="text"
						placeholder={ph(d?.vad_neg_threshold)}
						bind:value={settings.vadNegThreshold}
					/></label
				>
				<label
					title="Speech starts when the voice probability reaches this value, from 0 to 1. Higher ignores more noise and echo; lower catches a softer voice."
					>Threshold max <input
						type="text"
						placeholder={ph(d?.vad_threshold)}
						bind:value={settings.vadThreshold}
					/></label
				>
				<label
					title="Speech needed to open a turn while the assistant is silent. Lower lets a short 'yes' through; higher keeps coughs and clicks out."
					>Min speech ms <input
						type="text"
						placeholder={ph(d?.min_speech_ms)}
						bind:value={settings.minSpeechMs}
					/></label
				>
				<label
					title="Speech needed to cut the assistant while it talks. Longer than Min speech on purpose: interrupting costs more than a turn opened on a noise, and the echo canceller leaves a residue."
					>Barge-in ms <input
						type="text"
						placeholder={ph(d?.barge_in_ms)}
						bind:value={settings.bargeInMs}
					/></label
				>
				<label
					title="Silence that ends the speech and asks the end of turn detection whether the turn is over. Lower judges sooner; higher lets short pauses pass unjudged."
					>Min silence ms <input
						type="text"
						placeholder={ph(d?.min_silence_ms)}
						bind:value={settings.minSilenceMs}
					/></label
				>
				<label
					title="Speech needed to go on after a pause, in the same turn. Short, since the speaker is already talking and the first syllable must not be lost."
					>Continuation ms <input
						type="text"
						placeholder={ph(d?.min_speech_continuation_ms)}
						bind:value={settings.minSpeechContinuationMs}
					/></label
				>
				<label
					title="Audio kept before the speech starts and after it ends, so the recognizer gets the first consonant and the last one. The silence beyond it is left out."
					>Speech pad ms <input
						type="text"
						placeholder={ph(d?.speech_pad_ms)}
						bind:value={settings.speechPadMs}
					/></label
				>
			</div>
		</div>
	</details>

	<details class="has-clear">
		<summary>End of Turn Detection</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear end of turn detection"
			onclick={clearTurn}
			aria-label="Clear end of turn detection"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div class="meta-grid">
				<label
					title="Smart Turn decides at each pause whether the sentence is finished, from 0 to 1. Above this value the turn is committed; below, it waits for more speech. Higher waits more readily; lower answers sooner, at the risk of cutting a thought."
					>Threshold <input
						type="text"
						placeholder={ph(d?.turn_threshold)}
						bind:value={settings.turnThreshold}
					/></label
				>
				<label
					title="How long a pause judged unfinished stays free: going on within it costs nothing. After it the answer starts computing in silence, and going on then still continues the same turn, at the price of a wasted request. Shorter hides a slower endpoint; longer wastes less."
					>Incomplete delay ms <input
						type="text"
						placeholder={ph(d?.incomplete_delay_ms)}
						bind:value={settings.turnIncompleteDelayMs}
					/></label
				>
				<label
					title="Longest a pause judged unfinished can last before the answer is heard, so a conversation never stalls. It is also the delay of a single word the model finds unfinished."
					>Max wait ms <input
						type="text"
						placeholder={ph(d?.turn_max_wait_ms)}
						bind:value={settings.turnMaxWaitMs}
					/></label
				>
				<label
					title="Least silence kept after a finished sentence before the answer may be heard. Going on within it, or before any answer is heard, continues the same turn, recognized again as one sentence."
					>Reopen grace ms <input
						type="text"
						placeholder={ph(d?.reopen_grace_ms)}
						bind:value={settings.turnGraceMs}
					/></label
				>
			</div>
		</div>
	</details>

	{#if mode !== 'loopback'}
		<details class="has-clear">
			<summary>LLM, OpenAI compatible endpoint</summary>
			<button
				type="button"
				class="clear-btn details-clear"
				title="Clear LLM"
				onclick={clearLlm}
				aria-label="Clear LLM"
			>
				<X size={20} />
			</button>
			<div class="details-body">
				{#if !d?.llm_fixed}
					<div class="model-row">
						<span class="model-label">URL</span>
						<input
							class="model-select"
							type="text"
							placeholder={ENDPOINT_EXAMPLE}
							bind:value={settings.llmUrl}
						/>
					</div>

					<div class="model-row">
						<span class="model-label">API key</span>
						<input class="model-select" type="password" bind:value={settings.llmKey} />
					</div>

					<div class="model-row">
						<span class="model-label">Model</span>
						<select class="model-select" value={settings.llmModel} onchange={onModel}>
							{#each models as model (model)}
								<option value={model}>{model}</option>
							{/each}
						</select>
						<button
							type="button"
							class="clear-btn"
							onclick={reload}
							aria-label="Reload what the endpoint offers"
						>
							<RefreshCw size={16} />
						</button>
					</div>
				{/if}

				<label
					>System prompt <textarea
						rows="4"
						placeholder={ph(d?.instructions)}
						bind:value={settings.systemPrompt}
					></textarea></label
				>

				<div class="meta-grid">
					<label
						title="Longest the endpoint may stay silent, in seconds, before the answer is dropped with an error: reaching it, a model it loads, the prompt it reads before the first word, and every pause of the stream. A long answer never trips it. A real time voice has no use for a model slower than that."
						>Timeout <input
							type="text"
							placeholder={ph(d?.llm_timeout_sec)}
							bind:value={settings.llmTimeoutSec}
						/></label
					>
					<label
						title="Sent as reasoning_effort, OpenAI compatible. none is the best for a conversation: no thinking, the fastest answer, and llama.cpp takes it with any model. For other values, see the chat template of the model or the documentation of the provider."
						>Reasoning <input
							type="text"
							placeholder={ph(d?.reasoning_effort)}
							bind:value={settings.reasoningEffort}
						/></label
					>
					<label>Temperature <input type="text" bind:value={settings.temperature} /></label>
					<label>Top P <input type="text" bind:value={settings.topP} /></label>
					<label>Top K <input type="text" bind:value={settings.topK} /></label>
					<label>Min P <input type="text" bind:value={settings.minP} /></label>
					<label>Max tokens <input type="text" bind:value={settings.maxTokens} /></label>
					<label>Presence <input type="text" bind:value={settings.presencePenalty} /></label>
					<label>Frequency <input type="text" bind:value={settings.frequencyPenalty} /></label>
					<label>Seed <input type="text" bind:value={settings.seed} /></label>
				</div>
			</div>
		</details>

		{#if mode === 'agentic'}
			<details>
				<summary>Tools, run by the llama.cpp endpoint</summary>
				<div class="details-body">
					<div class="meta-grid">
						<label
							title="Rounds of tool calls one answer may take: the model calls, the endpoint runs, the model reads the results and may call again. Reaching the cap fails the answer."
							>Agentic rounds <input
								type="text"
								placeholder={ph(d?.max_rounds)}
								bind:value={settings.maxRounds}
							/></label
						>
						<label
							title="Longest a tool may work, in seconds, before its call fails: the endpoint runs it, a search or a fetch, and answers once it is done. Longer than the model timeout, a tool goes out to the network."
							>Timeout <input
								type="text"
								placeholder={ph(d?.tool_timeout_sec)}
								bind:value={settings.toolTimeoutSec}
							/></label
						>
					</div>

					{#each tools as tool (tool)}
						<label class="tool"
							><input
								type="checkbox"
								checked={settings.tools.includes(tool)}
								onchange={(e) => onTool(tool, e)}
							/>{tool}</label
						>
					{:else}
						<span class="tool">The endpoint lists no tool</span>
					{/each}
				</div>
			</details>
		{/if}
	{/if}

	<details class="has-clear">
		<summary>Text-to-Speech</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear TTS"
			onclick={clearTts}
			aria-label="Clear TTS"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div
				class="model-row"
				title="The voice every sentence is cloned from, read from voices/ at startup. reference speech continues a recording: timbre, pace and accent. speaker embedding only keeps the timbre, and starts faster."
			>
				<span class="model-label">Voice</span>
				<select class="model-select" value={sessionVoice() || d?.voice || ''} onchange={onVoice}>
					{#each d?.tts_voices ?? [] as name (name)}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>

			<div
				class="model-row"
				title="The language id the talker is given for every sentence. It only weighs on a lone word, which is read in the language of the id: a few words are read in the language they are written in, whatever the id. auto takes the language of the browser when the talker speaks it, English otherwise, where Merci comes out as the English mercy."
			>
				<span class="model-label">Language</span>
				<select
					class="model-select"
					value={sessionLanguage() || d?.language || 'auto'}
					onchange={onLanguage}
				>
					<option value="auto">auto</option>
					{#each d?.tts_languages ?? [] as name (name)}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>

			<div class="meta-grid">
				<label
					title="Loopback only: a transcript shorter than this is not spoken back, it is the tail of a noise the recognizer had to name. The answers of the model are always spoken."
					>Min chars <input
						type="text"
						placeholder={ph(d?.tts_min_chars)}
						bind:value={settings.ttsMinChars}
					/></label
				>
				<label
					title="Speaking rate the time budget of a sentence assumes: its length divided by this, plus the margin. A talker still speaking at the end of its budget is stopped. Lower gives a slow voice more room."
					>Chars per s <input
						type="text"
						placeholder={ph(d?.tts_chars_per_second)}
						bind:value={settings.ttsCharsPerSecond}
					/></label
				>
				<label
					title="Seconds added to the budget of every sentence, for its pauses and a slow start. It only cuts a talker that runs on without ending."
					>Margin s <input
						type="text"
						placeholder={ph(d?.tts_margin_seconds)}
						bind:value={settings.ttsMarginSeconds}
					/></label
				>
				<label
					title="Hard cap on the budget of any sentence, in codec frames at 12.5 per second, whatever its length."
					>Max frames <input
						type="text"
						placeholder={ph(d?.tts_max_new_tokens)}
						bind:value={settings.ttsMaxNewTokens}
					/></label
				>
				<label
					>Temperature <input
						type="text"
						placeholder={ph(d?.tts_temperature)}
						bind:value={settings.ttsTemperature}
					/></label
				>
				<label
					>Top K <input
						type="text"
						placeholder={ph(d?.tts_top_k)}
						bind:value={settings.ttsTopK}
					/></label
				>
				<label
					>Top P <input
						type="text"
						placeholder={ph(d?.tts_top_p)}
						bind:value={settings.ttsTopP}
					/></label
				>
				<label
					>Repetition <input
						type="text"
						placeholder={ph(d?.tts_repetition_penalty)}
						bind:value={settings.ttsRepetitionPenalty}
					/></label
				>
				<label
					>Sub temperature <input
						type="text"
						placeholder={ph(d?.tts_subtalker_temperature)}
						bind:value={settings.ttsSubtalkerTemperature}
					/></label
				>
				<label
					>Sub top K <input
						type="text"
						placeholder={ph(d?.tts_subtalker_top_k)}
						bind:value={settings.ttsSubtalkerTopK}
					/></label
				>
				<label
					>Sub top P <input
						type="text"
						placeholder={ph(d?.tts_subtalker_top_p)}
						bind:value={settings.ttsSubtalkerTopP}
					/></label
				>
				<label
					title="Empty draws a new seed for every sentence, shown in the log. Set one to replay a sentence exactly, to compare two settings on the same take."
					>Seed <input type="text" bind:value={settings.ttsSeed} /></label
				>
			</div>
		</div>
	</details>

	<div class="action-row">
		<button type="button" onclick={clearContext} title="Drop the conversation history"
			>Clear context</button
		>
		<button
			type="button"
			disabled={voice.state !== 'idle'}
			onclick={start}
			title="Open the microphone and connect">Start</button
		>
		<button
			type="button"
			disabled={voice.state === 'idle'}
			onclick={stop}
			title="Close the microphone and disconnect">Stop</button
		>
	</div>

	<div class="action-row">
		<button
			type="button"
			onclick={copySnippet}
			title="Ready to paste on your own page, with the settings shown here"
			>Copy the client code with your current settings</button
		>
	</div>
</div>

<style>
	.config {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.85rem;
		color: var(--fg-dim);
	}
	.has-clear {
		position: relative;
	}
	.details-clear {
		position: absolute;
		top: 0.4rem;
		right: 0;
	}
	.clear-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		border: none;
		background: transparent;
		color: var(--fg-dim);
		cursor: pointer;
		line-height: 0;
	}
	.clear-btn:hover {
		color: var(--fg);
	}
	textarea,
	input[type='text'],
	input[type='password'],
	select {
		font-family: inherit;
		font-size: 0.9rem;
		padding: 0.4rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-input);
		color: var(--fg);
		resize: vertical;
	}
	textarea:focus,
	input:focus,
	select:focus {
		outline: 2px solid var(--focus);
		outline-offset: -1px;
	}
	.meta-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(8rem, 1fr));
		gap: 0.5rem;
	}
	.tool {
		flex-direction: row;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.85rem;
		color: var(--fg-dim);
	}
	.model-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.model-label {
		font-size: 0.85rem;
		color: var(--fg-dim);
		flex-shrink: 0;
		min-width: 4rem;
	}
	.model-select {
		flex: 1;
		min-width: 0;
	}
	.model-files {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		font-size: 0.7rem;
		line-height: 1.4;
		color: var(--fg-dim);
		min-width: 0;
	}
	.model-file {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	details summary {
		cursor: pointer;
		font-size: 0.85rem;
		color: var(--fg);
		font-weight: 600;
		padding: 0.4rem 0;
	}
	.details-body {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.25rem 0 0.5rem;
	}
	.action-row {
		display: flex;
		gap: 0.5rem;
	}
	.action-row button {
		flex: 1;
	}
	button {
		padding: 0.5rem 1rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-btn);
		color: var(--fg);
		cursor: pointer;
		font-size: 0.85rem;
	}
	button:hover:not(:disabled) {
		background: var(--bg-btn-hover);
	}
	button:disabled {
		opacity: 0.4;
	}
	.clear-btn:hover {
		background: transparent;
	}
</style>
