<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { RefreshCw, X } from '@lucide/svelte';
	import { app, settings, toast } from '../lib/state.svelte.js';
	import { fetchModels, props } from '../lib/api.js';
	import { snippet } from '../lib/snippet.js';
	import { ph } from '../lib/fields.js';
	import { ENDPOINT_EXAMPLE, MODES, type Mode } from '../lib/config.js';
	import { ECHO_DEFAULT, ECHO_MODES, type S2SEcho } from '../lib/s2s.js';
	import { clearContext, getClient, voice } from '../lib/voice.svelte.js';

	let models = $state<string[]>([]);

	let d = $derived(app.props?.defaults);

	// the mode in force: the field when set, the server default otherwise
	let mode = $derived(settings.mode || d?.mode || '');

	// a language model is in the path only when the mode asks for one and the
	// endpoint answered: every consumer reads this, nobody recomputes it
	// a server that owns its endpoint answers for it, and keeps it out of the page
	let endpointReady = $derived(mode === 'conversation' && (app.endpointOk || !!d?.llm_fixed));

	// modes are lowercase on the wire, spelled out on screen
	const MODE_LABELS: Record<string, string> = {
		loopback: 'Loopback, to test the system',
		conversation: 'Conversation, to plug your llama.cpp'
	};

	function label(mode: string): string {
		return MODE_LABELS[mode] ?? mode;
	}

	function onMode(e: Event) {
		settings.mode = (e.target as HTMLSelectElement).value as Mode;
	}

	// A select shows the value in force, which is the field when it is set and
	// the server default otherwise: listing the default as an extra entry is
	// what put ryan and auto in there twice.
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

	function onSpeaker(e: Event) {
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
			app.endpointOk = false;
			return;
		}
		try {
			models = await fetchModels(settings.llmUrl, settings.llmKey);
			if (models.length > 0 && !models.includes(settings.llmModel)) {
				settings.llmModel = models[0];
			}
			app.endpointOk = models.length > 0;
		} catch {
			models = [];
			app.endpointOk = false;
		}
	}

	function loadProps() {
		props()
			.then((p) => (app.props = p))
			.catch(() => (app.props = null));
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
		settings.vadThreshold = '';
		settings.vadNegThreshold = '';
		settings.minSpeechMs = '';
		settings.minSpeechContinuationMs = '';
		settings.minSilenceMs = '';
		settings.speechPadMs = '';
	}

	function clearTurn() {
		settings.turnThreshold = '';
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
	let listsModels = $derived(!!d && mode === 'conversation' && !d.llm_fixed);

	$effect(() => {
		if (listsModels) {
			untrack(loadModels);
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
					>Threshold min <input
						type="text"
						placeholder={ph(d?.vad_neg_threshold)}
						bind:value={settings.vadNegThreshold}
					/></label
				>
				<label
					>Threshold max <input
						type="text"
						placeholder={ph(d?.vad_threshold)}
						bind:value={settings.vadThreshold}
					/></label
				>
				<label
					>Min speech ms <input
						type="text"
						placeholder={ph(d?.min_speech_ms)}
						bind:value={settings.minSpeechMs}
					/></label
				>
				<label
					>Continuation ms <input
						type="text"
						placeholder={ph(d?.min_speech_continuation_ms)}
						bind:value={settings.minSpeechContinuationMs}
					/></label
				>
				<label
					>Min silence ms <input
						type="text"
						placeholder={ph(d?.min_silence_ms)}
						bind:value={settings.minSilenceMs}
					/></label
				>
				<label
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
					>Threshold <input
						type="text"
						placeholder={ph(d?.turn_threshold)}
						bind:value={settings.turnThreshold}
					/></label
				>
				<label
					>Max wait ms <input
						type="text"
						placeholder={ph(d?.turn_max_wait_ms)}
						bind:value={settings.turnMaxWaitMs}
					/></label
				>
				<label
					>Reopen grace ms <input
						type="text"
						placeholder={ph(d?.reopen_grace_ms)}
						bind:value={settings.turnGraceMs}
					/></label
				>
			</div>
		</div>
	</details>

	{#if mode === 'conversation'}
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
							onclick={loadModels}
							aria-label="Reload the model list"
						>
							<RefreshCw size={16} />
						</button>
					</div>
				{/if}

				<label
					>System prompt <textarea
						rows="8"
						placeholder={ph(d?.instructions)}
						bind:value={settings.systemPrompt}
					></textarea></label
				>

				<div class="meta-grid">
					<label>Temperature <input type="text" bind:value={settings.temperature} /></label>
					<label>Top P <input type="text" bind:value={settings.topP} /></label>
					<label>Top K <input type="text" bind:value={settings.topK} /></label>
					<label>Min P <input type="text" bind:value={settings.minP} /></label>
					<label>Max tokens <input type="text" bind:value={settings.maxTokens} /></label>
					<label>Presence <input type="text" bind:value={settings.presencePenalty} /></label>
					<label>Frequency <input type="text" bind:value={settings.frequencyPenalty} /></label>
					<label>Seed <input type="text" bind:value={settings.seed} /></label>
					<label>Reasoning <input type="text" bind:value={settings.reasoningEffort} /></label>
					<label
						>Timeout s <input
							type="text"
							placeholder={ph(d?.llm_timeout_sec)}
							bind:value={settings.llmTimeoutSec}
						/></label
					>
				</div>
			</div>
		</details>
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
			<div class="model-row">
				<span class="model-label">Speaker</span>
				<select class="model-select" value={settings.voice || d?.voice || ''} onchange={onSpeaker}>
					{#each d?.tts_speakers ?? [] as speaker (speaker)}
						<option value={speaker}>{speaker}</option>
					{/each}
				</select>
			</div>

			<div class="model-row">
				<span class="model-label">Language</span>
				<select
					class="model-select"
					value={settings.language || d?.language || 'auto'}
					onchange={onLanguage}
				>
					<option value="auto">auto</option>
					{#each d?.tts_languages ?? [] as language (language)}
						<option value={language}>{language}</option>
					{/each}
				</select>
			</div>

			<div class="meta-grid">
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
					>Max frames <input
						type="text"
						placeholder={ph(d?.tts_max_new_tokens)}
						bind:value={settings.ttsMaxNewTokens}
					/></label
				>
				<label>Seed <input type="text" bind:value={settings.ttsSeed} /></label>
				<label
					>Min chars <input
						type="text"
						placeholder={ph(d?.tts_min_chars)}
						bind:value={settings.ttsMinChars}
					/></label
				>
				<label
					>Chars per s <input
						type="text"
						placeholder={ph(d?.tts_chars_per_second)}
						bind:value={settings.ttsCharsPerSecond}
					/></label
				>
				<label
					>Margin s <input
						type="text"
						placeholder={ph(d?.tts_margin_seconds)}
						bind:value={settings.ttsMarginSeconds}
					/></label
				>
			</div>
		</div>
	</details>

	<div class="action-row">
		<button
			type="button"
			disabled={!endpointReady}
			onclick={clearContext}
			title="Drop the conversation history">Clear context</button
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
