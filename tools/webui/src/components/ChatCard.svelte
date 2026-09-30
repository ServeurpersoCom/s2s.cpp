<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import { app, settings } from '../lib/state.svelte.js';
	import { voice, type ChatEntry, type ChatTool } from '../lib/voice.svelte.js';

	let body: HTMLDivElement | null = $state(null);

	// Log style prefixes, padded so both speakers start their text on the same
	// column. The one place indentation is welcome: it is alignment, not code.
	const ASSISTANT = '[Assistant]';
	const user = (revision: number) => `[User-${revision}]`.padStart(ASSISTANT.length);
	const SYSTEM = '[System]'.padStart(ASSISTANT.length);

	// The written text of an answer cut at its calls: every piece knows
	// whether the voice reached it, and every call sits where it was made.
	type Part = { text: string; ahead: boolean } | { tool: ChatTool };

	function parts(entry: ChatEntry): Part[] {
		const out: Part[] = [];
		let from = 0;
		const cut = (to: number) => {
			const spoken = Math.min(Math.max(entry.spokenEnd, from), to);
			if (spoken > from) {
				out.push({ text: entry.written.slice(from, spoken), ahead: false });
			}
			if (to > spoken) {
				out.push({ text: entry.written.slice(spoken, to), ahead: true });
			}
			from = to;
		};
		for (const tool of entry.tools) {
			cut(tool.at);
			out.push({ tool });
		}
		cut(entry.written.length);
		return out;
	}

	// its name while it runs, then the time it ran and the size of its result
	function tag(tool: ChatTool): string {
		if (!tool.done) {
			return `<${tool.name}`;
		}
		return tool.ok ? `<${tool.name} ${tool.ms} ms ${tool.bytes} bytes>` : `<${tool.name} failed>`;
	}

	const LOOPBACK_HELP =
		'Loopback: say something and the voice speaks back what it heard, word for word. No language model in the path, to tune the listening and the voice before switching to Conversation.';

	// The first line of the log: the system prompt the model reads, or in
	// loopback, where no model reads anything, what the mode is for.
	let d = $derived(app.props?.defaults);
	let mode = $derived(settings.mode || d?.mode || '');
	let system = $derived(
		mode === 'loopback' ? LOOPBACK_HELP : settings.systemPrompt || d?.instructions || ''
	);

	// follow the stream, the way a terminal does
	$effect(() => {
		void voice.chat.length;
		void voice.chat[voice.chat.length - 1]?.written;
		void voice.chat[voice.chat.length - 1]?.spoken;
		void voice.chat[voice.chat.length - 1]?.tools.at(-1)?.done;
		if (body) {
			body.scrollTop = body.scrollHeight;
		}
	});
</script>

<div class="card">
	<button class="card-header" onclick={() => (settings.chatOpen = !settings.chatOpen)}>
		{#if settings.chatOpen}
			<ChevronDown size={14} />
		{:else}
			<ChevronRight size={14} />
		{/if}
		<span class="card-label">Conversation</span>
	</button>
	{#if settings.chatOpen}
		<div class="chat-body" bind:this={body}>
			{#if system}
				<div class="turn">
					<span class="who system">{SYSTEM}</span>
					{system}
				</div>
			{/if}
			{#each voice.chat as entry, i (i)}
				<div class="turn">
					<span class="who {entry.role}"
						>{entry.role === 'user' ? user(entry.revision) : ASSISTANT}</span
					>
					{#if entry.role === 'user'}
						{entry.written}
					{:else if entry.written || entry.tools.length}
						{#each parts(entry) as part, j (j)}{#if 'tool' in part}<span
									class="tool"
									class:failed={part.tool.done && !part.tool.ok}>{tag(part.tool)}</span
								>{:else}<span class:ahead={part.ahead}>{part.text}</span>{/if}{/each}
					{:else}
						{entry.spoken}
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		border: none;
		border-radius: 4px;
		background: var(--bg-card);
		overflow: hidden;
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.5rem;
		background: var(--bg-input);
		border: none;
		cursor: pointer;
		color: var(--fg);
		font-size: 0.8rem;
		text-align: left;
	}
	.card-header:hover {
		background: var(--bg-btn-hover);
	}
	.card-label {
		font-weight: 600;
	}
	.chat-body {
		/* the log body, to the letter: same font, same size, same rhythm */
		padding: 0.4rem 0.5rem;
		/* half the log height, so the two cards share the column */
		max-height: 24rem;
		overflow: auto;
		font-family: monospace;
		font-size: 0.7rem;
		line-height: 1.4;
		color: var(--fg-dim);
	}
	/* the speaker is told by its colour, inline, with no line of its own */
	.who {
		white-space: pre;
	}
	.who.user {
		color: var(--who-user);
	}
	.who.assistant {
		color: var(--who-assistant);
	}
	.who.system {
		color: var(--fg-dim);
	}
	/* written by the model and not spoken: ahead of the voice, or never said */
	.ahead {
		color: var(--error);
	}
	/* a call of the model, inserted as is between two deltas */
	.tool {
		color: var(--fg);
	}
	.tool.failed {
		color: var(--error);
	}
</style>
