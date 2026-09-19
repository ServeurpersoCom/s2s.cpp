<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import { settings } from '../lib/state.svelte.js';
	import { voice } from '../lib/voice.svelte.js';

	let body: HTMLDivElement | null = $state(null);

	// Log style prefixes, padded so both speakers start their text on the same
	// column. The one place indentation is welcome: it is alignment, not code.
	const ASSISTANT = '[Assistant]';
	const USER = '[User]'.padStart(ASSISTANT.length);

	// follow the stream, the way a terminal does
	$effect(() => {
		void voice.chat.length;
		void voice.chat[voice.chat.length - 1]?.draft;
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
			{#each voice.chat as turn, i (i)}
				<div class="turn">
					<span class="who {turn.role}">{turn.role === 'user' ? USER : ASSISTANT}</span>
					{turn.draft.slice(0, turn.spokenEnd)}{#if turn.draft.length > turn.spokenEnd}<span
							class="ahead">{turn.draft.slice(turn.spokenEnd)}</span
						>{/if}
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
	/* written by the model and not spoken: ahead of the voice, or never said */
	.ahead {
		color: var(--error);
	}
</style>
