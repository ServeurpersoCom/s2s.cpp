<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { applySettings, applyVolume, createVoice, destroyVoice } from '../lib/voice.svelte.js';
	import ChatCard from './ChatCard.svelte';
	import LogCard from './LogCard.svelte';

	// The component itself draws nothing: this panel only owns its lifetime.
	// It arrives from ./s2s.js, the same file an integrator imports.
	onMount(createVoice);

	// Settings edited on the left reach a running session live: applySettings
	// reads the store, so the effect tracks exactly the fields it sends.
	$effect(applySettings);

	$effect(applyVolume);

	onDestroy(destroyVoice);
</script>

<div class="voice">
	<ChatCard />
	<LogCard />
</div>

<style>
	.voice {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		overflow-y: auto;
	}
</style>
