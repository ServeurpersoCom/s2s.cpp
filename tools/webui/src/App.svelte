<script lang="ts">
	import { Volume2 } from '@lucide/svelte';
	import { settings } from './lib/state.svelte.js';
	import ConfigPanel from './components/ConfigPanel.svelte';
	import VoicePanel from './components/VoicePanel.svelte';
	import Toast from './components/Toast.svelte';

	// sync dark/light class on <html> so CSS variables switch
	$effect(() => {
		document.documentElement.classList.toggle('dark', settings.dark);
		document.documentElement.classList.toggle('light', !settings.dark);
	});
</script>

<div class="s2s-app">
	<header>
		<span class="header-label">s2s.cpp</span>
		<span class="header-version">{__S2S_VERSION__}</span>
		<div class="spacer"></div>
		<label class="dark-toggle">
			<input type="checkbox" bind:checked={settings.dark} /> Dark
		</label>
		<div class="volume">
			<Volume2 size={14} />
			<input type="range" min="0" max="1" step="0.01" bind:value={settings.volume} />
		</div>
	</header>

	<main>
		<section class="panel config-panel">
			<ConfigPanel />
		</section>
		<section class="panel voice-panel">
			<VoicePanel />
		</section>
	</main>
</div>

<Toast />

<style>
	:global(:root) {
		--bg: #1a1a1a;
		--bg-input: #2a2a2a;
		--bg-card: #242424;
		--bg-btn: #333;
		--bg-btn-hover: #444;
		--fg: #eee;
		--fg-dim: #999;
		--border: #3a3a3a;
		--focus: #2ed573;
		--error: #c0392b;
		--ok: #27ae60;
		--who-user: #0e7490;
		--who-assistant: #b45309;
		color-scheme: dark;
	}
	:global(:root.light) {
		--bg: #f5f5f5;
		--bg-input: #fff;
		--bg-card: #fff;
		--bg-btn: #e0e0e0;
		--bg-btn-hover: #d0d0d0;
		--fg: #000;
		--fg-dim: #666;
		--border: #ccc;
		--focus: #27ae60;
		--error: #c0392b;
		--ok: #27ae60;
		--who-user: #0e7490;
		--who-assistant: #b45309;
		color-scheme: light;
	}
	:global(*, *::before, *::after) {
		box-sizing: border-box;
		margin: 0;
	}
	:global(body) {
		font-family:
			system-ui,
			-apple-system,
			sans-serif;
		background: var(--bg);
		color: var(--fg);
		min-height: 100dvh;
	}
	.s2s-app {
		display: flex;
		flex-direction: column;
		min-height: 100dvh;
	}
	header {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		padding: 1rem 1rem;
	}
	.header-label {
		font-size: 1.1rem;
		font-weight: 600;
		color: var(--fg);
	}
	.header-version {
		font-size: 0.7rem;
		color: var(--fg-dim);
		align-self: flex-end;
	}
	.spacer {
		flex: 1;
	}
	.dark-toggle {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.75rem;
		color: var(--fg);
		cursor: pointer;
	}
	.volume {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		color: var(--fg);
	}
	.volume input[type='range'] {
		width: 80px;
		cursor: pointer;
	}
	main {
		flex: 1;
		display: flex;
		gap: 1rem;
		padding: 0 1rem 1rem;
		background: var(--bg);
		overflow: hidden;
	}
	.panel {
		background: var(--bg);
		overflow-y: auto;
	}
	.config-panel {
		width: 400px;
		flex-shrink: 0;
	}
	.voice-panel {
		flex: 1;
	}
	@media (max-width: 800px) {
		main {
			flex-direction: column;
		}
		.config-panel {
			width: 100%;
		}
	}
</style>
