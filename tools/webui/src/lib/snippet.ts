import { toOptions } from './voice.svelte.js';

// Turns the current settings into the snippet an integrator pastes on their
// own page. Only the fields that were actually set appear: the rest is left
// to the server defaults, so the snippet stays as small as the intent behind
// it. That is also the honest test of the component API, since the snippet is
// nothing more than its options, written out.

function literal(value: unknown, indent: string): string {
	if (typeof value === 'string') {
		return `'${value.replace(/'/g, "\\'")}'`;
	}
	if (typeof value === 'number' || typeof value === 'boolean') {
		return String(value);
	}
	return object(value as Record<string, unknown>, indent);
}

// Drops empty strings, undefined values and objects left empty after that.
function prune(value: unknown): unknown {
	if (value === undefined || value === '' || value === null) {
		return undefined;
	}
	if (typeof value !== 'object') {
		return value;
	}

	const out: Record<string, unknown> = {};
	for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
		const kept = prune(nested);
		if (kept !== undefined) {
			out[key] = kept;
		}
	}
	return Object.keys(out).length > 0 ? out : undefined;
}

function object(value: Record<string, unknown>, indent: string): string {
	const inner = indent + '\t';
	const lines = Object.entries(value).map(
		([key, nested]) => `${inner}${key}: ${literal(nested, inner)}`
	);
	return `{\n${lines.join(',\n')}\n${indent}}`;
}

export function snippet(): string {
	const options = (prune(toOptions()) ?? {}) as Record<string, unknown>;

	// Everything stays relative to the page the snippet lands on: the module
	// comes from the s2s-server that serves it, and the component resolves
	// v1/realtime the same way. An integrator who hosts the page elsewhere
	// fills the Realtime URL field, and only then does an absolute URL appear
	// here. Pasting a foreign host would otherwise leech somebody else's
	// server, or point at a 127.0.0.1 that only exists on the developer's
	// machine.
	const body = `const s2s = new S2S(${object(options, '\t')});`;

	return `<button id="talk">Talk</button>

<script type="module">
	import { S2S } from './s2s.js';

	${body}

	s2s.on('state', (state) => console.log('state', state));
	s2s.on('user_text', (text) => console.log('user', text));
	s2s.on('assistant_text', (text) => console.log('assistant', text));
	s2s.on('error', (message) => console.error(message));

	// start() needs a user gesture and a secure context
	document.querySelector('#talk').onclick = () => s2s.start();
</script>`;
}
