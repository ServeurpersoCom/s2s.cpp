import { FETCH_TIMEOUT_MS } from './config.js';
import type { S2SProps } from './types.js';

// Every call goes to the server that served the page, relative to it, so any
// reverse proxy prefix just works.

// GET /props: version, audio rate and the defaults every empty field falls
// back to.
export async function props(): Promise<S2SProps> {
	const res = await fetch('props', {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	return res.json();
}

// POST /v1/models: the model list, proxied by s2s-server. The browser never
// calls the language model endpoint itself: no CORS to negotiate, an endpoint
// on a loopback address stays reachable, and the API key stays on the server.
export async function fetchModels(llmUrl: string, apiKey: string): Promise<string[]> {
	const res = await fetch('v1/models', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ url: llmUrl, key: apiKey }),
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	const body = await res.json().catch(() => ({}));
	if (!res.ok) throw new Error(body?.error || `${res.status} ${res.statusText}`);

	const list = Array.isArray(body?.data) ? body.data : [];
	return list.map((item: { id?: string }) => item.id).filter((id: unknown): id is string => !!id);
}

// POST /log: browser side events, so they land in the same stream as the
// server stages. One request in flight at a time keeps the lines in order,
// and the timeout keeps a stalled request from holding the ones behind it.
// Fire and forget: a failed log must never break a session.
let logTail: Promise<void> = Promise.resolve();

export function postLog(line: string) {
	logTail = logTail.then(() =>
		fetch('log', {
			method: 'POST',
			body: line,
			signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
		}).then(
			() => {},
			() => {}
		)
	);
}
