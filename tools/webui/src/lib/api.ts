import { FETCH_TIMEOUT_MS } from './config.js';
import type { S2SProps } from './types.js';

// relative to the page, so any reverse proxy prefix just works. An absolute
// server URL in the settings points the call somewhere else.
function url(path: string, serverUrl: string): string {
	return serverUrl ? new URL(path, serverUrl.replace(/\/?$/, '/')).toString() : path;
}

// GET /props: version, audio rate and the defaults every empty field falls
// back to.
export async function props(serverUrl: string): Promise<S2SProps> {
	const res = await fetch(url('props', serverUrl), {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	return res.json();
}

// POST /v1/models: the model list, proxied by s2s-server. The browser never
// calls the language model endpoint itself: no CORS to negotiate, an endpoint
// on a loopback address stays reachable, and the API key stays on the server.
export async function fetchModels(
	serverUrl: string,
	llmUrl: string,
	apiKey: string
): Promise<string[]> {
	const res = await fetch(url('v1/models', serverUrl), {
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
// server stages. Fire and forget: a failed log must never break a session.
export function postLog(serverUrl: string, line: string) {
	fetch(url('log', serverUrl), { method: 'POST', body: line }).catch(() => {});
}
