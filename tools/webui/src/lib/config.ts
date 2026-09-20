// UI constants

export const FETCH_TIMEOUT_MS = 2000;
export const SSE_RECONNECT_MS = 2000;
export const LOG_MAX_LINES = 25;
export const NEWLINE = '\n';

// Pipeline modes. Loopback skips the language model and speaks the recognized
// text back, which exercises the microphone, the turn detection, the
// recognizer and the voice without an endpoint in the path.
export const MODES = ['loopback', 'conversation'] as const;
export type Mode = (typeof MODES)[number];

// What an endpoint URL looks like, for the field and the copied client code.
export const ENDPOINT_EXAMPLE = 'http://127.0.0.1:8080/v1';
