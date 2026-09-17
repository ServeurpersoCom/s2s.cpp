// UI constants

export const FETCH_TIMEOUT_MS = 2000;
export const SSE_RECONNECT_MS = 2000;
export const LOG_MAX_LINES = 50;

// Pipeline modes. Loopback skips the language model and speaks the recognized
// text back, which exercises the microphone, the turn detection, the
// recognizer and the voice without an endpoint in the path.
export const MODES = ['loopback', 'conversation'] as const;
export type Mode = (typeof MODES)[number];
