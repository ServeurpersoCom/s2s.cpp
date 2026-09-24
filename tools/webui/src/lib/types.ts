// Server defaults, the single source of truth behind every empty field.
interface S2SDefaults {
	mode: string;
	llm_fixed: boolean; // the server owns the endpoint, the page names none
	mcp_fixed: boolean; // the server owns the MCP servers, the page names none
	instructions: string;
	voice: string;
	language: string;
	tts_effect: string;
	tts_effects: string[];
	tts_voices: string[];
	tts_languages: string[];
	tts_temperature: number;
	tts_top_k: number;
	tts_top_p: number;
	tts_repetition_penalty: number;
	tts_subtalker_temperature: number;
	tts_subtalker_top_k: number;
	tts_subtalker_top_p: number;
	tts_max_new_tokens: number;
	tts_min_chars: number;
	tts_chars_per_second: number;
	tts_margin_seconds: number;
	llm_timeout_sec: number;
	max_rounds: number;
	tool_timeout_sec: number;
	reasoning_effort: string;
	vad_neg_threshold: number;
	vad_threshold: number;
	min_speech_ms: number;
	barge_in_ms: number;
	min_silence_ms: number;
	min_speech_continuation_ms: number;
	speech_pad_ms: number;
	turn_threshold: number;
	incomplete_delay_ms: number;
	turn_max_wait_ms: number;
	reopen_grace_ms: number;
}

// The files really loaded, one per model.
interface S2SModels {
	vad: string;
	turn: string;
	asr: string;
	talker: string;
	codec: string;
	aec: string;
}

// What the page reads of /props.
export interface S2SProps {
	models: S2SModels;
	defaults: S2SDefaults;
}
