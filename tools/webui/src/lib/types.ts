// Server defaults, the single source of truth behind every empty field.
export interface S2SDefaults {
	mode: string;
	llm_fixed: boolean; // the server owns the endpoint, the page names none
	instructions: string;
	voice: string;
	language: string;
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
	reasoning_effort: string;
	vad_neg_threshold: number;
	vad_threshold: number;
	min_speech_ms: number;
	barge_in_ms: number;
	min_silence_ms: number;
	min_speech_continuation_ms: number;
	speech_pad_ms: number;
	turn_threshold: number;
	turn_max_wait_ms: number;
	reopen_grace_ms: number;
}

export interface S2SModels {
	vad: string;
	turn: string;
	asr: string;
	talker: string;
	codec: string;
}

export interface S2SProps {
	version: string;
	models: S2SModels;
	sample_rate: number;
	defaults: S2SDefaults;
}
