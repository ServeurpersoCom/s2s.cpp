// convert to number, undefined if empty/NaN
export function num(v: unknown): number | undefined {
	if (v == null || v === '') return undefined;
	const n = Number(v);
	return isNaN(n) ? undefined : n;
}

// placeholder text for a server default
export function ph(v: unknown): string {
	return v != null ? String(v) : '';
}
