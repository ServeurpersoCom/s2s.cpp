import { defineConfig } from 'vite';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { gzipSync } from 'zlib';
import { resolve } from 'path';

// Second build of the same sources: the client the page uses is also shipped
// as a standalone ES module, so an integrator can drop it on their own site
// and point it at any s2s-server. No Svelte runtime, no dependency.
//
// Output lands as ../public/s2s.js.gz, embedded into s2s-server next to the
// page and served on /s2s.js. The gzip is deterministic for the same reason
// the page is: the .gz is committed.
function s2sLibGzipPlugin() {
	return {
		name: 's2s:lib-gzip',
		apply: 'build' as const,
		closeBundle() {
			const source = resolve(__dirname, 'dist-lib', 's2s.js');
			const publicDir = resolve(__dirname, '..', 'public');
			const gzPath = resolve(publicDir, 's2s.js.gz');

			const code = readFileSync(source, 'utf-8');
			const compressed = gzipSync(Buffer.from(code, 'utf-8'), { level: 9 });

			// zero gzip header fields that vary between builds
			compressed[4] = 0; // mtime
			compressed[5] = 0;
			compressed[6] = 0;
			compressed[7] = 0;
			compressed[9] = 0; // OS

			mkdirSync(publicDir, { recursive: true });
			writeFileSync(gzPath, compressed);

			console.log(`  s2s.js: ${code.length} bytes -> s2s.js.gz: ${compressed.length} bytes`);
		}
	};
}

export default defineConfig({
	plugins: [s2sLibGzipPlugin()],

	build: {
		outDir: 'dist-lib',
		emptyOutDir: true,
		lib: {
			entry: resolve(__dirname, 'src/lib/s2s.ts'),
			formats: ['es'],
			fileName: () => 's2s.js'
		}
	}
});
