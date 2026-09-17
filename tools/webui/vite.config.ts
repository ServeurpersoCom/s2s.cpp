import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { viteSingleFile } from 'vite-plugin-singlefile';
import { execSync } from 'child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { gzipSync } from 'zlib';
import { resolve } from 'path';

// git version baked at build time (same format as C++ S2S_VERSION)
function gitVersion(): string {
	try {
		const hash = execSync('git rev-parse --short HEAD', { cwd: resolve(__dirname, '../..') })
			.toString()
			.trim();
		const date = execSync('git show -s --format=%cs HEAD', { cwd: resolve(__dirname, '../..') })
			.toString()
			.trim();
		return `${hash} (${date})`;
	} catch {
		return 'unknown';
	}
}

// deterministic gzip of the inlined index.html into ../public/index.html.gz.
// the .gz is committed to git so the C++ build works without npm.
// gzip timestamp and OS bytes are zeroed for reproducible output.
function s2sGzipPlugin() {
	return {
		name: 's2s:gzip',
		apply: 'build' as const,
		closeBundle() {
			const indexPath = resolve(__dirname, 'dist', 'index.html');
			const publicDir = resolve(__dirname, '..', 'public');
			const gzPath = resolve(publicDir, 'index.html.gz');

			const html = readFileSync(indexPath, 'utf-8');

			const compressed = gzipSync(Buffer.from(html, 'utf-8'), { level: 9 });

			// zero gzip header fields that vary between builds
			compressed[4] = 0; // mtime
			compressed[5] = 0;
			compressed[6] = 0;
			compressed[7] = 0;
			compressed[9] = 0; // OS

			mkdirSync(publicDir, { recursive: true });
			writeFileSync(gzPath, compressed);

			console.log(
				`  index.html: ${html.length} bytes -> index.html.gz: ${compressed.length} bytes`
			);
		}
	};
}

// One target for the dev proxy, and the routes s2s-server owns.
const DEV_SERVER = 'http://localhost:8088';
const DEV_ROUTES = ['/health', '/props', '/logs', '/log', '/v1/models', '/s2s.js'];

export default defineConfig({
	plugins: [svelte(), viteSingleFile(), s2sGzipPlugin()],

	define: {
		__S2S_VERSION__: JSON.stringify(gitVersion())
	},

	// dev server: everything the page asks for comes from a running
	// s2s-server, the published module included, since the page imports it at
	// runtime rather than bundling it.
	server: {
		proxy: {
			...Object.fromEntries(DEV_ROUTES.map((route) => [route, DEV_SERVER])),
			'/v1/realtime': { target: DEV_SERVER.replace('http', 'ws'), ws: true }
		}
	},

	build: {
		assetsInlineLimit: Infinity,
		cssCodeSplit: false
	}
});
