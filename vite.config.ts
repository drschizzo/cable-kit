// The examples, served and built by Vite; the tests (Vitest), in tests/.
//   npm run dev             → http://localhost:5173 (the list of examples)
//   npm run build:examples  → dist/
// The `source` condition reads the packages from their sources, not their dist/.
import { readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

const conditions = ['source', 'module', 'browser', 'development|production'];
const examples = readdirSync(resolve(__dirname, 'examples'), { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);

export default defineConfig({
  root: 'examples',
  base: './',
  resolve: { conditions },
  ssr: { resolve: { conditions: ['source'], externalConditions: ['source'] } },
  worker: { format: 'es' },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'es2022',
    chunkSizeWarningLimit: 1000, // three.js
    rollupOptions: {
      input: Object.fromEntries([['index', resolve(__dirname, 'examples/index.html')], ...examples.map((n) => [n, resolve(__dirname, `examples/${n}/index.html`)])]),
    },
  },
  test: {
    root: '.',
    include: ['tests/**/*.test.ts'],
  },
});
