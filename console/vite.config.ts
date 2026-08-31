import { fileURLToPath, URL } from 'node:url';

import react from '@vitejs/plugin-react';
// `defineConfig` from vitest/config rather than vite: it is the same function
// widened to accept the `test` block, so the app build and the test runner stay
// in one file with one set of path aliases.
import { defineConfig } from 'vitest/config';

/**
 * Vite configuration for the merchant console.
 *
 * The dev server proxies API calls to the FastAPI backend so that browser
 * requests are same-origin in development. That means the console exercises the
 * same relative URLs it will use in production, and CORS misconfiguration shows
 * up in a deployment check rather than only after deploying.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // Bound the proxy so a backend that accepts a connection and then
        // stalls surfaces as an error state rather than an indefinite spinner.
        //
        // This is a backstop, not a fix. A stalled backend is a backend bug and
        // belongs to the backend: the one time this actually happened, the
        // cause was libpq spending ~130s on a dead IPv6 address before falling
        // back to IPv4, and it was fixed there (see `CONNECT_TIMEOUT_SECONDS`
        // in `db/session.py`). What the console gets from a bound here is that
        // the next such bug shows the user an error instead of a spinner that
        // never resolves.
        timeout: 30_000,
        proxyTimeout: 30_000,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    css: false,
  },
});
