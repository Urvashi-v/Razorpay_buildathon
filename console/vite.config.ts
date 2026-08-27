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
