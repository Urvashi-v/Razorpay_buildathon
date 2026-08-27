import '@testing-library/jest-dom/vitest';

/**
 * Vitest setup.
 *
 * `fetch` is deliberately NOT polyfilled with a live implementation here. Tests
 * stub it explicitly per case, so a test that accidentally makes a real network
 * call fails loudly instead of silently depending on a running backend.
 */
