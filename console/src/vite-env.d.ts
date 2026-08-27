/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL for the scoring API. Empty in development so the Vite proxy
   * handles `/api`. Never put a secret behind a VITE_ prefix: anything with
   * that prefix is compiled into the client bundle and is public.
   */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
