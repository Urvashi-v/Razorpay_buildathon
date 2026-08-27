/**
 * The single HTTP client for the console.
 *
 * Every request to the backend goes through `request()`. Two reasons that
 * matters here:
 *
 *  1. **Errors arrive in one shape.** The backend wraps every failure in
 *     `{ error: { code, message, detail } }`, and `ApiError` preserves the
 *     `code`. A component can then distinguish `MODEL_UNAVAILABLE` (the system
 *     cannot score; show nothing) from `AGENT_UNAVAILABLE` (the sentence is
 *     missing; show the reason codes) - which a bare status code cannot express.
 *
 *  2. **No component fetches on its own.** That keeps base-URL handling, error
 *     translation and timeouts in one place, and it means there is nowhere for a
 *     component to quietly hardcode a fallback number when a call fails.
 *
 * NOTHING IN THE CONSOLE COMPUTES A METRIC. Every figure displayed is read from
 * a response. A chart that can invent its own numbers is a picture, not a report.
 */

import type { ApiErrorEnvelope, ErrorCode } from '@/types/api';

/**
 * Base URL for every request.
 *
 * Defaults to `/api`, which the Vite dev server proxies to the backend and
 * rewrites back to the root. That keeps development same-origin, so the console
 * exercises relative URLs exactly as it will in production and a CORS problem
 * surfaces in a deployment check rather than only after deploying.
 *
 * Set `VITE_API_BASE_URL` for a production build pointing at a separate host.
 * Never put a secret behind a VITE_ prefix - it is compiled into the bundle.
 */
const BASE_URL: string = import.meta.env['VITE_API_BASE_URL'] ?? '/api';

const DEFAULT_TIMEOUT_MS = 15_000;

/** A structured failure from the API, carrying the backend's error code. */
export class ApiError extends Error {
  readonly code: ErrorCode;
  readonly status: number;
  readonly detail: Record<string, unknown> | null;

  constructor(code: ErrorCode, message: string, status: number, detail: Record<string, unknown> | null) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.detail = detail;
  }

  /**
   * True when the system genuinely cannot do the work, as opposed to being
   * degraded. Callers use this to decide between hiding a panel and showing a
   * reduced one.
   */
  get isHardFailure(): boolean {
    return this.code === 'MODEL_UNAVAILABLE' || this.code === 'UNCALIBRATED_SCORE';
  }

  /** True when the language layer is off. The decision itself is unaffected. */
  get isDegradedExplanation(): boolean {
    return this.code === 'AGENT_UNAVAILABLE' || this.code === 'GROUNDING_REJECTED';
  }
}

interface RequestOptions {
  method?: 'GET' | 'POST';
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
  /**
   * Non-2xx statuses that still carry a usable body.
   *
   * `/readiness` is the case this exists for: a not-ready instance answers 503
   * with a full report of *which* component is down, and that report is far
   * more useful to an operator than an error banner. Any other endpoint leaves
   * this empty and a non-2xx is a failure.
   */
  acceptStatuses?: readonly number[];
}

function isErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (typeof value !== 'object' || value === null || !('error' in value)) {
    return false;
  }
  const inner = (value as { error: unknown }).error;
  return typeof inner === 'object' && inner !== null && 'code' in inner && 'message' in inner;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const {
    method = 'GET',
    body,
    signal,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    acceptStatuses,
  } = options;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  if (signal) {
    signal.addEventListener('abort', () => controller.abort(), { once: true });
  }

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? null : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (cause) {
    clearTimeout(timeout);
    // A network failure is not a backend error code, so it gets INTERNAL_ERROR
    // rather than being silently mapped onto something more specific.
    throw new ApiError('INTERNAL_ERROR', `Could not reach the API at ${path}.`, 0, {
      cause: String(cause),
    });
  }
  clearTimeout(timeout);

  const text = await response.text();
  const payload: unknown = text.length > 0 ? JSON.parse(text) : null;

  if (!response.ok && !(acceptStatuses ?? []).includes(response.status)) {
    if (isErrorEnvelope(payload)) {
      throw new ApiError(
        payload.error.code,
        payload.error.message,
        response.status,
        payload.error.detail ?? null,
      );
    }
    throw new ApiError('INTERNAL_ERROR', `Request to ${path} failed.`, response.status, null);
  }

  return payload as T;
}

export const get = <T>(path: string, signal?: AbortSignal): Promise<T> =>
  signal ? request<T>(path, { signal }) : request<T>(path);

export const post = <T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> =>
  signal ? request<T>(path, { method: 'POST', body, signal }) : request<T>(path, { method: 'POST', body });
