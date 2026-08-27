import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, get, post } from '@/api/client';

/**
 * The client preserves the backend's error codes.
 *
 * That matters more than it looks. `MODEL_UNAVAILABLE` and `AGENT_UNAVAILABLE`
 * are both 503 but mean opposite things operationally: the first means the
 * system cannot score and the UI must show nothing; the second means only the
 * sentence is missing and the decision is fine. A client that collapsed both
 * into "503" would force the UI to either over-react or under-react.
 */

function mockFetch(status: number, body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      text: async () => JSON.stringify(body),
    }),
  );
}

/**
 * Await a promise that must reject with an `ApiError`, and hand back the typed
 * error. Fails the test if the promise resolves - a silent success here would
 * mean the client swallowed a failure, which is the bug these tests exist for.
 */
async function expectApiError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error('expected the request to reject with an ApiError, but it resolved');
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('request', () => {
  it('returns the parsed body on success', async () => {
    mockFetch(200, { status: 'ok', version: '0.1.0', environment: 'test' });
    const result = await get<{ version: string }>('/health');
    expect(result.version).toBe('0.1.0');
  });

  it('preserves the error code from the shared envelope', async () => {
    mockFetch(501, {
      error: {
        code: 'NOT_IMPLEMENTED',
        message: 'Order scoring is not implemented yet; it lands in Phase 3.',
        detail: { phase: 'Phase 3' },
      },
    });

    await expect(post('/v1/score', {})).rejects.toThrow(ApiError);
    try {
      await post('/v1/score', {});
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const apiError = error as ApiError;
      expect(apiError.code).toBe('NOT_IMPLEMENTED');
      expect(apiError.status).toBe(501);
      expect(apiError.detail).toEqual({ phase: 'Phase 3' });
    }
  });

  it('distinguishes a missing model from a missing explanation', async () => {
    mockFetch(503, {
      error: { code: 'MODEL_UNAVAILABLE', message: 'No model artefact loaded.', detail: null },
    });
    const hard = await expectApiError(get('/v1/score'));
    expect(hard.isHardFailure).toBe(true);
    expect(hard.isDegradedExplanation).toBe(false);

    mockFetch(503, {
      error: { code: 'AGENT_UNAVAILABLE', message: 'Language layer is off.', detail: null },
    });
    const soft = await expectApiError(get('/v1/explanations/ORD-1'));
    expect(soft.isHardFailure).toBe(false);
    expect(soft.isDegradedExplanation).toBe(true);
  });

  it('reports an unreachable backend rather than inventing a result', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));
    const error = await expectApiError(get('/health'));
    expect(error.code).toBe('INTERNAL_ERROR');
    expect(error.status).toBe(0);
  });

  it('falls back to INTERNAL_ERROR when the body is not an error envelope', async () => {
    mockFetch(500, { unexpected: 'shape' });
    const error = await expectApiError(get('/health'));
    expect(error.code).toBe('INTERNAL_ERROR');
    expect(error.status).toBe(500);
  });
});
