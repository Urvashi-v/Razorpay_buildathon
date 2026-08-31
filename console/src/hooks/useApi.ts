/**
 * The one way this console fetches data.
 *
 * WHY A HOOK RATHER THAN FETCHING IN COMPONENTS
 * =============================================
 * Three states have to be distinguishable everywhere: loading, failed, and
 * loaded. A component that tracks them itself will eventually render `data ??
 * fallback` while a request is in flight, and a fallback is a number the console
 * invented. `AsyncState` is a discriminated union, so `data` does not exist
 * until the request succeeded and there is nothing to default.
 *
 * Requests abort on unmount and on dependency change, so a slow assessment that
 * resolves after the user has navigated away cannot write into a dead component
 * or overwrite a newer result.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError } from '@/api/client';

export type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; error: ApiError }
  | { status: 'success'; data: T };

/** Coerce any thrown value into an ApiError, so callers only handle one type. */
function toApiError(cause: unknown): ApiError {
  if (cause instanceof ApiError) {
    return cause;
  }
  return new ApiError('INTERNAL_ERROR', cause instanceof Error ? cause.message : String(cause), 0, null);
}

/**
 * Run `fetcher` whenever `deps` change.
 *
 * `enabled: false` holds the hook in `idle` - used where a request should not
 * fire until the user has chosen something, so the screen shows an empty state
 * rather than an error about a missing parameter.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  options: { enabled?: boolean } = {},
): AsyncState<T> & { reload: () => void } {
  const enabled = options.enabled ?? true;
  const [state, setState] = useState<AsyncState<T>>({ status: enabled ? 'loading' : 'idle' });
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) {
      setState({ status: 'idle' });
      return;
    }

    const controller = new AbortController();
    let live = true;
    setState({ status: 'loading' });

    fetcherRef
      .current(controller.signal)
      .then((data) => {
        if (live) {
          setState({ status: 'success', data });
        }
      })
      .catch((cause: unknown) => {
        // An abort is not a failure: it means the caller moved on.
        if (live && !controller.signal.aborted) {
          setState({ status: 'error', error: toApiError(cause) });
        }
      });

    return () => {
      live = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return { ...state, reload };
}

/**
 * An action the user triggers, rather than something that loads on mount.
 *
 * Used by the agent panel and the simulator: nothing runs until a button is
 * pressed, and the result replaces the previous one only when it arrives.
 */
export function useAction<TArgs extends unknown[], TResult>(
  action: (...args: [...TArgs, AbortSignal]) => Promise<TResult>,
): {
  state: AsyncState<TResult>;
  run: (...args: TArgs) => Promise<void>;
  reset: () => void;
} {
  const [state, setState] = useState<AsyncState<TResult>>({ status: 'idle' });
  const controllerRef = useRef<AbortController | null>(null);
  const actionRef = useRef(action);
  actionRef.current = action;

  useEffect(
    () => () => {
      controllerRef.current?.abort();
    },
    [],
  );

  const run = useCallback(async (...args: TArgs) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ status: 'loading' });

    try {
      const data = await actionRef.current(...args, controller.signal);
      if (!controller.signal.aborted) {
        setState({ status: 'success', data });
      }
    } catch (cause: unknown) {
      if (!controller.signal.aborted) {
        setState({ status: 'error', error: toApiError(cause) });
      }
    }
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    setState({ status: 'idle' });
  }, []);

  return { state, run, reset };
}
