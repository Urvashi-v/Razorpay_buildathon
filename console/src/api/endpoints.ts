/**
 * Typed wrappers for each backend endpoint the console uses.
 *
 * One function per endpoint, so a component names what it wants rather than
 * assembling a URL. When an endpoint changes, exactly one place here changes.
 *
 * Endpoints marked "Phase N" currently return 501 from the backend. They are
 * declared now because the contract is fixed and published in OpenAPI, so the
 * console can be built against a real schema. They are NOT stubbed with fake
 * data here - a call to one of them throws `ApiError('NOT_IMPLEMENTED')`, which
 * the UI surfaces plainly.
 */

import { get, post, request } from '@/api/client';
import type {
  CostInputs,
  EvaluationReport,
  Explanation,
  FairnessAudit,
  HealthResponse,
  ReadinessResponse,
  ScoreResponse,
  ThresholdDerivation,
  WhatIfResponse,
} from '@/types/api';

// --- health (implemented) ----------------------------------------------------

export const fetchHealth = (signal?: AbortSignal): Promise<HealthResponse> =>
  get<HealthResponse>('/health', signal);

/**
 * Readiness.
 *
 * A not-ready instance answers 503 with a complete report of which component is
 * down, so 503 is accepted as a valid response here rather than thrown. The
 * console then renders the real state - including "no model loaded" - instead of
 * a generic connection error that hides what is actually wrong.
 */
export const fetchReadiness = (signal?: AbortSignal): Promise<ReadinessResponse> =>
  request<ReadinessResponse>('/readiness', {
    acceptStatuses: [503],
    ...(signal ? { signal } : {}),
  });

// --- economics (Phase 2) -----------------------------------------------------

export const deriveThreshold = (
  inputs: CostInputs,
  signal?: AbortSignal,
): Promise<ThresholdDerivation> => post<ThresholdDerivation>('/v1/economics/threshold', inputs, signal);

export const runWhatIf = (
  inputs: CostInputs,
  split = 'validation',
  signal?: AbortSignal,
): Promise<WhatIfResponse> =>
  post<WhatIfResponse>('/v1/economics/what-if', { cost_inputs: inputs, split }, signal);

// --- scoring (Phase 3) -------------------------------------------------------

export const scoreOrder = (
  order: unknown,
  costInputs?: CostInputs,
  signal?: AbortSignal,
): Promise<ScoreResponse> =>
  post<ScoreResponse>('/v1/score', { order, cost_inputs: costInputs ?? null }, signal);

// --- evaluation (Phase 4) ----------------------------------------------------

export const fetchLadder = (
  split = 'validation',
  signal?: AbortSignal,
): Promise<EvaluationReport[]> =>
  get<EvaluationReport[]>(`/v1/evaluation/ladder?split=${encodeURIComponent(split)}`, signal);

export const fetchFairness = (split = 'validation', signal?: AbortSignal): Promise<FairnessAudit> =>
  get<FairnessAudit>(`/v1/evaluation/fairness?split=${encodeURIComponent(split)}`, signal);

// --- explanations (Phase 5, always optional) ---------------------------------

export const fetchExplanation = (orderId: string, signal?: AbortSignal): Promise<Explanation> =>
  post<Explanation>(`/v1/explanations/${encodeURIComponent(orderId)}`, undefined, signal);
