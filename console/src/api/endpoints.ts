/**
 * Typed wrappers for each backend endpoint the console uses.
 *
 * One function per endpoint, so a component names what it wants rather than
 * assembling a URL. When an endpoint changes, exactly one place here changes.
 *
 * There are no fixtures in this file and no fallbacks. Every function either
 * returns what the backend sent or throws `ApiError` with the backend's own
 * code, which the UI renders as the failure it is.
 */

import { get, post, request } from '@/api/client';
import type {
  AgentStatusResponse,
  CostInputs,
  DataStatusResponse,
  DecisionStatusResponse,
  FinalModelResponse,
  HealthResponse,
  InvestigationResponse,
  LadderResponse,
  ModelStatusResponse,
  OrderPageResponse,
  OrderSummary,
  ProfilesResponse,
  ReadinessResponse,
  RiskAssessmentResponse,
  SimulationResult,
  ThresholdDerivation,
  ToolCatalogueEntry,
  DriftReport,
  FairnessResponse,
  ShiftStudy,
} from '@/types/api';

/** Build a query string, dropping anything unset. */
type QueryValue = string | number | boolean | undefined;

function query(params: Readonly<Record<string, QueryValue>>): string {
  const parts = Object.entries(params)
    .filter(([, value]) => value !== undefined && value !== '')
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
  return parts.length > 0 ? `?${parts.join('&')}` : '';
}

// --- health ------------------------------------------------------------------

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

// --- orders ------------------------------------------------------------------

export interface OrderFilters {
  [key: string]: QueryValue;
  merchant_id?: string;
  customer_hash?: string;
  split?: string;
  payment_method?: string;
  dataset_run?: string;
  limit?: number;
  offset?: number;
}

export const fetchOrders = (
  filters: OrderFilters = {},
  signal?: AbortSignal,
): Promise<OrderPageResponse> => get<OrderPageResponse>(`/v1/orders${query(filters)}`, signal);

export const fetchOrder = (
  orderId: string,
  datasetRun?: string,
  signal?: AbortSignal,
): Promise<OrderSummary> =>
  get<OrderSummary>(
    `/v1/orders/${encodeURIComponent(orderId)}${query({ dataset_run: datasetRun })}`,
    signal,
  );

/**
 * The full assessment for one order.
 *
 * This is the call that runs the entire chain server-side: database row ->
 * feature pipeline -> trained model -> calibrator -> decision engine. It is slow
 * on a cold start because the artefact loads on first use, which is why the UI
 * shows a real loading state rather than a skeleton that implies instant data.
 */
export const fetchRiskAssessment = (
  orderId: string,
  options: { datasetRun?: string; includeContributions?: boolean } = {},
  signal?: AbortSignal,
): Promise<RiskAssessmentResponse> =>
  get<RiskAssessmentResponse>(
    `/v1/orders/${encodeURIComponent(orderId)}/risk${query({
      dataset_run: options.datasetRun,
      include_contributions: options.includeContributions ?? true,
    })}`,
    signal,
  );

// --- economics ---------------------------------------------------------------

export const fetchCostProfiles = (signal?: AbortSignal): Promise<ProfilesResponse> =>
  get<ProfilesResponse>('/v1/economics/profiles', signal);

export const deriveThreshold = (
  inputs: CostInputs,
  signal?: AbortSignal,
): Promise<ThresholdDerivation> =>
  post<ThresholdDerivation>('/v1/economics/threshold', inputs, signal);

/**
 * Recompute the whole policy under new merchant economics.
 *
 * The threshold, every band boundary, the assignment of each order to a band and
 * the rupee totals are all recalculated on the server. The console sends inputs
 * and renders what comes back; it does not scale a cached total, and there is no
 * economic arithmetic anywhere in this codebase.
 */
export const simulateEconomics = (
  inputs: CostInputs,
  compareToProfile?: string,
  signal?: AbortSignal,
): Promise<SimulationResult> =>
  post<SimulationResult>(
    '/v1/economics/simulate',
    {
      cost_inputs: inputs,
      split: 'validation',
      compare_to_profile: compareToProfile ?? null,
    },
    signal,
  );

// --- evaluation --------------------------------------------------------------

export const fetchLadder = (split = 'validation', signal?: AbortSignal): Promise<LadderResponse> =>
  get<LadderResponse>(`/v1/evaluation/ladder${query({ split })}`, signal);

export const fetchFinalModel = (
  split: 'validation' | 'test',
  signal?: AbortSignal,
): Promise<FinalModelResponse> =>
  get<FinalModelResponse>(`/v1/evaluation/final${query({ split })}`, signal);

// --- monitoring --------------------------------------------------------------

export const fetchModelStatus = (signal?: AbortSignal): Promise<ModelStatusResponse> =>
  get<ModelStatusResponse>('/v1/monitoring/model', signal);

export const fetchDataStatus = (
  datasetRun?: string,
  signal?: AbortSignal,
): Promise<DataStatusResponse> =>
  get<DataStatusResponse>(`/v1/monitoring/data${query({ dataset_run: datasetRun })}`, signal);

export const fetchDecisionStatus = (
  merchantId?: string,
  signal?: AbortSignal,
): Promise<DecisionStatusResponse> =>
  get<DecisionStatusResponse>(
    `/v1/monitoring/decisions${query({ merchant_id: merchantId })}`,
    signal,
  );

// --- agents ------------------------------------------------------------------

export const fetchAgentStatus = (signal?: AbortSignal): Promise<AgentStatusResponse> =>
  get<AgentStatusResponse>('/v1/explanations/status', signal);

export const fetchAgentTools = (signal?: AbortSignal): Promise<ToolCatalogueEntry[]> =>
  get<ToolCatalogueEntry[]>('/v1/explanations/tools', signal);

/**
 * Ask the investigation agent about one order.
 *
 * Throws `ApiError('AGENT_UNAVAILABLE')` when the language layer is off. The UI
 * shows that reason. It does not fall back to a canned explanation - the reason
 * codes are already on the risk screen and are the artefact of record.
 */
export const investigateOrder = (
  orderId: string,
  question: string,
  datasetRun?: string,
  signal?: AbortSignal,
): Promise<InvestigationResponse> =>
  post<InvestigationResponse>(
    `/v1/explanations/${encodeURIComponent(orderId)}/investigate${query({
      question,
      dataset_run: datasetRun,
    })}`,
    undefined,
    signal,
  );

// --- responsible AI ----------------------------------------------------------

/**
 * The cohort and disparate-impact audit.
 *
 * Throws `ApiError('NOT_IMPLEMENTED')` when the audit has not been run. The UI
 * shows that reason rather than an empty table, because an empty fairness table
 * reads as "we checked and found nothing".
 */
export const fetchFairness = (
  split: 'validation' | 'test' = 'validation',
  signal?: AbortSignal,
): Promise<FairnessResponse> =>
  get<FairnessResponse>(`/v1/evaluation/fairness${query({ split })}`, signal);

export const fetchShiftStudy = (signal?: AbortSignal): Promise<ShiftStudy> =>
  get<ShiftStudy>('/v1/evaluation/shift', signal);

export const fetchDriftReport = (signal?: AbortSignal): Promise<DriftReport> =>
  get<DriftReport>('/v1/monitoring/drift', signal);
