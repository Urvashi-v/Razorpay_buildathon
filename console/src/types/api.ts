/**
 * TypeScript mirrors of the backend contracts in `rto_sentinel.contracts`.
 *
 * These are hand-maintained rather than generated, because the console consumes
 * a deliberately small slice of the API and a generator would pull in the whole
 * OpenAPI surface plus a build step. The tradeoff is that they can drift - so
 * `src/api/client.test.ts` checks the shapes against the live schema when the
 * backend is reachable, and `npm run build` is not the only line of defence.
 *
 * WHAT IS NOT HERE, AND WHY
 * -------------------------
 * There is no type for "a risk score on its own". `ScoreResponse` always carries
 * the threshold that interpreted the probability, because a bare score invites a
 * comparison against 0.5 - which is the exact error this system exists to
 * correct. Keeping them welded together in the type makes that mistake
 * unavailable to a component author.
 */

export type RiskBand = 'LOW' | 'ELEVATED' | 'HIGH' | 'SEVERE';

export type InterventionAction =
  | 'none'
  | 'prepaid_nudge'
  | 'confirmation_required'
  | 'prepaid_only';

export type PaymentMethod = 'cod' | 'prepaid';

export type PincodeTier = 'tier_1' | 'tier_2' | 'tier_3' | 'unknown';

/** Machine-readable failure reasons. See `rto_sentinel.api.errors.ErrorCode`. */
export type ErrorCode =
  | 'VALIDATION_FAILED'
  | 'ORDER_NOT_FOUND'
  | 'DECISION_NOT_FOUND'
  | 'MODEL_UNAVAILABLE'
  | 'UNCALIBRATED_SCORE'
  | 'INVALID_COST_INPUTS'
  | 'AGENT_UNAVAILABLE'
  | 'GROUNDING_REJECTED'
  | 'NOT_IMPLEMENTED'
  | 'INTERNAL_ERROR';

export interface ApiErrorBody {
  code: ErrorCode;
  message: string;
  detail?: Record<string, unknown> | null;
}

export interface ApiErrorEnvelope {
  error: ApiErrorBody;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: 'ok';
  version: string;
  environment: string;
}

export interface ComponentStatus {
  ready: boolean;
  detail: string;
}

export interface ReadinessResponse {
  ready: boolean;
  version: string;
  environment: string;
  config_fingerprint: string | null;
  components: Record<string, ComponentStatus>;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Economics - the sliders
// ---------------------------------------------------------------------------

/** The four merchant inputs that derive the operating threshold. */
export interface CostInputs {
  rto_cost_inr: number;
  contribution_margin_inr: number;
  abandonment_on_friction: number;
  intervention_success_rate: number;
  friction_support_cost_inr: number;
}

/**
 * The derived threshold with its arithmetic attached.
 *
 * The intermediate terms travel with the result so the console can show the
 * working. A threshold that arrives without its derivation is a magic constant.
 */
export interface ThresholdDerivation {
  threshold: number;
  cost_false_positive_inr: number;
  saving_true_positive_inr: number;
  inputs: CostInputs;
  formula: string;
}

export interface WhatIfResponse {
  threshold: number;
  flag_rate: number;
  total_false_positive_cost_inr: number;
  net_inr_saved_per_1000_orders: number;
  n_orders: number;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

export interface FeatureContribution {
  feature: string;
  family: string;
  value: number | string | boolean | null;
  contribution: number;
}

export interface ScoreResponse {
  order_id: string;
  probability: number;
  threshold: number;
  band: RiskBand;
  action: InterventionAction;
  flagged: boolean;
  reason_codes: string[];
  expected_value_inr: number;
  appeal_available: boolean;
  human_review_required: boolean;
  contributions: FeatureContribution[];
  model_name: string;
  model_version: string;
  engine_version: string;
  scored_at: string;
  latency_ms: number | null;
  data_provenance: string;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/** Never a bare number: every estimate carries its bootstrap interval. */
export interface PointEstimate {
  value: number;
  ci_low: number;
  ci_high: number;
  confidence: number;
  n_bootstrap: number;
}

export interface CalibrationMetrics {
  expected_calibration_error: number;
  brier_score: number;
  n_bins: number;
  /** [mean predicted, observed frequency, count] per bin. */
  reliability_bins: Array<[number, number, number]>;
}

export interface EconomicResult {
  threshold: number;
  flag_rate: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  true_negatives: number;
  gross_saving_inr: number;
  /** Displayed on its own line. Never folded into the net figure. */
  total_false_positive_cost_inr: number;
  residual_false_negative_loss_inr: number;
  net_inr_saved_per_1000_orders: PointEstimate;
  baseline_net_inr_per_1000_orders: number;
}

export interface CohortResult {
  cohort: string;
  group: string;
  n_orders: number;
  flag_rate: number;
  precision: number | null;
  recall: number | null;
  net_inr_per_1000: number | null;
}

export interface FairnessAudit {
  slices: CohortResult[];
  max_flag_rate_ratio: number;
  worst_precision_drop: number;
  triggered: boolean;
  narrative: string;
}

export interface RankingMetrics {
  pr_auc: PointEstimate;
  roc_auc: PointEstimate;
  recall_at_precision_80: number | null;
  recall_at_precision_90: number | null;
  precision_at_k: Record<string, number>;
}

export interface EvaluationReport {
  model_name: string;
  model_version: string;
  rung_id: number;
  split: string;
  n_orders: number;
  evaluated_at: string;
  config_fingerprint: string;
  ranking: RankingMetrics;
  calibration: CalibrationMetrics;
  economics: EconomicResult;
  cohorts: CohortResult[];
  fairness: FairnessAudit | null;
  data_provenance: string;
}

// ---------------------------------------------------------------------------
// Explanations - everything here can legitimately be absent
// ---------------------------------------------------------------------------

export interface Explanation {
  order_id: string;
  sentence: string;
  reason_codes: Array<{
    code: string;
    feature: string;
    family: string;
    contribution: number;
    direction: string;
  }>;
  permitted_features: string[];
  generated_at: string;
  llm_model: string;
  /** False when the grounding validator rejected the generation. */
  grounded: boolean;
  rejection_reason: string | null;
}
