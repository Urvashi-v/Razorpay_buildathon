/**
 * Response types, mirroring the backend's published OpenAPI schema.
 *
 * Hand-written rather than generated, because a generated file invites nobody to
 * read it, and the comments here are the point: several of these fields carry
 * meaning that a type alone does not express. `is_rto: boolean | null` is not a
 * tri-state for tidiness - null means the outcome has not matured, and rendering
 * it as "delivered" would be the single most effective way to make this console
 * lie.
 *
 * NOTHING HERE HAS A DEFAULT VALUE. A missing field surfaces as undefined and
 * the component shows an absence, because a default is a number the console
 * invented.
 */

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

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

export interface ApiErrorEnvelope {
  error: {
    code: ErrorCode;
    message: string;
    detail?: Record<string, unknown> | null;
  };
}

// ---------------------------------------------------------------------------
// health
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
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
  /** Keyed by component name: database, model, configuration, language layer. */
  components: Record<string, ComponentStatus>;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// orders
// ---------------------------------------------------------------------------

export interface OrderSummary {
  order_id: string;
  merchant_id: string;
  customer_hash: string;
  ordered_at: string;
  payment_method: string;
  is_cod: boolean;
  order_value_inr: number;
  discount_inr: number;
  item_count: number;
  category: string;
  courier_partner: string | null;
  split: string;
  dataset_run_id: string | null;
  /** Null when the outcome has not matured. NOT the same as false. */
  is_rto: boolean | null;
  outcome: string | null;
  resolved_at: string | null;
}

export interface OrderPageResponse {
  orders: OrderSummary[];
  total: number;
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------------------
// risk assessment
// ---------------------------------------------------------------------------

export type RiskBand = 'LOW' | 'ELEVATED' | 'HIGH' | 'SEVERE';
export type InterventionAction =
  | 'none'
  | 'prepaid_nudge'
  | 'confirmation_required'
  | 'prepaid_only';

export interface FeatureContribution {
  feature: string;
  family: string;
  value: number | string | boolean | null;
  contribution: number;
}

export interface ModelProvenance {
  model_name: string;
  model_version: string;
  calibration_method: string | null;
  calibration_fitted_on: string | null;
  feature_version: string;
  feature_fingerprint: string;
  dataset_run_id: string;
  generator_version: string;
  trained_at: string;
  training_rows: number;
  n_features: number;
  selection_manifest_id: string;
}

export interface FeatureProvenance {
  feature_version: string;
  feature_fingerprint: string;
  n_features: number;
  /** Features with no value for this order. Cold start, not an error. */
  null_features: string[];
  context_rows: number;
}

export interface EconomicAssumptions {
  cost_profile: string;
  rto_cost_inr: number;
  contribution_margin_inr: number;
  friction_support_cost_inr: number;
  /** ASSUMED. Never measured. The UI must label it as such. */
  abandonment_on_friction: number;
  /** ASSUMED. Never measured. The UI must label it as such. */
  intervention_success_rate: number;
  cost_false_positive_inr: number;
  saving_true_positive_inr: number;
  threshold_formula: string;
  band_intervention_success_rate: number;
  band_abandonment_rate: number;
}

export interface RiskAssessmentResponse {
  order: OrderSummary;
  probability: number;
  raw_score: number | null;
  threshold: number;
  band: RiskBand;
  action: InterventionAction;
  flagged: boolean;
  reason_codes: string[];
  expected_value_inr: number;
  appeal_available: boolean;
  human_review_required: boolean;
  is_control_holdout: boolean;
  contributions: FeatureContribution[];
  model: ModelProvenance;
  features: FeatureProvenance;
  economics: EconomicAssumptions;
  engine_version: string;
  scored_at: string;
  latency_ms: number | null;
  outcome_is_known: boolean;
  data_provenance: string;
}

// ---------------------------------------------------------------------------
// economics
// ---------------------------------------------------------------------------

export interface CostInputs {
  rto_cost_inr: number;
  contribution_margin_inr: number;
  abandonment_on_friction: number;
  intervention_success_rate: number;
  friction_support_cost_inr: number;
}

export interface CostProfileSummary {
  key: string;
  label: string;
  inputs: CostInputs;
}

export interface ProfilesResponse {
  default_profile: string;
  profiles: CostProfileSummary[];
  bounds: Record<string, { min: number; max: number }>;
  assumption_warning: string;
}

export interface ThresholdDerivation {
  threshold: number;
  cost_false_positive_inr: number;
  saving_true_positive_inr: number;
  inputs: CostInputs;
  formula: string;
}

export interface SimulatorLadderRung {
  band: string;
  action: string;
  lower_bound: number;
  upper_bound: number | null;
  n_orders: number;
  share_of_book: number;
  expected_net_inr: number;
  intervention_success_rate: number;
  abandonment_rate: number;
}

export interface BandOutcome {
  band: string;
  action: string;
  n_orders: number;
  share_of_book: number;
  expected_rto_orders: number;
  realized_rto_orders: number | null;
  expected_net_inr: number;
}

export interface PortfolioEconomics {
  threshold: number;
  threshold_source: string;
  cost_profile: string;
  split: string;
  n_orders: number;
  flag_rate: number;
  intervention_rate: number;
  expected_orders_affected: number;
  expected_savings_inr: number;
  expected_false_positive_cost_inr: number;
  expected_false_negative_loss_inr: number;
  expected_total_cost_inr: number;
  expected_net_inr_per_1000_orders: number;
  realized_net_inr_per_1000_orders: number | null;
  realized_precision: number | null;
  realized_recall: number | null;
  do_nothing_loss_inr_per_1000_orders: number;
  holdout_fraction_of_flagged: number;
  net_inr_per_1000_after_holdout: number;
  bands: BandOutcome[];
  collapsed_bands: string[];
}

export interface SimulationResult {
  threshold: ThresholdDerivation;
  ladder: SimulatorLadderRung[];
  collapsed_bands: string[];
  economics: PortfolioEconomics;
  baseline_threshold: number | null;
  baseline_net_inr_per_1000_orders: number | null;
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

export interface LadderRungResult {
  rung_id: number;
  model_name: string;
  is_calibrated: boolean;
  pr_auc: number;
  pr_auc_ci_low: number;
  pr_auc_ci_high: number;
  roc_auc: number | null;
  recall_at_precision_80: number | null;
  expected_calibration_error: number;
  brier_score: number;
  flag_rate: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  net_inr_per_1000_orders: number | null;
  train_pr_auc: number | null;
  overfit_gap: number | null;
}

export interface LadderResponse {
  dataset_run_id: string;
  evaluated_split: string;
  seed: number;
  cost_profile: string;
  threshold: number;
  threshold_source: string;
  config_fingerprint: string;
  feature_fingerprint: string;
  created_at: string;
  rungs: LadderRungResult[];
}

/**
 * The final model's metrics for one split.
 *
 * `evaluated_split` is load-bearing in the UI: validation figures are
 * selection-contaminated (hyperparameters were chosen on them and the shipped
 * calibrator was refitted on them) while test figures are the honest read. The
 * console must never show them without saying which is which.
 */
export interface FinalModelResponse {
  manifest_id: string;
  model_name: string;
  model_version: string;
  evaluated_split: string;
  calibration_method: string;
  is_calibrated: boolean;
  n_rows: number;
  positive_rate: number;
  pr_auc: number;
  pr_auc_ci_low: number;
  pr_auc_ci_high: number;
  pr_auc_uncalibrated: number;
  roc_auc: number | null;
  recall_at_precision_80: number | null;
  recall_at_precision_90: number | null;
  brier_score: number;
  brier_score_uncalibrated: number;
  expected_calibration_error: number;
  expected_calibration_error_uncalibrated: number;
  threshold: number;
  flag_rate: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  true_negatives: number;
  net_inr_per_1000_orders: number;
  net_ci_low: number;
  net_ci_high: number;
  false_positive_cost_inr: number;
  do_nothing_loss_per_1000_orders: number;
  unseal_reason: string | null;
  evaluated_at: string;
  data_provenance: string;
}

// ---------------------------------------------------------------------------
// monitoring
// ---------------------------------------------------------------------------

export interface ModelStatusResponse {
  available: boolean;
  reason: string | null;
  model_name: string | null;
  model_version: string | null;
  calibration_method: string | null;
  calibration_fitted_on: string | null;
  feature_version: string | null;
  dataset_run_id: string | null;
  trained_at: string | null;
  training_rows: number | null;
  n_features: number | null;
  selection_manifest_id: string | null;
}

export interface DatasetRunSummary {
  run_id: string;
  generator_version: string;
  n_orders: number;
  created_at: string;
}

export interface DataStatusResponse {
  dataset_runs: DatasetRunSummary[];
  total_orders: number;
  orders_by_split: Record<string, number>;
  orders_by_payment_method: Record<string, number>;
  matured_orders: number;
  immature_orders: number;
  /** Computed over matured orders only. Null when none have matured. */
  observed_rto_rate: number | null;
}

export interface DecisionStatusResponse {
  total_decisions: number;
  decisions_by_band: Record<string, number>;
  awaiting_human_review: number;
  total_overrides: number;
  overrides_by_direction: Record<string, number>;
  override_rate: number | null;
}

// ---------------------------------------------------------------------------
// agents
// ---------------------------------------------------------------------------

export interface AgentStatusResponse {
  available: boolean;
  reason: string | null;
  provider: string;
  model: string;
  required_environment_variable: string;
  enable_switch: string;
  tools: string[];
  note: string;
}

export interface ToolInvocation {
  tool: string;
  arguments: Record<string, unknown>;
  found: boolean;
  reason: string | null;
  duration_ms: number;
  error: string | null;
}

export interface AgentAuditRecord {
  agent_type: string;
  request: string;
  subject_id: string | null;
  provider: string;
  model: string;
  duration_ms: number;
  tools_invoked: ToolInvocation[];
  llm_turns: number;
  input_tokens: number | null;
  output_tokens: number | null;
  grounded: boolean | null;
  rejection_reason: string | null;
  error: string | null;
}

/**
 * The agent's answer.
 *
 * `probability`, `band`, `threshold` and `model_version` are copied from the
 * tool results by the backend, not parsed from the model's prose. The console
 * displays those fields, never numbers scraped out of `summary` - which is what
 * makes it structurally impossible for a language model to change what this
 * screen reports.
 */
export interface RiskInvestigation {
  order_id: string;
  sufficient_evidence: boolean;
  summary: string;
  key_drivers: string[];
  evidence_used: string[];
  uncertainty: string;
  caveats: string[];
  probability: number | null;
  band: string | null;
  threshold: number | null;
  model_version: string | null;
  reason_codes: string[];
  generated_at: string;
  llm_model: string;
  grounded: boolean;
  rejection_reason: string | null;
}

export interface InvestigationResponse {
  investigation: RiskInvestigation;
  audit: AgentAuditRecord;
}

export interface ToolCatalogueEntry {
  name: string;
  purpose: string;
  permission: string;
  input_schema: Record<string, unknown>;
}

// --- responsible AI: cohorts, distribution shift, drift ----------------------

/**
 * One slice of the cohort audit.
 *
 * `sufficient` is the field that matters most. A group below the minimum support
 * is shown - suppressing thin groups would hide exactly what a fairness audit
 * exists to look at - but it is an observation, not evidence, and the console
 * must render it visibly differently.
 */
export interface CohortResult {
  cohort: string;
  group: string;
  n_orders: number;
  flag_rate: number;
  precision: number | null;
  recall: number | null;
  net_inr_per_1000: number | null;
  rto_rate: number | null;
  n_positives: number;
  n_flagged: number;
  flag_rate_ci: [number, number] | null;
  precision_ci: [number, number] | null;
  rto_rate_ci: [number, number] | null;
  sufficient: boolean;
  insufficient_reason: string;
}

export interface FairnessAudit {
  slices: CohortResult[];
  max_flag_rate_ratio: number;
  worst_precision_drop: number;
  triggered: boolean;
  narrative: string;
  cohorts_examined: string[];
  groups_below_support: string[];
  min_support: number;
  most_flagged_group: string;
  least_flagged_group: string;
}

export interface FairnessResponse {
  generated_at: string;
  dataset_run_id: string;
  split: string;
  model_version: string;
  threshold: number;
  disclaimer: string;
  audit: FairnessAudit;
  slices: CohortResult[];
}

export interface EnvironmentSpec {
  name: string;
  description: string;
  overrides: Record<string, number>;
  seed: number;
  n_orders: number;
}

export interface ShiftResult {
  environment: string;
  description: string;
  n_orders: number;
  observed_rto_rate: number;
  pr_auc: number;
  /**
   * PR-AUC divided by the base rate.
   *
   * Read this, not `pr_auc`, when comparing environments. A random ranker scores
   * PR-AUC equal to the positive rate, so an environment whose base rate moved
   * has a different floor - and raw PR-AUC rises when the world gets riskier.
   */
  pr_auc_lift: number;
  roc_auc: number;
  brier_score: number;
  expected_calibration_error: number;
  threshold: number;
  flag_rate: number;
  precision: number | null;
  recall: number | null;
  net_inr_per_1000: number;
  pr_auc_delta: number | null;
  pr_auc_lift_delta: number | null;
  net_delta: number | null;
  ece_delta: number | null;
}

export interface ShiftStudy {
  generated_at: string;
  model_version: string;
  feature_version: string;
  generator_version: string;
  threshold: number;
  environments: EnvironmentSpec[];
  results: ShiftResult[];
  findings: string[];
  data_provenance: string;
}

export type DriftSeverity = 'stable' | 'watch' | 'investigate';

export type DriftKind =
  | 'feature'
  | 'prediction'
  | 'outcome_rate'
  | 'flag_rate'
  | 'calibration';

export interface WindowSummary {
  label: string;
  n_orders: number;
  start: string | null;
  end: string | null;
  n_matured: number;
}

/** A distance between two windows. Deliberately carries no verdict. */
export interface DriftSignal {
  name: string;
  kind: DriftKind;
  statistic: string;
  distance: number;
  severity: DriftSeverity;
  baseline_value: number | null;
  current_value: number | null;
  baseline_n: number;
  current_n: number;
  sufficient: boolean;
  note: string;
}

/** A measured change in model quality. Only exists where labels do. */
export interface PerformanceDelta {
  metric: string;
  baseline: number;
  current: number;
  delta: number;
  n_baseline_matured: number;
  n_current_matured: number;
  sufficient: boolean;
  note: string;
}

export interface DriftReport {
  generated_at: string;
  baseline: WindowSummary;
  current: WindowSummary;
  signals: DriftSignal[];
  performance: PerformanceDelta[];
  warnings: string[];
  model_version: string;
  feature_version: string;
  /**
   * False when no labelled comparison was possible.
   *
   * The console must surface this prominently: an all-stable signal table
   * computed with no labels is the most misleading thing this system can show.
   */
  labels_available: boolean;
  data_provenance: string;
}

// --- ablation: what each feature family is worth ----------------------------

/**
 * One leave-one-family-out arm.
 *
 * `verdict` carries the reading rule, computed by the backend: an arm whose
 * bootstrap interval spans zero is "not established", which is NOT the same as
 * "contributes nothing". The console renders the verdict rather than
 * re-deriving it, so there is one implementation of that judgement.
 */
export interface AblationArm {
  family_removed: string;
  n_features: number;
  net_inr_per_1000: number;
  delta_vs_full: number;
  delta_ci_low: number;
  delta_ci_high: number;
  pr_auc: number;
  delta_pr_auc_vs_full: number;
  flag_rate: number;
  precision: number | null;
  threshold: number;
  chosen_candidate: string;
  calibration_method: string;
  verdict: string;
}

export interface AblationStudy {
  generated_at: string;
  dataset_run_id: string;
  split: string;
  seed: number;
  cost_profile: string;
  full_model: AblationArm;
  arms: AblationArm[];
  findings: string[];
  data_provenance: string;
}
