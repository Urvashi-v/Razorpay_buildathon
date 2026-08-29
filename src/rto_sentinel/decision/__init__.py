"""The deterministic economic decision layer.

Depends on: configuration, contracts. Depends on NOTHING downstream, and in
particular imports nothing from ``rto_sentinel.agents``, ``rto_sentinel.api`` or
any LLM SDK. That constraint is asserted mechanically by
``tests/architecture/test_layering.py``.
"""

from rto_sentinel.decision.cost_model import (
    OutcomeEconomics,
    band_outcome_economics,
    expected_value_of_flagging,
    outcome_economics,
)
from rto_sentinel.decision.engine import (
    ENGINE_VERSION,
    SCORE_ONLY_REASON,
    DecisionEngine,
    UncalibratedScoreError,
)
from rto_sentinel.decision.policy import (
    PolicyError,
    ResolvedLadder,
    action_for,
    band_economics,
    band_for,
    requires_human_review,
    requires_reason_code,
    resolve_boundaries,
)
from rto_sentinel.decision.portfolio import PortfolioError, evaluate_portfolio
from rto_sentinel.decision.reason_codes import FAMILY_CODE_STEMS, code_for, derive_reason_codes
from rto_sentinel.decision.simulation import (
    LadderRung,
    PolicyComparison,
    SimulationError,
    SimulationResult,
    compare_ladder_against_uniform,
    simulate,
)
from rto_sentinel.decision.threshold import derive_threshold, threshold_sensitivity
from rto_sentinel.decision.threshold_analysis import (
    SELECTION_METHODOLOGY,
    SweepError,
    sweep_thresholds,
    sweep_to_rows,
)

__all__ = [
    "ENGINE_VERSION",
    "FAMILY_CODE_STEMS",
    "SCORE_ONLY_REASON",
    "SELECTION_METHODOLOGY",
    "DecisionEngine",
    "LadderRung",
    "OutcomeEconomics",
    "PolicyComparison",
    "PolicyError",
    "PortfolioError",
    "ResolvedLadder",
    "SimulationError",
    "SimulationResult",
    "SweepError",
    "UncalibratedScoreError",
    "action_for",
    "band_economics",
    "band_for",
    "band_outcome_economics",
    "code_for",
    "compare_ladder_against_uniform",
    "derive_reason_codes",
    "derive_threshold",
    "evaluate_portfolio",
    "expected_value_of_flagging",
    "outcome_economics",
    "requires_human_review",
    "requires_reason_code",
    "resolve_boundaries",
    "simulate",
    "sweep_thresholds",
    "sweep_to_rows",
    "threshold_sensitivity",
]
