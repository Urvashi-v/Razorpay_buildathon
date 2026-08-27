"""The deterministic economic decision layer.

Depends on: configuration, contracts. Depends on NOTHING downstream, and in
particular imports nothing from ``rto_sentinel.agents``, ``rto_sentinel.api`` or
any LLM SDK. That constraint is asserted mechanically by
``tests/architecture/test_layering.py``.
"""

from rto_sentinel.decision.cost_model import (
    OutcomeEconomics,
    expected_value_of_flagging,
    outcome_economics,
)
from rto_sentinel.decision.engine import (
    ENGINE_VERSION,
    DecisionEngine,
    UncalibratedScoreError,
)
from rto_sentinel.decision.policy import (
    action_for,
    band_for,
    requires_human_review,
    resolve_boundaries,
)
from rto_sentinel.decision.reason_codes import FAMILY_CODE_STEMS, derive_reason_codes
from rto_sentinel.decision.threshold import derive_threshold, threshold_sensitivity

__all__ = [
    "ENGINE_VERSION",
    "FAMILY_CODE_STEMS",
    "DecisionEngine",
    "OutcomeEconomics",
    "UncalibratedScoreError",
    "action_for",
    "band_for",
    "derive_reason_codes",
    "derive_threshold",
    "expected_value_of_flagging",
    "outcome_economics",
    "requires_human_review",
    "resolve_boundaries",
    "threshold_sensitivity",
]
