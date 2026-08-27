"""The fairness audit. Not decoration.

SPEC section 07: "A model that concentrates its flags on tier-3 pincodes has not
found fraud, it has found poverty and bad municipal addressing."

WHAT THIS MODULE CHECKS
-----------------------
Flag rate and precision, reported *separately*, by pincode tier and by
order-value band. The question is not whether the flag rate is equal across
tiers - it will not be, and forcing equality would be its own kind of dishonesty.
The question is whether **precision holds up in the tiers that get flagged
most**. A group flagged twice as often but with materially worse precision is a
group having cost transferred onto it without justification, and that is exactly
the trigger condition configured in ``config/evaluation.yaml``.

WHAT HAPPENS WHEN IT TRIPS
--------------------------
The smoothed geography features get pulled back - stronger shrinkage, a higher
minimum support, or the family disabled outright - and the model is retrained and
re-audited. Both the trip and the remedy go into REPORT.md, including the runs
where the audit found nothing worth acting on.

For a payments company, a risk model with unexamined disparate impact is a
regulatory and reputational liability. That is why this is a build-time gate
rather than an appendix nobody reads.

STATUS: Phase 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

    from rto_sentinel.configuration.schemas import FairnessConfig
    from rto_sentinel.contracts.evaluation import CohortResult, FairnessAudit


def cohort_breakdown(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    cohort_column: str,
) -> tuple[CohortResult, ...]:
    """Flag rate, precision, recall and net rupees for each group in a cohort."""
    raise NotImplementedError("Cohort breakdown lands in Phase 4.")


def fairness_audit(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    config: FairnessConfig,
) -> FairnessAudit:
    """Run the disparate-impact review and report whether it tripped."""
    raise NotImplementedError("Fairness audit lands in Phase 4.")
