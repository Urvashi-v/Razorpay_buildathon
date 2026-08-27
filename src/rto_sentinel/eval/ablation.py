"""Leave-one-family-out ablation.

SPEC section 07: "how much does each feature family actually contribute?"

Answered by retraining with one family disabled at a time and reporting the
change in NET RUPEES, not in AUC. A family that adds ranking quality but no money
has not earned its place - and the geography family in particular has to justify
its fairness cost with a real economic contribution, not merely a lift.

STATUS: Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AblationResult:
    """One leave-one-out run, expressed as a delta against the full model."""

    family_removed: str
    net_inr_per_1000: float
    delta_vs_full: float
    pr_auc: float
    delta_pr_auc_vs_full: float


def run_ablation(families: list[str]) -> list[AblationResult]:
    """Retrain with each family removed in turn and report the deltas."""
    raise NotImplementedError("Ablation study lands in Phase 4.")
