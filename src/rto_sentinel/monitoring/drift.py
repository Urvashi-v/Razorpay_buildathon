"""Feature and score drift monitoring.

SPEC section 07 and section 11. Customer behaviour shifts with festival cycles,
sale events and courier changes, and a model trained in September is not
automatically valid in November.

WHAT IS MONITORED
-----------------
* **Feature distributions** against the training reference, per family.
* **Score distribution**, because a stable feature set with a drifting score
  usually means an upstream data change rather than a behavioural one.
* **Calibration**, which is the one that actually matters here. If the model
  drifts out of calibration, the derived threshold silently stops being the
  cost-optimal operating point and the rupee numbers become wrong while every
  ranking metric still looks fine. That failure is invisible to a PR-AUC monitor.

The evaluation harness also reports performance on the final two weeks alone,
which is the same question asked retrospectively at build time.

STATUS: Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd


@dataclass(frozen=True, slots=True)
class DriftSignal:
    """One monitored quantity and how far it has moved."""

    name: str
    statistic: str
    value: float
    reference_value: float
    threshold: float
    breached: bool


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str],
) -> list[DriftSignal]:
    """Population stability index per feature against the training reference."""
    raise NotImplementedError("Drift monitoring lands in Phase 6.")


def calibration_drift(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    reference_ece: float,
    tolerance: float = 0.02,
) -> DriftSignal:
    """Has the model drifted out of calibration since it was fitted.

    The most consequential drift check in this system, for the reason in the
    module docstring: an uncalibrated model still ranks well and still reports a
    healthy PR-AUC while quietly making every rupee figure wrong.
    """
    raise NotImplementedError("Calibration drift lands in Phase 6.")
