"""Monitoring: what moved between two periods, and what that does and does not mean.

The package is deliberately thin and label-aware. `drift` measures distances
between windows; `report` turns them into sentences an operations lead can act
on. Nothing here decides that a model has failed - that verdict needs labels, and
where labels are absent the report says the question is unanswered.
"""

from rto_sentinel.monitoring.drift import (
    MIN_WINDOW_ROWS,
    PSI_INVESTIGATE,
    PSI_WATCH,
    RATE_INVESTIGATE,
    RATE_WATCH,
    calibration_drift,
    categorical_psi,
    feature_drift,
    kolmogorov_smirnov,
    performance_delta,
    population_stability_index,
    prediction_drift,
    rate_drift,
)
from rto_sentinel.monitoring.outcomes import (
    MIN_PER_ARM,
    InterventionEffectiveness,
    OverrideSummary,
    intervention_effectiveness,
    override_summary,
)
from rto_sentinel.monitoring.report import build_drift_report, interpret

__all__ = [
    "MIN_PER_ARM",
    "MIN_WINDOW_ROWS",
    "PSI_INVESTIGATE",
    "PSI_WATCH",
    "RATE_INVESTIGATE",
    "RATE_WATCH",
    "InterventionEffectiveness",
    "OverrideSummary",
    "build_drift_report",
    "calibration_drift",
    "categorical_psi",
    "feature_drift",
    "interpret",
    "intervention_effectiveness",
    "kolmogorov_smirnov",
    "override_summary",
    "performance_delta",
    "population_stability_index",
    "prediction_drift",
    "rate_drift",
]
