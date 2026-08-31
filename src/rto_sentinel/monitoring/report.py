"""Turning distances into sentences an operations lead can act on.

A monitoring page that prints ``PSI=0.27`` has told an operations manager
nothing. One that prints "COD share rose from 61% to 74%. This moves the input
distribution, but COD share rising is an ordinary seasonal event and does not by
itself mean the model is worse - check the precision comparison below, which is
the question you actually care about" has told them what to do next.

That is the whole job of this module. It does not compute anything; it reads what
:mod:`rto_sentinel.monitoring.drift` measured and says what it means, including
saying "this cannot be answered yet" when the labels are not there.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from rto_sentinel.contracts.monitoring import (
    DriftReport,
    DriftSignal,
    PerformanceDelta,
    WindowSummary,
)
from rto_sentinel.monitoring.drift import (
    MIN_WINDOW_ROWS,
    calibration_drift,
    feature_drift,
    performance_delta,
    prediction_drift,
    rate_drift,
)

#: How many individual feature signals to name in the warnings. Beyond a handful
#: the prose stops being read; the full table is always available underneath.
MAX_NAMED_FEATURES = 5


def interpret(
    signals: tuple[DriftSignal, ...],
    performance: tuple[PerformanceDelta, ...],
    *,
    labels_available: bool,
) -> tuple[str, ...]:
    """Warnings, in the order an operator should read them.

    Ordered by what can actually be concluded: measured degradation first,
    because that is the only thing here that is evidence the model got worse;
    then calibration, because it invalidates the rupee arithmetic; then
    distribution moves, which are context.
    """
    warnings: list[str] = []

    # --- 1. what is actually known --------------------------------------
    if not labels_available:
        warnings.append(
            "No matured outcomes in the current window, so no measurement of model "
            "quality was possible. Everything below describes distributions that moved, "
            "not performance that changed. Treat an absence of alarms here as an absence "
            "of information, not as evidence the model is fine."
        )
    else:
        degraded = [
            delta
            for delta in performance
            if delta.sufficient and _is_worse(delta.metric, delta.delta)
        ]
        for delta in degraded:
            warnings.append(
                f"Measured change: {delta.metric} moved from {delta.baseline:.4f} to "
                f"{delta.current:.4f} ({delta.delta:+.4f}) on {delta.n_current_matured:,} "
                f"matured orders. This is a labelled comparison, so it is evidence about "
                f"model quality rather than a distribution shift."
            )
        if not degraded and performance:
            warnings.append(
                "No measured degradation: every labelled metric comparison is flat or "
                "improved between the two windows."
            )

    # --- 2. calibration, because the economics rest on it ----------------
    for signal in signals:
        if signal.kind != "calibration":
            continue
        if not signal.sufficient:
            warnings.append(
                "Calibration drift could not be measured - not enough matured outcomes. "
                "This matters more than the other gaps: the operating threshold is compared "
                "against a probability, so if calibration has moved, every rupee figure "
                "downstream is arithmetic on a number that no longer means what it says."
            )
        elif signal.severity != "stable":
            warnings.append(
                f"Calibration moved: expected calibration error went from "
                f"{signal.baseline_value:.4f} to {signal.current_value:.4f}. The economic "
                "threshold assumes the score is an honest probability, so this is the "
                "signal most likely to make the rupee figures wrong. Recalibrating on "
                "recent matured outcomes is the usual fix, and does not require retraining."
            )

    # --- 3. outcome and flag rates --------------------------------------
    # Outcome rate first, deliberately. A rising flag rate is usually the effect
    # and a rising RTO rate the cause; reading them in that order is the
    # difference between "the model went haywire" and "the book got riskier".
    rates = sorted(
        (
            signal
            for signal in signals
            if signal.kind in {"outcome_rate", "flag_rate"}
            and signal.severity != "stable"
            and signal.sufficient
        ),
        key=lambda signal: 0 if signal.kind == "outcome_rate" else 1,
    )
    for signal in rates:
        direction = "rose" if (signal.current_value or 0) > (signal.baseline_value or 0) else "fell"
        if signal.kind == "outcome_rate":
            warnings.append(
                f"The observed RTO rate {direction} from {signal.baseline_value:.1%} to "
                f"{signal.current_value:.1%}. A moving base rate changes what any fixed "
                "threshold does, but it is a fact about the book rather than a fault in "
                "the model - the model can be perfectly calibrated to a world that got "
                "riskier."
            )
        else:
            warnings.append(
                f"The flag rate {direction} from {signal.baseline_value:.1%} to "
                f"{signal.current_value:.1%}. More orders are receiving friction. Check "
                "this against the RTO rate above: if both moved together the model is "
                "tracking a genuinely riskier book; if the flag rate moved alone, the "
                "score distribution shifted without the outcomes following it."
            )

    # --- 4. inputs -------------------------------------------------------
    moved = [
        signal
        for signal in signals
        if signal.kind == "feature" and signal.severity != "stable" and signal.sufficient
    ]
    if moved:
        named = sorted(moved, key=lambda signal: signal.distance, reverse=True)[:MAX_NAMED_FEATURES]
        listed = ", ".join(f"{signal.name} (PSI {signal.distance:.3f})" for signal in named)
        extra = (
            f" and {len(moved) - len(named)} other feature(s)" if len(moved) > len(named) else ""
        )
        warnings.append(
            f"{len(moved)} input feature(s) moved between the windows: {listed}{extra}. "
            "Input drift is expected in a seasonal business and is not by itself a "
            "problem. It becomes one when it coincides with measured degradation or "
            "calibration movement above."
        )

    prediction_moved = [
        signal
        for signal in signals
        if signal.kind == "prediction" and signal.severity != "stable" and signal.sufficient
    ]
    if prediction_moved and not moved:
        warnings.append(
            "The score distribution moved while the input features did not. That "
            "combination usually means the model is receiving something different from "
            "what it was fed before - a feature that silently became null, or a pipeline "
            "version change - rather than a genuine change in the order book."
        )

    if not warnings:
        warnings.append(
            "Nothing moved beyond its watch band, on windows large enough for that to "
            "mean something."
        )

    return tuple(warnings)


#: Metrics where a larger number is worse. Everything else is read as
#: "higher is better", which is why the direction has to be declared rather than
#: guessed from the sign of the delta.
_LOWER_IS_BETTER = frozenset({"brier_score", "expected_calibration_error"})

#: How much a metric has to move before it is called out. Below this the
#: comparison is dominated by which orders happened to land in which window.
_MATERIAL_DELTA = 0.01


def _is_worse(metric: str, delta: float) -> bool:
    if abs(delta) < _MATERIAL_DELTA:
        return False
    return delta > 0 if metric in _LOWER_IS_BETTER else delta < 0


def build_drift_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    score_column: str = "score_calibrated",
    label_column: str = "label",
    feature_columns: list[str] | None = None,
    threshold: float,
    model_version: str = "",
    feature_version: str = "",
    baseline_label: str = "baseline",
    current_label: str = "current",
    time_column: str | None = "ordered_at",
    min_rows: int = MIN_WINDOW_ROWS,
) -> DriftReport:
    """Compare two windows and produce a report a human can read.

    ``label_column`` may be absent or all-null in the current window - that is the
    normal case for recent orders, which have not matured. Everything
    label-dependent is then reported as unmeasurable rather than skipped silently,
    because a drift page with no red on it and no labels behind it is the most
    misleading artefact this system could produce.
    """
    signals: list[DriftSignal] = []

    baseline_scores = baseline[score_column].to_numpy(dtype=float)
    current_scores = current[score_column].to_numpy(dtype=float)

    signals.extend(prediction_drift(baseline_scores, current_scores, min_rows=min_rows))

    if feature_columns:
        signals.extend(feature_drift(baseline, current, columns=feature_columns, min_rows=min_rows))

    # Flag rate needs no labels: it is a property of the scores and the threshold.
    signals.append(
        rate_drift(
            "flag_rate",
            "flag_rate",
            int((baseline_scores >= threshold).sum()),
            int(baseline_scores.size),
            int((current_scores >= threshold).sum()),
            int(current_scores.size),
            min_rows=min_rows,
        )
    )

    baseline_mature = _matured(baseline, label_column)
    current_mature = _matured(current, label_column)
    labels_available = len(current_mature) > 0 and len(baseline_mature) > 0

    performance: list[PerformanceDelta] = []
    if labels_available:
        baseline_labels = baseline_mature[label_column].to_numpy().astype(int)
        current_labels = current_mature[label_column].to_numpy().astype(int)
        baseline_mature_scores = baseline_mature[score_column].to_numpy(dtype=float)
        current_mature_scores = current_mature[score_column].to_numpy(dtype=float)

        signals.append(
            rate_drift(
                "rto_rate",
                "outcome_rate",
                int(baseline_labels.sum()),
                int(baseline_labels.size),
                int(current_labels.sum()),
                int(current_labels.size),
                min_rows=min_rows,
            )
        )
        signals.append(
            calibration_drift(
                baseline_mature_scores,
                baseline_labels,
                current_mature_scores,
                current_labels,
                min_rows=min_rows,
            )
        )
        performance.extend(
            _performance_deltas(
                baseline_labels,
                baseline_mature_scores,
                current_labels,
                current_mature_scores,
                threshold=threshold,
                min_rows=min_rows,
            )
        )
    else:
        signals.append(
            calibration_drift(
                np.array([]), np.array([]), np.array([]), np.array([]), min_rows=min_rows
            )
        )

    return DriftReport(
        generated_at=datetime.now(UTC),
        baseline=_window(baseline, baseline_label, len(baseline_mature), time_column),
        current=_window(current, current_label, len(current_mature), time_column),
        signals=tuple(signals),
        performance=tuple(performance),
        warnings=interpret(tuple(signals), tuple(performance), labels_available=labels_available),
        model_version=model_version,
        feature_version=feature_version,
        labels_available=labels_available,
    )


def _matured(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Rows with a known outcome. A null label is 'not yet', never 'delivered'."""
    if label_column not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame[label_column].notna()]


def _window(
    frame: pd.DataFrame, label: str, n_matured: int, time_column: str | None
) -> WindowSummary:
    start = end = None
    if time_column and time_column in frame.columns and len(frame):
        times = pd.to_datetime(frame[time_column])
        start = times.min().to_pydatetime()
        end = times.max().to_pydatetime()
    return WindowSummary(
        label=label, n_orders=len(frame), start=start, end=end, n_matured=n_matured
    )


def _performance_deltas(
    baseline_labels: np.ndarray,
    baseline_scores: np.ndarray,
    current_labels: np.ndarray,
    current_scores: np.ndarray,
    *,
    threshold: float,
    min_rows: int,
) -> list[PerformanceDelta]:
    """The labelled comparisons: ranking, calibration, and operating-point quality."""
    from rto_sentinel.eval.metrics import confusion_at_threshold, pr_auc

    deltas: list[PerformanceDelta] = []
    n_baseline = int(baseline_labels.size)
    n_current = int(current_labels.size)

    deltas.append(
        performance_delta(
            "pr_auc",
            pr_auc(baseline_labels, baseline_scores),
            pr_auc(current_labels, current_scores),
            n_baseline_matured=n_baseline,
            n_current_matured=n_current,
            min_rows=min_rows,
        )
    )

    baseline_confusion = confusion_at_threshold(baseline_labels, baseline_scores, threshold)
    current_confusion = confusion_at_threshold(current_labels, current_scores, threshold)
    operating_point: tuple[tuple[str, float, float], ...] = (
        ("precision", baseline_confusion.precision, current_confusion.precision),
        ("recall", baseline_confusion.recall, current_confusion.recall),
    )
    for metric, before, after in operating_point:
        deltas.append(
            performance_delta(
                metric,
                before,
                after,
                n_baseline_matured=n_baseline,
                n_current_matured=n_current,
                min_rows=min_rows,
            )
        )

    deltas.append(
        performance_delta(
            "brier_score",
            float(np.mean((baseline_scores - baseline_labels) ** 2)),
            float(np.mean((current_scores - current_labels) ** 2)),
            n_baseline_matured=n_baseline,
            n_current_matured=n_current,
            min_rows=min_rows,
        )
    )
    return deltas
