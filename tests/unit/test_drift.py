"""Drift arithmetic, and the line between "moved" and "got worse".

The central property under test is not a formula: it is that this module cannot
report degradation it did not measure. A drift page that shows green because it
had no labels is the most dangerous artefact the monitoring layer could produce,
so several tests here assert on prose - the warnings are the product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.monitoring import (
    MIN_WINDOW_ROWS,
    PSI_INVESTIGATE,
    PSI_WATCH,
    build_drift_report,
    calibration_drift,
    categorical_psi,
    feature_drift,
    interpret,
    kolmogorov_smirnov,
    performance_delta,
    population_stability_index,
    prediction_drift,
    rate_drift,
)


def book(n: int, *, shift: float = 0.0, seed: int = 0, matured: bool = True) -> pd.DataFrame:
    """A scored window with a controllable score shift."""
    rng = np.random.default_rng(seed)
    scores = np.clip(rng.beta(2, 8, n) + shift, 0.0, 1.0)
    labels = (rng.random(n) < scores).astype(float)
    return pd.DataFrame(
        {
            "score_calibrated": scores,
            "label": labels if matured else np.full(n, np.nan),
            "ordered_at": pd.date_range("2026-01-01", periods=n, freq="h"),
            "order_value_inr": rng.lognormal(6.85, 0.72, n),
            "is_cod": rng.random(n) < 0.62,
        }
    )


class TestPopulationStabilityIndex:
    def test_identical_samples_have_no_distance(self) -> None:
        sample = np.random.default_rng(1).normal(size=2000)
        assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_psi_is_non_negative(self) -> None:
        rng = np.random.default_rng(2)
        for shift in (-2.0, -0.5, 0.0, 0.5, 2.0):
            distance = population_stability_index(
                rng.normal(size=1000), rng.normal(size=1000) + shift
            )
            assert distance >= 0.0

    def test_psi_grows_with_the_size_of_the_shift(self) -> None:
        rng = np.random.default_rng(3)
        baseline = rng.normal(size=4000)
        small = population_stability_index(baseline, rng.normal(size=4000) + 0.2)
        large = population_stability_index(baseline, rng.normal(size=4000) + 1.5)
        assert large > small

    def test_an_empty_window_yields_zero_rather_than_an_error(self) -> None:
        assert population_stability_index(np.array([]), np.array([1.0, 2.0])) == 0.0

    def test_a_constant_baseline_cannot_drift_in_shape(self) -> None:
        """Shape drift is undefined with one bin; the level is reported separately."""
        assert population_stability_index(np.full(500, 3.0), np.full(500, 9.0)) == 0.0

    def test_an_empty_bin_produces_a_large_finite_distance(self) -> None:
        """log(0) is undefined; an absent value range is information, not a crash."""
        baseline = np.random.default_rng(4).normal(size=2000)
        current = baseline[baseline > 0]
        distance = population_stability_index(baseline, current)
        assert np.isfinite(distance)
        assert distance > PSI_INVESTIGATE

    def test_values_outside_the_baseline_range_are_counted_not_dropped(self) -> None:
        """A value the baseline never saw is exactly what this should catch."""
        baseline = np.random.default_rng(5).uniform(0, 1, 2000)
        current = np.concatenate([baseline[:1000], np.full(1000, 99.0)])
        assert population_stability_index(baseline, current) > PSI_WATCH

    def test_nan_values_are_excluded_rather_than_propagated(self) -> None:
        baseline = np.array([1.0, 2.0, 3.0, np.nan] * 100)
        current = np.array([1.0, 2.0, 3.0] * 100)
        assert np.isfinite(population_stability_index(baseline, current))


class TestCategoricalPsi:
    def test_identical_shares_have_no_distance(self) -> None:
        series = pd.Series(["a"] * 60 + ["b"] * 40)
        assert categorical_psi(series, series) == pytest.approx(0.0, abs=1e-9)

    def test_a_category_only_in_the_current_window_registers(self) -> None:
        """Scoring against the baseline's category list alone would discard it."""
        baseline = pd.Series(["a"] * 100)
        current = pd.Series(["a"] * 50 + ["brand_new"] * 50)
        assert categorical_psi(baseline, current) > PSI_INVESTIGATE

    def test_nulls_are_a_category_rather_than_a_silent_drop(self) -> None:
        baseline = pd.Series(["a"] * 100)
        current = pd.Series([None] * 100)
        assert categorical_psi(baseline, current) > PSI_INVESTIGATE


class TestKolmogorovSmirnov:
    def test_identical_samples_have_no_distance(self) -> None:
        sample = np.random.default_rng(6).normal(size=500)
        assert kolmogorov_smirnov(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_the_statistic_is_bounded_by_one(self) -> None:
        assert kolmogorov_smirnov(np.zeros(100), np.ones(100)) <= 1.0

    def test_disjoint_samples_reach_the_maximum(self) -> None:
        assert kolmogorov_smirnov(np.zeros(100), np.ones(100)) == pytest.approx(1.0)


class TestSeverityBands:
    def test_a_thin_window_never_reports_a_severity(self) -> None:
        """PSI on a handful of rows is noise with a number attached."""
        tiny = book(10, seed=7)
        other = book(10, shift=0.4, seed=8)
        signals = prediction_drift(
            tiny["score_calibrated"].to_numpy(), other["score_calibrated"].to_numpy()
        )
        assert all(signal.severity == "stable" for signal in signals)
        assert all(not signal.sufficient for signal in signals)

    def test_a_large_shift_on_a_large_window_reports_investigate(self) -> None:
        baseline = book(2000, seed=9)
        current = book(2000, shift=0.35, seed=10)
        signals = prediction_drift(
            baseline["score_calibrated"].to_numpy(), current["score_calibrated"].to_numpy()
        )
        assert any(signal.severity == "investigate" for signal in signals)

    def test_the_minimum_window_is_a_real_bound(self) -> None:
        assert MIN_WINDOW_ROWS > 1


class TestRateDrift:
    def test_the_statistic_is_the_plain_difference(self) -> None:
        """A person reading a monitoring page can act on percentage points."""
        signal = rate_drift("rto_rate", "outcome_rate", 100, 1000, 200, 1000)
        assert signal.baseline_value == pytest.approx(0.10)
        assert signal.current_value == pytest.approx(0.20)
        assert signal.distance == pytest.approx(0.10)

    def test_direction_does_not_change_the_distance(self) -> None:
        up = rate_drift("r", "outcome_rate", 100, 1000, 200, 1000)
        down = rate_drift("r", "outcome_rate", 200, 1000, 100, 1000)
        assert up.distance == pytest.approx(down.distance)

    def test_an_empty_window_does_not_divide_by_zero(self) -> None:
        signal = rate_drift("r", "flag_rate", 0, 0, 5, 100)
        assert signal.baseline_value == 0.0
        assert signal.sufficient is False


class TestCalibrationDrift:
    def test_without_labels_it_reports_insufficiency_not_stability(self) -> None:
        """The dangerous failure is reporting a reassuring number here."""
        signal = calibration_drift(np.array([]), np.array([]), np.array([]), np.array([]))
        assert signal.sufficient is False
        assert "cannot be measured without outcomes" in signal.note
        assert "not assumed to be unchanged" in signal.note

    def test_identical_windows_show_no_calibration_movement(self) -> None:
        rng = np.random.default_rng(11)
        scores = rng.beta(2, 5, 800)
        labels = (rng.random(800) < scores).astype(int)
        signal = calibration_drift(scores, labels, scores, labels)
        assert signal.distance == pytest.approx(0.0, abs=1e-9)
        assert signal.severity == "stable"

    def test_a_broken_calibration_is_detected(self) -> None:
        rng = np.random.default_rng(12)
        scores = rng.beta(2, 5, 1500)
        good = (rng.random(1500) < scores).astype(int)
        # Same scores, outcomes now unrelated to them at a much higher rate.
        broken = (rng.random(1500) < 0.8).astype(int)
        signal = calibration_drift(scores, good, scores, broken)
        assert signal.severity == "investigate"
        assert signal.current_value is not None
        assert signal.baseline_value is not None
        assert signal.current_value > signal.baseline_value


class TestPerformanceDelta:
    def test_the_delta_must_equal_current_minus_baseline(self) -> None:
        """The contract enforces it; a hand-set delta is a lie waiting to be read."""
        delta = performance_delta(
            "pr_auc", 0.50, 0.42, n_baseline_matured=500, n_current_matured=500
        )
        assert delta.delta == pytest.approx(-0.08)

    def test_a_thin_window_marks_the_comparison_insufficient(self) -> None:
        delta = performance_delta("pr_auc", 0.50, 0.10, n_baseline_matured=5, n_current_matured=5)
        assert delta.sufficient is False
        assert "sampling noise" in delta.note


class TestInterpretation:
    def test_missing_labels_lead_the_warnings(self) -> None:
        """Absence of alarms must be reported as absence of information."""
        warnings = interpret((), (), labels_available=False)
        assert "absence of information" in warnings[0]
        assert "not as evidence the model is fine" in warnings[0]

    def test_drift_alone_is_never_described_as_failure(self) -> None:
        baseline = book(1500, seed=13)
        current = book(1500, seed=14)
        current = current.assign(order_value_inr=current["order_value_inr"] * 3)
        signals = feature_drift(baseline, current, columns=["order_value_inr"])
        warnings = interpret(signals, (), labels_available=True)
        text = " ".join(warnings)
        assert "not by itself a problem" in text
        for forbidden in ("model has failed", "model is broken", "failure detected"):
            assert forbidden not in text.lower()

    def test_a_measured_regression_is_named_as_evidence(self) -> None:
        deltas = (
            performance_delta("pr_auc", 0.52, 0.40, n_baseline_matured=900, n_current_matured=900),
        )
        warnings = interpret((), deltas, labels_available=True)
        assert any("evidence about model quality" in warning for warning in warnings)

    def test_a_movement_below_the_material_threshold_is_not_called_out(self) -> None:
        deltas = (
            performance_delta(
                "pr_auc", 0.500, 0.4995, n_baseline_matured=900, n_current_matured=900
            ),
        )
        warnings = interpret((), deltas, labels_available=True)
        assert any("No measured degradation" in warning for warning in warnings)

    def test_lower_is_better_metrics_read_in_the_right_direction(self) -> None:
        """A rising Brier score is worse; a rising PR-AUC is better."""
        worse = interpret(
            (),
            (
                performance_delta(
                    "brier_score", 0.10, 0.20, n_baseline_matured=900, n_current_matured=900
                ),
            ),
            labels_available=True,
        )
        assert any("Measured change" in warning for warning in worse)

        better = interpret(
            (),
            (
                performance_delta(
                    "brier_score", 0.20, 0.10, n_baseline_matured=900, n_current_matured=900
                ),
            ),
            labels_available=True,
        )
        assert any("No measured degradation" in warning for warning in better)

    def test_the_outcome_rate_is_explained_before_the_flag_rate(self) -> None:
        """Cause before effect: a riskier book, not a model gone haywire."""
        signals = (
            rate_drift("flag_rate", "flag_rate", 100, 1000, 250, 1000),
            rate_drift("rto_rate", "outcome_rate", 150, 1000, 300, 1000),
        )
        warnings = interpret(signals, (), labels_available=True)
        joined = [w for w in warnings if "rate" in w]
        assert "RTO rate" in joined[0]
        assert "flag rate" in joined[1]


class TestDriftReport:
    def test_a_report_over_two_identical_windows_is_quiet(self) -> None:
        frame = book(1200, seed=15)
        report = build_drift_report(frame, frame.copy(), threshold=0.348)
        assert report.worst_severity == "stable"
        assert report.labels_available is True

    def test_an_unmatured_current_window_blocks_every_labelled_comparison(self) -> None:
        """The situation that actually occurs in production."""
        baseline = book(1200, seed=16)
        current = book(800, seed=17, matured=False)
        report = build_drift_report(baseline, current, threshold=0.348)

        assert report.labels_available is False
        assert report.performance == ()
        assert report.current.n_matured == 0
        # Rates that need labels are absent; rates that do not are present.
        kinds = {signal.kind for signal in report.signals}
        assert "outcome_rate" not in kinds
        assert "flag_rate" in kinds
        assert "absence of information" in report.warnings[0]

    def test_calibration_is_reported_unmeasurable_rather_than_stable(self) -> None:
        baseline = book(1200, seed=18)
        current = book(800, seed=19, matured=False)
        report = build_drift_report(baseline, current, threshold=0.348)
        calibration = [s for s in report.signals if s.kind == "calibration"]
        assert len(calibration) == 1
        assert calibration[0].sufficient is False

    def test_a_null_label_is_never_counted_as_a_non_rto(self) -> None:
        """Counting immature orders as delivered flatters every window."""
        frame = book(600, seed=20)
        frame.loc[frame.index[:300], "label"] = np.nan
        report = build_drift_report(frame, frame.copy(), threshold=0.348)
        assert report.baseline.n_orders == 600
        assert report.baseline.n_matured == 300

    def test_windows_record_their_time_bounds(self) -> None:
        frame = book(500, seed=21)
        report = build_drift_report(frame, frame.copy(), threshold=0.348)
        assert report.baseline.start is not None
        assert report.baseline.end is not None
        assert report.baseline.end > report.baseline.start

    def test_the_report_is_deterministic(self) -> None:
        """A drift alarm nobody can reproduce cannot be investigated."""
        baseline, current = book(900, seed=22), book(900, shift=0.1, seed=23)
        first = build_drift_report(baseline, current, threshold=0.348)
        second = build_drift_report(baseline, current, threshold=0.348)
        assert [s.model_dump() for s in first.signals] == [s.model_dump() for s in second.signals]
        assert first.warnings == second.warnings

    def test_a_schema_change_is_skipped_rather_than_reported_as_infinite_drift(self) -> None:
        """A missing column is a different problem with a different fix."""
        baseline = book(600, seed=24)
        current = book(600, seed=25).drop(columns=["order_value_inr"])
        report = build_drift_report(
            baseline, current, threshold=0.348, feature_columns=["order_value_inr", "is_cod"]
        )
        names = {s.name for s in report.signals if s.kind == "feature"}
        assert "order_value_inr" not in names
        assert "is_cod" in names
