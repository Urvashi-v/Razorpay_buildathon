"""The outcome loop: what it measures, and what it refuses to measure.

The one property that matters here is that `is_assumed` only ever flips to False
on a real measurement. Every other test in this file exists to make that hard to
break, because a success rate quoted off twenty control orders would carry the
authority of a measurement while being noise - and it would silently replace the
assumption that every rupee figure in this project currently rests on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.monitoring.outcomes import (
    MIN_PER_ARM,
    intervention_effectiveness,
    override_summary,
)


def arm(n: int, *, control: bool, rto_rate: float, band: str = "HIGH", seed: int = 1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "band": band,
            "is_control_holdout": control,
            "is_rto": rng.random(n) < rto_rate,
        }
    )


def book(*, treated: int, control: int, treated_rate: float, control_rate: float):
    return pd.concat(
        [
            arm(treated, control=False, rto_rate=treated_rate, seed=1),
            arm(control, control=True, rto_rate=control_rate, seed=2),
        ],
        ignore_index=True,
    )


class TestInsufficientData:
    def test_an_empty_log_stays_assumed(self) -> None:
        """Today's real state: nothing has ever been frictioned."""
        empty = pd.DataFrame(columns=["band", "is_control_holdout", "is_rto"])

        result = intervention_effectiveness(empty, "HIGH")

        assert result.measured_success_rate is None
        assert result.is_assumed is True
        assert result.n_treated == 0 and result.n_control == 0
        assert "remains the configured ASSUMPTION" in result.note

    def test_a_thin_control_arm_blocks_the_measurement(self) -> None:
        """The control arm is 2% of flagged orders, so it binds first.

        A large treated arm is not enough on its own - there is nothing to
        compare it against.
        """
        result = intervention_effectiveness(
            book(treated=5000, control=20, treated_rate=0.1, control_rate=0.3), "HIGH"
        )

        assert result.is_assumed is True
        assert result.measured_success_rate is None
        assert result.n_treated == 5000
        assert result.n_control == 20

    def test_a_thin_treated_arm_also_blocks_it(self) -> None:
        result = intervention_effectiveness(
            book(treated=10, control=500, treated_rate=0.1, control_rate=0.3), "HIGH"
        )
        assert result.is_assumed is True

    def test_rates_are_still_reported_when_the_measurement_is_blocked(self) -> None:
        """Insufficient is not the same as unknown. Show what there is."""
        result = intervention_effectiveness(
            book(treated=50, control=50, treated_rate=0.1, control_rate=0.3), "HIGH"
        )

        assert result.rto_rate_treated is not None
        assert result.rto_rate_control is not None
        assert result.measured_success_rate is None

    def test_a_control_arm_with_no_returns_is_not_a_success(self) -> None:
        """Zero would-be returns means nothing was prevented, not everything was."""
        result = intervention_effectiveness(
            book(
                treated=MIN_PER_ARM + 50,
                control=MIN_PER_ARM + 50,
                treated_rate=0.0,
                control_rate=0.0,
            ),
            "HIGH",
        )

        assert result.measured_success_rate is None
        assert result.is_assumed is True
        assert "Not a success" in result.note


class TestMeasurement:
    def test_effective_friction_produces_a_measured_rate(self) -> None:
        result = intervention_effectiveness(
            book(treated=900, control=400, treated_rate=0.12, control_rate=0.30), "HIGH"
        )

        assert result.is_assumed is False
        assert result.measured_success_rate is not None
        assert 0.4 < result.measured_success_rate < 0.75
        assert "replaces the configured assumption" in result.note

    def test_friction_that_does_nothing_measures_near_zero(self) -> None:
        result = intervention_effectiveness(
            book(treated=900, control=900, treated_rate=0.25, control_rate=0.25), "HIGH"
        )

        assert result.is_assumed is False
        assert result.measured_success_rate == pytest.approx(0.0, abs=0.15)

    def test_a_worse_treated_arm_clamps_to_zero_rather_than_going_negative(self) -> None:
        """A negative rate is noise at these sizes, not friction causing returns."""
        result = intervention_effectiveness(
            book(treated=900, control=900, treated_rate=0.40, control_rate=0.20), "HIGH"
        )

        assert result.measured_success_rate == 0.0

    def test_the_rate_never_exceeds_one(self) -> None:
        result = intervention_effectiveness(
            book(treated=900, control=900, treated_rate=0.0, control_rate=0.40), "HIGH"
        )
        assert result.measured_success_rate == 1.0


class TestMaturity:
    def test_immature_orders_are_dropped_not_counted_as_delivered(self) -> None:
        """Counting them would make every intervention look effective.

        A frictioned order that has not resolved yet has exactly the shape of a
        success, which is why this is the most dangerous shortcut available here.
        """
        frame = book(treated=300, control=300, treated_rate=0.2, control_rate=0.3)
        frame.loc[frame.index[:250], "is_rto"] = pd.NA

        result = intervention_effectiveness(frame, "HIGH")

        assert result.n_treated + result.n_control == 350

    def test_only_the_requested_band_is_measured(self) -> None:
        mixed = pd.concat(
            [
                book(treated=300, control=300, treated_rate=0.1, control_rate=0.3),
                book(treated=300, control=300, treated_rate=0.9, control_rate=0.9).assign(
                    band="ELEVATED"
                ),
            ],
            ignore_index=True,
        )

        result = intervention_effectiveness(mixed, "HIGH")

        assert result.n_treated == 300 and result.n_control == 300

    def test_a_frame_missing_a_column_is_refused(self) -> None:
        with pytest.raises(KeyError, match="missing"):
            intervention_effectiveness(pd.DataFrame({"band": ["HIGH"]}), "HIGH")


class TestOverrideSummary:
    def _decisions(self, bands: dict[str, int]) -> pd.DataFrame:
        rows = [{"band": band} for band, count in bands.items() for _ in range(count)]
        return pd.DataFrame(rows)

    def _overrides(self, entries: list[tuple[str, str]]) -> pd.DataFrame:
        return pd.DataFrame([{"band": band, "direction": direction} for band, direction in entries])

    def test_a_band_with_no_overrides_says_so(self) -> None:
        summaries = override_summary(self._overrides([]), self._decisions({"HIGH": 100}))
        assert summaries[0].reading.startswith("No overrides")
        assert summaries[0].override_rate == 0.0

    def test_consistent_relaxation_points_at_the_threshold(self) -> None:
        """The humans are not the ones being questioned here."""
        summaries = override_summary(
            self._overrides([("HIGH", "relaxed")] * 30 + [("HIGH", "escalated")] * 2),
            self._decisions({"HIGH": 100}),
        )

        assert "threshold being too low" in summaries[0].reading
        assert summaries[0].n_relaxed == 30
        assert summaries[0].override_rate == pytest.approx(0.32)

    def test_consistent_escalation_points_at_the_model(self) -> None:
        summaries = override_summary(
            self._overrides([("HIGH", "escalated")] * 30 + [("HIGH", "relaxed")] * 2),
            self._decisions({"HIGH": 100}),
        )
        assert "modelling gap" in summaries[0].reading

    def test_balanced_overrides_are_reported_as_balanced(self) -> None:
        summaries = override_summary(
            self._overrides([("HIGH", "relaxed")] * 10 + [("HIGH", "escalated")] * 10),
            self._decisions({"HIGH": 100}),
        )
        assert "roughly balanced" in summaries[0].reading

    def test_every_band_with_decisions_appears_even_with_no_overrides(self) -> None:
        summaries = override_summary(
            self._overrides([("HIGH", "relaxed")]),
            self._decisions({"LOW": 500, "ELEVATED": 200, "HIGH": 50}),
        )
        assert {s.band for s in summaries} == {"LOW", "ELEVATED", "HIGH"}

    def test_a_non_empty_frame_missing_a_column_is_refused(self) -> None:
        """An empty log is normal; a populated one missing `direction` is a bug."""
        with pytest.raises(KeyError, match="missing"):
            override_summary(pd.DataFrame({"band": ["HIGH"]}), pd.DataFrame({"band": ["HIGH"]}))

    def test_a_completely_empty_override_log_is_not_an_error(self) -> None:
        """A system nobody has overridden yet is the common case, not a fault."""
        summaries = override_summary(pd.DataFrame(), self._decisions({"HIGH": 10}))

        assert len(summaries) == 1
        assert summaries[0].n_relaxed == 0 and summaries[0].n_escalated == 0
