"""The cohort audit: arithmetic, small samples, and what it refuses to examine.

The most important tests here are the negative ones. A fairness audit that
computes the right numbers but can be pointed at an inferred sensitive attribute
is worse than no audit, because it launders the inference through a process that
looks like diligence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import FairnessConfig
from rto_sentinel.eval.fairness import (
    DEFAULT_MIN_FLAGGED,
    SENSITIVE_TOKENS,
    SensitiveCohortError,
    assert_no_sensitive_cohorts,
    band_column,
    cohort_breakdown,
    fairness_audit,
    history_band,
    shrink_towards,
    wilson_interval,
)


@pytest.fixture
def config() -> FairnessConfig:
    return FairnessConfig(
        group_by=["pincode_tier"],
        report=["flag_rate", "precision"],
        disparity_review_trigger={"flag_rate_ratio_above": 1.5, "precision_drop_below": 0.10},
    )


class TestSensitiveAttributes:
    """No sensitive characteristic may be examined, inferred, or approximated."""

    @pytest.mark.parametrize(
        "column",
        [
            "gender",
            "customer_gender",
            "inferred_religion",
            "caste_category",
            "name_derived_gender",
            "ethnicity",
            "estimated_age",
            "household_income",
            "marital_status",
            "mother_tongue",
        ],
    )
    def test_a_sensitive_cohort_is_refused(self, column: str) -> None:
        with pytest.raises(SensitiveCohortError, match="sensitive token"):
            assert_no_sensitive_cohorts([column])

    def test_refusal_is_fatal_rather_than_a_skipped_cohort(self, config: FairnessConfig) -> None:
        """Dropping the cohort with a warning would be the dangerous behaviour.

        An audit that silently examined less than its configuration claimed
        reports a cleaner result than it measured, which is how a fairness report
        starts being wrong in the direction of comfort.
        """
        frame = pd.DataFrame({"gender": ["a"] * 10})
        with pytest.raises(SensitiveCohortError):
            cohort_breakdown(
                frame,
                np.zeros(10, dtype=int),
                np.zeros(10),
                threshold=0.5,
                cohort_column="gender",
            )

    def test_the_audit_refuses_a_config_naming_a_sensitive_cohort(self) -> None:
        bad = FairnessConfig(
            group_by=["pincode_tier", "customer_gender"],
            report=["flag_rate"],
            disparity_review_trigger={},
        )
        frame = pd.DataFrame({"pincode_tier": ["tier_1"] * 10})
        with pytest.raises(SensitiveCohortError):
            fairness_audit(frame, np.zeros(10, dtype=int), np.zeros(10), threshold=0.5, config=bad)

    def test_operational_cohorts_are_permitted(self) -> None:
        """The cohorts this audit actually uses must all pass."""
        assert_no_sensitive_cohorts(
            ["pincode_tier", "order_value_band", "customer_history_band", "payment_method"]
        )

    def test_the_token_list_covers_the_named_prohibitions(self) -> None:
        """The specification names these explicitly; they may not quietly disappear."""
        for required in ("gender", "religion", "caste"):
            assert required in SENSITIVE_TOKENS


class TestWilsonInterval:
    def test_an_empty_group_yields_complete_ignorance(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_the_interval_always_lies_inside_the_unit_range(self) -> None:
        """The normal approximation fails here; Wilson is chosen for exactly this."""
        for successes, total in [(0, 5), (5, 5), (1, 3), (0, 1000), (1000, 1000)]:
            low, high = wilson_interval(successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_the_interval_contains_the_point_estimate_away_from_the_edges(self) -> None:
        low, high = wilson_interval(30, 100)
        assert low < 0.30 < high

    def test_more_data_narrows_the_interval(self) -> None:
        narrow = wilson_interval(200, 1000)
        wide = wilson_interval(2, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_a_higher_confidence_level_widens_the_interval(self) -> None:
        assert (
            wilson_interval(30, 100, confidence=0.99)[1]
            - wilson_interval(30, 100, confidence=0.99)[0]
        ) > (
            wilson_interval(30, 100, confidence=0.90)[1]
            - wilson_interval(30, 100, confidence=0.90)[0]
        )


class TestShrinkage:
    def test_an_empty_group_falls_back_to_the_prior(self) -> None:
        assert shrink_towards(0.9, n=0, prior=0.2, strength=50) == 0.2

    def test_a_thin_group_is_pulled_hard_towards_the_prior(self) -> None:
        shrunk = shrink_towards(1.0, n=5, prior=0.2, strength=50)
        assert shrunk < 0.3

    def test_a_large_group_is_left_almost_alone(self) -> None:
        shrunk = shrink_towards(0.8, n=100_000, prior=0.2, strength=50)
        assert shrunk == pytest.approx(0.8, abs=0.001)


class TestCohortBreakdown:
    @pytest.fixture
    def book(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """A deterministic book with a known answer.

        Group A: 10 orders, 4 flagged, 3 of those are positives.
        Group B: 10 orders, 2 flagged, 0 of those are positives.
        """
        groups = ["A"] * 10 + ["B"] * 10
        a_scores = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        b_scores = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        a_labels = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        b_labels = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        scores = [*a_scores, *b_scores]
        labels = [*a_labels, *b_labels]
        return (
            pd.DataFrame({"cohort": groups}),
            np.array(labels),
            np.array(scores),
        )

    def test_counts_and_rates_are_exact(
        self, book: tuple[pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        frame, labels, scores = book
        rows = {
            row.group: row
            for row in cohort_breakdown(
                frame,
                labels,
                scores,
                threshold=0.5,
                cohort_column="cohort",
                min_support=1,
                min_flagged=1,
            )
        }

        a = rows["A"]
        assert a.n_orders == 10
        assert a.n_flagged == 4
        assert a.n_positives == 4
        assert a.flag_rate == pytest.approx(0.4)
        assert a.rto_rate == pytest.approx(0.4)
        assert a.precision == pytest.approx(3 / 4)
        assert a.recall == pytest.approx(3 / 4)

        b = rows["B"]
        assert b.n_flagged == 2
        assert b.precision == pytest.approx(0.0)
        assert b.recall == pytest.approx(0.0)

    def test_a_probability_exactly_at_the_threshold_is_flagged(self) -> None:
        """`>=`, matching `confusion_at_threshold` and the serving path.

        If this used `>`, an order would be flagged in production and counted as
        unflagged in the audit, and the two would disagree about the same order.
        """
        frame = pd.DataFrame({"cohort": ["A"] * 4})
        rows = cohort_breakdown(
            frame,
            np.array([1, 1, 0, 0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            threshold=0.5,
            cohort_column="cohort",
            min_support=1,
            min_flagged=1,
        )
        assert rows[0].n_flagged == 4

    def test_precision_is_none_rather_than_zero_when_nothing_is_flagged(self) -> None:
        """Zero would claim a measurement; there is no denominator."""
        frame = pd.DataFrame({"cohort": ["A"] * 5})
        rows = cohort_breakdown(
            frame,
            np.array([1, 0, 0, 0, 0]),
            np.zeros(5),
            threshold=0.5,
            cohort_column="cohort",
            min_support=1,
        )
        assert rows[0].precision is None
        assert rows[0].flag_rate == 0.0

    def test_recall_is_none_rather_than_zero_when_there_are_no_positives(self) -> None:
        frame = pd.DataFrame({"cohort": ["A"] * 5})
        rows = cohort_breakdown(
            frame,
            np.zeros(5, dtype=int),
            np.ones(5),
            threshold=0.5,
            cohort_column="cohort",
            min_support=1,
            min_flagged=1,
        )
        assert rows[0].recall is None

    def test_misaligned_inputs_are_refused(self) -> None:
        """A silent misalignment attributes one group's outcomes to another."""
        frame = pd.DataFrame({"cohort": ["A"] * 10})
        with pytest.raises(ValueError, match="misaligned"):
            cohort_breakdown(
                frame, np.zeros(5, dtype=int), np.zeros(5), threshold=0.5, cohort_column="cohort"
            )

    def test_an_absent_cohort_column_is_refused(self) -> None:
        frame = pd.DataFrame({"cohort": ["A"] * 5})
        with pytest.raises(KeyError):
            cohort_breakdown(
                frame,
                np.zeros(5, dtype=int),
                np.zeros(5),
                threshold=0.5,
                cohort_column="not_here",
            )

    def test_net_rupees_are_computed_only_when_economics_are_supplied(
        self, book: tuple[pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        frame, labels, scores = book
        without = cohort_breakdown(
            frame,
            labels,
            scores,
            threshold=0.5,
            cohort_column="cohort",
            min_support=1,
            min_flagged=1,
        )
        assert all(row.net_inr_per_1000 is None for row in without)

        with_economics = cohort_breakdown(
            frame,
            labels,
            scores,
            threshold=0.5,
            cohort_column="cohort",
            min_support=1,
            min_flagged=1,
            cost_false_positive_inr=100.0,
            saving_true_positive_inr=200.0,
        )
        rows = {row.group: row for row in with_economics}
        # Group A: 3 true positives at +200, 1 false positive at -100, over 10 orders.
        assert rows["A"].net_inr_per_1000 == pytest.approx((3 * 200 - 1 * 100) / 10 * 1000)


class TestSmallSampleHandling:
    def test_a_thin_group_is_reported_but_marked_insufficient(self) -> None:
        """Suppressing thin groups would hide what an audit exists to look at."""
        frame = pd.DataFrame({"cohort": ["big"] * 500 + ["tiny"] * 12})
        scores = np.concatenate([np.full(500, 0.9), np.full(12, 0.9)])
        labels = np.concatenate(
            [
                np.ones(250, dtype=int),
                np.zeros(250, dtype=int),
                np.ones(6, dtype=int),
                np.zeros(6, dtype=int),
            ]
        )

        rows = {
            row.group: row
            for row in cohort_breakdown(
                frame, labels, scores, threshold=0.5, cohort_column="cohort", min_support=100
            )
        }

        assert rows["tiny"].n_orders == 12
        assert rows["tiny"].sufficient is False
        assert "below the minimum support" in rows["tiny"].insufficient_reason
        assert rows["big"].sufficient is True

    def test_a_large_group_with_few_flags_is_still_insufficient_for_precision(self) -> None:
        """Precision's denominator is the flag count, not the group size."""
        n = 400
        scores = np.concatenate([np.full(5, 0.9), np.full(n - 5, 0.1)])
        labels = np.zeros(n, dtype=int)
        labels[0] = 1
        frame = pd.DataFrame({"cohort": ["A"] * n})

        rows = cohort_breakdown(
            frame, labels, scores, threshold=0.5, cohort_column="cohort", min_support=100
        )
        assert rows[0].n_orders == n
        assert rows[0].n_flagged == 5
        assert rows[0].sufficient is False
        assert str(DEFAULT_MIN_FLAGGED) in rows[0].insufficient_reason

    def test_thin_groups_cannot_fire_the_disparity_trigger(self, config: FairnessConfig) -> None:
        """A precision computed on nine flagged orders must not raise an alarm.

        The tiny group here is flagged constantly and is always wrong - exactly
        the shape that would trip the review if it were allowed to count.
        """
        big = 800
        tiny = 15
        frame = pd.DataFrame({"pincode_tier": ["tier_1"] * big + ["tier_3"] * tiny})
        scores = np.concatenate(
            [
                np.concatenate([np.full(200, 0.9), np.full(big - 200, 0.1)]),
                np.full(tiny, 0.99),
            ]
        )
        labels = np.concatenate(
            [
                np.concatenate([np.ones(150, dtype=int), np.zeros(big - 150, dtype=int)]),
                np.zeros(tiny, dtype=int),
            ]
        )

        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)
        assert audit.triggered is False
        assert any("tier_3" in entry for entry in audit.groups_below_support)

    def test_thin_groups_cannot_suppress_a_trigger_either(self, config: FairnessConfig) -> None:
        """Exclusion works in both directions.

        A thin group with a very low flag rate would otherwise sit in the
        denominator of the ratio and inflate it, or sit in the numerator of the
        precision comparison and hide a real drop.
        """
        frame = pd.DataFrame({"pincode_tier": ["tier_1"] * 500 + ["tier_9"] * 5})
        scores = np.concatenate([np.full(500, 0.9), np.zeros(5)])
        labels = np.concatenate([np.ones(250, dtype=int), np.zeros(255, dtype=int)])

        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)
        # Only one group has support, so there is no comparison to make and the
        # audit says so rather than dividing by the thin group's zero flag rate.
        assert audit.max_flag_rate_ratio == 0.0
        assert audit.triggered is False


class TestDisparityTrigger:
    def _book(self, *, high_precision: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Two groups of 400. tier_3 is flagged 4x as often as tier_1."""
        rng = np.random.default_rng(3)
        n = 400
        tier1_flagged, tier3_flagged = 40, 160

        scores = np.concatenate(
            [
                np.full(tier1_flagged, 0.9),
                np.full(n - tier1_flagged, 0.1),
                np.full(tier3_flagged, 0.9),
                np.full(n - tier3_flagged, 0.1),
            ]
        )
        tier1_hits = int(tier1_flagged * 0.55)
        tier3_hits = int(tier3_flagged * high_precision)
        labels = np.concatenate(
            [
                np.ones(tier1_hits, dtype=int),
                np.zeros(tier1_flagged - tier1_hits, dtype=int),
                rng.integers(0, 2, n - tier1_flagged),
                np.ones(tier3_hits, dtype=int),
                np.zeros(tier3_flagged - tier3_hits, dtype=int),
                rng.integers(0, 2, n - tier3_flagged),
            ]
        )
        frame = pd.DataFrame({"pincode_tier": ["tier_1"] * n + ["tier_3"] * n})
        return frame, labels, scores

    def test_a_higher_flag_rate_alone_does_not_trigger(self, config: FairnessConfig) -> None:
        """A group that returns more parcels should be flagged more often.

        Forcing equal flag rates would make the system worse at its job while
        looking fairer, which is why the trigger is a conjunction.
        """
        frame, labels, scores = self._book(high_precision=0.60)
        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)
        assert audit.max_flag_rate_ratio > 1.5
        assert audit.triggered is False
        assert "does not trip" in audit.narrative

    def test_more_flags_and_worse_precision_together_do_trigger(
        self, config: FairnessConfig
    ) -> None:
        frame, labels, scores = self._book(high_precision=0.20)
        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)
        assert audit.max_flag_rate_ratio > 1.5
        assert audit.worst_precision_drop > 0.10
        assert audit.triggered is True
        assert "cost transferred without justification" in audit.narrative

    def test_a_near_miss_is_reported_rather_than_read_as_a_clean_pass(self) -> None:
        """A review reported only as a boolean loses "nowhere near" vs "just under".

        The second is the one that should change what happens at the next
        retrain, so the margin is stated in the narrative.
        """
        config = FairnessConfig(
            group_by=["pincode_tier"],
            report=["flag_rate"],
            disparity_review_trigger={"flag_rate_ratio_above": 1.5, "precision_drop_below": 0.10},
        )
        # Precision drops by ~0.08 against a 0.10 trigger: passes, but barely.
        frame, labels, scores = self._book(high_precision=0.47)
        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)

        assert audit.triggered is False
        assert "of the way to the precision-drop trigger" in audit.narrative
        assert "worth re-checking after the next retrain" in audit.narrative

    def test_a_comfortable_pass_is_not_dressed_up_as_a_near_miss(self) -> None:
        """The margin note must not fire when the audit was nowhere near tripping."""
        config = FairnessConfig(
            group_by=["pincode_tier"],
            report=["flag_rate"],
            disparity_review_trigger={"flag_rate_ratio_above": 1.5, "precision_drop_below": 0.10},
        )
        frame, labels, scores = self._book(high_precision=0.60)
        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)

        assert audit.worst_precision_drop < 0.05
        assert "of the way to the precision-drop trigger" not in audit.narrative

    def test_the_compared_pair_is_named_so_it_can_be_checked(self, config: FairnessConfig) -> None:
        frame, labels, scores = self._book(high_precision=0.60)
        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)
        assert audit.most_flagged_group == "pincode_tier=tier_3"
        assert audit.least_flagged_group == "pincode_tier=tier_1"

    def test_groups_are_never_compared_across_cohorts(self) -> None:
        """Two cohorts are two partitions of the same orders.

        A ratio between a pincode tier and an order-value band would be an
        artefact of the partitioning, not a disparity.
        """
        config = FairnessConfig(
            group_by=["pincode_tier", "order_value_band"],
            report=["flag_rate"],
            disparity_review_trigger={"flag_rate_ratio_above": 1.5, "precision_drop_below": 0.10},
        )
        n = 300
        frame = pd.DataFrame(
            {
                # One tier only: no within-cohort comparison is possible for it.
                "pincode_tier": ["tier_1"] * (2 * n),
                "order_value_band": ["v1"] * n + ["v2"] * n,
            }
        )
        # Both bands are flagged enough to be usable, v1 far more often than v2,
        # so the value-band cohort is the only one that yields a ratio.
        scores = np.concatenate(
            [
                np.full(200, 0.9),
                np.full(n - 200, 0.1),
                np.full(50, 0.9),
                np.full(n - 50, 0.1),
            ]
        )
        labels = np.ones(2 * n, dtype=int)

        audit = fairness_audit(frame, labels, scores, threshold=0.5, config=config, min_support=100)
        # The only ratio reported comes from the value band, never tier vs band.
        assert audit.most_flagged_group.startswith("order_value_band=")


class TestBanding:
    def test_quantile_bands_are_ordered_and_labelled_with_their_range(self) -> None:
        values = pd.Series(range(1000))
        bands = band_column(values, n_bands=4, prefix="v")
        assert bands.nunique() == 4
        assert all(label.startswith("v") for label in bands.unique())
        assert any("[" in label for label in bands.unique())

    def test_a_constant_column_does_not_crash_or_invent_bands(self) -> None:
        """Fewer bands than requested is correct: the data cannot support more."""
        bands = band_column(pd.Series([5.0] * 100), n_bands=4)
        assert bands.nunique() == 1

    def test_history_bands_give_new_customers_their_own_row(self) -> None:
        """A first-time customer is a different object from a second-time one."""
        bands = history_band(pd.Series([0, 1, 2, 3, 9, 10, 500]))
        assert bands.iloc[0] == "new (0 prior)"
        assert bands.iloc[1] == "light (1-2 prior)"
        assert bands.iloc[3] == "regular (3-9 prior)"
        assert bands.iloc[6] == "frequent (10+ prior)"

    def test_missing_history_counts_as_new_rather_than_unknown(self) -> None:
        bands = history_band(pd.Series([None, 0]))
        assert bands.iloc[0] == "new (0 prior)"


def test_the_audit_is_deterministic() -> None:
    """Same inputs, same audit. A fairness result nobody can reproduce is an anecdote."""
    config = FairnessConfig(
        group_by=["pincode_tier"],
        report=["flag_rate"],
        disparity_review_trigger={"flag_rate_ratio_above": 1.5, "precision_drop_below": 0.10},
    )
    rng = np.random.default_rng(11)
    n = 900
    frame = pd.DataFrame({"pincode_tier": rng.choice(["tier_1", "tier_2", "tier_3"], n)})
    scores = rng.random(n)
    labels = (rng.random(n) < scores).astype(int)

    first = fairness_audit(frame, labels, scores, threshold=0.4, config=config)
    second = fairness_audit(frame, labels, scores, threshold=0.4, config=config)
    assert first.model_dump() == second.model_dump()
