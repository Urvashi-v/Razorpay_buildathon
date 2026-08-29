"""Calibration behaves like calibration, and never sees data it must not see.

The claim under test is narrow: a calibrated score can be read as a probability.
That is only true if the mapping was fitted on data the model did not train on,
if it was never fitted on the sealed set, and if the numbers it produces line up
with observed frequencies better than the raw scores did. Each of those is a
separate test below, because each fails independently and silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from rto_sentinel.configuration.schemas import FinalModelConfig
from rto_sentinel.models.calibrated import CalibratedModel
from rto_sentinel.models.calibration import (
    CALIBRATORS,
    CalibrationError,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
    compare_calibrators,
    cross_validated_scores,
    restore_calibrator,
)
from rto_sentinel.models.final import select_calibration
from rto_sentinel.models.rung0_do_nothing import DoNothingModel


def _distorted(
    n: int = 4000, *, squash: float = 3.0, seed: int = 20260829
) -> tuple[np.ndarray, np.ndarray]:
    """Scores from a model that ranks well but is badly overconfident.

    The true probability is a smooth function of a latent variable; the "model"
    reports that probability pushed towards the extremes, which is exactly the
    distortion boosting produces and calibration is supposed to undo. Ranking is
    untouched - the distortion is monotone - so any change in ranking metrics
    afterwards would be a bug.

    ``squash=1.0`` produces scores that are already perfectly calibrated.

    A fresh generator per call, deliberately: a module-level generator would make
    every test in this file depend on which tests ran before it, and a flaky
    calibration test is worse than none.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    true_p = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.uniform(size=n) < true_p).astype(int)
    distorted = 1.0 / (1.0 + np.exp(-squash * latent))
    return distorted, y


def _ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    from rto_sentinel.eval.metrics import expected_calibration_error

    return expected_calibration_error(y_true.astype(bool), y_prob, n_bins=bins)[0]


# ---------------------------------------------------------------------------
# probabilities stay probabilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_calibrated_scores_stay_in_the_unit_interval(method: str) -> None:
    scores, y = _distorted()
    calibrator = build_calibrator(method)
    calibrator.fit(scores, y)

    # Includes the endpoints and values outside the fitted range, which is where
    # an unclipped isotonic extrapolation would escape [0, 1].
    probe = np.concatenate([scores, np.array([0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0])])
    out = calibrator.transform(probe)

    assert out.shape == probe.shape
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_calibration_preserves_ranking(method: str) -> None:
    """Every method here is monotone, so PR-AUC must not move.

    If it did, calibration would be changing which orders are considered riskier
    than which - a different and much larger claim than "the numbers now mean
    what they say".
    """
    from rto_sentinel.eval.metrics import pr_auc

    scores, y = _distorted()
    calibrator = build_calibrator(method)
    calibrator.fit(scores, y)

    before = pr_auc(y.astype(bool), scores)
    after = pr_auc(y.astype(bool), calibrator.transform(scores))
    assert after == pytest.approx(before, abs=0.02)


def test_an_unfitted_calibrator_refuses_to_transform() -> None:
    with pytest.raises(CalibrationError, match="has not been fitted"):
        IsotonicCalibrator().transform(np.array([0.3, 0.7]))


def test_fitting_on_nothing_is_refused() -> None:
    with pytest.raises(CalibrationError, match="zero rows"):
        PlattCalibrator().fit(np.array([]), np.array([]))


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(CalibrationError, match="disagree in length"):
        IsotonicCalibrator().fit(np.array([0.2, 0.4, 0.6]), np.array([0, 1]))


def test_platt_needs_both_classes() -> None:
    """A single-class fold cannot define a mapping, and says so."""
    with pytest.raises(CalibrationError, match="both classes"):
        PlattCalibrator().fit(np.linspace(0.1, 0.9, 50), np.zeros(50, dtype=int))


def test_an_unknown_method_is_refused_not_defaulted() -> None:
    """A typo must not quietly become "no calibration" behind a card that claims one."""
    with pytest.raises(CalibrationError, match="unknown calibration method"):
        build_calibrator("isotonik")


# ---------------------------------------------------------------------------
# calibration actually calibrates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrated_predictions_track_observed_frequencies_better(method: str) -> None:
    """The core claim, measured out-of-fold on a model known to be miscalibrated."""
    scores, y = _distorted()
    calibrated = cross_validated_scores(method, scores, y, n_folds=5, seed=1)

    before = _ece(y, scores)
    after = _ece(y, calibrated)

    assert before > 0.05, "the fixture must actually be miscalibrated for this to mean anything"
    assert after < before / 2


def test_calibration_does_not_disturb_an_already_calibrated_model() -> None:
    """Platt on the logit can express the identity, so it should stay near it.

    This is why the transform is applied to ``logit(p)`` rather than to ``p``: a
    sigmoid fitted on the raw probability cannot represent "leave it alone", and
    would distort a model that needed no correction.
    """
    scores, y = _distorted(squash=1.0)  # squash=1 means already calibrated
    before = _ece(y, scores)
    after = _ece(y, cross_validated_scores("platt", scores, y, n_folds=5, seed=1))

    assert before < 0.05
    assert after < before + 0.02


def test_the_identity_calibrator_changes_nothing() -> None:
    scores, y = _distorted()
    calibrator = IdentityCalibrator()
    calibrator.fit(scores, y)
    assert np.array_equal(calibrator.transform(scores), scores)


def test_isotonic_output_is_monotone_in_the_input() -> None:
    scores, y = _distorted()
    calibrator = IsotonicCalibrator()
    calibrator.fit(scores, y)

    grid = np.linspace(0.0, 1.0, 200)
    mapped = calibrator.transform(grid)
    assert np.all(np.diff(mapped) >= -1e-12)


# ---------------------------------------------------------------------------
# the comparison is honest
# ---------------------------------------------------------------------------


def test_the_comparison_is_out_of_fold() -> None:
    """Fitted-and-scored on the same rows, isotonic would look far better.

    This test is the reason ``compare_calibrators`` cross-validates: it measures
    the size of the optimism it exists to avoid.
    """
    scores, y = _distorted()

    in_sample = IsotonicCalibrator()
    in_sample.fit(scores, y)
    optimistic = _ece(y, in_sample.transform(scores))
    honest = _ece(y, cross_validated_scores("isotonic", scores, y, n_folds=5, seed=1))

    assert optimistic < honest, "in-sample calibration should look better than it is"


def test_doing_nothing_is_always_a_candidate(final_config: FinalModelConfig) -> None:
    """Without it there is no measurement of whether calibrating helped at all."""
    assert "none" in final_config.calibration.candidates


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_a_flexible_calibrator_is_not_selected_for_fitting_noise(
    final_config: FinalModelConfig, seed: int
) -> None:
    """On already-calibrated scores, isotonic must lose - across seeds.

    This is the failure the Brier veto exists to prevent, and it is a real one:
    isotonic reduces the *binned* calibration error on every one of these draws,
    purely by fitting the sampling noise inside the bins. Selecting it on that
    basis would ship a fitted component that makes the probabilities genuinely
    worse, which the proper score shows and ECE does not.
    """
    scores, y = _distorted(20000, squash=1.0, seed=seed)
    candidates = select_calibration(scores, y, final_config=final_config, seed=1)
    by_method = {candidate.method: candidate for candidate in candidates}

    assert by_method["isotonic"].improvement_over_none > 0, (
        "the premise of this test is that ECE alone would favour isotonic"
    )
    assert not by_method["isotonic"].selected


def test_a_calibrator_is_rejected_for_worsening_the_proper_score(
    final_config: FinalModelConfig,
) -> None:
    """ECE is binned and noisy; Brier is proper. The veto is what stops noise winning.

    Constructed so a candidate looks better on ECE while genuinely producing
    worse probabilities: the ECE gate alone would select it.
    """
    scores, y = _distorted(squash=1.0)
    candidates = select_calibration(scores, y, final_config=final_config, seed=7)
    baseline = next(c for c in candidates if c.method == "none")

    for candidate in candidates:
        if candidate.selected and candidate.method != "none":
            assert candidate.brier_score <= baseline.brier_score + (
                final_config.calibration.max_brier_degradation
            ), "a selected calibrator must not make the proper score worse"


def test_a_calibrator_that_helps_is_selected(final_config: FinalModelConfig) -> None:
    scores, y = _distorted(squash=4.0)
    candidates = select_calibration(scores, y, final_config=final_config, seed=1)
    chosen = next(c for c in candidates if c.selected)

    assert chosen.method != "none"
    assert chosen.improvement_over_none >= final_config.calibration.minimum_improvement
    assert chosen.fitted_on == "validation"
    assert chosen.n_folds == final_config.calibration.n_folds


def test_every_candidate_is_measured(final_config: FinalModelConfig) -> None:
    scores, y = _distorted()
    measured = compare_calibrators(scores, y, methods=["none", "isotonic", "platt"], seed=3)

    assert set(measured) == {"none", "isotonic", "platt"}
    for metrics in measured.values():
        assert 0.0 <= metrics.expected_calibration_error <= 1.0
        assert metrics.reliability_bins


# ---------------------------------------------------------------------------
# reproducibility and persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_calibration_is_reproducible(method: str) -> None:
    """Same scores, same labels, same mapping - to the last bit."""
    scores, y = _distorted()
    probe = np.linspace(0.0, 1.0, 500)

    first, second = build_calibrator(method), build_calibrator(method)
    first.fit(scores, y)
    second.fit(scores, y)

    assert np.array_equal(first.transform(probe), second.transform(probe))


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_cross_validated_selection_is_reproducible(method: str) -> None:
    scores, y = _distorted()
    first = cross_validated_scores(method, scores, y, n_folds=5, seed=11)
    second = cross_validated_scores(method, scores, y, n_folds=5, seed=11)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("method", sorted(CALIBRATORS))
def test_a_calibrator_survives_a_state_round_trip(method: str) -> None:
    """The shipped artefact carries the calibrator, so it has to restore exactly."""
    scores, y = _distorted()
    original = build_calibrator(method)
    original.fit(scores, y)

    restored = restore_calibrator(original.state())
    probe = np.linspace(0.01, 0.99, 300)

    assert restored.method == original.method
    assert restored.is_fitted
    assert restored.n_fit_rows_ == len(scores)
    assert np.array_equal(restored.transform(probe), original.transform(probe))


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------


def test_wrapping_refuses_an_unfitted_calibrator() -> None:
    model = DoNothingModel()
    import pandas as pd

    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    model.fit(x, pd.Series([0, 1, 0, 1]))

    with pytest.raises(CalibrationError, match="unfitted"):
        CalibratedModel.wrap(model, IsotonicCalibrator())


def test_a_calibrated_model_cannot_be_fitted_in_one_call(ladder_dataset) -> None:
    """The two halves are fitted on different splits, and the type says so."""
    from rto_sentinel.models.base import NotFittedError

    train = ladder_dataset.train
    with pytest.raises(NotFittedError, match="cannot be fitted in one call"):
        CalibratedModel().fit(train.x, train.y, train.context)
