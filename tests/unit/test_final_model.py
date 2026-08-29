"""The final model: what was selected, on what data, and what it may not touch.

The tests that matter most here are the negative ones. A selection pipeline that
quietly reads the sealed set produces better-looking numbers and no error, so the
only way to know it did not is to change the test labels and prove nothing moved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rto_sentinel.configuration.schemas import (
    CostModelConfig,
    FinalModelConfig,
    ModelCardConfig,
)
from rto_sentinel.contracts.final import SelectionManifest
from rto_sentinel.eval import DishonestReportError, render_model_card, write_comparison_csv
from rto_sentinel.features.dataset import ModelingDataset, TestSetAccessError
from rto_sentinel.models.calibrated import CalibratedModel
from rto_sentinel.models.experiment import cost_inputs_from_profile
from rto_sentinel.models.final import (
    GuardrailViolation,
    build_final_model,
    evaluate_final_model,
    load_manifest,
    save_evaluation,
    save_manifest,
)

SEED = 4242


@pytest.fixture(scope="module")
def final_model(
    ladder_dataset: ModelingDataset,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
):
    """One full selection run, shared. Bootstrap kept small to stay quick."""
    return build_final_model(
        ladder_dataset,
        final_config=final_config,
        cost_config=cost_config,
        seed=SEED,
        bootstrap_iterations=30,
    )


@pytest.fixture(scope="module")
def validation_evaluation(
    final_model, ladder_dataset: ModelingDataset, cost_config: CostModelConfig
):
    evaluation, calibrated, raw = evaluate_final_model(
        final_model.model,
        ladder_dataset.validation,
        manifest=final_model.manifest,
        cost_inputs=cost_inputs_from_profile(
            cost_config.profiles[final_model.manifest.cost_profile]
        ),
        bootstrap_iterations=30,
    )
    return evaluation, calibrated, raw


# ---------------------------------------------------------------------------
# the sealed set stays sealed
# ---------------------------------------------------------------------------


def test_selection_never_opens_the_test_split(final_model, ladder_dataset: ModelingDataset) -> None:
    assert ladder_dataset.test_is_sealed
    with pytest.raises(TestSetAccessError):
        _ = ladder_dataset.test


def test_selection_is_unaffected_by_the_test_labels(
    ladder_dataset: ModelingDataset,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
) -> None:
    """Flip every test label and prove nothing about the selection moves.

    The strongest available statement of "no test labels were used during
    fitting": if any step had read them, inverting them would change the chosen
    candidate, the chosen calibrator, or the scores - and the manifest hash is
    over exactly those decisions.
    """
    kwargs = {
        "final_config": final_config,
        "cost_config": cost_config,
        "seed": SEED,
        "bootstrap_iterations": 0,
    }
    before = build_final_model(ladder_dataset, **kwargs)

    corrupted = replace(
        ladder_dataset,
        labels=ladder_dataset.labels.where(
            ladder_dataset.splits != "test", ~ladder_dataset.labels.astype(bool)
        ).astype(ladder_dataset.labels.dtype),
    )
    assert not corrupted.labels.equals(ladder_dataset.labels), "the corruption must be real"

    after = build_final_model(corrupted, **kwargs)

    assert after.manifest.manifest_id == before.manifest.manifest_id
    assert after.manifest.chosen_candidate == before.manifest.chosen_candidate
    assert after.manifest.calibration_method == before.manifest.calibration_method
    assert np.array_equal(after.validation_scores, before.validation_scores)


def test_a_test_evaluation_requires_a_written_reason(
    final_model, ladder_dataset: ModelingDataset, cost_config: CostModelConfig
) -> None:
    ladder_dataset.unseal_test(reason="unit test of the reason requirement")
    try:
        with pytest.raises(ValueError, match="written reason"):
            evaluate_final_model(
                final_model.model,
                ladder_dataset.test,
                manifest=final_model.manifest,
                cost_inputs=cost_inputs_from_profile(
                    cost_config.profiles[final_model.manifest.cost_profile]
                ),
                bootstrap_iterations=0,
            )
    finally:
        # Re-seal, so the shared dataset fixture stays sealed for other tests.
        ladder_dataset._test_unsealed_reason = None


def test_the_threshold_comes_from_costs_not_from_labels(final_model) -> None:
    manifest = final_model.manifest
    assert manifest.threshold_source.startswith("derived from cost inputs")
    assert 0.0 < manifest.threshold < 1.0
    assert manifest.threshold != pytest.approx(0.5)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_every_candidate_is_measured_and_exactly_one_wins(
    final_model, final_config: FinalModelConfig
) -> None:
    candidates = final_model.manifest.candidates
    assert len(candidates) == len(final_config.search.candidates)
    assert sum(candidate.selected for candidate in candidates) == 1
    for candidate in candidates:
        assert 0.0 <= candidate.validation_pr_auc <= 1.0
        assert candidate.train_pr_auc is not None
        assert candidate.params["random_state"] == SEED


def test_the_losing_candidates_are_kept(final_model) -> None:
    """A selection record showing only the winner is an advertisement."""
    names = {candidate.name for candidate in final_model.manifest.candidates}
    assert len(names) > 1
    assert any(not candidate.selected for candidate in final_model.manifest.candidates)


def test_the_tie_rule_prefers_the_smaller_model_within_noise() -> None:
    """Two candidates a whisker apart: the smaller one wins."""
    from rto_sentinel.contracts.final import CandidateResult
    from rto_sentinel.models.final import _apply_tie_rule

    big = CandidateResult(
        name="big",
        params={"n_estimators": 600, "num_leaves": 31},
        validation_pr_auc=0.500,
        validation_pr_auc_ci=(0.450, 0.550),  # SE ~ 0.0255
        train_duration_seconds=1.0,
    )
    small = CandidateResult(
        name="small",
        params={"n_estimators": 100, "num_leaves": 7},
        validation_pr_auc=0.495,
        validation_pr_auc_ci=(0.445, 0.545),
        train_duration_seconds=1.0,
    )
    index = _apply_tie_rule([big, small], rule="one_standard_error_then_smallest")
    assert [big, small][index].name == "small"

    # Outside the interval it is not a tie, and the better model wins.
    far = small.model_copy(update={"validation_pr_auc": 0.300})
    index = _apply_tie_rule([big, far], rule="one_standard_error_then_smallest")
    assert [big, far][index].name == "big"


def test_the_tie_rule_needs_an_interval_and_says_so_by_falling_back() -> None:
    from rto_sentinel.contracts.final import CandidateResult
    from rto_sentinel.models.final import _apply_tie_rule

    big = CandidateResult(
        name="big",
        params={"n_estimators": 600, "num_leaves": 31},
        validation_pr_auc=0.500,
        train_duration_seconds=1.0,
    )
    small = CandidateResult(
        name="small",
        params={"n_estimators": 100, "num_leaves": 7},
        validation_pr_auc=0.495,
        train_duration_seconds=1.0,
    )
    index = _apply_tie_rule([big, small], rule="one_standard_error_then_smallest")
    assert [big, small][index].name == "big", "without an interval there is no tolerance to apply"


def test_selection_is_reproducible(
    ladder_dataset: ModelingDataset,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
) -> None:
    kwargs = {
        "final_config": final_config,
        "cost_config": cost_config,
        "seed": 11,
        "bootstrap_iterations": 0,
    }
    first = build_final_model(ladder_dataset, **kwargs)
    second = build_final_model(ladder_dataset, **kwargs)

    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.manifest.model_version == second.manifest.model_version
    assert np.array_equal(first.validation_scores, second.validation_scores)


def test_guardrails_refuse_an_unshippable_model(
    ladder_dataset: ModelingDataset,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
) -> None:
    """The guardrail must be able to fire, or it is decoration."""
    impossible = final_config.model_copy(
        update={
            "guardrails": final_config.guardrails.model_copy(
                update={"min_pr_auc_over_base_rate": 0.95}
            )
        }
    )
    with pytest.raises(GuardrailViolation, match="Refusing to freeze"):
        build_final_model(
            ladder_dataset,
            final_config=impossible,
            cost_config=cost_config,
            seed=SEED,
            bootstrap_iterations=0,
        )


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def test_the_manifest_pins_every_version(final_model, ladder_dataset: ModelingDataset) -> None:
    manifest, metadata = final_model.manifest, ladder_dataset.metadata
    assert manifest.generator_version == metadata.generator_version
    assert manifest.dataset_run_id == metadata.dataset_run_id
    assert manifest.feature_version == metadata.feature_version
    assert manifest.feature_fingerprint == metadata.feature_fingerprint
    assert manifest.config_fingerprint == metadata.config_fingerprint
    assert manifest.seed == SEED
    assert manifest.model_version
    assert manifest.calibration_fitted_on == "validation"


def test_the_manifest_id_covers_the_decisions(final_model) -> None:
    """Change a decision, get a different id. Change the clock, do not."""
    manifest = final_model.manifest
    from datetime import UTC, datetime

    same_decisions = manifest.model_copy(update={"frozen_at": datetime(2020, 1, 1, tzinfo=UTC)})
    assert same_decisions.manifest_id == manifest.manifest_id

    different = manifest.model_copy(update={"calibration_method": "none"})
    assert different.manifest_id != manifest.manifest_id


def test_the_manifest_round_trips(final_model, tmp_path: Path) -> None:
    path = save_manifest(final_model.manifest, tmp_path)
    assert path.is_file()

    reloaded = load_manifest(tmp_path, final_model.manifest.dataset_run_id)
    assert reloaded.manifest_id == final_model.manifest.manifest_id
    assert reloaded.chosen_params == final_model.manifest.chosen_params
    assert len(reloaded.candidates) == len(final_model.manifest.candidates)


def test_a_missing_manifest_is_an_explicit_refusal(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="frozen selection manifest"):
        load_manifest(tmp_path, "no-such-dataset")


def test_the_card_declares_the_calibration(final_model) -> None:
    card = final_model.card
    assert card.calibration_method == final_model.manifest.calibration_method
    assert card.calibration_fitted_on == "validation"
    assert card.feature_fingerprint
    assert card.dataset_run_id


# ---------------------------------------------------------------------------
# the calibrated model
# ---------------------------------------------------------------------------


def test_the_model_produces_probabilities(final_model, ladder_dataset: ModelingDataset) -> None:
    scores = final_model.model.predict_proba(
        ladder_dataset.validation.x, ladder_dataset.validation.context
    )
    assert scores.shape == (ladder_dataset.validation.n_rows,)
    assert np.all(np.isfinite(scores))
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_the_raw_score_is_still_reachable(final_model, ladder_dataset: ModelingDataset) -> None:
    """Before/after is a comparative claim, so both halves must be available."""
    view = ladder_dataset.validation
    raw = final_model.model.predict_raw(view.x, view.context)
    calibrated = final_model.model.predict_proba(view.x, view.context)
    assert raw.shape == calibrated.shape
    if final_model.manifest.calibration_method != "none":
        assert not np.array_equal(raw, calibrated)


def test_the_artefact_round_trips_with_its_calibrator(
    ladder_dataset: ModelingDataset,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
    tmp_path: Path,
) -> None:
    """The shipped file must score identically to the model that was measured."""
    built = build_final_model(
        ladder_dataset,
        final_config=final_config,
        cost_config=cost_config,
        seed=SEED,
        artifact_root=tmp_path,
        bootstrap_iterations=0,
    )
    assert built.artifact_path is not None
    loaded, card = CalibratedModel.load(built.artifact_path)

    view = ladder_dataset.validation
    assert np.array_equal(
        loaded.predict_proba(view.x, view.context),
        built.model.predict_proba(view.x, view.context),
    )
    assert loaded.calibration_method == built.model.calibration_method
    assert card.calibration_method == built.manifest.calibration_method


# ---------------------------------------------------------------------------
# evaluation and reporting
# ---------------------------------------------------------------------------


def test_the_evaluation_reports_both_calibrated_and_raw(validation_evaluation) -> None:
    evaluation, _, _ = validation_evaluation
    assert evaluation.ranking.pr_auc.is_defined
    assert evaluation.ranking.pr_auc.n_bootstrap > 0
    assert evaluation.calibration.reliability_bins
    assert evaluation.uncalibrated_calibration.reliability_bins
    assert 0.0 <= evaluation.uncalibrated_pr_auc <= 1.0


def test_calibration_does_not_change_the_ranking(validation_evaluation) -> None:
    """Every calibrator here is monotone, so PR-AUC must survive it unchanged."""
    evaluation, _, _ = validation_evaluation
    assert evaluation.ranking.pr_auc.value == pytest.approx(
        evaluation.uncalibrated_pr_auc, abs=0.02
    )


def test_the_operating_point_is_the_derived_threshold(validation_evaluation, final_model) -> None:
    evaluation, _, _ = validation_evaluation
    point = evaluation.operating_point
    assert point.threshold == pytest.approx(final_model.manifest.threshold)
    assert point.threshold_source == "cost-derived"
    assert (
        point.true_positives + point.false_positives + point.false_negatives + point.true_negatives
        == evaluation.evaluation_summary.n_rows
    )


def test_the_evaluation_round_trips(validation_evaluation, tmp_path: Path) -> None:
    evaluation, _, _ = validation_evaluation
    path = save_evaluation(evaluation, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "NaN" not in path.read_text(encoding="utf-8")
    assert payload["manifest_id"] == evaluation.manifest_id
    assert payload["calibration_method"] == evaluation.calibration_method
    assert "not real-world ground truth" in payload["data_provenance"]


def test_the_model_card_carries_its_numbers_and_its_disclaimer(
    model_card_config: ModelCardConfig, final_model, validation_evaluation
) -> None:
    evaluation, _, _ = validation_evaluation
    card = render_model_card(model_card_config, final_model.manifest, {"validation": evaluation})

    assert "not real-world ground truth" in card
    assert "Not intended for" in card
    assert "Fairness limitations" in card
    assert "Distribution-shift limitations" in card
    assert f"{evaluation.ranking.pr_auc.value:.3f}" in card
    assert final_model.manifest.manifest_id in card
    assert "SELECTION-CONTAMINATED" in card or "optimistic" in card


def test_a_card_without_an_evaluation_is_refused(
    model_card_config: ModelCardConfig, final_model
) -> None:
    with pytest.raises(DishonestReportError, match="no evaluation behind it"):
        render_model_card(model_card_config, final_model.manifest, {})


def test_the_comparison_csv_lists_the_final_model(validation_evaluation, tmp_path: Path) -> None:
    evaluation, _, _ = validation_evaluation
    path = write_comparison_csv({"validation": evaluation}, None, tmp_path / "comparison.csv")

    text = path.read_text(encoding="utf-8")
    assert "final model" in text
    assert evaluation.model_name in text


def test_manifest_json_is_valid_and_complete(final_model, tmp_path: Path) -> None:
    path = save_manifest(final_model.manifest, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    for key in (
        "chosen_candidate",
        "calibration_method",
        "threshold",
        "model_version",
        "feature_fingerprint",
        "config_fingerprint",
        "seed",
        "candidates",
        "calibration_candidates",
    ):
        assert key in payload, key
    assert SelectionManifest.model_validate(payload).manifest_id == final_model.manifest.manifest_id
