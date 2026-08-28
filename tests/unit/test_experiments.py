"""Experiment tracking: every run records what produced it, and can be replayed.

The claim these tests defend is narrow and important: **no metric in this project
is written as a literal**. Every number in a report came from
``rto_sentinel.eval`` applied to real predictions on held-out data, and the
artefact carries enough provenance to say which data, which features, which
configuration and which seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rto_sentinel.configuration.schemas import CostModelConfig, EvaluationConfig, LadderConfig
from rto_sentinel.contracts.experiment import LadderResults
from rto_sentinel.eval import (
    DishonestReportError,
    check_rendered_report,
    comparison_table,
    render_markdown,
    strongest_rung,
)
from rto_sentinel.features.dataset import ModelingDataset
from rto_sentinel.models import run_ladder, save_results, scores_frame


@pytest.fixture(scope="module")
def ladder_run(
    ladder_dataset: ModelingDataset, ladder_config: LadderConfig, cost_config: CostModelConfig
):
    """One full ladder run, shared. Bootstrap kept small to stay quick."""
    return run_ladder(
        ladder_dataset,
        ladder_config=ladder_config,
        cost_config=cost_config,
        seed=4242,
        bootstrap_iterations=40,
    )


# ---------------------------------------------------------------------------
# the ladder runs, uniformly
# ---------------------------------------------------------------------------


def test_every_enabled_rung_is_trained_and_recorded(
    ladder_run, ladder_config: LadderConfig
) -> None:
    results, trained = ladder_run
    expected = {rung.name for rung in ladder_config.rungs if rung.enabled}

    assert set(trained) == expected
    assert {record.model_name for record in results.records} == expected


def test_every_rung_is_scored_at_the_same_threshold(ladder_run) -> None:
    """SPEC section 05: identical footing, or the comparison means nothing."""
    results, _ = ladder_run
    for record in results.records:
        assert record.economics is not None
        assert record.economics.threshold == pytest.approx(results.threshold)
        assert record.threshold_metrics[0].threshold == pytest.approx(results.threshold)
        assert record.evaluated_split == results.evaluated_split


def test_the_threshold_is_derived_not_fitted(ladder_run) -> None:
    """It must come from cost inputs alone, never from the evaluation labels."""
    results, _ = ladder_run
    assert results.threshold_source.startswith("derived from cost inputs")
    assert 0.0 < results.threshold < 1.0
    assert results.threshold != pytest.approx(0.5)


def test_evaluation_happens_on_validation_leaving_the_test_split_sealed(
    ladder_run, ladder_dataset: ModelingDataset
) -> None:
    results, _ = ladder_run
    assert results.evaluated_split == "validation"
    assert ladder_dataset.test_is_sealed


# ---------------------------------------------------------------------------
# what a record contains
# ---------------------------------------------------------------------------


def test_a_record_pins_all_six_versions(ladder_run, ladder_dataset: ModelingDataset) -> None:
    """A result is reproducible only if every input that could change it is named."""
    results, _ = ladder_run
    metadata = ladder_dataset.metadata

    for record in results.records:
        assert record.generator_version == metadata.generator_version
        assert record.dataset_run_id == metadata.dataset_run_id
        assert record.feature_version == metadata.feature_version
        assert record.feature_fingerprint == metadata.feature_fingerprint
        assert record.config_fingerprint == metadata.config_fingerprint
        assert record.seed == 4242
        assert record.model_version


def test_a_record_captures_the_hyperparameters_actually_used(ladder_run) -> None:
    results, _ = ladder_run
    lightgbm = results.by_name("lightgbm")

    assert lightgbm.hyperparameters["random_state"] == 4242
    assert "n_estimators" in lightgbm.hyperparameters
    assert lightgbm.train_duration_seconds >= 0
    assert lightgbm.trained_at is not None


def test_a_record_describes_both_splits(ladder_run, ladder_dataset: ModelingDataset) -> None:
    results, _ = ladder_run
    record = results.records[0]

    assert record.train_summary.n_rows == ladder_dataset.train.n_rows
    assert record.evaluation_summary.n_rows == ladder_dataset.validation.n_rows
    assert record.train_summary.last_day < record.evaluation_summary.first_day, (
        "training must end before evaluation begins"
    )


def test_every_record_declares_itself_uncalibrated(ladder_run) -> None:
    """Phase 4 ships no calibrated model, and every artefact says so."""
    results, trained = ladder_run
    for record in results.records:
        assert record.is_calibrated is False
    for run in trained.values():
        assert run.card.calibration_method is None


def test_the_train_validation_gap_is_recorded(ladder_run) -> None:
    """So a memorising model is visible in the artefact, not just to whoever checks.

    Only asserted to be *present and finite* - the size of the gap is a finding to
    report, not a requirement to pin.
    """
    results, _ = ladder_run
    lightgbm = results.by_name("lightgbm")

    assert lightgbm.train_pr_auc is not None
    assert lightgbm.overfit_gap is not None


def test_metrics_are_computed_not_defaulted(ladder_run) -> None:
    """Guards against a record that serialises cleanly while holding nothing."""
    results, _ = ladder_run
    for record in results.records:
        assert record.ranking.pr_auc.is_defined
        assert record.ranking.pr_auc.n_bootstrap > 0
        assert record.economics is not None
        assert record.economics.n_orders == record.evaluation_summary.n_rows
        assert record.threshold_metrics


def test_the_do_nothing_rung_anchors_the_comparison(ladder_run) -> None:
    """Its PR-AUC is the base rate and its net saving is exactly zero."""
    results, _ = ladder_run
    baseline = results.by_name("do_nothing")

    assert baseline.primary_metric == pytest.approx(
        baseline.evaluation_summary.positive_rate, abs=0.01
    )
    assert baseline.economics is not None
    assert baseline.economics.net_inr_saved_per_1000_orders.value == pytest.approx(0.0)
    assert baseline.economics.flag_rate == 0.0
    assert not baseline.ranking.roc_auc.is_defined


# ---------------------------------------------------------------------------
# artefacts on disk
# ---------------------------------------------------------------------------


def test_results_are_written_and_reload_identically(ladder_run, tmp_path: Path) -> None:
    """A reported number must be recoverable from the artefact, not only from a rerun."""
    results, _ = ladder_run
    path = save_results(results, tmp_path)

    assert path.is_file()
    reloaded = LadderResults.model_validate_json(path.read_text(encoding="utf-8"))

    assert reloaded.dataset_run_id == results.dataset_run_id
    assert reloaded.threshold == pytest.approx(results.threshold)
    assert len(reloaded.records) == len(results.records)
    for original, restored in zip(results.ordered, reloaded.ordered, strict=True):
        assert restored.model_name == original.model_name
        assert restored.primary_metric == pytest.approx(original.primary_metric)


def test_a_per_rung_artefact_is_written_for_each_model(ladder_run, tmp_path: Path) -> None:
    """So one rung can be inspected without parsing the whole ladder."""
    results, _ = ladder_run
    save_results(results, tmp_path)
    directory = tmp_path / "experiments" / results.dataset_run_id

    for record in results.records:
        path = directory / f"{record.model_name}__validation.json"
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model_name"] == record.model_name
        assert payload["seed"] == record.seed


def test_the_written_record_carries_the_provenance_statement(ladder_run, tmp_path: Path) -> None:
    """Anyone reading the artefact meets the synthetic-data caveat there."""
    results, _ = ladder_run
    save_results(results, tmp_path)
    path = tmp_path / "experiments" / results.dataset_run_id / "lightgbm__validation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "not real-world ground truth" in payload["data_provenance"]


def test_undefined_metrics_serialise_as_null_not_nan(ladder_run, tmp_path: Path) -> None:
    """JSON has no NaN. Rung 0's precision must round-trip as null."""
    results, _ = ladder_run
    save_results(results, tmp_path)
    path = tmp_path / "experiments" / results.dataset_run_id / "do_nothing__validation.json"

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)  # would raise on a bare NaN
    assert payload["threshold_metrics"][0]["precision"] is None


def test_scores_are_exported_for_every_rung(ladder_run, ladder_dataset: ModelingDataset) -> None:
    """Plots and later analysis work from the same numbers the metrics used."""
    _, trained = ladder_run
    frame = scores_frame(ladder_dataset, trained, "validation")

    assert len(frame) == ladder_dataset.validation.n_rows
    for name in trained:
        assert f"score__{name}" in frame.columns
        assert frame[f"score__{name}"].between(0.0, 1.0).all()


def test_model_artefacts_are_written_when_a_root_is_given(
    ladder_dataset: ModelingDataset,
    ladder_config: LadderConfig,
    cost_config: CostModelConfig,
    tmp_path: Path,
) -> None:
    _, trained = run_ladder(
        ladder_dataset,
        ladder_config=ladder_config,
        cost_config=cost_config,
        seed=1,
        artifact_root=tmp_path,
        bootstrap_iterations=0,
    )
    for name, run in trained.items():
        assert run.artifact_path is not None, name
        assert (run.artifact_path / "model.joblib").is_file()
        assert (run.artifact_path / "card.json").is_file()
        assert (run.artifact_path / "checksum.txt").is_file()


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_whole_ladder(
    ladder_dataset: ModelingDataset,
    ladder_config: LadderConfig,
    cost_config: CostModelConfig,
) -> None:
    """Every rung's headline metric, identical across two runs."""
    kwargs = {
        "ladder_config": ladder_config,
        "cost_config": cost_config,
        "seed": 31,
        "bootstrap_iterations": 0,
    }
    first, _ = run_ladder(ladder_dataset, **kwargs)
    second, _ = run_ladder(ladder_dataset, **kwargs)

    for a, b in zip(first.ordered, second.ordered, strict=True):
        assert a.model_name == b.model_name
        assert a.primary_metric == pytest.approx(b.primary_metric, abs=1e-12)
        assert a.model_version == b.model_version
        assert a.experiment_id == b.experiment_id


def test_the_model_version_tracks_the_hyperparameters(
    ladder_dataset: ModelingDataset,
    ladder_config: LadderConfig,
    cost_config: CostModelConfig,
) -> None:
    """Two different seeds must not collide on one artefact directory."""
    first, _ = run_ladder(
        ladder_dataset,
        ladder_config=ladder_config,
        cost_config=cost_config,
        seed=1,
        bootstrap_iterations=0,
    )
    second, _ = run_ladder(
        ladder_dataset,
        ladder_config=ladder_config,
        cost_config=cost_config,
        seed=2,
        bootstrap_iterations=0,
    )
    assert first.by_name("lightgbm").model_version != second.by_name("lightgbm").model_version


# ---------------------------------------------------------------------------
# reporting rules
# ---------------------------------------------------------------------------


def test_the_comparison_table_renders(ladder_run) -> None:
    results, _ = ladder_run
    table = comparison_table(results)

    assert "PR-AUC" in table
    assert "flag" in table
    for record in results.records:
        assert record.model_name in table
    # Undefined values appear as a dash, never as a zero.
    assert "-" in table


def test_the_markdown_report_leads_with_pr_auc_not_roc(
    ladder_run, evaluation_config: EvaluationConfig
) -> None:
    results, _ = ladder_run
    markdown = render_markdown(results, evaluation_config)

    assert markdown.index("PR-AUC") < markdown.index("ROC-AUC")
    assert "not a claim about production performance" in markdown
    assert "Every rung here is uncalibrated" in markdown


def test_the_report_refuses_a_point_estimate_without_an_interval(
    ladder_dataset: ModelingDataset,
    ladder_config: LadderConfig,
    cost_config: CostModelConfig,
    evaluation_config: EvaluationConfig,
) -> None:
    """The prohibition must be able to fire, or it is decoration."""
    results, _ = run_ladder(
        ladder_dataset,
        ladder_config=ladder_config,
        cost_config=cost_config,
        seed=1,
        bootstrap_iterations=0,  # no intervals
    )
    with pytest.raises(DishonestReportError, match="no bootstrap interval"):
        render_markdown(results, evaluation_config)


def test_the_report_refuses_precision_without_a_flag_rate(
    evaluation_config: EvaluationConfig,
) -> None:
    """Precision alone flatters a model that flags almost nothing."""
    honest = "PR-AUC 0.48 at a flag rate of 12%, precision 0.31."
    check_rendered_report(honest, evaluation_config)  # does not raise

    with pytest.raises(DishonestReportError, match="without a flag rate"):
        check_rendered_report("Precision reached 0.92.", evaluation_config)


def test_the_strongest_rung_is_chosen_on_net_rupees(ladder_run) -> None:
    """Not PR-AUC. SPEC section 05: if a simpler rung wins on money, it ships."""
    results, _ = ladder_run
    best = strongest_rung(results)

    nets = [
        (record.model_name, record.economics.net_inr_saved_per_1000_orders.value)
        for record in results.records
        if record.economics is not None
    ]
    assert best.model_name == max(nets, key=lambda pair: pair[1])[0]
