"""Command-line entry point.

Subcommands mirror the pipeline stages, so the whole system is reproducible from
a shell without a notebook anywhere in the loop::

    rto-sentinel config check                 # validate every YAML file
    rto-sentinel db upgrade                   # run migrations
    rto-sentinel generate --seed 42 ...       # build the benchmark dataset
    rto-sentinel validate --run-id ...        # re-validate an artefact
    rto-sentinel seed-db --seed 42 ...        # migrate, generate, validate, load
    rto-sentinel db stats --run-id ...        # query the loaded data back
    rto-sentinel features list                # every feature and its definition
    rto-sentinel features docs                # regenerate docs/features.md
    rto-sentinel build-dataset --seed 42 ...  # generate + split + build features
    rto-sentinel train --seed 42 ...          # train and evaluate the whole ladder
    rto-sentinel evaluate                     # re-render from saved artefacts
    rto-sentinel serve                        # run the API

Every generation records its seed, generator version, configuration snapshot and
creation timestamp, so any dataset can be traced back to the exact inputs that
produced it - and regenerated from them.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import textwrap
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from rto_sentinel import __version__
from rto_sentinel.configuration import (
    ConfigurationError,
    config_fingerprint,
    load_app_config,
    load_evaluation_config,
    load_generator_config,
    load_splits_config,
)
from rto_sentinel.data.artifacts import latest_dataset_dir, read_dataset
from rto_sentinel.data.generator import SUPPORTED_GENERATOR_VERSIONS, GeneratorParams
from rto_sentinel.data.pipeline import build_dataset
from rto_sentinel.data.validation import validate_delivery_events, validate_orders
from rto_sentinel.settings import REPO_ROOT, Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.contracts.economics import PortfolioEconomics, ThresholdSweep
    from rto_sentinel.contracts.experiment import LadderResults
    from rto_sentinel.contracts.final import FinalEvaluation, SelectionManifest
    from rto_sentinel.decision.simulation import SimulationResult
    from rto_sentinel.features.dataset import ModelingDataset

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _cmd_config_check(_: argparse.Namespace) -> int:
    """Validate every configuration file and report the bundle fingerprint."""
    settings = get_settings()
    try:
        config = load_app_config(settings)
    except ConfigurationError as exc:
        print(f"configuration invalid:\n{exc}", file=sys.stderr)
        return 1

    print("configuration OK")
    print(f"  config dir       : {settings.config_path}")
    print(f"  fingerprint      : {config_fingerprint(settings)}")
    print(f"  generator version: {config.generator.generator_version}")
    print(f"  generator orders : {config.generator.horizon.n_orders:,}")
    print(
        f"  split (days)     : train {config.splits.temporal.train_days}, "
        f"val {config.splits.temporal.validation_days}, "
        f"test {config.splits.temporal.test_days}"
    )
    print(f"  pool shares      : {config.splits.group.pool_shares}")
    print(f"  feature families : {', '.join(config.features.enabled_families)}")
    print(f"  refused patterns : {len(config.features.refused_patterns)}")
    print(f"  cost profiles    : {', '.join(sorted(config.cost_model.profiles))}")
    print(f"  ladder rungs     : {', '.join(r.name for r in config.ladder.rungs if r.enabled)}")
    print(f"  primary metric   : {config.evaluation.primary_metric}")
    return 0


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _resolve_params(args: argparse.Namespace, settings: Settings) -> GeneratorParams:
    """Turn CLI arguments into generator parameters, falling back to config."""
    generator_config = load_generator_config(settings)

    seed = args.seed if args.seed is not None else settings.random_seed
    n_customers = (
        args.customers if args.customers is not None else generator_config.customers.n_customers
    )
    n_orders = args.orders if args.orders is not None else generator_config.horizon.n_orders
    version = (
        args.generator_version
        if args.generator_version is not None
        else generator_config.generator_version
    )

    if args.start_date is not None:
        start = datetime.fromisoformat(args.start_date).replace(tzinfo=UTC)
    else:
        start = datetime.fromisoformat(generator_config.horizon.start_date).replace(tzinfo=UTC)

    if args.end_date is not None:
        end = datetime.fromisoformat(args.end_date).replace(tzinfo=UTC)
    else:
        end = start + timedelta(days=generator_config.horizon.days - 1)

    return GeneratorParams(
        seed=seed,
        generator_version=version,
        n_customers=n_customers,
        n_orders=n_orders,
        start_date=start,
        end_date=end,
    )


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate, validate and write a benchmark dataset."""
    settings = get_settings()
    params = _resolve_params(args, settings)

    print(
        f"generating {params.n_orders:,} orders for {params.n_customers:,} customers "
        f"over {params.days} days (seed {params.seed}, generator {params.generator_version})"
    )
    result = build_dataset(
        generator_config=load_generator_config(settings),
        splits_config=load_splits_config(settings),
        params=params,
        artifact_root=None if args.no_write else settings.artifact_path,
        strict=not args.lenient,
    )
    print()
    print(result.render())

    if not result.ok:
        print(
            "\nvalidation FAILED; the dataset was written but must not be trusted.", file=sys.stderr
        )
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Re-validate a dataset artefact that is already on disk."""
    settings = get_settings()
    directory = (
        settings.artifact_path / "datasets" / args.run_id
        if args.run_id
        else latest_dataset_dir(settings.artifact_path)
    )
    if directory is None or not directory.is_dir():
        print("no dataset artefact found; run `rto-sentinel generate` first.", file=sys.stderr)
        return 1

    artifact = read_dataset(directory)
    print(f"validating {directory}")
    print(f"  generator version : {artifact.metadata.get('generator_version')}")
    print(f"  seed              : {artifact.metadata.get('seed')}")
    print(f"  created at        : {artifact.metadata.get('created_at')}")
    print()

    order_report = validate_orders(
        artifact.orders,
        config=load_generator_config(settings),
        customers=artifact.customers,
        strict=not args.lenient,
    )
    event_report = validate_delivery_events(artifact.delivery_events, artifact.orders)
    print(order_report.render())
    print()
    print(event_report.render())

    return 0 if (order_report.ok and event_report.ok) else 1


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------


def _cmd_db_upgrade(_: argparse.Namespace) -> int:
    """Run Alembic migrations up to head."""
    from alembic import command
    from alembic.config import Config

    from rto_sentinel.settings import REPO_ROOT

    settings = get_settings()
    print(f"migrating {settings.database.safe_url}")
    alembic_config = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(alembic_config, "head")
    print("migrations applied")
    return 0


def _cmd_db_stats(args: argparse.Namespace) -> int:
    """Query a loaded dataset back out of the database."""
    from rto_sentinel.db.repositories import DatasetRepository
    from rto_sentinel.db.session import session_scope

    with session_scope() as session:
        repository = DatasetRepository(session)
        runs = repository.list_runs()
        if not runs:
            print("no dataset runs are loaded. Run `rto-sentinel seed-db`.", file=sys.stderr)
            return 1

        run = (
            next((r for r in runs if r.run_id == args.run_id), runs[0]) if args.run_id else runs[0]
        )

        print(f"dataset run {run.run_id}")
        print(f"  generator version  : {run.generator_version}")
        print(f"  seed               : {run.seed}")
        print(f"  config fingerprint : {run.config_fingerprint[:16]}...")
        print(f"  created at         : {run.created_at.isoformat()}")
        print(f"  horizon            : {run.start_date.date()} to {run.end_date.date()}")
        print(f"  provenance         : {run.data_provenance}")
        print()
        print("row counts (from SQL)")
        for table, count in repository.table_counts(run.run_id).items():
            print(f"  {table:<20}: {count:,}")
        print()
        print("splits")
        for split, count in sorted(repository.split_counts(run.run_id).items()):
            print(f"  {split:<20}: {count:,}")
        print()
        print("outcomes")
        for outcome, count in sorted(repository.outcome_counts(run.run_id).items()):
            print(f"  {outcome:<20}: {count:,}")
        print()
        print("RTO rate by payment method (mature orders only)")
        for method, rate in sorted(repository.rto_rate_by_payment_method(run.run_id).items()):
            print(f"  {method:<20}: {rate:.4f}")
    return 0


def _cmd_seed_db(args: argparse.Namespace) -> int:
    """The full path: migrate, generate, validate, load, verify.

    Refuses to load a dataset that failed validation. Loading known-bad data is
    worse than having none: it looks like a working system.
    """
    from rto_sentinel.db.repositories import DatasetRepository
    from rto_sentinel.db.session import session_scope

    settings = get_settings()

    if not args.skip_migrations:
        print("== 1. migrations ==")
        if _cmd_db_upgrade(args) != 0:
            return 1
        print()

    print("== 2. generate ==")
    params = _resolve_params(args, settings)
    result = build_dataset(
        generator_config=load_generator_config(settings),
        splits_config=load_splits_config(settings),
        params=params,
        artifact_root=settings.artifact_path,
        strict=not args.lenient,
    )
    print(result.render())
    print()

    print("== 3. validate ==")
    if not result.ok:
        print("validation failed; refusing to load into the database.", file=sys.stderr)
        return 1
    print("validation passed")
    print()

    print("== 4. load ==")
    with session_scope() as session:
        counts = DatasetRepository(session).load(result.dataset)
    for table, count in counts.items():
        print(f"  {table:<20}: {count:,}")
    print()

    print("== 5. verify (queried back from the database) ==")
    with session_scope() as session:
        repository = DatasetRepository(session)
        run_id = result.dataset.metadata.run_id
        for table, count in repository.table_counts(run_id).items():
            print(f"  {table:<20}: {count:,}")
        print()
        print("  RTO rate by payment method (mature orders only)")
        for method, rate in sorted(repository.rto_rate_by_payment_method(run_id).items()):
            print(f"    {method:<18}: {rate:.4f}")
    return 0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def _cmd_features_list(args: argparse.Namespace) -> int:
    """Print every feature with its definition and timestamp provenance."""
    from rto_sentinel.configuration import load_features_config
    from rto_sentinel.features import FEATURE_VERSION, FeaturePipeline

    settings = get_settings()
    pipeline = FeaturePipeline(load_features_config(settings), load_generator_config(settings))
    pipeline.check_declarations()
    feature_set = pipeline.feature_set

    print(f"feature version    : {FEATURE_VERSION}")
    print(f"feature fingerprint: {feature_set.fingerprint()}")
    print(f"features           : {len(feature_set)} across {len(feature_set.families)} families")
    print()

    for family in feature_set.families:
        subset = feature_set.by_family(family)
        print(f"=== {family}  ({len(subset)} features) ===")
        for spec in subset:
            lookback = str(spec.lookback) if spec.lookback else "-"
            flag = "OK " if spec.is_available_at_prediction_time else "LEAK"
            print(f"  [{flag}] {spec.name}")
            print(f"         type={spec.dtype}  observed={spec.observation_point}")
            print(f"         lookback={lookback}")
            print(f"         sources={', '.join(spec.source_columns)}")
            print(f"         {spec.description}")
            if args.verbose:
                print(f"         risk: {spec.risk_note}")
        print()

    unavailable = feature_set.unavailable_at_prediction_time()
    if unavailable:
        print(f"REFUSED: {[s.name for s in unavailable]} are not available at prediction time")
        return 1
    print("every feature is declared available at order time.")
    return 0


def _cmd_features_docs(_: argparse.Namespace) -> int:
    """Regenerate docs/features.md from the feature declarations."""
    from rto_sentinel.configuration import load_features_config
    from rto_sentinel.features import FEATURE_VERSION, FeaturePipeline
    from rto_sentinel.features.catalogue import render_markdown

    settings = get_settings()
    pipeline = FeaturePipeline(load_features_config(settings), load_generator_config(settings))
    pipeline.check_declarations()
    feature_set = pipeline.feature_set

    target = REPO_ROOT / "docs" / "features.md"
    target.write_text(
        render_markdown(
            feature_set,
            feature_version=FEATURE_VERSION,
            fingerprint=feature_set.fingerprint(),
        ),
        encoding="utf-8",
    )
    print(f"wrote {target} ({len(feature_set)} features)")
    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    """Generate, split, build features, and report the modelling dataset."""
    from rto_sentinel.configuration import load_features_config
    from rto_sentinel.data.splits import assign_splits
    from rto_sentinel.features import build_modeling_dataset

    settings = get_settings()
    params = _resolve_params(args, settings)
    generator_config = load_generator_config(settings)
    splits_config = load_splits_config(settings)

    print(
        f"generating {params.n_orders:,} orders for {params.n_customers:,} customers "
        f"over {params.days} days (seed {params.seed})"
    )
    pipeline_result = build_dataset(
        generator_config=generator_config,
        splits_config=splits_config,
        params=params,
        artifact_root=None if args.no_write else settings.artifact_path,
        strict=not args.lenient,
    )
    if not pipeline_result.ok:
        print("dataset validation FAILED; refusing to build features.", file=sys.stderr)
        print(pipeline_result.render(), file=sys.stderr)
        return 1

    dataset = build_modeling_dataset(
        pipeline_result.dataset,
        features_config=load_features_config(settings),
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=assign_splits(pipeline_result.dataset.orders, splits_config).labels,
    )

    print()
    print(dataset.describe())
    print()
    print("label balance")
    for name in ("train", "validation"):
        view = getattr(dataset, name)
        print(f"  {view.describe()}")
    # The test split's label is NOT printed. Its shape appears above; its
    # outcomes stay sealed until the single final evaluation.
    print("  test        [SEALED - row counts and dates above, labels withheld]")

    print()
    print("feature null shares (highest 10)")
    null_shares = dataset.features.isna().mean().sort_values(ascending=False).head(10)
    for raw_name, share in null_shares.items():
        name = str(raw_name)
        expected = dataset.feature_set.get(name).expected_null_share
        print(f"  {name:<38} {share:6.1%}   (declared ~{expected:.0%})")
    return 0


# ---------------------------------------------------------------------------
# training and evaluation
# ---------------------------------------------------------------------------


def _load_or_build_dataset(
    args: argparse.Namespace, settings: Settings
) -> tuple[ModelingDataset, GeneratorParams]:
    """Build the modelling dataset the ladder trains on.

    Regenerated from the seed rather than loaded from a parquet artefact. The
    generator is deterministic, so this costs time and buys certainty that the
    features being trained on came from the configuration currently on disk.
    """
    from rto_sentinel.configuration import load_features_config
    from rto_sentinel.data.splits import assign_splits
    from rto_sentinel.features import build_modeling_dataset

    params = _resolve_params(args, settings)
    generator_config = load_generator_config(settings)
    splits_config = load_splits_config(settings)

    print(
        f"generating {params.n_orders:,} orders for {params.n_customers:,} customers "
        f"(seed {params.seed}, generator {params.generator_version})"
    )
    pipeline_result = build_dataset(
        generator_config=generator_config,
        splits_config=splits_config,
        params=params,
        artifact_root=None if args.no_write else settings.artifact_path,
        strict=not args.lenient,
    )
    if not pipeline_result.ok:
        print(pipeline_result.render(), file=sys.stderr)
        msg = "dataset validation failed; refusing to train on it"
        raise SystemExit(msg)

    dataset = build_modeling_dataset(
        pipeline_result.dataset,
        features_config=load_features_config(settings),
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=assign_splits(pipeline_result.dataset.orders, splits_config).labels,
    )
    return dataset, params


def _cmd_train(args: argparse.Namespace) -> int:
    """Train and evaluate every enabled rung of the baseline ladder."""
    from rto_sentinel.configuration import load_cost_model_config, load_ladder_config
    from rto_sentinel.eval import comparison_table, strongest_rung, write_report
    from rto_sentinel.models import run_ladder, save_results, scores_frame

    settings = get_settings()
    dataset, params = _load_or_build_dataset(args, settings)

    print(
        f"\ntrain={dataset.train.n_rows:,}  validation={dataset.validation.n_rows:,}  "
        f"features={len(dataset.feature_set)}  "
        f"positive rate (validation)={dataset.validation.positive_rate:.4f}"
    )
    if args.split == "test":
        # Deliberately loud. Reaching the sealed split is a one-time act at the
        # very end of the project, after the threshold has been fixed on
        # validation, and it should never happen by habit.
        print(
            "\n!! SEALED TEST SET REQUESTED. This is the single final evaluation.",
            file=sys.stderr,
        )
        dataset.unseal_test(reason=args.unseal_reason or "explicit --split test on the CLI")

    print("\ntraining the ladder...")
    results, trained = run_ladder(
        dataset,
        ladder_config=load_ladder_config(settings),
        cost_config=load_cost_model_config(settings),
        seed=params.seed,
        evaluation_split=args.split,
        artifact_root=settings.artifact_path if not args.no_write else None,
        bootstrap_iterations=args.bootstrap,
    )

    print(f"\noperating threshold: {results.threshold:.4f}")
    print(f"  {results.threshold_source}")
    print()
    print(comparison_table(results))

    baseline = results.records[0].economics
    if baseline is not None:
        print(
            f"\ndo-nothing absorbs INR {abs(baseline.baseline_net_inr_per_1000_orders):,.0f} "
            "per 1,000 orders. net/1k above is the saving relative to that."
        )

    best = strongest_rung(results)
    net = best.economics.net_inr_saved_per_1000_orders if best.economics else None
    print(
        f"\nstrongest on net rupees: {best.model_name} (rung {best.rung_id})"
        + (f" at INR {net.value:,.0f} per 1,000 orders" if net else "")
    )
    print("every rung here is UNCALIBRATED; `rto-sentinel final` calibrates.")

    if args.no_write:
        return 0

    results_path = save_results(results, settings.artifact_path)
    print(f"\nmachine-readable results : {results_path}")

    scores = scores_frame(dataset, trained, args.split)
    scores_path = (
        settings.artifact_path
        / "experiments"
        / results.dataset_run_id
        / f"scores__{args.split}.parquet"
    )
    scores.to_parquet(scores_path, index=False)
    print(f"per-order scores         : {scores_path}")

    report_path = write_report(
        results,
        REPO_ROOT / "docs" / "ladder_results.md",
        load_evaluation_config(settings),
    )
    print(f"report                   : {report_path}")

    for path in _write_plots(results, scores, settings):
        print(f"plot                     : {path}")

    for name, run in trained.items():
        if run.artifact_path is not None:
            print(f"model artefact           : {name} -> {run.artifact_path}")
    return 0


def _write_plots(results: LadderResults, scores: pd.DataFrame, settings: Settings) -> list[Path]:
    from rto_sentinel.configuration import load_cost_model_config
    from rto_sentinel.eval.plots import generate_all
    from rto_sentinel.models.experiment import cost_inputs_from_profile

    cost_config = load_cost_model_config(settings)
    cost_inputs = cost_inputs_from_profile(cost_config.profiles[results.cost_profile])
    output = settings.artifact_path / "reports" / results.dataset_run_id
    return generate_all(results, scores, cost_inputs, output)


# ---------------------------------------------------------------------------
# the final model: selection, calibration, and the one test-set read
# ---------------------------------------------------------------------------


def _or_dash(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_selection(manifest: SelectionManifest) -> None:
    print("\ncandidate search (fitted on train, scored on validation)")
    header = f"  {'candidate':<20}{'val PR-AUC':>12}{'train':>9}{'trn-val':>9}{'fit s':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for candidate in manifest.candidates:
        gap = candidate.overfit_gap
        mark = "*" if candidate.selected else " "
        train_pr = candidate.train_pr_auc if candidate.train_pr_auc is not None else float("nan")
        print(
            f" {mark}{candidate.name:<20}{candidate.validation_pr_auc:>12.4f}"
            f"{train_pr:>9.4f}{gap if gap is not None else float('nan'):>+9.3f}"
            f"{candidate.train_duration_seconds:>8.1f}"
        )

    print("\ncalibration (cross-validated inside validation, never fitted on test)")
    header = f"  {'method':<20}{'ECE':>12}{'Brier':>9}{'vs none':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in manifest.calibration_candidates:
        mark = "*" if method.selected else " "
        print(
            f" {mark}{method.method:<20}{method.expected_calibration_error:>12.4f}"
            f"{method.brier_score:>9.4f}{method.improvement_over_none:>+10.4f}"
        )


def _print_evaluation(evaluation: FinalEvaluation, *, caveat: str = "") -> None:
    ranking, point = evaluation.ranking, evaluation.operating_point
    net = evaluation.economics.net_inr_saved_per_1000_orders
    print(f"\n{evaluation.evaluated_split} split ({evaluation.evaluation_summary.n_rows:,} orders)")
    if caveat:
        print(f"  {caveat}")
    print(
        f"  PR-AUC        : {ranking.pr_auc.value:.4f} "
        f"[{ranking.pr_auc.ci_low:.4f}, {ranking.pr_auc.ci_high:.4f}]"
        f"   (uncalibrated {evaluation.uncalibrated_pr_auc:.4f})"
    )
    print(
        f"  ROC-AUC       : {ranking.roc_auc.value:.4f} "
        f"[{ranking.roc_auc.ci_low:.4f}, {ranking.roc_auc.ci_high:.4f}]"
    )
    print(f"  Recall@P80    : {_or_dash(ranking.recall_at_precision_80)}")
    print(f"  Recall@P90    : {_or_dash(ranking.recall_at_precision_90)}")
    print(
        f"  Brier         : {evaluation.calibration.brier_score:.4f}"
        f"   (uncalibrated {evaluation.uncalibrated_calibration.brier_score:.4f})"
    )
    print(
        f"  ECE           : {evaluation.calibration.expected_calibration_error:.4f}"
        f"   (uncalibrated "
        f"{evaluation.uncalibrated_calibration.expected_calibration_error:.4f})"
    )
    print(f"  threshold     : {point.threshold:.4f}  ({point.threshold_source})")
    print(
        f"  flag/prec/rec/F1: {point.flag_rate:.3f} / {_or_dash(point.precision)} / "
        f"{_or_dash(point.recall)} / {_or_dash(point.f1)}"
    )
    print(
        f"  confusion     : TP {point.true_positives:,}  FP {point.false_positives:,}  "
        f"FN {point.false_negatives:,}  TN {point.true_negatives:,}"
    )
    print(
        f"  net INR/1k    : {net.value:,.0f} [{net.ci_low:,.0f}, {net.ci_high:,.0f}]"
        f"   (FP cost INR {evaluation.economics.total_false_positive_cost_inr:,.0f})"
    )


def _latest_ladder_results(settings: Settings, dataset_run_id: str) -> LadderResults | None:
    """The Phase 4 ladder for this dataset, if one was measured.

    Matched on dataset run: comparing the final model against a ladder measured
    on different data would be a comparison of two different problems.
    """
    from rto_sentinel.contracts.experiment import LadderResults as Results

    directory = settings.artifact_path / "experiments" / dataset_run_id
    candidates = sorted(directory.glob("ladder__validation__*.json"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return Results.model_validate_json(newest.read_text(encoding="utf-8"))


def _write_final_outputs(
    manifest: SelectionManifest,
    evaluations: dict[str, FinalEvaluation],
    settings: Settings,
) -> None:
    """Metrics CSV, comparison CSV and the model card. Regenerated every run."""
    from rto_sentinel.configuration import load_model_card_config
    from rto_sentinel.eval import write_comparison_csv, write_metrics_csv, write_model_card
    from rto_sentinel.models import final_dir

    directory = final_dir(settings.artifact_path, manifest.dataset_run_id)
    metrics_csv = write_metrics_csv(evaluations, directory / "metrics.csv")
    print(f"metrics CSV              : {metrics_csv}")

    ladder = _latest_ladder_results(settings, manifest.dataset_run_id)
    comparison_csv = write_comparison_csv(evaluations, ladder, directory / "comparison.csv")
    print(f"comparison CSV           : {comparison_csv}")

    card_path = write_model_card(
        load_model_card_config(settings),
        manifest,
        evaluations,
        REPO_ROOT / "docs" / "model_card.md",
        ladder=ladder,
    )
    print(f"model card               : {card_path}")


def _final_artifact_path(settings: Settings, manifest: SelectionManifest) -> Path | None:
    """The saved artefact matching the frozen model version, if it is present."""
    from rto_sentinel.models import list_artifacts

    for path, card in list_artifacts(settings.artifact_path):
        if card.model_version == manifest.model_version:
            return path
    return None


def _cmd_final(args: argparse.Namespace) -> int:
    """Select, calibrate and freeze the final model. Reads validation only."""
    from rto_sentinel.configuration import load_cost_model_config, load_final_model_config
    from rto_sentinel.eval.plots import generate_final_plots
    from rto_sentinel.models import (
        build_final_model,
        evaluate_final_model,
        final_dir,
        save_evaluation,
        save_manifest,
    )
    from rto_sentinel.models.experiment import cost_inputs_from_profile
    from rto_sentinel.models.final import scores_frame as final_scores_frame

    settings = get_settings()
    dataset, params = _load_or_build_dataset(args, settings)
    final_config = load_final_model_config(settings)
    cost_config = load_cost_model_config(settings)

    print(
        f"\ntrain={dataset.train.n_rows:,}  validation={dataset.validation.n_rows:,}  "
        f"features={len(dataset.feature_set)}  "
        f"positive rate (validation)={dataset.validation.positive_rate:.4f}"
    )
    print(f"\nselecting over {len(final_config.search.candidates)} candidates...")

    final = build_final_model(
        dataset,
        final_config=final_config,
        cost_config=cost_config,
        seed=params.seed,
        artifact_root=settings.artifact_path if not args.no_write else None,
    )
    manifest = final.manifest
    _print_selection(manifest)

    print(f"\nselected      : {manifest.chosen_candidate} + {manifest.calibration_method}")
    print(f"tie rule      : {final_config.search.tie_rule}")
    print(f"model         : {final.model.name}  v{manifest.model_version}")
    print(f"manifest      : {manifest.manifest_id}")
    print(f"threshold     : {manifest.threshold:.4f}  ({manifest.threshold_source})")

    cost_inputs = cost_inputs_from_profile(cost_config.profiles[manifest.cost_profile])
    evaluation, calibrated, raw = evaluate_final_model(
        final.model,
        dataset.validation,
        manifest=manifest,
        cost_inputs=cost_inputs,
        bootstrap_iterations=args.bootstrap,
    )
    _print_evaluation(
        evaluation,
        caveat=(
            "SELECTION-CONTAMINATED: hyperparameters were chosen on this split and the "
            "shipped calibrator was refitted on it. The honest read is the test set."
        ),
    )
    print("\nthe test split has NOT been read. Run `rto-sentinel final-test` for that.")

    if args.no_write:
        return 0

    manifest_path = save_manifest(manifest, settings.artifact_path)
    print(f"\nfrozen manifest          : {manifest_path}")
    metrics_path = save_evaluation(evaluation, settings.artifact_path)
    print(f"validation metrics       : {metrics_path}")

    directory = final_dir(settings.artifact_path, manifest.dataset_run_id)
    scores_path = directory / "scores__validation.parquet"
    final_scores_frame(dataset.validation, calibrated, raw).to_parquet(scores_path, index=False)
    print(f"per-order scores         : {scores_path}")

    for path in generate_final_plots(
        evaluation,
        dataset.validation.y.to_numpy(dtype=bool),
        calibrated,
        manifest.threshold,
        directory,
    ):
        print(f"plot                     : {path}")

    _write_final_outputs(manifest, {"validation": evaluation}, settings)
    if final.artifact_path is not None:
        print(f"model artefact           : {final.artifact_path}")
    return 0


def _cmd_final_test(args: argparse.Namespace) -> int:
    """The single sealed-set evaluation. Requires a frozen manifest and a reason."""
    from rto_sentinel.configuration import load_cost_model_config
    from rto_sentinel.eval.plots import generate_final_plots
    from rto_sentinel.models import (
        evaluate_final_model,
        final_dir,
        load_evaluation,
        load_manifest,
        save_evaluation,
    )
    from rto_sentinel.models.calibrated import CalibratedModel
    from rto_sentinel.models.experiment import cost_inputs_from_profile
    from rto_sentinel.models.final import scores_frame as final_scores_frame

    settings = get_settings()
    dataset, _ = _load_or_build_dataset(args, settings)

    try:
        manifest = load_manifest(settings.artifact_path, dataset.metadata.dataset_run_id)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    directory = final_dir(settings.artifact_path, manifest.dataset_run_id)
    if (directory / "metrics__test.json").is_file() and not args.again:
        existing = load_evaluation(settings.artifact_path, manifest.dataset_run_id, "test")
        print(
            "the sealed test set has already been scored for this dataset and manifest.\n"
            f"  evaluated at : {existing.evaluated_at.isoformat()}\n"
            f"  reason       : {existing.unseal_reason}\n"
            "Scoring it repeatedly turns a held-out set into a validation set. Pass --again "
            "only if you understand that, and say why in --unseal-reason.",
            file=sys.stderr,
        )
        return 1

    # Load the artefact that was frozen rather than retraining. Retraining would
    # reproduce it - the pipeline is deterministic - but "the model we tested" and
    # "a model we rebuilt from the same recipe" are different claims.
    artifact = _final_artifact_path(settings, manifest)
    if artifact is None:
        print(
            f"no model artefact for version {manifest.model_version}. Re-run "
            "`rto-sentinel final` to write one.",
            file=sys.stderr,
        )
        return 1
    model, card = CalibratedModel.load(artifact)

    print("\n" + "=" * 78, file=sys.stderr)
    print(
        "SEALED TEST SET. This is the single final evaluation of the final model.\n"
        f"  manifest  : {manifest.manifest_id} frozen {manifest.frozen_at.isoformat()}\n"
        f"  model     : {card.model_name} v{card.model_version} "
        f"({card.calibration_method} calibration, fitted on {card.calibration_fitted_on})\n"
        f"  threshold : {manifest.threshold:.4f} - derived from costs, not from these labels\n"
        f"  reason    : {args.unseal_reason}",
        file=sys.stderr,
    )
    print("=" * 78 + "\n", file=sys.stderr)

    dataset.unseal_test(reason=args.unseal_reason)
    cost_config = load_cost_model_config(settings)
    cost_inputs = cost_inputs_from_profile(cost_config.profiles[manifest.cost_profile])

    evaluation, calibrated, raw = evaluate_final_model(
        model,
        dataset.test,
        manifest=manifest,
        cost_inputs=cost_inputs,
        bootstrap_iterations=args.bootstrap,
        unseal_reason=args.unseal_reason,
    )
    _print_evaluation(evaluation)

    if args.no_write:
        return 0

    metrics_path = save_evaluation(evaluation, settings.artifact_path)
    print(f"\ntest metrics             : {metrics_path}")

    scores_path = directory / "scores__test.parquet"
    final_scores_frame(dataset.test, calibrated, raw).to_parquet(scores_path, index=False)
    print(f"per-order scores         : {scores_path}")

    for path in generate_final_plots(
        evaluation,
        dataset.test.y.to_numpy(dtype=bool),
        calibrated,
        manifest.threshold,
        directory,
    ):
        print(f"plot                     : {path}")

    # The card shows validation beside test when both exist, so a reader can see
    # what selecting on validation cost. Test alone is still a complete card.
    evaluations: dict[str, FinalEvaluation] = {"test": evaluation}
    with contextlib.suppress(FileNotFoundError):
        evaluations = {
            "validation": load_evaluation(
                settings.artifact_path, manifest.dataset_run_id, "validation"
            ),
            "test": evaluation,
        }
    _write_final_outputs(manifest, evaluations, settings)
    return 0


def _cmd_final_report(args: argparse.Namespace) -> int:
    """Re-render the model card and CSVs from saved artefacts. Reads no split.

    Separate from `final` and `final-test` on purpose. Improving how a result is
    presented must never require re-running the measurement, because re-running
    the sealed measurement is the one thing the seal exists to prevent. This
    command touches the manifest and the metrics JSON and nothing else.
    """
    from rto_sentinel.models import load_evaluation, load_manifest

    settings = get_settings()
    root = settings.artifact_path / "final"
    runs = sorted(root.glob("*/selection_manifest.json")) if root.is_dir() else []
    if not runs:
        print(
            "no frozen selection manifest found. Run `rto-sentinel final` first.",
            file=sys.stderr,
        )
        return 1

    newest = max(runs, key=lambda path: path.stat().st_mtime)
    dataset_run_id = newest.parent.name
    manifest = load_manifest(settings.artifact_path, dataset_run_id)

    evaluations: dict[str, FinalEvaluation] = {}
    for split in ("validation", "test"):
        with contextlib.suppress(FileNotFoundError):
            evaluations[split] = load_evaluation(settings.artifact_path, dataset_run_id, split)
    if not evaluations:
        print(f"no evaluations saved for dataset {dataset_run_id}.", file=sys.stderr)
        return 1

    print(f"manifest {manifest.manifest_id} (dataset {dataset_run_id})")
    print(f"splits   : {', '.join(evaluations)}")
    _write_final_outputs(manifest, evaluations, settings)
    return 0


def _cmd_economics(args: argparse.Namespace) -> int:
    """Price the shipped policy, sweep the threshold, and write the report.

    Reads the calibrated validation book written by `rto-sentinel final`. It
    never touches the sealed split - the sweep and the simulator both refuse it -
    so this command can be re-run as often as anyone likes.
    """
    from rto_sentinel.configuration import load_cost_model_config, load_policy_config
    from rto_sentinel.decision.engine import ENGINE_VERSION
    from rto_sentinel.decision.portfolio import evaluate_portfolio
    from rto_sentinel.decision.simulation import compare_ladder_against_uniform, simulate
    from rto_sentinel.decision.threshold_analysis import sweep_thresholds
    from rto_sentinel.eval.economics_report import (
        render_economics_report,
        write_band_csv,
        write_economics_report,
        write_sweep_csv,
    )
    from rto_sentinel.models import load_scored_book
    from rto_sentinel.models.experiment import cost_inputs_from_profile

    settings = get_settings()
    try:
        book = load_scored_book(settings.artifact_path, split="validation")
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    cost_config = load_cost_model_config(settings)
    policy = load_policy_config(settings)
    profile_key = args.profile or cost_config.default_profile
    if profile_key not in cost_config.profiles:
        print(
            f"unknown cost profile {profile_key!r}; have {sorted(cost_config.profiles)}",
            file=sys.stderr,
        )
        return 1
    cost_inputs = cost_inputs_from_profile(cost_config.profiles[profile_key])

    print(f"\nscored book : {book.n_orders:,} orders, {book.split} split")
    print(f"model       : v{book.model_version}  (dataset {book.dataset_run_id})")
    print(f"profile     : {profile_key}")

    economics = evaluate_portfolio(
        book.probabilities,
        cost_inputs=cost_inputs,
        policy=policy,
        labels=book.labels,
        split=book.split,
        cost_profile=profile_key,
        engine_version=ENGINE_VERSION,
    )

    print(f"\nthreshold   : {economics.threshold:.4f}  ({economics.threshold_source})")
    print("\nintervention ladder")
    header = (
        f"  {'band':<9}{'action':<24}{'range':<20}{'orders':>8}{'share':>8}"
        f"{'E[RTO]':>9}{'net INR':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for band in economics.bands:
        upper = "1.0" if band.upper_bound is None else f"{band.upper_bound:.4f}"
        span = f"[{band.lower_bound:.4f}, {upper})"
        print(
            f"  {band.band.value:<9}{band.action.value:<24}{span:<20}{band.n_orders:>8,}"
            f"{band.share_of_book:>8.1%}{band.expected_rto_orders:>9.1f}"
            f"{band.expected_net_inr:>12,.0f}"
        )
    for entry in economics.collapsed_bands:
        print(f"  collapsed: {entry}")

    print("\neconomic outcome")
    print(f"  flag rate          : {economics.flag_rate:.4f}")
    print(f"  intervention rate  : {economics.intervention_rate:.4f}")
    print(f"  orders affected    : {economics.expected_orders_affected:,}")
    print(f"  expected savings   : INR {economics.expected_savings_inr:>12,.0f}")
    print(f"  false-positive cost: INR {economics.expected_false_positive_cost_inr:>12,.0f}")
    print(f"  residual FN loss   : INR {economics.expected_false_negative_loss_inr:>12,.0f}")
    print(f"  expected total cost: INR {economics.expected_total_cost_inr:>12,.0f}")
    print(f"  expected net/1k    : INR {economics.expected_net_inr_per_1000_orders:>12,.0f}")
    if economics.realized_net_inr_per_1000_orders is not None:
        print(f"  realized net/1k    : INR {economics.realized_net_inr_per_1000_orders:>12,.0f}")
        print(
            f"  calibration gap    : {economics.calibration_gap:+.1f} true positives "
            f"(expected {economics.expected_true_positives:.1f})"
        )
    print(f"  do-nothing loss/1k : INR {economics.do_nothing_loss_inr_per_1000_orders:>12,.0f}")
    print(f"  net after holdout  : INR {economics.net_inr_per_1000_after_holdout:>12,.0f}")

    sweep = sweep_thresholds(
        book.probabilities,
        book.labels,
        cost_inputs=cost_inputs,
        split=book.split,
        cost_profile=profile_key,
    )
    print(
        f"\nthreshold sweep    : derived {sweep.derived_threshold:.4f}, "
        f"curve peaks at {sweep.best_net_threshold:.4f}"
    )
    print("  the operating point is DERIVED from economics and is never read off the curve.")

    comparison = compare_ladder_against_uniform(
        book.probabilities, cost_inputs=cost_inputs, policy=policy, labels=book.labels
    )
    print(f"\ngraduated ladder   : INR {comparison.graduated_net_inr_per_1000:>10,.0f} per 1,000")
    for action, value in sorted(comparison.uniform_net_inr_per_1000.items(), key=lambda i: -i[1]):
        print(f"  uniform {action:<22}: INR {value:>10,.0f}")
    print(f"  graduated wins     : {comparison.graduated_wins}")

    # Merchant simulations: every configured profile, plus the margin change the
    # specification names explicitly.
    simulations = []
    for key, profile in cost_config.profiles.items():
        inputs = cost_inputs_from_profile(profile)
        simulations.append(
            (
                profile.label,
                simulate(
                    book.probabilities,
                    cost_inputs=inputs,
                    policy=policy,
                    labels=book.labels,
                    split=book.split,
                    cost_profile=key,
                    baseline=cost_inputs,
                ),
            )
        )
    for margin in (250.0, 400.0):
        simulations.append(
            (
                f"Margin changed to INR {margin:,.0f}",
                simulate(
                    book.probabilities,
                    cost_inputs=cost_inputs.model_copy(update={"contribution_margin_inr": margin}),
                    policy=policy,
                    labels=book.labels,
                    split=book.split,
                    cost_profile=profile_key,
                    baseline=cost_inputs,
                ),
            )
        )

    print("\nmerchant simulation (recomputed server-side, nothing scaled)")
    header = f"  {'scenario':<44}{'margin':>9}{'threshold':>11}{'flag':>8}{'net/1k':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, result in simulations:
        print(
            f"  {label[:43]:<44}{result.threshold.inputs.contribution_margin_inr:>9,.0f}"
            f"{result.threshold.threshold:>11.4f}{result.economics.flag_rate:>8.3f}"
            f"{result.economics.expected_net_inr_per_1000_orders:>10,.0f}"
        )

    if args.no_write:
        return 0

    directory = settings.artifact_path / "economics" / book.dataset_run_id
    directory.mkdir(parents=True, exist_ok=True)

    economics_path = directory / f"portfolio__{profile_key}.json"
    economics_path.write_text(economics.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nportfolio economics : {economics_path}")

    sweep_path = directory / f"sweep__{profile_key}.json"
    sweep_path.write_text(sweep.model_dump_json(indent=2), encoding="utf-8")
    print(f"threshold sweep     : {sweep_path}")

    print(f"sweep CSV           : {write_sweep_csv(sweep, directory / 'threshold_sweep.csv')}")
    print(f"band CSV            : {write_band_csv(economics, directory / 'bands.csv')}")

    document = render_economics_report(
        economics=economics,
        sweep=sweep,
        comparison=comparison,
        cost_inputs=cost_inputs,
        cost_config=cost_config,
        policy=policy,
        simulations=simulations,
    )
    report_path = write_economics_report(REPO_ROOT / "docs" / "economics.md", document)
    print(f"report              : {report_path}")

    # A CONTROLLED margin sweep for the figure: one input varied, everything else
    # held fixed. The profile table above varies three inputs at once, which is
    # the right comparison for a merchant and the wrong one for a causal claim.
    margin_sweep = [
        (
            margin,
            simulate(
                book.probabilities,
                cost_inputs=cost_inputs.model_copy(update={"contribution_margin_inr": margin}),
                policy=policy,
                labels=book.labels,
                split=book.split,
                cost_profile=profile_key,
            ),
        )
        for margin in (50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 900.0, 1400.0)
    ]

    for path in _write_economics_plots(economics, sweep, margin_sweep, directory):
        print(f"plot                : {path}")
    return 0


def _write_economics_plots(
    economics: PortfolioEconomics,
    sweep: ThresholdSweep,
    margin_sweep: list[tuple[float, SimulationResult]],
    directory: Path,
) -> list[Path]:
    from rto_sentinel.eval.plots import (
        plot_band_economics,
        plot_margin_response,
        plot_threshold_economics,
    )

    return [
        plot_threshold_economics(sweep, directory / "threshold_economics.png"),
        plot_band_economics(economics, directory / "band_economics.png"),
        plot_margin_response(margin_sweep, directory / "margin_response.png"),
    ]


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Re-render the comparison from saved experiment artefacts.

    Reads the machine-readable results rather than retraining, which is the
    point: a reported number should be reproducible from the artefact, not only
    from a fresh run.
    """
    from rto_sentinel.contracts.experiment import LadderResults
    from rto_sentinel.eval import comparison_table, strongest_rung

    settings = get_settings()
    root = settings.artifact_path / "experiments"
    candidates = sorted(root.rglob("ladder__*.json")) if root.is_dir() else []
    if not candidates:
        print("no ladder results found. Run `rto-sentinel train` first.", file=sys.stderr)
        return 1

    target = max(candidates, key=lambda path: path.stat().st_mtime)
    results = LadderResults.model_validate_json(target.read_text(encoding="utf-8"))

    print(f"results from {target}")
    print(
        f"  dataset {results.dataset_run_id}  split {results.evaluated_split}  seed {results.seed}"
    )
    print(f"  threshold {results.threshold:.4f}")
    print()
    print(comparison_table(results))
    best = strongest_rung(results)
    print(f"\nstrongest on net rupees: {best.model_name} (rung {best.rung_id})")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the API with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rto_sentinel.api.main:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )
    return 0


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"`{args.command}` is not implemented yet; it lands in {args.phase}.", file=sys.stderr)
    return 2


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """The generation parameters, shared by `generate` and `seed-db`.

    All optional: each falls back to ``config/generator.yaml``. Whatever is
    actually used is recorded on the dataset run, so a value taken from config is
    just as traceable as one passed here.
    """
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: from config)")
    parser.add_argument("--customers", type=int, default=None, help="number of customers")
    parser.add_argument("--orders", type=int, default=None, help="number of orders")
    parser.add_argument("--start-date", default=None, help="horizon start, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="horizon end, YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--generator-version",
        default=None,
        choices=sorted(SUPPORTED_GENERATOR_VERSIONS),
        help="version of the generative process to run",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="downgrade base-rate drift from an error to a warning (small samples)",
    )


# ---------------------------------------------------------------------------
# Phase 10: fairness, distribution shift, monitoring
# ---------------------------------------------------------------------------


def _cohort_frame(orders: pd.DataFrame) -> pd.DataFrame:
    """The operational cohorts this audit is allowed to examine.

    Every column here is an operational fact recorded on the order: where it is
    being delivered, how large it is, how much history the customer has, how it
    is being paid for. None is a sensitive characteristic and none is inferred
    from one - see `docs/responsible_ai.md`.
    """
    import pandas as pd

    from rto_sentinel.eval.fairness import band_column, history_band

    return pd.DataFrame(
        {
            "pincode_tier": orders["pincode_tier"].astype("object"),
            "order_value_band": band_column(orders["order_value_inr"], n_bands=4, prefix="v"),
            "customer_history_band": history_band(orders["prior_order_count"]),
            "payment_method": orders["payment_method"].astype("object"),
        }
    )


def _or_na(value: float | None, width: int, digits: int) -> str:
    """A number, or a right-aligned "n/a" of the same width.

    Printing 0 for an undefined metric would line up neatly and be a lie: a
    precision with nothing flagged has no denominator, and a zero there sorts and
    averages as though it had been measured.
    """
    if value is None:
        return "n/a".rjust(width)
    return f"{value:{width}.{digits}f}"


def _cmd_fairness(args: argparse.Namespace) -> int:
    """Run the cohort and fairness audit against the frozen scored book.

    Reads the calibrated scores written by `rto-sentinel final` and joins them
    back to the order attributes. Runs on validation by default; the sealed test
    split requires an explicit reason, exactly as every other test-set read does.
    """

    from rto_sentinel.configuration import load_cost_model_config, load_evaluation_config
    from rto_sentinel.eval.fairness import cohort_breakdown, fairness_audit
    from rto_sentinel.eval.responsible_report import write_fairness_artifacts
    from rto_sentinel.models import load_manifest
    from rto_sentinel.models.experiment import cost_inputs_from_profile

    settings = get_settings()
    split = args.split

    try:
        scores, orders, run_id = _load_scores_with_attributes(settings, split)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    manifest = load_manifest(settings.artifact_path, run_id)
    cost_config = load_cost_model_config(settings)
    evaluation_config = load_evaluation_config(settings)
    inputs = cost_inputs_from_profile(cost_config.profiles[manifest.cost_profile])
    min_support = args.min_support or evaluation_config.fairness.min_support_orders
    min_flagged = args.min_flagged or evaluation_config.fairness.min_flagged_orders

    from rto_sentinel.decision.cost_model import outcome_economics

    economics = outcome_economics(inputs)
    cohorts = _cohort_frame(orders)
    labels = scores["label"].to_numpy().astype(int)
    probabilities = scores["score_calibrated"].to_numpy(dtype=float)

    print(
        f"\nfairness audit on {split}: {len(labels):,} orders, threshold {manifest.threshold:.4f}"
    )
    print(f"cohorts: {', '.join(cohorts.columns)}")
    print(f"minimum support: {min_support} orders, {min_flagged} flagged\n")

    all_slices: list[object] = []
    for column in cohorts.columns:
        rows = cohort_breakdown(
            cohorts,
            labels,
            probabilities,
            threshold=manifest.threshold,
            cohort_column=column,
            min_support=min_support,
            min_flagged=min_flagged,
            cost_false_positive_inr=economics.false_positive_cost_inr,
            saving_true_positive_inr=economics.true_positive_saving_inr,
        )
        all_slices.extend(rows)
        print(f"  {column}")
        print(
            f"    {'group':34s} {'n':>6s} {'RTO':>7s} {'flag':>7s} {'prec':>7s} "
            f"{'recall':>7s} {'net/1k':>9s}"
        )
        for row in rows:
            mark = "" if row.sufficient else "  (thin)"
            print(
                f"    {row.group:34s} {row.n_orders:6d} {row.rto_rate:7.3f} "
                f"{row.flag_rate:7.3f} "
                f"{_or_na(row.precision, 7, 3)} "
                f"{_or_na(row.recall, 7, 3)} "
                f"{_or_na(row.net_inr_per_1000, 9, 0)}"
                f"{mark}"
            )
        print()

    audit = fairness_audit(
        cohorts,
        labels,
        probabilities,
        threshold=manifest.threshold,
        config=evaluation_config.fairness,
        min_support=min_support,
        min_flagged=min_flagged,
        cost_false_positive_inr=economics.false_positive_cost_inr,
        saving_true_positive_inr=economics.true_positive_saving_inr,
    )

    print(f"disparity review: {'TRIGGERED' if audit.triggered else 'not triggered'}")
    print(f"  max flag-rate ratio  : {audit.max_flag_rate_ratio:.2f}")
    print(f"  worst precision drop : {audit.worst_precision_drop:.3f}")
    print()
    print(textwrap.fill(audit.narrative, width=88))

    if not args.no_write:
        written = write_fairness_artifacts(
            audit,
            tuple(all_slices),  # type: ignore[arg-type]
            artifact_root=settings.artifact_path,
            dataset_run_id=run_id,
            split=split,
            model_version=manifest.model_version,
            threshold=manifest.threshold,
        )
        print()
        for path in written:
            print(f"wrote {path}")
    return 0


def _load_scores_with_attributes(
    settings: Settings, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Join the frozen scored book back to the order attributes it was built from.

    The scored book carries order ids, not attributes, which is deliberate: it is
    a record of what the model said, not a copy of the dataset. The cohort audit
    needs both, so the orders are reloaded from the dataset artefact and joined
    on order id. An inner join is used and its size asserted, because a silent
    partial join would produce a fairness table computed on a subset nobody chose.
    """
    import pandas as pd

    from rto_sentinel.models.final import FINAL_DIR

    root = settings.artifact_path / FINAL_DIR
    runs = sorted(root.glob("*/selection_manifest.json")) if root.is_dir() else []
    if not runs:
        msg = (
            f"no frozen final-model run under {root}. Run `rto-sentinel final` before "
            "auditing: there is no scored book to audit."
        )
        raise FileNotFoundError(msg)

    run_dir = max(runs, key=lambda path: path.stat().st_mtime).parent
    run_id = run_dir.name
    scores_path = run_dir / f"scores__{split}.parquet"
    if not scores_path.is_file():
        msg = (
            f"no {split} scores at {scores_path}. The sealed test split is scored only by "
            "`rto-sentinel final-test`."
        )
        raise FileNotFoundError(msg)

    orders_path = settings.artifact_path / "datasets" / run_id / "orders.parquet"
    if not orders_path.is_file():
        msg = (
            f"no order attributes at {orders_path}. The cohort audit needs the dataset "
            "the model was scored on; regenerate it with `rto-sentinel build-dataset`."
        )
        raise FileNotFoundError(msg)

    scores = pd.read_parquet(scores_path)
    orders = pd.read_parquet(orders_path)
    merged = scores.merge(orders, on="order_id", how="inner", suffixes=("", "_order"))
    if len(merged) != len(scores):
        msg = (
            f"joined {len(merged):,} of {len(scores):,} scored orders to their attributes. "
            "A partial join would audit a subset nobody selected; refusing."
        )
        raise FileNotFoundError(msg)
    return merged[list(scores.columns)], merged, run_id


def _cmd_shift(args: argparse.Namespace) -> int:
    """Run the controlled distribution-shift study against the frozen model."""
    from rto_sentinel.configuration import (
        load_cost_model_config,
        load_features_config,
    )
    from rto_sentinel.eval.responsible_report import write_shift_artifacts
    from rto_sentinel.eval.shift import default_environments, run_shift_study
    from rto_sentinel.models import load_manifest
    from rto_sentinel.models.calibrated import CalibratedModel
    from rto_sentinel.models.experiment import cost_inputs_from_profile
    from rto_sentinel.serving.model_registry import ModelRegistry, ModelUnavailableError

    settings = get_settings()
    registry = ModelRegistry(settings.artifact_path)
    try:
        artefact, card = registry.resolve()
    except ModelUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 1

    model, _ = CalibratedModel.load(artefact)
    manifest = load_manifest(settings.artifact_path, card.dataset_run_id)

    from rto_sentinel.decision.cost_model import outcome_economics

    cost_config = load_cost_model_config(settings)
    inputs = cost_inputs_from_profile(cost_config.profiles[manifest.cost_profile])
    economics = outcome_economics(inputs)

    environments = default_environments(seed=args.seed or manifest.seed, n_orders=args.n_orders)
    print(f"\nmodel      : {card.model_name} v{card.model_version} (frozen, not retrained)")
    print(f"threshold  : {manifest.threshold:.4f} (frozen)")
    print(f"environments: {len(environments)} x {args.n_orders:,} orders\n")

    def announce(spec: object) -> None:
        print(f"  generating {spec.name}...", flush=True)  # type: ignore[attr-defined]

    study = run_shift_study(
        environments,
        model,
        generator_config=load_generator_config(settings),
        features_config=load_features_config(settings),
        splits_config=load_splits_config(settings),
        threshold=manifest.threshold,
        cost_false_positive_inr=economics.false_positive_cost_inr,
        saving_true_positive_inr=economics.true_positive_saving_inr,
        model_version=card.model_version,
        feature_version=card.feature_version,
        feature_names=tuple(card.feature_names),
        progress=announce,
    )

    print()
    print(
        f"  {'environment':24s} {'n':>7s} {'RTO':>7s} {'PR-AUC':>7s} {'lift':>7s} "
        f"{'dLift':>7s} {'ECE':>6s} {'flag':>6s} {'prec':>6s} {'net/1k':>9s} {'dNet':>9s}"
    )
    for result in study.results:
        print(
            f"  {result.environment:24s} {result.n_orders:7,d} "
            f"{result.observed_rto_rate:7.3f} {result.pr_auc:7.3f} "
            f"{result.pr_auc_lift:7.2f} "
            f"{_or_na(result.pr_auc_lift_delta, 7, 2)} "
            f"{result.expected_calibration_error:6.3f} {result.flag_rate:6.3f} "
            f"{_or_na(result.precision, 6, 3)} "
            f"{result.net_inr_per_1000:9,.0f} "
            f"{_or_na(result.net_delta, 9, 0)}"
        )

    print()
    for finding in study.findings:
        print(textwrap.fill(f"- {finding}", width=88, subsequent_indent="  "))

    if not args.no_write:
        for path in write_shift_artifacts(study, artifact_root=settings.artifact_path):
            print(f"\nwrote {path}")
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    """Compare a baseline period against a current period on the scored book.

    Windows are cut by order time within the frozen scored book, which is what
    makes this runnable offline and reproducible. In a live deployment the same
    functions take the last N days of scored traffic; the arithmetic is identical
    and the module has no idea where the frames came from.
    """
    import pandas as pd

    from rto_sentinel.eval.responsible_report import write_drift_artifacts
    from rto_sentinel.models import load_manifest
    from rto_sentinel.monitoring import build_drift_report

    settings = get_settings()
    try:
        scores, orders, run_id = _load_scores_with_attributes(settings, args.split)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    manifest = load_manifest(settings.artifact_path, run_id)

    frame = scores.copy()
    frame["ordered_at"] = pd.to_datetime(orders["ordered_at"].to_numpy())
    for column in ("order_value_inr", "discount_depth", "is_cod", "prior_rto_rate", "item_count"):
        if column in orders.columns:
            frame[column] = orders[column].to_numpy()

    frame = frame.sort_values("ordered_at").reset_index(drop=True)
    cut = int(len(frame) * args.baseline_fraction)
    baseline, current = frame.iloc[:cut], frame.iloc[cut:]

    if args.simulate_immature:
        # Reproduce the situation that actually occurs in production: the recent
        # window has not matured, so no labelled comparison is possible. The
        # report must say the question is unanswered rather than showing green.
        current = current.copy()
        current["label"] = pd.NA

    report = build_drift_report(
        baseline,
        current,
        threshold=manifest.threshold,
        feature_columns=[
            column
            for column in ("order_value_inr", "discount_depth", "is_cod", "prior_rto_rate")
            if column in frame.columns
        ],
        model_version=manifest.model_version,
        feature_version=manifest.feature_version,
    )

    print(f"\nbaseline: {report.baseline.n_orders:,} orders, {report.baseline.n_matured:,} matured")
    print(f"current : {report.current.n_orders:,} orders, {report.current.n_matured:,} matured")
    print(f"labels available for comparison: {report.labels_available}")
    print(f"worst severity: {report.worst_severity}\n")

    print(f"  {'kind':14s} {'quantity':30s} {'stat':20s} {'distance':>9s}  severity")
    for signal in report.signals:
        flag = "" if signal.sufficient else "  (thin)"
        print(
            f"  {signal.kind:14s} {signal.name:30s} {signal.statistic:20s} "
            f"{signal.distance:9.4f}  {signal.severity}{flag}"
        )

    if report.performance:
        print(f"\n  {'metric':24s} {'baseline':>10s} {'current':>10s} {'delta':>10s}")
        for delta in report.performance:
            print(
                f"  {delta.metric:24s} {delta.baseline:10.4f} {delta.current:10.4f} "
                f"{delta.delta:+10.4f}"
            )

    print()
    for warning in report.warnings:
        print(textwrap.fill(f"- {warning}", width=88, subsequent_indent="  "))

    if not args.no_write:
        for path in write_drift_artifacts(
            report, artifact_root=settings.artifact_path, dataset_run_id=run_id
        ):
            print(f"\nwrote {path}")
    return 0


def _cmd_responsible_report(args: argparse.Namespace) -> int:
    """Render docs/responsible_ai.md from the saved fairness, shift and drift artefacts."""
    from rto_sentinel.eval.responsible_report import render_responsible_report

    settings = get_settings()
    try:
        path = render_responsible_report(artifact_root=settings.artifact_path)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def _cmd_evaluation_report(args: argparse.Namespace) -> int:
    """Render docs/evaluation_report.md from every saved measurement artefact."""
    from rto_sentinel.eval.final_report import NoArtefactsError, render

    settings = get_settings()
    try:
        path = render(artifact_root=settings.artifact_path)
    except NoArtefactsError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rto-sentinel",
        description="Cost-calibrated return-to-origin risk scoring for Indian COD commerce.",
    )
    parser.add_argument("--version", action="version", version=f"rto-sentinel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    config_parser = sub.add_parser("config", help="configuration utilities")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("check", help="validate all configuration files").set_defaults(
        func=_cmd_config_check
    )

    generate = sub.add_parser("generate", help="build the synthetic benchmark dataset")
    _add_generation_arguments(generate)
    generate.add_argument(
        "--no-write", action="store_true", help="generate and validate without writing artefacts"
    )
    generate.set_defaults(func=_cmd_generate)

    validate = sub.add_parser("validate", help="re-validate a dataset artefact on disk")
    validate.add_argument("--run-id", default=None, help="dataset run id (default: most recent)")
    validate.add_argument("--lenient", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    db_parser = sub.add_parser("db", help="database utilities")
    db_sub = db_parser.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("upgrade", help="run migrations to head").set_defaults(func=_cmd_db_upgrade)
    stats = db_sub.add_parser("stats", help="query a loaded dataset back")
    stats.add_argument("--run-id", default=None)
    stats.set_defaults(func=_cmd_db_stats)

    seed = sub.add_parser("seed-db", help="migrate, generate, validate and load")
    _add_generation_arguments(seed)
    seed.add_argument("--skip-migrations", action="store_true")
    seed.set_defaults(func=_cmd_seed_db)

    features_parser = sub.add_parser("features", help="feature pipeline utilities")
    features_sub = features_parser.add_subparsers(dest="features_command", required=True)
    listing = features_sub.add_parser("list", help="print every feature and its definition")
    listing.add_argument("--verbose", action="store_true", help="include risk notes")
    listing.set_defaults(func=_cmd_features_list)
    features_sub.add_parser("docs", help="regenerate docs/features.md").set_defaults(
        func=_cmd_features_docs
    )

    build = sub.add_parser("build-dataset", help="generate, split and build the modelling dataset")
    _add_generation_arguments(build)
    build.add_argument("--no-write", action="store_true")
    build.set_defaults(func=_cmd_build_dataset)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    train = sub.add_parser("train", help="train and evaluate the whole baseline ladder")
    _add_generation_arguments(train)
    train.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    train.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="which split to evaluate on. 'test' unseals the sealed set - once, at the end.",
    )
    train.add_argument(
        "--unseal-reason",
        default=None,
        help="written justification, required in spirit when --split test is used",
    )
    train.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="bootstrap iterations for confidence intervals (0 disables, tests only)",
    )
    train.set_defaults(func=_cmd_train)

    final = sub.add_parser(
        "final",
        help="select, calibrate and freeze the final model (reads validation, never test)",
    )
    _add_generation_arguments(final)
    final.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    final.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="bootstrap iterations for confidence intervals (0 disables, tests only)",
    )
    final.set_defaults(func=_cmd_final)

    final_test = sub.add_parser(
        "final-test",
        help="score the SEALED test set once, using the frozen manifest and artefact",
    )
    _add_generation_arguments(final_test)
    final_test.add_argument(
        "--unseal-reason",
        required=True,
        help="written justification for opening the sealed split. Recorded in the artefact.",
    )
    final_test.add_argument(
        "--again",
        action="store_true",
        help="score the sealed set again despite a previous test evaluation for this manifest",
    )
    final_test.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    final_test.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="bootstrap iterations for confidence intervals (0 disables, tests only)",
    )
    final_test.set_defaults(func=_cmd_final_test)

    final_report = sub.add_parser(
        "final-report",
        help="re-render the model card and CSVs from saved artefacts (reads no split)",
    )
    final_report.set_defaults(func=_cmd_final_report)

    economics = sub.add_parser(
        "economics",
        help="price the shipped policy, sweep the threshold, and write the economic report",
    )
    economics.add_argument(
        "--profile", default=None, help="cost profile to price (default: the configured default)"
    )
    economics.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    economics.set_defaults(func=_cmd_economics)

    evaluate = sub.add_parser(
        "evaluate", help="re-render the comparison from saved experiment artefacts"
    )
    evaluate.set_defaults(func=_cmd_evaluate)

    fairness_parser = sub.add_parser(
        "fairness", help="cohort and disparate-impact audit over operational cohorts"
    )
    fairness_parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="which scored book to audit (default: validation)",
    )
    # Defaults of None mean "use config/evaluation.yaml", so the shipped audit
    # and an ad-hoc one differ only where the operator said so.
    fairness_parser.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="orders a group needs before its rates count as evidence (default: config)",
    )
    fairness_parser.add_argument(
        "--min-flagged",
        type=int,
        default=None,
        help="flagged orders a group needs before its precision counts (default: config)",
    )
    fairness_parser.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    fairness_parser.set_defaults(func=_cmd_fairness)

    shift_parser = sub.add_parser(
        "shift", help="controlled distribution-shift study against the frozen model"
    )
    shift_parser.add_argument(
        "--n-orders", type=int, default=8000, help="orders per environment (default: 8000)"
    )
    shift_parser.add_argument(
        "--seed", type=int, default=None, help="base seed (default: the manifest seed)"
    )
    shift_parser.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    shift_parser.set_defaults(func=_cmd_shift)

    monitor_parser = sub.add_parser(
        "monitor", help="compare a baseline period against a current period"
    )
    monitor_parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="which scored book to window (default: validation)",
    )
    monitor_parser.add_argument(
        "--baseline-fraction",
        type=float,
        default=0.6,
        help="share of the book, earliest first, that forms the baseline window",
    )
    monitor_parser.add_argument(
        "--simulate-immature",
        action="store_true",
        help="blank the current window's labels, reproducing an unmatured recent period",
    )
    monitor_parser.add_argument("--no-write", action="store_true", help="skip writing artefacts")
    monitor_parser.set_defaults(func=_cmd_monitor)

    evaluation_report = sub.add_parser(
        "evaluation-report",
        help="render docs/evaluation_report.md from every saved measurement artefact",
    )
    evaluation_report.set_defaults(func=_cmd_evaluation_report)

    responsible = sub.add_parser(
        "responsible-report",
        help="render docs/responsible_ai.md from saved fairness, shift and drift artefacts",
    )
    responsible.set_defaults(func=_cmd_responsible_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
