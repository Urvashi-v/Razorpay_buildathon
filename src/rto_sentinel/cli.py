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
    rto-sentinel train --rung 4               # train one ladder rung   (Phase 3)
    rto-sentinel evaluate                     # score a model           (Phase 4)
    rto-sentinel serve                        # run the API

Every generation records its seed, generator version, configuration snapshot and
creation timestamp, so any dataset can be traced back to the exact inputs that
produced it - and regenerated from them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from rto_sentinel import __version__
from rto_sentinel.configuration import (
    ConfigurationError,
    config_fingerprint,
    load_app_config,
    load_generator_config,
    load_splits_config,
)
from rto_sentinel.data.artifacts import latest_dataset_dir, read_dataset
from rto_sentinel.data.generator import SUPPORTED_GENERATOR_VERSIONS, GeneratorParams
from rto_sentinel.data.pipeline import build_dataset
from rto_sentinel.data.validation import validate_delivery_events, validate_orders
from rto_sentinel.settings import REPO_ROOT, Settings, get_settings

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

    for name, phase, help_text in (
        ("train", "Phase 3", "train one rung of the baseline ladder"),
        ("evaluate", "Phase 4", "score a model and write an evaluation report"),
    ):
        stub = sub.add_parser(name, help=f"{help_text} ({phase})")
        stub.set_defaults(func=_cmd_not_implemented, command=name, phase=phase)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
