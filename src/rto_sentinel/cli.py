"""Command-line entry point.

Subcommands mirror the pipeline stages, so the whole system is reproducible from
a shell without a notebook anywhere in the loop::

    rto-sentinel config check      # validate every YAML file and print the fingerprint
    rto-sentinel generate          # build the synthetic dataset      (Phase 2)
    rto-sentinel split             # assign train/validation/test     (Phase 2)
    rto-sentinel train --rung 4    # train one ladder rung            (Phase 3)
    rto-sentinel evaluate          # score on validation              (Phase 4)
    rto-sentinel evaluate --sealed # the single, final test-set run   (Phase 4)
    rto-sentinel serve             # run the API

``config check`` is implemented now, because a configuration that does not parse
should be discoverable before anything else is built on top of it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rto_sentinel import __version__
from rto_sentinel.configuration import ConfigurationError, config_fingerprint, load_app_config
from rto_sentinel.settings import get_settings


def _cmd_config_check(_: argparse.Namespace) -> int:
    """Validate every configuration file and report the bundle fingerprint."""
    settings = get_settings()
    try:
        config = load_app_config(settings)
    except ConfigurationError as exc:
        print(f"configuration invalid:\n{exc}", file=sys.stderr)
        return 1

    fingerprint = config_fingerprint(settings)
    print("configuration OK")
    print(f"  config dir       : {settings.config_path}")
    print(f"  fingerprint      : {fingerprint}")
    print(f"  generator orders : {config.generator.horizon.n_orders:,}")
    print(
        f"  split (days)     : train {config.splits.temporal.train_days}, "
        f"val {config.splits.temporal.validation_days}, "
        f"test {config.splits.temporal.test_days}"
    )
    print(f"  feature families : {', '.join(config.features.enabled_families)}")
    print(f"  refused patterns : {len(config.features.refused_patterns)}")
    print(f"  cost profiles    : {', '.join(sorted(config.cost_model.profiles))}")
    print(f"  default profile  : {config.cost_model.default_profile}")
    print(f"  ladder rungs     : {', '.join(r.name for r in config.ladder.rungs if r.enabled)}")
    print(f"  primary metric   : {config.evaluation.primary_metric}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rto-sentinel",
        description="Cost-calibrated return-to-origin risk scoring for Indian COD commerce.",
    )
    parser.add_argument("--version", action="version", version=f"rto-sentinel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    config_parser = sub.add_parser("config", help="configuration utilities")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    check = config_sub.add_parser("check", help="validate all configuration files")
    check.set_defaults(func=_cmd_config_check)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="auto-reload on source changes")
    serve.set_defaults(func=_cmd_serve)

    for name, phase, help_text in (
        ("generate", "Phase 2", "build the synthetic dataset"),
        ("split", "Phase 2", "assign train/validation/test splits"),
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
