"""The data pipeline: generate, validate, split, validate again, write.

One function so the order of operations is fixed in exactly one place. The order
matters and is not arbitrary:

1. **Generate.** The simulator produces the frames.
2. **Validate the raw output.** Before anything is derived from it. A generator
   bug caught here is a puzzle; caught after splitting it is a mystery.
3. **Assign splits.** Temporal windows within customer pools.
4. **Validate again, post-split.** Because splitting is the step that can
   silently produce an empty or leaking split, and the checks for that need the
   split column to exist.
5. **Write.**

Every step is reported, including the ones that produce uncomfortable numbers -
the fraction of rows the split protocol drops is printed rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rto_sentinel.data import schema as cols
from rto_sentinel.data.artifacts import write_dataset
from rto_sentinel.data.generator import ConfiguredOrderGenerator, GenerationResult, GeneratorParams
from rto_sentinel.data.splits import (
    SplitAssignment,
    assign_splits,
    customers_are_disjoint,
    splits_are_time_ordered,
)
from rto_sentinel.data.validation import (
    ValidationReport,
    validate_delivery_events,
    validate_orders,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import GeneratorConfig, SplitsConfig


class PipelineError(RuntimeError):
    """Raised when a pipeline stage produces something unusable."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything one pipeline run produced, including its own report card."""

    dataset: GenerationResult
    splits: SplitAssignment
    order_report: ValidationReport
    event_report: ValidationReport
    output_dir: Path | None

    @property
    def ok(self) -> bool:
        return self.order_report.ok and self.event_report.ok

    def render(self) -> str:
        metadata = self.dataset.metadata
        orders = self.dataset.orders
        split_counts = self.splits.as_dict()
        retained = self.splits.n_modelling
        lines = [
            "dataset",
            f"  run id             : {metadata.run_id}",
            f"  generator version  : {metadata.generator_version}",
            f"  seed               : {metadata.seed}",
            f"  config fingerprint : {metadata.config_fingerprint[:16]}...",
            f"  created at         : {metadata.created_at.isoformat()}",
            f"  horizon            : {metadata.start_date.date()} to {metadata.end_date.date()}",
            "",
            "rows",
            f"  customers          : {len(self.dataset.customers):,}",
            f"  addresses          : {len(self.dataset.addresses):,}",
            f"  orders             : {len(orders):,}",
            f"  delivery events    : {len(self.dataset.delivery_events):,}",
            "",
            "outcomes (mature orders only)",
            f"  COD RTO rate       : {metadata.realised_rto_rate_cod:.4f}",
            f"  prepaid RTO rate   : {metadata.realised_rto_rate_prepaid:.4f}",
            f"  COD share          : {metadata.realised_cod_share:.4f}",
            f"  mature / immature  : {metadata.n_mature:,} / {metadata.n_immature:,}",
            "",
            "splits",
            f"  train              : {split_counts['train']:,}",
            f"  validation         : {split_counts['validation']:,}",
            f"  test               : {split_counts['test']:,}",
            f"  excluded (immature): {split_counts['excluded_immature']:,}",
            f"  excluded (protocol): {split_counts['excluded_group_protocol']:,}",
            f"  retained for models: {retained:,} ({retained / max(len(orders), 1):.1%})",
            "",
            "validation",
            f"  orders  : {len(self.order_report.errors)} errors, "
            f"{len(self.order_report.warnings)} warnings",
            f"  events  : {len(self.event_report.errors)} errors, "
            f"{len(self.event_report.warnings)} warnings",
        ]
        for message in self.order_report.errors + self.event_report.errors:
            lines.append(f"    ERROR   {message}")
        for message in self.order_report.warnings + self.event_report.warnings:
            lines.append(f"    WARNING {message}")
        if self.output_dir is not None:
            lines += ["", f"written to {self.output_dir}"]
        return "\n".join(lines)


def build_dataset(
    *,
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
    params: GeneratorParams,
    artifact_root: Path | None = None,
    strict: bool = True,
) -> PipelineResult:
    """Run the whole pipeline and return everything it produced.

    ``strict`` controls only base-rate drift tolerance, which is dominated by
    sampling noise on small datasets. Structural checks are never relaxed: a
    broken timestamp is broken at any sample size.
    """
    result = ConfiguredOrderGenerator().generate(generator_config, params)

    # --- 2. validate the raw output ------------------------------------------
    order_report = validate_orders(
        result.orders,
        config=generator_config,
        customers=result.customers,
        strict=strict,
    )
    event_report = validate_delivery_events(result.delivery_events, result.orders)

    # --- 3. splits -----------------------------------------------------------
    assignment = assign_splits(result.orders, splits_config)
    result.orders[cols.SPLIT] = assignment.labels

    # --- 4. post-split checks -------------------------------------------------
    # These cannot run earlier because they need the split column, and they are
    # the checks that catch the failure modes splitting itself introduces.
    if not customers_are_disjoint(result.orders):
        order_report.add_error(
            "customers appear in more than one modelling split; the grouped-split "
            "protocol has been violated"
        )
    if not splits_are_time_ordered(result.orders):
        order_report.add_error("split day ranges overlap; the temporal ordering has been violated")
    for name in ("train", "validation", "test"):
        if assignment.as_dict()[name] == 0:
            order_report.add_error(f"the {name} split is empty")

    output_dir = write_dataset(result, artifact_root) if artifact_root is not None else None

    return PipelineResult(
        dataset=result,
        splits=assignment,
        order_report=order_report,
        event_report=event_report,
        output_dir=output_dir,
    )
