"""Assembles evaluation reports, and refuses to write a dishonest one.

The builder enforces the ``forbidden`` list from ``config/evaluation.yaml``
mechanically:

* no single accuracy figure;
* ROC-AUC present, but never in the lead position;
* precision never emitted without its flag rate;
* false-positive cost never folded into a net figure;
* no point estimate without an interval;
* the sealed test set scored at most once.

A report that violates any of these raises rather than renders. It is easy to
promise these things in a README and quietly break one under deadline pressure;
it is much harder to defeat a builder that refuses to emit the file.

STATUS: Phase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.evaluation import EvaluationReport


class DishonestReportError(ValueError):
    """Raised when a report would violate a rule in the evaluation config."""


def write_report(report: EvaluationReport, path: Path) -> None:
    """Validate and write one evaluation report to disk."""
    raise NotImplementedError("Report writing lands in Phase 4.")


def render_markdown(reports: list[EvaluationReport]) -> str:
    """Render the ladder comparison table for REPORT.md.

    Every rung, scored identically, on the same sealed test set - including the
    rungs that beat the model, if any of them do.
    """
    raise NotImplementedError("Report rendering lands in Phase 4.")
