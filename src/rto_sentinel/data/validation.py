"""Structural validation of a raw order table.

Runs immediately after generation and again after any load from disk. This is
cheap insurance: a silently truncated parquet file or a column renamed during a
refactor should fail here, loudly, rather than surface as a mysteriously good
model three steps later.

STATUS: Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class DataValidationError(ValueError):
    """Raised when an order table violates the declared schema or an invariant."""


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating one table. Empty ``errors`` means it passed."""

    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            msg = f"order table failed validation:\n  - {joined}"
            raise DataValidationError(msg)


def validate_orders(frame: pd.DataFrame, *, strict: bool = True) -> ValidationReport:
    """Check a raw order table against ``data.schema``.

    Checks to implement in Phase 2:

    * every column in ``RAW_COLUMNS`` is present, with the expected dtype;
    * ``order_id`` is unique;
    * ``resolved_at`` is null exactly when ``outcome`` is PENDING, and otherwise
      strictly after ``ordered_at``;
    * ``is_rto`` agrees with ``outcome``;
    * ``day_index`` is consistent with ``ordered_at`` and the horizon start;
    * no null in a non-nullable column;
    * realised COD and prepaid RTO rates sit within the generator's tolerance of
      the configured base rates - a drifting generator is a silent problem.

    ``strict=False`` downgrades base-rate drift from an error to a warning, for
    exploratory runs on deliberately small samples.
    """
    raise NotImplementedError("Order-table validation lands in Phase 2.")
