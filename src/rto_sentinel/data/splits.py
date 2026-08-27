"""Dataset splitting and the test-set seal.

SPEC section 03. Three rules, implemented together because they interact:

1. **Temporal**, not random. Train on days 1-126, validate on 127-147, test on
   148-180.
2. **Grouped** on top. No ``customer_hash`` may appear in more than one split.
   Where a customer straddles a temporal boundary, their rows are resolved to a
   single split rather than duplicated - the resolution rule is deterministic and
   recorded, so the same seed always yields the same assignment.
3. **Sealed**. The test set is scored exactly once, after the threshold has been
   fixed on validation. :class:`TestSetSeal` makes that mechanical: it writes a
   receipt on first use and refuses a second scoring run.

STATUS: Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.configuration.schemas import SplitsConfig


class SealBrokenError(RuntimeError):
    """Raised on an attempt to score the sealed test set more than once."""


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """The result of splitting: a per-row label plus the counts to report."""

    labels: pd.Series
    n_train: int
    n_validation: int
    n_test: int
    n_excluded_immature: int
    n_customers_reassigned: int


def assign_splits(frame: pd.DataFrame, config: SplitsConfig) -> SplitAssignment:
    """Assign every row to train, validation, test, or the immature-tail bucket.

    Order of operations matters and is fixed:

    1. Drop rows whose label is not yet mature (SPEC rule 5) into
       ``EXCLUDED_IMMATURE``. Doing this *first* means an immature row can never
       be counted toward a split's size.
    2. Assign remaining rows by ``day_index`` against the temporal windows.
    3. Enforce customer disjointness, reassigning straddling customers to their
       earliest split so that no future behaviour bleeds backwards.
    """
    raise NotImplementedError("Split assignment lands in Phase 2.")


class TestSetSeal:
    """Enforces "scored exactly once" as code rather than as a promise.

    The receipt records what was scored, when, and under which configuration
    fingerprint. Deleting it to re-score is possible - it is a deliberate,
    visible act in the working tree, which is the point.
    """

    def __init__(self, receipt_path: Path) -> None:
        self._receipt_path = receipt_path

    @property
    def is_broken(self) -> bool:
        return self._receipt_path.is_file()

    def claim(self, *, model_name: str, config_fingerprint: str) -> None:
        """Record a test-set scoring run, or raise if one already happened."""
        raise NotImplementedError("Seal accounting lands in Phase 2.")
