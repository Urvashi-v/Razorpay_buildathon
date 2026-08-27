"""Dataset splitting and the test-set seal.

SPEC section 03. Three rules that interact, so they are implemented together.

1. **Temporal, not random.** Train on days 1-126, validate on 127-147, test on
   148-180. A random split leaks the future into the past.
2. **Grouped on top.** No ``customer_hash`` may appear in more than one split.
3. **Sealed.** The test set is scored exactly once, after the threshold has been
   fixed on validation.

HOW THE CONFLICT BETWEEN RULES 1 AND 2 IS RESOLVED, AND WHY IT MATTERS
----------------------------------------------------------------------
A customer who orders in both the training window and the test window violates
rule 2. There are three ways out and only one of them is sound.

**Move their orders into their earliest split.** Wrong. It drags genuinely later
orders backwards into train, putting future rows in the training set - breaking
rule 1 to satisfy rule 2, which reintroduces exactly the leakage the temporal
split exists to prevent.

**Assign each customer to their earliest split and drop their later orders.**
Satisfies both rules, and was this module's first implementation. It is *badly
biased*, which is worth stating plainly because the bias is invisible in the
split counts: validation and test end up composed almost entirely of customers
who had no orders in the training window. Measured on a 20,000-order sample, the
training set was 43% first-time customers while validation and test were 87% and
88%. A threshold fitted on that is fitted on cold-start orders, and a test score
measured on it is a cold-start score wearing a general-performance label.

**Partition customers into disjoint pools first, then apply the temporal window
within each pool.** What this module does. Pool assignment is a deterministic
hash of the customer identifier and a fixed salt, so it is independent of
behaviour, reproducible, and identical across runs. Every split therefore keeps
the same new-versus-repeat mix as the population.

The cost is rows: a customer in the validation pool contributes only their orders
inside the validation window, and the rest are dropped as
``excluded_group_protocol``. Roughly half the dataset ends up outside the three
modelling splits. That is the real price of enforcing both rules honestly, it is
reported in :class:`SplitAssignment` rather than hidden, and it is why the
default dataset is large.

Note what is *not* lost: a dropped row still contributed to its customer's as-of
history, so a validation-window order retains the prior-order counts earned by
that customer's earlier, excluded orders. That is correct - at serving time the
merchant genuinely does know that history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rto_sentinel.contracts.enums import DatasetSplit
from rto_sentinel.data import schema as cols

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import SplitsConfig

_SPLIT_RANK: dict[str, int] = {
    DatasetSplit.TRAIN.value: 0,
    DatasetSplit.VALIDATION.value: 1,
    DatasetSplit.TEST.value: 2,
}


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
    n_excluded_group_protocol: int
    n_customers_truncated: int

    def as_dict(self) -> dict[str, int]:
        return {
            "train": self.n_train,
            "validation": self.n_validation,
            "test": self.n_test,
            "excluded_immature": self.n_excluded_immature,
            "excluded_group_protocol": self.n_excluded_group_protocol,
            "customers_truncated": self.n_customers_truncated,
        }

    @property
    def n_modelling(self) -> int:
        return self.n_train + self.n_validation + self.n_test


def customer_pool(customer_hash: str, config: SplitsConfig) -> str:
    """Which split pool a customer belongs to.

    A deterministic hash of the identifier and a fixed salt, mapped onto the
    configured pool shares. Deterministic so the assignment is identical on every
    machine and every run; salted so the pools can be rotated deliberately if a
    future experiment needs a different partition.

    Crucially this depends only on the identifier - never on the customer's order
    count, timing, or outcomes. A pool assignment that looked at behaviour would
    reintroduce the very selection bias this scheme exists to remove.
    """
    digest = hashlib.sha256(f"{config.group.pool_salt}:{customer_hash}".encode()).digest()
    # First 8 bytes as a big-endian integer, scaled into [0, 1).
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)

    cumulative = 0.0
    for name in (
        DatasetSplit.TRAIN.value,
        DatasetSplit.VALIDATION.value,
        DatasetSplit.TEST.value,
    ):
        cumulative += config.group.pool_shares[name]
        if position < cumulative:
            return name
    return DatasetSplit.TEST.value  # pragma: no cover - the shares sum to 1


def assign_splits(frame: pd.DataFrame, config: SplitsConfig) -> SplitAssignment:
    """Assign every row to a split, or to one of the two exclusion buckets.

    Order of operations is fixed and matters:

    1. Rows whose label is not yet mature go to ``excluded_immature`` **first**,
       so an immature row can never be counted toward a split's size.
    2. Each customer is assigned to a pool by :func:`customer_pool`.
    3. A row joins a modelling split only when its ``day_index`` falls inside its
       own pool's temporal window. Everything else becomes
       ``excluded_group_protocol``.
    """
    labels = pd.Series(DatasetSplit.EXCLUDED_IMMATURE.value, index=frame.index, dtype="object")

    # --- 1. label maturity ---------------------------------------------------
    mature = frame[cols.IS_MATURE].astype(bool)
    if config.label_maturity.exclude_immature_tail:
        eligible = mature
    else:  # pragma: no cover - the config validator forbids this today
        eligible = pd.Series(True, index=frame.index)

    labels[eligible] = DatasetSplit.EXCLUDED_GROUP_PROTOCOL.value

    # --- 2. customer pools ---------------------------------------------------
    # Computed once per distinct customer rather than once per row: the hash is
    # cheap, but there are far fewer customers than orders.
    unique_customers = frame[cols.CUSTOMER_HASH].drop_duplicates()
    pool_by_customer = {customer: customer_pool(customer, config) for customer in unique_customers}
    pools = frame[cols.CUSTOMER_HASH].map(pool_by_customer)

    # --- 3. temporal window, within the customer's own pool ------------------
    day = frame[cols.DAY_INDEX].astype("int64")
    windows = (
        (DatasetSplit.TRAIN.value, config.temporal.train_days),
        (DatasetSplit.VALIDATION.value, config.temporal.validation_days),
        (DatasetSplit.TEST.value, config.temporal.test_days),
    )
    for name, (start, end) in windows:
        in_split = eligible & (pools == name) & (day >= start) & (day <= end)
        labels[in_split] = name

    counts = labels.value_counts()

    # Customers who contributed at least one row to a modelling split but also had
    # rows dropped by the window rule. Reported so the cost of the protocol is
    # visible rather than left to be inferred.
    assigned_mask = labels.isin(_SPLIT_RANK)
    contributing = set(frame.loc[assigned_mask, cols.CUSTOMER_HASH])
    dropped = set(
        frame.loc[labels == DatasetSplit.EXCLUDED_GROUP_PROTOCOL.value, cols.CUSTOMER_HASH]
    )

    return SplitAssignment(
        labels=labels,
        n_train=int(counts.get(DatasetSplit.TRAIN.value, 0)),
        n_validation=int(counts.get(DatasetSplit.VALIDATION.value, 0)),
        n_test=int(counts.get(DatasetSplit.TEST.value, 0)),
        n_excluded_immature=int(counts.get(DatasetSplit.EXCLUDED_IMMATURE.value, 0)),
        n_excluded_group_protocol=int(counts.get(DatasetSplit.EXCLUDED_GROUP_PROTOCOL.value, 0)),
        n_customers_truncated=len(contributing & dropped),
    )


def customers_are_disjoint(frame: pd.DataFrame, split_column: str = cols.SPLIT) -> bool:
    """True when no customer appears in more than one modelling split."""
    modelling = frame[frame[split_column].isin(_SPLIT_RANK)]
    if modelling.empty:
        return True
    per_customer = modelling.groupby(cols.CUSTOMER_HASH, sort=False)[split_column].nunique()
    return bool((per_customer <= 1).all())


def splits_are_time_ordered(frame: pd.DataFrame, split_column: str = cols.SPLIT) -> bool:
    """True when every train row precedes every validation row, and so on.

    Checked on the produced dataset rather than only on the config, because a
    correct configuration applied by a buggy assignment is still a leak.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for name in _SPLIT_RANK:
        subset = frame.loc[frame[split_column] == name, cols.DAY_INDEX]
        if not subset.empty:
            bounds[name] = (int(subset.min()), int(subset.max()))

    ordered = [bounds[name] for name in _SPLIT_RANK if name in bounds]
    return all(earlier[1] < later[0] for earlier, later in pairwise(ordered))


def drift_window_mask(frame: pd.DataFrame, config: SplitsConfig) -> pd.Series:
    """Rows in the final N days of the horizon, for the drift check.

    SPEC section 07 asks for performance on the final two weeks alone, because a
    model that has quietly gone stale still looks healthy on a full-period
    average.
    """
    last_day = int(frame[cols.DAY_INDEX].max())
    cutoff = last_day - config.drift_window.final_days + 1
    mask: pd.Series = frame[cols.DAY_INDEX].astype("int64") >= cutoff
    return mask


class TestSetSeal:
    """Enforces "scored exactly once" as code rather than as a promise.

    The receipt records what was scored, when, and under which configuration
    fingerprint. Deleting it to re-score is possible - it is a deliberate, visible
    act in the working tree, which is the point. The alternative, an honour
    system, has a perfect record of failing at 2am the night before a deadline.
    """

    #: Stops pytest trying to collect this as a test class because of its name.
    __test__ = False

    def __init__(self, receipt_path: Path) -> None:
        self._receipt_path = receipt_path

    @property
    def receipt_path(self) -> Path:
        return self._receipt_path

    @property
    def is_broken(self) -> bool:
        return self._receipt_path.is_file()

    def read_receipt(self) -> dict[str, object] | None:
        if not self.is_broken:
            return None
        loaded: dict[str, object] = json.loads(self._receipt_path.read_text(encoding="utf-8"))
        return loaded

    def claim(self, *, model_name: str, config_fingerprint: str, dataset_run_id: str) -> None:
        """Record a test-set scoring run, or raise if one already happened."""
        if self.is_broken:
            existing = self.read_receipt() or {}
            msg = (
                "The sealed test set has already been scored once, by "
                f"{existing.get('model_name')!r} at {existing.get('scored_at')}. "
                "Scoring it again invalidates the evaluation. If this is deliberate, "
                f"delete {self._receipt_path} explicitly."
            )
            raise SealBrokenError(msg)

        self._receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "model_name": model_name,
            "config_fingerprint": config_fingerprint,
            "dataset_run_id": dataset_run_id,
            "scored_at": datetime.now(UTC).isoformat(),
            "note": (
                "The test set is scored exactly once, after the operating threshold "
                "has been fixed on validation."
            ),
        }
        self._receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def split_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Row counts per split, for reporting."""
    counts = frame[cols.SPLIT].value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def rows_per_day(frame: pd.DataFrame) -> pd.Series:
    """Order volume by day index, used to sanity-check the sale-day calendar."""
    grouped: pd.Series = frame.groupby(cols.DAY_INDEX, sort=True).size()
    return grouped.astype(np.int64)
