"""Building a :class:`ModelingDataset` from a generated benchmark dataset.

The one place where raw orders become something a model can be trained on. Fixed
order of operations, because each step depends on the last being correct:

1. **Join the customer dimension.** Account age needs ``signup_at``, which lives
   on the customer table.
2. **Compute features over the FULL frame.** Every row, including later splits.
   Features are as-of, so this is safe - and filtering first would be worse. See
   the note in ``features/pipeline.py``.
3. **Drop unlabelled rows.** Immature orders have no known outcome. They were
   essential in step 2 - a pending order still tells you the customer placed an
   order - but they cannot be training examples.
4. **Drop rows outside the three modelling splits.** The group-protocol
   exclusions from Phase 2.
5. **Assemble and validate the contract.**

Step 2 before step 3 is the subtle one. Dropping immature orders first would
delete them from every customer's history, so a customer who ordered last week
would look like they had not. The merchant knows they ordered; only the outcome
is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from rto_sentinel.contracts.dataset import DatasetMetadata
from rto_sentinel.data import schema as cols
from rto_sentinel.features.pipeline import FEATURE_VERSION, FeaturePipeline
from rto_sentinel.features.spec import FeatureSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import (
        FeaturesConfig,
        GeneratorConfig,
        SplitsConfig,
    )
    from rto_sentinel.data.generator import GenerationResult

MODELLING_SPLITS = ("train", "validation", "test")
SplitName = Literal["train", "validation", "test"]


class TestSetAccessError(RuntimeError):
    """Raised on an attempt to read the sealed test split without unsealing it.

    SPEC section 03, rule 3: the test set is scored exactly once, at the end,
    after the operating threshold has been fixed on validation.

    The seal is not paranoia about malice. It is about the 2am accident: a quick
    ``dataset.test`` in a notebook to "just check", which contaminates the one
    number the whole submission rests on and leaves no trace that it happened. An
    honour system has a perfect record of failing under deadline pressure; a
    property that raises does not.
    """

    #: Stops pytest trying to collect this as a test class because of its name.
    __test__ = False


@dataclass(frozen=True, slots=True)
class SplitView:
    """One split's features, label and timestamps, kept aligned.

    A dataclass rather than a tuple so a caller cannot get the order wrong, and
    frozen so a downstream step cannot mutate the training set in place and leave
    the next consumer with something different.
    """

    name: SplitName
    x: pd.DataFrame
    y: pd.Series
    order_ids: pd.Series
    customer_hashes: pd.Series
    ordered_at: pd.Series
    day_index: pd.Series

    def __len__(self) -> int:
        return len(self.x)

    @property
    def n_rows(self) -> int:
        return len(self.x)

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else float("nan")

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.ordered_at.min(), self.ordered_at.max())

    @property
    def day_range(self) -> tuple[int, int]:
        return (int(self.day_index.min()), int(self.day_index.max()))

    def describe(self) -> str:
        start, end = self.date_range
        first_day, last_day = self.day_range
        return (
            f"{self.name:<11} rows={self.n_rows:>7,}  "
            f"positives={int(self.y.sum()):>6,} ({self.positive_rate:.4f})  "
            f"days {first_day:>3}-{last_day:<3}  "
            f"{start.date()} to {end.date()}  "
            f"customers={self.customer_hashes.nunique():,}"
        )


@dataclass
class ModelingDataset:
    """Features, label, timestamps, splits and provenance, bound together.

    Construct via :func:`build_modeling_dataset` below.

    Lives here rather than in ``contracts/`` because it holds pandas frames and a
    ``FeatureSet``. ``contracts/`` is the base of the dependency graph and stays
    light; the serialisable provenance half is
    :class:`~rto_sentinel.contracts.dataset.DatasetMetadata`.
    """

    features: pd.DataFrame
    labels: pd.Series
    splits: pd.Series
    order_ids: pd.Series
    customer_hashes: pd.Series
    ordered_at: pd.Series
    day_index: pd.Series
    feature_set: FeatureSet
    metadata: DatasetMetadata

    _test_unsealed_reason: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        lengths = {
            "features": len(self.features),
            "labels": len(self.labels),
            "splits": len(self.splits),
            "order_ids": len(self.order_ids),
            "ordered_at": len(self.ordered_at),
        }
        if len(set(lengths.values())) != 1:
            msg = f"dataset components have mismatched lengths: {lengths}"
            raise ValueError(msg)
        if self.labels.isna().any():
            msg = (
                "the modelling dataset contains unlabelled rows. Immature orders must be "
                "excluded before construction, never carried with a null label."
            )
            raise ValueError(msg)
        declared = list(self.feature_set.names)
        if list(self.features.columns) != declared:
            msg = "feature frame columns do not match the declared feature set"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # split access
    # ------------------------------------------------------------------

    def _view(self, name: SplitName) -> SplitView:
        mask = self.splits == name
        return SplitView(
            name=name,
            x=self.features.loc[mask],
            y=self.labels.loc[mask],
            order_ids=self.order_ids.loc[mask],
            customer_hashes=self.customer_hashes.loc[mask],
            ordered_at=self.ordered_at.loc[mask],
            day_index=self.day_index.loc[mask],
        )

    @property
    def train(self) -> SplitView:
        return self._view("train")

    @property
    def validation(self) -> SplitView:
        return self._view("validation")

    @property
    def test(self) -> SplitView:
        """The sealed test split. Raises unless :meth:`unseal_test` was called.

        SPEC section 03: scored exactly once, at the end, after the threshold has
        been fixed on validation.
        """
        if self._test_unsealed_reason is None:
            msg = (
                "The test split is sealed. It is scored exactly once, at the end, after the "
                "operating threshold has been fixed on validation.\n"
                "If this really is that moment, call dataset.unseal_test(reason=...) and say "
                "why. The reason is recorded on the dataset.\n"
                "If you wanted a held-out set to iterate against, you wanted .validation."
            )
            raise TestSetAccessError(msg)
        return self._view("test")

    def unseal_test(self, *, reason: str) -> None:
        """Unlock the test split, recording why.

        Deliberately awkward. The friction is the feature: it converts an
        accidental read into a decision someone had to write a sentence about.
        """
        if not reason or not reason.strip():
            msg = "unsealing the test set requires a written reason"
            raise ValueError(msg)
        self._test_unsealed_reason = reason.strip()

    @property
    def test_is_sealed(self) -> bool:
        return self._test_unsealed_reason is None

    @property
    def unseal_reason(self) -> str | None:
        return self._test_unsealed_reason

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    @property
    def split_counts(self) -> dict[str, int]:
        return {str(name): int(count) for name, count in self.splits.value_counts().items()}

    def describe(self) -> str:
        """A summary that reads the test split's *shape* without unsealing it.

        Row counts and date ranges are not outcomes, so reporting them costs
        nothing. The label is what stays sealed.
        """
        lines = ["dataset", *[f"  {line}" for line in self.metadata.summary_lines()], "", "splits"]
        for name in ("train", "validation", "test"):
            mask = self.splits == name
            subset_days = self.day_index.loc[mask]
            subset_dates = self.ordered_at.loc[mask]
            if subset_days.empty:
                lines.append(f"  {name:<11} rows=0")
                continue
            sealed_note = "" if name != "test" or not self.test_is_sealed else "  [SEALED]"
            lines.append(
                f"  {name:<11} rows={int(mask.sum()):>7,}  "
                f"days {int(subset_days.min()):>3}-{int(subset_days.max()):<3}  "
                f"{subset_dates.min().date()} to {subset_dates.max().date()}  "
                f"customers={self.customer_hashes.loc[mask].nunique():,}{sealed_note}"
            )
        lines += ["", f"features        : {len(self.feature_set)}"]
        lines.append(f"families        : {', '.join(self.feature_set.families)}")
        return "\n".join(lines)


def attach_customer_dimension(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Left-join the customer record onto the order table.

    Only ``signup_at`` is pulled across. The customer table also carries a
    ``generated_order_count`` - the customer's *total* order count over the whole
    horizon - which is a future-looking aggregate and must never reach a feature.
    Naming the wanted column explicitly rather than merging the whole frame is
    what keeps it out.
    """
    if "signup_at" not in customers.columns:
        return orders.copy()

    dimension = customers[[cols.CUSTOMER_HASH, "signup_at"]]
    merged = orders.merge(dimension, on=cols.CUSTOMER_HASH, how="left", validate="many_to_one")
    merged.index = orders.index
    return merged


def build_modeling_dataset(
    result: GenerationResult,
    *,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
    split_labels: pd.Series | None = None,
) -> ModelingDataset:
    """Turn a generated dataset into a validated, contract-bound modelling dataset."""
    orders = result.orders.copy()
    if split_labels is not None:
        orders[cols.SPLIT] = split_labels
    if cols.SPLIT not in orders.columns:
        msg = "the order table has no split column; assign splits before building features"
        raise ValueError(msg)

    # --- 1. customer dimension ------------------------------------------
    enriched = attach_customer_dimension(orders, result.customers)

    # --- 2. features over the full frame --------------------------------
    pipeline = FeaturePipeline(features_config, generator_config)
    matrix = pipeline.build(enriched)

    # --- 3 and 4. keep only labelled rows inside a modelling split -------
    labelled = orders[cols.IS_RTO].notna()
    in_split = orders[cols.SPLIT].isin(MODELLING_SPLITS)
    keep = labelled & in_split

    features = matrix.matrix.loc[keep].reset_index(drop=True)
    subset = orders.loc[keep].reset_index(drop=True)

    metadata = DatasetMetadata(
        dataset_run_id=result.metadata.run_id,
        generator_version=result.metadata.generator_version,
        seed=result.metadata.seed,
        config_fingerprint=result.metadata.config_fingerprint,
        feature_version=FEATURE_VERSION,
        feature_fingerprint=matrix.feature_fingerprint,
        families_used=matrix.families_used,
        split_strategy=splits_config.strategy,
        split_pool_shares=dict(splits_config.group.pool_shares),
        split_pool_salt=splits_config.group.pool_salt,
        train_days=splits_config.temporal.train_days,
        validation_days=splits_config.temporal.validation_days,
        test_days=splits_config.temporal.test_days,
    )

    return ModelingDataset(
        features=features,
        labels=subset[cols.IS_RTO].astype(bool),
        splits=subset[cols.SPLIT].astype(str),
        order_ids=subset[cols.ORDER_ID],
        customer_hashes=subset[cols.CUSTOMER_HASH],
        ordered_at=subset[cols.ORDERED_AT],
        day_index=subset[cols.DAY_INDEX].astype(int),
        feature_set=matrix.feature_set,
        metadata=metadata,
    )
