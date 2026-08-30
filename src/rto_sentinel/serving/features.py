"""Turning a stored order into the exact feature row the model was trained on.

THE HARD REQUIREMENT
====================
The row this produces must be identical to the row training produced for the same
order. Not similar - identical. A serving path that computes features slightly
differently from the training path is the classic silent failure: nothing errors,
the numbers move a little, and the model quietly stops being the model that was
evaluated.

So this module does not reimplement any feature. It reconstructs the *input
frame* from the database in the shape the generator produced, hands it to the
same :class:`~rto_sentinel.features.pipeline.FeaturePipeline` training used, and
takes the row it wants out of the result. Every transform, guard and as-of join
is the training code, unchanged.

``tests/api/test_serving_integration.py`` checks the claim rather than asserting
it: it scores an order through this service and compares the features against the
ones the offline pipeline builds for the same order.

WHY THE CONTEXT FRAME IS LARGER THAN ONE ROW
============================================
The feature families recompute customer history and geography aggregates from
the data. One row has no history and no aggregate, so scoring it alone would
produce a first-time customer in an unknown pincode - for every order, including
the ones the merchant has served fifty times.

:meth:`ServingRepository.context_frame` therefore loads the merchant's book up to
this order's ``ordered_at``, and the as-of machinery masks anything unresolved at
that instant. Including extra rows is safe precisely because that masking is what
the leakage suite tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.features.dataset import CONTEXT_COLUMNS
from rto_sentinel.features.pipeline import FEATURE_VERSION, FeaturePipeline

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import FeaturesConfig, GeneratorConfig
    from rto_sentinel.db.models import Order
    from rto_sentinel.db.repositories import ServingRepository


class FeatureServiceError(RuntimeError):
    """Raised when features cannot be built for an order that exists."""


@dataclass(frozen=True, slots=True)
class OrderFeatures:
    """One order's design-matrix row, plus what the heuristic layers need."""

    order_id: str
    #: Single-row frame, columns in the trained order.
    x: pd.DataFrame
    #: Raw operational columns the learned model is forbidden - pincode above
    #: all. Kept separate for the same reason it is separate in training.
    context: pd.DataFrame
    feature_version: str
    feature_fingerprint: str
    feature_names: tuple[str, ...]
    #: How many rows of merchant history the aggregates were computed over.
    #: Reported so a caller can tell a well-supported score from a cold-start one.
    context_rows: int

    @property
    def null_features(self) -> tuple[str, ...]:
        """Features with no value for this order.

        Not an error: a first-time customer genuinely has no prior RTO rate, and
        the model handles missingness natively. Surfaced because a score built
        mostly from nulls deserves less confidence than one that is not, and the
        API says so rather than leaving the caller to guess.
        """
        row = self.x.iloc[0]
        return tuple(str(name) for name in row.index[row.isna()])


class OrderFeatureService:
    """Builds the trained feature row for a stored order."""

    def __init__(
        self,
        repository: ServingRepository,
        *,
        features_config: FeaturesConfig,
        generator_config: GeneratorConfig,
        context_limit: int = 20000,
    ) -> None:
        self._repository = repository
        self._pipeline = FeaturePipeline(features_config, generator_config=generator_config)
        self._context_limit = context_limit

    @property
    def feature_fingerprint(self) -> str:
        return self._pipeline.feature_set.fingerprint()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self._pipeline.feature_set.names)

    @property
    def feature_version(self) -> str:
        return FEATURE_VERSION

    def build(self, order: Order) -> OrderFeatures:
        """Reconstruct the frame, run the training pipeline, take one row."""
        frame = self._repository.context_frame(order, limit=self._context_limit)
        if frame.empty:
            msg = (
                f"no context rows for order {order.order_id}. The order exists but its "
                "merchant history could not be read, so no feature row can be built."
            )
            raise FeatureServiceError(msg)

        prepared = self._prepare(frame)
        matrix = self._pipeline.build(prepared)

        mask = prepared[cols.ORDER_ID] == order.order_id
        if not mask.any():  # pragma: no cover - guarded upstream
            msg = f"order {order.order_id} vanished from its own context frame"
            raise FeatureServiceError(msg)

        position = int(mask.to_numpy().nonzero()[0][0])
        # The same CONTEXT_COLUMNS the training dataset carries, so the heuristic
        # rungs see exactly what they saw offline.
        context = prepared[list(CONTEXT_COLUMNS)].iloc[[position]].reset_index(drop=True)
        return OrderFeatures(
            order_id=order.order_id,
            x=matrix.matrix.iloc[[position]].reset_index(drop=True),
            context=context,
            feature_version=matrix.feature_version,
            feature_fingerprint=matrix.feature_fingerprint,
            feature_names=tuple(matrix.feature_names),
            context_rows=len(prepared),
        )

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce the SQL result into the dtypes the generator's frame had.

        The pipeline is strict about what it receives, and rightly so. SQLite
        stores booleans as integers and both engines hand back naive datetimes in
        some configurations; a timestamp that lost its timezone silently breaks
        every as-of comparison, so the coercion happens here, once, rather than
        being tolerated inside six feature families.
        """
        prepared = frame.copy()

        for column in (cols.ORDERED_AT, cols.RESOLVED_AT, "signup_at"):
            if column in prepared.columns:
                values = pd.to_datetime(prepared[column], utc=True, errors="coerce")
                prepared[column] = values

        boolean_columns = (
            cols.IS_COD,
            cols.CART_EDITED,
            cols.IS_LATE_NIGHT,
            cols.IS_SALE_DAY,
            cols.COD_AFTER_PREPAID_FAILURE,
            "addr_has_house_number",
            "addr_has_floor_number",
            "addr_has_landmark",
            "addr_pincode_city_consistent",
            "is_mature",
        )
        for column in boolean_columns:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("boolean").astype("object")

        # `is_rto` is deliberately nullable: an immature order has no outcome, and
        # `Int8` keeps NULL distinguishable from False. Casting it to bool here
        # would turn "not yet known" into "did not return", which is the single
        # most effective way to manufacture optimism in a benchmark.
        if cols.IS_RTO in prepared.columns:
            prepared[cols.IS_RTO] = prepared[cols.IS_RTO].astype("boolean")

        return prepared
