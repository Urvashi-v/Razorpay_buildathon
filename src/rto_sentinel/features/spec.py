"""Feature specifications: what every feature is, and when it is knowable.

THE PROBLEM THIS SOLVES
=======================
"Is this feature available at scoring time?" is the only question that matters in
a risk pipeline, and it is almost never written down. It lives in someone's head,
survives one refactor, and then a plausible-looking column quietly starts reading
the future.

So here every feature carries its own answer, as data rather than as a comment:

* ``source_columns`` - what it reads.
* ``observation_point`` - the instant its value is fixed.
* ``lookback`` - how far back it looks, when it aggregates.
* ``availability`` - whether a production system would genuinely have it at
  checkout.

That last field is not decoration. ``FeaturePipeline`` refuses to emit any
feature whose availability is not ``AT_ORDER_TIME``, and
``tests/leakage/test_feature_timestamp_integrity.py`` checks the declarations
against the data itself. A feature declared available that behaves otherwise
fails a test rather than shipping.

THE OBSERVATION POINT IS THE WHOLE GAME
=======================================
Two features can read the same column and differ entirely:

* "orders this customer has **placed** in the last 30 days" reads ``ordered_at``
  and is knowable instantly - the merchant watched them happen.
* "orders this customer has had **returned** in the last 30 days" reads
  ``resolved_at`` and is knowable only once those orders came back. An order
  placed on day 40 that returns on day 47 is invisible on day 42.

Both are legitimate. Confusing them is the single most common leak in this class
of problem, so ``ObservationPoint`` makes the distinction explicit and
``LookbackWindow`` records which clock the window runs on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class ObservationPoint(StrEnum):
    """The instant at which a feature's value becomes fixed."""

    #: Known from the order payload itself. No history, no aggregation.
    ORDER_PAYLOAD = "order_payload"

    #: Aggregated over the customer's prior orders, keyed by when those orders
    #: were **placed**. Legitimately available: the merchant saw them happen.
    PRIOR_ORDERS_PLACED = "prior_orders_placed"

    #: Aggregated over the customer's prior orders, keyed by when those orders
    #: **resolved**. This is the strict one - an order that has not come back yet
    #: contributes nothing.
    PRIOR_ORDERS_RESOLVED = "prior_orders_resolved"

    #: Aggregated across all customers' resolved orders, as-of this instant.
    #: Population statistics such as a smoothed per-pincode rate.
    POPULATION_RESOLVED = "population_resolved"

    #: Fixed when the customer account was created.
    CUSTOMER_RECORD = "customer_record"

    #: NOT knowable at scoring time. Declared so it can be refused explicitly.
    POST_ORDER = "post_order"


class Availability(StrEnum):
    """Whether a production system would have this value at checkout."""

    AT_ORDER_TIME = "at_order_time"
    POST_ORDER = "post_order"


#: Which clock a lookback window runs on. ``placed`` counts orders by their
#: ``ordered_at``; ``resolved`` counts them by ``resolved_at``. A window over
#: outcomes MUST use ``resolved`` - using ``placed`` is exactly the leak the
#: whole as-of apparatus exists to prevent.
WindowClock = Literal["placed", "resolved"]


@dataclass(frozen=True, slots=True)
class LookbackWindow:
    """A bounded historical window, and the clock it runs on."""

    days: int | None  # None means "all history"
    clock: WindowClock

    def __str__(self) -> str:
        span = "all history" if self.days is None else f"{self.days}d"
        return f"{span} by {self.clock}"


ALL_HISTORY_PLACED = LookbackWindow(days=None, clock="placed")
ALL_HISTORY_RESOLVED = LookbackWindow(days=None, clock="resolved")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The complete definition of one feature.

    ``risk_note`` is required and non-empty for a reason: a feature nobody can
    articulate a risk for has not been thought about. Several of the notes here
    say "low risk", which is a fine answer - the point is that someone wrote it.
    """

    name: str
    family: str
    dtype: Literal["float", "int", "bool", "category"]
    description: str
    source_columns: tuple[str, ...]
    observation_point: ObservationPoint
    availability: Availability
    lookback: LookbackWindow | None = None
    risk_note: str = ""
    #: Direction the model is permitted to learn, where the project is willing to
    #: defend one. Used for monotonic constraints in Phase 4.
    monotonic: Literal["increasing", "decreasing"] | None = None
    #: Fraction of rows expected to be null, roughly. Documentation only - a
    #: feature that is 99% null when it claims 5% is a bug worth noticing.
    expected_null_share: float = 0.0

    def __post_init__(self) -> None:
        if not self.risk_note.strip():
            msg = f"feature {self.name!r} has no risk note; every feature needs one"
            raise ValueError(msg)
        if not self.source_columns:
            msg = f"feature {self.name!r} declares no source columns"
            raise ValueError(msg)
        # A feature aggregating over history must say how far back it looks.
        aggregating = self.observation_point in {
            ObservationPoint.PRIOR_ORDERS_PLACED,
            ObservationPoint.PRIOR_ORDERS_RESOLVED,
            ObservationPoint.POPULATION_RESOLVED,
        }
        if aggregating and self.lookback is None:
            msg = f"feature {self.name!r} aggregates over history but declares no lookback"
            raise ValueError(msg)
        # An outcome-derived window must run on the resolution clock.
        if (
            self.lookback is not None
            and self.observation_point
            in {ObservationPoint.PRIOR_ORDERS_RESOLVED, ObservationPoint.POPULATION_RESOLVED}
            and self.lookback.clock != "resolved"
        ):
            msg = (
                f"feature {self.name!r} aggregates resolved outcomes but its lookback runs "
                f"on the {self.lookback.clock!r} clock. An order placed earlier has not "
                "necessarily come back yet."
            )
            raise ValueError(msg)

    @property
    def is_available_at_prediction_time(self) -> bool:
        return self.availability is Availability.AT_ORDER_TIME

    def as_row(self) -> dict[str, str]:
        """Flat form, for the generated feature dictionary."""
        return {
            "name": self.name,
            "family": self.family,
            "dtype": self.dtype,
            "description": self.description,
            "source_columns": ", ".join(self.source_columns),
            "observation_point": str(self.observation_point),
            "lookback": str(self.lookback) if self.lookback else "n/a",
            "available_at_prediction_time": "yes" if self.is_available_at_prediction_time else "NO",
            "monotonic": self.monotonic or "-",
            "risk_note": self.risk_note,
        }


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """An ordered, validated collection of specs.

    Order is fixed and meaningful: training and inference build the design matrix
    through the same object, so a reordered column cannot silently change what
    the model reads at serving time.
    """

    specs: tuple[FeatureSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.specs]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            msg = f"duplicate feature names: {sorted(duplicates)}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for spec in self.specs:
            if spec.family not in seen:
                seen.append(spec.family)
        return tuple(seen)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs if spec.dtype == "category")

    def by_family(self, family: str) -> FeatureSet:
        return FeatureSet(tuple(spec for spec in self.specs if spec.family == family))

    def get(self, name: str) -> FeatureSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        msg = f"unknown feature {name!r}"
        raise KeyError(msg)

    def unavailable_at_prediction_time(self) -> tuple[FeatureSpec, ...]:
        """Any spec that would leak. Should always be empty in a shipped set."""
        return tuple(spec for spec in self.specs if not spec.is_available_at_prediction_time)

    def fingerprint(self) -> str:
        """SHA-256 over the feature definitions.

        Changes when a feature is added, removed, renamed, or has its definition
        altered - so a model artefact can be tied to the exact feature set it was
        trained on. A silently changed lookback window would otherwise be
        invisible in a results table.
        """
        payload = json.dumps([spec.as_row() for spec in self.specs], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def merge(self, other: FeatureSet) -> FeatureSet:
        return FeatureSet(self.specs + other.specs)
