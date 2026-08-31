"""Geography and courier-lane features. HIGHEST FAIRNESS RISK IN THE MODEL.

SPEC section 04. A raw per-pincode RTO rate is an income and region proxy, and
with enough trees it becomes a redlining machine.

THREE CONSTRAINTS, ENFORCED HERE RATHER THAN REMEMBERED
=======================================================

**1. Bayesian shrinkage toward the global mean.** A pincode with 3 orders and 2
returns does not have a 67% return rate; it has almost no evidence. The shrunk
estimate is::

    rate = (returns + k * prior) / (orders + k)

with ``k`` from ``config/generator.yaml``. A thin pincode collapses to the
population base rate, which is the honest statement about a place we know
nothing about.

**2. Minimum support.** Below ``min_support`` resolved orders the feature is
**NaN**, not the shrunk value. Shrinkage alone still leaks a little signal from
two or three orders, and a place should not acquire a reputation from three
deliveries.

**3. Computed as-of, never globally.** The rate for an order on day 60 uses only
orders that had **resolved** by day 60. A global rate computed over the whole
dataset would let a training row see outcomes from the test window - the exact
leak this project is built to prevent, and the one that target encoding
introduces by default in most codebases.

WHAT IS DELIBERATELY ABSENT
===========================
There is no raw ``pincode`` feature. It is in ``FORBIDDEN_IN_FEATURES``, and the
refused-pattern check in the pipeline would reject it. Only the smoothed
aggregate and the tier are permitted.

If the Phase 5 fairness audit finds tier-3 flagged materially more often *with
materially worse precision*, this is the family that gets pulled back - stronger
shrinkage, higher support, or disabled outright. It is isolated behind its own
config switch precisely so that remedy is a one-line change.

A COVERAGE PROBLEM, MEASURED AND UNRESOLVED
===========================================
``geo_pincode_rto_rate_smoothed`` is 99% null at 18,000 orders and 92% null at
60,000. The minimum-support guard is working exactly as intended - it refuses to
give a place a reputation from a handful of deliveries - but the generator
spreads orders across 2,200 pincodes, which leaves roughly 12 to 27 orders per
pincode across the entire horizon, and far fewer *resolved before any given
order*.

The feature is correctly implemented. The simulated geography is the problem:
real merchant volume concentrates heavily in a few hundred high-traffic pincodes
with a long tail, rather than spreading uniformly. Fixing that means changing the
generator's pincode volume distribution, which would change every dataset
fingerprint - a Phase 2 change, not a feature change, and not one to make
silently while the pipeline is being validated.

Lowering ``min_support`` would restore coverage and is the wrong fix: it buys
usable data by weakening the guard that stops a place acquiring a reputation from
three deliveries.

Expected outcome: an ablation would likely show this family contributing close
to nothing. **That ablation has not been run** - `eval/ablation.py` is an
unimplemented interface - so the sentence above is a prediction, not a result.
If it is ever run, the finding should be reported as it lands rather than
engineered away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.spec import (
    ALL_HISTORY_RESOLVED,
    Availability,
    FeatureSet,
    FeatureSpec,
    ObservationPoint,
)

FAMILY = "geography_route"

#: Fallbacks when no generator config is supplied. The real values come from
#: ``config/generator.yaml`` so the shrinkage strength is tunable without a code
#: change - which is what makes the fairness remedy a config edit.
DEFAULT_SHRINKAGE = 50.0
DEFAULT_MIN_SUPPORT = 30
DEFAULT_PRIOR_RATE = 0.20


class GeographyRouteFamily(FeatureFamily):
    """Where it is going, who is carrying it, and how those lanes have performed."""

    name = FAMILY

    @property
    def _shrinkage(self) -> float:
        if self.generator_config is None:
            return DEFAULT_SHRINKAGE
        return float(self.generator_config.geography.shrinkage_prior_strength)

    @property
    def _min_support(self) -> int:
        if self.config.min_support is not None:
            return int(self.config.min_support)
        if self.generator_config is None:
            return DEFAULT_MIN_SUPPORT
        return int(self.generator_config.geography.min_support_for_feature)

    @property
    def _prior_rate(self) -> float:
        if self.generator_config is None:
            return DEFAULT_PRIOR_RATE
        # The blended base rate the population is calibrated to, weighted by COD
        # share - the honest prior for a place we have no evidence about.
        rates = self.generator_config.base_rates
        cod_share = self.generator_config.payment.cod_share
        return float(cod_share * rates.rto_given_cod + (1 - cod_share) * rates.rto_given_prepaid)

    @property
    def feature_set(self) -> FeatureSet:
        support = self._min_support
        shrinkage = self._shrinkage
        return FeatureSet(
            (
                FeatureSpec(
                    name="geo_pincode_tier",
                    family=FAMILY,
                    dtype="category",
                    description="Pincode tier: tier_1, tier_2 or tier_3.",
                    source_columns=(cols.PINCODE_TIER,),
                    observation_point=ObservationPoint.ORDER_PAYLOAD,
                    availability=Availability.AT_ORDER_TIME,
                    risk_note=(
                        "A coarse, three-level geography signal - deliberately coarser than "
                        "a pincode. It is also the primary axis of the fairness audit, so its "
                        "effect is measured rather than assumed."
                    ),
                ),
                FeatureSpec(
                    name="geo_pincode_rto_rate_smoothed",
                    family=FAMILY,
                    dtype="float",
                    description=(
                        f"Return rate for this pincode over orders resolved before this one, "
                        f"shrunk toward the population rate with strength {shrinkage:g}. "
                        f"NaN below {support} resolved orders."
                    ),
                    source_columns=(cols.PINCODE, cols.ORDERED_AT, cols.RESOLVED_AT, cols.IS_RTO),
                    observation_point=ObservationPoint.POPULATION_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    monotonic="increasing",
                    # MEASURED, not guessed: 99% null at 18k orders, 92% at 60k.
                    # The support threshold is doing its job; the simulated
                    # geography is simply too fragmented for it to have coverage.
                    # See the coverage note in the module docstring.
                    expected_null_share=0.92,
                    risk_note=(
                        "THE HIGHEST-RISK FEATURE IN THE PROJECT. A raw pincode rate is an "
                        "income and region proxy. Three guards: Bayesian shrinkage, a minimum "
                        "support threshold below which it is NaN, and as-of computation so it "
                        "cannot read the future. Never permitted as a top-3 SHAP feature "
                        "without written justification in REPORT.md."
                    ),
                ),
                FeatureSpec(
                    name="geo_pincode_resolved_count",
                    family=FAMILY,
                    dtype="int",
                    description="Orders in this pincode that had resolved before this one.",
                    source_columns=(cols.PINCODE, cols.ORDERED_AT, cols.RESOLVED_AT),
                    observation_point=ObservationPoint.POPULATION_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    risk_note=(
                        "The evidence count behind the rate. Emitted so the model can "
                        "distinguish a well-observed pincode from a thin one, rather than "
                        "treating a shrunk estimate as equally reliable everywhere."
                    ),
                ),
                FeatureSpec(
                    name="geo_courier_partner",
                    family=FAMILY,
                    dtype="category",
                    description="Courier assigned to this order.",
                    source_columns=(cols.COURIER_PARTNER,),
                    observation_point=ObservationPoint.ORDER_PAYLOAD,
                    availability=Availability.AT_ORDER_TIME,
                    risk_note=(
                        "Low fairness risk - this is a merchant logistics choice, not a "
                        "customer attribute. Assigned at order time in this dataset; a "
                        "production system where the courier is picked after scoring would "
                        "have to drop it."
                    ),
                ),
                FeatureSpec(
                    name="geo_courier_rto_rate_smoothed",
                    family=FAMILY,
                    dtype="float",
                    description=(
                        "Return rate for this courier over orders resolved before this one, "
                        f"shrunk with strength {shrinkage:g}."
                    ),
                    source_columns=(
                        cols.COURIER_PARTNER,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.IS_RTO,
                    ),
                    observation_point=ObservationPoint.POPULATION_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    monotonic="increasing",
                    expected_null_share=0.05,
                    risk_note=(
                        "Lane performance is a property of the carrier, not the customer, so "
                        "the fairness concern is much weaker here than for pincode. Still "
                        "as-of computed: a global courier rate would leak test-window outcomes "
                        "into training."
                    ),
                ),
            )
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        out["geo_pincode_tier"] = frame[cols.PINCODE_TIER].astype("category")
        out["geo_courier_partner"] = frame[cols.COURIER_PARTNER].astype("category")

        pincode_rate, pincode_count = self._smoothed_rate(frame, key=cols.PINCODE)
        out["geo_pincode_rto_rate_smoothed"] = pincode_rate
        out["geo_pincode_resolved_count"] = pincode_count.astype("int64")

        courier_rate, _ = self._smoothed_rate(
            frame, key=cols.COURIER_PARTNER, min_support_override=1
        )
        out["geo_courier_rto_rate_smoothed"] = courier_rate

        return out[list(self.feature_set.names)]

    def _smoothed_rate(
        self,
        frame: pd.DataFrame,
        *,
        key: str,
        min_support_override: int | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """As-of, shrunk group rate and its evidence count.

        Implemented directly rather than through ``as_of_aggregate`` because both
        the count and the sum are needed to apply shrinkage before division, and
        the min-support rule needs the raw count.

        The sort is on **resolution** time. A cumulative sum read at the order's
        timestamp then gives the totals over everything that had actually come
        back by then.
        """
        shrinkage = self._shrinkage
        prior = self._prior_rate
        support = self._min_support if min_support_override is None else min_support_override

        left = pd.DataFrame(
            {
                "_key": frame[key].to_numpy(),
                "_order_time": frame[cols.ORDERED_AT].to_numpy(),
                "_row": np.arange(len(frame)),
            }
        ).sort_values(["_order_time", "_row"], kind="stable")

        resolved = frame[frame[cols.RESOLVED_AT].notna()]
        right = pd.DataFrame(
            {
                "_key": resolved[key].to_numpy(),
                "_resolution_time": resolved[cols.RESOLVED_AT].to_numpy(),
                "_label": resolved[cols.IS_RTO].astype("float64").to_numpy(),
            }
        ).sort_values("_resolution_time", kind="stable")

        if right.empty:
            empty_rate = pd.Series(np.nan, index=frame.index, dtype="float64")
            empty_count = pd.Series(0, index=frame.index, dtype="int64")
            return empty_rate, empty_count

        right["_count"] = right.groupby("_key", sort=False).cumcount() + 1
        right["_sum"] = right.groupby("_key", sort=False)["_label"].cumsum()

        merged = pd.merge_asof(
            left,
            right[["_resolution_time", "_key", "_count", "_sum"]],
            left_on="_order_time",
            right_on="_resolution_time",
            by="_key",
            direction="backward",
            allow_exact_matches=False,  # strict <: an outcome at this instant is not yet known
        )

        counts = merged["_count"].to_numpy(dtype="float64", na_value=0.0)
        sums = merged["_sum"].to_numpy(dtype="float64", na_value=0.0)

        shrunk = (sums + shrinkage * prior) / (counts + shrinkage)
        # Below minimum support the estimate is withheld entirely. Shrinkage alone
        # still passes a little signal through from two or three orders, and a
        # place should not acquire a reputation from three deliveries.
        shrunk = np.where(counts >= support, shrunk, np.nan)

        row_order = merged["_row"].to_numpy()
        rate = pd.Series(shrunk, index=row_order).sort_index().set_axis(frame.index)
        count = pd.Series(counts, index=row_order).sort_index().set_axis(frame.index)
        return rate, count
