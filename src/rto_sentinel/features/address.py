"""Address-quality features.

SPEC section 04. Genuinely predictive and genuinely fair: a delivery address
missing a house number really does fail to deliver, and that has nothing to do
with who the customer is.

THE LINE THIS FAMILY WALKS
==========================
Every feature here measures **structural completeness** - is there a house
number, does the typed city match the pincode, are there tokens that look like
keyboard mash. None measures fluency, spelling, or language. An address written
in imperfect English is not a risky address, and a feature that conflated the two
would be a literacy proxy dressed as a delivery signal.

That distinction is enforced upstream, in ``data/address.py``, where the
observable signals are derived from text alone with no access to the latent
quality that produced it. This family only assembles them.

The one feature here with real as-of content is
``addr_is_new_for_customer``: whether this customer has used this address before.
That runs on the **placed** clock - the merchant knows an address was used
before, regardless of how those orders turned out.

The fairness audit in Phase 5 checks whether these features concentrate flags on
tier-3 pincodes without a matching precision. If they do, the remedy is here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.data.asof import as_of_aggregate
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.spec import (
    ALL_HISTORY_PLACED,
    Availability,
    FeatureSet,
    FeatureSpec,
    ObservationPoint,
)

FAMILY = "address_quality"

#: Components of the composite completeness score, and their weights. A house
#: number matters most: it is the component whose absence most reliably fails a
#: delivery. Weights sum to 1.0 so the score reads as a fraction.
COMPLETENESS_WEIGHTS = {
    cols.ADDR_HAS_HOUSE_NUMBER: 0.40,
    cols.ADDR_PINCODE_CITY_CONSISTENT: 0.30,
    cols.ADDR_HAS_LANDMARK: 0.20,
    cols.ADDR_HAS_FLOOR_NUMBER: 0.10,
}


def _payload_spec(
    name: str, dtype: str, description: str, source: str, risk_note: str
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family=FAMILY,
        dtype=dtype,  # type: ignore[arg-type]
        description=description,
        source_columns=(source,),
        observation_point=ObservationPoint.ORDER_PAYLOAD,
        availability=Availability.AT_ORDER_TIME,
        risk_note=risk_note,
    )


class AddressQualityFamily(FeatureFamily):
    """Structural completeness of the address the customer typed."""

    name = FAMILY

    @property
    def feature_set(self) -> FeatureSet:
        return FeatureSet(
            (
                _payload_spec(
                    "addr_token_count",
                    "int",
                    "Whitespace- and comma-separated tokens in the address line.",
                    cols.ADDR_TOKEN_COUNT,
                    "A very short address is usually an incomplete one. Measures length, not "
                    "quality of language.",
                ),
                _payload_spec(
                    "addr_has_house_number",
                    "bool",
                    "True when a house or building number is present.",
                    cols.ADDR_HAS_HOUSE_NUMBER,
                    "The component whose absence most reliably fails a delivery. Fair: a "
                    "courier genuinely cannot find an unnumbered building.",
                ),
                _payload_spec(
                    "addr_has_floor_number",
                    "bool",
                    "True when a floor or flat is present.",
                    cols.ADDR_HAS_FLOOR_NUMBER,
                    "Weaker than a house number and irrelevant for standalone houses, so it "
                    "carries the smallest weight in the composite score.",
                ),
                _payload_spec(
                    "addr_has_landmark",
                    "bool",
                    "True when a landmark reference is present.",
                    cols.ADDR_HAS_LANDMARK,
                    "How deliveries actually get completed across much of India. Its presence "
                    "is a positive signal, and its absence is weak evidence rather than a "
                    "red flag.",
                ),
                _payload_spec(
                    "addr_pincode_city_consistent",
                    "bool",
                    "True when the typed city matches the city the pincode belongs to.",
                    cols.ADDR_PINCODE_CITY_CONSISTENT,
                    "One of the most common structural defects in Indian address data, and "
                    "causally linked to delivery failure - a parcel addressed to the wrong "
                    "city struggles to arrive. Checkable against a lookup table, so it is a "
                    "fact rather than an inference about the person.",
                ),
                _payload_spec(
                    "addr_allcaps_ratio",
                    "float",
                    "Share of alphabetic tokens typed entirely in capitals.",
                    cols.ADDR_ALLCAPS_RATIO,
                    "WATCH THIS ONE. All-caps entry is common on older keyboards and among "
                    "less confident typists, so it risks proxying for something other than "
                    "deliverability. Kept because it is cheap to audit, and it is the first "
                    "candidate for removal if the fairness audit trips.",
                ),
                _payload_spec(
                    "addr_gibberish_ratio",
                    "float",
                    "Share of tokens that are keyboard mash or autocomplete garbage.",
                    cols.ADDR_GIBBERISH_RATIO,
                    "Matches a fixed list of consonant runs, not a language model. Cannot "
                    "flag an unfamiliar-looking real word, which is the failure mode a "
                    "learned gibberish detector would have.",
                ),
                FeatureSpec(
                    name="addr_completeness_score",
                    family=FAMILY,
                    dtype="float",
                    description=(
                        "Weighted share of present address components: house number 0.40, "
                        "pincode/city consistency 0.30, landmark 0.20, floor 0.10."
                    ),
                    source_columns=tuple(COMPLETENESS_WEIGHTS),
                    observation_point=ObservationPoint.ORDER_PAYLOAD,
                    availability=Availability.AT_ORDER_TIME,
                    monotonic="decreasing",
                    risk_note=(
                        "A hand-set composite, not a learned one. Weights are a judgement, "
                        "stated here so they can be argued with; the components are also "
                        "emitted individually so the model is not forced to accept them."
                    ),
                ),
                FeatureSpec(
                    name="addr_is_new_for_customer",
                    family=FAMILY,
                    dtype="bool",
                    description="True when this customer has not ordered to this address before.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ADDRESS_FINGERPRINT, cols.ORDERED_AT),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    risk_note=(
                        "Placed-clock: the merchant knows an address was used before "
                        "regardless of how those orders ended. A first delivery to a new "
                        "address is genuinely riskier - but so is every order from a new "
                        "customer, so this must not simply re-encode the cold-start cohort. "
                        "Reported alongside it in the evaluation."
                    ),
                ),
            )
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        out["addr_token_count"] = frame[cols.ADDR_TOKEN_COUNT].astype("int64")
        out["addr_has_house_number"] = frame[cols.ADDR_HAS_HOUSE_NUMBER].astype(bool)
        out["addr_has_floor_number"] = frame[cols.ADDR_HAS_FLOOR_NUMBER].astype(bool)
        out["addr_has_landmark"] = frame[cols.ADDR_HAS_LANDMARK].astype(bool)
        out["addr_pincode_city_consistent"] = frame[cols.ADDR_PINCODE_CITY_CONSISTENT].astype(bool)
        out["addr_allcaps_ratio"] = frame[cols.ADDR_ALLCAPS_RATIO].astype("float64")
        out["addr_gibberish_ratio"] = frame[cols.ADDR_GIBBERISH_RATIO].astype("float64")

        score = np.zeros(len(frame), dtype="float64")
        for column, weight in COMPLETENESS_WEIGHTS.items():
            score += frame[column].astype(bool).to_numpy(dtype="float64") * weight
        out["addr_completeness_score"] = score

        out["addr_is_new_for_customer"] = self._address_is_new(frame)
        return out[list(self.feature_set.names)]

    @staticmethod
    def _address_is_new(frame: pd.DataFrame) -> pd.Series:
        """Whether this (customer, address) pair has been used before.

        Grouped on the pair and counted on the placed clock, so a prior order to
        the same address counts from the moment it was placed.
        """
        working = frame[[cols.CUSTOMER_HASH, cols.ADDRESS_FINGERPRINT, cols.ORDERED_AT]].copy()
        working["_pair"] = (
            working[cols.CUSTOMER_HASH].astype(str)
            + "|"
            + working[cols.ADDRESS_FINGERPRINT].astype(str)
        )
        working["_placed_clock"] = working[cols.ORDERED_AT]

        prior_uses = as_of_aggregate(
            working,
            group_key="_pair",
            value_column=None,
            order_time_column=cols.ORDERED_AT,
            resolution_time_column="_placed_clock",
            aggregate="count",
        )
        return prior_uses == 0
