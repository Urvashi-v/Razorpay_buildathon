"""Maps ladder rung names to their implementations.

One lookup table so the training script, the evaluation harness and the API all
resolve a rung the same way. Adding a rung means touching this file and
``config/models/ladder.yaml`` and nothing else.

Rung 5 (a text encoder over address strings) is deliberately absent: it is
disabled in config and unimplemented, and registering a name with no
implementation would let a configuration typo produce a confusing failure much
later than it should.
"""

from __future__ import annotations

from rto_sentinel.models.base import RiskModel
from rto_sentinel.models.rung0_do_nothing import DoNothingModel
from rto_sentinel.models.rung1_blanket_block import BlanketCodBlockModel
from rto_sentinel.models.rung2_pincode_blocklist import PincodeBlocklistModel
from rto_sentinel.models.rung3_logistic import LogisticRegressionModel
from rto_sentinel.models.rung4_lightgbm import LightGbmModel

RUNG_REGISTRY: dict[str, type[RiskModel]] = {
    "do_nothing": DoNothingModel,
    "blanket_cod_block": BlanketCodBlockModel,
    "pincode_blocklist": PincodeBlocklistModel,
    "logistic_regression": LogisticRegressionModel,
    "lightgbm": LightGbmModel,
}


class UnknownRungError(KeyError):
    """Raised when a configured rung name has no implementation."""


def resolve_rung(name: str) -> type[RiskModel]:
    """Look up a rung implementation by its configured name."""
    try:
        return RUNG_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(RUNG_REGISTRY))
        msg = f"unknown ladder rung {name!r}; known rungs are: {known}"
        raise UnknownRungError(msg) from exc
