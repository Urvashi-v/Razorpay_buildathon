"""The baseline ladder, rungs 0-5, plus calibration and artefact handling.

Depends on: configuration, contracts, features. Knows nothing about the decision
engine, the API, the database or the agent layer. A model produces a probability
and stops there.
"""

from rto_sentinel.models.base import HeuristicModel, RiskModel
from rto_sentinel.models.calibration import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
)
from rto_sentinel.models.registry import RUNG_REGISTRY, UnknownRungError, resolve_rung
from rto_sentinel.models.rung0_do_nothing import DoNothingModel
from rto_sentinel.models.rung1_blanket_block import BlanketCodBlockModel
from rto_sentinel.models.rung2_pincode_blocklist import PincodeBlocklistModel
from rto_sentinel.models.rung3_logistic import LogisticRegressionModel
from rto_sentinel.models.rung4_lightgbm import LightGbmModel
from rto_sentinel.models.rung5_address_text import LightGbmAddressTextModel

__all__ = [
    "RUNG_REGISTRY",
    "BlanketCodBlockModel",
    "Calibrator",
    "DoNothingModel",
    "HeuristicModel",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "LightGbmAddressTextModel",
    "LightGbmModel",
    "LogisticRegressionModel",
    "PincodeBlocklistModel",
    "RiskModel",
    "UnknownRungError",
    "resolve_rung",
]
