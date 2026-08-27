"""Shared, typed contracts between every layer.

Import direction is one-way: contracts depend on nothing inside this package
except each other. Data, features, models, decision, api, db and agents all
depend on contracts. That keeps a schema change visible in one place rather than
diffused across the codebase.
"""

from rto_sentinel.contracts.decision import (
    BandBoundary,
    CostInputs,
    Decision,
    OpsOverride,
    ThresholdDerivation,
)
from rto_sentinel.contracts.enums import (
    BAND_ORDER,
    DatasetSplit,
    DeviceClass,
    InterventionAction,
    OrderOutcome,
    OverrideDirection,
    PaymentMethod,
    PincodeTier,
    RiskBand,
    band_rank,
)
from rto_sentinel.contracts.evaluation import (
    CalibrationMetrics,
    CohortResult,
    EconomicResult,
    EvaluationReport,
    FairnessAudit,
    PointEstimate,
    RankingMetrics,
)
from rto_sentinel.contracts.explanation import (
    AddressRepairSuggestion,
    ConfirmationMessage,
    DigestSection,
    Explanation,
    GroundedOutput,
    MerchantDigest,
    ReasonCode,
)
from rto_sentinel.contracts.orders import (
    AddressPayload,
    OrderLineItem,
    OrderOutcomeUpdate,
    OrderPayload,
    SessionContext,
)
from rto_sentinel.contracts.risk import FeatureContribution, ModelCard, RiskScore

__all__ = [
    "BAND_ORDER",
    "AddressPayload",
    "AddressRepairSuggestion",
    "BandBoundary",
    "CalibrationMetrics",
    "CohortResult",
    "ConfirmationMessage",
    "CostInputs",
    "DatasetSplit",
    "Decision",
    "DeviceClass",
    "DigestSection",
    "EconomicResult",
    "EvaluationReport",
    "Explanation",
    "FairnessAudit",
    "FeatureContribution",
    "GroundedOutput",
    "InterventionAction",
    "MerchantDigest",
    "ModelCard",
    "OpsOverride",
    "OrderLineItem",
    "OrderOutcome",
    "OrderOutcomeUpdate",
    "OrderPayload",
    "OverrideDirection",
    "PaymentMethod",
    "PincodeTier",
    "PointEstimate",
    "RankingMetrics",
    "ReasonCode",
    "RiskBand",
    "RiskScore",
    "SessionContext",
    "ThresholdDerivation",
    "band_rank",
]
