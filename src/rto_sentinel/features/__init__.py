"""Feature engineering: one module per family, assembled by the pipeline.

Depends on: configuration, contracts, data. Knows nothing about models, the API
or the database.
"""

from rto_sentinel.features.address import AddressQualityFamily
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.customer_history import CustomerHistoryFamily
from rto_sentinel.features.geography import GeographyRouteFamily
from rto_sentinel.features.order_shape import OrderShapeFamily
from rto_sentinel.features.pipeline import (
    FeatureMatrix,
    FeaturePipeline,
    RefusedFeatureError,
    TargetLeakageError,
)
from rto_sentinel.features.session_intent import SessionIntentFamily

__all__ = [
    "AddressQualityFamily",
    "CustomerHistoryFamily",
    "FeatureFamily",
    "FeatureMatrix",
    "FeaturePipeline",
    "GeographyRouteFamily",
    "OrderShapeFamily",
    "RefusedFeatureError",
    "SessionIntentFamily",
    "TargetLeakageError",
]
