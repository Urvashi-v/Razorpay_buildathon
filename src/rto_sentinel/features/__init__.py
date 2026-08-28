"""Feature engineering: one module per family, assembled by the pipeline.

Depends on: configuration, contracts, data. Knows nothing about models, the API
or the database.

Every feature carries its own answer to "would this exist at checkout?" as data
rather than as a comment - see :mod:`rto_sentinel.features.spec`. The pipeline
refuses to emit any feature whose answer is no.
"""

from rto_sentinel.features.address import AddressQualityFamily
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.customer_history import CustomerHistoryFamily
from rto_sentinel.features.dataset import (
    ModelingDataset,
    SplitView,
    TestSetAccessError,
    attach_customer_dimension,
    build_modeling_dataset,
)
from rto_sentinel.features.geography import GeographyRouteFamily
from rto_sentinel.features.order_shape import OrderShapeFamily
from rto_sentinel.features.pipeline import (
    FAMILY_REGISTRY,
    FEATURE_VERSION,
    FeatureContractError,
    FeatureMatrix,
    FeaturePipeline,
    RefusedFeatureError,
    TargetLeakageError,
)
from rto_sentinel.features.session_intent import SessionIntentFamily
from rto_sentinel.features.spec import (
    Availability,
    FeatureSet,
    FeatureSpec,
    LookbackWindow,
    ObservationPoint,
)
from rto_sentinel.features.temporal import TemporalFamily

__all__ = [
    "FAMILY_REGISTRY",
    "FEATURE_VERSION",
    "AddressQualityFamily",
    "Availability",
    "CustomerHistoryFamily",
    "FeatureContractError",
    "FeatureFamily",
    "FeatureMatrix",
    "FeaturePipeline",
    "FeatureSet",
    "FeatureSpec",
    "GeographyRouteFamily",
    "LookbackWindow",
    "ModelingDataset",
    "ObservationPoint",
    "OrderShapeFamily",
    "RefusedFeatureError",
    "SessionIntentFamily",
    "SplitView",
    "TargetLeakageError",
    "TemporalFamily",
    "TestSetAccessError",
    "attach_customer_dimension",
    "build_modeling_dataset",
]
