"""Data layer: generation, validation, as-of joins and split assignment.

Depends on: configuration, contracts. Depends on nothing downstream - the data
layer has no idea a model, an API or a database exists.
"""

from rto_sentinel.data.address import (
    AddressSignals,
    RenderedAddress,
    address_fingerprint,
    observable_signals,
    render_address,
)
from rto_sentinel.data.artifacts import (
    ArtifactError,
    DatasetArtifact,
    latest_dataset_dir,
    read_dataset,
    write_dataset,
)
from rto_sentinel.data.asof import (
    AsOfLeakageError,
    as_of_aggregate,
    assert_no_future_information,
    brute_force_as_of,
)
from rto_sentinel.data.generator import (
    SUPPORTED_GENERATOR_VERSIONS,
    ConfiguredOrderGenerator,
    DatasetRunMetadata,
    GenerationResult,
    GeneratorParams,
    OrderGenerator,
    UnsupportedGeneratorVersionError,
)
from rto_sentinel.data.pipeline import PipelineError, PipelineResult, build_dataset
from rto_sentinel.data.splits import (
    SealBrokenError,
    SplitAssignment,
    TestSetSeal,
    assign_splits,
    customers_are_disjoint,
    drift_window_mask,
    split_summary,
)
from rto_sentinel.data.validation import (
    DataValidationError,
    ValidationReport,
    validate_delivery_events,
    validate_orders,
)

__all__ = [
    "SUPPORTED_GENERATOR_VERSIONS",
    "AddressSignals",
    "ArtifactError",
    "AsOfLeakageError",
    "ConfiguredOrderGenerator",
    "DataValidationError",
    "DatasetArtifact",
    "DatasetRunMetadata",
    "GenerationResult",
    "GeneratorParams",
    "OrderGenerator",
    "PipelineError",
    "PipelineResult",
    "RenderedAddress",
    "SealBrokenError",
    "SplitAssignment",
    "TestSetSeal",
    "UnsupportedGeneratorVersionError",
    "ValidationReport",
    "address_fingerprint",
    "as_of_aggregate",
    "assert_no_future_information",
    "assign_splits",
    "brute_force_as_of",
    "build_dataset",
    "customers_are_disjoint",
    "drift_window_mask",
    "latest_dataset_dir",
    "observable_signals",
    "read_dataset",
    "render_address",
    "split_summary",
    "validate_delivery_events",
    "validate_orders",
    "write_dataset",
]
