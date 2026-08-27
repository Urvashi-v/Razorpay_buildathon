"""Data layer: generation, validation, as-of joins and split assignment.

Depends on: configuration, contracts. Depends on nothing downstream - the data
layer has no idea a model or an API exists.
"""

from rto_sentinel.data.asof import as_of_aggregate, assert_no_future_information
from rto_sentinel.data.generator import ConfiguredOrderGenerator, GenerationResult, OrderGenerator
from rto_sentinel.data.splits import SealBrokenError, SplitAssignment, TestSetSeal, assign_splits
from rto_sentinel.data.validation import DataValidationError, ValidationReport, validate_orders

__all__ = [
    "ConfiguredOrderGenerator",
    "DataValidationError",
    "GenerationResult",
    "OrderGenerator",
    "SealBrokenError",
    "SplitAssignment",
    "TestSetSeal",
    "ValidationReport",
    "as_of_aggregate",
    "assert_no_future_information",
    "assign_splits",
    "validate_orders",
]
