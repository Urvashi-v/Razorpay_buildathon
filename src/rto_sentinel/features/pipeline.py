"""Assembles feature families into a design matrix, and polices what gets in.

This module is the single choke point between raw orders and any model. Three
guarantees are enforced here, once, rather than trusted to five separate family
implementations:

1. **No target leakage.** Nothing in ``data.schema.FORBIDDEN_IN_FEATURES``
   reaches the matrix - not the label, not the resolution timestamp, not an
   identity column.
2. **No refused features.** Nothing matching a pattern in the ``refused`` block
   of ``config/features.yaml`` reaches the matrix. Name-derived features, raw
   pincode categoricals, protected attributes and cross-merchant signals are
   rejected by name, with the configured reason quoted in the error.
3. **Consistent column order.** Training and inference build the matrix through
   the same call, so a reordered column cannot silently change what the model
   reads at serving time.

STATUS: Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.configuration.schemas import FeaturesConfig
    from rto_sentinel.features.base import FeatureFamily


class RefusedFeatureError(ValueError):
    """Raised when a family tries to emit a feature this project refuses to use.

    The error message quotes the reason from ``config/features.yaml`` verbatim,
    so whoever hits it reads *why* rather than just *no*.
    """


class TargetLeakageError(ValueError):
    """Raised when a forbidden column reaches the design matrix."""


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A design matrix plus the metadata a model artefact needs to reproduce it."""

    matrix: pd.DataFrame
    feature_names: tuple[str, ...]
    families_used: tuple[str, ...]
    n_rows: int


class FeaturePipeline:
    """Builds the design matrix from the enabled families.

    Constructed from config so that an ablation run - the leave-one-family-out
    study in SPEC section 07 - is a matter of flipping ``enabled`` and rebuilding,
    with no code path that differs between the ablated and full runs.
    """

    def __init__(self, config: FeaturesConfig, families: list[FeatureFamily] | None = None) -> None:
        self.config = config
        self._families = families

    def build(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Run every enabled family and assemble the validated matrix."""
        raise NotImplementedError("Feature assembly lands in Phase 2.")

    def check_refused(self, columns: list[str]) -> None:
        """Reject any column matching a refused pattern.

        Called after every family and again on the assembled matrix. Cheap, and
        the redundancy is deliberate: this is the check that keeps a fairness
        commitment from decaying into a comment.
        """
        raise NotImplementedError("Refused-feature checking lands in Phase 2.")
