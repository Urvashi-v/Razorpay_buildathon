"""The feature-family abstraction.

Every family in ``config/features.yaml`` is one class implementing
:class:`FeatureFamily`. That is what makes the ablation study in SPEC section 07
a configuration change rather than a code change: switch a family off and the
pipeline stops calling it.

Two rules bind every implementation:

* **As-of discipline.** Any family whose config sets ``as_of: true`` must build
  its aggregates through ``rto_sentinel.data.asof``. It may not compute a
  groupby over the whole frame.
* **No refused columns.** A family may not emit a column matching a pattern in
  the ``refused`` block of the config. The pipeline checks this after every
  family runs, so a well-meaning addition cannot slip a name-derived or raw
  pincode feature into the matrix.

STATUS: Phase 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.configuration.schemas import FeatureFamilyConfig


class FeatureFamily(ABC):
    """Base class for one coherent group of features."""

    #: Must match a key under ``families`` in config/features.yaml.
    name: str

    def __init__(self, config: FeatureFamilyConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def output_columns(self) -> tuple[str, ...]:
        """Columns this family emits. Declared up front so the pipeline can
        detect collisions and refused patterns before any computation runs."""

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like ``frame`` holding only this family's columns.

        Implementations must be pure: no mutation of the input, no reliance on
        state carried between calls, no I/O. That is what makes a family
        independently testable and independently ablatable.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, enabled={self.config.enabled})"
