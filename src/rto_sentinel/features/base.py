"""The feature-family abstraction.

Every family in ``config/features.yaml`` is one class implementing
:class:`FeatureFamily`. That is what makes the ablation study in SPEC section 07
a configuration change rather than a code change: switch a family off and the
pipeline stops calling it.

THREE RULES BIND EVERY IMPLEMENTATION
=====================================

**Declare before you compute.** A family publishes its :class:`FeatureSet`
without touching data, so the pipeline can check names, availability and refused
patterns *before* anything is calculated. A leak caught at declaration time is a
typo; the same leak caught after training is a retracted result.

**As-of discipline.** Any family aggregating over history must go through
``rto_sentinel.data.asof``. It may not compute a groupby over the whole frame,
because a plain groupby includes orders that had not resolved yet.

**Emit exactly what you declared.** ``transform`` must return the columns named
in ``feature_set``, no more and no fewer. The pipeline verifies this, which is
what stops a family quietly adding a debugging column that nobody audits.

Implementations are pure: no mutation of the input, no state carried between
calls, no I/O. That is what makes a family independently testable and
independently ablatable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.configuration.schemas import FeatureFamilyConfig, GeneratorConfig
    from rto_sentinel.features.spec import FeatureSet


class FeatureFamily(ABC):
    """Base class for one coherent group of features."""

    #: Must match a key under ``families`` in config/features.yaml.
    name: str

    def __init__(
        self,
        config: FeatureFamilyConfig,
        generator_config: GeneratorConfig | None = None,
    ) -> None:
        self.config = config
        # Some families need generator-level constants - the shrinkage strength
        # and minimum support for geography, for instance. Passed explicitly
        # rather than re-read from disk so a family stays a pure function of its
        # inputs.
        self.generator_config = generator_config

    @property
    @abstractmethod
    def feature_set(self) -> FeatureSet:
        """The features this family emits, declared without touching data."""

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like ``frame``, holding only this family's columns.

        The input is the raw order table with every row in it - including rows
        from later splits. That is deliberate and safe: features are computed
        as-of each row's own order time, so a training row cannot see a test row's
        outcome regardless of what else is in the frame. Filtering to a split
        *before* computing features would be worse, not better: it would deprive a
        validation-window order of the customer history it genuinely had.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, enabled={self.config.enabled})"
