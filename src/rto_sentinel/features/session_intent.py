"""Session and intent features.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.features.base import FeatureFamily

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class SessionIntentFamily(FeatureFamily):
    """Session and intent features.

    SPEC section 04. Weak individually, and included on that understanding:
    low-intent late-night impulse purchases are a documented RTO driver, and the
    ablation study reports honestly how little this family adds."""

    name = "session_intent"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.config.signals)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Feature family SessionIntentFamily lands in Phase 2.")
