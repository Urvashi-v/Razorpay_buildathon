"""Features to a calibrated probability, with the provenance to justify it.

The whole of this module is four lines of real work: build the features, check
they match what the model was trained on, run the artefact, wrap the result. The
value is in what it refuses to do.

**It does not fall back.** No default probability, no last-known score, no
"model unavailable so assume low risk". Every failure raises and the API turns it
into a 503 or a 409 with the reason attached.

**It does not score against a mismatched pipeline.** The fingerprint check runs
before inference, not after, because a model scored on the wrong feature set does
not fail - it returns a number.

**It does not hand back a bare probability.** A :class:`RiskScore` carries the
model name, version and calibration method, so a downstream consumer physically
cannot receive a score without knowing what produced it and whether it is
calibrated. The decision engine then refuses any score whose calibration method
is null, which is what stops an uncalibrated number reaching a rupee figure.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rto_sentinel.contracts.risk import FeatureContribution, RiskScore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.db.models import Order
    from rto_sentinel.serving.features import OrderFeatures, OrderFeatureService
    from rto_sentinel.serving.model_registry import LoadedModel, ModelRegistry


class ScoringService:
    """Runs the trained artefact over a stored order's features."""

    def __init__(self, registry: ModelRegistry, features: OrderFeatureService) -> None:
        self._registry = registry
        self._features = features

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def score(
        self, order: Order, *, include_contributions: bool = False
    ) -> tuple[RiskScore, OrderFeatures, LoadedModel]:
        """Score one order. Returns the score, its features and the model used.

        All three are returned rather than just the score because the caller -
        the assessment service, the API - needs to cite the model and may want to
        report which features were null. Recomputing either would mean building
        the feature row twice per request.
        """
        started = time.perf_counter()
        loaded = self._registry.load()
        built = self._features.build(order)

        # Before inference, not after: a mismatched pipeline produces a number,
        # not an error, and the number is wrong.
        self._registry.verify_features(built.feature_fingerprint, built.feature_names)

        probability = float(loaded.model.predict_proba(built.x, built.context)[0])
        raw = float(loaded.model.predict_raw(built.x, built.context)[0])
        latency_ms = (time.perf_counter() - started) * 1000.0

        contributions: list[FeatureContribution] = []
        if include_contributions:
            contributions = self._contributions(loaded, built)

        score = RiskScore(
            order_id=order.order_id,
            probability=probability,
            raw_score=raw,
            model_name=loaded.card.model_name,
            model_version=loaded.card.model_version,
            calibration_method=loaded.card.calibration_method,
            scored_at=datetime.now(UTC),
            latency_ms=latency_ms,
            contributions=contributions,
        )
        return score, built, loaded

    def _contributions(
        self, loaded: LoadedModel, built: OrderFeatures
    ) -> list[FeatureContribution]:
        """Per-feature attributions, or an empty list when none are available.

        Empty is a legitimate answer, not a failure. Heuristic rungs have no
        attributions at all, and SHAP can be unavailable. The decision layer
        already handles that case by emitting a score-only reason code, so
        raising here would break scoring for a presentational nicety.
        """
        try:
            per_row = loaded.model.explain(built.x, top_k=5)
        except Exception:
            return []
        return list(per_row[0]) if per_row else []
