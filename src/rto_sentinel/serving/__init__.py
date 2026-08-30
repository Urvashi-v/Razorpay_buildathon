"""The serving layer: database plus artefacts plus engine, composed.

WHY THIS PACKAGE EXISTS AT ALL
==============================
Every other layer is deliberately unable to reach the ones beside it. ``features``
and ``models`` cannot import ``db`` - a model must be retrainable offline from
config and a seed, with no server anywhere. ``decision`` cannot import either -
it stays pure so the same inputs always produce the same action. ``api.routers``
cannot import an ML library - a route handler that reaches for LightGBM has
become the place risk logic lives.

Those constraints are enforced mechanically in
``tests/architecture/test_layering.py``, and they leave a gap: something has to
put the pieces together for a live request. That something is this package.

WHAT IT COMPOSES, IN ORDER
==========================
::

    database row
      -> OrderFeatureService   (reconstructs the frame, runs the feature pipeline)
      -> ModelRegistry         (the artefact that was actually trained and frozen)
      -> calibrator            (inside the artefact; the card names the method)
      -> calibrated probability
      -> DecisionEngine        (cost-derived threshold, friction ladder)
      -> structured response

Every arrow in that chain executes on every request. Nothing is cached except the
loaded artefact itself, and nothing is synthesised: when the model is missing the
services raise, and the API returns 503 rather than a plausible number.

WHAT THIS PACKAGE MAY IMPORT
============================
``db``, ``features``, ``models``, ``decision``, ``eval``, ``contracts``,
``configuration``. It may **not** import ``api`` (composition does not depend on
its caller) or ``agents`` (a language model has no path into a score, a threshold
or an action). Asserted in the layering tests.
"""

from rto_sentinel.serving.assessment import (
    AssessmentService,
    OrderAssessment,
    OrderNotFoundError,
)
from rto_sentinel.serving.features import (
    FeatureServiceError,
    OrderFeatures,
    OrderFeatureService,
)
from rto_sentinel.serving.model_registry import (
    LoadedModel,
    ModelRegistry,
    ModelUnavailableError,
)
from rto_sentinel.serving.scoring import ScoringService

__all__ = [
    "AssessmentService",
    "FeatureServiceError",
    "LoadedModel",
    "ModelRegistry",
    "ModelUnavailableError",
    "OrderAssessment",
    "OrderFeatureService",
    "OrderFeatures",
    "OrderNotFoundError",
    "ScoringService",
]
