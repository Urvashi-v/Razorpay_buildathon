"""Controlled model loading. One artefact, verified, cached, and versioned.

WHAT "CONTROLLED" MEANS HERE
===========================
Four properties, each of which has a failure mode this module exists to prevent:

**It loads a real artefact or it raises.** There is no default model, no
zero-probability fallback, no "score unavailable so assume low risk". A service
that cannot score an order says so with a 503; it does not guess and let the
guess flow into a rupee figure. :class:`ModelUnavailableError` carries the path
it looked in and the command that would produce one.

**It verifies before it trusts.** ``load_artifact`` checks a SHA-256 checksum
before unpickling, so a truncated or edited file fails loudly rather than
producing subtly wrong scores. The registry then checks that the card's feature
fingerprint matches the pipeline the server is running: a model loaded against a
different feature set is a silent wrong answer, and it is the single most likely
production failure in a system where features and models version separately.

**It refuses an uncalibrated artefact.** The decision layer's entire expected-value
argument depends on the probability being honest. An artefact whose card says
``calibration_method: null`` cannot be served, and that is checked at load time
rather than at the threshold comparison, so the failure surfaces at startup
instead of on a customer's order.

**It is loaded once.** Deserialising a booster per request would be slow and,
worse, would make the served model a function of whatever is on disk at that
instant. The registry caches the artefact and reports the version it holds, so
every response says which model produced it and a redeploy is visible rather than
inferred.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rto_sentinel.models.artifacts import ArtifactError, list_artifacts
from rto_sentinel.models.calibrated import CalibratedModel
from rto_sentinel.models.final import load_manifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from rto_sentinel.contracts.final import SelectionManifest
    from rto_sentinel.contracts.risk import ModelCard


class ModelUnavailableError(RuntimeError):
    """Raised when no servable model artefact exists.

    Deliberately not a warning and deliberately not recoverable inside the
    request path. The API maps it to 503 with the message intact, because "we
    have no model" is operational information the caller needs, and every
    alternative involves inventing a probability.
    """


class ModelMismatchError(RuntimeError):
    """Raised when the loaded artefact does not match the running feature pipeline.

    A model scored against a different feature set does not fail - it returns a
    number, and the number is wrong. This is the check that turns that into an
    error.
    """


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """The artefact in memory, with everything a response needs to cite it."""

    model: CalibratedModel
    card: ModelCard
    manifest: SelectionManifest
    artifact_path: Path

    @property
    def model_version(self) -> str:
        return self.card.model_version

    @property
    def feature_version(self) -> str:
        return self.card.feature_version

    @property
    def calibration_method(self) -> str | None:
        return self.card.calibration_method

    def describe(self) -> dict[str, object]:
        """The provenance block every scoring response carries."""
        return {
            "model_name": self.card.model_name,
            "model_version": self.card.model_version,
            "rung_id": self.card.rung_id,
            "calibration_method": self.card.calibration_method,
            "calibration_fitted_on": self.card.calibration_fitted_on,
            "feature_version": self.card.feature_version,
            "feature_fingerprint": self.card.feature_fingerprint,
            "dataset_run_id": self.card.dataset_run_id,
            "generator_version": self.card.generator_version,
            "config_fingerprint": self.card.config_fingerprint,
            "trained_at": self.card.trained_at,
            "training_rows": self.card.training_rows,
            "n_features": len(self.card.feature_names),
            "selection_manifest_id": self.manifest.manifest_id,
            "artifact_path": str(self.artifact_path),
        }


class ModelRegistry:
    """Loads and holds the servable artefact. Thread-safe, cached, verifiable."""

    def __init__(self, artifact_root: Path, *, require_calibration: bool = True) -> None:
        self._artifact_root = artifact_root
        self._require_calibration = require_calibration
        self._loaded: LoadedModel | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def load(self) -> LoadedModel:
        """Return the servable model, loading it on first use.

        Double-checked under a lock: two simultaneous first requests must not
        deserialise the booster twice, and must not observe a half-built
        registry.
        """
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load_uncached()
        return self._loaded

    def _load_uncached(self) -> LoadedModel:
        candidate = self._newest_calibrated_artifact()
        try:
            model, card = CalibratedModel.load(candidate)
        except ArtifactError as error:
            msg = (
                f"the model artefact at {candidate} failed its integrity check: {error}. "
                "Refusing to serve scores from a file that may have been truncated or "
                "edited."
            )
            raise ModelUnavailableError(msg) from error

        if self._require_calibration and not card.calibration_method:
            msg = (
                f"the artefact at {candidate} is uncalibrated (calibration_method is null). "
                "An uncalibrated probability compared against an expected-value threshold "
                "produces rupee figures that are fiction, so it is refused at load time."
            )
            raise ModelUnavailableError(msg)

        try:
            manifest = load_manifest(self._artifact_root, card.dataset_run_id)
        except FileNotFoundError as error:
            msg = (
                f"the artefact at {candidate} has no frozen selection manifest for dataset "
                f"{card.dataset_run_id}. A servable model must be traceable to the decisions "
                "that produced it. Re-run `rto-sentinel final`."
            )
            raise ModelUnavailableError(msg) from error

        return LoadedModel(model=model, card=card, manifest=manifest, artifact_path=candidate)

    def _newest_calibrated_artifact(self) -> Path:
        """The most recently trained calibrated artefact under the root."""
        try:
            found = list_artifacts(self._artifact_root)
        except OSError as error:  # pragma: no cover - unreadable artefact store
            msg = f"could not read the artefact store at {self._artifact_root}: {error}"
            raise ModelUnavailableError(msg) from error

        servable = [
            (path, card)
            for path, card in found
            if card.calibration_method or not self._require_calibration
        ]
        if not servable:
            uncalibrated = len(found) - len(servable)
            detail = (
                f" {uncalibrated} uncalibrated artefact(s) were found and skipped;"
                " Phase 4 ladder rungs are not servable."
                if uncalibrated
                else ""
            )
            msg = (
                f"no calibrated model artefact under {self._artifact_root / 'models'}.{detail} "
                "Run `rto-sentinel final` to train, calibrate and freeze one. The API will "
                "not serve a synthesised probability in its place."
            )
            raise ModelUnavailableError(msg)

        # `list_artifacts` returns newest first by `trained_at`.
        return servable[0][0]

    # ------------------------------------------------------------------
    # verification and introspection
    # ------------------------------------------------------------------

    def verify_features(self, feature_fingerprint: str, feature_names: tuple[str, ...]) -> None:
        """Refuse to score when the running pipeline is not the trained one.

        Two checks rather than one. The fingerprint catches a changed *set* of
        features; the name comparison produces a message a human can act on,
        naming what appeared and what went missing.
        """
        loaded = self.load()
        if loaded.card.feature_fingerprint == feature_fingerprint:
            return

        trained = set(loaded.card.feature_names)
        running = set(feature_names)
        added = sorted(running - trained)
        removed = sorted(trained - running)
        msg = (
            f"the loaded model {loaded.card.model_version} was trained on feature set "
            f"{loaded.card.feature_fingerprint[:16]}... but this server is running "
            f"{feature_fingerprint[:16]}.... Scoring would silently produce wrong numbers. "
            f"Features added since training: {added or 'none'}. Features missing now: "
            f"{removed or 'none'}."
        )
        raise ModelMismatchError(msg)

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def status(self) -> dict[str, object]:
        """What the readiness and monitoring endpoints report.

        Never raises. "No model is loaded" is a state the operator needs
        described, not an exception thrown at a health check.
        """
        try:
            loaded = self.load()
        except ModelUnavailableError as error:
            return {"available": False, "reason": str(error)}
        return {"available": True, **loaded.describe()}

    def invalidate(self) -> None:
        """Drop the cached artefact so the next request reloads from disk."""
        with self._lock:
            self._loaded = None
