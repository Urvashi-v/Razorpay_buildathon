"""Model artefact persistence and provenance.

Every saved model is a directory containing three things:

``model.joblib``
    The fitted estimator's state.
``card.json``
    A :class:`~rto_sentinel.contracts.risk.ModelCard`: which rung, which seed,
    which configuration fingerprint, which feature fingerprint, how many rows it
    saw, which families were enabled.
``checksum.txt``
    SHA-256 of ``model.joblib``.

WHY THE CHECKSUM
================
Loading a joblib file executes pickle, which executes code. For a local artefact
store written by this project's own training runs that is an acceptable risk, but
"acceptable" is not "ignorable". The checksum is recorded at save time and
verified at load, so a corrupted or substituted file fails loudly instead of
silently producing predictions from something nobody intended.

It is an integrity check, not an authenticity one: anyone who can replace the
model can replace the checksum. It catches corruption and accident, which is what
actually happens. A real deployment would sign artefacts.

WHY THE CARD IS SEPARATE FROM THE MODEL
=======================================
So provenance can be read without unpickling anything. Listing what is in the
artefact store, checking whether a model matches the current configuration, or
building a comparison table should not require executing code from a file on
disk. ``read_card`` opens JSON and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
from pydantic import ValidationError

from rto_sentinel.contracts.risk import ModelCard

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

MODEL_FILE = "model.joblib"
CARD_FILE = "card.json"
CHECKSUM_FILE = "checksum.txt"


class ArtifactError(RuntimeError):
    """Raised when an artefact is missing, corrupt, or fails its integrity check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_dir(root: Path, model_name: str, model_version: str) -> Path:
    """Where one model artefact lives. Keyed so versions never collide."""
    return root / "models" / f"{model_name}__{model_version}"


def save_artifact(directory: Path, state: Any, card: ModelCard) -> Path:
    """Write the fitted state, its card and a checksum. Returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / MODEL_FILE
    joblib.dump(state, model_path)

    checksum = _sha256(model_path)
    (directory / CHECKSUM_FILE).write_text(checksum, encoding="utf-8")
    (directory / CARD_FILE).write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return directory


def read_card(directory: Path) -> ModelCard:
    """Read and validate a model card without touching the pickled state."""
    path = directory / CARD_FILE
    if not path.is_file():
        msg = f"no model card at {path}"
        raise ArtifactError(msg)
    return ModelCard.model_validate_json(path.read_text(encoding="utf-8"))


def load_artifact(directory: Path) -> tuple[Any, ModelCard]:
    """Load a persisted model state and its card, verifying integrity first."""
    if not directory.is_dir():
        msg = f"no artefact directory at {directory}"
        raise ArtifactError(msg)

    model_path = directory / MODEL_FILE
    checksum_path = directory / CHECKSUM_FILE
    for required in (model_path, checksum_path):
        if not required.is_file():
            msg = f"artefact at {directory} is incomplete; missing {required.name}"
            raise ArtifactError(msg)

    expected = checksum_path.read_text(encoding="utf-8").strip()
    actual = _sha256(model_path)
    if actual != expected:
        msg = (
            f"checksum mismatch for {model_path}.\n"
            f"  recorded: {expected}\n"
            f"  actual  : {actual}\n"
            "The artefact has been modified or corrupted since it was written. Refusing to "
            "load it: predictions from an unknown model are worse than no predictions."
        )
        raise ArtifactError(msg)

    return joblib.load(model_path), read_card(directory)


def verify_provenance(card: ModelCard, *, config_fingerprint: str) -> list[str]:
    """Warnings when an artefact was trained under a different configuration.

    Returns rather than raises: a fingerprint mismatch is not always fatal - the
    console may legitimately display an older model's results - but it must never
    pass unnoticed. The API surfaces these on ``/readiness``.
    """
    warnings: list[str] = []
    if card.config_fingerprint != config_fingerprint:
        warnings.append(
            f"model {card.model_name} v{card.model_version} was trained under configuration "
            f"{card.config_fingerprint[:12]}... but the current configuration is "
            f"{config_fingerprint[:12]}.... Its metrics describe a different setup."
        )
    if card.calibration_method is None:
        warnings.append(
            f"model {card.model_name} v{card.model_version} is UNCALIBRATED. Its scores are "
            "not honest probabilities and must not reach the decision engine."
        )
    return warnings


def list_artifacts(root: Path) -> list[tuple[Path, ModelCard]]:
    """Every artefact in the store, newest first, read from JSON only."""
    directory = root / "models"
    if not directory.is_dir():
        return []

    found: list[tuple[Path, ModelCard]] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or not (child / CARD_FILE).is_file():
            continue
        try:
            found.append((child, read_card(child)))
        except (OSError, ValueError, ValidationError) as error:
            # An unreadable or malformed card must not hide the rest of the
            # store, but it must not vanish silently either.
            LOGGER.warning("skipping artefact %s: %s", child.name, error)
    return sorted(found, key=lambda pair: pair[1].trained_at, reverse=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a machine-readable artefact, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
