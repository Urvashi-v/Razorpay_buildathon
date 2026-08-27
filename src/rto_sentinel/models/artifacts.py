"""Model artefact persistence and provenance.

Every saved model is accompanied by a :class:`~rto_sentinel.contracts.risk.ModelCard`
recording the configuration fingerprint, the seed, the feature names and which
families were enabled. That is what makes a number in REPORT.md traceable back to
the exact state that produced it, months later, without trusting anyone's memory.

Artefacts are written under ``artifacts/models/`` and are git-ignored: they are
reproducible from config plus seed, so they are output, not source.

STATUS: Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.risk import ModelCard


class ArtifactError(RuntimeError):
    """Raised when an artefact is missing, corrupt, or fails its provenance check."""


def write_card(path: Path, card: ModelCard) -> None:
    """Write the JSON model card that sits beside a serialised model."""
    raise NotImplementedError("Artefact persistence lands in Phase 3.")


def read_card(path: Path) -> ModelCard:
    """Read and validate a model card."""
    raise NotImplementedError("Artefact persistence lands in Phase 3.")


def verify_provenance(card: ModelCard, current_fingerprint: str) -> list[str]:
    """Return warnings when an artefact was trained under a different configuration.

    Returns rather than raises: a fingerprint mismatch is not always fatal (the
    console may legitimately display an older model's results), but it must never
    pass unnoticed. The API surfaces these warnings on ``/health``.
    """
    raise NotImplementedError("Provenance checking lands in Phase 3.")
