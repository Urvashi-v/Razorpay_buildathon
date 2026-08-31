"""The registry's resolution rules.

`resolve` is the single answer to "which artefact does this instance serve".
Readiness and scoring both ask it, which is the point: when they each derived
the answer independently, readiness reported the instance unready while scoring
was working perfectly off the newest artefact in the store.

These tests use card JSON only. `resolve` deliberately does not deserialise a
booster - a readiness probe that loads a model is a readiness probe nobody can
poll - so a directory with a valid card is enough to exercise every branch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rto_sentinel.contracts.risk import ModelCard
from rto_sentinel.models.artifacts import CARD_FILE
from rto_sentinel.serving.model_registry import ModelRegistry, ModelUnavailableError


def write_artefact(
    root: Path,
    name: str,
    *,
    calibration: str | None = "platt",
    trained_at: datetime | None = None,
) -> Path:
    """A minimal but valid artefact directory: a card, nothing else."""
    directory = root / "models" / name
    directory.mkdir(parents=True, exist_ok=True)
    card = ModelCard(
        model_name=name,
        model_version=f"v-{name}",
        rung_id=5,
        trained_at=trained_at or datetime.now(UTC),
        training_rows=1000,
        feature_names=("a", "b"),
        enabled_families=("customer",),
        random_seed=7,
        config_fingerprint="deadbeef",
        calibration_method=calibration,
    )
    (directory / CARD_FILE).write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return directory


class TestUnpinned:
    def test_serves_the_newest_calibrated_artefact(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        write_artefact(tmp_path, "older", trained_at=now - timedelta(days=3))
        newest = write_artefact(tmp_path, "newer", trained_at=now)

        path, card = ModelRegistry(tmp_path).resolve()

        assert path == newest
        assert card.model_name == "newer"

    def test_skips_uncalibrated_artefacts(self, tmp_path: Path) -> None:
        """A Phase 4 ladder rung is not servable, however recently it was trained."""
        calibrated = write_artefact(
            tmp_path, "final", trained_at=datetime.now(UTC) - timedelta(days=10)
        )
        write_artefact(tmp_path, "rung", calibration=None, trained_at=datetime.now(UTC))

        path, _ = ModelRegistry(tmp_path).resolve()

        assert path == calibrated

    def test_refuses_an_empty_store_rather_than_inventing_one(self, tmp_path: Path) -> None:
        with pytest.raises(ModelUnavailableError, match="no calibrated model artefact"):
            ModelRegistry(tmp_path).resolve()

    def test_says_when_the_only_artefacts_are_uncalibrated(self, tmp_path: Path) -> None:
        write_artefact(tmp_path, "rung", calibration=None)

        with pytest.raises(ModelUnavailableError, match="uncalibrated artefact"):
            ModelRegistry(tmp_path).resolve()


class TestPinned:
    def test_a_pin_wins_over_the_newest_artefact(self, tmp_path: Path) -> None:
        """The whole purpose of a pin: serve this version, not whatever is newest."""
        old = write_artefact(tmp_path, "old", trained_at=datetime.now(UTC) - timedelta(days=30))
        write_artefact(tmp_path, "new", trained_at=datetime.now(UTC))

        path, card = ModelRegistry(tmp_path, pinned=old).resolve()

        assert path == old
        assert card.model_name == "old"

    def test_a_broken_pin_does_not_fall_back(self, tmp_path: Path) -> None:
        """Falling back would silently serve a different model than was asked for.

        That is precisely the accident a pin exists to prevent, so an unresolvable
        pin is an error even when a perfectly good artefact sits next to it.
        """
        write_artefact(tmp_path, "healthy")

        registry = ModelRegistry(tmp_path, pinned=tmp_path / "models" / "does-not-exist")

        with pytest.raises(ModelUnavailableError, match="not a readable artefact directory"):
            registry.resolve()

    def test_a_pin_at_a_plain_file_is_refused(self, tmp_path: Path) -> None:
        """An artefact is a directory. A file of arbitrary bytes is not a model."""
        stray = tmp_path / "model.pkl"
        stray.write_bytes(b"not a real model, but a real file")

        with pytest.raises(ModelUnavailableError, match="not a readable artefact directory"):
            ModelRegistry(tmp_path, pinned=stray).resolve()

    def test_an_uncalibrated_pin_is_refused(self, tmp_path: Path) -> None:
        pinned = write_artefact(tmp_path, "rung", calibration=None)

        with pytest.raises(ModelUnavailableError, match="uncalibrated"):
            ModelRegistry(tmp_path, pinned=pinned).resolve()

    def test_a_malformed_card_is_refused(self, tmp_path: Path) -> None:
        pinned = write_artefact(tmp_path, "corrupt")
        (pinned / CARD_FILE).write_text("{ this is not json", encoding="utf-8")

        with pytest.raises(ModelUnavailableError, match="unreadable card"):
            ModelRegistry(tmp_path, pinned=pinned).resolve()


def test_resolve_does_not_load_the_booster(tmp_path: Path) -> None:
    """Resolution stays cheap enough for a readiness probe to poll.

    The artefacts written above contain no pickled state at all. If `resolve`
    deserialised anything, every test in this module would fail.
    """
    write_artefact(tmp_path, "only")
    registry = ModelRegistry(tmp_path)

    registry.resolve()

    assert registry.is_loaded is False
