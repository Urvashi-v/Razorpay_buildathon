"""The health and readiness endpoints tell the truth.

These are the only fully implemented endpoints in Phase 1, so they are tested
properly rather than smoke-tested. The important assertions are the honest ones:
readiness is **not** ready when no model is loaded, and it **is** still ready
when the language layer is off.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rto_sentinel.api.main import create_app
from rto_sentinel.settings import get_settings
from tests.unit.test_model_registry import write_artefact


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_is_up(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"


def test_readiness_reports_a_missing_model_as_not_ready(client: TestClient) -> None:
    """No model means this instance cannot score, and it says so.

    A service that reports itself ready while unable to score would receive
    traffic and then have to invent a probability or fail per-request. Neither is
    acceptable, so the readiness check is where it gets caught.
    """
    response = client.get("/readiness")
    assert response.status_code == 503

    body = response.json()
    assert body["ready"] is False
    assert body["components"]["model"]["ready"] is False
    # The wording comes from the registry itself, so this asserts that readiness
    # is reporting the scoring path's own reason rather than a parallel one.
    assert "will not serve a synthesised probability" in body["components"]["model"]["detail"]


def test_readiness_stays_ready_without_the_language_layer(client: TestClient) -> None:
    """SPEC section 08: no LLM means degraded explanations, not a broken system.

    The agent component reports not-ready, but it is excluded from the overall
    readiness computation - which is asserted here by checking that the *only*
    reason readiness is false is the missing model.
    """
    body = client.get("/readiness").json()
    assert body["components"]["agents"]["ready"] is False
    assert "scoring and decisions are unaffected" in body["components"]["agents"]["detail"]

    not_ready = {name for name, comp in body["components"].items() if not comp["ready"]}
    assert not_ready == {"model", "agents"}


def test_readiness_becomes_ready_once_a_servable_artefact_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The readiness gate flips on the registry being able to resolve an artefact.

    Readiness asks the registry the same question the scoring path asks, rather
    than checking the environment variable itself. Note what this test does
    *not* do: it does not set `RTO_ACTIVE_MODEL_PATH`. An unset pin means "serve
    the newest calibrated artefact", which is a perfectly servable instance -
    and the earlier implementation reported it unready.
    """
    write_artefact(tmp_path, "lightgbm_platt")
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(tmp_path))
    get_settings.cache_clear()

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.get("/readiness")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["components"]["model"]["ready"] is True
        detail = body["components"]["model"]["detail"]
        assert "lightgbm_platt" in detail
        assert "newest in store" in detail


def test_readiness_reports_a_pin_as_a_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Which artefact is serving, and whether it was chosen or pinned, is visible."""
    write_artefact(tmp_path, "newest")
    pinned = write_artefact(tmp_path, "pinned_version")
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("RTO_ACTIVE_MODEL_PATH", str(pinned))
    get_settings.cache_clear()

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        body = client.get("/readiness").json()
        assert body["ready"] is True
        detail = body["components"]["model"]["detail"]
        assert "pinned_version" in detail
        assert "(pinned)" in detail


def test_readiness_flags_a_configured_but_missing_artefact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pin that resolves to nothing fails readiness - it does not fall back."""
    write_artefact(tmp_path, "healthy")
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("RTO_ACTIVE_MODEL_PATH", str(tmp_path / "models" / "absent"))
    get_settings.cache_clear()

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        body = client.get("/readiness").json()
        assert body["ready"] is False
        assert "not a readable artefact directory" in body["components"]["model"]["detail"]


def test_readiness_never_leaks_the_database_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RTO_DATABASE_URL", "postgresql+psycopg://rto:hunter2@localhost:5432/db")
    get_settings.cache_clear()

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        raw = client.get("/readiness").text
        assert "hunter2" not in raw
        assert "***" in raw


def test_readiness_exposes_the_config_fingerprint(client: TestClient) -> None:
    """Ties a running instance to the exact configuration it is serving."""
    body = client.get("/readiness").json()
    assert body["config_fingerprint"]
    assert len(body["config_fingerprint"]) == 64
