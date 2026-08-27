"""Shared pytest fixtures.

Two things this file is careful about:

* **No live PostgreSQL.** The suite points ``RTO_DATABASE_URL`` at a temporary
  SQLite file, so ``pytest`` runs on a clean checkout with nothing installed but
  the Python dependencies. Tests that genuinely need Postgres are marked
  ``requires_db`` and skipped by default.
* **No API key.** The language layer is explicitly disabled for every test.
  Nothing in this suite makes a network call, and a test that silently started
  billing an Anthropic account would be a bad surprise.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from rto_sentinel.settings import REPO_ROOT, Settings, get_settings

# Environment variables that must not leak in from the developer's own shell or
# a stray .env file and change what the suite is testing.
_ISOLATED_VARS = (
    "RTO_ENV",
    "RTO_DATABASE_URL",
    "RTO_ACTIVE_MODEL_PATH",
    "RTO_AGENTS_ENABLED",
    "ANTHROPIC_API_KEY",
    "RTO_CONFIG_DIR",
    "RTO_ARTIFACT_DIR",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give every test a clean, offline, database-free environment."""
    for name in _ISOLATED_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("RTO_ENV", "test")
    monkeypatch.setenv("RTO_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("RTO_AGENTS_ENABLED", "false")
    monkeypatch.setenv("RTO_CONFIG_DIR", str(REPO_ROOT / "config"))

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Settings built from the isolated test environment."""
    return get_settings()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def source_root() -> Path:
    return REPO_ROOT / "src" / "rto_sentinel"
