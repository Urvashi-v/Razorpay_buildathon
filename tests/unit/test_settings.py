"""Settings load correctly and never leak a secret.

The redaction test is the one that earns its place. ``safe_url`` is what
``/readiness`` returns, and a readiness endpoint that echoes a database password
is the kind of mistake that survives review because nobody reads the health
response carefully.
"""

from __future__ import annotations

import pytest

from rto_sentinel.settings import LLMSettings, Settings, get_settings


def test_defaults_are_sane(settings: Settings) -> None:
    assert settings.env == "test"
    assert settings.api_port == 8000
    assert settings.random_seed > 0
    assert settings.config_path.is_dir()


def test_database_url_redacts_the_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RTO_DATABASE_URL", "postgresql+psycopg://rto:hunter2@db.internal:5432/rto_sentinel"
    )
    get_settings.cache_clear()
    settings = get_settings()

    assert "hunter2" in settings.database.url, "the real URL must still work"
    assert "hunter2" not in settings.database.safe_url
    assert (
        settings.database.safe_url == "postgresql+psycopg://rto:***@db.internal:5432/rto_sentinel"
    )


def test_sqlite_url_survives_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL with no credentials passes through rather than being mangled."""
    monkeypatch.setenv("RTO_DATABASE_URL", "sqlite+pysqlite:///./local.db")
    get_settings.cache_clear()
    assert get_settings().database.safe_url == "sqlite+pysqlite:///./local.db"


def test_cors_origins_are_parsed_as_a_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTO_CORS_ORIGINS", "http://localhost:5173, https://console.example.com")
    get_settings.cache_clear()
    assert get_settings().cors_origins == [
        "http://localhost:5173",
        "https://console.example.com",
    ]


def test_blank_model_path_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RTO_ACTIVE_MODEL_PATH=`` in a .env file means "no model", not path ""."""
    monkeypatch.setenv("RTO_ACTIVE_MODEL_PATH", "   ")
    get_settings.cache_clear()
    assert get_settings().active_model_path is None


# ---------------------------------------------------------------------------
# The LLM off-switch
# ---------------------------------------------------------------------------


def test_agents_are_off_without_a_key() -> None:
    llm = LLMSettings(agents_enabled=True, api_key=None)
    assert llm.enabled is False
    assert llm.unavailable_reason == "ANTHROPIC_API_KEY is not set"


def test_agents_are_off_when_the_switch_is_off() -> None:
    """A key present is not sufficient. The operator has to opt in."""
    from pydantic import SecretStr

    llm = LLMSettings(agents_enabled=False, api_key=SecretStr("placeholder-not-a-real-key"))
    assert llm.enabled is False
    assert llm.unavailable_reason == "RTO_AGENTS_ENABLED is false"


def test_agents_are_on_only_with_both(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import SecretStr

    llm = LLMSettings(agents_enabled=True, api_key=SecretStr("placeholder-not-a-real-key"))
    assert llm.enabled is True
    assert llm.unavailable_reason is None


def test_an_empty_key_string_counts_as_absent() -> None:
    """``ANTHROPIC_API_KEY=`` in a .env must not read as "configured"."""
    from pydantic import SecretStr

    llm = LLMSettings(agents_enabled=True, api_key=SecretStr("   "))
    assert llm.enabled is False


def test_api_key_is_not_exposed_by_repr() -> None:
    """A settings object in a traceback must not print the key."""
    from pydantic import SecretStr

    llm = LLMSettings(agents_enabled=True, api_key=SecretStr("placeholder-not-a-real-key"))
    assert "placeholder-not-a-real-key" not in repr(llm)
    assert "placeholder-not-a-real-key" not in str(llm)
