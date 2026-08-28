"""Settings load correctly and never leak a secret.

The redaction test is the one that earns its place. ``safe_url`` is what
``/readiness`` returns, and a readiness endpoint that echoes a database password
is the kind of mistake that survives review because nobody reads the health
response carefully.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import SecretStr

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

#: A syntactically plausible but entirely fictional key. It is never sent
#: anywhere: these tests only exercise the local gate, which decides whether a
#: call would be attempted at all.
FAKE_KEY = "placeholder-not-a-real-key"


@pytest.fixture
def build_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., LLMSettings]:
    """A factory building :class:`LLMSettings` from its arguments and nothing else.

    WHY THIS HELPER EXISTS
    ----------------------
    ``LLMSettings`` declares its fields with environment aliases
    (``agents_enabled`` <- ``RTO_AGENTS_ENABLED``). pydantic-settings merges the
    init keyword arguments and the environment into one mapping before
    validating, and the two arrive under *different keys* - the field name from
    the constructor, the alias from the environment. Pydantic then prefers the
    alias, so an environment value silently outranks an explicit constructor
    argument, which inverts the documented source priority.

    The suite's ``_isolated_env`` fixture sets ``RTO_AGENTS_ENABLED=false``, and
    a developer ``.env`` supplies ``ANTHROPIC_API_KEY=``. Without this helper
    both of those quietly replace whatever the test passed, and every assertion
    below would pass or fail for reasons unrelated to the gate logic.

    So: remove the aliases from the environment and switch the dotenv file off,
    leaving the constructor arguments as the only source. ``monkeypatch`` puts
    the environment back afterwards.
    """
    for name in ("RTO_AGENTS_ENABLED", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def build(**overrides: object) -> LLMSettings:
        return LLMSettings(_env_file=None, **overrides)  # type: ignore[arg-type]

    return build


def test_agents_are_off_without_a_key(build_llm: Callable[..., LLMSettings]) -> None:
    llm = build_llm(agents_enabled=True, api_key=None)
    assert llm.enabled is False
    assert llm.unavailable_reason == "ANTHROPIC_API_KEY is not set"


def test_agents_are_off_when_the_switch_is_off(build_llm: Callable[..., LLMSettings]) -> None:
    """A key present is not sufficient. The operator has to opt in."""
    llm = build_llm(agents_enabled=False, api_key=SecretStr(FAKE_KEY))
    assert llm.enabled is False
    assert llm.unavailable_reason == "RTO_AGENTS_ENABLED is false"


def test_agents_are_on_only_with_both(build_llm: Callable[..., LLMSettings]) -> None:
    llm = build_llm(agents_enabled=True, api_key=SecretStr(FAKE_KEY))
    assert llm.enabled is True
    assert llm.unavailable_reason is None


def test_an_empty_key_string_counts_as_absent(build_llm: Callable[..., LLMSettings]) -> None:
    """``ANTHROPIC_API_KEY=`` in a .env must not read as "configured"."""
    llm = build_llm(agents_enabled=True, api_key=SecretStr("   "))
    assert llm.enabled is False


def test_the_environment_can_turn_agents_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real path an operator uses: environment variables, by alias."""
    monkeypatch.setenv("RTO_AGENTS_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    get_settings.cache_clear()

    assert get_settings().llm.enabled is True


def test_the_suite_itself_runs_with_agents_off(settings: Settings) -> None:
    """No test in this project may reach the Anthropic API.

    ``_isolated_env`` pins the switch off; this asserts it, so a future change to
    that fixture cannot quietly enable outbound calls during a test run.
    """
    assert settings.llm.enabled is False


def test_api_key_is_not_exposed_by_repr(build_llm: Callable[..., LLMSettings]) -> None:
    """A settings object in a traceback must not print the key."""
    llm = build_llm(agents_enabled=True, api_key=SecretStr(FAKE_KEY))
    assert FAKE_KEY not in repr(llm)
    assert FAKE_KEY not in str(llm)
