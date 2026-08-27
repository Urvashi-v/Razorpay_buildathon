"""Typed application settings, loaded once from the environment.

This is the *only* module in the codebase permitted to read ``os.environ``.
Every other module receives configuration through :func:`get_settings` or through
an explicit function argument. ``tests/architecture/test_layering.py`` enforces
that mechanically.

Secrets never appear here as literals. ``ANTHROPIC_API_KEY`` has no default: if
it is absent the agent layer reports itself unavailable rather than pretending to
work (see ``agents/provider.py``).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/rto_sentinel/settings.py -> src/rto_sentinel -> src -> root
REPO_ROOT = Path(__file__).resolve().parents[2]

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class DatabaseSettings(BaseSettings):
    """Connection details for the orders / decisions / outcomes store."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    user: str = Field(default="rto", alias="POSTGRES_USER")
    password: SecretStr = Field(default=SecretStr(""), alias="POSTGRES_PASSWORD")
    name: str = Field(default="rto_sentinel", alias="POSTGRES_DB")
    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")

    # A full URL, when provided, wins over the assembled parts above. Tests set
    # this to an ephemeral SQLite file so the suite never needs a live server.
    url_override: str | None = Field(default=None, alias="RTO_DATABASE_URL")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """SQLAlchemy URL. Never logged - it carries the password."""
        if self.url_override:
            return self.url_override
        pwd = self.password.get_secret_value()
        return f"postgresql+psycopg://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def safe_url(self) -> str:
        """URL with the password removed, safe to log or return from /health."""
        raw = self.url
        if "://" not in raw or "@" not in raw:
            return raw
        scheme, rest = raw.split("://", 1)
        _credentials, location = rest.rsplit("@", 1)
        user = _credentials.split(":", 1)[0]
        return f"{scheme}://{user}:***@{location}"


class LLMSettings(BaseSettings):
    """Configuration for the OPTIONAL downstream language layer.

    EXTERNAL SERVICE: Anthropic Claude API (https://api.anthropic.com).

    The risk engine does not depend on this. When ``enabled`` resolves False the
    agent endpoints return an explicit "unavailable" response; they do not return
    canned prose dressed up as model output.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-sonnet-5", alias="RTO_LLM_MODEL")
    max_tokens: int = Field(default=1024, ge=1, le=8192, alias="RTO_LLM_MAX_TOKENS")
    timeout_seconds: float = Field(default=20.0, gt=0, alias="RTO_LLM_TIMEOUT_SECONDS")
    # Hard off-switch. Even with a key present, agents stay off unless this is on.
    agents_enabled: bool = Field(default=False, alias="RTO_AGENTS_ENABLED")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def enabled(self) -> bool:
        """True only when the operator switched agents on AND a key is present."""
        has_key = self.api_key is not None and bool(self.api_key.get_secret_value().strip())
        return self.agents_enabled and has_key

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unavailable_reason(self) -> str | None:
        """Human-readable reason the agent layer is off, or None when it is on."""
        if not self.agents_enabled:
            return "RTO_AGENTS_ENABLED is false"
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            return "ANTHROPIC_API_KEY is not set"
        return None


class Settings(BaseSettings):
    """Root settings object. Construct once via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    env: Environment = Field(default="development", alias="RTO_ENV")
    log_level: LogLevel = Field(default="INFO", alias="RTO_LOG_LEVEL")

    api_host: str = Field(default="127.0.0.1", alias="RTO_API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="RTO_API_PORT")
    cors_origins_raw: str = Field(default="http://localhost:5173", alias="RTO_CORS_ORIGINS")

    config_dir: Path = Field(default=Path("config"), alias="RTO_CONFIG_DIR")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="RTO_ARTIFACT_DIR")

    # Global seed for generator, splits and training. Recorded in run metadata:
    # changing it changes every reported number.
    random_seed: int = Field(default=20260827, alias="RTO_RANDOM_SEED")

    # Empty means no model is loaded, and the scoring endpoint says so rather
    # than inventing a probability.
    active_model_path: Path | None = Field(default=None, alias="RTO_ACTIVE_MODEL_PATH")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @field_validator("active_model_path", mode="before")
    @classmethod
    def _blank_path_is_none(cls, value: object) -> object:
        """Treat ``RTO_ACTIVE_MODEL_PATH=`` in a .env file as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    def resolve(self, path: Path) -> Path:
        """Resolve a possibly-relative configured path against the repo root."""
        return path if path.is_absolute() else (REPO_ROOT / path)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def config_path(self) -> Path:
        return self.resolve(self.config_dir)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_path(self) -> Path:
        return self.resolve(self.artifact_dir)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that the environment is read once. Tests that need different values
    call ``get_settings.cache_clear()`` after patching the environment; see
    ``tests/conftest.py``.
    """
    return Settings()
