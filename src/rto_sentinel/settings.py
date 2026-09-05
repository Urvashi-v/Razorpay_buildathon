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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/rto_sentinel/settings.py -> src/rto_sentinel -> src -> root
REPO_ROOT = Path(__file__).resolve().parents[2]

Environment = Literal["development", "test", "production"]

#: Environments that are real deployments, and therefore must be authenticated.
#:
#: Named positively so a new environment is secured by default: anything
#: added to `Environment` and to this set requires a key, whereas an
#: allow-list of "safe" environments would silently exempt it.
DEPLOYED_ENVIRONMENTS: frozenset[str] = frozenset({"production"})
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


#: What an API key is allowed to do.
#:
#: Only two, because there are only two mutating endpoints in the whole API:
#: appending a decision and appending an override. A finer taxonomy would be
#: scopes nobody could explain the difference between.
API_SCOPES: frozenset[str] = frozenset({"read", "write"})


@dataclass(frozen=True, slots=True)
class ApiKey:
    """One configured credential.

    ``secret`` is compared in constant time and never rendered. ``name`` and
    ``scope`` are safe to log - and are logged, because an audit trail that
    cannot say who did something is not an audit trail.
    """

    name: str
    secret: str
    scope: str

    @property
    def may_write(self) -> bool:
        return self.scope == "write"


class Settings(BaseSettings):
    """Root settings object. Construct once via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    env: Environment = Field(default="development", alias="RTO_ENV")
    log_level: LogLevel = Field(default="INFO", alias="RTO_LOG_LEVEL")

    api_host: str = Field(default="127.0.0.1", alias="RTO_API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="RTO_API_PORT")

    #: API keys that may call `/v1/*`, as `name:secret[:scope]` triples.
    #:
    #: Example::
    #:
    #:     RTO_API_KEYS=console:sk_live_abc...,ops:sk_live_def...:write
    #:
    #: The name is not a credential - it identifies the caller in rate-limit
    #: accounting and in the audit trail, so a leaked key can be revoked without
    #: guessing who was using it.
    #:
    #: **The scope defaults to `read` when omitted.** Least privilege by default:
    #: a key added in a hurry cannot append a decision or an override, and
    #: granting that power is a deliberate act of typing `:write`. The opposite
    #: default would mean every key ever issued could mutate the audit log.
    #:
    #: WHY API KEYS AND NOT OAUTH OR mTLS. There is no identity provider in this
    #: system for OAuth to talk to, and no certificate infrastructure for mTLS.
    #: Inventing either would mean shipping a login screen with nothing behind
    #: it. API keys are what this shape of service - a merchant console plus
    #: server-to-server callers - actually uses.
    api_keys_raw: SecretStr = Field(default=SecretStr(""), alias="RTO_API_KEYS")

    #: Requests per minute per key. 0 disables the limit.
    rate_limit_per_minute: int = Field(default=120, ge=0, alias="RTO_RATE_LIMIT_PER_MINUTE")

    #: Where the counters live.
    #:
    #: `memory` is correct for ONE uvicorn worker and wrong for several - each
    #: keeps its own buckets, so N workers permit N times the rate. `database`
    #: shares one counter across every worker at the cost of a round trip per
    #: request, which is noise beside a scoring endpoint that takes seconds.
    #:
    #: Default `memory` because the single-worker case should pay nothing.
    rate_limit_backend: Literal["memory", "database"] = Field(
        default="memory", alias="RTO_RATE_LIMIT_BACKEND"
    )

    cors_origins_raw: str = Field(default="http://localhost:5173", alias="RTO_CORS_ORIGINS")
    config_dir: Path = Field(default=Path("config"), alias="RTO_CONFIG_DIR")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="RTO_ARTIFACT_DIR")

    # Global seed for generator, splits and training. Recorded in run metadata:
    # changing it changes every reported number.
    random_seed: int = Field(default=20260827, alias="RTO_RANDOM_SEED")

    # Empty means no model is loaded, and the scoring endpoint says so rather
    # than inventing a probability.
    active_model_path: Path | None = Field(default=None, alias="RTO_ACTIVE_MODEL_PATH")

    # How many rows of merchant history the serving feature pipeline reads to
    # compute one order's aggregates. Lower is faster and gives the geography
    # features less evidence, so they shrink harder towards their prior; it never
    # makes them wrong, only less informed. Exposed because the right value
    # depends on book size, and the honest default is "enough".
    serving_context_limit: int = Field(
        default=20000, ge=100, le=1_000_000, alias="RTO_SERVING_CONTEXT_LIMIT"
    )

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
        """Allowed browser origins. A wildcard is refused, not silently accepted.

        The API sends ``Access-Control-Allow-Credentials: true``. Combined with a
        wildcard origin that is the classic CORS hole: Starlette cannot legally
        return ``*`` alongside credentials, so it echoes back whatever ``Origin``
        the request carried. The practical effect is that *any* site can make
        credentialed cross-origin requests and read the responses - the opposite
        of what someone setting ``*`` usually believes they are configuring.

        Verified rather than assumed: with ``RTO_CORS_ORIGINS=*`` the response to
        a request from ``https://evil.example.com`` came back with that exact
        origin echoed and credentials allowed.

        Refusing at startup is deliberate. Quietly dropping the wildcard would
        leave an operator believing a config that is not in force, and quietly
        dropping ``allow_credentials`` would change the API's semantics based on
        an unrelated variable.
        """
        origins = [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
        if any(origin == "*" for origin in origins):
            msg = (
                "RTO_CORS_ORIGINS contains '*'. This API allows credentials, and a "
                "wildcard origin with credentials makes every site a permitted origin - "
                "Starlette echoes the caller's own Origin header back. List the origins "
                "explicitly, comma-separated (e.g. "
                "'http://localhost:5173,https://console.example.com')."
            )
            raise ValueError(msg)
        return origins

    @property
    def api_keys(self) -> dict[str, ApiKey]:
        """Caller name -> credential. Empty when authentication is not configured.

        Not a ``computed_field``: a computed field is serialised, and the secret
        must never appear in ``/readiness``, a log line or an error body.
        """
        raw = self.api_keys_raw.get_secret_value().strip()
        if not raw:
            return {}

        keys: dict[str, ApiKey] = {}
        for entry in raw.split(","):
            item = entry.strip()
            if not item:
                continue

            parts = item.split(":")
            if len(parts) not in (2, 3):
                msg = (
                    "RTO_API_KEYS entries must be 'name:secret' or 'name:secret:scope' "
                    "so a leaked key can be attributed and revoked. Found an entry "
                    f"with {len(parts)} colon-separated part(s)."
                )
                raise ValueError(msg)

            name, secret = parts[0].strip(), parts[1].strip()
            scope = (parts[2].strip().lower() if len(parts) == 3 else "read") or "read"

            if not name or not secret:
                msg = "RTO_API_KEYS entries need a non-empty name and a non-empty secret."
                raise ValueError(msg)
            if scope not in API_SCOPES:
                msg = (
                    f"RTO_API_KEYS entry {name!r} has scope {scope!r}; valid scopes are "
                    f"{sorted(API_SCOPES)}. Omit the scope for read-only access."
                )
                raise ValueError(msg)

            keys[name] = ApiKey(name=name, secret=secret, scope=scope)
        return keys

    @computed_field  # type: ignore[prop-decorator]
    @property
    def authentication_enabled(self) -> bool:
        """Whether `/v1/*` requires a key. Reported by `/readiness`; the keys are not."""
        return bool(self.api_keys)

    @model_validator(mode="after")
    def _production_requires_authentication(self) -> Settings:
        """Refuse to start unauthenticated outside development.

        An API that silently runs open is how a risk system ends up serving every
        order in the book to anyone who can reach the port.

        The check names the DEPLOYED environments rather than allow-listing the
        local ones. `development` and `test` are not deployments - a local console
        with no key should work, and the suite should not have to invent
        credentials to exercise unrelated routes. Adding a `staging` environment
        later should require a key by default, and listing what must be secured
        is what makes that the default rather than something to remember.

        `/readiness` reports the state either way, so "open" is never a surprise.
        """
        if self.env in DEPLOYED_ENVIRONMENTS and not self.api_keys:
            msg = (
                f"RTO_ENV is {self.env!r} and RTO_API_KEYS is empty. This API would "
                "serve every order to anyone who can reach the port. Set "
                "RTO_API_KEYS='name:secret,...' before deploying."
            )
            raise ValueError(msg)
        return self

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
