"""FastAPI dependencies.

Everything a route handler needs arrives through this module: settings, config,
a database session, the decision engine, the LLM provider. Two consequences that
are the point of doing it this way:

* A handler never constructs a service, so a handler never accidentally becomes
  the place where policy is decided.
* A test can override any dependency without patching module globals, which is
  how the API tests run with no database and no API key.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, status
from sqlalchemy.orm import Session

from rto_sentinel.agents.provider import LLMProvider, get_provider
from rto_sentinel.api.errors import ApiError, ErrorCode
from rto_sentinel.configuration import AppConfig, load_app_config
from rto_sentinel.db.session import get_session_factory
from rto_sentinel.decision.engine import DecisionEngine
from rto_sentinel.models.final import ScoredBook, load_scored_book
from rto_sentinel.settings import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


def app_config_dep(settings: SettingsDep) -> AppConfig:
    """The validated YAML bundle.

    Loaded per request rather than cached at import time so a configuration edit
    is picked up by a restart-free reload in development. It is a handful of
    small files; the parse cost is not worth the staleness risk.
    """
    return load_app_config(settings)


AppConfigDep = Annotated[AppConfig, Depends(app_config_dep)]


def db_session_dep() -> Iterator[Session]:
    """A request-scoped session. Commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DbSession = Annotated[Session, Depends(db_session_dep)]


def decision_engine_dep(config: AppConfigDep) -> DecisionEngine:
    """The deterministic engine, built from policy config.

    Note what is *not* injected here: no LLM provider. The engine has no
    parameter through which a language model could reach it, which is the
    architectural rule expressed as a function signature.
    """
    return DecisionEngine(policy=config.policy)


DecisionEngineDep = Annotated[DecisionEngine, Depends(decision_engine_dep)]


def scored_book_dep(settings: SettingsDep) -> ScoredBook:
    """The calibrated validation book the economics endpoints price.

    VALIDATION, ALWAYS. There is no parameter through which a caller can request
    the sealed test split, which is the only reliable way to keep a slider from
    consuming it: a flag that defaults to safe still has an unsafe value.

    When no final model has been trained, this raises a 503 naming the command to
    run. It does not return an empty book or synthesised scores - an economics
    endpoint answering from nothing would produce rupee figures for a model that
    does not exist.
    """
    try:
        return load_scored_book(settings.artifact_path, split="validation")
    except FileNotFoundError as error:
        raise ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            str(error),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from error


ScoredBookDep = Annotated[ScoredBook, Depends(scored_book_dep)]


def llm_provider_dep(settings: SettingsDep) -> LLMProvider:
    """The language provider, or an unavailable one carrying its reason.

    Never raises here. A missing API key is a normal operating state for this
    system, and the routes that need language handle it explicitly.
    """
    return get_provider(settings.llm)


LLMProviderDep = Annotated[LLMProvider, Depends(llm_provider_dep)]
