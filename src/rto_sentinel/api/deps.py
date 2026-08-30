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

import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, status
from sqlalchemy.orm import Session

from rto_sentinel.agents.provider import LLMProvider, get_provider
from rto_sentinel.api.errors import ApiError, ErrorCode
from rto_sentinel.configuration import AppConfig, load_app_config
from rto_sentinel.contracts.decision import CostInputs
from rto_sentinel.db.repositories import (
    DecisionLogRepository,
    OpsOverrideRepository,
    ServingRepository,
)
from rto_sentinel.db.session import get_session_factory
from rto_sentinel.decision.engine import DecisionEngine
from rto_sentinel.models.final import ScoredBook, load_scored_book
from rto_sentinel.serving import (
    AssessmentService,
    ModelRegistry,
    OrderFeatureService,
    ScoringService,
)
from rto_sentinel.serving.agent_tools import ApplicationToolset
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


def agent_toolset_dep(
    repository: ServingRepositoryDep,
    assessments: AssessmentServiceDep,
    session: DbSession,
) -> ApplicationToolset:
    """The read-only tools an agent is handed.

    Constructed per request so its per-order assessment cache lives exactly as
    long as one agent run. An agent has no other route to data: it cannot import
    a repository, and the layering tests assert that it never will.
    """
    return ApplicationToolset(repository, assessments, session)


AgentToolsetDep = Annotated[ApplicationToolset, Depends(agent_toolset_dep)]


def llm_provider_dep(settings: SettingsDep) -> LLMProvider:
    """The language provider, or an unavailable one carrying its reason.

    Never raises here. A missing API key is a normal operating state for this
    system, and the routes that need language handle it explicitly.
    """
    return get_provider(settings.llm)


LLMProviderDep = Annotated[LLMProvider, Depends(llm_provider_dep)]


# ---------------------------------------------------------------------------
# the serving path
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _registry_for(artifact_root: Path) -> ModelRegistry:
    """One registry per artefact root, for the life of the process.

    Cached deliberately. The registry holds a deserialised booster; building one
    per request would be slow and would make the served model a function of
    whatever is on disk at that instant rather than a stable, reportable version.
    The cache is keyed on the path so a test pointing at a temporary artefact
    store gets its own registry instead of the production one.
    """
    return ModelRegistry(artifact_root)


def model_registry_dep(settings: SettingsDep) -> ModelRegistry:
    """The process-wide model registry. Never loads the artefact by itself."""
    return _registry_for(settings.artifact_path)


ModelRegistryDep = Annotated[ModelRegistry, Depends(model_registry_dep)]


def serving_repository_dep(session: DbSession) -> ServingRepository:
    return ServingRepository(session)


ServingRepositoryDep = Annotated[ServingRepository, Depends(serving_repository_dep)]


def decision_log_dep(session: DbSession) -> DecisionLogRepository:
    return DecisionLogRepository(session)


DecisionLogDep = Annotated[DecisionLogRepository, Depends(decision_log_dep)]


def override_repository_dep(session: DbSession) -> OpsOverrideRepository:
    return OpsOverrideRepository(session)


OverrideRepositoryDep = Annotated[OpsOverrideRepository, Depends(override_repository_dep)]


def feature_service_dep(
    repository: ServingRepositoryDep, config: AppConfigDep, settings: SettingsDep
) -> OrderFeatureService:
    """The feature service, built from the same config the trainer used."""
    return OrderFeatureService(
        repository,
        features_config=config.features,
        generator_config=config.generator,
        context_limit=settings.serving_context_limit,
    )


FeatureServiceDep = Annotated[OrderFeatureService, Depends(feature_service_dep)]


def scoring_service_dep(registry: ModelRegistryDep, features: FeatureServiceDep) -> ScoringService:
    return ScoringService(registry, features)


ScoringServiceDep = Annotated[ScoringService, Depends(scoring_service_dep)]


def cost_inputs_for(config: AppConfig, profile_key: str | None = None) -> tuple[CostInputs, str]:
    """The merchant economics a request should use, and which profile they are.

    Shared by every router that needs them so a handler cannot quietly assemble a
    different set of cost inputs than the one the report was written against.
    """
    cost_model = config.cost_model
    key = profile_key or cost_model.default_profile
    profile = cost_model.profiles.get(key)
    if profile is None:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"unknown cost profile {key!r}",
            detail={"available": sorted(cost_model.profiles)},
        )
    return (
        CostInputs(
            rto_cost_inr=profile.rto_cost_inr,
            contribution_margin_inr=profile.contribution_margin_inr,
            abandonment_on_friction=profile.abandonment_on_friction,
            intervention_success_rate=profile.intervention_success_rate,
            friction_support_cost_inr=profile.friction_support_cost_inr,
        ),
        key,
    )


def assessment_service_dep(
    repository: ServingRepositoryDep,
    scoring: ScoringServiceDep,
    engine: DecisionEngineDep,
    config: AppConfigDep,
) -> AssessmentService:
    """The whole chain, assembled. Route handlers call this and nothing else."""
    cost_inputs, profile = cost_inputs_for(config)
    return AssessmentService(
        repository,
        scoring,
        engine,
        default_cost_inputs=cost_inputs,
        default_cost_profile=profile,
    )


AssessmentServiceDep = Annotated[AssessmentService, Depends(assessment_service_dep)]
