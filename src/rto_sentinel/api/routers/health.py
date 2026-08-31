"""Health and readiness. Fully implemented - this endpoint tells the truth now.

The distinction between the two endpoints matters operationally:

``/health``
    Is the process up? Cheap, no I/O, safe to poll every second.

``/readiness``
    Can this instance actually do its job? Reports whether a model artefact is
    loaded, whether the configuration parses, and whether the language layer is
    available. **A missing model makes this endpoint unhealthy**, because an
    instance that cannot score an order should not be receiving traffic. A
    missing API key does not: the system is designed to run without it.

Nothing here leaks a secret. The database URL is reported with its password
redacted, and the presence of an API key is reported as a boolean - never the key,
never a prefix of it.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from rto_sentinel import __version__
from rto_sentinel.api.deps import ModelRegistryDep, SettingsDep
from rto_sentinel.configuration import ConfigurationError, config_fingerprint, load_app_config
from rto_sentinel.serving.model_registry import ModelUnavailableError

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: str


class ComponentStatus(BaseModel):
    """One component's readiness, with a reason when it is not ready."""

    ready: bool
    detail: str


class ReadinessResponse(BaseModel):
    ready: bool = Field(description="False when any component required to score is not ready")
    version: str
    environment: str
    config_fingerprint: str | None = Field(
        default=None, description="SHA-256 over the configuration bundle"
    )
    components: dict[str, ComponentStatus]
    warnings: list[str] = Field(default_factory=list)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: SettingsDep) -> HealthResponse:
    """Is the process up. No I/O, safe to poll aggressively."""
    return HealthResponse(version=__version__, environment=settings.env)


@router.get("/readiness", response_model=ReadinessResponse, summary="Readiness probe")
def readiness(
    settings: SettingsDep, registry: ModelRegistryDep, response: Response
) -> ReadinessResponse:
    """Can this instance score an order, and what is degraded if not."""
    components: dict[str, ComponentStatus] = {}
    warnings: list[str] = []
    fingerprint: str | None = None

    # --- configuration --------------------------------------------------
    try:
        load_app_config(settings)
        fingerprint = config_fingerprint(settings)
        components["configuration"] = ComponentStatus(
            ready=True, detail="All configuration files parsed and validated."
        )
    except ConfigurationError as exc:
        components["configuration"] = ComponentStatus(ready=False, detail=str(exc))

    # --- model artefact -------------------------------------------------
    # No model means this instance cannot score. It reports that plainly rather
    # than accepting traffic and returning a default probability.
    #
    # This asks the registry - the same object the scoring path uses - which
    # artefact it would serve, rather than re-deriving the answer here. An
    # earlier version checked `RTO_ACTIVE_MODEL_PATH` directly and reported the
    # instance unready whenever the pin was unset, while scoring worked fine off
    # the newest artefact in the store. A readiness probe that disagrees with
    # the code path it is reporting on is worse than no probe: it drains a
    # healthy instance out of the load balancer.
    #
    # `resolve` reads card JSON only, so this stays cheap enough to poll.
    try:
        artefact, card = registry.resolve()
    except ModelUnavailableError as exc:
        components["model"] = ComponentStatus(ready=False, detail=str(exc))
    else:
        pin = " (pinned)" if settings.active_model_path is not None else " (newest in store)"
        loaded = " and loaded" if registry.is_loaded else ""
        components["model"] = ComponentStatus(
            ready=True,
            detail=(
                f"{card.model_name} {card.model_version}{pin}, calibration "
                f"{card.calibration_method}, resolved{loaded} from {artefact.name}"
            ),
        )

    # --- database -------------------------------------------------------
    # Configuration only: no connection is opened here. A liveness or readiness
    # probe that opens a connection turns a slow database into a restart loop.
    components["database"] = ComponentStatus(
        ready=True, detail=f"Configured: {settings.database.safe_url}"
    )

    # --- language layer (optional) --------------------------------------
    # Never counted toward readiness. SPEC section 08: if every LLM call fails,
    # the system still scores, still thresholds, still acts.
    llm_reason = settings.llm.unavailable_reason
    if llm_reason is None:
        components["agents"] = ComponentStatus(
            ready=True, detail=f"Enabled, model {settings.llm.model}"
        )
    else:
        components["agents"] = ComponentStatus(
            ready=False,
            detail=(
                f"Unavailable ({llm_reason}). Explanations degrade to raw reason codes; "
                "scoring and decisions are unaffected."
            ),
        )
        warnings.append("Language layer unavailable - explanations will be reason codes only.")

    required = ("configuration", "model")
    ready = all(components[name].ready for name in required)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=ready,
        version=__version__,
        environment=settings.env,
        config_fingerprint=fingerprint,
        components=components,
        warnings=warnings,
    )
