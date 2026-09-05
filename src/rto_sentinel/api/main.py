"""FastAPI application factory.

Assembles routers, error handlers and CORS. Contains no business logic - the app
factory is wiring, and wiring is all it should ever be.

CORS is restricted to the origins in ``RTO_CORS_ORIGINS`` (the Vite dev server by
default). It is not ``*``: this API returns decisions about real orders, and a
wildcard origin on a service like that is a mistake that is easy to make on day
one and expensive to notice later.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rto_sentinel import __version__
from rto_sentinel.api.errors import register_exception_handlers
from rto_sentinel.api.routers import (
    decisions,
    economics,
    evaluation,
    explanations,
    health,
    monitoring,
    orders,
    scoring,
)
from rto_sentinel.api.security import AccessLogMiddleware
from rto_sentinel.settings import Settings, get_settings

DESCRIPTION = """\
Return-to-origin risk scoring for Indian cash-on-delivery commerce.

**How this API is layered.** A model returns a calibrated probability. A
deterministic engine converts that probability into an action using an explicit
rupee cost model, where the threshold is *derived* from the merchant's own margin
and freight numbers rather than assumed at 0.5. The language endpoints under
`/v1/explanations` run strictly downstream: they describe decisions that have
already been made, and if every one of them fails the system still scores orders
and still takes the right action.

**Data provenance.** Models are trained on synthetic data generated from
published Indian RTO base rates. Absolute metric values are not a claim about
production performance. See the project README for what synthetic data can and
cannot honestly demonstrate.

**Posture.** Defense-only. No component generates, simulates or optimises
fraudulent behaviour.
"""


def configure_logging(settings: Settings) -> None:
    """Honour `RTO_LOG_LEVEL`, and give the access log somewhere to go.

    This did not exist. `RTO_LOG_LEVEL` was a documented setting that configured
    nothing: the root logger stayed at WARNING, so every `logger.info` in the
    project was discarded - including, once it was added, the request audit
    trail. A knob that does nothing is worse than no knob, because someone sets
    it and believes the result.

    `force=True` because uvicorn installs its own handlers first. Without it this
    call is a no-op under the server and works only under the test client, which
    is the most confusing possible outcome.
    """
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        force=True,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Called by uvicorn and by the test suite alike."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="RTO Sentinel",
        description=DESCRIPTION,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Outermost, so it records the request whatever happens inside - including a
    # 401 from authentication and a 429 from the limiter, which are exactly the
    # lines worth having after a credential leaks.
    app.add_middleware(AccessLogMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(orders.router)
    app.include_router(scoring.router)
    app.include_router(economics.router)
    app.include_router(decisions.router)
    app.include_router(evaluation.router)
    app.include_router(explanations.router)
    app.include_router(monitoring.router)

    return app


# Module-level app for `uvicorn rto_sentinel.api.main:app`.
app = create_app()
