"""A single error envelope for the whole API.

One shape for every failure, so the console has one error path to render and one
place to look when something goes wrong.

The ``code`` field matters more than the HTTP status here. ``MODEL_UNAVAILABLE``
and ``AGENT_UNAVAILABLE`` are both 503, but they mean very different things: the
first means this system cannot score an order and the caller must not proceed;
the second means the explanation will be a bar chart instead of a sentence and
everything else is fine. A frontend that can only see the status cannot tell
those apart, and would either over-react to a missing sentence or under-react to
a missing model.

Nothing in this module ever puts a database URL, a settings object, or an
exception's raw string into a response body. Internal detail goes to the log;
the caller gets a code and a message written for them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rto_sentinel.decision.engine import UncalibratedScoreError
from rto_sentinel.serving.features import FeatureServiceError
from rto_sentinel.serving.model_registry import ModelMismatchError, ModelUnavailableError

LOGGER = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    """Machine-readable failure reasons."""

    VALIDATION_FAILED = "VALIDATION_FAILED"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    DECISION_NOT_FOUND = "DECISION_NOT_FOUND"

    #: No trained model artefact is loaded. The system cannot score, and says so
    #: rather than returning a default probability.
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    #: The score reaching the engine was not calibrated. Hard failure by design.
    UNCALIBRATED_SCORE = "UNCALIBRATED_SCORE"
    #: Cost inputs from which no finite threshold can be derived.
    INVALID_COST_INPUTS = "INVALID_COST_INPUTS"

    #: The language layer is off or unreachable. Degraded, not broken.
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    #: The generation named something it was not given; output withheld.
    GROUNDING_REJECTED = "GROUNDING_REJECTED"

    #: No usable API key was presented. Deliberately does not distinguish a
    #: missing key from a wrong one - telling a caller their format was right is
    #: telling an attacker their format was right.
    UNAUTHENTICATED = "UNAUTHENTICATED"
    #: The caller is authenticated but the credential does not carry the power
    #: this endpoint needs. 403, not 401 - re-checking a working key is wasted
    #: effort.
    FORBIDDEN = "FORBIDDEN"
    #: The caller exceeded its per-key allowance. The detail carries the retry
    #: delay so a client can back off rather than hammer.
    RATE_LIMITED = "RATE_LIMITED"

    #: The endpoint exists and its contract is fixed, but the phase that
    #: implements it has not landed. Honest 501 rather than a plausible stub.
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorBody(BaseModel):
    """The error payload. Every failing response has exactly this shape."""

    code: ErrorCode
    message: str
    detail: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    """Raised by routers and services to produce a structured error response."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail

    def to_response(self) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorBody(code=self.code, message=self.message, detail=self.detail)
        )
        return JSONResponse(status_code=self.status_code, content=body.model_dump(mode="json"))


def not_implemented(feature: str, phase: str) -> ApiError:
    """Build the standard 501 for a contract that exists but has no implementation.

    Used deliberately in place of a plausible-looking stub response. A frontend
    developer wiring against this endpoint should discover immediately that the
    data is not real, rather than build against fabricated numbers and find out
    later.
    """
    return ApiError(
        ErrorCode.NOT_IMPLEMENTED,
        f"{feature} is not implemented yet; it lands in {phase}.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"feature": feature, "phase": phase},
    )


def sanitise_validation_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Reduce Pydantic's error list to a JSON-serialisable, safe subset.

    Two problems with returning ``exc.errors()`` directly, both of which this
    solves:

    * **It does not serialise.** When a field validator raises ``ValueError``,
      Pydantic puts the *exception object itself* into ``ctx``. Encoding that
      raises ``PydanticSerializationError``, turning every custom-validator
      rejection into a 500 - so a caller sending a phone number where a hash
      belongs would get "internal server error" instead of being told what was
      wrong with their payload.
    * **``input`` echoes the caller's value back.** For most fields that is
      helpful. For ``customer_hash`` it would mean reflecting an identifier - and
      possibly a raw phone number, which is exactly the mistake the validator
      just caught - into a response body and any log that records it.

    So: keep the location, the type and the message. Drop the context and the
    input.
    """
    cleaned: list[dict[str, Any]] = []
    for error in errors:
        if not isinstance(error, dict):  # pragma: no cover - defensive
            continue
        cleaned.append(
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "type": str(error.get("type", "unknown")),
                "msg": str(error.get("msg", "")),
            }
        )
    return cleaned


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so every failure leaves through the same envelope."""

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return ApiError(
            ErrorCode.VALIDATION_FAILED,
            "The request payload failed validation.",
            # Literal 422 rather than the framework constant: Starlette renamed
            # HTTP_422_UNPROCESSABLE_ENTITY and deprecated the old spelling, and the
            # number is stable across every version we support.
            status_code=422,
            detail={"errors": sanitise_validation_errors(exc.errors())},
        ).to_response()

    # -- the serving path ------------------------------------------------
    #
    # Each of these is a *known* operational state with a specific meaning, and
    # each carries its message through to the client verbatim. That is
    # deliberate: "no calibrated model artefact exists, run `rto-sentinel final`"
    # is information an operator needs, and none of these messages contains a
    # path outside the artefact store, a credential, or a stack frame. Anything
    # not enumerated here falls through to the generic 500 below, which reveals
    # nothing.

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        return ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            str(exc),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        ).to_response()

    @app.exception_handler(ModelMismatchError)
    async def _model_mismatch(_: Request, exc: ModelMismatchError) -> JSONResponse:
        # 409, not 503: the service is up and a model is loaded. The deployment
        # is internally inconsistent, which is a different problem needing a
        # different response from whoever is paged.
        return ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            str(exc),
            status_code=status.HTTP_409_CONFLICT,
        ).to_response()

    @app.exception_handler(UncalibratedScoreError)
    async def _uncalibrated(_: Request, exc: UncalibratedScoreError) -> JSONResponse:
        return ApiError(
            ErrorCode.UNCALIBRATED_SCORE,
            str(exc),
            status_code=status.HTTP_409_CONFLICT,
        ).to_response()

    @app.exception_handler(FeatureServiceError)
    async def _feature_failure(_: Request, exc: FeatureServiceError) -> JSONResponse:
        return ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ).to_response()

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """The catch-all. Logs everything, returns nothing.

        The traceback goes to the server log where an operator can read it; the
        client gets a code and a request path. A stack trace in an HTTP response
        tells an attacker the framework, the file layout and often a query - and
        it is the single easiest way for a connection string to end up in
        somebody's browser console.
        """
        LOGGER.exception("unhandled error serving %s %s", request.method, request.url.path)
        return ApiError(
            ErrorCode.INTERNAL_ERROR,
            "The server failed to handle this request. The failure has been logged.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"path": request.url.path},
        ).to_response()
