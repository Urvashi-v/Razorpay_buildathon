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

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


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
