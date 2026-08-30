"""The LLM client boundary.

EXTERNAL SERVICE
================
Anthropic Claude API - https://api.anthropic.com

Required environment variable: ``ANTHROPIC_API_KEY``
Also read: ``RTO_LLM_MODEL``, ``RTO_LLM_MAX_TOKENS``, ``RTO_LLM_TIMEOUT_SECONDS``,
``RTO_AGENTS_ENABLED``, ``ANTHROPIC_BASE_URL`` (optional, for a gateway).

There is no default key, no bundled key, and no development fallback that
pretends to be the service. If the key is absent, :func:`get_provider` returns
:class:`UnavailableProvider`, whose every call raises
:class:`AgentUnavailableError`, and the API surfaces that to the caller as an
explicit "explanation unavailable". The console then shows the raw reason codes.

WHY IT IS BUILT THIS WAY
------------------------
SPEC section 08: "If every LLM call fails, the system still scores orders, still
applies the correct threshold, and still takes the right action - it just
explains itself less gracefully."

Degrading the wording is acceptable. Substituting canned text that looks like
model output is not: it would make a demo look complete while hiding the fact
that a dependency is down, and that is the sort of thing that gets discovered in
production by a customer rather than by an engineer.

**There is no scripted responder in this module, and adding one would be a bug.**
The test suite drives the agents through a recording double that lives in
``tests/``, never in ``src/``. That distinction is the whole point: a double in a
test proves the orchestration works; a double in the product hides that it does
not.

WHAT THE SURFACE INTENTIONALLY OMITS
------------------------------------
No streaming, no conversation persistence, no retry-into-the-request. An
explanation that takes eight seconds to arrive is worse than one that never
arrives, because the ops queue has already moved on. Tool use *is* supported,
because the risk investigation agent has to fetch its own evidence - but the loop
that drives it lives in ``agents.investigator``, and every tool it can reach is
read-only by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.settings import LLMSettings

#: Environment variable the operator must set. Named once, here, so an error
#: message and the documentation cannot drift apart.
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"
ENABLE_VARIABLE = "RTO_AGENTS_ENABLED"


class AgentUnavailableError(RuntimeError):
    """Raised when the language layer cannot run.

    Carries the reason so the API can say *why* rather than returning a bare
    503: no key configured, agents switched off, upstream timeout, and so on.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the model asked to run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Completion:
    """One turn of the model's output.

    ``tool_calls`` is non-empty when the model wants evidence before answering.
    The caller runs them and sends the results back; nothing in this module
    executes a tool, because deciding what an agent may run is a permission
    question and belongs with the toolset, not with the HTTP client.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    """The narrow surface the agent jobs are allowed to use.

    Deliberately minimal. There is no streaming and no conversation state,
    because none of the language jobs needs them. A smaller surface is a smaller
    thing to reason about when asking whether an LLM can influence a decision -
    and it cannot, because nothing here returns a number the decision layer
    reads.
    """

    @property
    def available(self) -> bool: ...

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        """Return the model's text response, or raise :class:`AgentUnavailableError`."""
        ...

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """One turn, optionally with tools offered. Raises when unavailable."""
        ...


class UnavailableProvider:
    """The provider used when no key is configured or agents are switched off.

    Every method raises. This class exists so that "no LLM" is a well-typed,
    testable state rather than a ``None`` check scattered through five call sites.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def model(self) -> str:
        return "unavailable"

    @property
    def reason(self) -> str:
        return self._reason

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        raise AgentUnavailableError(self._reason)

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        raise AgentUnavailableError(self._reason)


class AnthropicProvider:
    """Thin wrapper over the Anthropic Messages API.

    * The ``anthropic`` SDK is imported lazily in the constructor, so the core
      system installs and runs without it. A missing SDK is reported as an
      unavailable language layer, not as an import error at startup.
    * The key comes from :class:`~rto_sentinel.settings.LLMSettings` as a
      ``SecretStr``, is read exactly once here, and is never logged, echoed in an
      error, or returned from an endpoint.
    * Timeouts are short and failures are not retried into a user-facing request.
    """

    def __init__(self, settings: LLMSettings) -> None:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - depends on the environment
            msg = (
                "the `anthropic` SDK is not installed, so the language layer cannot run. "
                "Install it with `pip install anthropic` (it is declared in pyproject.toml). "
                "The risk system is unaffected: scoring, calibration and the decision "
                "engine do not depend on it."
            )
            raise AgentUnavailableError(msg) from error

        if settings.api_key is None:  # pragma: no cover - guarded by get_provider
            raise AgentUnavailableError(f"{API_KEY_VARIABLE} is not set")

        self._settings = settings
        self._model = settings.model
        self._max_tokens = settings.max_tokens
        # `base_url=None` lets the SDK use its own default; an operator pointing
        # ANTHROPIC_BASE_URL at a gateway is honoured rather than ignored.
        self._client = anthropic.Anthropic(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=0,
        )
        self._errors = (
            anthropic.APIError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        )

    @property
    def available(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        result = self.converse(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return result.text

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """One Messages API turn.

        Failures are translated into :class:`AgentUnavailableError` with the
        exception *type* in the message and never the response body. An upstream
        error payload can echo the request, and the request contains order data.
        """
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools

        try:
            response = self._client.messages.create(**request)
        except self._errors as error:
            msg = (
                f"the Anthropic API call failed ({type(error).__name__}). The decision "
                "itself is unaffected; only its explanation is unavailable."
            )
            raise AgentUnavailableError(msg) from error

        return _to_completion(response)


def _to_completion(response: Any) -> Completion:
    """Flatten a Messages response into the shape the agent loop consumes."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    raw: list[dict[str, Any]] = []

    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
            raw.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            )
            raw.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input or {}),
                }
            )

    usage = getattr(response, "usage", None)
    return Completion(
        text="\n".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        stop_reason=getattr(response, "stop_reason", None),
        raw_content=raw,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def get_provider(settings: LLMSettings) -> LLMProvider:
    """Return a live provider, or an :class:`UnavailableProvider` with a reason.

    Never raises. "No language layer" is a normal operating state for this
    system, and the endpoints that need one handle it explicitly.
    """
    reason = settings.unavailable_reason
    if reason is not None:
        return UnavailableProvider(reason)
    try:
        return AnthropicProvider(settings)
    except AgentUnavailableError as error:
        return UnavailableProvider(error.reason)
