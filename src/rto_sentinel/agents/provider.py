"""The LLM client boundary.

EXTERNAL SERVICE
================
Anthropic Claude API - https://api.anthropic.com

Required environment variable: ``ANTHROPIC_API_KEY``
Also read: ``RTO_LLM_MODEL``, ``RTO_LLM_MAX_TOKENS``, ``RTO_LLM_TIMEOUT_SECONDS``,
``RTO_AGENTS_ENABLED``.

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

STATUS: Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.settings import LLMSettings


class AgentUnavailableError(RuntimeError):
    """Raised when the language layer cannot run.

    Carries the reason so the API can say *why* rather than returning a bare
    503: no key configured, agents switched off, upstream timeout, and so on.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LLMProvider(Protocol):
    """The narrow surface the agent jobs are allowed to use.

    Deliberately minimal. There is no streaming, no tool-calling loop and no
    conversation state, because none of the four language jobs needs them. A
    smaller surface is a smaller thing to reason about when asking whether an LLM
    can influence a decision - and it cannot, because nothing here returns a
    number the decision layer reads.
    """

    @property
    def available(self) -> bool: ...

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        """Return the model's text response, or raise :class:`AgentUnavailableError`."""
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

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        raise AgentUnavailableError(self._reason)


class AnthropicProvider:
    """Thin wrapper over the Anthropic Messages API.

    Implementation notes for Phase 5:

    * The ``anthropic`` SDK is an OPTIONAL dependency (``pip install -e
      '.[agents]'``). Import it lazily inside the constructor so the core system
      installs and runs without it.
    * The key comes from :class:`~rto_sentinel.settings.LLMSettings` as a
      ``SecretStr`` and is never logged, never echoed in an error, and never
      returned from an endpoint.
    * Timeouts are short and failures are not retried into a user-facing request.
      An explanation that takes eight seconds to arrive is worse than one that
      never arrives, because the ops queue has already moved on.
    """

    def __init__(self, settings: LLMSettings) -> None:
        raise NotImplementedError("Anthropic provider lands in Phase 5.")

    @property
    def available(self) -> bool:
        raise NotImplementedError("Anthropic provider lands in Phase 5.")

    @property
    def model(self) -> str:
        raise NotImplementedError("Anthropic provider lands in Phase 5.")

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        raise NotImplementedError("Anthropic provider lands in Phase 5.")


def get_provider(settings: LLMSettings) -> LLMProvider:
    """Return a live provider, or an :class:`UnavailableProvider` with a reason."""
    reason = settings.unavailable_reason
    if reason is not None:
        return UnavailableProvider(reason)
    return AnthropicProvider(settings)
