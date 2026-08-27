"""Tools the agent layer may call. All read-only, all narrow, all logged.

This is the complete list of capabilities available to the language layer. It is
short on purpose, and everything absent from it is absent deliberately.

WHAT THE AGENTS CAN DO
----------------------
* Read a decision that has already been made, with its reason codes.
* Read aggregate figures computed by SQL, for the weekly digest.
* Read the merchant's message templates.

WHAT NO TOOL HERE PERMITS
-------------------------
* Writing a risk score, a threshold, a band or an action.
* Overriding, re-running, or re-scoring a decision.
* Reading raw customer PII. Tools return hashed identities and derived
  attributes; nothing returns a name, a phone number, or a full address.
* Reaching any system outside this application's own database.
* Sending a message to a customer. The agent *drafts*; a human-reviewed
  template and the merchant's own messaging system send.

The asymmetry is the whole design: agents can read what happened and describe
it, and they cannot change what happens. ``tests/architecture/test_layering.py``
checks that no write-capable repository method is importable from this package.

STATUS: Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.decision import Decision


@dataclass(frozen=True, slots=True)
class DigestFigures:
    """Aggregate figures for one merchant over one period, computed in SQL.

    Handed to the digest writer as the complete set of numbers it may mention.
    The LLM does not compute, sum, or infer any figure - a wrong number in the
    digest is therefore a bug in a query, which is findable, rather than a
    hallucination, which is not.
    """

    merchant_id: str
    period_start: datetime
    period_end: datetime
    figures: dict[str, float]


class AgentToolset(Protocol):
    """Read-only accessors available to the language jobs."""

    def get_decision(self, order_id: str) -> Decision | None:
        """Fetch a decision that has already been made and logged."""
        ...

    def get_digest_figures(
        self, merchant_id: str, period_start: datetime, period_end: datetime
    ) -> DigestFigures:
        """Compute the weekly aggregate figures in SQL."""
        ...

    def get_message_template(self, template_id: str) -> str:
        """Fetch a human-reviewed message template with its variable slots."""
        ...
