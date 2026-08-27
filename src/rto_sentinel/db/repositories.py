"""Repositories - the only place that reads or writes the database.

Every query in this application lives here. Route handlers do not build queries,
the decision engine does not touch a session, and the agent layer gets read-only
accessors and nothing else. Two reasons this boundary earns its keep:

* **The decision log is append-only in practice, not just in intent.** There is
  no ``update_decision``. A changed decision is a new row, which is what an audit
  trail requires. The absence of the method is the enforcement.
* **The agent layer cannot write.** :class:`ReadOnlyRepository` is the interface
  the agent toolset receives. It has no write methods to call, so "the LLM must
  not modify a decision" is a fact about the type it holds rather than a rule
  someone has to remember.

STATUS: Phase 4. Interfaces are fixed; the queries are not yet written.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

    from rto_sentinel.contracts.decision import Decision as DecisionContract
    from rto_sentinel.contracts.decision import OpsOverride
    from rto_sentinel.contracts.orders import OrderOutcomeUpdate, OrderPayload
    from rto_sentinel.db.models import Decision, Order


class ReadOnlyRepository(Protocol):
    """The read surface. This is what the agent layer is handed."""

    def get_order(self, order_id: str) -> Order | None: ...

    def get_latest_decision(self, order_id: str) -> Decision | None: ...

    def list_review_queue(self, merchant_id: str, limit: int = 50) -> list[Decision]: ...

    def digest_figures(
        self, merchant_id: str, period_start: datetime, period_end: datetime
    ) -> dict[str, float]:
        """Aggregate rupee figures for the weekly digest, computed in SQL.

        The digest writer receives the output of this method as the complete set
        of numbers it may mention. It does not compute anything itself.
        """
        ...


class OrderRepository:
    """Reads and writes orders and their outcomes."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, payload: OrderPayload) -> Order:
        raise NotImplementedError("Order persistence lands in Phase 4.")

    def get_order(self, order_id: str) -> Order | None:
        raise NotImplementedError("Order persistence lands in Phase 4.")

    def record_outcome(self, update: OrderOutcomeUpdate) -> None:
        """Record a terminal delivery state.

        Rejects an outcome whose ``resolved_at`` precedes the order's
        ``ordered_at``: that is not a late label, it is corrupt data, and letting
        it through would poison every as-of aggregate computed afterwards.
        """
        raise NotImplementedError("Outcome persistence lands in Phase 4.")


class DecisionRepository:
    """Append-only decision log. Note the absence of an update method."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, decision: DecisionContract, *, config_fingerprint: str) -> Decision:
        raise NotImplementedError("Decision logging lands in Phase 4.")

    def get_latest_decision(self, order_id: str) -> Decision | None:
        raise NotImplementedError("Decision logging lands in Phase 4.")

    def list_review_queue(self, merchant_id: str, limit: int = 50) -> list[Decision]:
        """SEVERE-band decisions awaiting a human, oldest first.

        Oldest first on purpose: a queue sorted by risk score leaves the least
        risky appeals waiting forever, and those are disproportionately the false
        positives - the customers who did nothing wrong.
        """
        raise NotImplementedError("Review queue lands in Phase 4.")


class OverrideRepository:
    """Ops overrides. Always available, always logged."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, override: OpsOverride) -> None:
        raise NotImplementedError("Override logging lands in Phase 4.")
