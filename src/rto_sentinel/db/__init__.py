"""Database layer: declarative models, session management, repositories.

Depends on: contracts, settings. Nothing in the ML pipeline imports this package
- training reads parquet files, not tables - so the model can be retrained
without a database running, and the database can be migrated without touching the
pipeline.
"""

from rto_sentinel.db.base import Base, TimestampMixin, utc_now
from rto_sentinel.db.models import (
    Decision,
    ModelRun,
    OpsOverrideRecord,
    Order,
    OrderOutcomeRecord,
)
from rto_sentinel.db.repositories import (
    DecisionRepository,
    OrderRepository,
    OverrideRepository,
    ReadOnlyRepository,
)
from rto_sentinel.db.session import get_engine, get_session_factory, reset_engine, session_scope

__all__ = [
    "Base",
    "Decision",
    "DecisionRepository",
    "ModelRun",
    "OpsOverrideRecord",
    "Order",
    "OrderOutcomeRecord",
    "OrderRepository",
    "OverrideRepository",
    "ReadOnlyRepository",
    "TimestampMixin",
    "get_engine",
    "get_session_factory",
    "reset_engine",
    "session_scope",
    "utc_now",
]
