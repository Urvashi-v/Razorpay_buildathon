"""Declarative base and shared column conventions.

Why SQLAlchemy 2.0 declarative rather than SQLModel: the API contracts in
``rto_sentinel.contracts`` and the storage schema here are *different things*
that happen to look similar today. A wire contract that changes for a frontend
reason should not force a migration, and a column added for auditing should not
appear in a public response. Keeping them as two type hierarchies makes that
separation structural rather than a matter of remembering to write a mapper.

Naming conventions are declared up front so Alembic autogenerates stable,
predictable constraint names instead of database-generated ones that differ
between engines and make migrations unreviewable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from sqlalchemy import DateTime, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every table in this application.

    ``type_annotation_map`` maps every ``datetime`` column to
    ``TIMESTAMP WITH TIME ZONE``. This is not a stylistic preference. The entire
    project turns on temporal reasoning - what was knowable at the instant an
    order was placed - and a naive timestamp column silently discards the offset
    on write, so a value round-trips as a different instant depending on the
    server's timezone. An as-of join computed against those is wrong in a way
    that produces plausible numbers, which is the worst kind of wrong.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # RUF012 does not know this is a SQLAlchemy declarative hook rather than a
    # mutable default that instances might share.
    type_annotation_map: ClassVar[dict[object, object]] = {datetime: DateTime(timezone=True)}


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp.

    Used as a column default rather than ``datetime.utcnow`` (naive) or a
    database ``now()`` (engine-dependent), so timestamps compare correctly across
    SQLite in tests and PostgreSQL in development.
    """
    return datetime.now(UTC)


class TimestampMixin:
    """``created_at`` on every row. Append-only tables never get ``updated_at``."""

    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False, index=True)
