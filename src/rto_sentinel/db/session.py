"""Engine and session management.

One engine per process, created lazily so that importing the package does not
attempt a database connection - which matters for the ML pipeline and the test
suite, neither of which needs a database at all.

``session_scope`` is the only supported way to get a transaction. It commits on
success and rolls back on any exception, so a handler that raises halfway
through cannot leave a partially written decision log behind.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rto_sentinel.settings import Settings, get_settings


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide SQLAlchemy engine.

    ``pool_pre_ping`` is on because a scoring service that has been idle
    overnight should reconnect rather than fail the first order of the morning.
    """
    settings: Settings = get_settings()
    url = settings.database.url
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        # SQLite is used by the test suite only; allow cross-thread use so the
        # FastAPI TestClient can share one in-process database.
        connect_args["check_same_thread"] = False
    return create_engine(url, pool_pre_ping=True, future=True, connect_args=connect_args)


@functools.lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine and session factory.

    Used by tests after pointing ``RTO_DATABASE_URL`` at a temporary file. Not
    called anywhere in application code - an engine that can be swapped at
    runtime is a source of surprising behaviour in a service.
    """
    get_engine.cache_clear()
    get_session_factory.cache_clear()
