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

#: Seconds libpq may spend establishing one TCP connection before moving on.
#:
#: libpq's default is the operating system's, which on Windows is roughly two
#: minutes per address. That default is how an unreachable database becomes an
#: indefinite hang instead of an error: the request never reaches the handler,
#: nothing is logged, and the caller sees a spinner. Worse, it is silent in the
#: common case - a host that resolves to both ``::1`` and ``127.0.0.1`` where
#: only the IPv4 address is listening (a Docker container published as
#: ``127.0.0.1:5442->5432``) burns the full OS timeout on IPv6 and *then*
#: succeeds, so the first query after every restart takes two minutes and every
#: subsequent one is instant because the pool is warm.
#:
#: Five seconds is far longer than a healthy connect on any network worth
#: serving from, and short enough that a dead address surfaces as a 503 while
#: the operator is still looking at the screen.
CONNECT_TIMEOUT_SECONDS = 5


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide SQLAlchemy engine.

    ``pool_pre_ping`` is on because a scoring service that has been idle
    overnight should reconnect rather than fail the first order of the morning.
    """
    settings: Settings = get_settings()
    url = settings.database.url
    return create_engine(url, pool_pre_ping=True, future=True, connect_args=connect_args_for(url))


def connect_args_for(url: str) -> dict[str, object]:
    """Driver-level connection arguments for a database URL.

    Separated from ``get_engine`` so the policy can be asserted directly. The
    alternative is reaching into ``engine.pool._creator`` to recover what was
    passed, which is a private attribute whose shape differs between drivers.
    """
    if url.startswith("sqlite"):
        # SQLite is used by the test suite only; allow cross-thread use so the
        # FastAPI TestClient can share one in-process database. `connect_timeout`
        # is a libpq parameter and sqlite3 rejects it outright.
        return {"check_same_thread": False}
    return {"connect_timeout": CONNECT_TIMEOUT_SECONDS}


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
