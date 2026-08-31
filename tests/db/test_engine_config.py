"""Engine construction. Specifically: an unreachable database must fail, not hang.

WHY THIS TEST EXISTS
====================
Without an explicit `connect_timeout`, libpq uses the operating system's, which
on Windows is roughly two minutes *per resolved address*. That turned a
mundane local setup - a Docker container published as `127.0.0.1:5442->5432`,
with `localhost` in the URL resolving to `::1` first - into a 130-second first
request after every server restart, followed by instant responses once the pool
was warm. Nothing was logged, because the request never reached a handler.

The console showed it as one panel stuck on a spinner while its siblings, which
touch no database, rendered normally. Every plausible-looking explanation for
that shape - React StrictMode aborts, dev-proxy socket reuse, a request race -
was wrong.
"""

from __future__ import annotations

import pytest

from rto_sentinel.db.session import (
    CONNECT_TIMEOUT_SECONDS,
    connect_args_for,
    get_engine,
    reset_engine,
)
from rto_sentinel.settings import get_settings


@pytest.fixture(autouse=True)
def _clean_engine() -> object:
    """Each test builds its own engine and leaves none behind for the next."""
    reset_engine()
    get_settings.cache_clear()
    yield
    reset_engine()
    get_settings.cache_clear()


def test_postgres_connections_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead address surfaces as an error while an operator is still watching."""
    monkeypatch.setenv("RTO_DATABASE_URL", "postgresql+psycopg://user:pw@127.0.0.1:5999/nope")
    get_settings.cache_clear()

    engine = get_engine()

    assert engine.dialect.name == "postgresql"
    # `create_engine` does not connect, so this asserts the configuration that
    # will be handed to libpq rather than observing a connection.
    args = connect_args_for(str(engine.url))
    assert args["connect_timeout"] == CONNECT_TIMEOUT_SECONDS
    assert CONNECT_TIMEOUT_SECONDS <= 30, "a bound nobody waits out is not a bound"


def test_sqlite_does_not_receive_a_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """`connect_timeout` is a libpq parameter; sqlite3 rejects it outright."""
    monkeypatch.setenv("RTO_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()

    engine = get_engine()
    args = connect_args_for(str(engine.url))

    assert "connect_timeout" not in args
    assert args["check_same_thread"] is False


def test_an_unreachable_postgres_fails_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end behaviour the constant exists to produce.

    Port 5999 has nothing on it. The connection must be refused or time out
    promptly - what must not happen is the multi-minute block that made the
    console look broken.
    """
    import time

    from sqlalchemy.exc import OperationalError

    monkeypatch.setenv("RTO_DATABASE_URL", "postgresql+psycopg://user:pw@127.0.0.1:5999/nope")
    get_settings.cache_clear()

    started = time.perf_counter()
    with pytest.raises(OperationalError), get_engine().connect():
        pass
    elapsed = time.perf_counter() - started

    assert elapsed < CONNECT_TIMEOUT_SECONDS + 5, f"took {elapsed:.1f}s to give up"
