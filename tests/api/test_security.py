"""Authentication and rate limiting.

The tests that matter most are the negative ones, and one positive one that is
easy to forget: an API which refuses everything is not secure, it is broken.

Two properties are load-bearing:

1. **An unauthenticated instance cannot exist outside development.** `Settings`
   refuses to construct, so there is no way to deploy this open by accident.
2. **`/health` and `/readiness` stay open.** A probe that needs a credential
   fails during exactly the incident it exists to report.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from rto_sentinel.api.main import create_app
from rto_sentinel.api.security import (
    ACCESS_LOG,
    API_KEY_HEADER,
    AccessLogMiddleware,
    DatabaseRateLimiter,
    RateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)
from rto_sentinel.db.models import RateLimitWindow
from rto_sentinel.settings import get_settings

CONSOLE_KEY = "sk_test_console_9f3a2b7c1d4e"
OPS_KEY = "sk_test_ops_5e8d1a6b0c2f"

#: A route that needs neither a model nor a database, so these tests measure
#: authentication rather than artefact availability.
PROTECTED = "/v1/explanations/status"


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    get_settings.cache_clear()
    reset_rate_limiter()
    yield
    get_settings.cache_clear()
    reset_rate_limiter()


def client_with(monkeypatch: pytest.MonkeyPatch, **env: str) -> TestClient:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    reset_rate_limiter()
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture
def secured(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with client_with(
        monkeypatch,
        RTO_API_KEYS=f"console:{CONSOLE_KEY},ops:{OPS_KEY}",
        RTO_RATE_LIMIT_PER_MINUTE="120",
    ) as client:
        yield client


@pytest.fixture
def open_instance(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with client_with(monkeypatch, RTO_API_KEYS="", RTO_ENV="development") as client:
        yield client


class TestAuthentication:
    def test_a_valid_key_is_accepted(self, secured: TestClient) -> None:
        """An API that refuses everything is broken, not secure."""
        response = secured.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY})
        assert response.status_code == 200

    def test_a_bearer_token_is_accepted(self, secured: TestClient) -> None:
        response = secured.get(PROTECTED, headers={"Authorization": f"Bearer {CONSOLE_KEY}"})
        assert response.status_code == 200

    def test_no_key_is_refused(self, secured: TestClient) -> None:
        response = secured.get(PROTECTED)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"

    def test_a_wrong_key_is_refused(self, secured: TestClient) -> None:
        response = secured.get(PROTECTED, headers={API_KEY_HEADER: "sk_test_not_a_key"})
        assert response.status_code == 401

    def test_a_prefix_of_a_valid_key_is_refused(self, secured: TestClient) -> None:
        """Guards against a comparison that stops at the first difference."""
        response = secured.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY[:-1]})
        assert response.status_code == 401

    def test_the_error_does_not_reveal_whether_the_format_was_right(
        self, secured: TestClient
    ) -> None:
        """A message distinguishing "no key" from "wrong key" helps an attacker.

        Both must be 401 with the same code; only the human-facing sentence
        differs, and neither confirms that a presented value looked plausible.
        """
        missing = secured.get(PROTECTED).json()["error"]
        wrong = secured.get(PROTECTED, headers={API_KEY_HEADER: "x"}).json()["error"]

        assert missing["code"] == wrong["code"] == "UNAUTHENTICATED"
        for body in (missing, wrong):
            assert CONSOLE_KEY not in str(body)
            assert "console" not in str(body).lower()

    def test_no_response_ever_contains_a_configured_key(self, secured: TestClient) -> None:
        for path in ("/health", "/readiness", PROTECTED):
            raw = secured.get(path, headers={API_KEY_HEADER: CONSOLE_KEY}).text
            assert CONSOLE_KEY not in raw
            assert OPS_KEY not in raw

    @pytest.mark.parametrize("path", ["/health", "/readiness"])
    def test_probes_stay_open(self, secured: TestClient, path: str) -> None:
        """A probe needing a credential fails during the incident it reports on."""
        assert secured.get(path).status_code in {200, 503}

    def test_an_open_instance_serves_without_a_key(self, open_instance: TestClient) -> None:
        assert open_instance.get(PROTECTED).status_code == 200

    def test_an_open_instance_says_so_on_readiness(self, open_instance: TestClient) -> None:
        """Open is allowed in development; being quiet about it is not."""
        body = open_instance.get("/readiness").json()

        assert "DISABLED" in body["components"]["authentication"]["detail"]
        assert any("Authentication is disabled" in w for w in body["warnings"])

    def test_a_secured_instance_lists_key_names_but_not_secrets(self, secured: TestClient) -> None:
        detail = secured.get("/readiness").json()["components"]["authentication"]["detail"]

        assert "console" in detail and "ops" in detail
        assert CONSOLE_KEY not in detail and OPS_KEY not in detail


class TestProductionGuard:
    def test_production_without_keys_refuses_to_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no way to deploy this open by accident."""
        monkeypatch.setenv("RTO_ENV", "production")
        monkeypatch.setenv("RTO_API_KEYS", "")
        get_settings.cache_clear()

        with pytest.raises(ValueError, match="RTO_API_KEYS is empty"):
            get_settings()

    def test_production_with_keys_starts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RTO_ENV", "production")
        monkeypatch.setenv("RTO_API_KEYS", f"console:{CONSOLE_KEY}")
        get_settings.cache_clear()

        assert get_settings().authentication_enabled is True

    def test_development_without_keys_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RTO_ENV", "development")
        monkeypatch.setenv("RTO_API_KEYS", "")
        get_settings.cache_clear()

        assert get_settings().authentication_enabled is False

    @pytest.mark.parametrize("bad", ["nocolon", "name:", ":secret"])
    def test_a_malformed_key_entry_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        """A key with no name cannot be attributed or revoked."""
        monkeypatch.setenv("RTO_API_KEYS", bad)
        get_settings.cache_clear()

        with pytest.raises(ValueError, match="RTO_API_KEYS"):
            _ = get_settings().api_keys


class TestRateLimiter:
    def test_requests_under_the_limit_pass(self) -> None:
        limiter = RateLimiter(limit_per_minute=5)
        for _ in range(5):
            allowed, _, _ = limiter.check("console")
            assert allowed

    def test_the_request_over_the_limit_is_refused(self) -> None:
        limiter = RateLimiter(limit_per_minute=3)
        for _ in range(3):
            limiter.check("console")

        allowed, remaining, retry_after = limiter.check("console")

        assert allowed is False
        assert remaining == 0
        assert 0 < retry_after <= 60

    def test_callers_have_separate_buckets(self) -> None:
        """One noisy key must not throttle another merchant."""
        limiter = RateLimiter(limit_per_minute=2)
        limiter.check("console")
        limiter.check("console")

        assert limiter.check("console")[0] is False
        assert limiter.check("ops")[0] is True

    def test_the_window_slides_rather_than_resetting_on_the_minute(self) -> None:
        """A fixed window allows a double burst across the boundary.

        Spending the whole allowance at t=59 and again at t=61 is twice the
        intended rate at exactly the moment a burst is most likely.
        """
        limiter = RateLimiter(limit_per_minute=2)
        limiter.check("console", now=0.0)
        limiter.check("console", now=1.0)

        assert limiter.check("console", now=30.0)[0] is False
        # The first hit ages out at t=60, freeing exactly one slot.
        assert limiter.check("console", now=60.5)[0] is True
        assert limiter.check("console", now=60.6)[0] is False

    def test_a_zero_limit_disables_limiting(self) -> None:
        limiter = RateLimiter(limit_per_minute=0)
        for _ in range(1000):
            assert limiter.check("console")[0] is True


class TestRateLimitEndToEnd:
    def test_exceeding_the_limit_returns_429_with_a_retry_delay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with client_with(
            monkeypatch,
            RTO_API_KEYS=f"console:{CONSOLE_KEY}",
            RTO_RATE_LIMIT_PER_MINUTE="3",
        ) as client:
            headers = {API_KEY_HEADER: CONSOLE_KEY}
            for _ in range(3):
                assert client.get(PROTECTED, headers=headers).status_code == 200

            response = client.get(PROTECTED, headers=headers)

        assert response.status_code == 429
        body = response.json()["error"]
        assert body["code"] == "RATE_LIMITED"
        assert body["detail"]["retry_after_seconds"] > 0

    def test_the_limit_is_per_key_not_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two merchants share a deployment; one must not exhaust the other."""
        with client_with(
            monkeypatch,
            RTO_API_KEYS=f"console:{CONSOLE_KEY},ops:{OPS_KEY}",
            RTO_RATE_LIMIT_PER_MINUTE="2",
        ) as client:
            for _ in range(2):
                client.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY})

            assert client.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY}).status_code == 429
            assert client.get(PROTECTED, headers={API_KEY_HEADER: OPS_KEY}).status_code == 200

    def test_probes_are_not_rate_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Throttling a health probe turns a busy service into a restarting one."""
        with client_with(
            monkeypatch,
            RTO_API_KEYS=f"console:{CONSOLE_KEY}",
            RTO_RATE_LIMIT_PER_MINUTE="2",
        ) as client:
            for _ in range(10):
                assert client.get("/health").status_code == 200


class TestScopes:
    """Least privilege: a read key must not be able to touch the audit log."""

    #: The only two endpoints in this API that change stored state.
    WRITES = ("/v1/decisions", "/v1/decisions/override")

    def test_a_key_without_a_scope_is_read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The safe default. A key added in a hurry cannot mutate anything."""
        monkeypatch.setenv("RTO_API_KEYS", f"console:{CONSOLE_KEY}")
        get_settings.cache_clear()

        assert get_settings().api_keys["console"].scope == "read"
        assert get_settings().api_keys["console"].may_write is False

    def test_the_write_scope_is_granted_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RTO_API_KEYS", f"ops:{OPS_KEY}:write")
        get_settings.cache_clear()

        assert get_settings().api_keys["ops"].may_write is True

    def test_an_unknown_scope_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`admin` looks plausible and grants nothing. Fail loudly instead."""
        monkeypatch.setenv("RTO_API_KEYS", f"ops:{OPS_KEY}:admin")
        get_settings.cache_clear()

        with pytest.raises(ValueError, match="valid scopes are"):
            _ = get_settings().api_keys

    @pytest.mark.parametrize("path", WRITES)
    def test_a_read_key_is_forbidden_from_writing(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        """403, not 401: the key is valid, it simply lacks the power.

        A 401 would send the caller off to re-check a key that works.
        """
        with client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client:
            response = client.post(
                path,
                json={"order_id": "ORD-00000001"},
                headers={API_KEY_HEADER: CONSOLE_KEY},
            )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_the_refusal_names_the_key_but_not_the_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming the key is the point - the operator has to know which one."""
        with client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client:
            body = client.post(
                "/v1/decisions",
                json={"order_id": "ORD-00000001"},
                headers={API_KEY_HEADER: CONSOLE_KEY},
            ).text

        assert "console" in body
        assert CONSOLE_KEY not in body

    def test_a_read_key_may_still_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scoping must not break the thing the key exists for."""
        with client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client:
            assert client.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY}).status_code == 200

    @pytest.mark.parametrize("path", WRITES)
    def test_a_write_key_passes_the_scope_check(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        """Past the guard.

        The request goes on to fail on a missing order or a missing model, which
        is exactly the point: the scope check is no longer what stopped it.
        """
        with client_with(monkeypatch, RTO_API_KEYS=f"ops:{OPS_KEY}:write") as client:
            response = client.post(
                path, json={"order_id": "ORD-00000001"}, headers={API_KEY_HEADER: OPS_KEY}
            )

        assert response.status_code != 403

    def test_an_open_instance_may_write(self, open_instance: TestClient) -> None:
        """Restricting an open dev instance would break the console for nothing.

        `Settings` already refuses to run open in a deployed environment, so this
        branch cannot be reached anywhere it would matter.
        """
        response = open_instance.post("/v1/decisions", json={"order_id": "ORD-00000001"})
        assert response.status_code != 403


class _Recorder(logging.Handler):
    """Collects access-log lines.

    Not `caplog`: `create_app` calls `configure_logging`, which uses
    `basicConfig(force=True)` because uvicorn installs its own handlers first.
    That removes every root handler - including the one pytest installed - so a
    `caplog` block entered before the client sees nothing. Attaching straight to
    the access logger after the app exists sidesteps the ordering entirely.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@contextmanager
def recording() -> Iterator[_Recorder]:
    recorder = _Recorder()
    previous = ACCESS_LOG.level
    ACCESS_LOG.setLevel(logging.INFO)
    ACCESS_LOG.addHandler(recorder)
    try:
        yield recorder
    finally:
        ACCESS_LOG.removeHandler(recorder)
        ACCESS_LOG.setLevel(previous)


class TestAccessLog:
    """Which key read which order. The first question asked after a leak."""

    def test_a_successful_request_records_the_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with (
            client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client,
            recording() as log,
        ):
            client.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY})

        assert "caller=console" in log.text
        assert "scope=read" in log.text
        assert "status=200" in log.text

    def test_a_refused_request_is_recorded_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run of 401s is what probing looks like, so it must leave a trace."""
        with (
            client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client,
            recording() as log,
        ):
            client.get(PROTECTED)

        assert "caller=unresolved" in log.text
        assert "status=401" in log.text

    def test_the_key_is_never_written_to_the_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A secret in a log is a secret in every backup and every shipper."""
        with (
            client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client,
            recording() as log,
        ):
            client.get(PROTECTED, headers={API_KEY_HEADER: CONSOLE_KEY})
            client.get(PROTECTED, headers={"Authorization": f"Bearer {CONSOLE_KEY}"})

        assert log.lines, "the requests produced no audit line at all"
        assert CONSOLE_KEY not in log.text

    def test_the_query_string_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """*Which* orders were listed is the part worth auditing."""
        with (
            client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client,
            recording() as log,
        ):
            client.get(f"{PROTECTED}?verbose=1", headers={API_KEY_HEADER: CONSOLE_KEY})

        assert "query=verbose=1" in log.text

    def test_probes_are_not_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """At one per second they would bury every line that matters."""
        with (
            client_with(monkeypatch, RTO_API_KEYS=f"console:{CONSOLE_KEY}") as client,
            recording() as log,
        ):
            client.get("/health")
            client.get("/readiness")

        assert log.text == ""

    def test_probe_paths_are_the_only_exemption(self) -> None:
        """Widening this set silently is how an audit trail develops holes."""
        assert sorted(AccessLogMiddleware.SKIP) == ["/health", "/readiness"]


class TestDatabaseRateLimiter:
    """The backend that makes one limit survive more than one worker."""

    @pytest.fixture
    def sessions(self, tmp_path: Path) -> sessionmaker[Session]:
        engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'rate_limit.db'}", future=True)
        RateLimitWindow.__table__.create(engine)
        return sessionmaker(bind=engine, expire_on_commit=False, future=True)

    def test_two_workers_share_one_counter(self, sessions: sessionmaker[Session]) -> None:
        """The whole point. Two instances stand in for two uvicorn workers.

        With the in-memory backend all ten of these pass, because each worker
        keeps its own bucket - the bug this backend removes.
        """
        worker_a = DatabaseRateLimiter(limit_per_minute=5, session_factory=sessions)
        worker_b = DatabaseRateLimiter(limit_per_minute=5, session_factory=sessions)

        allowed = sum(
            (worker_a if index % 2 else worker_b).check("console", now=1000.0)[0]
            for index in range(10)
        )

        assert allowed == 5

    def test_the_in_memory_backend_does_not_share(self) -> None:
        """The contrast that makes the test above mean something.

        This is not a defect being asserted into permanence - it is why
        `RTO_RATE_LIMIT_BACKEND=database` exists and why the docs tell a
        multi-worker deployment to set it.
        """
        worker_a = RateLimiter(limit_per_minute=5)
        worker_b = RateLimiter(limit_per_minute=5)

        allowed = sum(
            (worker_a if index % 2 else worker_b).check("console", now=1000.0)[0]
            for index in range(10)
        )

        assert allowed == 10

    def test_callers_still_have_separate_allowances(self, sessions: sessionmaker[Session]) -> None:
        limiter = DatabaseRateLimiter(limit_per_minute=2, session_factory=sessions)
        limiter.check("console", now=1000.0)
        limiter.check("console", now=1000.0)

        assert limiter.check("console", now=1000.0)[0] is False
        assert limiter.check("ops", now=1000.0)[0] is True

    def test_a_refusal_reports_when_to_retry(self, sessions: sessionmaker[Session]) -> None:
        limiter = DatabaseRateLimiter(limit_per_minute=1, session_factory=sessions)
        limiter.check("console", now=1000.0)

        allowed, remaining, retry_after = limiter.check("console", now=1000.0)

        assert allowed is False
        assert remaining == 0
        assert 0 < retry_after <= 60

    def test_the_previous_window_is_weighted_rather_than_forgotten(
        self, sessions: sessionmaker[Session]
    ) -> None:
        """A plain fixed window allows a double burst across the boundary.

        Spending the allowance at 12:00:59 and again at 12:01:00 is twice the
        intended rate, at exactly the moment a burst is most likely.
        """
        limiter = DatabaseRateLimiter(limit_per_minute=10, session_factory=sessions)
        for _ in range(10):
            limiter.check("console", now=1259.0)

        # One second into the next window: the previous one is still ~98% in view.
        assert limiter.check("console", now=1261.0)[0] is False

    def test_an_old_burst_stops_counting_once_it_has_scrolled_away(
        self, sessions: sessionmaker[Session]
    ) -> None:
        """The other half of that: the window has to slide, not just block."""
        limiter = DatabaseRateLimiter(limit_per_minute=10, session_factory=sessions)
        for _ in range(10):
            limiter.check("console", now=1259.0)

        assert limiter.check("console", now=1315.0)[0] is True

    def test_expired_windows_are_swept(self, sessions: sessionmaker[Session]) -> None:
        """Anything older than two windows can never affect an estimate again.

        Sweeping opportunistically because a cron job for three rows is
        infrastructure nobody should have to own.
        """
        limiter = DatabaseRateLimiter(limit_per_minute=10, session_factory=sessions)
        limiter.check("console", now=1000.0)
        limiter.check("console", now=5000.0)

        with sessions() as session:
            rows = session.execute(select(func.count()).select_from(RateLimitWindow)).scalar_one()

        assert rows == 1

    def test_a_zero_limit_never_touches_the_database(self, sessions: sessionmaker[Session]) -> None:
        limiter = DatabaseRateLimiter(limit_per_minute=0, session_factory=sessions)
        for _ in range(50):
            assert limiter.check("console")[0] is True

        with sessions() as session:
            rows = session.execute(select(func.count()).select_from(RateLimitWindow)).scalar_one()

        assert rows == 0

    def test_the_backend_setting_selects_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A setting that configures nothing is worse than no setting."""
        monkeypatch.setenv("RTO_RATE_LIMIT_BACKEND", "database")
        get_settings.cache_clear()
        reset_rate_limiter()

        assert isinstance(get_rate_limiter(), DatabaseRateLimiter)

    def test_memory_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The single-worker case should pay nothing for this."""
        monkeypatch.delenv("RTO_RATE_LIMIT_BACKEND", raising=False)
        get_settings.cache_clear()
        reset_rate_limiter()

        assert isinstance(get_rate_limiter(), RateLimiter)
