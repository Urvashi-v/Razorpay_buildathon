"""Authentication and rate limiting for the scoring API.

WHAT THIS PROTECTS
==================
Every `/v1/*` route. `/health` and `/readiness` stay open, because a liveness
probe that needs a credential is a probe that fails during exactly the incident
it exists to report - and neither reveals an order, a score or a secret.

WHY API KEYS
============
There is no identity provider in this system for OAuth to talk to, and no
certificate infrastructure for mTLS. Building either would mean shipping a login
screen with nothing behind it. API keys are what this shape of service - a
merchant console plus server-to-server callers - actually uses.

THE BROWSER IS NOT A PLACE TO PUT A KEY
=======================================
A key compiled into a frontend bundle is readable by anyone who opens dev tools.
Shipping one there would be theatre: it would look like authentication and
protect nothing.

So the console does not hold a key. In development the Vite proxy injects it
server-side (it is a Node process, not the browser); in production the console is
served behind a reverse proxy that does the same. The key never reaches the
client, which is the only arrangement where this means anything. See
`docs/deployment.md`.

CONSTANT-TIME COMPARISON
========================
`secrets.compare_digest`, not `==`. A naive comparison returns as soon as two
bytes differ, so response time leaks how many leading characters were right and a
key can be recovered one character at a time over enough requests. The window is
small over a network and it is not zero, and the fix costs one import.

SCOPES
======
A key carries `read` or `write`, and `read` is the default when the scope is
omitted. Least privilege by default: granting a key the power to append to the
audit log is a deliberate act of typing `:write`, not something that happens to
every key ever issued. `write` guards exactly the two routes that change stored
state - appending a decision and appending an override.

RATE LIMITING
=============
Two backends, chosen by `RTO_RATE_LIMIT_BACKEND`:

* `memory` (default) holds per-key buckets in process memory. Correct for ONE
  uvicorn worker and **wrong across several** - each worker keeps its own, so N
  workers permit N times the configured rate.
* `database` counts in PostgreSQL, so every worker shares one counter. One round
  trip per request, which is noise beside a scoring endpoint that takes seconds.

A multi-worker deployment must set `database`. The default stays `memory` so the
single-worker case pays nothing for a problem it does not have.

THE REQUEST AUDIT LOG
=====================
`AccessLogMiddleware` writes one line per request: caller, scope, method, path,
query, status, duration. Never the key. "Which key read which order" is the
first question asked after a credential leaks, and reads are the entire surface
a leaked read-only key exposes.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Any

from fastapi import Depends, Header, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from rto_sentinel.api.errors import ApiError, ErrorCode
from rto_sentinel.settings import ApiKey, Settings, get_settings

#: The header a caller presents. `Authorization: Bearer <key>` is also accepted.
API_KEY_HEADER = "X-API-Key"

#: Identity used for rate limiting when authentication is not configured.
ANONYMOUS = "anonymous"


@dataclass(frozen=True, slots=True)
class Caller:
    """Who is making this request, and what they may do.

    ``name`` is the label from `RTO_API_KEYS`, never the secret. It reaches log
    lines and rate-limit accounting, so a leaked key can be attributed and
    revoked without guessing who was using it.
    """

    name: str
    authenticated: bool
    scope: str = "write"

    @property
    def may_write(self) -> bool:
        """Whether this caller may append a decision or an override.

        An UNAUTHENTICATED caller - only possible in development, where no keys
        are configured - defaults to `write`. Restricting an open instance would
        be security theatre that only broke the local console, and `Settings`
        already refuses to run open in a deployed environment.
        """
        return self.scope == "write"


def _present_key(header_key: str | None, authorization: str | None) -> str | None:
    """The key the caller presented, from either accepted header."""
    if header_key and header_key.strip():
        return header_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
        return candidate or None
    return None


def _match(presented: str, configured: dict[str, ApiKey]) -> ApiKey | None:
    """The credential whose secret matches, in constant time.

    Every configured key is compared even after a match is found. Short-circuiting
    would make the response time depend on the caller's position in the
    dictionary, which leaks how many keys exist and roughly where a guess landed.
    """
    matched: ApiKey | None = None
    for key in configured.values():
        if secrets.compare_digest(presented, key.secret):
            matched = key
    return matched


def authenticate(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Caller:
    """Resolve the caller, or refuse the request.

    When no keys are configured the API is open and every caller is `anonymous`.
    That is allowed only in development - `Settings` refuses to start otherwise -
    and `/readiness` reports `authentication: disabled` so an open instance is
    never a surprise.
    """
    settings: Settings = get_settings()
    configured = settings.api_keys

    if not configured:
        caller = Caller(name=ANONYMOUS, authenticated=False)
        request.state.caller = caller
        return caller

    presented = _present_key(x_api_key, authorization)
    if presented is None:
        raise ApiError(
            ErrorCode.UNAUTHENTICATED,
            f"This endpoint requires an API key. Send it as the {API_KEY_HEADER} header "
            "or as `Authorization: Bearer <key>`.",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    key = _match(presented, configured)
    if key is None:
        # Deliberately identical to the missing-key message in everything but the
        # first clause: an error that distinguishes "no key" from "wrong key"
        # tells an attacker their format was right.
        raise ApiError(
            ErrorCode.UNAUTHENTICATED,
            "The API key presented is not recognised. Check the key, or ask whoever "
            "issued it whether it has been revoked.",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    caller = Caller(name=key.name, authenticated=True, scope=key.scope)
    request.state.caller = caller
    return caller


CallerDep = Annotated[Caller, Depends(authenticate)]


@dataclass
class _Bucket:
    """Request timestamps inside the current window, oldest first."""

    hits: deque[float] = field(default_factory=deque)


class RateLimiter:
    """A sliding-window-per-caller limiter, held in process memory.

    Sliding rather than a fixed calendar minute: a fixed window lets a caller
    spend its whole allowance in the last second of one minute and again in the
    first second of the next, which is twice the intended rate at exactly the
    moment a burst is most likely.

    **Per process.** N uvicorn workers permit N times the limit, because each
    keeps its own buckets. Use `DatabaseRateLimiter` for anything running more
    than one worker.
    """

    def __init__(self, *, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._window = 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    @property
    def limit_per_minute(self) -> int:
        return self._limit

    def check(self, caller: str, *, now: float | None = None) -> tuple[bool, int, float]:
        """Record one request. Returns (allowed, remaining, seconds_until_reset)."""
        if self._limit <= 0:
            return (True, 0, 0.0)

        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.setdefault(caller, _Bucket())
            cutoff = moment - self._window
            while bucket.hits and bucket.hits[0] <= cutoff:
                bucket.hits.popleft()

            if len(bucket.hits) >= self._limit:
                retry_after = bucket.hits[0] + self._window - moment
                return (False, 0, max(retry_after, 0.0))

            bucket.hits.append(moment)
            return (True, self._limit - len(bucket.hits), 0.0)

    def reset(self) -> None:
        """Drop every bucket. For tests, and for a deliberate operational flush."""
        with self._lock:
            self._buckets.clear()


class DatabaseRateLimiter:
    """The same limit, counted in PostgreSQL so every worker shares it.

    WHY THIS EXISTS
    ---------------
    `RateLimiter` holds its buckets in process memory, so a deployment running
    four uvicorn workers permits four times the configured rate. That was a
    documented limitation; this removes it.

    Postgres rather than Redis because Postgres is already deployed. A second
    datastore to provision, monitor and back up for the sake of one integer is a
    poor trade.

    THE SLIDING WINDOW, APPROXIMATED
    --------------------------------
    Rows are fixed windows, which alone would allow a caller to spend its whole
    allowance at 12:00:59 and again at 12:01:00. So the count is the current
    window plus the previous one weighted by how much of it is still in view:

        estimate = current + previous * (1 - elapsed_fraction)

    That is the standard sliding-window-counter approximation and it keeps the
    same semantics as the in-memory limiter, at one round trip per request.
    """

    def __init__(self, *, limit_per_minute: int, session_factory: Any) -> None:
        self._limit = limit_per_minute
        self._window = 60
        self._sessions = session_factory

    @property
    def limit_per_minute(self) -> int:
        return self._limit

    def check(self, caller: str, *, now: float | None = None) -> tuple[bool, int, float]:
        if self._limit <= 0:
            return (True, 0, 0.0)

        from sqlalchemy import delete, select
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        from rto_sentinel.db.models import RateLimitWindow

        moment = time.time() if now is None else now
        window_start = int(moment // self._window) * self._window
        previous_start = window_start - self._window
        elapsed = (moment - window_start) / self._window

        with self._sessions() as session:
            dialect = session.get_bind().dialect.name
            insert = pg_insert if dialect == "postgresql" else sqlite_insert
            statement = (
                insert(RateLimitWindow)
                .values(caller=caller, window_start=window_start, hits=1)
                .on_conflict_do_update(
                    index_elements=["caller", "window_start"],
                    set_={"hits": RateLimitWindow.__table__.c.hits + 1},
                )
                .returning(RateLimitWindow.__table__.c.hits)
            )
            current = int(session.execute(statement).scalar_one())

            previous = int(
                session.execute(
                    select(RateLimitWindow.__table__.c.hits).where(
                        RateLimitWindow.__table__.c.caller == caller,
                        RateLimitWindow.__table__.c.window_start == previous_start,
                    )
                ).scalar_one_or_none()
                or 0
            )

            # Sweep opportunistically rather than on a schedule: anything older
            # than two windows can never contribute to an estimate again, and a
            # cron job for three rows is infrastructure nobody should own.
            session.execute(
                delete(RateLimitWindow).where(
                    RateLimitWindow.__table__.c.window_start < previous_start
                )
            )
            session.commit()

        estimate = current + previous * (1.0 - elapsed)
        if estimate > self._limit:
            return (False, 0, max((1.0 - elapsed) * self._window, 0.0))
        return (True, max(self._limit - int(estimate), 0), 0.0)


#: One limiter per process. Module-level because in-memory buckets must outlive a
#: request; the database backend is stateless and shares this slot for symmetry.
_LIMITER: RateLimiter | DatabaseRateLimiter | None = None


def get_rate_limiter() -> RateLimiter | DatabaseRateLimiter:
    global _LIMITER
    if _LIMITER is None:
        settings = get_settings()
        if settings.rate_limit_backend == "database":
            from rto_sentinel.db.session import get_session_factory

            _LIMITER = DatabaseRateLimiter(
                limit_per_minute=settings.rate_limit_per_minute,
                session_factory=get_session_factory(),
            )
        else:
            _LIMITER = RateLimiter(limit_per_minute=settings.rate_limit_per_minute)
    return _LIMITER


def reset_rate_limiter() -> None:
    """Rebuild the limiter from current settings. Used by tests."""
    global _LIMITER
    _LIMITER = None


def enforce_rate_limit(request: Request, caller: CallerDep) -> Caller:
    """Refuse a caller that has exceeded its allowance.

    Depends on `authenticate`, so limiting is per key rather than per IP. Per-IP
    would punish every merchant behind one corporate NAT for the behaviour of
    one, and would not limit a single key spread across many addresses.
    """
    limiter = get_rate_limiter()
    allowed, remaining, retry_after = limiter.check(caller.name)

    request.state.rate_limit_remaining = remaining
    if not allowed:
        raise ApiError(
            ErrorCode.RATE_LIMITED,
            f"Rate limit exceeded: {limiter.limit_per_minute} requests per minute for "
            f"this key. Retry in {retry_after:.0f}s.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"retry_after_seconds": round(retry_after, 1)},
        )
    return caller


#: The dependency every `/v1` router carries. Authenticates, then rate limits.
ProtectedDep = Annotated[Caller, Depends(enforce_rate_limit)]


def require_write(caller: ProtectedDep) -> Caller:
    """Refuse a read-only key on a mutating endpoint.

    Guards exactly two routes - appending a decision and appending an override -
    because those are the only two things in this API that change stored state.
    Both write to an append-only audit log, which is precisely the kind of thing
    a compromised read key should not be able to touch.

    403 rather than 401: the caller is who they say they are, and the credential
    simply does not carry this power. A 401 would send them off to re-check a key
    that is working correctly.
    """
    if not caller.may_write:
        raise ApiError(
            ErrorCode.FORBIDDEN,
            f"This endpoint requires a key with the 'write' scope. The key '{caller.name}' "
            "has 'read'. Scopes are set in RTO_API_KEYS as 'name:secret:write'.",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return caller


WriteDep = Annotated[Caller, Depends(require_write)]


# ---------------------------------------------------------------------------
# request audit
# ---------------------------------------------------------------------------

#: One logger, so a deployment can route access records separately from
#: application logs - they have different retention needs and different readers.
ACCESS_LOG = logging.getLogger("rto_sentinel.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One line per request: who, what, the outcome, and how long it took.

    WHY THIS EXISTS
    ---------------
    A risk system holds every order a merchant has. "Which key read which order"
    is the first question asked after a credential leaks, and without this the
    answer is "we cannot tell". Decisions and overrides were already logged;
    *reads* were not, and reads are the whole surface a leaked read-only key
    exposes.

    WHAT IS NEVER LOGGED
    --------------------
    The key. Query strings are recorded because a filtered order listing is the
    thing worth auditing, but the `X-API-Key` header and the `Authorization`
    header are not touched - a secret in a log file is a secret in every backup,
    every log shipper and every screen it is ever displayed on.

    Health probes are skipped. They are unauthenticated, carry no data, and at
    one per second would bury every line that matters.
    """

    #: Paths that produce no audit line. Probes only.
    SKIP = frozenset({"/health", "/readiness"})

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.url.path in self.SKIP:
            return await call_next(request)

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000.0

        # `request.state.caller` is set by `authenticate`. It is absent when the
        # request never reached a route - a 404 or a validation failure - and
        # those are worth recording precisely because they are what probing
        # looks like.
        caller = getattr(request.state, "caller", None)
        name = caller.name if caller is not None else "unresolved"
        scope = caller.scope if caller is not None else "-"

        ACCESS_LOG.info(
            "caller=%s scope=%s method=%s path=%s query=%s status=%s duration_ms=%.1f",
            name,
            scope,
            request.method,
            request.url.path,
            request.url.query or "-",
            response.status_code,
            duration_ms,
        )
        return response
