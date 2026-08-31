"""Every endpoint the console calls exists on the backend, with the right method.

WHY THIS TEST EXISTS
====================
The console's request layer and the API's route table are edited in different
languages, in different directories, by different tools. Nothing else in this
repository fails when they drift apart: `tsc` type-checks the console against
hand-written interfaces in `types/api.ts`, not against the server, and the
console's own tests stub `fetch`, so a request to a path the backend removed
returns whatever the stub was told to return.

The failure that produces is quiet and late. A renamed route ships green, and the
first person to discover it is a merchant looking at an error panel.

So this parses the URLs out of `console/src/api/endpoints.ts` and asserts each
one resolves against the real OpenAPI schema. It deliberately reads the
TypeScript as text rather than importing it: the point is to check what the
console will actually request at runtime, and a mock of the request layer cannot
tell us that.

WHAT THIS DOES NOT CHECK
========================
Response *shapes*. `types/api.ts` is hand-written and could still disagree with a
response model field for field. That is a real remaining gap and it is listed as
such in the Phase 11 report rather than implied away by this file's existence.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from rto_sentinel.api.main import create_app
from rto_sentinel.settings import REPO_ROOT

ENDPOINTS_FILE = REPO_ROOT / "console" / "src" / "api" / "endpoints.ts"

#: Matches the path literal in a `get<T>(...)` / `post<T>(...)` call.
#:
#: The console builds URLs with template literals, so a path arrives here looking
#: like `/v1/orders/${encodeURIComponent(orderId)}/risk${query({...})}`. The
#: interpolations are replaced with a placeholder segment below rather than being
#: parsed - what matters is the route shape, not the value.
_PATH = re.compile(r"""[`'"](/v1/[^`'"]*)[`'"]""")

#: `${...}` inside a path is either a parameter or an appended query string.
_INTERPOLATION = re.compile(r"\$\{[^}]*\}")


def console_paths() -> set[str]:
    """Route templates the console will request, normalised to OpenAPI form."""
    source = ENDPOINTS_FILE.read_text(encoding="utf-8")
    found: set[str] = set()

    for raw in _PATH.findall(source):
        # A trailing `${query({...})}` is a query string, not a path segment.
        path = re.sub(r"\$\{query\([^`]*$", "", raw)
        path = _INTERPOLATION.sub("{param}", path).rstrip("/")
        if path.startswith("/v1/"):
            found.add(path)
    return found


def openapi_paths(client: TestClient) -> dict[str, set[str]]:
    """Path template -> the HTTP methods the backend serves on it."""
    schema = client.get("/openapi.json").json()
    return {
        # Normalise `{order_id}` to `{param}` so a rename of the path variable
        # does not read as a missing route.
        re.sub(r"\{[^}]+\}", "{param}", path): {method.upper() for method in operations}
        for path, operations in schema["paths"].items()
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def test_the_endpoints_file_is_where_it_is_expected_to_be() -> None:
    """A guard on the guard.

    If the console's request layer moves, `console_paths()` silently returns an
    empty set and every assertion below passes vacuously.
    """
    assert ENDPOINTS_FILE.is_file(), f"console request layer not found at {ENDPOINTS_FILE}"
    assert console_paths(), "no /v1 paths parsed out of the console request layer"


def test_every_path_the_console_requests_exists_on_the_backend(client: TestClient) -> None:
    """No console request can 404 because the route was renamed or removed."""
    served = openapi_paths(client)
    missing = sorted(path for path in console_paths() if path not in served)

    assert not missing, (
        "the console requests paths the API does not serve. Either the route was "
        f"renamed and the console not updated, or vice versa: {missing}"
    )


def test_the_console_requests_only_read_and_simulate_methods(client: TestClient) -> None:
    """The console is a client of a read-mostly API.

    Every path it calls must be served over GET or POST. A console path that
    resolved only to DELETE or PUT would mean the request layer had been pointed
    at something that mutates, which is not what this console is for - it reads,
    and it simulates.
    """
    served = openapi_paths(client)
    for path in sorted(console_paths()):
        assert served[path] & {"GET", "POST"}, f"{path} is served, but not over GET or POST"


def test_the_backend_serves_no_destructive_methods_at_all(client: TestClient) -> None:
    """There is no DELETE or PUT anywhere in this API.

    Decisions are an append-only log and overrides are appended, never edited.
    That is a property of the whole surface, not of the routes the console
    happens to call, so it is asserted over the entire schema.
    """
    served = openapi_paths(client)
    destructive = {
        path: sorted(methods & {"DELETE", "PUT", "PATCH"})
        for path, methods in served.items()
        if methods & {"DELETE", "PUT", "PATCH"}
    }

    assert not destructive, f"the API exposes mutating methods: {destructive}"
