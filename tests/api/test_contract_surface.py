"""The API contract exists and is documented, and unimplemented routes say so.

Two things being asserted:

1. **The contract is published now.** Every endpoint the console will consume
   appears in the OpenAPI document during Phase 1, so the frontend can be built
   against a real schema rather than a guess.
2. **Unimplemented endpoints return 501, not a plausible response.** This is the
   test that keeps the project honest. It would be easy - and it would demo
   better - to return a nice-looking fake score from ``/v1/score``. That number
   would then flow into a chart, into a screenshot, and into a claim. A 501 with
   ``NOT_IMPLEMENTED`` makes the gap impossible to miss.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rto_sentinel.api.main import create_app

EXPECTED_PATHS = {
    "/health",
    "/readiness",
    "/v1/score",
    "/v1/score/batch",
    "/v1/orders",
    "/v1/orders/{order_id}",
    "/v1/orders/{order_id}/risk",
    "/v1/economics/profiles",
    "/v1/economics/threshold",
    "/v1/economics/simulate",
    "/v1/economics/what-if",
    "/v1/economics/sweep",
    "/v1/decisions",
    "/v1/decisions/queue",
    "/v1/decisions/{order_id}",
    "/v1/decisions/override",
    "/v1/evaluation/ladder",
    "/v1/evaluation/final",
    "/v1/evaluation/selection",
    "/v1/evaluation/reliability",
    "/v1/evaluation/fairness",
    "/v1/evaluation/shift",
    "/v1/evaluation/ablation",
    "/v1/monitoring/model",
    "/v1/monitoring/decisions",
    "/v1/monitoring/data",
    "/v1/monitoring/drift",
    "/v1/explanations/{order_id}",
    "/v1/explanations/{order_id}/investigate",
    "/v1/explanations/{order_id}/confirmation",
    "/v1/explanations/address-repair",
    "/v1/explanations/digest",
    "/v1/explanations/tools",
    "/v1/explanations/status",
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def test_openapi_publishes_the_whole_contract(client: TestClient) -> None:
    document = client.get("/openapi.json").json()
    assert set(document["paths"]) == EXPECTED_PATHS


def test_openapi_states_the_data_provenance(client: TestClient) -> None:
    """The synthetic-data caveat is in the API description, not only the README.

    Someone integrating against this service may never read the repository. The
    limitation belongs where they will actually encounter it.
    """
    raw = client.get("/openapi.json").json()["info"]["description"]
    # The description is a wrapped block; normalise whitespace before matching so
    # a reflow cannot silently drop the caveat from this assertion.
    description = " ".join(raw.lower().split())
    assert "synthetic" in description
    assert "not a claim about production performance" in description


@pytest.mark.parametrize(
    "path",
    [
        "/v1/evaluation/fairness",
        "/v1/evaluation/shift",
        "/v1/evaluation/ablation",
        "/v1/monitoring/drift",
    ],
)
def test_endpoints_with_no_artefact_return_an_explicit_501(client: TestClient, path: str) -> None:
    """No plausible-looking placeholder data anywhere in the API.

    These three are implemented, and each serves a saved artefact. This client
    points at an empty artefact root, so none of the experiments has been run -
    and the correct answer is 501 with the reason, not an empty table that reads
    as "we checked and found nothing".

    The fairness case is the one that matters most: a fairness report nobody
    computed, presented as evidence the model was audited, would be the most
    damaging fake this API could serve.
    """
    response = client.get(path)

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "NOT_IMPLEMENTED"
    # The message must say the experiment was not run AND name the command that
    # runs it. A 501 that leaves the caller guessing is only half an answer.
    assert "been run" in error["message"]
    assert "rto-sentinel" in error["message"]


def test_the_implemented_economics_endpoints_return_real_arithmetic(
    client: TestClient,
) -> None:
    """The threshold endpoint needs no model artefact: economics alone derive it."""
    response = client.post(
        "/v1/economics/threshold",
        json={
            "rto_cost_inr": 220.0,
            "contribution_margin_inr": 250.0,
            "abandonment_on_friction": 0.25,
            "intervention_success_rate": 0.60,
            "friction_support_cost_inr": 0.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == pytest.approx(0.3214, abs=1e-4)
    assert body["cost_false_positive_inr"] == pytest.approx(62.5)


def test_scoring_endpoint_does_not_fabricate_a_score(client: TestClient) -> None:
    """With no model artefact loaded, scoring refuses. It never invents a number.

    The Phase 1 version of this test asserted a 501 against an unimplemented
    endpoint. The endpoint is implemented now, and the invariant it was defending
    survives unchanged: no code path returns a probability the model did not
    produce.
    """
    response = client.post("/v1/score", json={"order_id": "ORD-00000001"})

    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "probability" not in {key.lower() for key in body["error"].get("detail") or {}}
