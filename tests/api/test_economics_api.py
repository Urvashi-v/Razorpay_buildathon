"""The economics endpoints recalculate on the server, and refuse the sealed set.

The claim these tests defend is the one the specification is most sceptical of:
that dragging a slider produces a real recomputation rather than a number scaled
in a browser. So the tests change one input, assert that *several* independent
downstream quantities move in the directions the formula implies, and check that
the band boundaries themselves moved - a scaling trick would leave those fixed.

The scored book comes from the Phase 5 artefacts. When none exists the endpoints
return 503 naming the command to run, and the tests assert that too: an economics
endpoint answering from nothing would produce rupee figures for a model that does
not exist.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rto_sentinel.api.deps import scored_book_dep
from rto_sentinel.api.main import create_app
from rto_sentinel.models.final import ScoredBook

SPEC_BODY = {
    "rto_cost_inr": 220.0,
    "contribution_margin_inr": 250.0,
    "abandonment_on_friction": 0.25,
    "intervention_success_rate": 0.60,
    "friction_support_cost_inr": 0.0,
}


@pytest.fixture
def book() -> ScoredBook:
    """A deterministic stand-in for the Phase 5 scored book.

    Injected rather than read from disk so the API tests do not depend on a
    training run having happened, and so the arithmetic under test is over a book
    whose contents are known.
    """
    import numpy as np

    rng = np.random.default_rng(4242)
    scores = rng.uniform(0.0, 1.0, size=1500)
    labels = rng.uniform(size=1500) < scores
    return ScoredBook(
        probabilities=scores,
        labels=labels,
        split="validation",
        dataset_run_id="testrun",
        model_version="vtest",
        n_orders=1500,
    )


@pytest.fixture
def client(book: ScoredBook) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[scored_book_dep] = lambda: book
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# the derivation, exposed with its working
# ---------------------------------------------------------------------------


def test_the_threshold_endpoint_reproduces_the_specification(client: TestClient) -> None:
    response = client.post("/v1/economics/threshold", json=SPEC_BODY)
    body = response.json()

    assert response.status_code == 200
    assert body["cost_false_positive_inr"] == pytest.approx(62.5)
    assert body["saving_true_positive_inr"] == pytest.approx(132.0)
    assert body["threshold"] == pytest.approx(0.3214, abs=1e-4)
    assert body["threshold"] != pytest.approx(0.5)


def test_the_derivation_shows_its_arithmetic(client: TestClient) -> None:
    """A threshold arriving without its derivation is a magic constant."""
    body = client.post("/v1/economics/threshold", json=SPEC_BODY).json()
    assert "C_fp" in body["formula"]
    assert body["inputs"]["contribution_margin_inr"] == 250.0


def test_profiles_list_their_bounds_and_the_assumption_warning(client: TestClient) -> None:
    body = client.get("/v1/economics/profiles").json()

    assert body["default_profile"] in {profile["key"] for profile in body["profiles"]}
    assert body["bounds"]["contribution_margin_inr"]["max"] > 0
    assert "ASSUMPTION" in body["assumption_warning"]


# ---------------------------------------------------------------------------
# the recomputation is real
# ---------------------------------------------------------------------------


def _simulate(client: TestClient, margin: float) -> dict:
    response = client.post(
        "/v1/economics/simulate",
        json={"cost_inputs": {**SPEC_BODY, "contribution_margin_inr": margin}},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_changing_the_margin_moves_everything_downstream(client: TestClient) -> None:
    """Margin 250 -> 400, the specification's own example.

    Four independent quantities must move, and the band boundaries with them. A
    frontend scaling a cached total would leave the boundaries untouched.
    """
    before = _simulate(client, 250.0)
    after = _simulate(client, 400.0)

    assert after["threshold"]["threshold"] > before["threshold"]["threshold"]
    assert after["economics"]["flag_rate"] < before["economics"]["flag_rate"]
    assert (
        after["economics"]["expected_orders_affected"]
        < before["economics"]["expected_orders_affected"]
    )
    assert (
        after["economics"]["expected_false_positive_cost_inr"]
        != before["economics"]["expected_false_positive_cost_inr"]
    )

    # The ladder itself was rebuilt, not rescaled.
    before_bounds = [rung["lower_bound"] for rung in before["ladder"]]
    after_bounds = [rung["lower_bound"] for rung in after["ladder"]]
    assert after_bounds[1] > before_bounds[1]


def test_the_simulation_returns_a_delta_against_a_named_profile(client: TestClient) -> None:
    response = client.post(
        "/v1/economics/simulate",
        json={
            "cost_inputs": {**SPEC_BODY, "contribution_margin_inr": 400.0},
            "compare_to_profile": "mid_margin_d2c",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["baseline_threshold"] is not None
    assert body["baseline_net_inr_per_1000_orders"] is not None


def test_an_unknown_profile_is_refused(client: TestClient) -> None:
    response = client.post(
        "/v1/economics/simulate",
        json={"cost_inputs": SPEC_BODY, "compare_to_profile": "no_such_profile"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


def test_what_if_returns_the_four_numbers_that_move_together(client: TestClient) -> None:
    body = client.post("/v1/economics/what-if", json={"cost_inputs": SPEC_BODY}).json()

    assert set(body) == {
        "threshold",
        "flag_rate",
        "total_false_positive_cost_inr",
        "net_inr_saved_per_1000_orders",
        "n_orders",
    }
    assert body["n_orders"] == 1500


def test_what_if_and_simulate_agree(client: TestClient) -> None:
    """One implementation of the arithmetic, so the two surfaces cannot diverge."""
    compact = client.post("/v1/economics/what-if", json={"cost_inputs": SPEC_BODY}).json()
    full = client.post("/v1/economics/simulate", json={"cost_inputs": SPEC_BODY}).json()

    assert compact["threshold"] == pytest.approx(full["threshold"]["threshold"])
    assert compact["flag_rate"] == pytest.approx(full["economics"]["flag_rate"])
    assert compact["net_inr_saved_per_1000_orders"] == pytest.approx(
        full["economics"]["expected_net_inr_per_1000_orders"]
    )


def test_the_ladder_carries_its_assumed_rates(client: TestClient) -> None:
    """A console rendering the ladder must be able to label the assumptions."""
    body = _simulate(client, 250.0)
    acting = [rung for rung in body["ladder"] if rung["action"] != "none"]

    assert acting
    for rung in acting:
        assert 0.0 <= rung["intervention_success_rate"] <= 1.0
        assert 0.0 <= rung["abandonment_rate"] <= 1.0


def test_the_response_carries_provenance_for_every_headline(client: TestClient) -> None:
    body = _simulate(client, 250.0)
    quantities = {q["name"]: q for q in body["economics"]["quantities"]}

    assert quantities["intervention_success_rate"]["provenance"] == "assumed_intervention"
    assert quantities["contribution_margin_inr"]["provenance"] == "merchant_input"
    assert quantities["net_inr_saved_per_1000_orders"]["provenance"] == "derived"


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_the_sealed_split_is_refused(client: TestClient) -> None:
    """A slider wired to the test set would consume it in the first minute."""
    response = client.post(
        "/v1/economics/simulate", json={"cost_inputs": SPEC_BODY, "split": "test"}
    )
    assert response.status_code == 400
    assert "sealed test split" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contribution_margin_inr", -50.0),
        ("rto_cost_inr", -1.0),
        ("abandonment_on_friction", 1.5),
        ("intervention_success_rate", -0.2),
    ],
)
def test_invalid_economics_are_rejected_not_clamped(
    client: TestClient, field: str, value: float
) -> None:
    response = client.post("/v1/economics/threshold", json={**SPEC_BODY, field: value})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


def test_degenerate_economics_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/economics/threshold",
        json={**SPEC_BODY, "abandonment_on_friction": 0.0, "intervention_success_rate": 0.0},
    )
    assert response.status_code == 422


def test_missing_economics_are_rejected(client: TestClient) -> None:
    response = client.post("/v1/economics/threshold", json={"rto_cost_inr": 220.0})
    assert response.status_code == 422


def test_the_endpoints_say_so_when_no_model_has_been_trained() -> None:
    """503 naming the command to run, never invented scores."""
    app = create_app()

    def missing() -> ScoredBook:
        from rto_sentinel.api.errors import ApiError, ErrorCode

        raise ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            "no final-model run under artifacts/final",
            status_code=503,
        )

    app.dependency_overrides[scored_book_dep] = missing
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/economics/simulate", json={"cost_inputs": SPEC_BODY})
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def test_the_sweep_carries_its_selection_methodology(client: TestClient) -> None:
    """The caveat is a required contract field, so it cannot be dropped in transit."""
    body = client.get("/v1/economics/sweep").json()

    assert "derived" in body["selection_methodology"].lower()
    assert body["derived_threshold"] != body.get("selection_methodology")
    assert len(body["points"]) > 50
    assert sum(point["is_derived_operating_point"] for point in body["points"]) == 1


def test_the_sweep_rejects_an_unknown_profile(client: TestClient) -> None:
    response = client.get("/v1/economics/sweep", params={"profile": "nope"})
    assert response.status_code == 400
