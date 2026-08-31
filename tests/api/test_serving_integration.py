"""The integration test: a database order, through the real chain, to a decision.

WHAT MAKES THIS A REAL INTEGRATION TEST
=======================================
Nothing is stubbed. A dataset is generated, loaded into a database, and a model
is trained, calibrated and frozen to a real artefact on disk. The API then scores
an order from that database through the real feature pipeline and the real
artefact, and the decision comes from the real engine.

The test that matters most is :func:`test_the_api_fails_when_the_model_is_missing`.
A serving path that silently substitutes a plausible number when its model is
unavailable is the single most dangerous failure this system could have: nothing
errors, a probability appears, and it flows into a rupee figure and a customer's
order. So the absence of the artefact is asserted to produce a 503 - and the
assertion is written to fail if a score comes back instead.

WHY THIS RUNS ON SQLITE
=======================
The schema is engine-independent and the suite must run with no server. The same
code path is exercised against PostgreSQL by `rto-sentinel seed-db` and by the
manual verification recorded in the phase report; what is under test here is the
composition, not the dialect.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from rto_sentinel.agents.provider import Completion, ToolCall
from rto_sentinel.api.deps import db_session_dep
from rto_sentinel.api.main import create_app
from rto_sentinel.configuration import load_app_config
from rto_sentinel.configuration.schemas import (
    CostModelConfig,
    FeaturesConfig,
    FinalModelConfig,
    GeneratorConfig,
    SplitsConfig,
)
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import ConfiguredOrderGenerator, GeneratorParams
from rto_sentinel.data.splits import assign_splits
from rto_sentinel.db.base import Base
from rto_sentinel.db.repositories import DatasetRepository
from rto_sentinel.features.dataset import build_modeling_dataset
from rto_sentinel.models.final import build_final_model, save_manifest
from rto_sentinel.serving.model_registry import ModelRegistry, ModelUnavailableError
from rto_sentinel.settings import get_settings

#: Small enough to train in seconds, large enough that customers accumulate
#: history and the geography aggregates are not entirely prior.
INTEGRATION_DATASET = GeneratorParams(
    seed=90210,
    generator_version="1.0.0",
    n_customers=1500,
    n_orders=4500,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


@pytest.fixture(scope="module")
def generated(generator_config: GeneratorConfig, splits_config: SplitsConfig):
    result = ConfiguredOrderGenerator().generate(generator_config, INTEGRATION_DATASET)
    result.orders[cols.SPLIT] = assign_splits(result.orders, splits_config).labels
    return result


@pytest.fixture(scope="module")
def artifact_root(
    generated,
    generator_config: GeneratorConfig,
    features_config: FeaturesConfig,
    splits_config: SplitsConfig,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Train, calibrate and freeze a real model into a temporary artefact store."""
    root = tmp_path_factory.mktemp("artifacts")
    dataset = build_modeling_dataset(
        generated,
        features_config=features_config,
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=generated.orders[cols.SPLIT],
    )
    final = build_final_model(
        dataset,
        final_config=final_config,
        cost_config=cost_config,
        seed=INTEGRATION_DATASET.seed,
        artifact_root=root,
        bootstrap_iterations=0,
    )
    save_manifest(final.manifest, root)
    assert final.artifact_path is not None
    return root


@pytest.fixture(scope="module")
def database(generated, tmp_path_factory: pytest.TempPathFactory) -> Iterator[sessionmaker]:
    """A real database holding the generated dataset."""
    path = tmp_path_factory.mktemp("db") / "integration.db"
    engine = create_engine(f"sqlite+pysqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    with factory() as session:
        DatasetRepository(session).load(generated)
        session.commit()

    yield factory
    engine.dispose()


@pytest.fixture
def client(
    database: sessionmaker, artifact_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(artifact_root))
    get_settings.cache_clear()

    app = create_app()

    def session_override() -> Iterator[Session]:
        with database() as session:
            yield session
            session.commit()

    app.dependency_overrides[db_session_dep] = session_override
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def scorable_order_id(client: TestClient) -> str:
    """A COD order from late in the horizon, so it has history behind it."""
    response = client.get("/v1/orders", params={"payment_method": "cod", "limit": 1, "offset": 0})
    assert response.status_code == 200, response.text
    orders = response.json()["orders"]
    assert orders, "the generated dataset must contain COD orders"
    return orders[0]["order_id"]


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------


def test_database_order_through_model_to_decision(
    client: TestClient, scorable_order_id: str
) -> None:
    """The whole chain, asserted link by link.

    Each block below corresponds to one arrow in the pipeline, so a failure names
    the stage that broke rather than reporting that "the response was wrong".
    """
    response = client.get(f"/v1/orders/{scorable_order_id}/risk")
    assert response.status_code == 200, response.text
    body = response.json()

    # -- the database link ------------------------------------------------
    assert body["order"]["order_id"] == scorable_order_id
    assert body["order"]["merchant_id"]
    assert body["order"]["ordered_at"]

    # -- the feature link -------------------------------------------------
    features = body["features"]
    assert features["n_features"] > 0
    assert features["feature_fingerprint"]
    assert features["context_rows"] > 1, (
        "the feature row must be built from real history, not from the order alone"
    )

    # -- the model link ---------------------------------------------------
    model = body["model"]
    assert model["model_version"]
    assert model["calibration_method"], "an uncalibrated model must not reach a decision"
    assert model["feature_fingerprint"] == features["feature_fingerprint"], (
        "the serving pipeline must match the one the model was trained on"
    )

    # -- the probability --------------------------------------------------
    assert 0.0 <= body["probability"] <= 1.0
    assert body["raw_score"] is not None

    # -- the decision link ------------------------------------------------
    assert body["band"] in {"LOW", "ELEVATED", "HIGH", "SEVERE"}
    assert 0.0 < body["threshold"] < 1.0
    assert body["threshold"] != pytest.approx(0.5), "the threshold is derived, never 0.5"
    assert body["appeal_available"] is True
    assert body["engine_version"]

    # -- the economics ----------------------------------------------------
    economics = body["economics"]
    assert economics["cost_false_positive_inr"] > 0
    assert economics["saving_true_positive_inr"] > 0
    assert "C_fp" in economics["threshold_formula"]


def test_the_api_fails_when_the_model_is_missing(
    database: sessionmaker, scorable_order_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503, never a fabricated score. The most important test in this file.

    Pointed at an empty artefact store, the endpoint must refuse. A serving path
    that substitutes a plausible probability when its model is unavailable fails
    silently, and the number flows into a rupee figure and a customer's order.
    """
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(tmp_path / "empty"))
    get_settings.cache_clear()

    app = create_app()

    def session_override() -> Iterator[Session]:
        with database() as session:
            yield session

    app.dependency_overrides[db_session_dep] = session_override
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/v1/orders/{scorable_order_id}/risk")
    app.dependency_overrides.clear()
    get_settings.cache_clear()

    assert response.status_code == 503, (
        f"expected a refusal, got {response.status_code} with body {response.text[:400]}. "
        "A missing model must never produce a score."
    )
    error = response.json()["error"]
    assert error["code"] == "MODEL_UNAVAILABLE"
    assert "rto-sentinel final" in error["message"], "the error should say how to fix it"
    # No score-shaped field anywhere in the body. The point of the refusal is
    # that nothing numeric comes back that a caller could mistake for a result.
    body = response.json()
    assert set(body) == {"error"}
    detail = error.get("detail") or {}
    assert not any(key in detail for key in ("probability", "band", "threshold", "score"))


def test_serving_features_match_the_offline_pipeline(
    database: sessionmaker,
    generated,
    generator_config: GeneratorConfig,
    features_config: FeaturesConfig,
    splits_config: SplitsConfig,
) -> None:
    """The row served must be the row training would have built. Identical.

    This is the claim the whole serving design rests on. A serving path that
    computes features slightly differently from the training path does not fail -
    the numbers just move, and the model quietly stops being the model that was
    evaluated.
    """
    import numpy as np

    from rto_sentinel.db.repositories import ServingRepository
    from rto_sentinel.serving.features import OrderFeatureService

    offline = build_modeling_dataset(
        generated,
        features_config=features_config,
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=generated.orders[cols.SPLIT],
    )
    # An order the offline dataset actually contains. The API listing returns the
    # newest orders, which are typically immature and therefore excluded from the
    # modelling frame - a fine order to score, but there is no offline row to
    # compare it against.
    order_ids = offline.order_ids.to_numpy().tolist()
    comparable_id = order_ids[len(order_ids) // 2]
    position = order_ids.index(comparable_id)
    expected = offline.features.iloc[position]

    with database() as session:
        repository = ServingRepository(session)
        order = repository.get_order(comparable_id)
        assert order is not None
        served = OrderFeatureService(
            repository,
            features_config=features_config,
            generator_config=generator_config,
        ).build(order)

    assert list(served.x.columns) == list(offline.features.columns), "column order must match"

    actual = served.x.iloc[0]
    mismatched = []
    for name in offline.features.columns:
        left, right = expected[name], actual[name]
        if isinstance(left, float) and isinstance(right, float):
            if not (np.isnan(left) and np.isnan(right)) and not np.isclose(
                left, right, equal_nan=True
            ):
                mismatched.append((name, left, right))
        elif left != right and not (pd_isna(left) and pd_isna(right)):
            mismatched.append((name, left, right))

    assert not mismatched, (
        "the serving feature row differs from the offline one. The model would be "
        f"scoring something it was not trained on: {mismatched[:5]}"
    )


def pd_isna(value: object) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - non-scalar
        return False


# ---------------------------------------------------------------------------
# the decision log and overrides, against the real database
# ---------------------------------------------------------------------------


def test_a_decision_is_logged_and_retrievable(client: TestClient, scorable_order_id: str) -> None:
    created = client.post("/v1/decisions", json={"order_id": scorable_order_id})
    assert created.status_code == 201, created.text
    decision = created.json()

    fetched = client.get(f"/v1/decisions/{scorable_order_id}")
    assert fetched.status_code == 200
    assert fetched.json()["band"] == decision["band"]
    assert fetched.json()["model_version"] == decision["model_version"]


def test_an_override_is_recorded_with_its_reason(
    client: TestClient, scorable_order_id: str
) -> None:
    """Reason, timestamp, operator and the decision it attaches to."""
    created = client.post("/v1/decisions", json={"order_id": scorable_order_id})
    assert created.status_code == 201
    original = created.json()["band"]
    target = "SEVERE" if original != "SEVERE" else "LOW"

    response = client.post(
        "/v1/decisions/override",
        json={
            "order_id": scorable_order_id,
            "override_band": target,
            "operator_id": "op-hashed-identity",
            "reason": "Customer confirmed the address by phone; escalating for manual review.",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["accepted"] is True
    assert body["original_band"] == original
    assert body["new_band"] == target
    assert body["direction"] in {"escalated", "relaxed"}
    assert body["logged_at"]
    assert "address by phone" in body["note"]
    assert body["original_decision_unchanged"] is True

    # The original decision row is untouched: an audit trail that can be edited
    # is not an audit trail.
    after = client.get(f"/v1/decisions/{scorable_order_id}").json()
    assert after["band"] == original


def test_an_override_without_a_prior_decision_is_refused(client: TestClient) -> None:
    response = client.post(
        "/v1/decisions/override",
        json={
            "order_id": "ORD-99999999",
            "override_band": "HIGH",
            "operator_id": "op-1",
            "reason": "a reason long enough to satisfy the minimum length rule",
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DECISION_NOT_FOUND"


def test_monitoring_reports_what_is_actually_loaded(client: TestClient) -> None:
    model = client.get("/v1/monitoring/model").json()
    assert model["available"] is True
    assert model["calibration_method"]

    data = client.get("/v1/monitoring/data").json()
    assert data["total_orders"] == INTEGRATION_DATASET.n_orders
    assert data["matured_orders"] + data["immature_orders"] == data["total_orders"]
    assert 0.0 <= (data["observed_rto_rate"] or 0.0) <= 1.0


# ---------------------------------------------------------------------------
# Phase 11: failure behaviour
# ---------------------------------------------------------------------------
#
# THE PROPERTY UNDER TEST IN THIS SECTION
# ======================================
# Every one of these asserts that a broken dependency produces an honest error
# rather than a plausible number. That is a harder property to hold than it
# sounds: each of these failures has an obvious "graceful degradation" that would
# make the system look better and be worse - a default probability, a cached
# score, a zero where a measurement should be. The assertions are written to fail
# if any of those appear.


def test_the_database_being_unavailable_produces_an_error_not_a_score(
    artifact_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead database must not yield a probability from anywhere else.

    The model is loaded and perfectly capable of scoring a feature vector. The
    danger is that the serving path treats "no history rows" as "a new customer"
    and returns a confident cold-start score for an order it never actually read.
    """
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(artifact_root))
    get_settings.cache_clear()

    app = create_app()

    def broken_session() -> Iterator[Session]:
        msg = "connection to server at 127.0.0.1 port 5442 failed"
        raise OperationalError("SELECT 1", {}, Exception(msg))
        yield  # pragma: no cover - unreachable, present for the generator contract

    app.dependency_overrides[db_session_dep] = broken_session
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/v1/orders/ORD-00000001/risk")

    assert response.status_code >= 500
    body = response.json()
    assert "probability" not in body
    assert body["error"]["code"] in {"INTERNAL_ERROR", "DATABASE_UNAVAILABLE"}
    get_settings.cache_clear()


def test_a_database_failure_does_not_leak_the_connection_string(
    artifact_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error a caller sees must not contain credentials or SQL."""
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(artifact_root))
    get_settings.cache_clear()

    app = create_app()

    def broken_session() -> Iterator[Session]:
        raise OperationalError(
            "SELECT * FROM orders WHERE secret_col = 'x'",
            {},
            Exception("password authentication failed for user 'rto' (hunter2)"),
        )
        yield  # pragma: no cover

    app.dependency_overrides[db_session_dep] = broken_session
    with TestClient(app, raise_server_exceptions=False) as client:
        raw = client.get("/v1/orders/ORD-00000001/risk").text

    for leak in ("hunter2", "password authentication", "SELECT * FROM", "Traceback"):
        assert leak not in raw, f"the error response leaked {leak!r}"
    get_settings.cache_clear()


def test_a_missing_order_is_a_404_with_no_probability(client: TestClient) -> None:
    """Not found means not found. It does not mean scored-as-average."""
    response = client.get("/v1/orders/ORD-DOES-NOT-EXIST/risk")

    assert response.status_code == 404
    body = response.json()
    assert "probability" not in body
    assert body["error"]["code"] in {"ORDER_NOT_FOUND", "NOT_FOUND"}


@pytest.mark.parametrize(
    "order_id",
    [
        "'; DROP TABLE orders; --",
        "../../../etc/passwd",
        "<script>alert(1)</script>",
        "\x00null",
        "a" * 200,
    ],
)
def test_hostile_order_ids_are_rejected_by_validation(client: TestClient, order_id: str) -> None:
    """Rejected at the edge, and never reaching a query or the filesystem.

    Every query in this system is built with SQLAlchemy's expression language and
    bound parameters, so none of these could inject. They are rejected anyway:
    an identifier that cannot exist should not consume a database round trip.
    """
    response = client.get(f"/v1/orders/{quote(order_id, safe='')}/risk")

    assert response.status_code in {404, 422}
    assert "probability" not in response.json()


def test_an_unknown_split_is_rejected_rather_than_answered_with_zero(
    client: TestClient,
) -> None:
    """ "No orders in that split" and "that split does not exist" are different.

    Returning an empty page for a mistyped split tells a merchant their book is
    empty. `payment_method` was always constrained this way; `split` was not.
    """
    response = client.get("/v1/orders", params={"split": "trian"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contribution_margin_inr", -50.0),
        ("rto_cost_inr", -1.0),
        ("intervention_success_rate", 1.5),
        ("intervention_success_rate", -0.1),
        ("abandonment_on_friction", 2.0),
        ("friction_support_cost_inr", -10.0),
    ],
)
def test_invalid_merchant_economics_are_refused(
    client: TestClient, field: str, value: float
) -> None:
    """Impossible economics must not silently produce a threshold.

    A negative RTO cost or a success rate above 1 yields a threshold that is
    arithmetically defined and economically meaningless. Every rupee figure
    downstream would inherit it while looking entirely normal.
    """
    payload = {
        "contribution_margin_inr": 250.0,
        "rto_cost_inr": 220.0,
        "intervention_success_rate": 0.6,
        "abandonment_on_friction": 0.25,
        "friction_support_cost_inr": 8.0,
    }
    payload[field] = value

    # `/threshold` rather than `/simulate`: the derivation is a function of the
    # economics alone and needs no scored book, so this isolates the validation
    # from artefact availability.
    response = client.post("/v1/economics/threshold", json=payload)

    assert response.status_code == 422, response.text
    assert "threshold" not in response.json()


def test_zero_margin_economics_are_handled_without_dividing_by_zero(
    client: TestClient,
) -> None:
    """A zero contribution margin is legal, unusual, and must not crash.

    With no margin to lose, a false positive costs only the support call - so the
    threshold drops and the system flags more. That is the correct answer, not an
    error, and it is asserted here so nobody "fixes" it into one.
    """
    response = client.post(
        "/v1/economics/threshold",
        json={
            "contribution_margin_inr": 0.0,
            "rto_cost_inr": 220.0,
            "intervention_success_rate": 0.6,
            "abandonment_on_friction": 0.25,
            "friction_support_cost_inr": 8.0,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["threshold"] <= 1.0
    # C_fp is only the support cost now, so the threshold collapses towards zero
    # and the system flags far more than at the default profile's 0.348.
    assert body["threshold"] < 0.35


def test_an_order_with_no_customer_history_still_scores_and_says_so(
    client: TestClient,
) -> None:
    """Missing history is a fact about the customer, not a failure.

    A first-time customer genuinely has no prior RTO rate. The model handles
    missingness natively, so the order scores - but the response reports how many
    features were null and how many history rows backed the aggregates, because a
    score built largely from nulls deserves less confidence than one that is not.
    """
    earliest = client.get(
        "/v1/orders", params={"limit": 200, "offset": 0, "split": "train"}
    ).json()["orders"]
    assert earliest, "the generated dataset must contain train-split orders"

    # The oldest order in the book has the least history behind it.
    oldest = min(earliest, key=lambda order: order["ordered_at"])
    response = client.get(f"/v1/orders/{oldest['order_id']}/risk")

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert "null_features" in body["features"]
    assert body["features"]["context_rows"] >= 0


def test_the_agent_refuses_rather_than_answering_when_the_llm_is_unavailable(
    client: TestClient, scorable_order_id: str
) -> None:
    """No key means no answer. It does not mean a scripted explanation.

    This is the failure with the largest blast radius in the whole system: a
    canned sentence here would be indistinguishable from a model-generated one to
    every consumer, and would carry the authority of an "AI explanation" with
    nothing behind it.
    """
    response = client.post(
        f"/v1/explanations/{scorable_order_id}/investigate",
        params={"question": "Why did this order receive its risk level?"},
    )

    assert response.status_code in {501, 503}
    body = response.json()
    assert "investigation" not in body
    assert "summary" not in body
    message = body["error"]["message"]
    assert "ANTHROPIC_API_KEY" in message or "RTO_AGENTS_ENABLED" in message


def test_the_agent_status_endpoint_states_why_it_is_unavailable(
    client: TestClient,
) -> None:
    """An operator must be able to find out what to configure, from the API."""
    body = client.get("/v1/explanations/status").json()

    assert body["available"] is False
    assert body["reason"]
    assert body["required_environment_variable"] == "ANTHROPIC_API_KEY"


# ---------------------------------------------------------------------------
# calibration, asserted in the live chain
# ---------------------------------------------------------------------------


def test_the_served_probability_is_calibrated_not_the_raw_score(
    client: TestClient, scorable_order_id: str
) -> None:
    """The decision is made on the calibrated number, and both are reported.

    The raw score travels with the response for debugging, but it must never be
    the one compared against the threshold: an uncalibrated boosting output is
    not a probability, and the entire expected-value argument rests on it being
    one. Asserting the calibration method is present is what stops a future
    change quietly serving `predict_raw`.
    """
    body = client.get(f"/v1/orders/{scorable_order_id}/risk").json()

    assert body["model"]["calibration_method"] is not None
    assert body["model"]["calibration_fitted_on"] == "validation"
    assert body["raw_score"] is not None
    assert 0.0 <= body["probability"] <= 1.0
    # The band follows the calibrated probability against the threshold.
    flagged = body["probability"] >= body["threshold"]
    assert (body["band"] != "LOW") == flagged


def test_an_uncalibrated_artefact_is_refused_at_load(tmp_path: Path) -> None:
    """The registry will not serve a model whose card says it was never calibrated.

    This is the structural guard behind the test above: even if a caller wanted
    raw scores, there is no path that produces them from a servable artefact.
    """
    from tests.unit.test_model_registry import write_artefact

    write_artefact(tmp_path, "uncalibrated_rung", calibration=None)
    registry = ModelRegistry(tmp_path)

    with pytest.raises(ModelUnavailableError, match="uncalibrated"):
        registry.resolve()


def test_scoring_the_same_order_twice_returns_the_same_probability(
    client: TestClient, scorable_order_id: str
) -> None:
    """Determinism. A risk score that moves between identical requests is not auditable."""
    first = client.get(f"/v1/orders/{scorable_order_id}/risk").json()
    second = client.get(f"/v1/orders/{scorable_order_id}/risk").json()

    assert first["probability"] == second["probability"]
    assert first["raw_score"] == second["raw_score"]
    assert first["band"] == second["band"]
    assert first["reason_codes"] == second["reason_codes"]


def test_evaluation_retrieval_404s_rather_than_recomputing(client: TestClient) -> None:
    """Metrics come from the artefact a run wrote, or they do not come at all.

    This fixture trains and freezes a model but never runs an evaluation, so no
    metrics artefact exists. The endpoint must say so. The tempting alternative -
    scoring the validation split on demand and returning the result - would look
    identical to a caller and would report numbers that no frozen artefact backs,
    computed at request time against whatever data happened to be present.

    Retrieval *with* artefacts present is covered in `test_responsible_api.py`,
    which writes them first.
    """
    response = client.get("/v1/evaluation/final", params={"split": "validation"})

    assert response.status_code == 404
    body = response.json()
    assert "ranking" not in body
    assert "economics" not in body


def _answer_turn(**fields: Any) -> Completion:
    """A completion carrying the agent's JSON answer."""
    payload = {
        "sufficient_evidence": True,
        "summary": "",
        "key_drivers": [],
        "evidence_used": [],
        "uncertainty": "",
        "caveats": [],
        **fields,
    }
    return Completion(text=json.dumps(payload), stop_reason="end_turn")


def _tool_turn(*names: str, order_id: str = "") -> Completion:
    """A completion asking for tools by name.

    The order id is threaded through rather than hardcoded, because these tools
    hit a real database - a fixed id would look like a passing test while every
    tool returned "not found".
    """
    arguments = {"order_id": order_id} if order_id else {}
    return Completion(
        text="",
        tool_calls=tuple(
            ToolCall(id=f"call_{index}", name=name, arguments=dict(arguments))
            for index, name in enumerate(names)
        ),
        stop_reason="tool_use",
        raw_content=[
            {"type": "tool_use", "id": f"call_{index}", "name": name, "input": dict(arguments)}
            for index, name in enumerate(names)
        ],
    )


# ---------------------------------------------------------------------------
# Phase 11: the agent chain against real application data
# ---------------------------------------------------------------------------
#
# WHAT IS REAL HERE AND WHAT IS NOT
# =================================
# Real: the database, the feature pipeline, the trained and calibrated artefact,
# the decision engine, the six application tools, the tool-dispatch loop, the
# grounding validators and the audit trail.
#
# Stubbed: the HTTP call to Anthropic, and only that. `ScriptedProvider` replays
# prepared assistant turns in place of the network.
#
# That boundary is deliberate and it is where the honest line falls. No API key
# is configured, and the project's rule is that the system refuses rather than
# inventing a model response - which `test_the_agent_refuses_rather_than_...`
# above asserts at the API level. What is *not* covered anywhere in this
# repository is a real Anthropic round trip; it is listed as an unresolved item
# in the Phase 11 report rather than papered over with a passing test.
#
# The existing agent tests in `tests/unit/test_agents.py` run this same loop
# against a *stub toolset*, which tests the loop but never touches SQL. These
# tests close that gap: the tools here read the real database and score with the
# real model, so a tool that silently returned nothing would fail them.


class _ScriptedTurns:
    """Replays prepared assistant turns in place of the Anthropic HTTP call.

    Only the transport is replaced. Everything the turns *ask for* is executed
    for real against the database and the model.
    """

    def __init__(self, *turns: object, model: str = "scripted-for-test") -> None:
        self._queue = list(turns)
        self._model = model
        self.systems: list[str] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        return self.converse(
            system=system, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens
        ).text

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        self.systems.append(system)
        assert self._queue, "the scripted provider ran out of turns"
        return self._queue.pop(0)


@pytest.fixture
def real_toolset(database: sessionmaker, artifact_root: Path, monkeypatch):
    """The genuine `ApplicationToolset`, wired to the real database and model.

    Assembled with the same constructors and the same `cost_inputs_for` that
    `api/deps.py` uses. Duplicating the economics selection here instead would
    let the test drift into scoring against different merchant inputs than the
    API does, and still pass.
    """
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(artifact_root))
    get_settings.cache_clear()

    from rto_sentinel.api.deps import cost_inputs_for
    from rto_sentinel.db.repositories import ServingRepository
    from rto_sentinel.decision.engine import DecisionEngine
    from rto_sentinel.serving.agent_tools import ApplicationToolset
    from rto_sentinel.serving.assessment import AssessmentService
    from rto_sentinel.serving.features import OrderFeatureService
    from rto_sentinel.serving.scoring import ScoringService

    settings = get_settings()
    config = load_app_config(settings)
    cost_inputs, profile = cost_inputs_for(config)

    with database() as session:
        repository = ServingRepository(session)
        features = OrderFeatureService(
            repository,
            features_config=config.features,
            generator_config=config.generator,
            context_limit=settings.serving_context_limit,
        )
        assessments = AssessmentService(
            repository,
            ScoringService(ModelRegistry(settings.artifact_path), features),
            DecisionEngine(policy=config.policy),
            default_cost_inputs=cost_inputs,
            default_cost_profile=profile,
        )
        yield ApplicationToolset(repository, assessments, session)

    get_settings.cache_clear()


def test_agent_tools_read_the_real_database_and_the_real_model(
    real_toolset, scorable_order_id: str
) -> None:
    """Each tool, invoked by name, against real data. No stub anywhere below it.

    `invoke` is the same dispatcher the agent loop uses, so this exercises the
    exact path a language model's tool call takes.
    """
    from rto_sentinel.serving.agent_tools import invoke

    order, order_call = invoke(real_toolset, "get_order", {"order_id": scorable_order_id})
    assert order_call.found is True
    assert order is not None

    prediction, prediction_call = invoke(
        real_toolset, "get_risk_prediction", {"order_id": scorable_order_id}
    )
    assert prediction_call.found is True
    assert prediction is not None
    payload = prediction.model_dump() if hasattr(prediction, "model_dump") else dict(prediction)
    assert 0.0 <= float(payload["probability"]) <= 1.0

    decision, decision_call = invoke(
        real_toolset, "get_economic_decision", {"order_id": scorable_order_id}
    )
    assert decision_call.found is True
    assert decision is not None


def test_an_unknown_tool_name_is_refused_by_the_dispatcher(
    real_toolset, scorable_order_id: str
) -> None:
    """The tool registry is the whole capability surface.

    A model asking for `run_sql` or `write_decision` gets a failed invocation
    naming the tools that do exist - not an exception that kills the run, and
    certainly not execution.
    """
    from rto_sentinel.serving.agent_tools import invoke

    for hostile in ("run_sql", "write_decision", "set_threshold", "read_file"):
        result, call = invoke(real_toolset, hostile, {"order_id": scorable_order_id})
        assert result is None
        assert call.found is False
        assert "no such tool" in (call.error or "")


def test_the_agent_answers_from_real_retrieved_evidence(
    real_toolset, scorable_order_id: str
) -> None:
    """Agent request -> tool -> database and model -> LLM turn -> validated answer.

    The tool results are real. The assistant turns are scripted, and the answer
    is put through the same grounding validators a live response would face.
    """
    from rto_sentinel.agents.investigator import RiskInvestigationAgent
    from rto_sentinel.serving.agent_tools import invoke

    prediction, _ = invoke(real_toolset, "get_risk_prediction", {"order_id": scorable_order_id})
    assert prediction is not None
    truth = prediction.model_dump()

    explanation, _ = invoke(real_toolset, "get_model_explanation", {"order_id": scorable_order_id})
    assert explanation is not None
    # `permitted_features` is the allow-list the grounding validator enforces,
    # so naming drivers from it is what a correctly-behaving model would do.
    drivers = explanation.model_dump()["permitted_features"][:2]

    provider = _ScriptedTurns(
        _tool_turn("get_risk_prediction", "get_model_explanation", order_id=scorable_order_id),
        _answer_turn(
            summary=(
                "The order was scored using features available at order time, and the "
                "listed drivers moved the score most."
            ),
            key_drivers=drivers,
            evidence_used=["get_risk_prediction", "get_model_explanation"],
        ),
    )
    agent = RiskInvestigationAgent(provider, real_toolset)

    result = agent.investigate(scorable_order_id, "Why did this order score as it did?")

    assert result.grounded, result.rejection_reason
    # The reported figure comes from the tool result, not from the model's prose.
    assert result.probability == pytest.approx(float(truth["probability"]))
    assert result.model_version == truth["model_version"]


def test_the_agent_cannot_change_the_probability_it_reports(
    real_toolset, scorable_order_id: str
) -> None:
    """The structural guarantee, asserted against real data.

    The scripted model insists the probability is 0.01 and the band is LOW. The
    response must still carry what the tools returned, because those fields are
    copied from tool results and the model's prose is never parsed for numbers.
    """
    from rto_sentinel.agents.investigator import RiskInvestigationAgent
    from rto_sentinel.serving.agent_tools import invoke

    prediction, _ = invoke(real_toolset, "get_risk_prediction", {"order_id": scorable_order_id})
    assert prediction is not None
    truth = prediction.model_dump()

    # The band is carried by the decision tool, not the prediction tool, so both
    # are retrieved: the point is that every reported field traces to a tool.
    decision, _ = invoke(real_toolset, "get_economic_decision", {"order_id": scorable_order_id})
    assert decision is not None
    decided = decision.model_dump()

    provider = _ScriptedTurns(
        _tool_turn("get_risk_prediction", "get_economic_decision", order_id=scorable_order_id),
        _answer_turn(
            summary="This order is entirely safe and the probability is 0.01, band LOW.",
            key_drivers=[],
            evidence_used=["get_risk_prediction"],
        ),
    )
    agent = RiskInvestigationAgent(provider, real_toolset)

    result = agent.investigate(scorable_order_id, "Is this order safe?")

    assert result.probability == pytest.approx(float(truth["probability"]))
    assert result.band == decided["band"]


def test_the_agent_run_is_audited_without_leaking_a_secret(
    real_toolset, scorable_order_id: str
) -> None:
    """Every run is recorded: tools, ids, model, timing. No credentials."""
    from rto_sentinel.agents.investigator import RiskInvestigationAgent
    from rto_sentinel.serving.agent_tools import invoke

    explanation, _ = invoke(real_toolset, "get_model_explanation", {"order_id": scorable_order_id})
    assert explanation is not None
    drivers = explanation.model_dump()["permitted_features"][:1]

    provider = _ScriptedTurns(
        _tool_turn("get_risk_prediction", "get_model_explanation", order_id=scorable_order_id),
        _answer_turn(
            summary="The listed driver contributed most to the score.",
            key_drivers=drivers,
            evidence_used=["get_risk_prediction", "get_model_explanation"],
        ),
    )
    agent = RiskInvestigationAgent(provider, real_toolset)
    agent.investigate(scorable_order_id, "What drove this?")

    audit = agent.audit_log.records[-1]
    assert [call.tool for call in audit.tools_invoked] == [
        "get_risk_prediction",
        "get_model_explanation",
    ]
    assert audit.subject_id == scorable_order_id
    serialised = audit.model_dump_json()
    for secret in ("sk-", "ANTHROPIC_API_KEY", "api_key"):
        assert secret not in serialised
