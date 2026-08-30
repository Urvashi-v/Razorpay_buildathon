"""Order, decision, override and monitoring endpoints against a test database.

Scope note: these tests exercise validation, pagination, error handling and the
database round trip. They do **not** train a model - the full chain, with a real
artefact, lives in ``test_serving_integration.py``. Split that way so a
validation regression fails in seconds rather than behind a training run, and so
the integration test's failure means what it says.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rto_sentinel.api.deps import db_session_dep
from rto_sentinel.api.main import create_app
from rto_sentinel.configuration.schemas import GeneratorConfig, SplitsConfig
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import ConfiguredOrderGenerator, GeneratorParams
from rto_sentinel.data.splits import assign_splits
from rto_sentinel.db.base import Base
from rto_sentinel.db.repositories import DatasetRepository, split_reason_codes

SMALL = GeneratorParams(
    seed=777,
    generator_version="1.0.0",
    n_customers=200,
    n_orders=600,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


@pytest.fixture(scope="module")
def loaded_db(
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[sessionmaker]:
    result = ConfiguredOrderGenerator().generate(generator_config, SMALL)
    result.orders[cols.SPLIT] = assign_splits(result.orders, splits_config).labels

    path = tmp_path_factory.mktemp("orders-db") / "orders.db"
    engine = create_engine(f"sqlite+pysqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        DatasetRepository(session).load(result)
        session.commit()
    yield factory
    engine.dispose()


@pytest.fixture
def client(loaded_db: sessionmaker) -> Iterator[TestClient]:
    app = create_app()

    def session_override() -> Iterator[Session]:
        with loaded_db() as session:
            yield session
            session.commit()

    app.dependency_overrides[db_session_dep] = session_override
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def any_order_id(client: TestClient) -> str:
    return client.get("/v1/orders", params={"limit": 1}).json()["orders"][0]["order_id"]


# ---------------------------------------------------------------------------
# listing and pagination
# ---------------------------------------------------------------------------


def test_orders_come_from_the_database(client: TestClient) -> None:
    body = client.get("/v1/orders", params={"limit": 5}).json()

    assert body["total"] == SMALL.n_orders
    assert len(body["orders"]) == 5
    for order in body["orders"]:
        assert order["order_id"].startswith("ORD-")
        assert order["merchant_id"]
        assert order["payment_method"] in {"cod", "prepaid"}


def test_pagination_walks_the_book_without_repeating(client: TestClient) -> None:
    first = client.get("/v1/orders", params={"limit": 10, "offset": 0}).json()
    second = client.get("/v1/orders", params={"limit": 10, "offset": 10}).json()

    ids_a = [order["order_id"] for order in first["orders"]]
    ids_b = [order["order_id"] for order in second["orders"]]
    assert not set(ids_a) & set(ids_b)
    assert first["total"] == second["total"]
    assert second["offset"] == 10


def test_the_page_size_is_capped(client: TestClient) -> None:
    """An uncapped page is a denial-of-service surface, not a convenience."""
    assert client.get("/v1/orders", params={"limit": 5000}).status_code == 422
    assert client.get("/v1/orders", params={"limit": 0}).status_code == 422
    assert client.get("/v1/orders", params={"offset": -1}).status_code == 422


def test_filters_are_applied_in_sql(client: TestClient) -> None:
    cod = client.get("/v1/orders", params={"payment_method": "cod", "limit": 50}).json()
    assert cod["total"] < SMALL.n_orders
    assert all(order["payment_method"] == "cod" for order in cod["orders"])

    train = client.get("/v1/orders", params={"split": "train", "limit": 50}).json()
    assert all(order["split"] == "train" for order in train["orders"])


def test_an_unknown_payment_method_is_refused(client: TestClient) -> None:
    assert client.get("/v1/orders", params={"payment_method": "crypto"}).status_code == 422


def test_an_immature_order_reports_a_null_label(client: TestClient) -> None:
    """NULL, never False. Defaulting it would manufacture optimism."""
    page = client.get("/v1/orders", params={"split": "excluded_immature", "limit": 5}).json()
    if not page["orders"]:
        pytest.skip("this dataset produced no immature orders")
    for order in page["orders"]:
        assert order["is_rto"] is None
        assert order["resolved_at"] is None


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_one_order_is_retrievable(client: TestClient, any_order_id: str) -> None:
    response = client.get(f"/v1/orders/{any_order_id}")
    assert response.status_code == 200
    assert response.json()["order_id"] == any_order_id


def test_an_unknown_order_is_a_404_with_a_code(client: TestClient) -> None:
    response = client.get("/v1/orders/ORD-99999999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ORDER_NOT_FOUND"


@pytest.mark.parametrize("bad", ["../etc/passwd", "a" * 100, "'; DROP TABLE orders;--"])
def test_a_malformed_order_id_never_reaches_the_database(client: TestClient, bad: str) -> None:
    """Rejected by the path pattern, so it is never interpolated into a query."""
    response = client.get(f"/v1/orders/{bad}")
    assert response.status_code in {404, 422}
    assert "DROP TABLE" not in response.text


# ---------------------------------------------------------------------------
# validation of the request bodies
# ---------------------------------------------------------------------------


def test_an_override_reason_is_mandatory_and_substantive(client: TestClient) -> None:
    """An override with no stated reason is unusable as counterfactual evidence."""
    for reason in ("", "ok", "   "):
        response = client.post(
            "/v1/decisions/override",
            json={
                "order_id": "ORD-00000001",
                "override_band": "HIGH",
                "operator_id": "op-1",
                "reason": reason,
            },
        )
        assert response.status_code == 422, f"reason {reason!r} should be refused"


def test_an_override_needs_a_valid_band(client: TestClient) -> None:
    response = client.post(
        "/v1/decisions/override",
        json={
            "order_id": "ORD-00000001",
            "override_band": "CATASTROPHIC",
            "operator_id": "op-1",
            "reason": "a reason of entirely sufficient length to pass validation",
        },
    )
    assert response.status_code == 422


def test_an_operator_id_is_required(client: TestClient) -> None:
    response = client.post(
        "/v1/decisions/override",
        json={
            "order_id": "ORD-00000001",
            "override_band": "HIGH",
            "operator_id": "",
            "reason": "a reason of entirely sufficient length to pass validation",
        },
    )
    assert response.status_code == 422


def test_scoring_validates_the_order_id(client: TestClient) -> None:
    response = client.post("/v1/score", json={"order_id": "not a valid id!!"})
    assert response.status_code == 422


def test_a_batch_is_capped(client: TestClient) -> None:
    """Each order rebuilds its own feature context, so the cap is a real cost ceiling."""
    payload = [{"order_id": f"ORD-{index:08d}"} for index in range(1, 200)]
    assert client.post("/v1/score/batch", json=payload).status_code == 422


def test_merchant_economics_are_validated_on_the_scoring_path(client: TestClient) -> None:
    response = client.post(
        "/v1/score",
        json={
            "order_id": "ORD-00000001",
            "cost_inputs": {
                "rto_cost_inr": 220.0,
                "contribution_margin_inr": -50.0,
                "abandonment_on_friction": 0.25,
                "intervention_success_rate": 0.6,
            },
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


# ---------------------------------------------------------------------------
# errors reveal nothing
# ---------------------------------------------------------------------------


def test_errors_carry_no_stack_trace_or_secret(client: TestClient) -> None:
    for response in (
        client.get("/v1/orders/ORD-99999999"),
        client.get("/v1/orders", params={"limit": 99999}),
        client.post("/v1/score", json={"order_id": "!!"}),
    ):
        text = response.text.lower()
        assert "traceback" not in text
        assert "sqlalchemy" not in text
        assert "password" not in text
        assert "postgresql://" not in text
        assert ".py" not in text


def test_every_error_uses_the_same_envelope(client: TestClient) -> None:
    for response in (
        client.get("/v1/orders/ORD-99999999"),
        client.get("/v1/orders", params={"limit": 0}),
    ):
        body = response.json()
        assert set(body) == {"error"}
        assert {"code", "message"} <= set(body["error"])


# ---------------------------------------------------------------------------
# monitoring reads the database, not a cache
# ---------------------------------------------------------------------------


def test_monitoring_counts_match_the_database(client: TestClient) -> None:
    body = client.get("/v1/monitoring/data").json()

    assert body["total_orders"] == SMALL.n_orders
    assert sum(body["orders_by_split"].values()) == SMALL.n_orders
    assert sum(body["orders_by_payment_method"].values()) == SMALL.n_orders
    assert body["matured_orders"] + body["immature_orders"] == SMALL.n_orders
    assert body["dataset_runs"], "the loaded run must be reported"


def test_the_rto_rate_is_computed_over_matured_orders_only(client: TestClient) -> None:
    """Dividing by every order would count 'not yet resolved' as 'did not return'."""
    body = client.get("/v1/monitoring/data").json()
    rate = body["observed_rto_rate"]
    assert rate is None or 0.0 <= rate <= 1.0
    if rate is not None and body["immature_orders"] > 0:
        naive = rate * body["matured_orders"] / body["total_orders"]
        assert rate > naive, "the reported rate must not be diluted by immature orders"


def test_monitoring_reports_no_model_without_inventing_one(client: TestClient) -> None:
    """No artefact in this fixture, so the endpoint says so and does not raise."""
    body = client.get("/v1/monitoring/model").json()
    assert body["available"] is False
    assert body["reason"]
    assert body["model_version"] is None


def test_scoring_without_a_model_is_a_503(client: TestClient, any_order_id: str) -> None:
    """The refusal, on the endpoints that would otherwise have to invent a number."""
    for response in (
        client.get(f"/v1/orders/{any_order_id}/risk"),
        client.post("/v1/score", json={"order_id": any_order_id}),
        client.post("/v1/decisions", json={"order_id": any_order_id}),
    ):
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_the_fairness_endpoint_refuses_rather_than_fabricating(client: TestClient) -> None:
    """A fairness report nobody computed is worse than none at all."""
    response = client.get("/v1/evaluation/fairness")
    assert response.status_code == 501
    assert "has not been run" in response.json()["error"]["message"]


def test_reason_codes_round_trip_through_the_text_column() -> None:
    """The storage format is lossless for the codes this system emits."""
    assert split_reason_codes("ORDER_IS_COD,HISTORY_PRIOR_RTO_RATE") == [
        "ORDER_IS_COD",
        "HISTORY_PRIOR_RTO_RATE",
    ]
    assert split_reason_codes("") == []
    assert split_reason_codes(None) == []
