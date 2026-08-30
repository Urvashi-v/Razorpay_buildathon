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

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rto_sentinel.api.deps import db_session_dep
from rto_sentinel.api.main import create_app
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
