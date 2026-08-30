"""Shared pytest fixtures.

Two things this file is careful about:

* **No live PostgreSQL.** The suite points ``RTO_DATABASE_URL`` at a temporary
  SQLite file, so ``pytest`` runs on a clean checkout with nothing installed but
  the Python dependencies. Tests that genuinely need PostgreSQL are marked
  ``requires_db`` and skipped by default.
* **No API key.** The language layer is explicitly disabled for every test.
  Nothing in this suite makes a network call, and a test that silently started
  billing an Anthropic account would be a bad surprise.

The generated-dataset fixtures are session-scoped. Generation is not free - the
base-rate calibration runs the whole simulation two or three times - and every
test that needs a dataset needs the *same* one anyway.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from rto_sentinel.configuration import (
    load_cost_model_config,
    load_evaluation_config,
    load_features_config,
    load_final_model_config,
    load_generator_config,
    load_ladder_config,
    load_model_card_config,
    load_policy_config,
    load_splits_config,
)
from rto_sentinel.configuration.schemas import (
    CostModelConfig,
    EvaluationConfig,
    FeaturesConfig,
    FinalModelConfig,
    GeneratorConfig,
    LadderConfig,
    ModelCardConfig,
    PolicyConfig,
    SplitsConfig,
)
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import ConfiguredOrderGenerator, GenerationResult, GeneratorParams
from rto_sentinel.data.splits import assign_splits
from rto_sentinel.features import (
    ModelingDataset,
    attach_customer_dimension,
    build_modeling_dataset,
)
from rto_sentinel.settings import REPO_ROOT, Settings, get_settings

# Environment variables that must not leak in from the developer's own shell or a
# stray .env file and change what the suite is testing.
_ISOLATED_VARS = (
    "RTO_ENV",
    "RTO_DATABASE_URL",
    "RTO_ACTIVE_MODEL_PATH",
    "RTO_AGENTS_ENABLED",
    "ANTHROPIC_API_KEY",
    "RTO_CONFIG_DIR",
    "RTO_ARTIFACT_DIR",
)

#: A deliberately small dataset. Big enough that customers accumulate history and
#: every split is populated; small enough that the suite stays fast.
SMALL_DATASET = GeneratorParams(
    seed=1234,
    generator_version="1.0.0",
    n_customers=600,
    n_orders=2000,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give every test a clean, offline, database-free environment."""
    for name in _ISOLATED_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("RTO_ENV", "test")
    monkeypatch.setenv("RTO_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("RTO_AGENTS_ENABLED", "false")
    monkeypatch.setenv("RTO_CONFIG_DIR", str(REPO_ROOT / "config"))
    # An EMPTY artefact store by default. Without this the suite reads whatever
    # the developer last trained into `artifacts/`, so a test asserting "no model
    # is loaded" would pass or fail depending on someone's local state - and,
    # worse, a test asserting a score would pass against a model it never
    # created. A test that needs an artefact builds one into a tmp_path and
    # points RTO_ARTIFACT_DIR at it explicitly.
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    get_settings.cache_clear()

    # Create the schema in the per-test SQLite file. Without it, any endpoint
    # that touches the database fails on a missing table - which masks the thing
    # a test was actually asserting. A test database with no schema is not a test
    # database, it is a different failure wearing the same clothes.
    _create_schema(get_settings().database.url)

    yield
    get_settings.cache_clear()


def _create_schema(url: str) -> None:
    from sqlalchemy import create_engine

    from rto_sentinel.db.base import Base

    engine = create_engine(url, future=True)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


@pytest.fixture
def settings() -> Settings:
    """Settings built from the isolated test environment."""
    return get_settings()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def source_root() -> Path:
    return REPO_ROOT / "src" / "rto_sentinel"


# ---------------------------------------------------------------------------
# generated data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def generator_config() -> GeneratorConfig:
    """The shipped generator configuration, loaded once."""
    return load_generator_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def splits_config() -> SplitsConfig:
    return load_splits_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def small_dataset(generator_config: GeneratorConfig) -> GenerationResult:
    """One small generated dataset, shared across the suite."""
    return ConfiguredOrderGenerator().generate(generator_config, SMALL_DATASET)


@pytest.fixture(scope="session")
def features_config() -> FeaturesConfig:
    return load_features_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def split_labels(small_dataset: GenerationResult, splits_config: SplitsConfig) -> pd.Series:
    """Split assignment for the shared dataset, computed once."""
    return assign_splits(small_dataset.orders, splits_config).labels


@pytest.fixture(scope="session")
def feature_frame(small_dataset: GenerationResult, split_labels: pd.Series) -> pd.DataFrame:
    """The raw order table with splits assigned and the customer dimension joined.

    This is exactly what ``FeaturePipeline.build`` is handed, so the leakage tests
    exercise the real input rather than a convenient reconstruction of it.
    """
    orders = small_dataset.orders.copy()
    orders[cols.SPLIT] = split_labels
    return attach_customer_dimension(orders, small_dataset.customers)


@pytest.fixture(scope="session")
def modeling_dataset(
    small_dataset: GenerationResult,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
    split_labels: pd.Series,
) -> ModelingDataset:
    """One built modelling dataset, shared across the suite."""
    return build_modeling_dataset(
        small_dataset,
        features_config=features_config,
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=split_labels,
    )


# ---------------------------------------------------------------------------
# ladder fixtures
# ---------------------------------------------------------------------------

#: Larger than SMALL_DATASET. The ladder needs enough rows for a tree model to
#: fit at all and for the pincode blocklist to find any pincode with support;
#: at 2,000 orders the blocklist comes out empty and the tests would be asserting
#: nothing. Still small enough to keep the suite under a minute.
LADDER_DATASET = GeneratorParams(
    seed=4242,
    generator_version="1.0.0",
    n_customers=5000,
    n_orders=15000,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


@pytest.fixture(scope="session")
def ladder_config() -> LadderConfig:
    return load_ladder_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def cost_config() -> CostModelConfig:
    return load_cost_model_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def final_config() -> FinalModelConfig:
    return load_final_model_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def model_card_config() -> ModelCardConfig:
    return load_model_card_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def policy_config() -> PolicyConfig:
    return load_policy_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def evaluation_config() -> EvaluationConfig:
    """The real evaluation config, so the reporting prohibitions under test are
    the ones the project actually ships."""
    return load_evaluation_config(Settings(RTO_CONFIG_DIR=str(REPO_ROOT / "config")))


@pytest.fixture(scope="session")
def ladder_dataset(
    generator_config: GeneratorConfig,
    features_config: FeaturesConfig,
    splits_config: SplitsConfig,
) -> ModelingDataset:
    """A modelling dataset big enough to train the ladder on, built once."""
    result = ConfiguredOrderGenerator().generate(generator_config, LADDER_DATASET)
    labels = assign_splits(result.orders, splits_config).labels
    return build_modeling_dataset(
        result,
        features_config=features_config,
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=labels,
    )


@pytest.fixture
def model_card_factory(ladder_dataset: ModelingDataset):
    """Build a minimal, valid model card for artefact round-trip tests."""
    from rto_sentinel.contracts.risk import ModelCard

    def build(name: str, model) -> ModelCard:  # type: ignore[no-untyped-def]
        metadata = ladder_dataset.metadata
        return ModelCard(
            model_name=name,
            model_version="testv1",
            rung_id=model.rung_id,
            trained_at=datetime(2026, 1, 1, tzinfo=UTC),
            training_rows=ladder_dataset.train.n_rows,
            feature_names=tuple(ladder_dataset.train.x.columns),
            enabled_families=metadata.families_used,
            random_seed=11,
            config_fingerprint=metadata.config_fingerprint,
            feature_fingerprint=metadata.feature_fingerprint,
            dataset_run_id=metadata.dataset_run_id,
            generator_version=metadata.generator_version,
        )

    return build
