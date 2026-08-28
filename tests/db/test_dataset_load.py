"""A generated dataset survives a round-trip through the database.

Runs against SQLite so the suite needs no server. The PostgreSQL path is
exercised by ``rto-sentinel seed-db`` and is not asserted here - what these tests
check is the mapping, the null handling, and the constraints, all of which are
engine-independent.

The null-handling test is the one that earns its place. ``float('nan')`` inserted
into a nullable column is *not* NULL, it is the float NaN, and it would quietly
turn "this customer has no history" into "this customer has an unrepresentable
history" - a distinction no model can recover from.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import pairwise

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from rto_sentinel.configuration.schemas import GeneratorConfig, SplitsConfig
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import ConfiguredOrderGenerator, GeneratorParams
from rto_sentinel.data.splits import assign_splits
from rto_sentinel.db.base import Base
from rto_sentinel.db.models import (
    Address,
    Customer,
    DatasetRun,
    DeliveryEvent,
    Order,
    OrderOutcomeRecord,
    SimulationLatent,
)
from rto_sentinel.db.repositories import DatasetRepository

TINY = GeneratorParams(
    seed=555,
    generator_version="1.0.0",
    n_customers=120,
    n_orders=400,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


@pytest.fixture(scope="module")
def tiny_dataset(generator_config: GeneratorConfig, splits_config: SplitsConfig):
    result = ConfiguredOrderGenerator().generate(generator_config, TINY)
    result.orders[cols.SPLIT] = assign_splits(result.orders, splits_config).labels
    return result


@pytest.fixture
def session(tmp_path) -> Iterator[Session]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'roundtrip.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as db_session:
        yield db_session
    engine.dispose()


@pytest.fixture
def loaded(session: Session, tiny_dataset) -> tuple[Session, str]:
    counts = DatasetRepository(session).load(tiny_dataset)
    session.commit()
    assert counts["orders"] == len(tiny_dataset.orders)
    return session, tiny_dataset.metadata.run_id


# ---------------------------------------------------------------------------
# the round-trip
# ---------------------------------------------------------------------------


def test_every_table_receives_its_rows(loaded, tiny_dataset) -> None:
    session, run_id = loaded
    counts = DatasetRepository(session).table_counts(run_id)

    assert counts["customers"] == len(tiny_dataset.customers)
    assert counts["addresses"] == len(tiny_dataset.addresses)
    assert counts["orders"] == len(tiny_dataset.orders)
    assert counts["order_outcomes"] == len(tiny_dataset.orders)
    assert counts["delivery_events"] == len(tiny_dataset.delivery_events)
    assert counts["simulation_latents"] == len(tiny_dataset.orders)


def test_provenance_is_recorded(loaded, tiny_dataset) -> None:
    """Seed, generator version, configuration and creation time all persist."""
    session, run_id = loaded
    run = session.get(DatasetRun, run_id)

    assert run is not None
    assert run.seed == TINY.seed
    assert run.generator_version == TINY.generator_version
    assert run.n_orders == len(tiny_dataset.orders)
    assert run.config_fingerprint == tiny_dataset.metadata.config_fingerprint
    assert run.config_snapshot["base_rates"]["rto_given_cod"] == pytest.approx(0.26)
    assert run.created_at is not None
    assert "not real-world ground truth" in run.data_provenance


def test_foreign_keys_resolve(loaded) -> None:
    """Orders point at real customers and real addresses."""
    session, run_id = loaded
    orphan_customers = session.execute(
        select(func.count())
        .select_from(Order)
        .outerjoin(Customer, Customer.id == Order.customer_pk)
        .where(Order.dataset_run_id == run_id)
        .where(Customer.id.is_(None))
    ).scalar_one()
    orphan_addresses = session.execute(
        select(func.count())
        .select_from(Order)
        .outerjoin(Address, Address.id == Order.address_pk)
        .where(Order.dataset_run_id == run_id)
        .where(Address.id.is_(None))
    ).scalar_one()

    assert orphan_customers == 0
    assert orphan_addresses == 0


def test_missing_history_persists_as_null_not_nan(loaded, tiny_dataset) -> None:
    """NaN must become SQL NULL on the way in.

    A NaN in ``prior_rto_rate`` would read back as a float that is neither a rate
    nor a null, and every downstream comparison against it silently returns false.
    """
    session, run_id = loaded
    null_rates = session.execute(
        select(func.count())
        .select_from(Order)
        .where(Order.dataset_run_id == run_id)
        .where(Order.prior_rto_rate.is_(None))
    ).scalar_one()

    expected = int(tiny_dataset.orders[cols.PRIOR_RTO_RATE].isna().sum())
    assert null_rates == expected
    assert null_rates > 0, "the fixture should contain first-time customers"


def test_immature_orders_persist_with_a_null_label(loaded, tiny_dataset) -> None:
    """The maturity rule survives the database, not only the frame."""
    session, run_id = loaded
    immature = (
        session.execute(
            select(OrderOutcomeRecord)
            .join(Order, Order.id == OrderOutcomeRecord.order_pk)
            .where(Order.dataset_run_id == run_id)
            .where(OrderOutcomeRecord.is_mature.is_(False))
        )
        .scalars()
        .all()
    )

    assert len(immature) == int((~tiny_dataset.orders[cols.IS_MATURE]).sum())
    for record in immature:
        assert record.is_rto is None
        assert record.resolved_at is None
        assert record.outcome == "pending"


def test_timestamps_survive_the_round_trip(loaded, tiny_dataset) -> None:
    """Ordering relationships must hold after a write and a read.

    Every piece of temporal reasoning in this project depends on it, and a naive
    timestamp column would silently shift instants by the server's offset.
    """
    session, run_id = loaded
    rows = session.execute(
        select(Order.ordered_at, Order.dispatched_at, OrderOutcomeRecord.resolved_at)
        .join(OrderOutcomeRecord, OrderOutcomeRecord.order_pk == Order.id)
        .where(Order.dataset_run_id == run_id)
        .where(Order.dispatched_at.is_not(None))
    ).all()

    assert len(rows) > 0
    for ordered_at, dispatched_at, resolved_at in rows:
        assert dispatched_at >= ordered_at
        if resolved_at is not None:
            assert resolved_at > ordered_at


def test_delivery_events_are_ordered_within_an_order(loaded) -> None:
    session, run_id = loaded
    order_id = session.execute(
        select(Order.id).where(Order.dataset_run_id == run_id).limit(1)
    ).scalar_one()

    events = (
        session.execute(
            select(DeliveryEvent)
            .where(DeliveryEvent.order_pk == order_id)
            .order_by(DeliveryEvent.sequence)
        )
        .scalars()
        .all()
    )

    assert len(events) >= 2
    assert events[0].event_type == "order_placed"
    for earlier, later in pairwise(events):
        assert earlier.sequence < later.sequence
        assert earlier.occurred_at <= later.occurred_at


def test_simulation_ground_truth_is_stored_separately(loaded) -> None:
    """The true probability lives in its own table, never on ``orders``."""
    session, run_id = loaded
    latent_columns = set(SimulationLatent.__table__.columns.keys())
    order_columns = set(Order.__table__.columns.keys())

    assert "true_rto_probability" in latent_columns
    assert not (order_columns & {"true_rto_probability", "latent_logit"})

    probabilities = (
        session.execute(
            select(SimulationLatent.true_rto_probability).where(
                SimulationLatent.dataset_run_id == run_id
            )
        )
        .scalars()
        .all()
    )
    assert all(0.0 <= p <= 1.0 for p in probabilities)


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_reloading_the_same_run_replaces_rather_than_duplicates(
    session: Session, tiny_dataset
) -> None:
    """Re-running the seed script must not silently double every count.

    ``run_id`` is a deterministic digest of the generator parameters, so the same
    inputs identify the same dataset. Loading twice is an upsert.
    """
    repository = DatasetRepository(session)
    repository.load(tiny_dataset)
    session.commit()
    first = repository.table_counts(tiny_dataset.metadata.run_id)

    repository.load(tiny_dataset)
    session.commit()
    second = repository.table_counts(tiny_dataset.metadata.run_id)

    assert first == second


def test_deleting_a_run_removes_its_children(session: Session, tiny_dataset) -> None:
    repository = DatasetRepository(session)
    repository.load(tiny_dataset)
    session.commit()

    repository.delete_run(tiny_dataset.metadata.run_id)
    session.commit()

    counts = repository.table_counts(tiny_dataset.metadata.run_id)
    assert all(count == 0 for count in counts.values())


# ---------------------------------------------------------------------------
# constraints are real
# ---------------------------------------------------------------------------


def test_a_labelled_immature_outcome_is_rejected_by_the_database(
    loaded,
) -> None:
    """The maturity rule is a database constraint, not only a Python check.

    Belt and braces on purpose: this is the invariant whose violation would make
    every downstream metric optimistic, so it is enforced in the type, in the
    validator, and here.
    """
    session, run_id = loaded
    order_pk = session.execute(
        select(Order.id).where(Order.dataset_run_id == run_id).limit(1)
    ).scalar_one()

    session.add(
        OrderOutcomeRecord(
            order_pk=order_pk,
            order_id="ORD-BOGUS",
            outcome="pending",
            is_rto=False,  # a label on an immature order
            resolved_at=None,
            maturity_days=None,
            is_mature=False,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_negative_order_values_are_rejected_by_the_database(session: Session, tiny_dataset) -> None:
    repository = DatasetRepository(session)
    repository.load(tiny_dataset)
    session.commit()

    order = session.execute(select(Order).limit(1)).scalar_one()
    order.order_value_inr = -50.0
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---------------------------------------------------------------------------
# several dataset runs in one database
# ---------------------------------------------------------------------------

#: A second run. Same generator, different seed, so it renumbers its orders from
#: ORD-00000001 again and re-renders many of the same addresses.
SECOND = GeneratorParams(
    seed=777,
    generator_version="1.0.0",
    n_customers=120,
    n_orders=400,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


@pytest.fixture(scope="module")
def second_dataset(generator_config: GeneratorConfig, splits_config: SplitsConfig):
    result = ConfiguredOrderGenerator().generate(generator_config, SECOND)
    result.orders[cols.SPLIT] = assign_splits(result.orders, splits_config).labels
    return result


def test_two_dataset_runs_coexist(session: Session, tiny_dataset, second_dataset) -> None:
    """The regression this file exists to prevent from recurring.

    ``order_id``, ``customer_hash`` and ``address_fingerprint`` were originally
    globally unique, which made the database able to hold exactly one benchmark
    dataset: the second `seed-db` died on

        duplicate key value violates unique constraint
        "ix_addresses_address_fingerprint"

    Both runs number their orders from ORD-00000001 and both render some of the
    same address text, so the collision is guaranteed rather than incidental.
    """
    repository = DatasetRepository(session)
    repository.load(tiny_dataset)
    repository.load(second_dataset)
    session.commit()

    first_id = tiny_dataset.metadata.run_id
    second_id = second_dataset.metadata.run_id
    assert first_id != second_id

    assert repository.table_counts(first_id)["orders"] == len(tiny_dataset.orders)
    assert repository.table_counts(second_id)["orders"] == len(second_dataset.orders)

    shared = set(tiny_dataset.orders[cols.ORDER_ID]) & set(second_dataset.orders[cols.ORDER_ID])
    assert shared, "the runs must actually share order ids for this test to mean anything"

    duplicated = (
        session.execute(select(Order.order_id).where(Order.order_id == next(iter(shared))))
        .scalars()
        .all()
    )
    assert len(duplicated) == 2, "the same order id must exist once per run"


def test_an_order_id_cannot_repeat_within_one_run(loaded) -> None:
    """Scoping uniqueness to the run must not weaken it inside the run."""
    session, run_id = loaded
    existing = session.execute(
        select(Order).where(Order.dataset_run_id == run_id).limit(1)
    ).scalar_one()

    # Copy every column except the surrogate key, so the only thing under test is
    # the repeated `order_id` rather than an incidentally invalid row.
    values = {
        column.key: getattr(existing, column.key)
        for column in Order.__mapper__.column_attrs
        if column.key != "id"
    }
    session.add(Order(**values))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
