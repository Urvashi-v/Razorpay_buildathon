"""The leakage suite. SPEC section 03.

    "Judges can run pytest and see them pass. That is a more persuasive claim
    about rigour than any number on a slide."

The four tests the specification names are declared here with their final names
and their final assertions written out in the docstrings. They are **skipped**
until the pipeline they test exists, and they are skipped loudly - a skipped test
appears in the pytest summary, whereas a test that passes vacuously against a
missing implementation would be worse than not having written it.

What Phase 1 *can* assert is checked below, unskipped: that the split protocol
and the forbidden-column set are configured such that these tests will have
something real to verify.
"""

from __future__ import annotations

import pytest

from rto_sentinel.configuration import load_splits_config
from rto_sentinel.data import schema
from rto_sentinel.settings import Settings

pytestmark = pytest.mark.leakage

PHASE_2 = "Implemented in Phase 2, when the data pipeline lands."


@pytest.mark.skip(reason=PHASE_2)
def test_no_future_aggregates() -> None:
    """No feature may use information that post-dates the order.

    Every historical aggregate is recomputed under a shifted clock that hides
    everything after each row's ``ordered_at``. A value that changes was reading
    the future.

    The subtle case this catches: an aggregate built from orders *placed* earlier
    rather than *resolved* earlier. An order placed on day 40 that comes back on
    day 47 was not known to be an RTO on day 42.
    """


@pytest.mark.skip(reason=PHASE_2)
def test_customer_disjoint_splits() -> None:
    """No ``customer_hash`` appears in more than one split.

    Asserts empty pairwise intersections between the train, validation and test
    customer sets. Without this the model memorises individuals and reports a
    score it cannot reproduce on anyone new.
    """


@pytest.mark.skip(reason=PHASE_2)
def test_label_maturity() -> None:
    """No order is labelled before its terminal state is known.

    Asserts that every labelled row has ``resolved_at`` within the horizon, and
    that rows whose resolution would fall outside it are marked
    ``EXCLUDED_IMMATURE`` rather than optimistically labelled "delivered".
    """


@pytest.mark.skip(reason=PHASE_2)
def test_target_not_in_features() -> None:
    """No forbidden column reaches the design matrix.

    Asserts that the matrix built by ``FeaturePipeline`` shares no column with
    ``data.schema.FORBIDDEN_IN_FEATURES`` - the label, its timestamp, the split
    marker, and the identity columns.
    """


# ---------------------------------------------------------------------------
# What Phase 1 can already verify
# ---------------------------------------------------------------------------


def test_split_protocol_is_configured_for_the_leakage_tests(settings: Settings) -> None:
    """The protocol the four tests above will check is actually configured."""
    splits = load_splits_config(settings)
    assert splits.group.disjoint_across_splits, "test_customer_disjoint_splits needs this"
    assert splits.as_of_join.enforced, "test_no_future_aggregates needs this"
    assert splits.label_maturity.exclude_immature_tail, "test_label_maturity needs this"
    assert splits.label_maturity.max_resolution_days == 9


def test_as_of_rule_uses_resolution_time_not_order_time(settings: Settings) -> None:
    """The as-of rule keys on when an order *resolved*, not when it was placed.

    This distinction is the single most important line in the split protocol, and
    getting it wrong produces a model that looks excellent and is worthless.
    """
    as_of = load_splits_config(settings).as_of_join
    assert as_of.resolution_timestamp_column == "resolved_at"
    assert as_of.order_timestamp_column == "ordered_at"
    assert "resolved_at" in as_of.rule
    assert "ordered_at" in as_of.rule


def test_forbidden_column_set_is_not_empty() -> None:
    """A guard against someone emptying the set and making test 4 vacuous."""
    assert len(schema.FORBIDDEN_IN_FEATURES) >= 8
    assert schema.TARGET_COLUMN in schema.FORBIDDEN_IN_FEATURES
