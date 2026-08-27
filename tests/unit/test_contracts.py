"""The contracts enforce the safety invariants at construction time.

A :class:`Decision` that removes the appeal path, or applies friction with no
reason code, cannot be *built* - not by a route handler, not by a repository
replaying a log row, not by a test. Putting the invariant in the type means there
is no code path that produces an unsafe decision, rather than a convention that
every call site has to remember.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rto_sentinel.contracts import (
    AddressPayload,
    CostInputs,
    Decision,
    InterventionAction,
    OpsOverride,
    OrderLineItem,
    OrderPayload,
    OverrideDirection,
    PaymentMethod,
    PointEstimate,
    RiskBand,
    RiskScore,
    band_rank,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _address(**overrides: object) -> AddressPayload:
    payload: dict[str, object] = {
        "line": "12/3 MG Road, near the water tank",
        "city": "Pune",
        "state": "Maharashtra",
        "pincode": "411001",
    }
    payload.update(overrides)
    return AddressPayload(**payload)  # type: ignore[arg-type]


def _order(**overrides: object) -> OrderPayload:
    payload: dict[str, object] = {
        "order_id": "ORD-1001",
        "merchant_id": "M-1",
        "customer_hash": "a3f5c9d1e7b20486",
        "ordered_at": NOW,
        "payment_method": PaymentMethod.COD,
        "order_value_inr": 1499.0,
        "discount_inr": 500.0,
        "address": _address(),
        "items": [OrderLineItem(sku="S1", category="fashion", quantity=2, unit_price_inr=750.0)],
    }
    payload.update(overrides)
    return OrderPayload(**payload)  # type: ignore[arg-type]


def _decision(**overrides: object) -> Decision:
    payload: dict[str, object] = {
        "order_id": "ORD-1001",
        "probability": 0.44,
        "threshold": 0.3214,
        "band": RiskBand.HIGH,
        "action": InterventionAction.CONFIRMATION_REQUIRED,
        "flagged": True,
        "reason_codes": ("ADDRESS_INCOMPLETE",),
        "expected_value_inr": 41.2,
        "decided_at": NOW,
        "engine_version": "0.1.0",
    }
    payload.update(overrides)
    return Decision(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Orders: the privacy boundary
# ---------------------------------------------------------------------------


def test_order_accepts_a_hashed_identity() -> None:
    order = _order()
    assert order.customer_hash == "a3f5c9d1e7b20486"
    assert order.item_count == 2
    assert order.discount_depth == pytest.approx(500.0 / 1999.0)


def test_order_rejects_an_unhashed_identity() -> None:
    """A phone number in the identity field is rejected, not silently stored."""
    with pytest.raises(ValidationError, match="hex digest"):
        _order(customer_hash="9876543210")


def test_order_has_no_field_for_a_customer_name() -> None:
    """SPEC section 04: names are hashed for identity only, never featurised.

    The strongest form of that commitment is that the wire contract has nowhere
    to put a name. ``extra="forbid"`` means an integration that tries is rejected.
    """
    assert "customer_name" not in OrderPayload.model_fields
    with pytest.raises(ValidationError):
        _order(customer_name="Someone")


def test_order_rejects_a_malformed_pincode() -> None:
    with pytest.raises(ValidationError, match="6 digits"):
        _address(pincode="41100")
    with pytest.raises(ValidationError, match="6 digits"):
        _address(pincode="011001")


# ---------------------------------------------------------------------------
# Decisions: the human safeguards
# ---------------------------------------------------------------------------


def test_a_valid_decision_constructs() -> None:
    decision = _decision()
    assert decision.flagged
    assert decision.appeal_available


def test_the_appeal_path_cannot_be_removed() -> None:
    """SPEC section 09: no hard block without an appeal path."""
    with pytest.raises(ValidationError, match="appeal path"):
        _decision(appeal_available=False)


def test_friction_without_a_reason_code_is_rejected() -> None:
    """Every action carries a reason code. No exceptions, including SEVERE."""
    with pytest.raises(ValidationError, match="reason code"):
        _decision(reason_codes=())


def test_severe_must_route_to_a_human() -> None:
    with pytest.raises(ValidationError, match="human review queue"):
        _decision(band=RiskBand.SEVERE, action=InterventionAction.PREPAID_ONLY)

    ok = _decision(
        band=RiskBand.SEVERE,
        action=InterventionAction.PREPAID_ONLY,
        human_review_required=True,
    )
    assert ok.human_review_required


def test_flagged_and_action_must_agree() -> None:
    with pytest.raises(ValidationError, match="cannot be flagged"):
        _decision(action=InterventionAction.NONE, band=RiskBand.LOW)
    with pytest.raises(ValidationError, match="must be marked flagged"):
        _decision(flagged=False)


def test_bands_are_ordered() -> None:
    assert band_rank(RiskBand.LOW) < band_rank(RiskBand.ELEVATED)
    assert band_rank(RiskBand.ELEVATED) < band_rank(RiskBand.HIGH)
    assert band_rank(RiskBand.HIGH) < band_rank(RiskBand.SEVERE)


# ---------------------------------------------------------------------------
# Cost inputs
# ---------------------------------------------------------------------------


def test_degenerate_cost_inputs_are_rejected() -> None:
    """Inputs from which no threshold can be derived fail loudly.

    Returning 0.5 here would be the exact failure this project exists to correct.
    """
    with pytest.raises(ValidationError, match="degenerate"):
        CostInputs(
            rto_cost_inr=220.0,
            contribution_margin_inr=250.0,
            abandonment_on_friction=0.0,
            intervention_success_rate=0.0,
            friction_support_cost_inr=0.0,
        )


def test_cost_inputs_reject_out_of_range_probabilities() -> None:
    with pytest.raises(ValidationError):
        CostInputs(
            rto_cost_inr=220.0,
            contribution_margin_inr=250.0,
            abandonment_on_friction=1.4,
            intervention_success_rate=0.6,
        )


# ---------------------------------------------------------------------------
# Risk scores and calibration
# ---------------------------------------------------------------------------


def test_a_score_knows_whether_it_is_calibrated() -> None:
    """The decision engine refuses an uncalibrated score; this is how it can tell."""
    uncalibrated = RiskScore(
        order_id="ORD-1",
        probability=0.4,
        model_name="lightgbm_isotonic",
        model_version="v1",
        scored_at=NOW,
    )
    assert not uncalibrated.is_calibrated

    calibrated = RiskScore(
        order_id="ORD-1",
        probability=0.4,
        model_name="lightgbm_isotonic",
        model_version="v1",
        calibration_method="isotonic",
        scored_at=NOW,
    )
    assert calibrated.is_calibrated


def test_probability_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        RiskScore(
            order_id="ORD-1",
            probability=1.3,
            model_name="m",
            model_version="v1",
            scored_at=NOW,
        )


# ---------------------------------------------------------------------------
# Evaluation: no point estimate without an interval
# ---------------------------------------------------------------------------


def test_point_estimate_requires_a_containing_interval() -> None:
    """ "A point estimate on 5,000 rows is not a result" - SPEC section 07."""
    ok = PointEstimate(value=0.61, ci_low=0.58, ci_high=0.64, n_bootstrap=2000)
    assert ok.ci_low <= ok.value <= ok.ci_high

    with pytest.raises(ValidationError, match="outside its interval"):
        PointEstimate(value=0.61, ci_low=0.62, ci_high=0.64)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_override_direction_must_match_the_band_change() -> None:
    ok = OpsOverride(
        order_id="ORD-1",
        original_band=RiskBand.SEVERE,
        override_band=RiskBand.ELEVATED,
        direction=OverrideDirection.RELAXED,
        operator_id="op-hash-1",
        created_at=NOW,
    )
    assert ok.direction is OverrideDirection.RELAXED

    with pytest.raises(ValidationError, match="contradicts"):
        OpsOverride(
            order_id="ORD-1",
            original_band=RiskBand.LOW,
            override_band=RiskBand.SEVERE,
            direction=OverrideDirection.RELAXED,
            operator_id="op-hash-1",
            created_at=NOW,
        )


def test_override_must_change_something() -> None:
    with pytest.raises(ValidationError, match="must change the band"):
        OpsOverride(
            order_id="ORD-1",
            original_band=RiskBand.HIGH,
            override_band=RiskBand.HIGH,
            direction=OverrideDirection.ESCALATED,
            operator_id="op-hash-1",
            created_at=NOW,
        )
