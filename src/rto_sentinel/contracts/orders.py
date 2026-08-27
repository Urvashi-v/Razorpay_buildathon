"""Order payloads - the input side of the scoring contract.

PRIVACY BOUNDARY (SPEC section 09): this schema accepts a *hashed* customer
identifier and never a name. There is deliberately no field for a customer name,
gender, age, or anything from which those could be inferred. The address line is
accepted because address *quality* is a legitimate and fair delivery-risk signal;
it is consumed only to derive quality metrics and is never stored beyond the
merchant boundary.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rto_sentinel.contracts.enums import DeviceClass, OrderOutcome, PaymentMethod

_HASH_PATTERN = re.compile(r"^[a-f0-9]{16,64}$")
_PINCODE_PATTERN = re.compile(r"^[1-9][0-9]{5}$")


class OrderLineItem(BaseModel):
    """One line of the basket. Item count and value shape are the useful signals."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(max_length=64)
    category: str = Field(max_length=64)
    quantity: int = Field(ge=1, le=999)
    unit_price_inr: float = Field(ge=0)


class SessionContext(BaseModel):
    """Weak-individually intent signals. See features.yaml -> session_intent."""

    model_config = ConfigDict(extra="forbid")

    product_page_seconds: float | None = Field(default=None, ge=0)
    sessions_before_purchase: int | None = Field(default=None, ge=0)
    device_class: DeviceClass = DeviceClass.UNKNOWN
    time_to_checkout_seconds: float | None = Field(default=None, ge=0)
    cart_edited: bool = False
    cod_after_prepaid_failure: bool = False


class AddressPayload(BaseModel):
    """Delivery address, consumed for *quality* signals only.

    The raw text is not featurised as text in rungs 0-4; only derived quality
    metrics (token count, house-number presence, pincode/city consistency, and so
    on) reach the model.
    """

    model_config = ConfigDict(extra="forbid")

    line: str = Field(max_length=512, description="Free-text address line as entered")
    city: str = Field(max_length=128)
    state: str = Field(max_length=128)
    pincode: str = Field(description="6-digit Indian PIN code")

    @field_validator("pincode")
    @classmethod
    def _valid_pincode(cls, value: str) -> str:
        if not _PINCODE_PATTERN.match(value):
            msg = "pincode must be 6 digits and must not start with 0"
            raise ValueError(msg)
        return value


class OrderPayload(BaseModel):
    """A single order presented for scoring at checkout.

    Every field is order metadata. Nothing here identifies a person by name.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(max_length=64, description="Merchant's own order reference")
    merchant_id: str = Field(max_length=64)
    customer_hash: str = Field(
        description="Stable, salted hash of the customer identity. Never a raw name or phone."
    )
    ordered_at: datetime = Field(description="Order timestamp; the as-of point for all aggregates")

    payment_method: PaymentMethod
    order_value_inr: float = Field(gt=0, le=1_000_000)
    discount_inr: float = Field(default=0.0, ge=0)

    address: AddressPayload
    items: list[OrderLineItem] = Field(min_length=1, max_length=200)
    session: SessionContext = Field(default_factory=SessionContext)

    courier_partner: str | None = Field(default=None, max_length=64)

    @field_validator("customer_hash")
    @classmethod
    def _looks_hashed(cls, value: str) -> str:
        """Reject anything that is plainly not a hash.

        This is a guard rail, not cryptography: it stops a caller from posting a
        phone number or an email into the identity field by accident.
        """
        if not _HASH_PATTERN.match(value.lower()):
            msg = "customer_hash must be a 16-64 character lowercase hex digest"
            raise ValueError(msg)
        return value.lower()

    @property
    def discount_depth(self) -> float:
        """Discount as a fraction of gross order value, in [0, 1]."""
        gross = self.order_value_inr + self.discount_inr
        return self.discount_inr / gross if gross > 0 else 0.0

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items)


class OrderOutcomeUpdate(BaseModel):
    """Terminal state reported back once the courier resolves the shipment.

    This is the label source. An order is only labelled once its terminal state
    is known - SPEC section 03, rule 5.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(max_length=64)
    outcome: OrderOutcome
    resolved_at: datetime

    @field_validator("outcome")
    @classmethod
    def _terminal_only(cls, value: OrderOutcome) -> OrderOutcome:
        if value is OrderOutcome.PENDING:
            msg = "an outcome update must report a terminal state, not PENDING"
            raise ValueError(msg)
        return value
