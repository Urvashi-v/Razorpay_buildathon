"""The canonical column schema for the raw order table.

Declared once here so that the generator, the validator, the as-of join, the
splitter and the feature pipeline all agree on what a row looks like. A column
that is not in this list does not exist as far as the pipeline is concerned.

Note ``TARGET_COLUMN`` and ``FORBIDDEN_IN_FEATURES``: those two constants are
what ``tests/leakage/test_target_not_in_features.py`` checks against, so the
"do not feed the label to the model" rule is mechanical.
"""

from __future__ import annotations

from typing import Final

# --- identity and time -------------------------------------------------------
ORDER_ID: Final = "order_id"
MERCHANT_ID: Final = "merchant_id"
CUSTOMER_HASH: Final = "customer_hash"
ORDERED_AT: Final = "ordered_at"
RESOLVED_AT: Final = "resolved_at"
DAY_INDEX: Final = "day_index"

# --- order shape -------------------------------------------------------------
PAYMENT_METHOD: Final = "payment_method"
ORDER_VALUE_INR: Final = "order_value_inr"
DISCOUNT_INR: Final = "discount_inr"
DISCOUNT_DEPTH: Final = "discount_depth"
ITEM_COUNT: Final = "item_count"
CATEGORY: Final = "category"
CART_EDITED: Final = "cart_edited"

# --- session -----------------------------------------------------------------
PRODUCT_PAGE_SECONDS: Final = "product_page_seconds"
SESSIONS_BEFORE_PURCHASE: Final = "sessions_before_purchase"
DEVICE_CLASS: Final = "device_class"
HOUR_OF_DAY: Final = "hour_of_day"
TIME_TO_CHECKOUT_SECONDS: Final = "time_to_checkout_seconds"
COD_AFTER_PREPAID_FAILURE: Final = "cod_after_prepaid_failure"

# --- address (quality inputs, not identity) ----------------------------------
ADDRESS_LINE: Final = "address_line"
ADDRESS_CITY: Final = "address_city"
ADDRESS_STATE: Final = "address_state"
PINCODE: Final = "pincode"
PINCODE_TIER: Final = "pincode_tier"

# --- route -------------------------------------------------------------------
COURIER_PARTNER: Final = "courier_partner"

# --- label and bookkeeping ---------------------------------------------------
OUTCOME: Final = "outcome"
IS_RTO: Final = "is_rto"
SPLIT: Final = "split"

TARGET_COLUMN: Final = IS_RTO

RAW_COLUMNS: Final[tuple[str, ...]] = (
    ORDER_ID,
    MERCHANT_ID,
    CUSTOMER_HASH,
    ORDERED_AT,
    RESOLVED_AT,
    DAY_INDEX,
    PAYMENT_METHOD,
    ORDER_VALUE_INR,
    DISCOUNT_INR,
    DISCOUNT_DEPTH,
    ITEM_COUNT,
    CATEGORY,
    CART_EDITED,
    PRODUCT_PAGE_SECONDS,
    SESSIONS_BEFORE_PURCHASE,
    DEVICE_CLASS,
    HOUR_OF_DAY,
    TIME_TO_CHECKOUT_SECONDS,
    COD_AFTER_PREPAID_FAILURE,
    ADDRESS_LINE,
    ADDRESS_CITY,
    ADDRESS_STATE,
    PINCODE,
    PINCODE_TIER,
    COURIER_PARTNER,
    OUTCOME,
    IS_RTO,
    SPLIT,
)

# Columns that must never appear in a design matrix.
#
# ``outcome``/``is_rto`` are the label. ``resolved_at`` is the label's timestamp
# and knowing it is knowing the future. ``order_id``/``customer_hash`` are
# identity: a tree given them memorises individuals. ``address_line`` is raw
# text, admitted to the pipeline for quality extraction and dropped afterwards.
FORBIDDEN_IN_FEATURES: Final[frozenset[str]] = frozenset(
    {
        OUTCOME,
        IS_RTO,
        RESOLVED_AT,
        SPLIT,
        ORDER_ID,
        CUSTOMER_HASH,
        MERCHANT_ID,
        ADDRESS_LINE,
        PINCODE,  # raw pincode is refused as a feature; only smoothed aggregates
    }
)
