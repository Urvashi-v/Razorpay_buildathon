"""The canonical column schema for the generated order dataset.

Declared once here so the generator, the validator, the as-of join, the splitter,
the database loader and the (Phase 3) feature pipeline all agree on what a row
looks like. A column not in this list does not exist as far as the pipeline is
concerned.

THREE COLUMN SETS, AND WHY THEY ARE SEPARATE
--------------------------------------------
``RAW_COLUMNS``
    Everything the generator emits. This is the benchmark dataset.

``ORDER_TIME_COLUMNS``
    The subset knowable *at the moment the order is placed*. Every historical
    aggregate in here was computed during simulation from orders that had already
    **resolved** before this order's ``ordered_at`` - so the set is leak-free by
    construction rather than by later filtering.

``FORBIDDEN_IN_FEATURES``
    Columns that must never reach a design matrix: the label, its timestamps,
    identity columns, and the simulator's own latent variables.

``tests/leakage/`` checks the relationships between these three sets, so the
separation is mechanical rather than a convention.
"""

from __future__ import annotations

from typing import Final

# --- identity and time -------------------------------------------------------
ORDER_ID: Final = "order_id"
MERCHANT_ID: Final = "merchant_id"
CUSTOMER_HASH: Final = "customer_hash"
ADDRESS_FINGERPRINT: Final = "address_fingerprint"
ORDERED_AT: Final = "ordered_at"
DISPATCHED_AT: Final = "dispatched_at"
FIRST_ATTEMPT_AT: Final = "first_attempt_at"
RESOLVED_AT: Final = "resolved_at"
DAY_INDEX: Final = "day_index"

# --- order shape -------------------------------------------------------------
PAYMENT_METHOD: Final = "payment_method"
IS_COD: Final = "is_cod"
ORDER_VALUE_INR: Final = "order_value_inr"
DISCOUNT_INR: Final = "discount_inr"
DISCOUNT_DEPTH: Final = "discount_depth"
ITEM_COUNT: Final = "item_count"
CATEGORY: Final = "category"
CART_EDITED: Final = "cart_edited"

# --- session and timing ------------------------------------------------------
PRODUCT_PAGE_SECONDS: Final = "product_page_seconds"
SESSIONS_BEFORE_PURCHASE: Final = "sessions_before_purchase"
DEVICE_CLASS: Final = "device_class"
HOUR_OF_DAY: Final = "hour_of_day"
DAY_OF_WEEK: Final = "day_of_week"
IS_LATE_NIGHT: Final = "is_late_night"
IS_SALE_DAY: Final = "is_sale_day"
TIME_TO_CHECKOUT_SECONDS: Final = "time_to_checkout_seconds"
COD_AFTER_PREPAID_FAILURE: Final = "cod_after_prepaid_failure"

# --- address (raw text kept for audit; observable signals for features) -------
ADDRESS_LINE: Final = "address_line"
ADDRESS_CITY: Final = "address_city"
ADDRESS_STATE: Final = "address_state"
PINCODE: Final = "pincode"
PINCODE_TIER: Final = "pincode_tier"

ADDR_TOKEN_COUNT: Final = "addr_token_count"  # noqa: S105  (a column name, not a secret)
ADDR_HAS_HOUSE_NUMBER: Final = "addr_has_house_number"
ADDR_HAS_FLOOR_NUMBER: Final = "addr_has_floor_number"
ADDR_HAS_LANDMARK: Final = "addr_has_landmark"
ADDR_PINCODE_CITY_CONSISTENT: Final = "addr_pincode_city_consistent"
ADDR_ALLCAPS_RATIO: Final = "addr_allcaps_ratio"
ADDR_GIBBERISH_RATIO: Final = "addr_gibberish_ratio"

# --- route -------------------------------------------------------------------
COURIER_PARTNER: Final = "courier_partner"

# --- customer history, computed AS-OF ordered_at -----------------------------
PRIOR_ORDER_COUNT: Final = "prior_order_count"
PRIOR_RTO_COUNT: Final = "prior_rto_count"
PRIOR_RTO_RATE: Final = "prior_rto_rate"
DAYS_SINCE_LAST_ORDER: Final = "days_since_last_order"
PREPAID_TO_COD_RATIO: Final = "prepaid_to_cod_ratio"
MEAN_RESOLUTION_DAYS: Final = "mean_resolution_days"
IS_NEW_CUSTOMER: Final = "is_new_customer"

# --- label and maturity ------------------------------------------------------
OUTCOME: Final = "outcome"
IS_RTO: Final = "is_rto"
MATURITY_DAYS: Final = "maturity_days"
IS_MATURE: Final = "is_mature"
SPLIT: Final = "split"

# --- simulator ground truth (NEVER a feature) --------------------------------
TRUE_RTO_PROBABILITY: Final = "true_rto_probability"
LATENT_LOGIT: Final = "latent_logit"

TARGET_COLUMN: Final = IS_RTO


ORDER_TIME_COLUMNS: Final[tuple[str, ...]] = (
    # order shape
    PAYMENT_METHOD,
    IS_COD,
    ORDER_VALUE_INR,
    DISCOUNT_INR,
    DISCOUNT_DEPTH,
    ITEM_COUNT,
    CATEGORY,
    CART_EDITED,
    # session and timing
    PRODUCT_PAGE_SECONDS,
    SESSIONS_BEFORE_PURCHASE,
    DEVICE_CLASS,
    HOUR_OF_DAY,
    DAY_OF_WEEK,
    IS_LATE_NIGHT,
    IS_SALE_DAY,
    TIME_TO_CHECKOUT_SECONDS,
    COD_AFTER_PREPAID_FAILURE,
    # address quality (observable from the text the customer typed)
    ADDR_TOKEN_COUNT,
    ADDR_HAS_HOUSE_NUMBER,
    ADDR_HAS_FLOOR_NUMBER,
    ADDR_HAS_LANDMARK,
    ADDR_PINCODE_CITY_CONSISTENT,
    ADDR_ALLCAPS_RATIO,
    ADDR_GIBBERISH_RATIO,
    # geography and route
    PINCODE_TIER,
    COURIER_PARTNER,
    # customer history, as-of ordered_at
    PRIOR_ORDER_COUNT,
    PRIOR_RTO_COUNT,
    PRIOR_RTO_RATE,
    DAYS_SINCE_LAST_ORDER,
    PREPAID_TO_COD_RATIO,
    MEAN_RESOLUTION_DAYS,
    IS_NEW_CUSTOMER,
)

RAW_COLUMNS: Final[tuple[str, ...]] = (
    ORDER_ID,
    MERCHANT_ID,
    CUSTOMER_HASH,
    ADDRESS_FINGERPRINT,
    ORDERED_AT,
    DISPATCHED_AT,
    FIRST_ATTEMPT_AT,
    RESOLVED_AT,
    DAY_INDEX,
    *ORDER_TIME_COLUMNS,
    ADDRESS_LINE,
    ADDRESS_CITY,
    ADDRESS_STATE,
    PINCODE,
    OUTCOME,
    IS_RTO,
    MATURITY_DAYS,
    IS_MATURE,
    SPLIT,
)

# Columns that must never appear in a design matrix.
#
# ``outcome``/``is_rto`` are the label. ``resolved_at`` and the other post-order
# timestamps are the label's timing - knowing when an order resolved is close to
# knowing how it resolved. ``order_id``/``customer_hash`` are identity: a tree
# given them memorises individuals. ``address_line`` is raw text, admitted for
# quality extraction and audit, then dropped. ``pincode`` is refused as a raw
# categorical (only smoothed aggregates are permitted). The last two are the
# simulator's own latent variables, which exist for calibration diagnostics and
# would be perfect leakage if fed to a model.
FORBIDDEN_IN_FEATURES: Final[frozenset[str]] = frozenset(
    {
        OUTCOME,
        IS_RTO,
        RESOLVED_AT,
        DISPATCHED_AT,
        FIRST_ATTEMPT_AT,
        MATURITY_DAYS,
        IS_MATURE,
        SPLIT,
        ORDER_ID,
        CUSTOMER_HASH,
        MERCHANT_ID,
        ADDRESS_FINGERPRINT,
        ADDRESS_LINE,
        PINCODE,
        TRUE_RTO_PROBABILITY,
        LATENT_LOGIT,
    }
)

#: The simulator's own ground truth. These live in a SEPARATE frame and a
#: separate table, never in ``RAW_COLUMNS`` - which is precisely why they are
#: listed here as well: ``FORBIDDEN_IN_FEATURES`` names them, and a test asserts
#: every forbidden column is a real column somewhere, so this is what stops that
#: assertion from being satisfied by a typo.
LATENT_COLUMNS: Final[tuple[str, ...]] = (
    TRUE_RTO_PROBABILITY,
    LATENT_LOGIT,
    "customer_reliability",
    "latent_address_quality",
    "address_deliverability",
    "pincode_effect",
    "label_flipped",
)

#: Categorical columns and their permitted values. The validator rejects anything
#: outside these, so a renamed category or a typo in config fails loudly instead
#: of silently creating a new level the model treats as its own signal.
CATEGORICAL_DOMAINS: Final[dict[str, frozenset[str]]] = {
    PAYMENT_METHOD: frozenset({"cod", "prepaid"}),
    DEVICE_CLASS: frozenset({"mobile_web", "mobile_app", "desktop", "unknown"}),
    PINCODE_TIER: frozenset({"tier_1", "tier_2", "tier_3"}),
    OUTCOME: frozenset({"pending", "delivered", "rto", "cancelled"}),
    SPLIT: frozenset(
        {"train", "validation", "test", "excluded_immature", "excluded_group_protocol"}
    ),
}

#: Columns that may never be null in a generated dataset.
NON_NULLABLE: Final[tuple[str, ...]] = (
    ORDER_ID,
    MERCHANT_ID,
    CUSTOMER_HASH,
    ADDRESS_FINGERPRINT,
    ORDERED_AT,
    DAY_INDEX,
    PAYMENT_METHOD,
    ORDER_VALUE_INR,
    DISCOUNT_INR,
    ITEM_COUNT,
    CATEGORY,
    PINCODE,
    PINCODE_TIER,
    COURIER_PARTNER,
    OUTCOME,
    IS_MATURE,
    SPLIT,
)
