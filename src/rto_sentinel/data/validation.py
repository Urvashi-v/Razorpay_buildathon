"""Structural and semantic validation of a generated order table.

Runs immediately after generation, again before the database load, and again
after any load from disk. That repetition is cheap insurance: a silently
truncated parquet file, a column renamed during a refactor, or a timezone lost in
a round-trip should fail here, loudly, rather than surface later as a
mysteriously good model.

WHAT IS CHECKED, AND WHY EACH ONE EARNS ITS PLACE
-------------------------------------------------
Every check below corresponds to a way this dataset could be quietly wrong in a
direction that *flatters* a model. That is the selection criterion - a validator
that only catches obvious corruption is not worth running.

``nulls``
    A null in a non-nullable column usually means a partial write.
``duplicate identifiers``
    A duplicated ``order_id`` inflates a customer's history and can put the same
    row in two splits.
``negative or impossible values``
    Negative order values, negative discounts, discounts exceeding gross value,
    zero item counts.
``invalid categorical values``
    A typo creates a new level that a tree model will happily split on.
``impossible timestamps``
    Dispatch before order, attempt before dispatch, resolution before order.
    Any of these means the temporal reasoning underneath the whole project is
    broken.
``inconsistent outcome timestamps``
    A resolved order with no resolution time, a pending order that has one, a
    "delivered" row flagged as an RTO. These are the errors that would corrupt
    the label itself.
``label maturity``
    An immature row must carry a NULL label, not an optimistic "delivered", and a
    mature row must resolve inside the configured window.
``customer / order relationships``
    An order for a customer who does not exist, or placed before that customer
    existed.
``base-rate drift``
    The realised COD and prepaid RTO rates must sit within tolerance of the
    configured targets. A generator that has silently drifted off its anchors is
    no longer the benchmark it claims to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from rto_sentinel.data import schema as cols

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import GeneratorConfig


class DataValidationError(ValueError):
    """Raised when an order table violates the declared schema or an invariant."""


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating one table. Empty ``errors`` means it passed."""

    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            msg = f"order table failed validation ({len(self.errors)} problems):\n  - {joined}"
            raise DataValidationError(msg)

    def render(self) -> str:
        lines = [
            f"rows checked : {self.n_rows:,}",
            f"checks run   : {self.checks_run}",
            f"errors       : {len(self.errors)}",
            f"warnings     : {len(self.warnings)}",
        ]
        for error in self.errors:
            lines.append(f"  ERROR   {error}")
        for warning in self.warnings:
            lines.append(f"  WARNING {warning}")
        return "\n".join(lines)


def _count(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def validate_orders(
    frame: pd.DataFrame,
    *,
    config: GeneratorConfig | None = None,
    customers: pd.DataFrame | None = None,
    strict: bool = True,
) -> ValidationReport:
    """Check a generated order table against the declared schema and invariants.

    ``strict=False`` downgrades base-rate drift from an error to a warning, which
    is appropriate for deliberately small samples where the marginal is dominated
    by sampling noise. Nothing else is relaxed - a broken timestamp is broken at
    any sample size.
    """
    report = ValidationReport(n_rows=len(frame))

    # --- 1. columns ----------------------------------------------------------
    report.checks_run += 1
    missing = [column for column in cols.RAW_COLUMNS if column not in frame.columns]
    if missing:
        report.add_error(f"missing columns: {missing}")
        # Every later check indexes columns, so stop here rather than cascade.
        return report

    unexpected = [column for column in frame.columns if column not in cols.RAW_COLUMNS]
    if unexpected:
        report.add_warning(f"unexpected columns present: {unexpected}")

    if frame.empty:
        report.add_error("order table is empty")
        return report

    # --- 2. nulls ------------------------------------------------------------
    report.checks_run += 1
    for column in cols.NON_NULLABLE:
        nulls = _count(frame[column].isna())
        if nulls:
            report.add_error(f"{column}: {nulls:,} null values in a non-nullable column")

    # --- 3. duplicate identifiers -------------------------------------------
    report.checks_run += 1
    duplicate_orders = _count(frame[cols.ORDER_ID].duplicated())
    if duplicate_orders:
        report.add_error(f"{cols.ORDER_ID}: {duplicate_orders:,} duplicate identifiers")

    # --- 4. numeric ranges ---------------------------------------------------
    report.checks_run += 1
    negative_value = _count(frame[cols.ORDER_VALUE_INR] <= 0)
    if negative_value:
        report.add_error(f"{cols.ORDER_VALUE_INR}: {negative_value:,} rows at or below zero")

    negative_discount = _count(frame[cols.DISCOUNT_INR] < 0)
    if negative_discount:
        report.add_error(f"{cols.DISCOUNT_INR}: {negative_discount:,} negative values")

    bad_depth = _count((frame[cols.DISCOUNT_DEPTH] < 0) | (frame[cols.DISCOUNT_DEPTH] > 1))
    if bad_depth:
        report.add_error(f"{cols.DISCOUNT_DEPTH}: {bad_depth:,} rows outside [0, 1]")

    bad_items = _count(frame[cols.ITEM_COUNT] < 1)
    if bad_items:
        report.add_error(f"{cols.ITEM_COUNT}: {bad_items:,} rows with fewer than one item")

    bad_prior = _count(frame[cols.PRIOR_RTO_COUNT] > frame[cols.PRIOR_ORDER_COUNT])
    if bad_prior:
        report.add_error(
            f"{cols.PRIOR_RTO_COUNT}: {bad_prior:,} rows where prior RTOs exceed prior orders"
        )

    bad_hour = _count((frame[cols.HOUR_OF_DAY] < 0) | (frame[cols.HOUR_OF_DAY] > 23))
    if bad_hour:
        report.add_error(f"{cols.HOUR_OF_DAY}: {bad_hour:,} rows outside 0-23")

    # --- 5. categorical domains ---------------------------------------------
    report.checks_run += 1
    for column, permitted in cols.CATEGORICAL_DOMAINS.items():
        observed = set(frame[column].dropna().unique())
        invalid = observed - set(permitted)
        if invalid:
            report.add_error(f"{column}: invalid values {sorted(invalid)}")

    if config is not None:
        known_categories = {category.name for category in config.catalogue.categories}
        invalid_categories = set(frame[cols.CATEGORY].dropna().unique()) - known_categories
        if invalid_categories:
            report.add_error(
                f"{cols.CATEGORY}: values not in the catalogue {sorted(invalid_categories)}"
            )

        known_couriers = {courier.name for courier in config.couriers}
        invalid_couriers = set(frame[cols.COURIER_PARTNER].dropna().unique()) - known_couriers
        if invalid_couriers:
            report.add_error(
                f"{cols.COURIER_PARTNER}: values not in the courier list {sorted(invalid_couriers)}"
            )

    # --- 6. impossible timestamps -------------------------------------------
    report.checks_run += 1
    ordered = frame[cols.ORDERED_AT]
    dispatched = frame[cols.DISPATCHED_AT]
    attempted = frame[cols.FIRST_ATTEMPT_AT]
    resolved = frame[cols.RESOLVED_AT]

    dispatch_before_order = _count(dispatched.notna() & (dispatched < ordered))
    if dispatch_before_order:
        report.add_error(
            f"{cols.DISPATCHED_AT}: {dispatch_before_order:,} rows dispatched before the order"
        )

    attempt_before_dispatch = _count(
        attempted.notna() & dispatched.notna() & (attempted < dispatched)
    )
    if attempt_before_dispatch:
        report.add_error(
            f"{cols.FIRST_ATTEMPT_AT}: {attempt_before_dispatch:,} rows attempted before dispatch"
        )

    resolved_before_order = _count(resolved.notna() & (resolved < ordered))
    if resolved_before_order:
        report.add_error(
            f"{cols.RESOLVED_AT}: {resolved_before_order:,} rows resolved before the order"
        )

    resolved_before_attempt = _count(resolved.notna() & attempted.notna() & (resolved < attempted))
    if resolved_before_attempt:
        report.add_error(
            f"{cols.RESOLVED_AT}: {resolved_before_attempt:,} rows resolved before the "
            "first delivery attempt"
        )

    # --- 7. outcome / timestamp consistency ---------------------------------
    report.checks_run += 1
    is_pending = frame[cols.OUTCOME] == "pending"
    pending_with_resolution = _count(is_pending & resolved.notna())
    if pending_with_resolution:
        report.add_error(f"{pending_with_resolution:,} pending orders carry a resolution timestamp")

    terminal_without_resolution = _count(~is_pending & resolved.isna())
    if terminal_without_resolution:
        report.add_error(
            f"{terminal_without_resolution:,} terminal orders have no resolution timestamp"
        )

    pending_with_label = _count(is_pending & frame[cols.IS_RTO].notna())
    if pending_with_label:
        report.add_error(
            f"{pending_with_label:,} pending orders carry a non-null label. An order whose "
            "outcome is unknown must not be labelled."
        )

    labelled = frame[cols.IS_RTO].notna()
    claims_rto = frame[cols.IS_RTO].astype("boolean").fillna(False).astype(bool)
    rto_mismatch = _count(labelled & (claims_rto != (frame[cols.OUTCOME] == "rto")))
    if rto_mismatch:
        report.add_error(f"{rto_mismatch:,} rows where is_rto disagrees with outcome")

    # --- 8. label maturity ---------------------------------------------------
    report.checks_run += 1
    mature = frame[cols.IS_MATURE].astype(bool)
    immature_labelled = _count(~mature & frame[cols.IS_RTO].notna())
    if immature_labelled:
        report.add_error(
            f"{immature_labelled:,} immature orders carry a label. Orders whose terminal state "
            "is not yet known must not be optimistically labelled."
        )

    mature_unlabelled = _count(mature & frame[cols.IS_RTO].isna())
    if mature_unlabelled:
        report.add_error(f"{mature_unlabelled:,} mature orders have no label")

    mature_without_maturity_days = _count(mature & frame[cols.MATURITY_DAYS].isna())
    if mature_without_maturity_days:
        report.add_error(
            f"{mature_without_maturity_days:,} mature orders have no maturity_days value"
        )

    if config is not None:
        limit = config.label_maturity.max_resolution_days
        overlong = _count(frame[cols.MATURITY_DAYS] > limit + 1e-9)
        if overlong:
            report.add_error(
                f"{overlong:,} orders resolve later than the configured maximum of {limit} days"
            )

    # --- 9. customer / order relationships ----------------------------------
    report.checks_run += 1
    if customers is not None and not customers.empty:
        known = set(customers[cols.CUSTOMER_HASH])
        orphaned = _count(~frame[cols.CUSTOMER_HASH].isin(known))
        if orphaned:
            report.add_error(f"{orphaned:,} orders reference a customer that does not exist")

        if "signup_at" in customers.columns:
            signup = customers.set_index(cols.CUSTOMER_HASH)["signup_at"]
            mapped = frame[cols.CUSTOMER_HASH].map(signup)
            before_signup = _count(mapped.notna() & (ordered < mapped))
            if before_signup:
                report.add_warning(
                    f"{before_signup:,} orders precede their customer's signup timestamp"
                )

    # --- 10. base-rate drift -------------------------------------------------
    report.checks_run += 1
    if config is not None:
        mature_frame = frame[mature]
        tolerance = config.base_rates.marginal_tolerance
        for label, subset, target in (
            (
                "COD",
                mature_frame[mature_frame[cols.IS_COD].astype(bool)],
                config.base_rates.rto_given_cod,
            ),
            (
                "prepaid",
                mature_frame[~mature_frame[cols.IS_COD].astype(bool)],
                config.base_rates.rto_given_prepaid,
            ),
        ):
            if subset.empty:
                report.add_warning(f"no mature {label} orders to check the base rate against")
                continue
            realised = float(subset[cols.IS_RTO].astype(float).mean())
            drift = abs(realised - target)
            if drift > tolerance:
                message = (
                    f"{label} RTO rate is {realised:.4f}, target {target:.4f}, "
                    f"drift {drift:.4f} exceeds tolerance {tolerance:.4f}"
                )
                if strict:
                    report.add_error(message)
                else:
                    report.add_warning(message)

    return report


def validate_delivery_events(events: pd.DataFrame, orders: pd.DataFrame) -> ValidationReport:
    """Check the event trail is complete, ordered, and consistent with the orders."""
    report = ValidationReport(n_rows=len(events))

    report.checks_run += 1
    required = {"order_id", "sequence", "event_type", "occurred_at"}
    missing = required - set(events.columns)
    if missing:
        report.add_error(f"delivery_events is missing columns: {sorted(missing)}")
        return report

    report.checks_run += 1
    orphaned = _count(~events["order_id"].isin(set(orders[cols.ORDER_ID])))
    if orphaned:
        report.add_error(f"{orphaned:,} delivery events reference an unknown order")

    report.checks_run += 1
    duplicated = _count(events.duplicated(subset=["order_id", "sequence"]))
    if duplicated:
        report.add_error(f"{duplicated:,} duplicate (order_id, sequence) event pairs")

    report.checks_run += 1
    # Events must be non-decreasing in time within an order. A trail that goes
    # backwards makes any reconstruction of "what was known when" meaningless.
    ordered_events = events.sort_values(["order_id", "sequence"], kind="stable")
    grouped = ordered_events.groupby("order_id", sort=False)["occurred_at"]
    out_of_order = int((grouped.diff() < pd.Timedelta(0)).sum())
    if out_of_order:
        report.add_error(f"{out_of_order:,} delivery events occur before the previous event")

    report.checks_run += 1
    without_placement = set(orders[cols.ORDER_ID]) - set(
        events.loc[events["event_type"] == "order_placed", "order_id"]
    )
    if without_placement:
        report.add_error(f"{len(without_placement):,} orders have no order_placed event")

    return report
