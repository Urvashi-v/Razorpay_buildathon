"""The synthetic order generator - a controlled RTO benchmark, not ground truth.

SPEC section 03 (data strategy) and section 09 (defense-only compliance).

READ THIS BEFORE USING ANY NUMBER MEASURED ON THIS DATA
=======================================================
The labels this module produces are **simulated outcomes of a documented
process**, not observations of the world. They are calibrated so their marginal
rates match published Indian RTO figures, which makes the benchmark realistic in
aggregate; it does not make any individual label true, and it does not make a
metric measured here a claim about production performance.

The full write-up of the process, its assumptions, and what it can and cannot
demonstrate is in ``docs/simulator.md``.

HOW THE SIMULATION WORKS, IN SHORT
==================================
1. **Populations.** Customers, pincodes and couriers are drawn once, each with
   latent traits: customer reliability and writing quality, a per-pincode random
   effect, a per-courier lane quality.
2. **Order stream.** Order timestamps are sampled across the horizon using
   hour-of-day weights, a weekend uplift and a sale-day calendar, then sorted.
   Customers are assigned by activity weight, so each customer's orders arrive in
   chronological order.
3. **Per order, in time order.** Order attributes are drawn, then the customer's
   history features are computed **as-of that instant** from orders that had
   already *resolved* - the mechanism that makes the dataset leak-free by
   construction rather than by later filtering.
4. **Latent logit.** A linear combination of the documented causal drivers, plus
   a per-order Gaussian shock. Several drivers are latent and never exposed.
5. **Calibration.** A per-payment-method intercept is found by a fixed-point
   iteration over the whole pass, so the realised marginal matches the configured
   base rate. The iteration is needed rather than a closed-form solve because the
   simulation has a real feedback loop - a customer's prior RTO rate is an input
   to their next order's risk - and that loop is worth keeping.
6. **Sampling.** The label is a Bernoulli draw, with a small symmetric flip rate
   standing in for courier miscoding.
7. **Fulfilment timeline.** Dispatch, first attempt and resolution timestamps
   follow, with RTOs resolving later than deliveries.
8. **Maturity.** Orders whose resolution falls beyond the horizon get outcome
   ``pending``, a null label, and ``excluded_immature`` - never an optimistic
   "delivered".

WHY THE TASK IS NOT REVERSE-ENGINEERABLE
========================================
Three drivers are unobservable (customer reliability, the per-pincode effect,
and for address quality the model sees only rendered text signals rather than the
latent). Add the per-order Gaussian shock, the Bernoulli draw and the label flip
rate, and there is a genuine Bayes-optimal ceiling well below perfect separation.
A model that scores near-perfectly on this data has found a bug, not a signal.

WHAT THIS DOES NOT PRODUCE
==========================
Working payment credentials. Valid identity documents. Deliverable addresses.
Real customer records. Anything usable outside this repository's own evaluation
harness. Customer identifiers are digests of a synthetic index; pincodes are
synthetic six-digit identifiers, not a map of real Indian postcodes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.data.address import (
    CITY_STEMS,
    AddressSignals,
    RenderedAddress,
    observable_signals,
    render_address,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import GeneratorConfig

#: Behavioural versions of the generative process that this module implements.
#: A dataset generated under one version is not comparable to one generated under
#: another, which is why the version is recorded on every dataset run and an
#: unknown value is refused rather than silently accepted.
SUPPORTED_GENERATOR_VERSIONS: frozenset[str] = frozenset({"1.0.0"})

DEFAULT_MERCHANT_ID = "M-DEMO-001"

#: Upper bound on the base-rate fixed-point iteration. In practice it converges
#: in two or three passes; the cap exists so a pathological configuration fails
#: with a reported gap rather than spinning.
MAX_CALIBRATION_ITERATIONS: int = 8


class UnsupportedGeneratorVersionError(ValueError):
    """Raised when a requested generator version has no implementation here."""


@dataclass(frozen=True, slots=True)
class GeneratorParams:
    """Everything that must be recorded to reproduce a dataset exactly."""

    seed: int
    generator_version: str
    n_customers: int
    n_orders: int
    start_date: datetime
    end_date: datetime

    @property
    def days(self) -> int:
        """Inclusive horizon length in days."""
        return (self.end_date.date() - self.start_date.date()).days + 1

    def __post_init__(self) -> None:
        if self.generator_version not in SUPPORTED_GENERATOR_VERSIONS:
            known = ", ".join(sorted(SUPPORTED_GENERATOR_VERSIONS))
            msg = (
                f"generator version {self.generator_version!r} is not implemented; "
                f"known versions: {known}"
            )
            raise UnsupportedGeneratorVersionError(msg)
        if self.end_date < self.start_date:
            msg = "end_date must not precede start_date"
            raise ValueError(msg)
        if self.n_customers <= 0 or self.n_orders <= 0:
            msg = "n_customers and n_orders must both be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DatasetRunMetadata:
    """Provenance for one generated dataset.

    Recorded on the dataset artefact and in the ``dataset_runs`` table. Everything
    needed to regenerate the identical dataset is here: the seed, the generator
    version, the parameters, and a fingerprint of the configuration.
    """

    run_id: str
    generator_version: str
    seed: int
    config_fingerprint: str
    config_snapshot: dict[str, Any]
    n_customers: int
    n_orders: int
    start_date: datetime
    end_date: datetime
    created_at: datetime
    realised_rto_rate_cod: float
    realised_rto_rate_prepaid: float
    realised_cod_share: float
    n_mature: int
    n_immature: int

    def to_json(self) -> str:
        payload = {
            "run_id": self.run_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "config_fingerprint": self.config_fingerprint,
            "config_snapshot": self.config_snapshot,
            "n_customers": self.n_customers,
            "n_orders": self.n_orders,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "realised_rto_rate_cod": self.realised_rto_rate_cod,
            "realised_rto_rate_prepaid": self.realised_rto_rate_prepaid,
            "realised_cod_share": self.realised_cod_share,
            "n_mature": self.n_mature,
            "n_immature": self.n_immature,
            "data_provenance": (
                "Synthetic benchmark data produced by rto_sentinel.data.generator. "
                "Labels are simulated outcomes of a documented process, not real-world "
                "ground truth."
            ),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A generated dataset, as separate frames matching the database tables.

    ``orders`` is the ML-facing benchmark table. ``latents`` holds the simulator's
    own ground truth (the true per-order probability), which exists for
    calibration diagnostics and is excluded from every feature path - see
    ``schema.FORBIDDEN_IN_FEATURES``.
    """

    customers: pd.DataFrame
    addresses: pd.DataFrame
    orders: pd.DataFrame
    delivery_events: pd.DataFrame
    latents: pd.DataFrame
    metadata: DatasetRunMetadata

    def summary(self) -> dict[str, float | int | str]:
        orders = self.orders
        mature = orders[orders[cols.IS_MATURE]]
        return {
            "run_id": self.metadata.run_id,
            "generator_version": self.metadata.generator_version,
            "seed": self.metadata.seed,
            "customers": len(self.customers),
            "addresses": len(self.addresses),
            "orders": len(orders),
            "delivery_events": len(self.delivery_events),
            "mature_orders": int(mature.shape[0]),
            "immature_orders": int(orders.shape[0] - mature.shape[0]),
            "cod_share": float(orders[cols.IS_COD].mean()),
            "rto_rate_overall": float(mature[cols.IS_RTO].mean()) if len(mature) else float("nan"),
            "rto_rate_cod": self.metadata.realised_rto_rate_cod,
            "rto_rate_prepaid": self.metadata.realised_rto_rate_prepaid,
        }


class OrderGenerator(Protocol):
    """Anything that can produce a labelled order table."""

    def generate(self, config: GeneratorConfig, params: GeneratorParams) -> GenerationResult: ...


# ---------------------------------------------------------------------------
# Population draws and per-customer state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CustomerState:
    """Mutable per-customer history used for as-of feature computation.

    ``resolved`` holds only orders whose terminal state is already known. An order
    enters this list at its *resolution* time, not at its order time, which is the
    whole point: on day 42 a customer's history contains only what had actually
    come back by day 42.
    """

    index: int
    customer_hash: str
    reliability: float
    address_quality: float
    prepaid_affinity: float
    pincode_index: int
    signup_at: datetime
    # (resolved_at, is_rto, is_cod, resolution_days, order_value)
    history: list[tuple[datetime, bool, bool, float, float]] = field(default_factory=list)
    order_values: list[float] = field(default_factory=list)
    last_order_at: datetime | None = None
    order_count: int = 0


def _customer_hash(index: int, seed: int) -> str:
    """Opaque, reproducible customer identifier.

    A digest of a synthetic index and the run seed. There is no pre-image to
    recover because there is no real identity behind it - this is a stable label
    for a simulated entity, nothing more.
    """
    return hashlib.sha256(f"cust:{seed}:{index}".encode()).hexdigest()[:32]


def _pincode(index: int) -> str:
    """Synthetic six-digit identifier. Not a real Indian postcode."""
    return f"{100000 + (index * 37) % 800000:06d}"


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Turn a non-negative weight vector into a probability vector."""
    total = float(weights.sum())
    if total <= 0:
        msg = "weight vector must have a positive sum"
        raise ValueError(msg)
    normalised: np.ndarray = weights / total
    return normalised


class ConfiguredOrderGenerator:
    """The project's generator, driven entirely by ``config/generator.yaml``.

    Deterministic given ``(config, params)``. Every random draw comes from a
    single ``numpy.random.Generator`` seeded from ``params.seed`` and consumed in
    a fixed order, so two runs with the same inputs produce identical frames.
    ``tests/unit/test_generator_reproducibility.py`` asserts this.
    """

    def generate(self, config: GeneratorConfig, params: GeneratorParams) -> GenerationResult:
        if params.generator_version != "1.0.0":  # pragma: no cover - guarded in params
            msg = f"no implementation for generator version {params.generator_version!r}"
            raise UnsupportedGeneratorVersionError(msg)
        return _simulate_v1(config, params)


# ---------------------------------------------------------------------------
# Version 1.0.0 of the generative process
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Draws:
    """Every random number the simulation needs, drawn once, up front.

    Pre-drawing matters for a specific reason. The base-rate intercepts are solved
    by running the sequential simulation repeatedly (see :func:`_simulate_v1`),
    and if the passes consumed randomness they would consume *different*
    randomness each time - the customer histories would change between iterations
    and the fixed point would not exist. With the draws frozen here, a pass is a
    pure function of the intercepts, so the iteration converges and the result is
    reproducible.
    """

    # populations
    reliability: np.ndarray
    writing_quality: np.ndarray
    prepaid_affinity: np.ndarray
    customer_pincode_idx: np.ndarray
    signup_lag_hours: np.ndarray
    activity_probs: np.ndarray
    pincode_tier_idx: np.ndarray
    pincode_effect: np.ndarray
    pincode_city_idx: np.ndarray
    # per-order, already in chronological order
    customer_choice: np.ndarray
    order_days: np.ndarray
    order_hours: np.ndarray
    timestamps: np.ndarray
    category_idx: np.ndarray
    order_values: np.ndarray
    item_counts: np.ndarray
    base_discount: np.ndarray
    courier_idx: np.ndarray
    device_idx: np.ndarray
    product_page_seconds: np.ndarray
    sessions_before: np.ndarray
    time_to_checkout: np.ndarray
    cart_edited: np.ndarray
    cod_flag: np.ndarray
    prepaid_failure_roll: np.ndarray
    logit_noise: np.ndarray
    label_draw: np.ndarray
    flip_draw: np.ndarray
    dispatch_hours: np.ndarray
    transit_days: np.ndarray
    rto_extra: np.ndarray
    cancel_draw: np.ndarray
    # rendered once, because address text depends on nothing the intercepts touch
    addresses: list[RenderedAddress]
    address_signals: list[AddressSignals]
    latent_address_quality: np.ndarray
    address_deliverability: np.ndarray


def _draw_everything(config: GeneratorConfig, params: GeneratorParams) -> _Draws:
    """Draw the whole simulation's randomness, in one fixed order."""
    rng = np.random.default_rng(params.seed)
    tiers = ("tier_1", "tier_2", "tier_3")
    n_days = params.days
    n_orders = params.n_orders
    n_customers = params.n_customers

    # --- pincode population --------------------------------------------------
    n_pincodes = max(8, min(config.geography.n_pincodes, max(8, n_customers // 4)))
    tier_probs = _normalise(
        np.array([config.geography.tier_shares[tier] for tier in tiers], dtype=float)
    )
    pincode_tier_idx = rng.choice(len(tiers), size=n_pincodes, p=tier_probs)
    pincode_effect = rng.normal(0.0, config.noise.pincode_effect_sigma, size=n_pincodes)
    pincode_city_idx = rng.integers(0, len(CITY_STEMS), size=n_pincodes)

    # --- customer population -------------------------------------------------
    reliability = rng.beta(
        config.customers.reliability_beta.a, config.customers.reliability_beta.b, size=n_customers
    )
    writing_quality = rng.beta(
        config.customers.address_quality_beta.a,
        config.customers.address_quality_beta.b,
        size=n_customers,
    )
    prepaid_affinity = rng.beta(
        config.customers.prepaid_affinity_beta.a,
        config.customers.prepaid_affinity_beta.b,
        size=n_customers,
    )
    customer_pincode_idx = rng.integers(0, n_pincodes, size=n_customers)
    # Account age at first purchase. In Indian e-commerce an account is usually
    # created at or just before the first order, so signup is anchored to that
    # order rather than drawn independently - an independent draw produces
    # orders that precede their own customer, which is not a thing.
    signup_lag_hours = rng.integers(0, 72, size=n_customers)

    # A long tail, so most customers order once and a few order often. A uniform
    # order count would give everyone a history and quietly delete the cold-start
    # cohort the specification cares about.
    #
    # The raw Pareto head is clipped: unclipped, a single customer takes a
    # double-digit share of the merchant's book, which no real store sees and
    # which would let one person's latent reliability dominate the whole dataset.
    activity = rng.pareto(config.customers.orders_per_customer_alpha, size=n_customers) + 1.0
    ceiling = float(np.quantile(activity, config.customers.activity_clip_quantile))
    activity = np.minimum(activity, ceiling)
    activity_probs = _normalise(activity)

    # --- order timeline ------------------------------------------------------
    day_weights = np.ones(n_days, dtype=float)
    for day_offset in range(n_days):
        if (params.start_date + timedelta(days=day_offset)).weekday() >= 5:
            day_weights[day_offset] *= config.timing.weekend_uplift
    for sale_day in config.timing.sale_days:
        if 0 <= sale_day < n_days:
            day_weights[sale_day] *= config.timing.sale_day_volume_multiplier
    day_probs = _normalise(day_weights)
    hour_probs = _normalise(np.array(config.timing.hour_weights, dtype=float))

    raw_days = rng.choice(n_days, size=n_orders, p=day_probs)
    raw_hours = rng.choice(24, size=n_orders, p=hour_probs)
    raw_minutes = rng.integers(0, 60, size=n_orders)
    raw_seconds = rng.integers(0, 60, size=n_orders)
    raw_timestamps = (
        np.array(
            [params.start_date.timestamp()] * n_orders,
            dtype=float,
        )
        + raw_days * 86400.0
        + raw_hours * 3600.0
        + raw_minutes * 60.0
        + raw_seconds
    )
    # Sorting here is what makes every customer's orders arrive chronologically in
    # the sequential pass, which is what makes the as-of computation correct
    # without any later sorting or filtering.
    order = np.argsort(raw_timestamps, kind="stable")

    raw_customer_choice = rng.choice(n_customers, size=n_orders, p=activity_probs)

    category_probs = _normalise(
        np.array([category.share for category in config.catalogue.categories], dtype=float)
    )
    raw_category_idx = rng.choice(len(config.catalogue.categories), size=n_orders, p=category_probs)

    raw_values = np.clip(
        rng.lognormal(
            config.catalogue.order_value.mu, config.catalogue.order_value.sigma, n_orders
        ),
        config.catalogue.order_value.min,
        config.catalogue.order_value.max,
    )
    raw_items = 1 + rng.poisson(max(config.catalogue.items_per_order_lambda - 1.0, 0.0), n_orders)
    raw_discount = rng.beta(1.6, 6.0, size=n_orders)

    courier_probs = _normalise(
        np.array([courier.share for courier in config.couriers], dtype=float)
    )
    raw_courier_idx = rng.choice(len(config.couriers), size=n_orders, p=courier_probs)
    raw_device_idx = rng.choice(3, size=n_orders, p=np.array([0.52, 0.33, 0.15]))

    raw_page_seconds = np.round(rng.gamma(2.0, 55.0, size=n_orders), 1)
    raw_sessions = 1 + rng.poisson(0.7, size=n_orders)
    raw_checkout = np.round(rng.gamma(1.6, 90.0, size=n_orders), 1)
    raw_cart_edited = rng.random(n_orders) < 0.28

    # --- payment method ------------------------------------------------------
    # Centred on the observed mean affinity so the realised COD share matches the
    # configured one. Centring on the distribution's theoretical mean would leave
    # a systematic offset at small sample sizes.
    affinity_by_order = prepaid_affinity[raw_customer_choice]
    cod_probability = np.clip(
        config.payment.cod_share + 0.55 * (float(affinity_by_order.mean()) - affinity_by_order),
        0.02,
        0.98,
    )
    raw_cod_flag = rng.random(n_orders) < cod_probability
    raw_prepaid_failure = rng.random(n_orders)

    raw_logit_noise = rng.normal(0.0, config.noise.logit_sigma, size=n_orders)
    raw_label_draw = rng.random(n_orders)
    raw_flip_draw = rng.random(n_orders)

    raw_dispatch = rng.integers(
        config.fulfilment.dispatch_lag_hours.min,
        config.fulfilment.dispatch_lag_hours.max + 1,
        size=n_orders,
    )
    raw_transit = rng.integers(
        config.fulfilment.transit_days.min, config.fulfilment.transit_days.max + 1, size=n_orders
    )
    raw_rto_extra = rng.integers(
        config.fulfilment.rto_extra_days.min,
        config.fulfilment.rto_extra_days.max + 1,
        size=n_orders,
    )
    raw_cancel = rng.random(n_orders)

    # --- addresses -----------------------------------------------------------
    # Rendered here rather than inside the pass because address text depends on
    # latent writing quality and tier, and on nothing the intercepts touch.
    #
    # Each customer has ONE home address, reused across their orders, because that
    # is what real order books look like and because repeat structure is what
    # makes an address dimension worth having. A small share of orders go to an
    # alternate address - a gift, an office - which is drawn fresh.
    customer_choice = raw_customer_choice[order]
    address_noise = rng.normal(0.0, 0.12, size=n_customers)

    home_quality = np.empty(n_customers, dtype=float)
    home_addresses: list[RenderedAddress] = []
    home_signals: list[AddressSignals] = []
    home_inconsistency = rng.random(n_customers)
    for customer_index in range(n_customers):
        pincode_index = int(customer_pincode_idx[customer_index])
        tier = tiers[int(pincode_tier_idx[pincode_index])]
        quality = float(
            np.clip(
                writing_quality[customer_index]
                - config.address_quality.tier_degradation[tier]
                + address_noise[customer_index],
                0.0,
                1.0,
            )
        )
        home_quality[customer_index] = quality
        expected_city = CITY_STEMS[int(pincode_city_idx[pincode_index])]
        consistent = (
            home_inconsistency[customer_index]
            >= config.address_quality.pincode_city_inconsistency[tier]
        )
        rendered = render_address(
            rng,
            quality=quality,
            city=expected_city,
            pincode=_pincode(pincode_index),
            consistent_pincode=consistent,
        )
        home_addresses.append(rendered)
        home_signals.append(
            observable_signals(rendered.line, city=rendered.city, expected_city=expected_city)
        )

    alternate_roll = rng.random(n_orders)
    alternate_inconsistency = rng.random(n_orders)
    alternate_noise = rng.normal(0.0, 0.12, size=n_orders)

    latent_quality = np.empty(n_orders, dtype=float)
    # Deliverability is the quantity that actually drives the outcome: writing
    # quality, minus a penalty when the typed city disagrees with the pincode.
    # Kept separate from `latent_quality` (which drives *rendering*) so that the
    # observable text signals stay a noisy proxy for risk rather than a lossless
    # encoding of it.
    deliverability = np.empty(n_orders, dtype=float)
    penalty = config.address_quality.pincode_city_inconsistency_penalty
    addresses: list[RenderedAddress] = []
    signals: list[AddressSignals] = []

    for position in range(n_orders):
        customer_index = int(customer_choice[position])
        if alternate_roll[position] >= config.address_quality.alternate_address_rate:
            latent_quality[position] = home_quality[customer_index]
            home = home_signals[customer_index]
            deliverability[position] = float(
                np.clip(
                    home_quality[customer_index]
                    - (0.0 if home.pincode_city_consistent else penalty),
                    0.0,
                    1.0,
                )
            )
            addresses.append(home_addresses[customer_index])
            signals.append(home)
            continue

        pincode_index = int(customer_pincode_idx[customer_index])
        tier = tiers[int(pincode_tier_idx[pincode_index])]
        quality = float(
            np.clip(
                writing_quality[customer_index]
                - config.address_quality.tier_degradation[tier]
                + alternate_noise[position],
                0.0,
                1.0,
            )
        )
        latent_quality[position] = quality
        expected_city = CITY_STEMS[int(pincode_city_idx[pincode_index])]
        consistent = (
            alternate_inconsistency[position]
            >= config.address_quality.pincode_city_inconsistency[tier]
        )
        rendered = render_address(
            rng,
            quality=quality,
            city=expected_city,
            pincode=_pincode(pincode_index),
            consistent_pincode=consistent,
        )
        rendered_signals = observable_signals(
            rendered.line, city=rendered.city, expected_city=expected_city
        )
        deliverability[position] = float(
            np.clip(
                quality - (0.0 if rendered_signals.pincode_city_consistent else penalty),
                0.0,
                1.0,
            )
        )
        addresses.append(rendered)
        signals.append(rendered_signals)

    return _Draws(
        reliability=reliability,
        writing_quality=writing_quality,
        prepaid_affinity=prepaid_affinity,
        customer_pincode_idx=customer_pincode_idx,
        signup_lag_hours=signup_lag_hours,
        activity_probs=activity_probs,
        pincode_tier_idx=pincode_tier_idx,
        pincode_effect=pincode_effect,
        pincode_city_idx=pincode_city_idx,
        customer_choice=customer_choice,
        order_days=raw_days[order],
        order_hours=raw_hours[order],
        timestamps=raw_timestamps[order],
        category_idx=raw_category_idx[order],
        order_values=raw_values[order],
        item_counts=raw_items[order],
        base_discount=raw_discount[order],
        courier_idx=raw_courier_idx[order],
        device_idx=raw_device_idx[order],
        product_page_seconds=raw_page_seconds[order],
        sessions_before=raw_sessions[order],
        time_to_checkout=raw_checkout[order],
        cart_edited=raw_cart_edited[order],
        cod_flag=raw_cod_flag[order],
        prepaid_failure_roll=raw_prepaid_failure[order],
        logit_noise=raw_logit_noise[order],
        label_draw=raw_label_draw[order],
        flip_draw=raw_flip_draw[order],
        dispatch_hours=raw_dispatch[order],
        transit_days=raw_transit[order],
        rto_extra=raw_rto_extra[order],
        cancel_draw=raw_cancel[order],
        addresses=addresses,
        address_signals=signals,
        latent_address_quality=latent_quality,
        address_deliverability=deliverability,
    )


@dataclass(slots=True)
class _PassResult:
    rows: list[dict[str, Any]]
    latents: list[dict[str, Any]]
    customers: list[_CustomerState]
    realised_cod_rate: float
    realised_prepaid_rate: float


def _run_pass(
    config: GeneratorConfig,
    params: GeneratorParams,
    draws: _Draws,
    *,
    cod_intercept: float,
    prepaid_intercept: float,
) -> _PassResult:
    """One full sequential simulation at the given intercepts.

    Pure: consumes no randomness, only the pre-drawn values in ``draws``. This is
    what allows the intercept solver to iterate without the dataset shifting
    underneath it.

    THE ORDER OF OPERATIONS INSIDE THE LOOP IS THE POINT. For each order, in
    chronological order:

    1. read the customer's history, filtered to orders that had already RESOLVED
       before this order's timestamp;
    2. compute the latent logit and sample the label;
    3. compute the resolution timestamp;
    4. append the outcome to the customer's history *keyed by resolution time*.

    Step 4 is why a later order sees this outcome only if it had genuinely come
    back first. An order placed on day 40 that returns on day 47 is invisible to
    an order placed by the same customer on day 42.
    """
    tiers = ("tier_1", "tier_2", "tier_3")
    weights = {name: driver.weight for name, driver in config.causal_drivers.items()}
    category_names = [category.name for category in config.catalogue.categories]
    category_offsets = [category.rto_logit_offset for category in config.catalogue.categories]
    courier_names = [courier.name for courier in config.couriers]
    courier_lane_quality = [courier.lane_quality for courier in config.couriers]
    device_names = ("mobile_app", "mobile_web", "desktop")
    late_night = frozenset(config.timing.late_night_hours)
    sale_days = frozenset(config.timing.sale_days)
    max_days = config.label_maturity.max_resolution_days
    horizon_end = params.end_date

    customers = [
        _CustomerState(
            index=i,
            customer_hash=_customer_hash(i, params.seed),
            reliability=float(draws.reliability[i]),
            address_quality=float(draws.writing_quality[i]),
            prepaid_affinity=float(draws.prepaid_affinity[i]),
            pincode_index=int(draws.customer_pincode_idx[i]),
            # Provisional: overwritten with (first order - lag) on the first order.
            signup_at=params.start_date,
        )
        for i in range(params.n_customers)
    ]

    rows: list[dict[str, Any]] = []
    latents: list[dict[str, Any]] = []
    cod_labels: list[bool] = []
    prepaid_labels: list[bool] = []

    for position in range(params.n_orders):
        customer = customers[int(draws.customer_choice[position])]
        ordered_at = datetime.fromtimestamp(float(draws.timestamps[position]), tz=UTC)
        pincode_index = customer.pincode_index
        tier = tiers[int(draws.pincode_tier_idx[pincode_index])]

        # ---- 1. as-of customer history -----------------------------------
        resolved_prior = [row for row in customer.history if row[0] < ordered_at]
        prior_order_count = len(resolved_prior)
        prior_rto_count = sum(1 for row in resolved_prior if row[1])
        prior_rto_rate = prior_rto_count / prior_order_count if prior_order_count else float("nan")
        prior_cod = sum(1 for row in resolved_prior if row[2])
        prepaid_to_cod_ratio = (
            (prior_order_count - prior_cod) / prior_cod if prior_cod else float("nan")
        )
        mean_resolution_days = (
            sum(row[3] for row in resolved_prior) / prior_order_count
            if prior_order_count
            else float("nan")
        )
        days_since_last_order = (
            (ordered_at - customer.last_order_at).total_seconds() / 86400.0
            if customer.last_order_at is not None
            else float("nan")
        )
        is_new_customer = customer.order_count == 0

        is_cod = bool(draws.cod_flag[position])
        cod_after_prepaid_failure = bool(
            is_cod and draws.prepaid_failure_roll[position] < config.payment.prepaid_failure_to_cod
        )

        day_offset = int(draws.order_days[position])
        is_sale_day = day_offset in sale_days
        discount_depth = float(
            np.clip(
                draws.base_discount[position]
                + (config.timing.sale_day_discount_uplift if is_sale_day else 0.0),
                0.0,
                0.85,
            )
        )
        gross_value = float(draws.order_values[position])
        discount_inr = round(gross_value * discount_depth, 2)
        net_value = round(gross_value - discount_inr, 2)

        if customer.order_values:
            median = float(np.median(customer.order_values))
            spread = float(np.std(customer.order_values)) or max(median * 0.35, 1.0)
            value_z = float(np.clip((net_value - median) / spread, -4.0, 4.0))
        else:
            value_z = 0.0

        hour = int(draws.order_hours[position])
        is_late_night = hour in late_night
        fast_checkout = float(draws.time_to_checkout[position]) < 45.0
        rendered = draws.addresses[position]
        signals = draws.address_signals[position]

        # ---- 2. latent logit, then the label -------------------------------
        logit = (
            weights["address_quality"] * (float(draws.address_deliverability[position]) - 0.5) * 2.0
            + weights["prior_rto_rate"] * (0.0 if prior_order_count == 0 else prior_rto_rate)
            + weights["discount_depth"] * discount_depth
            + weights["order_value_zscore"] * value_z
            + weights["late_night_order"] * float(is_late_night)
            + weights["pincode_tier_effect"] * config.geography.tier_risk_offset[tier]
            + weights["is_new_customer"] * float(is_new_customer)
            + weights["fast_checkout"] * float(fast_checkout)
            + weights["courier_lane_quality"]
            * float(courier_lane_quality[int(draws.courier_idx[position])])
            + weights["customer_reliability"] * (customer.reliability - 0.5) * 2.0
            + weights["pincode_effect"] * float(draws.pincode_effect[pincode_index])
            + float(category_offsets[int(draws.category_idx[position])])
            + float(draws.logit_noise[position])
        )
        calibrated = logit + (cod_intercept if is_cod else prepaid_intercept)
        probability = float(1.0 / (1.0 + math.exp(-max(min(calibrated, 60.0), -60.0))))

        label = bool(draws.label_draw[position] < probability)
        flipped = bool(draws.flip_draw[position] < config.noise.label_flip_rate)
        if flipped:
            label = not label

        # ---- 3. fulfilment timeline ----------------------------------------
        cancelled = bool(draws.cancel_draw[position] < config.fulfilment.cancellation_rate)
        dispatched_at = ordered_at + timedelta(hours=int(draws.dispatch_hours[position]))
        first_attempt_at = dispatched_at + timedelta(days=int(draws.transit_days[position]))

        if cancelled:
            outcome = "cancelled"
            is_rto: bool | None = False
            dispatched_out: datetime | None = None
            attempt_out: datetime | None = None
            resolved_at = ordered_at + timedelta(hours=int(draws.dispatch_hours[position]))
        elif label:
            outcome = "rto"
            is_rto = True
            dispatched_out = dispatched_at
            attempt_out = first_attempt_at
            resolved_at = first_attempt_at + timedelta(days=int(draws.rto_extra[position]))
        else:
            outcome = "delivered"
            is_rto = False
            dispatched_out = dispatched_at
            attempt_out = first_attempt_at
            resolved_at = first_attempt_at

        latest_allowed = ordered_at + timedelta(days=max_days)
        resolved_at = min(resolved_at, latest_allowed)
        maturity_days = (resolved_at - ordered_at).total_seconds() / 86400.0

        # LABEL MATURITY. An order whose terminal state falls beyond the horizon
        # is not yet known. Marked pending with a NULL label - never optimistically
        # labelled "delivered", which is how a benchmark manufactures optimism.
        is_mature = resolved_at <= horizon_end
        if not is_mature and config.label_maturity.exclude_unresolved_tail:
            outcome = "pending"
            is_rto = None
            resolved_out: datetime | None = None
        else:
            resolved_out = resolved_at

        # ---- 4. history entry, keyed by RESOLUTION time --------------------
        # Cancellations are deliberately excluded from delivery history. An order
        # cancelled before dispatch was never presented to the customer, so it
        # says nothing about whether they accept deliveries - counting it would
        # dilute the prior-RTO rate with events that carry no delivery signal.
        if is_mature and is_rto is not None and outcome != "cancelled":
            customer.history.append((resolved_at, bool(is_rto), is_cod, maturity_days, net_value))
            if is_cod:
                cod_labels.append(bool(is_rto))
            else:
                prepaid_labels.append(bool(is_rto))

        if customer.order_count == 0:
            customer.signup_at = ordered_at - timedelta(
                hours=int(draws.signup_lag_hours[customer.index])
            )
        customer.order_count += 1
        customer.order_values.append(net_value)
        customer.last_order_at = ordered_at

        rows.append(
            {
                cols.ORDER_ID: f"ORD-{position + 1:08d}",
                cols.MERCHANT_ID: DEFAULT_MERCHANT_ID,
                cols.CUSTOMER_HASH: customer.customer_hash,
                cols.ADDRESS_FINGERPRINT: rendered.fingerprint,
                cols.ORDERED_AT: ordered_at,
                cols.DISPATCHED_AT: dispatched_out,
                cols.FIRST_ATTEMPT_AT: attempt_out,
                cols.RESOLVED_AT: resolved_out,
                cols.DAY_INDEX: day_offset + 1,
                cols.PAYMENT_METHOD: "cod" if is_cod else "prepaid",
                cols.IS_COD: is_cod,
                cols.ORDER_VALUE_INR: net_value,
                cols.DISCOUNT_INR: discount_inr,
                cols.DISCOUNT_DEPTH: round(discount_depth, 4),
                cols.ITEM_COUNT: int(draws.item_counts[position]),
                cols.CATEGORY: category_names[int(draws.category_idx[position])],
                cols.CART_EDITED: bool(draws.cart_edited[position]),
                cols.PRODUCT_PAGE_SECONDS: float(draws.product_page_seconds[position]),
                cols.SESSIONS_BEFORE_PURCHASE: int(draws.sessions_before[position]),
                cols.DEVICE_CLASS: device_names[int(draws.device_idx[position])],
                cols.HOUR_OF_DAY: hour,
                cols.DAY_OF_WEEK: ordered_at.weekday(),
                cols.IS_LATE_NIGHT: is_late_night,
                cols.IS_SALE_DAY: is_sale_day,
                cols.TIME_TO_CHECKOUT_SECONDS: float(draws.time_to_checkout[position]),
                cols.COD_AFTER_PREPAID_FAILURE: cod_after_prepaid_failure,
                cols.ADDRESS_LINE: rendered.line,
                cols.ADDRESS_CITY: rendered.city,
                cols.ADDRESS_STATE: rendered.state,
                cols.PINCODE: rendered.pincode,
                cols.PINCODE_TIER: tier,
                cols.ADDR_TOKEN_COUNT: signals.token_count,
                cols.ADDR_HAS_HOUSE_NUMBER: signals.has_house_number,
                cols.ADDR_HAS_FLOOR_NUMBER: signals.has_floor_number,
                cols.ADDR_HAS_LANDMARK: signals.has_landmark,
                cols.ADDR_PINCODE_CITY_CONSISTENT: signals.pincode_city_consistent,
                cols.ADDR_ALLCAPS_RATIO: round(signals.allcaps_ratio, 4),
                cols.ADDR_GIBBERISH_RATIO: round(signals.gibberish_ratio, 4),
                cols.COURIER_PARTNER: courier_names[int(draws.courier_idx[position])],
                cols.PRIOR_ORDER_COUNT: prior_order_count,
                cols.PRIOR_RTO_COUNT: prior_rto_count,
                cols.PRIOR_RTO_RATE: prior_rto_rate,
                cols.DAYS_SINCE_LAST_ORDER: days_since_last_order,
                cols.PREPAID_TO_COD_RATIO: prepaid_to_cod_ratio,
                cols.MEAN_RESOLUTION_DAYS: mean_resolution_days,
                cols.IS_NEW_CUSTOMER: is_new_customer,
                cols.OUTCOME: outcome,
                cols.IS_RTO: is_rto,
                cols.MATURITY_DAYS: round(maturity_days, 4) if is_mature else None,
                cols.IS_MATURE: is_mature,
                cols.SPLIT: "train" if is_mature else "excluded_immature",
            }
        )

        latents.append(
            {
                cols.ORDER_ID: f"ORD-{position + 1:08d}",
                cols.TRUE_RTO_PROBABILITY: probability,
                cols.LATENT_LOGIT: calibrated,
                "customer_reliability": customer.reliability,
                "latent_address_quality": float(draws.latent_address_quality[position]),
                "address_deliverability": float(draws.address_deliverability[position]),
                "pincode_effect": float(draws.pincode_effect[pincode_index]),
                "label_flipped": flipped,
            }
        )

    return _PassResult(
        rows=rows,
        latents=latents,
        customers=customers,
        realised_cod_rate=(sum(cod_labels) / len(cod_labels)) if cod_labels else float("nan"),
        realised_prepaid_rate=(sum(prepaid_labels) / len(prepaid_labels))
        if prepaid_labels
        else float("nan"),
    )


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _simulate_v1(config: GeneratorConfig, params: GeneratorParams) -> GenerationResult:
    """Generator version 1.0.0.

    The base-rate intercepts are found by a fixed-point iteration rather than a
    closed-form solve, because the simulation has a genuine feedback loop: a
    customer's prior RTO rate is itself an input to their next order's risk. Change
    the intercept and the histories change, which changes the logits, which changes
    the marginal. Iterating the whole pass is the only way to land on the published
    base rate *with* that feedback intact - and the feedback is worth keeping,
    because "history predicts the future" is the single most important honest
    signal in this problem.

    The iteration is deterministic: the draws are fixed up front, so a pass is a
    pure function of the intercepts.
    """
    draws = _draw_everything(config, params)

    # Correct the target for the symmetric label-flip rate, so the *observed*
    # marginal matches the published figure rather than the pre-noise one.
    flip = config.noise.label_flip_rate
    denominator = 1.0 - 2.0 * flip
    cod_target = (config.base_rates.rto_given_cod - flip) / denominator
    prepaid_target = (config.base_rates.rto_given_prepaid - flip) / denominator
    cod_target = min(max(cod_target, 1e-4), 1.0 - 1e-4)
    prepaid_target = min(max(prepaid_target, 1e-4), 1.0 - 1e-4)

    cod_intercept = 0.0
    prepaid_intercept = 0.0
    tolerance = min(config.base_rates.marginal_tolerance * 0.25, 0.005)

    result = _run_pass(
        config, params, draws, cod_intercept=cod_intercept, prepaid_intercept=prepaid_intercept
    )
    for _ in range(MAX_CALIBRATION_ITERATIONS):
        cod_gap = _logit(config.base_rates.rto_given_cod) - _logit(result.realised_cod_rate)
        prepaid_gap = _logit(config.base_rates.rto_given_prepaid) - _logit(
            result.realised_prepaid_rate
        )
        if abs(cod_gap) < 1e-3 and abs(prepaid_gap) < 1e-3:
            break
        cod_intercept += cod_gap
        prepaid_intercept += prepaid_gap
        result = _run_pass(
            config, params, draws, cod_intercept=cod_intercept, prepaid_intercept=prepaid_intercept
        )
        if (
            abs(result.realised_cod_rate - config.base_rates.rto_given_cod) < tolerance
            and abs(result.realised_prepaid_rate - config.base_rates.rto_given_prepaid) < tolerance
        ):
            break

    return _assemble(config, params, draws, result)


def _assemble(
    config: GeneratorConfig,
    params: GeneratorParams,
    draws: _Draws,
    result: _PassResult,
) -> GenerationResult:
    """Turn a converged pass into the five frames the database and ML layer use."""
    tiers = ("tier_1", "tier_2", "tier_3")

    orders = pd.DataFrame(result.rows)[list(cols.RAW_COLUMNS)]
    latents = pd.DataFrame(result.latents)

    event_rows: list[dict[str, Any]] = []
    for row in result.rows:
        event_rows.extend(
            _delivery_events(
                order_id=row[cols.ORDER_ID],
                ordered_at=row[cols.ORDERED_AT],
                dispatched_at=row[cols.DISPATCHED_AT],
                first_attempt_at=row[cols.FIRST_ATTEMPT_AT],
                resolved_at=row[cols.RESOLVED_AT],
                outcome=row[cols.OUTCOME],
            )
        )
    delivery_events = pd.DataFrame(event_rows)

    # Address dimension: first occurrence wins, so the frame is a stable set of
    # distinct addresses rather than one row per order.
    address_columns = [
        cols.ADDRESS_FINGERPRINT,
        cols.ADDRESS_LINE,
        cols.ADDRESS_CITY,
        cols.ADDRESS_STATE,
        cols.PINCODE,
        cols.PINCODE_TIER,
        cols.ADDR_TOKEN_COUNT,
        cols.ADDR_HAS_HOUSE_NUMBER,
        cols.ADDR_HAS_FLOOR_NUMBER,
        cols.ADDR_HAS_LANDMARK,
        cols.ADDR_PINCODE_CITY_CONSISTENT,
        cols.ADDR_ALLCAPS_RATIO,
        cols.ADDR_GIBBERISH_RATIO,
    ]
    addresses = (
        orders[[*address_columns, cols.ORDERED_AT]]
        .sort_values(cols.ORDERED_AT, kind="stable")
        .drop_duplicates(subset=[cols.ADDRESS_FINGERPRINT], keep="first")
        .rename(columns={cols.ORDERED_AT: "first_seen_at"})
        .reset_index(drop=True)
    )

    used = set(orders[cols.CUSTOMER_HASH].unique())
    customers_frame = (
        pd.DataFrame(
            [
                {
                    cols.CUSTOMER_HASH: customer.customer_hash,
                    "signup_at": customer.signup_at,
                    "home_pincode": _pincode(customer.pincode_index),
                    "home_pincode_tier": tiers[int(draws.pincode_tier_idx[customer.pincode_index])],
                    "generated_order_count": customer.order_count,
                }
                for customer in result.customers
                if customer.customer_hash in used
            ]
        )
        .sort_values(cols.CUSTOMER_HASH, kind="stable")
        .reset_index(drop=True)
    )

    mature = orders[orders[cols.IS_MATURE]]
    mature_cod = mature[mature[cols.IS_COD]]
    mature_prepaid = mature[~mature[cols.IS_COD]]

    metadata = DatasetRunMetadata(
        run_id=_run_id(params),
        generator_version=params.generator_version,
        seed=params.seed,
        config_fingerprint=_config_fingerprint(config),
        config_snapshot=config.model_dump(mode="json"),
        n_customers=int(customers_frame.shape[0]),
        n_orders=int(orders.shape[0]),
        start_date=params.start_date,
        end_date=params.end_date,
        created_at=datetime.now(UTC),
        realised_rto_rate_cod=float(mature_cod[cols.IS_RTO].mean())
        if len(mature_cod)
        else float("nan"),
        realised_rto_rate_prepaid=float(mature_prepaid[cols.IS_RTO].mean())
        if len(mature_prepaid)
        else float("nan"),
        realised_cod_share=float(orders[cols.IS_COD].mean()),
        n_mature=int(mature.shape[0]),
        n_immature=int(orders.shape[0] - mature.shape[0]),
    )

    return GenerationResult(
        customers=customers_frame,
        addresses=addresses,
        orders=orders,
        delivery_events=delivery_events,
        latents=latents,
        metadata=metadata,
    )


def _delivery_events(
    *,
    order_id: str,
    ordered_at: datetime,
    dispatched_at: datetime | None,
    first_attempt_at: datetime | None,
    resolved_at: datetime | None,
    outcome: str,
) -> list[dict[str, Any]]:
    """The event trail for one order.

    Stored so a prediction can be reconstructed rather than merely believed: the
    events carry the timestamps that decide what was knowable when, which is what
    an as-of join needs and what an audit of a past decision needs.
    """
    events: list[dict[str, Any]] = [
        {
            "order_id": order_id,
            "sequence": 1,
            "event_type": "order_placed",
            "occurred_at": ordered_at,
        }
    ]
    sequence = 2
    if dispatched_at is not None:
        events.append(
            {
                "order_id": order_id,
                "sequence": sequence,
                "event_type": "dispatched",
                "occurred_at": dispatched_at,
            }
        )
        sequence += 1
    if first_attempt_at is not None:
        events.append(
            {
                "order_id": order_id,
                "sequence": sequence,
                "event_type": "delivery_attempted",
                "occurred_at": first_attempt_at,
            }
        )
        sequence += 1
    if resolved_at is not None:
        terminal = {
            "delivered": "delivered",
            "rto": "returned_to_origin",
            "cancelled": "cancelled",
        }[outcome]
        events.append(
            {
                "order_id": order_id,
                "sequence": sequence,
                "event_type": terminal,
                "occurred_at": resolved_at,
            }
        )
    return events


def _run_id(params: GeneratorParams) -> str:
    """Deterministic identifier for a (version, seed, parameters) combination.

    Deterministic on purpose: regenerating the same dataset produces the same
    ``run_id``, so a re-seed is visible as an upsert rather than as a second,
    subtly different dataset sitting alongside the first.
    """
    payload = (
        f"{params.generator_version}|{params.seed}|{params.n_customers}|{params.n_orders}"
        f"|{params.start_date.isoformat()}|{params.end_date.isoformat()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _config_fingerprint(config: GeneratorConfig) -> str:
    """SHA-256 over the generator configuration as loaded."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
