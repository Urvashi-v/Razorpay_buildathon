"""Applying the ladder to a whole book of orders, and pricing the result.

This is where a per-order decision becomes a merchant-level number. The engine
answers "what happens to this order"; this module answers "what happens to the
business", which is the question the specification's headline metric asks.

WHY THE EXPECTED FIGURES NEED NO LABELS
=======================================
A calibrated probability *is* an expectation. If the model says 0.62 and it is
calibrated, then 0.62 of an RTO is the honest expected value for that order, and
summing across the book gives an expected RTO count without any outcome data.
That is what makes the merchant simulator possible: a merchant can drag a margin
slider on today's unlabelled orders and get a real recomputation rather than a
projection from last quarter's measurement.

It is also entirely contingent on the calibration being good. Where labels do
exist, this module computes the realized figures alongside, and
``PortfolioEconomics.calibration_gap`` is the difference. A large gap invalidates
the expected figures, and it is reported rather than left for someone to notice.

THE ARITHMETIC, ONCE, EXPLICITLY
================================
For a band ``B`` with an action, over the orders that land in it::

    S_tp(B) = success_rate x success_multiplier(B) x rto_cost
    C_fp(B) = abandonment x abandonment_multiplier(B) x margin + support_cost(B)

    expected_saving(B)  = sum_{i in B} p_i        x S_tp(B)
    expected_fp_cost(B) = sum_{i in B} (1 - p_i)  x C_fp(B)
    expected_net(B)     = expected_saving(B) - expected_fp_cost(B)

Bands with no action contribute zero to both. The residual false-negative loss is
every RTO the policy fails to prevent: unflagged orders in full, plus the share of
flagged ones the intervention does not save.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rto_sentinel.contracts.economics import BandOutcome, PortfolioEconomics
from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.contracts.provenance import Provenance, Quantity
from rto_sentinel.decision.cost_model import band_outcome_economics, outcome_economics
from rto_sentinel.decision.policy import band_economics, resolve_boundaries
from rto_sentinel.decision.threshold import derive_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import PolicyConfig
    from rto_sentinel.contracts.decision import CostInputs

PER_ORDERS = 1000.0

#: Provenances the headline rupee figures rest on. Attached to every derived
#: quantity so a reader can see the assumption without reading the source.
_RUPEE_DEPENDENCIES = (
    Provenance.MEASURED,
    Provenance.MERCHANT_INPUT,
    Provenance.ASSUMED_INTERVENTION,
)


class PortfolioError(ValueError):
    """Raised when a book cannot be priced."""


def _rates_for_band(
    cost_inputs: CostInputs, policy: PolicyConfig, band: RiskBand
) -> tuple[float, float]:
    """Effective success and abandonment rates for a band, after clamping."""
    economics = band_economics(band, policy)
    success = min(
        max(cost_inputs.intervention_success_rate * economics.intervention_success_multiplier, 0.0),
        1.0,
    )
    abandonment = min(
        max(cost_inputs.abandonment_on_friction * economics.abandonment_multiplier, 0.0), 1.0
    )
    return success, abandonment


def evaluate_portfolio(
    probabilities: np.ndarray,
    *,
    cost_inputs: CostInputs,
    policy: PolicyConfig,
    labels: np.ndarray | None = None,
    split: str = "unlabelled",
    cost_profile: str = "custom",
    engine_version: str = "1.0.0",
) -> PortfolioEconomics:
    """Price one policy against one book of calibrated probabilities.

    ``labels`` are optional. Without them the realized fields stay ``None`` and
    only the expected figures are produced - which is the live case, and the case
    the merchant simulator runs in.
    """
    scores = np.asarray(probabilities, dtype="float64").ravel()
    if scores.size == 0:
        msg = "refusing to price an empty book"
        raise PortfolioError(msg)
    if not np.all(np.isfinite(scores)):
        msg = "the book contains non-finite probabilities"
        raise PortfolioError(msg)
    if scores.min() < 0.0 or scores.max() > 1.0:
        msg = f"probabilities must lie in [0, 1]; got [{scores.min()}, {scores.max()}]"
        raise PortfolioError(msg)

    y: np.ndarray | None = None
    if labels is not None:
        y = np.asarray(labels).astype(bool).ravel()
        if y.shape != scores.shape:
            msg = f"labels and probabilities disagree in length: {y.shape} vs {scores.shape}"
            raise PortfolioError(msg)

    n = int(scores.size)
    derivation = derive_threshold(cost_inputs)
    threshold = derivation.threshold
    ladder = resolve_boundaries(threshold, policy)
    base_economics = outcome_economics(cost_inputs)

    band_outcomes: list[BandOutcome] = []
    expected_savings = 0.0
    expected_fp_cost = 0.0
    residual_loss = 0.0
    support_on_true_positives = 0.0
    affected = 0
    assigned = np.zeros(n, dtype=bool)

    for boundary in ladder.boundaries:
        if boundary.upper_bound is None:
            mask = scores >= boundary.lower_bound
        else:
            mask = (scores >= boundary.lower_bound) & (scores < boundary.upper_bound)
        # Orders already assigned to a lower band cannot be claimed twice; the
        # boundaries are disjoint by construction, and this makes that a fact
        # rather than a belief about the construction.
        mask &= ~assigned
        assigned |= mask

        in_band = scores[mask]
        count = int(in_band.size)
        expected_rtos = float(in_band.sum())
        expected_goods = float((1.0 - in_band).sum())

        economics = band_outcome_economics(cost_inputs, band_economics(boundary.band, policy))
        success, abandonment = _rates_for_band(cost_inputs, policy, boundary.band)
        applies = boundary.action is not InterventionAction.NONE

        saving = expected_rtos * economics.true_positive_saving_inr if applies else 0.0
        fp_cost = expected_goods * economics.false_positive_cost_inr if applies else 0.0

        if applies:
            affected += count
            expected_savings += saving
            expected_fp_cost += fp_cost
            # The RTOs the intervention fails to save still come back.
            residual_loss += expected_rtos * (1.0 - success) * cost_inputs.rto_cost_inr
            # The support cost the spec's C_fp omits: it charges support only on
            # false positives, but ops pays it on every frictioned order.
            support_on_true_positives += (
                expected_rtos * band_economics(boundary.band, policy).support_cost_inr
            )
        else:
            # Unflagged: every expected RTO lands in full.
            residual_loss += expected_rtos * cost_inputs.rto_cost_inr

        band_outcomes.append(
            BandOutcome(
                band=boundary.band,
                action=boundary.action,
                lower_bound=boundary.lower_bound,
                upper_bound=boundary.upper_bound,
                n_orders=count,
                share_of_book=count / n,
                expected_rto_orders=expected_rtos,
                realized_rto_orders=int(y[mask].sum()) if y is not None else None,
                intervention_success_rate=success,
                abandonment_rate=abandonment,
                support_cost_inr=band_economics(boundary.band, policy).support_cost_inr,
                expected_saving_inr=saving,
                expected_false_positive_cost_inr=fp_cost,
                expected_net_inr=saving - fp_cost,
            )
        )

    unassigned = int((~assigned).sum())
    if unassigned:  # pragma: no cover - the ladder covers [0, 1] by construction
        msg = f"{unassigned} orders fell outside every band at threshold {threshold}"
        raise PortfolioError(msg)

    expected_net = expected_savings - expected_fp_cost
    flagged_mask = scores >= threshold
    flag_rate = float(flagged_mask.mean())

    realized: dict[str, float | int | None] = {
        "realized_net_inr_per_1000_orders": None,
        "realized_true_positives": None,
        "realized_false_positives": None,
        "realized_false_negatives": None,
        "realized_true_negatives": None,
        "realized_precision": None,
        "realized_recall": None,
    }
    if y is not None:
        acted = np.zeros(n, dtype=bool)
        realized_net = 0.0
        for boundary in ladder.boundaries:
            if boundary.action is InterventionAction.NONE:
                continue
            if boundary.upper_bound is None:
                mask = scores >= boundary.lower_bound
            else:
                mask = (scores >= boundary.lower_bound) & (scores < boundary.upper_bound)
            acted |= mask
            economics = band_outcome_economics(cost_inputs, band_economics(boundary.band, policy))
            realized_net += economics.net_versus_doing_nothing(
                tp=int((mask & y).sum()), fp=int((mask & ~y).sum())
            )

        true_positives = int((acted & y).sum())
        false_positives = int((acted & ~y).sum())
        false_negatives = int((~acted & y).sum())
        true_negatives = int((~acted & ~y).sum())
        realized = {
            "realized_net_inr_per_1000_orders": (realized_net / n) * PER_ORDERS,
            "realized_true_positives": true_positives,
            "realized_false_positives": false_positives,
            "realized_false_negatives": false_negatives,
            "realized_true_negatives": true_negatives,
            "realized_precision": (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives)
                else None
            ),
            "realized_recall": (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives)
                else None
            ),
        }

    # The do-nothing reference uses expected RTOs when there are no labels, so a
    # merchant sees the size of their problem either way.
    expected_or_observed_rtos = float(y.sum()) if y is not None else float(scores.sum())
    do_nothing = -(expected_or_observed_rtos * cost_inputs.rto_cost_inr / n) * PER_ORDERS

    holdout = policy.holdout_control.fraction_of_flagged if policy.holdout_control.enabled else 0.0
    net_per_1000 = (expected_net / n) * PER_ORDERS

    return PortfolioEconomics(
        threshold=threshold,
        threshold_source=(
            f"derived from cost inputs: C_fp={derivation.cost_false_positive_inr:.2f}, "
            f"S_tp={derivation.saving_true_positive_inr:.2f}"
        ),
        cost_profile=cost_profile,
        split=split,
        n_orders=n,
        engine_version=engine_version,
        flag_rate=flag_rate,
        intervention_rate=affected / n,
        expected_orders_affected=affected,
        expected_savings_inr=expected_savings,
        expected_false_positive_cost_inr=expected_fp_cost,
        expected_false_negative_loss_inr=residual_loss,
        expected_total_cost_inr=expected_fp_cost + residual_loss,
        expected_net_inr=expected_net,
        expected_net_inr_per_1000_orders=net_per_1000,
        support_cost_on_true_positives_inr=support_on_true_positives,
        do_nothing_loss_inr_per_1000_orders=do_nothing,
        holdout_fraction_of_flagged=holdout,
        net_inr_per_1000_after_holdout=net_per_1000 * (1.0 - holdout),
        bands=band_outcomes,
        collapsed_bands=[f"{band.value}: {reason}" for band, reason in ladder.collapsed],
        quantities=_quantities(
            cost_inputs=cost_inputs,
            threshold=threshold,
            base_fp=base_economics.false_positive_cost_inr,
            base_tp=base_economics.true_positive_saving_inr,
            flag_rate=flag_rate,
            net_per_1000=net_per_1000,
            expected_fp_cost=expected_fp_cost,
        ),
        **realized,
    )


def _quantities(
    *,
    cost_inputs: CostInputs,
    threshold: float,
    base_fp: float,
    base_tp: float,
    flag_rate: float,
    net_per_1000: float,
    expected_fp_cost: float,
) -> list[Quantity]:
    """Every headline number, labelled with where it came from.

    The report renders from this list, which is what stops a merchant input and a
    measured metric appearing side by side as equally established facts.
    """
    return [
        Quantity(
            name="rto_cost_inr",
            value=cost_inputs.rto_cost_inr,
            unit="INR",
            provenance=Provenance.MERCHANT_INPUT,
            source="config/cost_model.yaml or the merchant's own figure",
        ),
        Quantity(
            name="contribution_margin_inr",
            value=cost_inputs.contribution_margin_inr,
            unit="INR",
            provenance=Provenance.MERCHANT_INPUT,
            source="config/cost_model.yaml or the merchant's own figure",
        ),
        Quantity(
            name="abandonment_on_friction",
            value=cost_inputs.abandonment_on_friction,
            unit="probability",
            provenance=Provenance.ASSUMED_INTERVENTION,
            source="No measurement exists. Requires a controlled holdout to establish.",
        ),
        Quantity(
            name="intervention_success_rate",
            value=cost_inputs.intervention_success_rate,
            unit="probability",
            provenance=Provenance.ASSUMED_INTERVENTION,
            source="No measurement exists. Requires a controlled holdout to establish.",
        ),
        Quantity(
            name="cost_of_false_positive",
            value=base_fp,
            unit="INR",
            provenance=Provenance.DERIVED,
            source="C_fp = abandonment x margin + support cost",
            depends_on=(Provenance.MERCHANT_INPUT, Provenance.ASSUMED_INTERVENTION),
        ),
        Quantity(
            name="saving_per_true_positive",
            value=base_tp,
            unit="INR",
            provenance=Provenance.DERIVED,
            source="S_tp = intervention success x RTO cost",
            depends_on=(Provenance.MERCHANT_INPUT, Provenance.ASSUMED_INTERVENTION),
        ),
        Quantity(
            name="operating_threshold",
            value=threshold,
            unit="probability",
            provenance=Provenance.DERIVED,
            source="threshold = C_fp / (C_fp + S_tp). Never 0.5, never fitted to labels.",
            depends_on=(Provenance.MERCHANT_INPUT, Provenance.ASSUMED_INTERVENTION),
        ),
        Quantity(
            name="flag_rate",
            value=flag_rate,
            unit="probability",
            provenance=Provenance.DERIVED,
            source="Share of the scored book at or above the derived threshold",
            depends_on=(Provenance.MEASURED, Provenance.DERIVED),
        ),
        Quantity(
            name="expected_false_positive_cost",
            value=expected_fp_cost,
            unit="INR",
            provenance=Provenance.DERIVED,
            source="Reported separately and never netted away",
            depends_on=_RUPEE_DEPENDENCIES,
        ),
        Quantity(
            name="net_inr_saved_per_1000_orders",
            value=net_per_1000,
            unit="INR/1000 orders",
            provenance=Provenance.DERIVED,
            source="TP x S_tp - FP x C_fp, measured against doing nothing",
            depends_on=_RUPEE_DEPENDENCIES,
        ),
    ]
