"""Merchant simulation: change the economics, recompute everything downstream.

WHAT "GENUINELY RECALCULATE" MEANS HERE
=======================================
A margin slider that multiplies a cached saving by a ratio is a frontend trick.
This module does the real thing: new cost inputs go in, and the threshold, the
band boundaries, the action assigned to every order, and the full rupee picture
all come back out, recomputed from the scored book.

Nothing about a simulation is cached or interpolated, and nothing is computed in
a browser. Changing the margin from 250 to 400 moves the threshold, moves every
band cut point with it, moves orders between bands, and therefore moves the flag
rate, the false-positive cost and the net saving - each of which is recalculated
from the same arithmetic the production path uses. There is exactly one
implementation of that arithmetic, in ``decision.portfolio``, and both the API
and the report call it.

WHAT A SIMULATION MAY NOT TOUCH
===============================
The sealed test split. A slider is dragged dozens of times in a demo; a slider
wired to the sealed set would destroy the seal in the first minute of use, and
nobody would notice. :func:`simulate` refuses ``split="test"``.

THE LADDER IS NOT ASSUMED TO BE WORTH IT
========================================
:func:`compare_ladder_against_uniform` prices the graduated ladder against
applying a single action to everything above the threshold. Under the multipliers
declared in ``config/policy.yaml`` those two are not equal, and which one wins is
an empirical question about assumptions nobody has measured. Answering it in code
rather than assuming the ladder is better is the honest treatment - a graduated
ladder that loses money is a finding, not a feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rto_sentinel.contracts.decision import CostInputs, ThresholdDerivation
from rto_sentinel.contracts.economics import PortfolioEconomics
from rto_sentinel.decision.portfolio import evaluate_portfolio
from rto_sentinel.decision.threshold import derive_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import PolicyConfig


class SimulationError(ValueError):
    """Raised when a simulation is requested against data it may not use."""


class LadderRung(BaseModel):
    """One rung as the console renders it: bounds, action, and what it costs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band: str
    action: str
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float | None = Field(default=None, ge=0.0, le=1.0)
    n_orders: int = Field(ge=0)
    share_of_book: float = Field(ge=0.0, le=1.0)
    expected_net_inr: float
    intervention_success_rate: float = Field(
        ge=0.0, le=1.0, description="ASSUMED. Not measured on this or any data."
    )
    abandonment_rate: float = Field(
        ge=0.0, le=1.0, description="ASSUMED. Not measured on this or any data."
    )


class SimulationResult(BaseModel):
    """Everything that moves when a merchant changes one number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: ThresholdDerivation
    ladder: list[LadderRung]
    collapsed_bands: list[str]
    economics: PortfolioEconomics

    #: Set when a baseline was supplied, so the console can show the delta rather
    #: than making the reader diff two tables by eye.
    baseline_threshold: float | None = None
    baseline_net_inr_per_1000_orders: float | None = None

    @property
    def threshold_delta(self) -> float | None:
        if self.baseline_threshold is None:
            return None
        return self.threshold.threshold - self.baseline_threshold

    @property
    def net_delta_inr_per_1000_orders(self) -> float | None:
        if self.baseline_net_inr_per_1000_orders is None:
            return None
        return (
            self.economics.expected_net_inr_per_1000_orders - self.baseline_net_inr_per_1000_orders
        )


def simulate(
    probabilities: np.ndarray,
    *,
    cost_inputs: CostInputs,
    policy: PolicyConfig,
    labels: np.ndarray | None = None,
    split: str = "validation",
    cost_profile: str = "custom",
    baseline: CostInputs | None = None,
) -> SimulationResult:
    """Recompute the entire decision policy under one set of merchant economics.

    ``baseline`` is optional: supplying it adds the deltas a console needs to
    show "your threshold moved from X to Y" without recomputing the comparison
    on the client.
    """
    if split == "test":
        msg = (
            "refusing to simulate against the sealed test split. A slider is dragged "
            "dozens of times in a demo; wiring one to the sealed set would consume it "
            "silently. Simulate on validation, or on live unlabelled orders."
        )
        raise SimulationError(msg)

    economics = evaluate_portfolio(
        probabilities,
        cost_inputs=cost_inputs,
        policy=policy,
        labels=labels,
        split=split,
        cost_profile=cost_profile,
    )

    rungs = [
        LadderRung(
            band=band.band.value,
            action=band.action.value,
            lower_bound=band.lower_bound,
            upper_bound=band.upper_bound,
            n_orders=band.n_orders,
            share_of_book=band.share_of_book,
            expected_net_inr=band.expected_net_inr,
            intervention_success_rate=band.intervention_success_rate,
            abandonment_rate=band.abandonment_rate,
        )
        for band in economics.bands
    ]

    baseline_threshold: float | None = None
    baseline_net: float | None = None
    if baseline is not None:
        baseline_threshold = derive_threshold(baseline).threshold
        baseline_net = evaluate_portfolio(
            probabilities,
            cost_inputs=baseline,
            policy=policy,
            labels=labels,
            split=split,
            cost_profile=cost_profile,
        ).expected_net_inr_per_1000_orders

    return SimulationResult(
        threshold=derive_threshold(cost_inputs),
        ladder=rungs,
        collapsed_bands=list(economics.collapsed_bands),
        economics=economics,
        baseline_threshold=baseline_threshold,
        baseline_net_inr_per_1000_orders=baseline_net,
    )


class PolicyComparison(BaseModel):
    """The graduated ladder priced against simpler alternatives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graduated_net_inr_per_1000: float
    uniform_net_inr_per_1000: dict[str, float] = Field(
        description="Net if a single band's action were applied to everything above threshold"
    )
    best_uniform_action: str
    graduated_wins: bool
    note: str


def compare_ladder_against_uniform(
    probabilities: np.ndarray,
    *,
    cost_inputs: CostInputs,
    policy: PolicyConfig,
    labels: np.ndarray | None = None,
) -> PolicyComparison:
    """Is the graduated ladder worth having, under its own assumptions?

    Prices the configured ladder against a flat policy: one action, applied to
    every order above the derived threshold. The comparison is entirely a
    function of the assumed multipliers - a gentler rung saves less, and if it
    covers most of the flagged volume the ladder can lose to simply asking
    everyone to confirm.

    This answers a question the specification's ladder design raises and does not
    settle. It is not evidence that either policy is better in reality, because
    the multipliers it turns on have never been measured.
    """
    graduated = evaluate_portfolio(
        probabilities, cost_inputs=cost_inputs, policy=policy, labels=labels
    ).expected_net_inr_per_1000_orders

    uniform: dict[str, float] = {}
    for band in policy.bands:
        if band.action == "none":
            continue
        # A ladder where every acting rung carries this band's economics is the
        # same thing as applying that one action to everything above threshold.
        flat = policy.model_copy(
            update={
                "bands": [
                    entry
                    if entry.action == "none"
                    else entry.model_copy(
                        update={"economics": band.economics, "action": band.action}
                    )
                    for entry in policy.bands
                ]
            }
        )
        uniform[band.action] = evaluate_portfolio(
            probabilities, cost_inputs=cost_inputs, policy=flat, labels=labels
        ).expected_net_inr_per_1000_orders

    best_action = max(uniform, key=lambda key: uniform[key]) if uniform else "none"
    best_value = uniform.get(best_action, float("-inf"))

    return PolicyComparison(
        graduated_net_inr_per_1000=graduated,
        uniform_net_inr_per_1000=uniform,
        best_uniform_action=best_action,
        graduated_wins=graduated >= best_value,
        note=(
            "Both figures rest on the same ASSUMED intervention multipliers in "
            "config/policy.yaml. This comparison tests whether graduation pays under those "
            "assumptions; it is not evidence about which policy is better in reality."
        ),
    )
