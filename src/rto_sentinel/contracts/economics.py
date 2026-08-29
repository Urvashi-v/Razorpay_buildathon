"""Portfolio economics: what a policy does to a book of orders, in rupees.

TWO MODES, AND THE DIFFERENCE BETWEEN THEM IS ITSELF A RESULT
=============================================================
Every figure here exists in two versions:

**Expected** - computed from calibrated probabilities alone, with no labels. This
is what a merchant can compute on today's unlabelled order book, and it is what
the merchant simulator returns. It is only as good as the calibration.

**Realized** - computed from observed outcomes on a held-out split. This is a
measurement, and it is only available after the labels mature.

Reporting both is not duplication. If the model is calibrated, the expected
true-positive count and the realized one should agree; a large gap between them
means the probabilities are not honest, and it shows up here before it shows up
in a reliability diagram anyone thought to look at.

WHAT THE HEADLINE NETS AND WHAT IT DOES NOT
===========================================
``net_inr_per_1000_orders`` is measured against doing nothing, so the
false-negative term cancels - the derivation is written out in
``OutcomeEconomics.net_versus_doing_nothing``. The false-negative loss is still
reported, on its own line, because "how much RTO loss remains after the policy
runs" is a question a merchant asks and a net figure cannot answer.

A KNOWN UNDERSTATEMENT, STATED RATHER THAN FIXED
================================================
The specification folds the per-friction support cost into ``C_fp``, which
applies it only to false positives. In reality an ops team pays it on every
frictioned order, true positives included. Keeping the spec's formula means the
headline understates support spend; ``support_cost_on_true_positives_inr``
reports the omitted amount separately so the gap is visible rather than silently
favourable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.contracts.provenance import Quantity


class BandOutcome(BaseModel):
    """What one rung of the ladder does to the orders that land in it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band: RiskBand
    action: InterventionAction
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float | None = Field(default=None, ge=0.0, le=1.0)

    n_orders: int = Field(ge=0)
    share_of_book: float = Field(ge=0.0, le=1.0)

    #: Sum of calibrated probabilities in this band: how many RTOs to expect.
    expected_rto_orders: float = Field(ge=0.0)
    #: Observed RTOs, where labels exist.
    realized_rto_orders: int | None = Field(default=None, ge=0)

    #: ASSUMED effectiveness applied to this band's action.
    intervention_success_rate: float = Field(ge=0.0, le=1.0)
    abandonment_rate: float = Field(ge=0.0, le=1.0)
    support_cost_inr: float = Field(ge=0.0)

    expected_saving_inr: float
    expected_false_positive_cost_inr: float = Field(ge=0.0)
    expected_net_inr: float

    @property
    def applies_friction(self) -> bool:
        return self.action is not InterventionAction.NONE

    @property
    def expected_precision(self) -> float | None:
        """Expected share of frictioned orders in this band that would have returned."""
        if self.n_orders == 0:
            return None
        return self.expected_rto_orders / self.n_orders


class PortfolioEconomics(BaseModel):
    """The full economic picture of one policy on one book of orders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- what produced this ------------------------------------------------
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: str
    cost_profile: str
    split: str = Field(description="Which book was scored. Never 'test' from a live slider.")
    n_orders: int = Field(ge=0)
    engine_version: str

    # --- rates -------------------------------------------------------------
    flag_rate: float = Field(
        ge=0.0, le=1.0, description="Share of orders at or above the derived threshold"
    )
    intervention_rate: float = Field(
        ge=0.0, le=1.0, description="Share of orders receiving any action"
    )
    expected_orders_affected: int = Field(
        ge=0, description="Count of orders receiving an intervention"
    )

    # --- expected, from probabilities only ---------------------------------
    expected_savings_inr: float
    expected_false_positive_cost_inr: float = Field(ge=0.0)
    expected_false_negative_loss_inr: float = Field(
        ge=0.0, description="Residual RTO loss the policy does not prevent"
    )
    expected_total_cost_inr: float = Field(
        ge=0.0, description="False-positive cost plus residual false-negative loss"
    )
    expected_net_inr: float
    expected_net_inr_per_1000_orders: float

    #: The support spend the spec's C_fp leaves out. See the module docstring.
    support_cost_on_true_positives_inr: float = Field(ge=0.0)

    # --- realized, where labels exist --------------------------------------
    realized_net_inr_per_1000_orders: float | None = None
    realized_true_positives: int | None = Field(default=None, ge=0)
    realized_false_positives: int | None = Field(default=None, ge=0)
    realized_false_negatives: int | None = Field(default=None, ge=0)
    realized_true_negatives: int | None = Field(default=None, ge=0)
    realized_precision: float | None = None
    realized_recall: float | None = None

    # --- the do-nothing reference -----------------------------------------
    do_nothing_loss_inr_per_1000_orders: float = Field(
        description="Negative: the loss absorbed today, and what the net is measured against"
    )

    # --- the control slice --------------------------------------------------
    holdout_fraction_of_flagged: float = Field(ge=0.0, le=1.0)
    net_inr_per_1000_after_holdout: float = Field(
        description="Net once the randomised no-friction slice is withheld"
    )

    bands: list[BandOutcome]
    collapsed_bands: list[str] = Field(
        default_factory=list, description="Rungs that cannot fire at this threshold, with reasons"
    )

    #: Every headline number with its provenance attached. The report renders
    #: from this, so a rupee figure cannot appear without its qualifier.
    quantities: list[Quantity] = Field(default_factory=list)

    @model_validator(mode="after")
    def _band_shares_are_coherent(self) -> PortfolioEconomics:
        counted = sum(band.n_orders for band in self.bands)
        if counted != self.n_orders:
            msg = f"bands account for {counted} orders but the book holds {self.n_orders}"
            raise ValueError(msg)
        return self

    @property
    def expected_true_positives(self) -> float:
        """Expected count of frictioned orders that would have returned."""
        return sum(band.expected_rto_orders for band in self.bands if band.applies_friction)

    @property
    def calibration_gap(self) -> float | None:
        """Expected minus realized true positives. Near zero means calibrated.

        A large positive gap means the model expected more RTOs among flagged
        orders than actually occurred - overconfidence - and every rupee figure
        derived from those probabilities is correspondingly optimistic.
        """
        if self.realized_true_positives is None:
            return None
        return self.expected_true_positives - self.realized_true_positives


class ThresholdPoint(BaseModel):
    """One row of the threshold sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    flag_rate: float = Field(ge=0.0, le=1.0)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    expected_cost_inr: float
    expected_net_inr_per_1000_orders: float
    realized_net_inr_per_1000_orders: float | None = None
    true_positives: int | None = Field(default=None, ge=0)
    false_positives: int | None = Field(default=None, ge=0)
    is_derived_operating_point: bool = False


class ThresholdSweep(BaseModel):
    """The sweep, plus a statement of how the operating point was actually chosen.

    ``selection_methodology`` is a required field rather than documentation
    because the sweep is exactly the artefact someone would use to pick the
    highest point on the curve. It was not chosen that way, and the artefact says
    so wherever it travels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str
    cost_profile: str
    derived_threshold: float = Field(ge=0.0, le=1.0)
    best_net_threshold: float = Field(
        ge=0.0, le=1.0, description="Where the curve peaks. NOT the operating point."
    )
    selection_methodology: str
    points: list[ThresholdPoint]

    @model_validator(mode="after")
    def _methodology_is_stated(self) -> ThresholdSweep:
        if "derived" not in self.selection_methodology.lower():
            msg = (
                "selection_methodology must state that the operating threshold is derived "
                "from merchant economics, not read off this curve"
            )
            raise ValueError(msg)
        return self
