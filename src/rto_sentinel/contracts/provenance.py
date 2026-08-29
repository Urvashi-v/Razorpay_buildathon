"""Where a number came from, carried by the number itself.

THE PROBLEM THIS SOLVES
=======================
An economic report mixes five kinds of number, and they look identical on a
dashboard:

* a **measured** model metric - PR-AUC on a held-out split, computed from
  predictions against labels;
* a **merchant-provided** input - the margin a merchant typed into a form;
* a **published** assumption - a figure taken from a cited external source;
* a **simulated** assumption - a parameter this project chose when writing the
  benchmark generator;
* a **derived** output - arithmetic over the four above.

"₹5,169 saved per 1,000 orders" is derived from a measured confusion matrix and
two assumed rates. Presenting it beside a measured PR-AUC without saying so
invites the reader to treat both as equally established, and they are not: one
is a measurement on synthetic labels and the other is arithmetic over numbers
nobody has verified.

So provenance is attached to the value rather than to a footnote. A
:class:`Quantity` cannot be constructed without it, and the report renderer
groups by it.

THE RULE THAT MATTERS MOST
==========================
``ASSUMED_INTERVENTION`` exists as its own category, separate from
``PUBLISHED``. Intervention effectiveness - "60% of risky orders are saved when
we ask for confirmation" - is the single most load-bearing number in this
system's rupee figures and **nobody has measured it here**. It cannot be
measured without running the intervention and observing the counterfactual,
which is what the holdout control slice in ``config/policy.yaml`` exists to make
possible later. Until then it is an assumption, labelled as one everywhere it
appears.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Provenance(StrEnum):
    """The five kinds of number an economic report contains."""

    #: Computed from model predictions against held-out labels by
    #: ``rto_sentinel.eval``. On this project the labels are simulated, so a
    #: measured metric is a measurement *of the benchmark*.
    MEASURED = "measured"

    #: Supplied by the merchant: margin, RTO cost, support cost. True for that
    #: merchant by definition, unverifiable by us.
    MERCHANT_INPUT = "merchant_input"

    #: Taken from a cited external source. See ``docs/sources.md``.
    PUBLISHED = "published"

    #: A parameter this project chose when writing the benchmark generator.
    #: Not evidence about the world.
    SIMULATED = "simulated"

    #: An assumption about how well an intervention works. Kept separate from
    #: PUBLISHED because it is the assumption the rupee figures are most
    #: sensitive to, and because measuring it requires a controlled holdout that
    #: has not been run.
    ASSUMED_INTERVENTION = "assumed_intervention"

    #: Arithmetic over the categories above. Inherits every weakness of its
    #: inputs, which is why ``Quantity.depends_on`` exists.
    DERIVED = "derived"


#: Provenances that are assumptions rather than observations. A report must never
#: present one of these as established fact.
ASSUMPTION_PROVENANCES: frozenset[Provenance] = frozenset(
    {Provenance.SIMULATED, Provenance.ASSUMED_INTERVENTION}
)

#: Human-readable qualifiers, used verbatim by the report renderer so that the
#: wording cannot drift between surfaces.
PROVENANCE_LABELS: dict[Provenance, str] = {
    Provenance.MEASURED: "measured on held-out data",
    Provenance.MERCHANT_INPUT: "merchant-provided input",
    Provenance.PUBLISHED: "published external figure",
    Provenance.SIMULATED: "simulator assumption, not evidence",
    Provenance.ASSUMED_INTERVENTION: "ASSUMED intervention effectiveness, never measured",
    Provenance.DERIVED: "derived by arithmetic",
}


class Quantity(BaseModel):
    """One number, its units, and where it came from.

    ``depends_on`` lists the provenances feeding a derived value, so a reader can
    see at a glance that a rupee saving rests on an unmeasured intervention rate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(max_length=80)
    value: float
    unit: str = Field(max_length=32, description="e.g. INR, INR/1000 orders, probability, count")
    provenance: Provenance
    source: str = Field(
        default="", max_length=400, description="Citation, config key, or code path"
    )
    depends_on: tuple[Provenance, ...] = ()

    @property
    def is_assumption(self) -> bool:
        """True when this value, or anything it rests on, is an assumption."""
        if self.provenance in ASSUMPTION_PROVENANCES:
            return True
        return any(parent in ASSUMPTION_PROVENANCES for parent in self.depends_on)

    @property
    def label(self) -> str:
        return PROVENANCE_LABELS[self.provenance]

    def qualified(self) -> str:
        """The value with its provenance attached, for prose and tables."""
        marker = " [ASSUMPTION]" if self.is_assumption else ""
        return f"{self.value:,.4g} {self.unit} ({self.label}){marker}"
