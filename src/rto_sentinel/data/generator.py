"""Synthetic order generator.

SPEC section 03 (data strategy) and section 09 (defense-only compliance).

WHAT THIS COMPONENT IS, STATED PLAINLY
--------------------------------------
A labelled tabular sampler. It draws order metadata from the distributions in
``config/generator.yaml`` and assigns a probabilistic RTO label from published
aggregate base rates. It is the one component in this repository that deserves
scrutiny under a defense-only rule, so it is named explicitly and kept short
enough for a reviewer to verify in a minute.

WHAT IT DOES NOT PRODUCE
------------------------
Working payment credentials. Valid identity documents. Deliverable addresses.
Anything usable outside this repository's own evaluation harness. Customer
identifiers are opaque hashes of a row index and a salt; addresses are token
patterns, not real locations; pincodes are synthetic six-digit identifiers drawn
from a generated pool rather than a map of real Indian postcodes.

HONESTY REQUIREMENT
-------------------
Every artefact derived from this generator carries the provenance string in
``rto_sentinel.contracts.evaluation.EvaluationReport.data_provenance``. Absolute
metric values reflect the generator's assumptions, not reality.

STATUS: Phase 2. The interface below is fixed; the sampling is not yet written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.configuration.schemas import GeneratorConfig


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A generated dataset plus the provenance needed to reproduce it."""

    orders: pd.DataFrame
    config_fingerprint: str
    seed: int
    n_orders: int
    realised_rto_rate_cod: float
    realised_rto_rate_prepaid: float


class OrderGenerator(Protocol):
    """Anything that can produce a labelled order table."""

    def generate(self, config: GeneratorConfig, seed: int) -> GenerationResult:
        """Produce ``config.horizon.n_orders`` rows with terminal outcomes."""
        ...


class ConfiguredOrderGenerator:
    """The project's generator, driven entirely by ``config/generator.yaml``.

    Design notes for the implementation (Phase 2):

    * The latent risk score is a linear combination of the documented causal
      drivers on the logit scale. Those coefficients are the generator's
      assumptions and are never exposed to a model.
    * Marginal RTO rates are calibrated back to ``base_rates`` within
      ``marginal_tolerance``; the generator's own test asserts this rather than
      trusting the parameterisation.
    * Resolution timestamps are drawn from ``label_maturity``, and rows whose
      resolution would fall beyond the horizon are marked
      ``DatasetSplit.EXCLUDED_IMMATURE`` rather than labelled "delivered".
    """

    def generate(self, config: GeneratorConfig, seed: int) -> GenerationResult:
        raise NotImplementedError(
            "Synthetic generation lands in Phase 2. The contract above is fixed."
        )
