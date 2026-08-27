# Sources

Every market figure used to anchor the simulator's base rates and the cost
model's bands, with its citation. These are industry estimates from public
sources; they vary by category, region and reporting methodology, and they are
cited rather than asserted.

**No proprietary or customer data is used anywhere in this project.**

## Market figures

| # | Figure | Source | Used for |
|---|--------|--------|----------|
| 1 | India's average e-commerce RTO rate is roughly **23%**, with category peaks approaching 40% (analysis across 180M+ shoppers) | GoKwik | Framing the loss; sanity bound on blended RTO rate |
| 2 | RTO around **26%** on COD / non-prepaid versus **under 2%** on prepaid; some city cohorts reported near 35% | Shipway ShipNotes, FY25 | `generator.yaml` → `base_rates.rto_given_cod` and `rto_given_prepaid` |
| 3 | COD accounts for roughly **60–65%** of Indian e-commerce orders | ET Prime Research, via Dazeinfo | `generator.yaml` → `payment.cod_share` |
| 4 | Direct per-RTO cost of roughly **₹150–300**, covering forward freight, reverse freight, repackaging and handling | Indian D2C logistics cost analyses | `cost_model.yaml` → `rto_cost_inr` and its documented bounds |
| 5 | Over **₹8,000 crore** annually attributed to RTO across Indian D2C brands | Aggregate D2C loss estimate | Framing the scale of the loss class |
| 6 | Context on digital payment fraud reporting and the sector's supervisory direction, including ML-based detection initiatives such as MuleHunter.ai | Reserve Bank of India, Annual Report 2025–26 | Regulatory context for the fairness and auditability posture |

## Public datasets used for realism, not labels

| Dataset | Role | Explicitly NOT used for |
|---------|------|-------------------------|
| UCI Online Retail II | Shaping realistic marginals for basket size, order-value distribution and repeat-purchase cadence | Supplying an RTO label |

`generator.yaml` records this as `realism_anchors.used_for_labels: false`, and
`GeneratorConfig` refuses to load if that value is anything but false. Labels come
from this project's own documented generative process and nowhere else.

## Assumptions that are NOT sourced

Stated separately, because the distinction matters.

| Assumption | Default | Status |
|---|---|---|
| `intervention_success_rate` | 0.60 | Taken from published intervention studies, **not measured here**. Replacing it with a measured number requires A/B testing the friction ladder — a research question in its own right. See `monitoring/outcomes.py`. |
| `abandonment_on_friction` | 0.25 | A merchant-specific input, exposed as a slider. Not a universal truth. |
| `contribution_margin_inr` | 250 | Entirely merchant-specific. The default exists to reproduce the specification's worked example. |
| Causal driver weights in `generator.yaml` | see file | The generator's own assumptions, stated openly so a reviewer can attack them. The model never sees these values. |

## A note on citation precision

Live URLs belong in this table. They are omitted at Phase 1 rather than
reproduced from memory, because a plausible-looking URL that does not resolve is
worse than an attributed figure without one — it looks verified and is not. They
will be added when each source is re-checked directly, before any of these
figures is used in a published claim.
