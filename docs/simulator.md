# The RTO Sentinel simulator

**What this document is:** a complete description of how the benchmark data is
produced, and an explicit separation between four things that are routinely
blurred together in synthetic-data projects.

| Category | What it means here | Where it lives |
|---|---|---|
| **External published assumptions** | Figures taken from public sources about real Indian e-commerce. Cited, not invented. | [docs/sources.md](sources.md) |
| **Simulator assumptions** | Choices *this project* made about causal structure and magnitudes. Not evidence. Stated so they can be attacked. | [`config/generator.yaml`](../config/generator.yaml) |
| **Simulated labels** | The RTO outcomes this generator produces. **Not real-world ground truth.** | `orders.is_rto` |
| **Measured model results** | What a model scores on this data. A statement about the simulator, not about production. | [REPORT.md](../REPORT.md) — currently empty |

Nothing in the first column is a claim by this project. Nothing in the third
column is a fact about the world. The distinction is load-bearing: a metric
measured on simulated labels is evidence about the *pipeline*, and only becomes
evidence about *reality* after validation on a real merchant's history.

---

## 1. What the generator is, and is not

It is a **controlled benchmark generator**: a documented causal process that emits
order metadata and samples a probabilistic RTO outcome, calibrated so its marginal
rates match published Indian figures.

It is **not**:

- a source of ground truth about any real merchant, customer, or address;
- a model of any specific company's book;
- capable of producing working payment credentials, valid identity documents, or
  deliverable addresses.

Customer identifiers are SHA-256 digests of a synthetic index and the run seed —
there is no pre-image because there is no real identity behind them. Pincodes are
synthetic six-digit identifiers from a generated pool, not a map of real Indian
postcodes. Street names come from a ten-item vocabulary. The whole of
[`data/address.py`](../src/rto_sentinel/data/address.py) is short enough to read
in a minute, which is the point.

---

## 2. The generative process, step by step

Implemented in [`data/generator.py`](../src/rto_sentinel/data/generator.py),
version `1.0.0`. The version is recorded on every dataset; an unknown version is
refused rather than silently falling back.

### Step 1 — Populations

Drawn once per run.

**Customers** (`n_customers`) each receive three latent traits, all Beta-distributed:

| Trait | Distribution | Meaning | Observable? |
|---|---|---|---|
| `reliability` | Beta(6.0, 2.2) | Propensity to accept a delivery | **No** |
| `address_quality` | Beta(3.2, 1.8) | How completely they write addresses | Partially |
| `prepaid_affinity` | Beta(2.0, 2.4) | Preference for prepaid over COD | Partially |

Activity weights come from a Pareto(1.8) tail, **clipped at the 99th percentile**.
Unclipped, a single customer takes a double-digit share of the book — which no
real store sees, and which would let one person's latent reliability dominate the
entire dataset.

**Pincodes** (`n_pincodes`) get a tier and a latent Gaussian random effect
(σ = 0.35). **Couriers** get a share and a latent lane quality.

### Step 2 — The order stream

Timestamps are sampled across the horizon using:

- **hour-of-day weights** — bimodal, with a lunchtime bump, a large evening peak,
  and a deliberate late-night tail;
- **a weekend uplift** (×1.15);
- **a sale-day calendar** (×2.4 volume, +0.18 discount depth).

The stream is then **sorted by time**, and customers are assigned by activity
weight. Sorting is what makes each customer's orders arrive chronologically,
which is what makes the as-of computation in step 3 correct without any later
sorting or filtering.

### Step 3 — Per-order, in chronological order

For each order, in time order:

**a. As-of history.** The customer's history is filtered to orders that had
already **resolved** before this order's timestamp. Not orders placed earlier —
*resolved* earlier. An order placed on day 40 that comes back on day 47 was not
known to be an RTO on day 42, and does not appear.

Cancellations are excluded from delivery history: an order cancelled before
dispatch was never presented to the customer and carries no delivery signal.

**b. Order attributes.** Payment method (from the customer's prepaid affinity,
centred so the realised COD share matches the configured one), value, discount
(deepened on sale days), category, courier, session signals.

**c. The latent logit.** A linear combination of the documented drivers:

```
logit =  w_address        × (deliverability − 0.5) × 2
       + w_prior_rto      × prior_rto_rate            (as-of)
       + w_discount       × discount_depth
       + w_value_z        × order_value_zscore
       + w_late_night     × is_late_night
       + w_tier           × tier_risk_offset
       + w_new_customer   × is_new_customer
       + w_fast_checkout  × fast_checkout
       + w_courier        × courier_lane_quality      (latent)
       + w_reliability    × (customer_reliability − 0.5) × 2   (LATENT)
       + w_pincode        × pincode_effect                     (LATENT)
       + category_offset
       + N(0, 0.90)                                            (irreducible)
       + intercept                                             (see step 4)
```

**Deliverability** is address writing quality minus a penalty (0.35) when the
typed city disagrees with the pincode's city. This is a *causal* penalty, not a
cosmetic one — a parcel addressed to the wrong city genuinely struggles to
arrive, and it is one of the most common structural defects in Indian address
data.

**d. The label.** A Bernoulli draw at `sigmoid(logit)`, then a symmetric flip at
rate 0.004 standing in for courier miscoding. Real outcome data is not perfectly
recorded, and a benchmark with a noiseless label overstates what any model can
achieve.

**e. Fulfilment timeline.** Dispatch (4–36h), first attempt (1–4 days), then
resolution — with RTOs resolving 2–5 days later than deliveries, because failed
attempts, reattempts and the return leg all take time. Total resolution is capped
at `max_resolution_days` (9).

**f. History entry, keyed by resolution time.** The outcome joins the customer's
history at its *resolution* instant, so a later order by the same customer sees it
only if it had genuinely come back first.

### Step 4 — Base-rate calibration

The per-payment-method intercepts are found by a **fixed-point iteration over the
whole simulation pass**, not a closed-form solve.

This is necessary because the simulation contains a genuine feedback loop: a
customer's prior RTO rate is itself an input to their next order's risk. Change
the intercept and the histories change, which changes the logits, which changes
the marginal. Iterating the whole pass is the only way to land on the published
base rate *with* that loop intact — and the loop is worth keeping, because
"history predicts the future" is the single most important honest signal in this
problem.

The iteration is deterministic: all randomness is drawn up front, so a pass is a
pure function of the intercepts. It converges in two or three passes.

The target is corrected for the label-flip rate, so the **observed** marginal
matches the published figure rather than the pre-noise one.

### Step 5 — Label maturity

An order whose resolution falls beyond the horizon end gets:

- `outcome = "pending"`
- `is_rto = NULL`
- `split = "excluded_immature"`

It is **never** optimistically labelled "delivered". That substitution is the
single most common way a benchmark manufactures optimism: it converts every
not-yet-returned order into a clean negative.

---

## 3. Why the task is not reverse-engineerable

A simulator whose every driver is visible to the model is a deterministic rule
waiting to be recovered, and a model trained on it reports a score that means
nothing. Four things prevent that here:

1. **Three latent drivers.** `customer_reliability` (weight −1.40) and
   `pincode_effect` (weight 1.00) are never exposed as columns. The model sees a
   courier *name*, not its lane quality.
2. **Partial observability of address quality.** The simulator uses latent
   deliverability; the model sees *rendered text signals* — token count, house
   number present, city/pincode consistency. The gap between them is real noise,
   not a lossless encoding. A test asserts the correlation stays between 0.05 and
   0.95.
3. **A per-order Gaussian shock** (σ = 0.90) on the logit.
4. **The Bernoulli draw itself**, plus the 0.4% label flip.

Together these create a genuine Bayes-optimal ceiling well below perfect
separation. `GeneratorConfig` refuses to load a configuration in which every
driver is observable.

**A model that scores near-perfectly on this data has found a bug, not a signal.**

---

## 4. What the simulator deliberately makes true

Stated plainly, because these are assumptions and a reader should be able to
disagree with them.

| Relationship | Direction | Tested in |
|---|---|---|
| Prior RTO history → future RTO | Strongest honest signal | `test_prior_rto_history_predicts_future_rto` |
| Missing house number → higher risk | Positive | `test_worse_addresses_return_more_often` |
| City/pincode mismatch → higher risk | Positive | `test_pincode_city_inconsistency_raises_risk` |
| Deeper discount → higher risk | Positive | `test_deeper_discounts_return_more_often` |
| Tier-3 pincode → higher risk | Positive | `test_tier_3_pincodes_carry_more_risk` |
| COD vs prepaid | ~26% vs ~2% | `test_prepaid_rto_rate_is_far_below_cod` |
| Sale days | More volume, deeper discounts | `test_sale_days_carry_more_volume` |

### On the tier-3 gradient specifically

The simulator **does** make tier-3 pincodes riskier on average
(`tier_risk_offset: +0.45` versus `−0.25` for tier-1). This is not hidden, and it
is the mechanism the Phase 4 fairness audit exists to scrutinise.

The audit's question is *not* whether the gradient exists — it does, by
construction. The question is whether a model trained on it **transfers cost onto
tier-3 customers beyond what its precision justifies**. A model that flags tier-3
twice as often with materially worse precision has found poverty and bad
municipal addressing, not fraud.

---

## 5. Assumptions that are NOT sourced

The most important table in this document.

| Assumption | Value | Status |
|---|---|---|
| `intervention_success_rate` | 0.60 | From published intervention studies, **not measured here**. Replacing it with a measured number requires A/B testing the friction ladder. |
| `abandonment_on_friction` | 0.25 | Merchant-specific input, exposed as a console slider. Not a universal truth. |
| `contribution_margin_inr` | 250 | Entirely merchant-specific. The default reproduces the specification's worked example. |
| All `causal_drivers` weights | see config | The simulator's own assumptions. No model ever sees these values. |
| `logit_sigma`, `pincode_effect_sigma` | 0.90, 0.35 | Chosen to produce a plausible difficulty level. Not estimated from anything. |
| `label_flip_rate` | 0.004 | A stand-in for courier miscoding. Order of magnitude only. |

---

## 6. Reproducibility

The same configuration and seed produce the same data. This is asserted from both
directions in
[`tests/unit/test_generator_reproducibility.py`](../tests/unit/test_generator_reproducibility.py):
identical inputs give identical frames, and changing *any* input — seed, customer
count, order count, either date — changes the `run_id`.

Every dataset records:

| Field | Purpose |
|---|---|
| `seed` | The random seed |
| `generator_version` | Which generative process produced it |
| `config_snapshot` | The full configuration, as loaded |
| `config_fingerprint` | SHA-256 over that configuration |
| `created_at` | Wall-clock creation time |
| `run_id` | Deterministic digest of the parameters |

`run_id` being deterministic means regenerating a dataset is an **upsert**, not a
second near-identical copy silently doubling every count.

To regenerate exactly:

```bash
rto-sentinel generate --seed 20260827 --customers 45000 --orders 120000 --start-date 2025-09-01 --end-date 2026-02-27
```

---

## 7. The split protocol, and what it costs

Two rules are in genuine tension: temporal splits and customer-disjoint splits.
With customers active across the whole horizon, satisfying both loses data.

The resolution — **partition customers into disjoint pools first, then apply the
temporal window within each pool** — and the reasoning behind rejecting the two
obvious alternatives is documented at the top of
[`data/splits.py`](../src/rto_sentinel/data/splits.py).

The short version: the naive "assign each customer to their earliest split"
approach satisfies both rules and is *badly biased*. Measured on a 20,000-order
sample it produced 43% first-time customers in train against 87% and 88% in
validation and test. A threshold fitted on that is a cold-start threshold.

Pool assignment is a deterministic hash of the customer identifier and a fixed
salt — independent of behaviour, so every split keeps the population's
new-versus-repeat mix.

**The cost is roughly half the dataset**, reported explicitly in
`SplitAssignment` rather than hidden. That is the honest price of enforcing both
rules, and it is why the default dataset is large.

A residual difference remains and *should*: later windows contain more returning
customers, because the customer base matures over the horizon. That is real
temporal drift, not selection bias, and it points the opposite way to the bug it
replaced.

---

## 8. What this data can and cannot demonstrate

**Can honestly demonstrate:**

- that the pipeline is leak-free — the four leakage tests run against real
  generated data;
- that the cost-optimal threshold behaves as the theory predicts;
- that a model beats sensible baselines under identical conditions;
- that calibration can be measured against a *known* true probability, which real
  data can never offer;
- that the system degrades gracefully.

**Cannot demonstrate:**

- any absolute performance number that transfers to production;
- that the causal structure resembles real RTO causation;
- that the fairness properties of a real deployment would match these;
- anything at all about a specific merchant.

Real RTO has messier causes and a fatter tail. The only way to know whether any
of this holds is to validate on a real merchant's 90-day history before trusting
a single number.

---

## 9. Defense-only compliance

The generator is the one component in this repository that deserves scrutiny
under a defense-only rule, so it is named explicitly.

It produces order records with a probabilistic RTO label drawn from published
aggregate base rates. It does **not** produce working payment credentials, valid
identity documents, deliverable fake addresses, realistic synthetic identities, or
anything usable outside this repository's own evaluation harness. It contains no
adversarial or evasion-testing capability, and nothing in it transfers to
attacking any system.

It is a labelled tabular sampler, and the code is short enough for a reviewer to
verify that.
