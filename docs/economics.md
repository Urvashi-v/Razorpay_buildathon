# Economic evaluation

**Generated from the scored book and the configuration. Do not edit by hand.**

> Every rupee figure below is arithmetic over declared inputs applied to measured model output. The model output was measured on **synthetic** labels, and two of the inputs are **assumptions nobody has measured**. Neither fact is a reason not to compute these numbers; both are reasons to read them with the provenance table in hand.

## The decision rule

```
C_fp      = abandonment_on_friction x contribution_margin + friction_support_cost
S_tp      = intervention_success_rate x rto_cost
threshold = C_fp / (C_fp + S_tp)
```

At the `mid_margin_d2c` profile:

```
C_fp      = 0.25 x 250.0 + 8.0 = 70.50
S_tp      = 0.6 x 220.0 = 132.00
threshold = 70.50 / (70.50 + 132.00) = 0.3481
```

**Not 0.5.** The threshold is a function of the merchant's economics and moves when they move; the simulation section below shows it moving. No absolute probability is hardcoded anywhere in the policy.

Scored book: `validation` split, 2,034 orders, engine version `1.0.0`.

## Where every number comes from

Five kinds of number appear below and they are not equally established. This table separates them; every rupee total further down inherits the weakest provenance among its inputs.

| Quantity | Value | Kind | Source |
|---|---|---|---|
| `rto_cost_inr` | 220 INR | merchant-provided input | config/cost_model.yaml or the merchant's own figure |
| `contribution_margin_inr` | 250 INR | merchant-provided input | config/cost_model.yaml or the merchant's own figure |
| `abandonment_on_friction` | 0.25 probability | ASSUMED intervention effectiveness, never measured **[ASSUMPTION]** | No measurement exists. Requires a controlled holdout to establish. |
| `intervention_success_rate` | 0.6 probability | ASSUMED intervention effectiveness, never measured **[ASSUMPTION]** | No measurement exists. Requires a controlled holdout to establish. |
| `cost_of_false_positive` | 70.5 INR | derived by arithmetic **[ASSUMPTION]** | C_fp = abandonment x margin + support cost |
| `saving_per_true_positive` | 132 INR | derived by arithmetic **[ASSUMPTION]** | S_tp = intervention success x RTO cost |
| `operating_threshold` | 0.3481 probability | derived by arithmetic **[ASSUMPTION]** | threshold = C_fp / (C_fp + S_tp). Never 0.5, never fitted to labels. |
| `flag_rate` | 0.1908 probability | derived by arithmetic | Share of the scored book at or above the derived threshold |
| `expected_false_positive_cost` | 5,622 INR | derived by arithmetic **[ASSUMPTION]** | Reported separately and never netted away |
| `net_inr_saved_per_1000_orders` | 4,648 INR/1000 orders | derived by arithmetic **[ASSUMPTION]** | TP x S_tp - FP x C_fp, measured against doing nothing |

**`abandonment_on_friction`, `intervention_success_rate` have never been measured.** They cannot be measured without running the interventions and observing what would otherwise have happened, which is what the randomised control slice in `config/policy.yaml` exists to make possible. Until that has run, every rupee figure in this document is arithmetic over an unverified rate, and the sensitivity section below is the closest thing to a bound on how wrong it could be.

## The intervention ladder at this threshold

Band cut points are multipliers on the derived threshold, so the whole ladder moves when the merchant's economics move. No absolute probability is configured anywhere.

| Band | Action | Range | Orders | Share | Expected RTOs | Observed RTOs | Assumed success | Expected net INR |
|---|---|---|---|---|---|---|---|---|
| LOW | `none` | [0.0000, 0.3481) | 1,646 | 80.9% | 201.4 | 201 | 0.000 | 0 |
| ELEVATED | `prepaid_nudge` | [0.3481, 0.5570) | 304 | 14.9% | 131.6 | 131 | 0.270 | 4,241 |
| HIGH | `confirmation_required` | [0.5570, 0.8356) | 84 | 4.1% | 55.0 | 56 | 0.600 | 5,213 |
| SEVERE | `prepaid_only` | [0.8356, 1.0) | 0 | 0.0% | 0.0 | 0 | 0.780 | 0 |

**SEVERE received no orders.** The tier exists in configuration and cannot fire at this threshold on this book - no order scores high enough. Reported rather than hidden: a ladder rung that never fires is a rung the merchant is not actually using.

## Economic outcome

**Expected** figures are computed from calibrated probabilities alone and need no labels - that is what a merchant can compute on today's unlabelled book. **Realized** figures are measured against observed outcomes. The gap between them is a calibration check.

| Quantity | Expected | Realized |
|---|---|---|
| Orders scored | 2,034 | 2,034 |
| Flag rate (at or above threshold) | 0.1908 | 0.1908 |
| Intervention rate (any action) | 0.1908 | 0.1908 |
| Orders affected | 388 | 388 |
| Savings, INR | 15,076 | — |
| False-positive cost, INR | 5,622 | — |
| Residual false-negative loss, INR | 70,281 | — |
| Total cost, INR | 75,903 | — |
| **Net INR per 1,000 orders** | **4,648** | **4,725** |
| True positives | 186.6 | 187 |
| Precision | — | 0.482 |
| Recall | — | 0.482 |

The do-nothing baseline absorbs **INR 41,967 per 1,000 orders**. Net is the saving relative to that, so zero means the policy changed nothing and a negative figure means the friction costs more than it saves.

**Calibration check.** The probabilities predicted 186.6 true positives among frictioned orders; 187 occurred, a gap of -0.4. That is small relative to the volume, so the expected figures can be read as meaning roughly what they say.

**A known understatement.** The specification folds the per-friction support cost into `C_fp`, which charges it only to false positives. Ops pays it on every frictioned order, so a further **INR 703** of support spend on true positives is not reflected in the net figure above. The spec's formula is kept rather than silently improved; the omission is reported instead.

**After the control slice.** 2.0% of flagged orders receive no friction so that precision stays measurable once the system acts. Net after withholding that slice: **INR 4,555 per 1,000 orders**. The difference is the price of continuing to know whether the model works.

## Threshold analysis

> **The operating threshold is DERIVED from merchant economics as C_fp / (C_fp + S_tp) and is never read off this curve. The curve is a diagnostic: it shows what other operating points would cost, how flat the region around the derived point is, and where the peak sits. Selecting the peak would fit the operating point to the evaluation labels, which is what the derivation exists to avoid. The sweep is computed on validation only; running it on the sealed test split is refused.**

Derived operating point: **0.3481**. The curve peaks at **0.3200**.

| Threshold | Flag rate | Precision | Recall | F1 | Expected cost INR | Expected net INR/1k | Realized net INR/1k |
|---|---|---|---|---|---|---|---|
| 0.0100 | 1.000 | 0.191 | 1.000 | 0.320 | 150,187 | -31,873 | -31,872 |
| 0.1100 | 0.585 | 0.303 | 0.928 | 0.456 | 96,456 | -5,457 | -5,406 |
| 0.2100 | 0.381 | 0.381 | 0.760 | 0.508 | 80,794 | 2,244 | 2,542 |
| 0.3100 | 0.235 | 0.459 | 0.567 | 0.507 | 75,271 | 4,959 | 5,300 |
| 0.3481 **<-- operating point** | 0.191 | 0.482 | 0.482 | 0.482 | 74,924 | 5,130 | 5,169 |
| 0.4000 | 0.135 | 0.498 | 0.353 | 0.413 | 75,555 | 4,819 | 4,108 |
| 0.5000 | 0.067 | 0.610 | 0.214 | 0.317 | 78,295 | 3,472 | 3,549 |
| 0.6000 | 0.028 | 0.719 | 0.106 | 0.184 | 81,403 | 1,944 | 2,106 |
| 0.7000 | 0.011 | 0.783 | 0.046 | 0.088 | 83,520 | 903 | 995 |
| 0.8000 | 0.001 | 1.000 | 0.005 | 0.010 | 85,172 | 91 | 130 |
| 0.9000 | 0.000 | — | 0.000 | — | 85,357 | 0 | 0 |

The derived point sits 0.0281 from the peak. That is close, which is reassuring but not a validation: the two agreeing means the merchant's stated economics happen to match what the labels imply, and the derivation would still be the right choice if they did not.

## Sensitivity: how wrong can the assumptions be?

The threshold is a function of four inputs, two of which are assumptions. This is how far it moves when each is wrong by up to 30%.

| Parameter | -30% | -15% | baseline | +15% | +30% |
|---|---|---|---|---|---|
| `rto_cost_inr` | 0.4328 | 0.3859 | 0.3481 | 0.3171 | 0.2912 |
| `contribution_margin_inr` | 0.2816 | 0.3165 | 0.3481 | 0.3770 | 0.4034 |
| `abandonment_on_friction` | 0.2816 | 0.3165 | 0.3481 | 0.3770 | 0.4034 |
| `intervention_success_rate` | 0.4328 | 0.3859 | 0.3481 | 0.3171 | 0.2912 |

A threshold that swings widely under a 30% error in an assumed rate is a threshold resting on that assumption. The two intervention rates are exactly the ones nobody has measured.

## Is the graduated ladder worth having?

The ladder applies a gentler action to the lower flagged band and a stronger one above. Whether graduation pays is a question about the assumed multipliers, so it is answered here rather than assumed.

| Policy | Net INR per 1,000 orders |
|---|---|
| Graduated ladder (configured) | 4,648 |
| Uniform `confirmation_required` above threshold | 5,130 |
| Uniform `prepaid_nudge` above threshold | 3,395 |
| Uniform `prepaid_only` above threshold | 742 |

**The graduated ladder loses.** Applying `confirmation_required` uniformly to everything above the threshold is worth INR 482 per 1,000 orders more. The mechanism is visible in the ladder table: the gentlest rung carries most of the flagged volume and is assumed to convert least often, so graduating downwards gives up more saving than it avoids in abandonment. This is a finding about the assumed multipliers, not about reality - but under the assumptions the system actually ships with, the simpler policy is better, and that is worth saying rather than burying.

Both figures rest on the same ASSUMED intervention multipliers in config/policy.yaml. This comparison tests whether graduation pays under those assumptions; it is not evidence about which policy is better in reality.

## Merchant simulation

Each row below was produced by the same server-side recomputation the API exposes at `POST /v1/economics/simulate`: new economics in, new threshold, new band boundaries, new per-order assignment, new rupee totals out. Nothing is scaled from a cached result and nothing is computed in a browser.

| Scenario | Margin INR | RTO cost INR | Threshold | Flag rate | Orders affected | Net INR/1k | Rungs that fire |
|---|---|---|---|---|---|---|---|
| Mid-margin D2C brand (spec worked example) | 250 | 220 | 0.3481 | 0.191 | 388 | 4,648 | ELEVATED, HIGH, SEVERE |
| Thin-margin reseller - flags MORE (friction is cheap here) | 90 | 180 | 0.2612 | 0.296 | 602 | 6,154 | ELEVATED, HIGH, SEVERE |
| High-margin beauty brand - flags LESS (a lost order costs more) | 520 | 240 | 0.3944 | 0.141 | 286 | 4,034 | ELEVATED, HIGH, SEVERE |
| Margin changed to INR 250 | 250 | 220 | 0.3481 | 0.191 | 388 | 4,648 | ELEVATED, HIGH, SEVERE |
| Margin changed to INR 400 | 400 | 220 | 0.4500 | 0.096 | 195 | 2,199 | ELEVATED, HIGH |

A higher margin raises the threshold - losing a good customer costs more, so the bar for frictioning one rises - and the flag rate falls with it. That direction is a property of the formula, asserted in `tests/unit/test_decision_engine.py`, not an artefact of this particular book.

## Safeguards

- No hard block exists at any rung: `hard_block_allowed` is `False` and `Decision` refuses `appeal_available=False` at construction.
- SEVERE routes to a human review queue and carries an appeal path.
- Ops overrides are enabled (True) and logged (True); overrides are counterfactual evidence, not noise.
- A randomised 2% of flagged orders receives no friction, so precision remains measurable after the system starts acting. Those orders are exempt from review as well - routing them to a human would destroy the counterfactual they exist to preserve.

## What this document does not establish

- **That the interventions work.** Their effectiveness is assumed. The rupee figures scale linearly with those assumptions and would move accordingly.
- **That these numbers transfer to a real merchant.** The labels are simulated; the model was measured on a synthetic benchmark.
- **That the flagged population is fair.** The cohort audit defined in `config/evaluation.yaml` has not been run.
