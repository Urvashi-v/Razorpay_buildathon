# RTO Sentinel — evaluation report

Generated 2026-09-01 03:58 UTC by `rto-sentinel evaluation-report`. **Every figure is read from a saved artefact under `artifacts/`; none is written by hand.**

> **The labels are simulated.** They are outcomes of the process in `docs/simulator.md`, not observations of real returns. Every metric here is a statement about that simulator. A good number means the model learned the simulator; whether the simulator resembles Indian COD commerce is a separate question this project does not claim to have settled.

---

## How to read this

| Column | What it is |
|---|---|
| **Validation** | **Selection-contaminated.** Hyperparameters were chosen on this split and the shipped calibrator was refitted on it. Useful for comparison, not for a performance claim. |
| **Sealed test** | **The honest read.** Opened once, after model selection, calibration and threshold methodology were frozen in the manifest. |

† Rests on `intervention_success_rate` and `abandonment_on_friction`, which are **stated assumptions, never measured** on this or any data. Every rupee figure inherits their uncertainty on top of the sampling interval shown.

---

## 1. What was measured

Every artefact records all seven identifiers, so any figure below can be traced to the exact code, data and configuration that produced it.

| Identifier | Value |
|---|---|
| Model | `lightgbm` + `platt` calibration |
| Model version | `a0d780424b79` |
| Selection manifest | `4f17cd1f1279d897d589` |
| Dataset run | `7b5ae86219ac7cafe45e7d51` |
| Generator version | `1.0.0` |
| Feature version | `1.0.0` |
| Feature fingerprint | `798aef57ad3cefe9…` |
| Config fingerprint | `f3943b6937cf0b36…` |
| Seed | `7` |
| Calibration fitted on | validation (5-fold CV) |
| Threshold | 0.3481 — derived from cost inputs: C_fp=70.50, S_tp=132.00 |
| Cost profile | `mid_margin_d2c` |
| Frozen at | 2026-08-29 02:17 UTC |

### Split sizes

| Split | Orders | Positives | Positive rate | Customers |
|---|---:|---:|---:|---:|
| validation | 2,034 | 388 | 0.1908 | 1,529 |
| test | 1,698 | 266 | 0.1567 | 1,180 |

---

## 2. Headline

On the sealed test set, at the cost-derived threshold 0.3481:

- **Net ₹716 <sub>[₹-1,009, ₹2,603]</sub> per 1,000 orders**, against doing nothing†
- Doing nothing costs the merchant ₹34,464 per 1,000 orders in absorbed RTO losses
- Flag rate 15.9%, precision 0.370, recall 0.376
- PR-AUC 0.365 <sub>[0.315, 0.426]</sub> against a base rate of 0.1567

> **The interval crosses zero.** On 1,698 sealed orders this measurement cannot distinguish the model from doing nothing. The point estimate is positive; the evidence does not establish it. This is the single most important sentence in the report and it is stated first rather than buried under the figures that look better.

---

## 3. Ranking quality

**PR-AUC leads.** ROC-AUC is reported but not led with: it flatters an imbalanced problem by rewarding the model for ranking the large negative class correctly, which is not the task.

| Metric | Validation<br><sub>selection-contaminated</sub> | Sealed test<br><sub>the honest read</sub> |
|---|---|---|
| PR-AUC | 0.484 <sub>[0.439, 0.536]</sub> | 0.365 <sub>[0.315, 0.426]</sub> |
| ROC-AUC | 0.806 <sub>[0.785, 0.827]</sub> | 0.781 <sub>[0.755, 0.807]</sub> |
| Base rate (PR-AUC floor) | 0.1908 | 0.1567 |
| Recall @ precision 80% | 0.036 | — |
| Recall @ precision 90% | 0.010 | — |
| Precision @ top 1% | 0.750 | 0.588 |
| Precision @ top 5% | 0.667 | 0.518 |
| Precision @ top 10% | 0.522 | 0.429 |

**Recall@P80 and Recall@P90 are em-dashes on the sealed set because the model never reaches those precisions at any threshold.** That is a real result about the ceiling of this problem, not a missing measurement: with a base rate near 16% and substantial irreducible noise in the simulator, 80% precision is not attainable. Reporting 0.0 would have claimed it was attained and missed.

---

## 4. Calibration

Calibration is a headline metric here, not a footnote. The operating threshold is compared against a probability; if the score is not an honest probability, the comparison is arithmetic on a number that denotes nothing and every rupee figure below is fiction.

| Metric | Validation<br><sub>selection-contaminated</sub> | Sealed test<br><sub>the honest read</sub> |
|---|---|---|
| Expected calibration error (calibrated) | 0.0136 | 0.0290 |
| Expected calibration error (uncalibrated) | 0.0285 | 0.0175 |
| Brier score (calibrated) | 0.1239 | 0.1162 |
| Brier score (uncalibrated) | 0.1247 | 0.1152 |

> **Calibration improved ECE on validation by +0.0150 and made it worse on the sealed set by -0.0115.** The Platt mapping was fitted on validation by cross-validation and did not transfer. This is reported because it is the kind of result that quietly disappears from a write-up: the calibrator was selected honestly, and it still failed to generalise.

### Reliability diagrams

Generated at evaluation time from the same predictions as the table above:

- `E:/Razorpay Buildthon/artifacts/final/7b5ae86219ac7cafe45e7d51/reliability__validation.png`
- `E:/Razorpay Buildthon/artifacts/final/7b5ae86219ac7cafe45e7d51/reliability__test.png`

The console renders the same bins live from `GET /v1/evaluation/reliability`, so the picture and the endpoint cannot disagree.

---

## 5. The operating point and the confusion matrix

The threshold is **0.3481**, not 0.5. It is derived from the merchant's economics as `C_fp / (C_fp + S_tp)` and recomputed whenever they change - see `docs/economics.md`.

| Metric | Validation<br><sub>selection-contaminated</sub> | Sealed test<br><sub>the honest read</sub> |
|---|---|---|
| Threshold | 0.3481 | 0.3481 |
| Flag rate | 19.1% | 15.9% |
| Precision | 0.482 | 0.370 |
| Recall | 0.482 | 0.376 |
| F1 | 0.482 | 0.373 |
| True positives | 187 | 100 |
| False positives | 201 | 170 |
| False negatives | 201 | 166 |
| True negatives | 1,445 | 1,262 |

**Precision is never reported without the flag rate.** A precision figure alone is meaningless: any model can reach high precision by flagging almost nothing, and the pair is what describes the operating point.

---

## 6. Economic results

† Rests on `intervention_success_rate` and `abandonment_on_friction`, which are **stated assumptions, never measured** on this or any data. Every rupee figure inherits their uncertainty on top of the sampling interval shown.

| Metric | Validation<br><sub>selection-contaminated</sub> | Sealed test<br><sub>the honest read</sub> |
|---|---|---|
| **Net ₹ per 1,000 orders**† | ₹5,169 <sub>[₹3,279, ₹7,173]</sub> | ₹716 <sub>[₹-1,009, ₹2,603]</sub> |
| Do-nothing loss per 1,000 | ₹41,967 | ₹34,464 |
| Gross saving† | ₹24,684 | ₹13,200 |
| **False-positive cost**† | ₹14,170 | ₹11,985 |
| Residual RTO loss (not flagged) | ₹44,220 | ₹36,520 |

**The false-positive cost is reported separately and is never netted away inside the savings figure.** `EconomicResult` keeps it as a required field with nowhere to hide it, because a net number that has quietly absorbed the cost of frictioning good customers is the most flattering thing this system could report.

Full derivation, the friction ladder, the threshold sweep and the sensitivity analysis: `docs/economics.md`.

---

## 7. Baseline comparison

Every rung on the **validation** split, at the same cost-derived threshold (0.3481), scored identically. The question a baseline ladder answers is not "is the model good" but "does it beat the thing a merchant could build in an afternoon" - and if a simpler rung wins on money, it ships.

| Rung | Model | PR-AUC | Train-val gap | Flag rate | Precision | Net ₹/1k† |
|---:|---|---|---:|---:|---:|---:|
| 0 | `do_nothing` | 0.191 <sub>[0.173, 0.206]</sub> | — | 0.000 | — | ₹0 |
| 1 | `blanket_cod_block` | 0.234 <sub>[0.209, 0.262]</sub> | — | 0.088 | 0.417 | ₹1,228 |
| 2 | `pincode_blocklist` | 0.196 <sub>[0.177, 0.214]</sub> | — | 0.048 | 0.265 | ₹-808 |
| 3 | `logistic_regression` | 0.483 <sub>[0.439, 0.535]</sub> | — | 0.146 | 0.468 | ₹3,544 |
| 4 | `lightgbm` | 0.443 <sub>[0.397, 0.501]</sub> | — | 0.140 | 0.465 | ₹3,298 |
| — | **`lightgbm_platt`** (shipped) | **0.484 <sub>[0.439, 0.536]</sub>** | — | 0.191 | 0.482 | **₹5,169** |

**A large positive train-val gap means the rung memorised the training window**; its validation score describes an overfitting configuration rather than a ceiling. That is why the ladder is reported with the gap beside the score.

---

## 8. Fairness across operational cohorts

**No sensitive characteristic is examined, inferred or approximated.** There is no gender, religion, caste, ethnicity, age or income field in this data - not withheld, not present - and none is derived from names or addresses. `eval/fairness.py` refuses by name any cohort matching a sensitive token, and the refusal is a hard error rather than a skipped cohort.

The cohorts are operational: delivery-area tier, order-value band, customer-history depth, payment method.

### Validation split

**Disparity review: not triggered.** Maximum flag-rate ratio 3.15 (order_value_band=v4 [1,135-8,481) vs order_value_band=v1 [39-420)); worst precision drop 0.055.

| Cohort | Group | n | RTO rate | Flag rate | Precision | Evidence |
|---|---|---:|---:|---:|---:|---|
| pincode_tier | tier_1 | 701 | 14.4% | 12.7% | 0.461 | yes |
| pincode_tier | tier_2 | 751 | 20.0% | 18.1% | 0.456 | yes |
| pincode_tier | tier_3 | 582 | 23.5% | 28.0% | 0.515 | yes |
| order_value_band | v1 [39-420) | 509 | 14.7% | 10.6% | 0.519 | yes |
| order_value_band | v2 [420-698) | 508 | 18.1% | 12.8% | 0.492 | yes |
| order_value_band | v3 [698-1,135) | 508 | 19.9% | 19.5% | 0.455 | yes |
| order_value_band | v4 [1,135-8,481) | 509 | 23.6% | 33.4% | 0.482 | yes |
| customer_history_band | frequent (10+ prior) | 213 | 20.2% | 22.5% | 0.458 | yes |
| customer_history_band | light (1-2 prior) | 770 | 18.6% | 21.3% | 0.482 | yes |
| customer_history_band | new (0 prior) | 257 | 19.8% | 10.9% | 0.357 | **too thin** |
| customer_history_band | regular (3-9 prior) | 794 | 19.0% | 18.6% | 0.514 | yes |
| payment_method | cod | 1,225 | 29.8% | 31.7% | 0.482 | yes |
| payment_method | prepaid | 809 | 2.8% | 0.0% | — | **too thin** |

The most-flagged group with sufficient support is order_value_band=v4 [1,135-8,481), flagged 3.15x as often as order_value_band=v1 [39-420). The precision gap between the most-flagged group and the best-performing group in its cohort is 0.055. This does not trip the configured review, which requires BOTH a ratio above 1.5 AND a precision drop above 0.1. A higher flag rate on its own is not a finding: a group that returns more parcels should be flagged more often, and equalising flag rates would make the system worse at its job while looking fairer. It is, however, 55% of the way to the precision-drop trigger. That is a margin worth re-checking after the next retrain rather than treating as settled. 2 group(s) fell below the minimum support of 100 and are shown in the table but excluded from this comparison in both directions.

### Test split

**Disparity review: not triggered.** Maximum flag-rate ratio 3.57 (order_value_band=v4 [1,180-6,551) vs order_value_band=v1 [73-425)); worst precision drop 0.071.

| Cohort | Group | n | RTO rate | Flag rate | Precision | Evidence |
|---|---|---:|---:|---:|---:|---|
| pincode_tier | tier_1 | 564 | 11.7% | 9.8% | 0.327 | yes |
| pincode_tier | tier_2 | 687 | 18.0% | 15.6% | 0.374 | yes |
| pincode_tier | tier_3 | 447 | 17.0% | 24.2% | 0.389 | yes |
| order_value_band | v1 [73-425) | 425 | 13.4% | 8.2% | 0.286 | yes |
| order_value_band | v2 [425-703) | 424 | 13.4% | 12.5% | 0.358 | yes |
| order_value_band | v3 [703-1,180) | 424 | 15.8% | 13.4% | 0.439 | yes |
| order_value_band | v4 [1,180-6,551) | 425 | 20.0% | 29.4% | 0.368 | yes |
| customer_history_band | frequent (10+ prior) | 271 | 12.5% | 10.7% | 0.345 | **too thin** |
| customer_history_band | light (1-2 prior) | 536 | 14.7% | 20.0% | 0.336 | yes |
| customer_history_band | new (0 prior) | 153 | 19.0% | 9.8% | 0.400 | **too thin** |
| customer_history_band | regular (3-9 prior) | 738 | 16.8% | 16.1% | 0.403 | yes |
| payment_method | cod | 1,036 | 24.1% | 26.1% | 0.370 | yes |
| payment_method | prepaid | 662 | 2.4% | 0.0% | — | **too thin** |

The most-flagged group with sufficient support is order_value_band=v4 [1,180-6,551), flagged 3.57x as often as order_value_band=v1 [73-425). The precision gap between the most-flagged group and the best-performing group in its cohort is 0.071. This does not trip the configured review, which requires BOTH a ratio above 1.5 AND a precision drop above 0.1. A higher flag rate on its own is not a finding: a group that returns more parcels should be flagged more often, and equalising flag rates would make the system worse at its job while looking fairer. It is, however, 71% of the way to the precision-drop trigger. That is a margin worth re-checking after the next retrain rather than treating as settled. 3 group(s) fell below the minimum support of 100 and are shown in the table but excluded from this comparison in both directions.

Full audit, with Wilson intervals on every rate and the support thresholds: `docs/responsible_ai.md`.

---

## 9. Distribution shift

Nine named perturbations of the generator, with the model **frozen and not retrained** and the threshold held fixed. The `reference` environment is a fresh draw from the *unshifted* distribution, so subtracting it removes sampling variance and leaves the effect of the perturbation.

**Read the lift column, not raw PR-AUC.** A random ranker scores PR-AUC equal to the positive rate, so an environment whose base rate moved has a different floor. Raw PR-AUC *rises* when the world gets riskier, and reading that as robustness reports the arithmetic of the base rate as a property of the model.

| Environment | RTO rate | PR-AUC | Lift | ΔLift | ECE | Net ₹/1k† |
|---|---:|---:|---:|---:|---:|---:|
| `reference` | 16.7% | 0.430 | 2.57x | — | 0.025 | ₹2,276 |
| `cod_surge` | 21.4% | 0.465 | 2.17x | -0.40x | 0.034 | ₹4,122 |
| `cod_collapse` | 10.2% | 0.432 | 4.25x | +1.68x | 0.013 | ₹1,707 |
| `rto_base_rate_up` | 23.7% | 0.560 | 2.37x | -0.20x | 0.038 | ₹8,730 |
| `rto_base_rate_down` | 9.6% | 0.299 | 3.10x | +0.53x | 0.083 | ₹-2,130 |
| `category_mix_fashion` | 16.8% | 0.447 | 2.67x | +0.10x | 0.031 | ₹3,088 |
| `customer_mix_new` | 16.8% | 0.441 | 2.63x | +0.07x | 0.027 | ₹2,968 |
| `geography_tier3` | 16.6% | 0.437 | 2.63x | +0.07x | 0.041 | ₹2,281 |
| `order_value_up` | 17.1% | 0.448 | 2.62x | +0.05x | 0.026 | ₹2,861 |
| `combined_festive` | 26.7% | 0.551 | 2.07x | -0.50x | 0.015 | ₹9,267 |

### Findings

- Reference: PR-AUC 0.430 at a base rate of 16.7%, which is a lift of 2.57x over chance. Comparisons below use lift, because raw PR-AUC is not comparable across environments whose base rates differ - a random ranker scores PR-AUC equal to the positive rate.
- combined_festive: ranking lift fell by 0.50x to 2.07x (raw PR-AUC 0.551 at a 26.7% base rate). The model ranks orders less well when cod share, rto base rate and fashion share all rise together, as they plausibly would during a festive peak
- cod_surge: ranking lift fell by 0.40x to 2.17x (raw PR-AUC 0.465 at a 21.4% base rate). The model ranks orders less well when cod share rises from 62% to 80%, as in a festive-season shift
- rto_base_rate_up: ranking lift fell by 0.20x to 2.37x (raw PR-AUC 0.560 at a 23.7% base rate). The model ranks orders less well when the cod rto base rate rises from 26% to 38%
- rto_base_rate_down: calibration error rose by 0.058 to 0.083. This is the more serious failure mode: the threshold is compared against a probability, so a miscalibrated score makes every rupee figure downstream wrong even where ranking held up.
- rto_base_rate_down: net economics turned non-positive (INR -2,130 per 1,000 orders). Under this shift the system stops paying for itself at the frozen threshold.
- 4 of 9 shifted environments left ranking lift within 0.15x of the reference: category_mix_fashion, customer_mix_new, geography_tier3, order_value_up.

---

## 10. Drift monitoring

**Drift is not failure.** A moved input distribution is a fact about the world; whether quality degraded needs labels. The two are kept structurally apart - a drift signal has no field for a verdict, and a performance delta cannot be constructed without matured outcomes.

Baseline 1,220 orders (1,220 matured) versus current 814 (814 matured). Labelled comparison possible: **yes**.

| Kind | Quantity | Statistic | Distance | Reading |
|---|---|---|---:|---|
| prediction | `predicted_probability` | psi | 0.0115 | stable |
| prediction | `predicted_probability` | ks | 0.0420 | stable |
| feature | `discount_depth` | psi | 0.5474 | investigate |
| feature | `is_cod` | psi | 0.0000 | stable |
| feature | `order_value_inr` | psi | 0.1016 | watch |
| feature | `prior_rto_rate` | psi | 0.0084 | stable |
| flag_rate | `flag_rate` | absolute_difference | 0.0170 | stable |
| outcome_rate | `rto_rate` | absolute_difference | 0.0149 | stable |
| calibration | `expected_calibration_error` | absolute_difference | 0.0115 | stable |

### Labelled comparisons

| Metric | Baseline | Current | Δ |
|---|---:|---:|---:|
| pr_auc | 0.4966 | 0.4664 | -0.0302 |
| precision | 0.4772 | 0.4898 | +0.0126 |
| recall | 0.4792 | 0.4865 | +0.0073 |
| brier_score | 0.1264 | 0.1201 | -0.0063 |

---

## 11. What this evaluation does and does not establish

**Established:**

- The pipeline is leak-free under seven explicit tests, and the sealed test set was opened exactly once, after the manifest was frozen.
- The model ranks materially better than chance on the sealed set (PR-AUC well above the base rate).
- The decision threshold is derived from merchant economics, not chosen, and it moves in the direction the arithmetic requires.
- The cohort audit did not trip its disparity review, and the shift study found a specific, reproducible failure mode.

**Not established:**

- **That the system saves money.** The sealed-set interval [₹-1,009, ₹2,603] crosses zero.
- **Anything about production.** The labels are simulated; these are statements about the simulator.
- **Any fairness property of a protected group.** No such attribute exists in the data and none was inferred.
- **The rupee figures†**, which rest on two rates that have never been measured.
- **That any feature family earns its place.** The leave-one-family-out ablation is an unimplemented interface and has never been run.

Complete limitations: `docs/phase11_report.md` and `docs/responsible_ai.md`.

