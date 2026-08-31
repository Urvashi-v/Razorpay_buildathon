# Responsible AI and robustness report

Generated 2026-08-31 14:29 UTC by `rto-sentinel responsible-report`. Every number below is read from a saved artefact under `artifacts/responsible/`; none is written by hand.

> **Read this first.**
> Controlled benchmark experiment on synthetic data.
> Cohorts, shifts and drift windows are properties of the documented simulator in docs/simulator.md, not observations of real customers, real distribution shift, or real production behaviour.
> These results are evidence that the audit machinery works and that this model behaves a certain way on this benchmark.
> They are NOT evidence of production fairness or production robustness, and no such claim should be made from them.

---

## 1. Fairness across operational cohorts

### What was and was not examined

The cohorts are operational: delivery-area tier, order-value band, customer-history depth, and payment method. Every one is a fact recorded on the order.

**No sensitive characteristic was examined, inferred, or approximated.** There is no gender, religion, caste, ethnicity, age or income field in this data - not withheld, not present - and none is derived from names, addresses or any other field. `eval/fairness.py` refuses to group by a column whose name matches a sensitive token, and that refusal is a hard error rather than a skipped cohort, so a misconfigured audit fails loudly instead of quietly examining less than it claims.

Pincode tier is the closest thing here to a proxy for something sensitive, and that is exactly why it is the headline cohort rather than an omitted one. A delivery-area tier is an operational fact about logistics and it is also correlated with income. Auditing it openly is the alternative to pretending the correlation is not there.

### Results on the validation split

Model `a0d780424b79` at threshold 0.3481, minimum support 100 orders.

| Cohort | Group | n | RTO rate | Flag rate | Precision | Recall | Net INR/1k | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pincode_tier | tier_1 | 701 | 14.4% <sub>[0.12, 0.17]</sub> | 12.7% <sub>[0.10, 0.15]</sub> | 0.461 <sub>[0.36, 0.56]</sub> | 0.406 | ₹2,893 | yes |
| pincode_tier | tier_2 | 751 | 20.0% <sub>[0.17, 0.23]</sub> | 18.1% <sub>[0.16, 0.21]</sub> | 0.456 <sub>[0.37, 0.54]</sub> | 0.413 | ₹3,951 | yes |
| pincode_tier | tier_3 | 582 | 23.5% <sub>[0.20, 0.27]</sub> | 28.0% <sub>[0.25, 0.32]</sub> | 0.515 <sub>[0.44, 0.59]</sub> | 0.613 | ₹9,482 | yes |
| order_value_band | v1 [39-420) | 509 | 14.7% <sub>[0.12, 0.18]</sub> | 10.6% <sub>[0.08, 0.14]</sub> | 0.519 <sub>[0.39, 0.65]</sub> | 0.373 | ₹3,660 | yes |
| order_value_band | v2 [420-698) | 508 | 18.1% <sub>[0.15, 0.22]</sub> | 12.8% <sub>[0.10, 0.16]</sub> | 0.492 <sub>[0.37, 0.61]</sub> | 0.348 | ₹3,735 | yes |
| order_value_band | v3 [698-1,135) | 508 | 19.9% <sub>[0.17, 0.24]</sub> | 19.5% <sub>[0.16, 0.23]</sub> | 0.455 <sub>[0.36, 0.55]</sub> | 0.446 | ₹4,199 | yes |
| order_value_band | v4 [1,135-8,481) | 509 | 23.6% <sub>[0.20, 0.27]</sub> | 33.4% <sub>[0.29, 0.38]</sub> | 0.482 <sub>[0.41, 0.56]</sub> | 0.683 | ₹9,077 | yes |
| customer_history_band | frequent (10+ prior) | 213 | 20.2% <sub>[0.15, 0.26]</sub> | 22.5% <sub>[0.17, 0.29]</sub> | 0.458 <sub>[0.33, 0.60]</sub> | 0.512 | ₹5,028 | yes |
| customer_history_band | light (1-2 prior) | 770 | 18.6% <sub>[0.16, 0.21]</sub> | 21.3% <sub>[0.19, 0.24]</sub> | 0.482 <sub>[0.41, 0.56]</sub> | 0.552 | ₹5,760 | yes |
| customer_history_band | new (0 prior) | 257 | 19.8% <sub>[0.15, 0.25]</sub> | 10.9% <sub>[0.08, 0.15]</sub> | 0.357 <sub>[0.21, 0.54]</sub> | 0.196 | ₹198 | too thin |
| customer_history_band | regular (3-9 prior) | 794 | 19.0% <sub>[0.16, 0.22]</sub> | 18.6% <sub>[0.16, 0.21]</sub> | 0.514 <sub>[0.43, 0.59]</sub> | 0.503 | ₹6,242 | yes |
| payment_method | cod | 1,225 | 29.8% <sub>[0.27, 0.32]</sub> | 31.7% <sub>[0.29, 0.34]</sub> | 0.482 <sub>[0.43, 0.53]</sub> | 0.512 | ₹8,582 | yes |
| payment_method | prepaid | 809 | 2.8% <sub>[0.02, 0.04]</sub> | 0.0% <sub>[0.00, 0.00]</sub> | —  | 0.000 | ₹0 | too thin |

**Disparity review: not triggered.** Maximum flag-rate ratio 3.15 (order_value_band=v4 [1,135-8,481) vs order_value_band=v1 [39-420)); worst precision drop 0.055.

The most-flagged group with sufficient support is order_value_band=v4 [1,135-8,481), flagged 3.15x as often as order_value_band=v1 [39-420). The precision gap between the most-flagged group and the best-performing group in its cohort is 0.055. This does not trip the configured review, which requires BOTH a ratio above 1.5 AND a precision drop above 0.1. A higher flag rate on its own is not a finding: a group that returns more parcels should be flagged more often, and equalising flag rates would make the system worse at its job while looking fairer. It is, however, 55% of the way to the precision-drop trigger. That is a margin worth re-checking after the next retrain rather than treating as settled. 2 group(s) fell below the minimum support of 100 and are shown in the table but excluded from this comparison in both directions.

Groups shown but excluded from the comparison for insufficient support: customer_history_band=new (0 prior) (257 orders), payment_method=prepaid (809 orders).

### Results on the test split

Model `a0d780424b79` at threshold 0.3481, minimum support 100 orders.

| Cohort | Group | n | RTO rate | Flag rate | Precision | Recall | Net INR/1k | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pincode_tier | tier_1 | 564 | 11.7% <sub>[0.09, 0.15]</sub> | 9.8% <sub>[0.08, 0.12]</sub> | 0.327 <sub>[0.22, 0.46]</sub> | 0.273 | ₹-412 | yes |
| pincode_tier | tier_2 | 687 | 18.0% <sub>[0.15, 0.21]</sub> | 15.6% <sub>[0.13, 0.18]</sub> | 0.374 <sub>[0.29, 0.47]</sub> | 0.323 | ₹810 | yes |
| pincode_tier | tier_3 | 447 | 17.0% <sub>[0.14, 0.21]</sub> | 24.2% <sub>[0.20, 0.28]</sub> | 0.389 <sub>[0.30, 0.48]</sub> | 0.553 | ₹1,993 | yes |
| order_value_band | v1 [73-425) | 425 | 13.4% <sub>[0.10, 0.17]</sub> | 8.2% <sub>[0.06, 0.11]</sub> | 0.286 <sub>[0.16, 0.45]</sub> | 0.175 | ₹-1,041 | yes |
| order_value_band | v2 [425-703) | 424 | 13.4% <sub>[0.11, 0.17]</sub> | 12.5% <sub>[0.10, 0.16]</sub> | 0.358 <sub>[0.24, 0.49]</sub> | 0.333 | ₹262 | yes |
| order_value_band | v3 [703-1,180) | 424 | 15.8% <sub>[0.13, 0.20]</sub> | 13.4% <sub>[0.11, 0.17]</sub> | 0.439 <sub>[0.32, 0.57]</sub> | 0.373 | ₹2,462 | yes |
| order_value_band | v4 [1,180-6,551) | 425 | 20.0% <sub>[0.16, 0.24]</sub> | 29.4% <sub>[0.25, 0.34]</sub> | 0.368 <sub>[0.29, 0.46]</sub> | 0.541 | ₹1,182 | yes |
| customer_history_band | frequent (10+ prior) | 271 | 12.5% <sub>[0.09, 0.17]</sub> | 10.7% <sub>[0.08, 0.15]</sub> | 0.345 <sub>[0.20, 0.53]</sub> | 0.294 | ₹-72 | too thin |
| customer_history_band | light (1-2 prior) | 536 | 14.7% <sub>[0.12, 0.18]</sub> | 20.0% <sub>[0.17, 0.24]</sub> | 0.336 <sub>[0.25, 0.43]</sub> | 0.456 | ₹-473 | yes |
| customer_history_band | new (0 prior) | 153 | 19.0% <sub>[0.14, 0.26]</sub> | 9.8% <sub>[0.06, 0.16]</sub> | 0.400 <sub>[0.20, 0.64]</sub> | 0.207 | ₹1,029 | too thin |
| customer_history_band | regular (3-9 prior) | 738 | 16.8% <sub>[0.14, 0.20]</sub> | 16.1% <sub>[0.14, 0.19]</sub> | 0.403 <sub>[0.32, 0.49]</sub> | 0.387 | ₹1,803 | yes |
| payment_method | cod | 1,036 | 24.1% <sub>[0.22, 0.27]</sub> | 26.1% <sub>[0.23, 0.29]</sub> | 0.370 <sub>[0.31, 0.43]</sub> | 0.400 | ₹1,173 | yes |
| payment_method | prepaid | 662 | 2.4% <sub>[0.01, 0.04]</sub> | 0.0% <sub>[0.00, 0.01]</sub> | —  | 0.000 | ₹0 | too thin |

**Disparity review: not triggered.** Maximum flag-rate ratio 3.57 (order_value_band=v4 [1,180-6,551) vs order_value_band=v1 [73-425)); worst precision drop 0.071.

The most-flagged group with sufficient support is order_value_band=v4 [1,180-6,551), flagged 3.57x as often as order_value_band=v1 [73-425). The precision gap between the most-flagged group and the best-performing group in its cohort is 0.071. This does not trip the configured review, which requires BOTH a ratio above 1.5 AND a precision drop above 0.1. A higher flag rate on its own is not a finding: a group that returns more parcels should be flagged more often, and equalising flag rates would make the system worse at its job while looking fairer. It is, however, 71% of the way to the precision-drop trigger. That is a margin worth re-checking after the next retrain rather than treating as settled. 3 group(s) fell below the minimum support of 100 and are shown in the table but excluded from this comparison in both directions.

Groups shown but excluded from the comparison for insufficient support: customer_history_band=frequent (10+ prior) (271 orders), customer_history_band=new (0 prior) (153 orders), payment_method=prepaid (662 orders).

---

## 2. Distribution shift

Each environment is a **named change to a generator parameter**, not a fresh draw from the same distribution. Regenerating with a new seed and calling that robustness would measure sampling variance: every draw would come from the distribution the model was trained on.

The `reference` environment is itself a fresh draw from the *unshifted* distribution, generated the same way as every other environment and differing only in that it applies no overrides. That is what makes the deltas meaningful: sampling variance is present in the reference too, so subtracting it leaves the effect of the perturbation rather than the effect of having generated new data. A study that compared shifted worlds against the original training run would be measuring both at once.

The model (`a0d780424b79`) is **not retrained** between environments, and the threshold is held fixed at 0.3481. Re-deriving the threshold per environment would repair part of the damage and understate what a deployed model suffers - in production the threshold is a configuration value and does not follow the distribution around.

### Environments

| Environment | What changed | Overrides |
|---|---|---|
| `reference` | the unshifted world, generated with the configured parameters | _none (control)_ |
| `cod_surge` | COD share rises from 62% to 80%, as in a festive-season shift | `payment.cod_share=0.8` |
| `cod_collapse` | COD share falls to 35% as prepaid adoption accelerates | `payment.cod_share=0.35` |
| `rto_base_rate_up` | the COD RTO base rate rises from 26% to 38% | `base_rates.rto_given_cod=0.38` |
| `rto_base_rate_down` | the COD RTO base rate falls from 26% to 15% | `base_rates.rto_given_cod=0.15` |
| `category_mix_fashion` | the catalogue swings towards fashion, the highest-return category | `catalogue.categories.accessories.share=0.08`, `catalogue.categories.beauty.share=0.12`, `catalogue.categories.electronics.share=0.08`, `catalogue.categories.fashion.share=0.6`, `catalogue.categories.home.share=0.12` |
| `customer_mix_new` | the book fills with first-time customers, so history features go null | `customers.orders_per_customer_alpha=4.5` |
| `geography_tier3` | delivery mix shifts towards tier-3 pincodes | `geography.tier_shares.tier_1=0.2`, `geography.tier_shares.tier_2=0.3`, `geography.tier_shares.tier_3=0.5` |
| `order_value_up` | basket sizes rise; the median order value roughly doubles | `catalogue.order_value.mu=7.55` |
| `combined_festive` | COD share, RTO base rate and fashion share all rise together, as they plausibly would during a festive peak | `base_rates.rto_given_cod=0.34`, `catalogue.categories.accessories.share=0.1`, `catalogue.categories.beauty.share=0.16`, `catalogue.categories.electronics.share=0.1`, `catalogue.categories.fashion.share=0.48`, `catalogue.categories.home.share=0.16`, `payment.cod_share=0.78` |

### Measured degradation

**Read the lift column, not the raw PR-AUC column.** A random ranker scores PR-AUC equal to the positive rate, so an environment whose base rate moved hands the model a different floor for free. In `rto_base_rate_up` the raw PR-AUC rises, and reading that as robustness would be reporting the arithmetic of the base rate as a property of the model. Lift divides the floor out.

| Environment | n | RTO rate | PR-AUC | Lift | ΔLift | ECE | ΔECE | Flag rate | Precision | Net ₹/1k | ΔNet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reference` | 8,766 | 16.7% | 0.430 | 2.57x | — | 0.025 | — | 18.1% | 0.410 | ₹2,276 | — |
| `cod_surge` | 8,782 | 21.4% | 0.465 | 2.17x | -0.40x | 0.034 | +0.009 | 24.8% | 0.430 | ₹4,122 | +1,846 |
| `cod_collapse` | 8,805 | 10.2% | 0.432 | 4.25x | +1.68x | 0.013 | -0.012 | 8.5% | 0.447 | ₹1,707 | -569 |
| `rto_base_rate_up` | 8,767 | 23.7% | 0.560 | 2.37x | -0.20x | 0.038 | +0.013 | 21.6% | 0.548 | ₹8,730 | +6,454 |
| `rto_base_rate_down` | 8,797 | 9.6% | 0.299 | 3.10x | +0.53x | 0.083 | +0.058 | 15.0% | 0.278 | ₹-2,130 | -4,406 |
| `category_mix_fashion` | 8,789 | 16.8% | 0.447 | 2.67x | +0.10x | 0.031 | +0.005 | 19.4% | 0.427 | ₹3,088 | +812 |
| `customer_mix_new` | 8,764 | 16.8% | 0.441 | 2.63x | +0.07x | 0.027 | +0.002 | 18.1% | 0.429 | ₹2,968 | +692 |
| `geography_tier3` | 8,727 | 16.6% | 0.437 | 2.63x | +0.07x | 0.041 | +0.016 | 21.4% | 0.401 | ₹2,281 | +5 |
| `order_value_up` | 8,758 | 17.1% | 0.448 | 2.62x | +0.05x | 0.026 | +0.001 | 19.0% | 0.423 | ₹2,861 | +585 |
| `combined_festive` | 8,741 | 26.7% | 0.551 | 2.07x | -0.50x | 0.015 | -0.011 | 28.3% | 0.510 | ₹9,267 | +6,991 |

### Findings

- Reference: PR-AUC 0.430 at a base rate of 16.7%, which is a lift of 2.57x over chance. Comparisons below use lift, because raw PR-AUC is not comparable across environments whose base rates differ - a random ranker scores PR-AUC equal to the positive rate.
- combined_festive: ranking lift fell by 0.50x to 2.07x (raw PR-AUC 0.551 at a 26.7% base rate). The model ranks orders less well when cod share, rto base rate and fashion share all rise together, as they plausibly would during a festive peak
- cod_surge: ranking lift fell by 0.40x to 2.17x (raw PR-AUC 0.465 at a 21.4% base rate). The model ranks orders less well when cod share rises from 62% to 80%, as in a festive-season shift
- rto_base_rate_up: ranking lift fell by 0.20x to 2.37x (raw PR-AUC 0.560 at a 23.7% base rate). The model ranks orders less well when the cod rto base rate rises from 26% to 38%
- rto_base_rate_down: calibration error rose by 0.058 to 0.083. This is the more serious failure mode: the threshold is compared against a probability, so a miscalibrated score makes every rupee figure downstream wrong even where ranking held up.
- rto_base_rate_down: net economics turned non-positive (INR -2,130 per 1,000 orders). Under this shift the system stops paying for itself at the frozen threshold.
- 4 of 9 shifted environments left ranking lift within 0.15x of the reference: category_mix_fashion, customer_mix_new, geography_tier3, order_value_up.

---

## 3. Monitoring and drift

**Drift is not failure.** A moved input distribution is a fact about the world; whether quality degraded is a separate question that needs labels. The two are kept structurally apart: a drift signal has no field in which to record a verdict, and a performance delta cannot be constructed without mature outcomes.

This matters operationally. Indian e-commerce moves hard during festive season - COD share rises, order values rise, category mix swings. Every one of those shows up as real drift and none of them means the model stopped working. A monitor that pages someone every Diwali gets muted by March.

Baseline: 1,220 orders (1,220 matured). Current: 814 orders (814 matured). Labelled comparison possible: **yes**.

| Kind | Quantity | Statistic | Baseline | Current | Distance | Reading |
|---|---|---|---:|---:|---:|---|
| prediction | `predicted_probability` | psi | 0.195 | 0.184 | 0.0115 | stable |
| prediction | `predicted_probability` | ks | 0.195 | 0.184 | 0.0420 | stable |
| feature | `discount_depth` | psi | 0.216 | 0.322 | 0.5474 | investigate |
| feature | `is_cod` | psi | 0.609 | 0.592 | 0.0000 | stable |
| feature | `order_value_inr` | psi | 1011.711 | 797.715 | 0.1016 | watch |
| feature | `prior_rto_rate` | psi | 0.173 | 0.175 | 0.0084 | stable |
| flag_rate | `flag_rate` | absolute_difference | 0.198 | 0.181 | 0.0170 | stable |
| outcome_rate | `rto_rate` | absolute_difference | 0.197 | 0.182 | 0.0149 | stable |
| calibration | `expected_calibration_error` | absolute_difference | 0.012 | 0.023 | 0.0115 | stable |

### Labelled comparisons

| Metric | Baseline | Current | Δ | Evidence |
|---|---:|---:|---:|---|
| pr_auc | 0.4966 | 0.4664 | -0.0302 | yes |
| precision | 0.4772 | 0.4898 | +0.0126 | yes |
| recall | 0.4792 | 0.4865 | +0.0073 | yes |
| brier_score | 0.1264 | 0.1201 | -0.0063 | yes |

### Warnings

- Measured change: pr_auc moved from 0.4966 to 0.4664 (-0.0302) on 814 matured orders. This is a labelled comparison, so it is evidence about model quality rather than a distribution shift.
- 2 input feature(s) moved between the windows: discount_depth (PSI 0.547), order_value_inr (PSI 0.102). Input drift is expected in a seasonal business and is not by itself a problem. It becomes one when it coincides with measured degradation or calibration movement above.

---

## 4. Limitations

**These are controlled benchmark experiments, not evidence of production fairness or production robustness.** Stated plainly because this is the sentence most likely to be dropped when results are quoted onward.

1. **The labels are simulated.** Every outcome here was drawn from the causal process in `docs/simulator.md`. A cohort disparity measured on this data is a property of that process. If the simulator makes tier-3 riskier - and it does, by an explicit `tier_risk_offset` - then a model that flags tier-3 more is recovering a fact the simulator put there, not discovering one about India.

2. **The cohorts are operational, not demographic.** This audit cannot answer whether the system disadvantages any protected group, because no such attribute exists in the data and inferring one would be worse than not asking. A production deployment would need a separate, consented, legally-reviewed process for that question.

3. **The shift environments are the ones we thought of.** Robustness against nine named perturbations is not robustness in general. The failure modes that matter most in production are usually the ones nobody enumerated - a courier changing its scanning behaviour, a checkout redesign changing session features, an upstream field going null.

4. **Drift bands are conventions, not calibrated thresholds.** The PSI bands (0.10 / 0.25) come from credit-risk practice. They were not tuned against a labelled history of this system's incidents, because no such history exists. They select the words "watch" and "investigate" rather than "warn" and "fail" for exactly that reason.

5. **The audit ran on one dataset run and one model version.** Both are recorded in the artefacts. Neither result transfers automatically to a retrained model, and re-running the audit is part of shipping one.

6. **2 cohort group(s) were too small to support a conclusion.** They are shown in the table because suppressing them would hide exactly what an audit exists to look at, but they are excluded from the disparity comparison in both directions - they cannot fire the trigger and they cannot hold down a ratio that would otherwise have fired it.

