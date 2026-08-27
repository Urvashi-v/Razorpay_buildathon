# RTO Sentinel — Results

**This document is deliberately empty of results.**

No model has been trained. No dataset has been generated. The sealed test set has
never been scored. Writing provisional numbers here — even clearly labelled ones —
is how a placeholder becomes a claim three commits later, so there are none.

## What will go here

Per the specification, this report will contain, whichever way the numbers land:

### The ladder, scored identically

Every rung on the same sealed test set, against the same cost metric:

| Rung | Model | PR-AUC | ECE | Flag rate | FP cost (₹) | Net ₹/1,000 |
|------|-------|--------|-----|-----------|-------------|-------------|
| 0 | Ship everything, flag nothing | — | — | — | — | — |
| 1 | Blanket COD block above ₹X | — | — | — | — | — |
| 2 | Pincode blocklist (top decile) | — | — | — | — | — |
| 3 | Logistic regression | — | — | — | — | — |
| 4 | LightGBM + isotonic | — | — | — | — | — |

If a simpler rung wins on net rupees, it ships, and this table will say so.

### Calibration

Reliability diagram, Expected Calibration Error, Brier score. Reported as a
headline metric, because the entire decision layer depends on the score being an
honest probability. If the model says 0.30 and the true rate for that bucket is
0.55, the rupee numbers are fiction.

### Economics

Net ₹ saved per 1,000 orders at the cost-optimal threshold, with a bootstrap
confidence interval, alongside total false-positive cost stated separately, and
the flag rate. Plus the sensitivity analysis: how fast the savings degrade if the
cost inputs are wrong by 30%.

### Fairness audit

Flag rate and precision by pincode tier and order-value band, whether or not the
disparity trigger fires, and what was done about it if it did.

### Robustness

Performance on the final two weeks alone (drift), cohort breakdowns (new vs
returning, COD vs prepaid, by value quartile), and the leave-one-family-out
ablation.

### Limitations

Stated plainly, including:

- the synthetic-data ceiling — absolute metrics reflect the generator's
  assumptions, not reality;
- the counterfactual problem — once friction is applied, the true outcome of that
  order is never observed;
- cold start for a new merchant with no pincode priors;
- adaptation over festival cycles and courier changes;
- that the 60% intervention success rate in the cost model is an assumption from
  published studies, not something measured here.

## The rules this report will be held to

From `config/evaluation.yaml`, enforced by the report builder, which raises
rather than renders a report that violates any of them:

- no single accuracy figure
- ROC-AUC reported but never led with
- the threshold never tuned on the test set
- precision never quoted without the flag rate
- false-positive cost never netted away inside a savings number
- no point estimate presented without an interval

## Test-set discipline

The test set is scored **exactly once**, at the end, after the threshold has been
fixed on validation. `data/splits.py::TestSetSeal` writes a receipt on first use
and refuses a second run. When this document is filled in, the receipt will exist
at `artifacts/reports/test_set_seal.json` and its fingerprint will be quoted here.
