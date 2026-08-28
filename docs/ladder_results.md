# Baseline ladder results

**Generated from measured predictions. Do not edit by hand.**

> Every number below was computed by `rto_sentinel.eval` from actual model predictions against held-out data. No metric in this project is written as a literal anywhere in source.

## Run provenance

| Field | Value |
|---|---|
| Evaluated split | `validation` |
| Dataset run | `7b5ae86219ac7cafe45e7d51` |
| Config fingerprint | `f3943b6937cf0b36...` |
| Feature fingerprint | `798aef57ad3cefe9...` |
| Seed | `7` |
| Cost profile | `mid_margin_d2c` |
| Operating threshold | **0.3481** |
| Threshold source | derived from cost inputs: C_fp=70.50, S_tp=132.00 |
| Generated at | 2026-08-28T18:05:13.591020+00:00 |
| Train rows | 23,058 (days 1-126) |
| Evaluation rows | 2,034 (days 127-147) |
| Evaluation positive rate | 0.1908 |
| Features | 54 across 6 families |

## Comparison

PR-AUC leads because it is not inflated by the large negative class. ROC-AUC is reported but deliberately not led with - it flatters imbalanced problems. Precision is always adjacent to flag rate, because precision without it is a half-truth.

`Trn-val` is training PR-AUC minus validation PR-AUC. A large positive gap means the rung has memorised the training split, and its validation score should be read as the score of an overfitting model rather than as a ceiling on what that model family can do. A small negative gap is unremarkable - it means the validation window happened to be slightly easier.

| Rung | Model | PR-AUC (95% CI) | Trn-val | Flag rate | Precision | Recall | F1 | R@P80 | ROC-AUC | ECE | Net ₹/1k | FP cost ₹ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `do_nothing` | **0.191** [0.173, 0.206] | -0.023 | 0.000 | — | 0.000 | — | — | — | 0.023 | 0 | 0 |
| 1 | `blanket_cod_block` | **0.234** [0.209, 0.262] | -0.032 | 0.088 | 0.417 | 0.193 | 0.264 | — | 0.565 | 0.206 | 1,228 | 7,402 |
| 2 | `pincode_blocklist` | **0.196** [0.177, 0.214] | +0.018 | 0.048 | 0.265 | 0.067 | 0.107 | — | 0.512 | 0.213 | -808 | 5,076 |
| 3 | `logistic_regression` | **0.483** [0.439, 0.535] | -0.043 | 0.146 | 0.468 | 0.358 | 0.406 | 0.098 | 0.799 | 0.024 | 3,544 | 11,139 |
| 4 | `lightgbm` | **0.443** [0.397, 0.501] | +0.519 | 0.140 | 0.465 | 0.340 | 0.393 | 0.044 | 0.790 | 0.056 | 3,298 | 10,716 |

The do-nothing baseline absorbs **₹41,967 per 1,000 orders**. Net ₹/1k above is the saving *relative to that*, so a rung scoring 0 has changed nothing and a negative figure means the intervention costs more than it saves.

## Which rung is strongest

On the declared headline metric - net rupees per 1,000 orders - **rung 3 `logistic_regression`** wins, at **₹3,544** [1,867, 5,307] per 1,000 orders at a 14.6% flag rate. It also leads on PR-AUC (0.483).

`lightgbm` is the rung to be careful with. It scores 0.962 PR-AUC on the training split against 0.443 on validation - a gap of +0.519 - so it has substantially memorised the training window. Its validation score describes an overfitting configuration, not the ceiling of what that model family can reach.

This ordering is a measurement on **this synthetic benchmark**, and the simulator's own structure is part of what is being measured. It is not evidence about which model family wins on real RTO data.

## What each rung answers

| Rung | Question |
|---|---|
| 0 `do_nothing` | What happens without intervention? Defines the loss absorbed today. |
| 1 `blanket_cod_block` | What happens under a maximally aggressive policy? |
| 2 `pincode_blocklist` | Can a simple location-based rule perform meaningfully? |
| 3 `logistic_regression` | What does a simple, interpretable model achieve? |
| 4 `lightgbm` | What does the stronger nonlinear model achieve? |

## Calibration status

**Every rung here is uncalibrated.** ECE is reported as a *diagnostic*, not as a claim that any rung produces honest probabilities. Isotonic calibration on the validation fold is Phase 5; the model cards carry `calibration_method: null`, and the decision engine refuses a score whose calibration method is null - so none of these models can currently reach a decision.

## Data provenance

Synthetic benchmark data. Labels are simulated outcomes of the documented process in [docs/simulator.md](simulator.md), not real-world ground truth. **Absolute metric values are not a claim about production performance.**
