# RTO Sentinel API

Hand-written, unlike the results documents in this repository — the machine-readable
contract lives at `/openapi.json` and is generated. This page explains what the
endpoints mean, what they refuse to do, and why.

Every example below is a real request and a real response captured against
PostgreSQL with a trained artefact loaded. Nothing here is illustrative.

```bash
rto-sentinel serve          # or: uvicorn rto_sentinel.api.main:app
```

Interactive docs at `/docs`, the schema at `/openapi.json`.

---

## The one thing to understand first

**The API never invents a number.** There is no default probability, no
last-known score, no "model unavailable so assume low risk". When any link in the
scoring chain is missing, the endpoint fails with a code that says which link:

| Situation | Status | Code |
|---|---|---|
| No calibrated model artefact on disk | 503 | `MODEL_UNAVAILABLE` |
| Model trained on a different feature set than the server runs | 409 | `MODEL_UNAVAILABLE` |
| Score reached the engine uncalibrated | 409 | `UNCALIBRATED_SCORE` |
| Order not in the database | 404 | `ORDER_NOT_FOUND` |
| No decision logged for that order | 404 | `DECISION_NOT_FOUND` |
| Bad input — pagination, economics, override reason | 422 | `VALIDATION_FAILED` |
| Fairness audit requested, never run | 501 | `NOT_IMPLEMENTED` |

`tests/api/test_serving_integration.py::test_the_api_fails_when_the_model_is_missing`
asserts the first row, and is written to fail if a score comes back instead.

Errors share one envelope and never carry a stack trace, a file path, a query or
a connection string:

```json
{"error": {"code": "ORDER_NOT_FOUND", "message": "no order 'ORD-99999999' in the database", "detail": {"order_id": "ORD-99999999", "dataset_run": null}}}
```

---

## The inference chain

`GET /v1/orders/{order_id}/risk` executes all of this on every call:

```
database row          ServingRepository       the order, its address, its outcome
  -> feature service  OrderFeatureService     merchant history -> training pipeline -> one row
  -> model            ModelRegistry           the frozen artefact, checksum + fingerprint verified
  -> calibration      inside the artefact     Platt, fitted on validation
  -> probability      RiskScore               calibrated P(RTO), with provenance
  -> decision engine  DecisionEngine          cost-derived threshold, friction ladder
  -> response
```

Nothing is cached between the order and the decision. The only cached thing is
the deserialised model artefact, which is loaded once per process and reported by
version on every response.

### Why scoring takes a reference, not a payload

There is no endpoint that accepts an arbitrary order object and scores it. The
features this model needs are not all on the order: customer history and
geography aggregates are computed from the merchant's book *as of that order's
own timestamp*. An order that has never been persisted has no history, so scoring
it would silently treat every customer as a first-time buyer in an unknown
pincode — and return a confident number for it.

Ingesting new orders is a separate concern from scoring stored ones. Conflating
them is how a serving path starts lying.

### Latency, honestly

**About 0.8–1.9 seconds per order** on the benchmark database. The feature
pipeline reads the merchant's book (8,874 rows for the example below) to rebuild
the aggregates, because that is what the features genuinely require.

A production system would precompute those aggregates into a feature store and
read one row. This does not, and reports the cost rather than hiding it. The
window is bounded by `RTO_SERVING_CONTEXT_LIMIT` (default 20,000); a lower value
is faster and gives the geography features less evidence, so they shrink harder
towards their prior — never wrong, only less informed.

---

## Orders

### `GET /v1/orders`

Newest first. `limit` is capped at 200 — an uncapped page is a denial-of-service
surface, not a convenience.

Filters: `merchant_id`, `customer_hash`, `split`, `payment_method` (`cod` |
`prepaid`), `dataset_run`, `limit`, `offset`.

```bash
curl -s "localhost:8000/v1/orders?limit=2&split=test&payment_method=cod"
```

```json
{
  "orders": [
    {
      "order_id": "ORD-00008874",
      "merchant_id": "M-DEMO-001",
      "customer_hash": "546e7a386f62c4f41bc0d00dc4c45bb5",
      "ordered_at": "2026-02-25T17:00:20Z",
      "payment_method": "cod",
      "is_cod": true,
      "order_value_inr": 3902.0,
      "discount_inr": 972.85,
      "item_count": 1,
      "category": "electronics",
      "courier_partner": "courier_b",
      "split": "test",
      "dataset_run_id": "d1efe6b75393a8f95dac4c6a",
      "is_rto": false,
      "outcome": "delivered",
      "resolved_at": "2026-02-26T21:00:20Z"
    }
  ],
  "total": 322,
  "limit": 2,
  "offset": 0
}
```

`is_rto` is `null` until the order resolves. It is **never** defaulted to
`false`: an immature order has no outcome, and saying otherwise manufactures
optimism in exactly the direction that flatters the model.

### Order ids are scoped to a dataset run

Each generator run numbers its orders from `ORD-00000001`, so a database holding
two benchmark runs holds two different orders with the same id. Migration
`4f1c2a7d8e30` scoped the uniqueness constraint accordingly. Pass `dataset_run`
to disambiguate; without it the most recent run wins, which is defined rather
than arbitrary.

### `GET /v1/orders/{order_id}` — one order

Same shape as a list row, with the outcome joined.

---

## Risk

### `GET /v1/orders/{order_id}/risk`

The full chain. Also available as `POST /v1/score`, which is identical except
that it accepts custom `cost_inputs` in a body rather than a query string.

```bash
curl -s "localhost:8000/v1/orders/ORD-00008874/risk?dataset_run=d1efe6b75393a8f95dac4c6a"
```

```json
{
  "order": { "order_id": "ORD-00008874", "...": "as above" },
  "probability": 0.5719063451176762,
  "raw_score": 0.5327335652564399,
  "threshold": 0.34814814814814815,
  "band": "HIGH",
  "action": "confirmation_required",
  "flagged": true,
  "reason_codes": ["ORDER_IS_COD", "HISTORY_VALUE_VS_PRIOR_MEAN", "ADDRESS_COMPLETENESS_SCORE"],
  "expected_value_inr": 45.31103488632944,
  "appeal_available": true,
  "human_review_required": false,
  "is_control_holdout": false,
  "contributions": [
    {"feature": "order_is_cod", "family": "order_shape", "value": true, "contribution": 1.1611399926081218},
    {"feature": "cust_value_vs_prior_mean", "family": "customer_history", "value": 3.850477403608548, "contribution": 0.601939028163669},
    {"feature": "addr_completeness_score", "family": "address_quality", "value": 0.4, "contribution": 0.22251179112636865}
  ],
  "model": {
    "model_name": "lightgbm_platt",
    "model_version": "a0d780424b79",
    "calibration_method": "platt",
    "calibration_fitted_on": "validation",
    "feature_version": "1.0.0",
    "feature_fingerprint": "798aef57ad3cefe9...",
    "dataset_run_id": "7b5ae86219ac7cafe45e7d51",
    "generator_version": "1.0.0",
    "trained_at": "2026-08-29T02:17:52Z",
    "training_rows": 23058,
    "n_features": 54,
    "selection_manifest_id": "4f17cd1f1279d897d589"
  },
  "features": {
    "feature_version": "1.0.0",
    "feature_fingerprint": "798aef57ad3cefe9...",
    "n_features": 54,
    "null_features": ["geo_pincode_rto_rate_smoothed"],
    "context_rows": 8874
  },
  "economics": {
    "cost_profile": "mid_margin_d2c",
    "rto_cost_inr": 220.0,
    "contribution_margin_inr": 250.0,
    "friction_support_cost_inr": 8.0,
    "abandonment_on_friction": 0.25,
    "intervention_success_rate": 0.6,
    "cost_false_positive_inr": 70.5,
    "saving_true_positive_inr": 132.0,
    "threshold_formula": "threshold = C_fp / (C_fp + S_tp)",
    "band_intervention_success_rate": 0.6,
    "band_abandonment_rate": 0.25
  },
  "engine_version": "1.0.0",
  "scored_at": "2026-08-30T06:09:12.441Z",
  "latency_ms": 1860.59,
  "outcome_is_known": true,
  "data_provenance": "Model trained on synthetic benchmark data. Labels are simulated, not real-world ground truth; see docs/model_card.md."
}
```

**Why so much provenance.** Probability, threshold, band and action travel
together deliberately: a bare score invites comparison against 0.5, which is the
error this whole system exists to correct. And the response says what it does not
know — `null_features` (a cold-start customer, not an error), `context_rows` (how
much history the aggregates saw), `outcome_is_known` (whether this is a live
decision or a re-score of a resolved order).

**`abandonment_on_friction` and `intervention_success_rate` are assumptions.**
Neither has been measured on this or any data; measuring them requires a
controlled holdout that has not been run. Every rupee figure that rests on them —
including `expected_value_inr` — inherits that. See
[docs/economics.md](economics.md).

### `POST /v1/score` — with custom economics

Changing the margin re-derives the threshold and re-bands the order, server-side:

```bash
curl -s -X POST localhost:8000/v1/score -H 'content-type: application/json' -d '{
  "order_id": "ORD-00008874",
  "dataset_run_id": "d1efe6b75393a8f95dac4c6a",
  "cost_inputs": {"rto_cost_inr": 220, "contribution_margin_inr": 400,
                  "abandonment_on_friction": 0.25, "intervention_success_rate": 0.6,
                  "friction_support_cost_inr": 8}
}'
```

```json
{"probability": 0.5719063451176762, "threshold": 0.45, "band": "ELEVATED",
 "action": "prepaid_nudge", "flagged": true, "expected_value_inr": 20.272239943755615}
```

Same order, same probability, **different action** — the margin rose from ₹250 to
₹400, so the threshold rose from 0.348 to 0.450 and the order dropped from HIGH
to ELEVATED. A higher margin means a false positive costs more, so the bar for
frictioning rises. (This direction is the opposite of the common intuition; the
repository documented it backwards until Phase 6.)

### `POST /v1/score/batch`

Up to 25 orders. Each rebuilds its own feature context, so a batch is genuinely
N times the work rather than a vectorised shortcut. Fails on the first missing
order rather than returning partial results.

---

## Decisions and overrides

### `POST /v1/decisions` — score and log

```json
{
  "order_id": "ORD-00008874",
  "probability": 0.5719063451176762,
  "threshold": 0.34814814814814815,
  "band": "HIGH",
  "action": "confirmation_required",
  "flagged": true,
  "reason_codes": ["ORDER_IS_COD", "HISTORY_VALUE_VS_PRIOR_MEAN", "ADDRESS_COMPLETENESS_SCORE"],
  "expected_value_inr": 45.31103488632944,
  "appeal_available": true,
  "human_review_required": false,
  "is_control_holdout": false,
  "model_name": "lightgbm_platt",
  "model_version": "a0d780424b79",
  "engine_version": "1.0.0",
  "decided_at": "2026-08-30T06:11:15.060139Z"
}
```

`is_control_holdout: true` marks the randomised no-friction slice: the order is
banded exactly as usual, no action is taken, and no human review is triggered —
routing it to a queue would let an operator act on it and destroy the
counterfactual the slice exists to preserve.

**The log is append-only.** There is no update endpoint and no update method on
the repository.

### `POST /v1/decisions/override`

```bash
curl -s -X POST localhost:8000/v1/decisions/override -H 'content-type: application/json' -d '{
  "order_id": "ORD-00008874",
  "override_band": "ELEVATED",
  "operator_id": "op-3f1a9c",
  "reason": "Repeat customer with a verified address; a confirmation call is unnecessary friction here."
}'
```

```json
{
  "accepted": true,
  "order_id": "ORD-00008874",
  "original_band": "HIGH",
  "new_band": "ELEVATED",
  "direction": "relaxed",
  "logged_at": "2026-08-30T06:11:15.188941Z",
  "note": "Repeat customer with a verified address; a confirmation call is unnecessary friction here.",
  "original_decision_unchanged": true
}
```

Three things this endpoint enforces:

- **The reason is mandatory**, minimum 10 characters. An override with no stated
  reason is unusable as the counterfactual evidence it is supposed to be — "an
  operator disagreed" tells the outcome loop nothing about *why*.
- **The direction is derived**, never accepted from the client. A caller claiming
  "relaxed" while raising the band would corrupt every aggregate built on it.
- **The original decision is not mutated.** The override is a new row. An audit
  trail that can be edited is not an audit trail.

### `GET /v1/decisions/{order_id}` and `GET /v1/decisions/queue`

The queue is ordered **oldest first**, not by risk score. A queue sorted by score
leaves the least risky appeals waiting longest, and those are disproportionately
the false positives — the customers who did nothing wrong, and the people the
appeal path exists for.

---

## Economics

Documented in [docs/economics.md](economics.md). Five endpoints:
`GET /v1/economics/profiles`, `POST /v1/economics/threshold`,
`POST /v1/economics/simulate`, `POST /v1/economics/what-if`,
`GET /v1/economics/sweep`.

`POST /v1/economics/threshold` needs no model and no database — the derivation is
a function of merchant economics alone, which is exactly why it can be published
before a sealed evaluation:

```bash
curl -s -X POST localhost:8000/v1/economics/threshold -H 'content-type: application/json' \
  -d '{"rto_cost_inr":220,"contribution_margin_inr":250,"abandonment_on_friction":0.25,"intervention_success_rate":0.60,"friction_support_cost_inr":0}'
```

```json
{"threshold": 0.3213367609254499, "cost_false_positive_inr": 62.5,
 "saving_true_positive_inr": 132.0, "formula": "threshold = C_fp / (C_fp + S_tp)"}
```

The specification's worked example, reproduced exactly. **Not 0.5.**

---

## Evaluation

Read from the frozen evaluation artefacts, never recomputed from live traffic —
a metric is a measurement against held-out labels, and live orders have not
resolved yet.

| Endpoint | Returns |
|---|---|
| `GET /v1/evaluation/ladder?split=validation` | Every rung, on identical footing, including the ones that beat the model |
| `GET /v1/evaluation/final?split=test` | The shipped model's sealed-set measurement, with its unseal reason |
| `GET /v1/evaluation/selection` | Every hyperparameter and calibration candidate that was tried, not only the winner |
| `GET /v1/evaluation/reliability?split=...` | Reliability bins, ECE and Brier — bins rather than an image, so a reviewer can recompute |
| `GET /v1/evaluation/fairness` | **501.** See below. |

The ladder endpoint returns rung 3 (logistic regression) at PR-AUC 0.483 and net
₹3,544/1k against rung 4 (LightGBM) at 0.443 and ₹3,298/1k. The simpler model
wins on this benchmark and the API says so.

### The fairness endpoint returns 501, deliberately

The cohort audit is defined in `config/evaluation.yaml` and has **never been
run**. Returning a plausible-looking breakdown would be the most damaging fake in
this API: a fairness report nobody computed, presented as evidence the model was
checked. The 501 carries that reason. No fairness claim about this model should
be made until the audit exists.

---

## Monitoring

Operational state, not model quality. Every number is a count from the database
or a field from the loaded model card.

| Endpoint | Returns |
|---|---|
| `GET /v1/monitoring/model` | The loaded artefact and its full provenance |
| `GET /v1/monitoring/data` | Dataset runs, order counts by split and payment method, maturity |
| `GET /v1/monitoring/decisions` | Decisions by band, review backlog, override counts and rate |

`GET /v1/monitoring/model` never raises: "no model is loaded" is exactly what an
operator queries monitoring to find out, and returning 503 would mean the
endpoint that answers the question fails whenever the answer is interesting.

```json
{"available": true, "model_name": "lightgbm_platt", "model_version": "a0d780424b79",
 "calibration_method": "platt", "feature_version": "1.0.0", "n_features": 54,
 "selection_manifest_id": "4f17cd1f1279d897d589"}
```

`observed_rto_rate` is computed over **matured orders only**. Dividing by every
order would count "not yet resolved" as "did not return" and report a rate lower
than reality.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RTO_DATABASE_URL` | assembled from `POSTGRES_*` | Connection string. Never logged; `/readiness` returns it password-redacted |
| `RTO_ARTIFACT_DIR` | `artifacts` | Where the model registry looks for a calibrated artefact |
| `RTO_SERVING_CONTEXT_LIMIT` | `20000` | Rows of merchant history the feature pipeline reads per score |
| `RTO_CORS_ORIGINS` | `http://localhost:5173` | Allowed browser origins. Not `*` |
| `ANTHROPIC_API_KEY` | unset | Optional. Agent endpoints report themselves unavailable without it |
| `RTO_AGENTS_ENABLED` | `false` | Hard off-switch. A key alone does not enable the language layer |

---

## What this API does not do

- **Ingest orders.** There is no `POST /v1/orders`. Scoring an order that is not
  in the database cannot be done honestly, so it is not offered.
- **Let a language model touch a decision.** Not the probability, the threshold,
  the band or the action. `tests/architecture/test_layering.py` asserts that no
  module in `decision` or `serving` imports an LLM SDK.
- **Serve an uncalibrated model.** The registry refuses one at load time rather
  than at the threshold comparison, so the failure surfaces at startup instead of
  on a customer's order.
- **Return a fairness audit.** Because none has been computed.
