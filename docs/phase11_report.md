# Phase 11 — integration, hardening and test report

An audit of the assembled system, not a feature phase. What follows is what was
found, what was fixed, what was measured, and what is still wrong.

**This project is not production-ready.** Section 7 says why, specifically.

---

## 1. Defects found and fixed

Seven real problems. Six were found by looking rather than by a test failing; the seventh (§1.6) surfaced only when the full suite ran against the fix for §1.3.

### 1.1 CORS: wildcard origin with credentials (security)

`RTO_CORS_ORIGINS=*` combined with `allow_credentials=True` produced a
same-origin bypass. Starlette cannot legally return `*` alongside credentials, so
it echoes the caller's own `Origin` header back. Verified empirically before
fixing:

```
Origin: https://evil.example.com
→ Access-Control-Allow-Origin: https://evil.example.com
→ Access-Control-Allow-Credentials: true
```

Any site could have made credentialed cross-origin requests and read the
responses — the opposite of what someone setting `*` believes they are
configuring. `Settings.cors_origins` now refuses a wildcard at startup with a
message naming the fix. Refusing beats quietly dropping the wildcard (an operator
would believe a config that is not in force) or quietly dropping credentials
(that changes API semantics based on an unrelated variable).

### 1.2 `split` filter accepted anything and answered with zero (honesty)

`GET /v1/orders?split=trian` returned `{"orders": [], "total": 0}` with a 200.
No injection was possible — every query is built with SQLAlchemy's expression
language and bound parameters — but "no orders are in that split" is
indistinguishable from "that split does not exist". A merchant filtering by a
typo would conclude their book was empty. `payment_method` beside it was already
constrained. `split` is now a `Literal` of the five real values and returns 422.

### 1.3 Three dead repository classes, one of them a trap (dead code)

`OrderRepository`, `DecisionRepository` and `OverrideRepository` were Phase 1
sketches whose write methods all raised `NotImplementedError("lands in Phase 4")`.
Phase 7 built the real ones (`ServingRepository`, `DecisionLogRepository`,
`OpsOverrideRepository`) and nothing was migrated, but the sketches stayed
exported from `db/__init__.py`.

`OrderRepository.get_order` was implemented, which made the class look live;
`OrderRepository.create` raised. A stub that half-works is worse than no stub.
Deleted.

### 1.4 Two unused dependencies, one of them heavy

An AST walk over `src/` found neither `shap` nor `structlog` imported anywhere.
Feature attributions come from LightGBM's own `pred_contrib=True`
(`models/rung4_lightgbm.py`) — they are genuinely SHAP values, computed by
LightGBM's built-in TreeSHAP, so the term is correct in the docs, but the `shap`
package was never involved. Nothing was ever wired to structured logging. Both
removed, along with their mypy overrides.

`scipy` and `pyarrow` are *not* imported directly either, and were kept: both are
required at runtime (scikit-learn imports scipy; pandas needs a parquet engine)
and the explicit pins turn an upstream dropping them into a resolver error rather
than an `ImportError` on the first `to_parquet`. The distinction is now recorded
in `pyproject.toml`.

### 1.5 Stale phase labels asserting work that never happened

`eval/ablation.py` said "STATUS: Phase 4"; `monitoring/outcomes.py` said "Phase
6". Neither has ever run, and we are at Phase 11. Worse, three feature modules
made claims in the *indicative*: "the Phase 5 ablation study **will show** this
family contributing close to nothing".

That is a prediction wearing the clothes of a result. All corrected to say
plainly that no ablation has been run and nothing has earned its place on that
evidence. The `NotImplementedError` messages now explain what is missing instead
of naming a phase that has passed.

### 1.6 A test that had been passing vacuously

Deleting the dead repositories (§1.3) broke
`test_decision_log_is_append_only_by_construction`, which asserted that
`DecisionRepository` exposed no `update` or `delete` method.

It had never been a real check. `DecisionRepository` could not write *at all* -
every write method raised `NotImplementedError` - so "it exposes no update
method" held trivially while saying nothing whatsoever about the class that
actually appends to the decision log. The test guarded an audit-trail property
this project genuinely depends on, and it was guarding nothing.

Now repointed at `DecisionLogRepository` and `OpsOverrideRepository` - the two
classes that really write - and strengthened to assert that `append` is present
as well as that no mutating method is. This is the clearest argument in the phase
for deleting dead code rather than leaving it: the sketch was not inert, it was
absorbing a test.

### 1.7 Agent→serving boundary was unenforced

The agent layer's only route to data is
`from rto_sentinel.serving.agent_tools import invoke`, which dispatches through a
registry of six read-only `get_` tools. Sound — but `serving` was not on the
forbidden-imports list, so nothing would have caught a later
`from rto_sentinel.serving.scoring import score` in an agent module. `serving`
also contains `assessment`, `features` and `model_registry` — a decision, a
feature matrix and a loaded booster. New layering test restricts the agent
package to the tool registry alone.

---

## 2. Test results

| Area | Tests | Result |
|---|---:|---|
| `tests/unit` | 603 | pass |
| `tests/api` | 116 | pass |
| `tests/leakage` | 29 | pass |
| `tests/db` | 27 | pass |
| `tests/architecture` | 16 | pass |
| **Backend total** | **791** | **pass** |
| Console (vitest) | 30 | pass |

**Failed: 0** in the final run. Nothing is skipped or xfailed.

One failure occurred on the way there and is worth recording rather than
smoothing over: removing the dead repositories (§1.3) broke
`test_decision_log_is_append_only_by_construction`, which turned out to have been
asserting an audit-trail property against a class that could not write. Found by
the full suite, diagnosed, and fixed by pointing the test at the live path (§1.6).

Quality gates: `ruff check`, `ruff format --check` (157 files), `mypy --strict`
(113 files), `tsc --noEmit`, `eslint`, `vite build` — all clean.

### 2.1 The two required chains

**Database → backend → model → decision engine → API response**
(`test_database_order_through_model_to_decision`). Nothing stubbed: a dataset is
generated, loaded into a real database, and a model is trained, calibrated and
frozen to disk. The API then scores an order through the real feature pipeline,
the real artefact and the real engine.

**Agent request → agent tool → database/backend → LLM → validated response**
(`test_the_agent_answers_from_real_retrieved_evidence`). The tools read the real
database and score with the real model; the assistant turns are scripted. **The
LLM transport is the only stubbed component, and no real Anthropic round trip is
covered anywhere in this repository** — see §7.1.

### 2.2 New coverage added this phase

Counts are collected test cases, so a parametrised function contributes each of
its cases. The integration file went from 7 tests to 26.

| Area | Cases | Notes |
|---|---:|---|
| Failure behaviour | 18 | §3 |
| Agent chain on real data | 7 | Closes the gap where agent tests used a stub toolset and never touched SQL |
| Frontend↔backend contract | 4 | 17 console paths parsed from `endpoints.ts`, each resolved against the live OpenAPI schema |
| Calibration in the live chain | 3 | The served probability is the calibrated one; an uncalibrated artefact is refused at load |
| Agent→serving boundary | 1 | §1.7 |
| **Total added** | **33** | |

---

## 3. Failure behaviour

Every scenario the phase brief names, asserted to produce an honest error rather
than a plausible number. Each of these has an obvious "graceful degradation"
that would make the system look better and be worse; the assertions are written
to fail if any of those appear.

| Scenario | Behaviour | Asserted |
|---|---|---|
| Database unavailable | 5xx, no probability in the body | `test_the_database_being_unavailable_produces_an_error_not_a_score` |
| Database error message | No password, SQL or traceback reaches the client | `test_a_database_failure_does_not_leak_the_connection_string` |
| Model artefact unavailable | 503 `MODEL_UNAVAILABLE` | `test_the_api_fails_when_the_model_is_missing` |
| Missing order | 404, no probability | `test_a_missing_order_is_a_404_with_no_probability` |
| Invalid input | 422 for injection strings, traversal, oversized ids | `test_hostile_order_ids_are_rejected_by_validation` |
| Unknown split | 422, not an empty page | `test_an_unknown_split_is_rejected_rather_than_answered_with_zero` |
| Invalid merchant economics | 422 across six invalid fields | `test_invalid_merchant_economics_are_refused` |
| Zero margin | 200 with a lower threshold — legal, not an error | `test_zero_margin_economics_are_handled_without_dividing_by_zero` |
| Missing historical data | Scores, and reports null-feature count and history depth | `test_an_order_with_no_customer_history_still_scores_and_says_so` |
| LLM unavailable | 501/503 naming the env var; no `summary` field | `test_the_agent_refuses_rather_than_answering_when_the_llm_is_unavailable` |
| Malformed LLM response | Rejected, not repaired | `test_invalid_structured_output_is_rejected_not_repaired` and 3 others |
| Evaluation artefact absent | 404, never a recomputation at request time | `test_evaluation_retrieval_404s_rather_than_recomputing` |

---

## 4. Security

| Check | Finding |
|---|---|
| SQL injection | **No raw SQL anywhere.** Every query uses SQLAlchemy's expression language with bound parameters |
| Secret handling | `SecretStr` for the DB password and API key; read once; never logged, echoed or serialised. Asserted by `test_the_audit_record_carries_no_secret` |
| Hardcoded credentials | None. `.env` is gitignored |
| CORS | **Was vulnerable** (§1.1), now refuses a wildcard |
| Error leakage | No stack traces, SQL or credentials in any response; one error envelope for the whole API |
| Input validation | Pydantic at every edge; `limit` capped at 200; ids pattern-constrained |
| Agent tool permissions | Six read-only `get_` tools. An unknown name returns a failed invocation naming the real tools — no exception, no execution |
| Agent filesystem/SQL access | None. No `open`, `Path`, `subprocess`, `eval` or `exec` anywhere in `agents/` |
| Personal data exposure | `/v1/orders` returns a `customer_hash` only — no address line, pincode, name or phone in any response model |
| Destructive HTTP methods | None. No DELETE, PUT or PATCH on any route, asserted over the whole schema |

### 4.1 Agents cannot alter what the system decides

Confirmed by four independent mechanisms, each tested:

1. **Import ban.** `agents/` cannot import `decision`, `models`, `features`,
   `data` or `eval` — there is no route to code that produces a probability.
2. **Narrowed door.** `agents/` may reach only `serving.agent_tools` (§1.6).
3. **No write capability.** The package cannot name a repository, `session_scope`,
   `.commit(`, `.flush(` or `session.add(`.
4. **Fields come from tools, not prose.** `test_the_agent_cannot_change_the_probability_it_reports`
   scripts a model that insists the probability is 0.01 and the band is LOW,
   against real data. The response still carries what the tools returned.

---

## 5. Performance

Measured against the live stack (FastAPI on uvicorn, PostgreSQL in Docker,
21,000 orders). Medians over 8–12 requests after warm-up. **Not optimised** — the
brief says not to, and nothing here was tuned.

| Endpoint | Median | p95 |
|---|---:|---:|
| `GET /health` | 4 ms | 22 ms |
| `GET /v1/evaluation/final` (artefact read) | 7 ms | 21 ms |
| `GET /v1/orders?limit=50` (DB query) | 40 ms | 90 ms |
| `GET /v1/monitoring/data` (DB aggregates) | 47 ms | 67 ms |
| `GET /readiness` | 123 ms | 218 ms |
| `GET /v1/orders/{id}/risk` (**full chain**) | **2,998 ms** | 3,616 ms |

### 5.1 Where the three seconds go

Attributed by measurement, not guessed. Model inference was timed in isolation
against the loaded artefact:

| Component | Cost | Share of the chain |
|---|---:|---:|
| Feature construction (8,874 history rows, full pipeline) | ~2,300 ms | **~77%** |
| TreeSHAP contributions | ~670 ms | ~22% |
| **Model inference** (`predict_proba`, one order) | **~16 ms** | **~0.5%** |
| Artefact load | 28 ms | once per process, not per request |

**The model is not the bottleneck — it is a rounding error.** Anyone optimising
this should not touch LightGBM. Inference amortises hard with batching (190 µs
per order at 100, 29 µs at 1,000), so the per-order cost is almost entirely fixed
overhead, and the real expense is rebuilding the as-of feature context from the
merchant's book on every request.

Dropping `include_contributions` saves the 670 ms and is already a query
parameter; the console requests them because the investigation screen shows the
contribution table.

This is a design consequence, not a bug. `RTO_SERVING_CONTEXT_LIMIT` (default
20,000) exists because the geography and customer-history aggregates are computed
as-of the order's timestamp from the merchant's book. Lowering it makes those
features shrink harder towards their prior — less informed, never wrong. Three
seconds is defensible for a human-triggered investigation and is not viable for
synchronous checkout scoring; see §7.4.

`/readiness` at 123 ms is slower than a probe should be: it re-parses the config
bundle and reads every artefact card. Noted, not changed.

---

## 6. Suspicious-pattern search

The whole repository, for each pattern the brief names.

| Pattern | Result |
|---|---|
| `Math.random` / random risk values in the console | **None** |
| Fake predictions, mock responses, dummy keys | **None.** Every textual match is prose *about* not faking |
| Placeholder LLM responses | **None.** `test_no_scripted_responder_ships_in_the_product` asserts it |
| Hardcoded metrics in the frontend | **None.** Every displayed metric is `formatNumber(<backend field>)` |
| Hardcoded order records | **None** |
| `TODO` / `FIXME` / `XXX` / `HACK` | **None in the entire repository** |
| Numeric fallbacks in render paths | **None.** Every `??` yields an em-dash or empty string, never a number |
| `NotImplementedError` | 8 remain, all in honestly-unimplemented interfaces (§1.5). None reachable from a route that claims to work |

**Controlled simulation randomness was left alone**, as instructed. `np.random`
in `data/generator.py`, `data/address.py` and `data/splits.py` is the seeded
benchmark sampler — the thing the entire project measures against, reproducible
from `(config, seed)` and asserted so by `test_generator_reproducibility.py`. The
distinction that matters: randomness that produces *documented simulated data* is
the product; randomness that produces *application behaviour* would be a fake.
There is none of the second kind.

---

## 7. Known limitations and unresolved issues

### 7.1 No real LLM round trip is covered anywhere — UNRESOLVED

No `ANTHROPIC_API_KEY` is configured. The agent layer refuses honestly (tested),
and the tool loop, grounding validators and audit trail are tested against real
application data with the transport scripted. **The one leg never executed is the
actual HTTP call to Anthropic.** Provider error handling is tested against a
`FailingProvider`, not a real failure. This is a genuine hole and no test in this
repository closes it.

### 7.2 The API has no authentication and no rate limiting — UNRESOLVED

Anyone who can reach the port can list every order and score any of them. There
is no auth middleware, no API key, no rate limit, and `/docs`, `/redoc` and
`/openapi.json` are open. This was never in scope for any phase, and it is the
single largest reason the system must not be deployed as-is.

### 7.3 No TLS, and no request-size limit beyond validation

The API is served over plain HTTP by uvicorn with no terminating proxy in this
repository. Combined with §7.2 that means order data crosses the network in
clear. Body size is bounded only by Pydantic's per-field constraints; there is no
global request-size cap.

### 7.4 Frontend response *shapes* are unverified against the backend

The new contract test proves every path the console requests exists and is served
over the right method. It does **not** check that `types/api.ts` — which is
hand-written — agrees with the response models field for field. A renamed
response field would still ship green and render an em-dash.

### 7.5 Three-second scoring is not viable for synchronous checkout

§5.1. Acceptable for merchant-initiated investigation; a checkout-time
integration would need precomputed aggregates or a feature store. Nothing here
does that — and note from §5.1 that the fix is not a faster model.

### 7.6 Integration tests run on SQLite, not PostgreSQL

The schema is engine-independent and PostgreSQL is exercised by `seed-db` and by
manual verification, but the automated integration suite uses SQLite so it can
run with no server. Dialect-specific behaviour is not covered.

### 7.7 Carried forward, unchanged

- **No fairness claim.** The cohort audit runs on synthetic data with operational
  cohorts. Not evidence of production fairness (`docs/responsible_ai.md`).
- **The sealed-set economic result is not statistically established.** Net ₹716
  per 1,000 with a 95% interval of ₹−1,009 to ₹2,603, on 1,698 orders. The
  interval crosses zero.
- **Intervention effectiveness and abandonment rates are assumptions**, never
  measured. Every rupee figure inherits that.
- **The ablation study has never been run.** No feature family's economic
  contribution is established.
- **The outcome loop is not implemented.** Measured precision will decay once the
  system acts, and only the 2% control holdout keeps it measurable.
- **Labels are simulated.** Absolute metric values describe this simulator.

---

## 8. What this phase did not do

No new features, as instructed. The ablation study, the outcome loop and live
drift monitoring remain unimplemented interfaces — building them here would have
been the opposite of the brief. `docs/responsible_ai.md` and this report both say
so where the numbers would otherwise go.
