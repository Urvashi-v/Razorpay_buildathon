# RTO Sentinel — Architecture

> A cost-calibrated return-risk scorer for Indian COD commerce — that knows what
> a false positive costs.

This document defines the components, their responsibilities, and the boundaries
between them. Where a boundary is enforced by a test rather than by convention,
the test is named. Where something is not built yet, it says so.

**Status: the end-to-end path is built and running.** Data layer, feature
pipeline, calibrated model, deterministic decision engine, FastAPI backend, agent
layer and merchant console are implemented and tested. What remains unbuilt is
named explicitly: retraining on realised outcomes, live drift monitoring against
production traffic, and address repair — which is deferred with reasons rather than faked.
The cohort fairness audit and the robustness study now run; their results are in
`docs/responsible_ai.md`, and they are **controlled benchmark experiments on
synthetic data, not evidence of production fairness or robustness**. See
[Implementation status](#8-implementation-status).

The diagrams live in [docs/architecture.md](docs/architecture.md); this document
is the module inventory and the boundary rules.

**It is not production-ready, and the reasons are specific rather than
rhetorical.** No test in this repository executes a real LLM call; scoring one
order takes ~3 seconds because the feature context is rebuilt from thousands of
history rows per request; the rate limiter holds its buckets in process memory,
so a multi-worker deployment permits more than the configured rate; and the
integration suite runs on SQLite. All of it is enumerated in
[docs/phase11_report.md](docs/phase11_report.md).

---

## 1. The one rule everything else serves

```
   ┌──────────┐     probability      ┌──────────────────┐    action    ┌─────────┐
   │  MODEL   │ ───────────────────▶ │ DECISION ENGINE  │ ───────────▶ │   API   │
   │ (rung 4) │   calibrated P(RTO)  │  deterministic   │   + reasons  │         │
   └──────────┘                      └──────────────────┘              └────┬────┘
                                              │                             │
                                     cost inputs from                       │
                                     the merchant                           ▼
                                                                    ┌───────────────┐
                                                                    │  AGENT LAYER  │
                                                                    │  (describes)  │
                                                                    └───────────────┘
```

**The ML model outputs a probability. The economic decision engine converts that
probability into a decision. The LLM is downstream assistance only.**

The LLM may not:

| Prohibition | Where it is enforced |
|---|---|
| Generate the risk probability | `agents/` cannot import `models/` — `test_agents_cannot_reach_the_decision_engine_or_models` |
| Choose the economic threshold | Threshold is derived in `decision/threshold.py` from `CostInputs` alone |
| Override the deterministic decision engine | `decision/` cannot import `agents/` — `test_decision_layer_never_imports_an_llm` |
| Fabricate evidence | `agents/grounding.py` rejects any output naming something outside its allow-list |
| Modify model predictions | Agent outputs are typed as `contracts.explanation.*`, none of which carries a probability, threshold, band or action |
| Silently approve or block orders | `Decision` rejects `appeal_available=False`; `PolicyConfig` rejects `hard_block_allowed: true` |

If every LLM call fails, the system still scores orders, still applies the
correct threshold, and still takes the right action. It just explains itself less
gracefully. That is the design, not a fallback.

---

## 2. Directory map

```
.
├── config/                  # every knob, in version-controlled YAML
│   ├── generator.yaml       #   synthetic sampler parameters
│   ├── splits.yaml          #   the split protocol, fixed before modelling
│   ├── features.yaml        #   feature families + the refused list
│   ├── cost_model.yaml      #   merchant cost profiles + bounds
│   ├── policy.yaml          #   the four-rung friction ladder
│   ├── evaluation.yaml      #   metrics, fairness triggers, prohibitions
│   └── models/ladder.yaml   #   baseline ladder rungs 0–5
│
├── src/rto_sentinel/
│   ├── settings.py          # THE ONLY module that reads the environment
│   ├── cli.py               # pipeline entry points
│   ├── configuration/       # typed schemas + validating loader + fingerprint
│   ├── contracts/           # shared types; bottom of the dependency graph
│   ├── data/                # generator, address, validation, as-of, splits,
│   │                        #   artefacts, pipeline
│   ├── features/            # one module per family + the assembling pipeline
│   ├── models/              # ladder rungs 0–5, calibration, artefacts
│   ├── decision/            # cost model, threshold, policy, engine ◀ AUTHORITY
│   ├── eval/                # metrics, economics, bootstrap, fairness, report
│   ├── monitoring/          # drift + outcome loop
│   ├── agents/              # the four language jobs ◀ DOWNSTREAM ONLY
│   ├── db/                  # declarative models, session, repositories
│   └── api/                 # FastAPI app, deps, errors, routers
│
├── console/                 # React + TypeScript + Vite merchant console
├── migrations/              # Alembic (initial revision applied to PostgreSQL)
├── artifacts/datasets/      # generated benchmark datasets (git-ignored)
├── tests/
│   ├── unit/                # contracts, config, settings, refused features
│   ├── architecture/        # the layering rules, mechanically enforced
│   ├── api/                 # health, contract surface
│   ├── db/                  # schema shape, privacy, and the load round-trip
│   └── leakage/             # the four leakage tests from the spec (running)
├── docker/                  # API and console images
├── docs/                    # simulator write-up, sources, ADRs
└── scripts/                 # setup, check, seed_db
```

The specification's appendix A3 sketches `src/data/`, `src/models/`, `api/`,
`console/`. That structure is preserved conceptually; the Python parts live under
`src/rto_sentinel/` so the project is an installable, importable package rather
than a collection of scripts that only run from the repository root.

---

## 3. Components and responsibilities

### 3.1 Configuration (`configuration/`)

Loads and validates every YAML file into frozen Pydantic models, and computes a
SHA-256 **fingerprint** over the bundle.

The fingerprint is the mechanism that makes "the split protocol was fixed before
modelling" checkable rather than merely asserted: it is stamped onto every
dataset, model artefact and evaluation report, so a configuration edited after
results were produced is detectable.

Validation is not passive. `PolicyConfig` refuses to load a ladder whose bands
are reordered or whose `hard_block_allowed` is true. `SplitsConfig` refuses a
temporal ordering where validation precedes training. `LadderConfig` refuses
`smote: true`. Weakening a commitment requires deleting a validator, which shows
up in a diff.

### 3.2 Contracts (`contracts/`)

Shared types. The bottom of the dependency graph — it imports nothing above
itself (`test_contracts_depend_on_nothing_but_contracts`).

Three separate type hierarchies exist on purpose:

| Hierarchy | Purpose | Why separate |
|---|---|---|
| `contracts/` | API wire format | Changes for frontend reasons |
| `configuration/schemas.py` | YAML shape | Changes for tuning reasons |
| `db/models.py` | Storage schema | Changes require a migration |

Collapsing them would mean a field added for auditing appears in a public
response, or a wire-format rename forces a migration.

Safety invariants live in the types:

- `Decision` cannot be constructed with `appeal_available=False`.
- `Decision` cannot apply friction without a reason code.
- A `SEVERE` decision must route to human review.
- `CostInputs` rejects degenerate inputs from which no threshold exists — rather
  than silently returning 0.5.
- `PointEstimate` cannot exist without a confidence interval.
- `OrderPayload` has **no field** for a customer name, gender or age.

### 3.3 Data pipeline (`data/`)

| Module | Responsibility |
|---|---|
| `generator.py` | The benchmark simulator, driven entirely by `generator.yaml` |
| `address.py` | Address rendering (latent → text) and observable signals (text → measurements) |
| `schema.py` | Canonical columns, `ORDER_TIME_COLUMNS`, `FORBIDDEN_IN_FEATURES`, `LATENT_COLUMNS` |
| `validation.py` | Ten groups of structural and semantic checks |
| `asof.py` | As-of joins, plus a brute-force reference and a leakage assertion |
| `splits.py` | Pool-then-window split assignment, and the test-set seal |
| `artifacts.py` | Parquet dataset artefacts with a JSON provenance sidecar |
| `pipeline.py` | generate → validate → split → validate → write, in that fixed order |

**The labels are simulated, not observed.** Everything the generator emits is the
output of the documented process in [docs/simulator.md](docs/simulator.md). A
metric measured on this data is a statement about the simulator; it becomes a
statement about reality only after validation on a real merchant's history.

Three drivers are deliberately **latent** — customer reliability, the per-pincode
random effect, and (partially) address quality. Together with a per-order
Gaussian shock and the Bernoulli draw, they create a genuine Bayes-optimal
ceiling. A model that scores near-perfectly here has found a bug.

The base-rate intercepts are solved by a fixed-point iteration over the whole
simulation pass rather than in closed form, because the simulation has a real
feedback loop: a customer's prior RTO rate is an input to their next order's risk.

The as-of rule keys on **resolution** time, not order time. An order placed on
day 40 that comes back on day 47 was not known to be an RTO on day 42. That
distinction is the single most important line in the split protocol, and it lives
in one module so it is reviewed once rather than re-derived per feature family.

`TestSetSeal` writes a receipt on first use and refuses a second scoring run.
Deleting the receipt to re-score is possible — it is a visible act in the working
tree, which is the point.

### 3.4 Features (`features/`)

One class per family, each independently ablatable via `enabled` in
`features.yaml`. That is what makes the leave-one-family-out study a
configuration change rather than a code change, with no code path differing
between the ablated and full runs.

`pipeline.py` is the single choke point between raw orders and any model, and
enforces three things once rather than trusting five implementations: no target
leakage, no refused features, consistent column order between training and
serving.

**Geography is isolated behind its own switch** because it carries the highest
fairness risk and is the family that gets pulled back if the audit trips.

### 3.5 Models (`models/`)

Rungs 0–4 behind one interface, so the evaluation harness scores a do-nothing
baseline and a LightGBM identically and has no idea which is which.

`predict_proba` returns probabilities. **Nothing in this layer returns a band, an
action, a rupee figure or a threshold** — those belong to `decision/`, and
keeping them out of the interface is what stops a model quietly deciding policy.

**Why `fit`/`predict_proba` take a second `context` frame.** Rungs 1 and 2 are
operational rules, and they need columns the learned rungs are forbidden:
rung 2's whole question is whether a pincode blocklist works, and `pincode` is
withheld from the feature matrix precisely so no learned rung can key on raw
geography. Passing it through a separate `context` argument means the heuristics
get what they need without widening what the model matrix contains, and
`test_learned_rungs_ignore_context` asserts rungs 3 and 4 produce identical
scores with the context present and absent.

**Every rung records its training PR-AUC**, not only its validation PR-AUC.
Rung 4 at its configured size scores far higher on train than on validation, and
a record showing only the validation number would present a badly overfitting
model as merely a mediocre one. Those call for different responses, so the gap
travels with the result.

`artifacts.py` writes a model, its card and a SHA-256 checksum. `load_artifact`
verifies the checksum **before** unpickling, and `read_card` reads the JSON card
without unpickling anything, so provenance can be inspected without executing a
file. The API will load these artefacts in a later phase.

`calibration.py` is a first-class component, not a postprocessing step: the
entire decision layer depends on the score being an honest probability. The
calibrator is fitted on **validation** — never train (calibrating to one's own
overfitting), never test (contaminating the seal). The Phase 4 ladder rungs
remain uncalibrated and carry `calibration_method: null`; the Phase 5 model is a
`CalibratedModel` composing one of them with a fitted calibrator.

**The method is chosen by cross-validation inside the validation split**, not by
fitting each candidate on validation and reading its error off the same rows.
Isotonic regression in particular can drive that in-sample number close to zero
while generalising worse than doing nothing, and `test_the_comparison_is_out_of_fold`
measures exactly that optimism. `none` is always a candidate, so "calibrating did
not help" is a result the pipeline can reach.

Selection uses **two gates**: a candidate must improve expected calibration error
by a configured margin *and* must not worsen the Brier score. ECE is computed
over ten equal-width bins and is noisy — on two thousand rows a perfectly
calibrated model still scores about 0.015 from sampling alone — so a flexible
calibrator can "improve" it by fitting binning noise. Brier is a proper scoring
rule and cannot be improved that way, so it acts as a veto.

**`CalibratedModel` is a wrapper, not a flag.** The base model saw the training
split and the calibrator saw the validation split; folding them into one object
would hide the first thing anyone auditing a probability needs to know. It
refuses a single-call `fit` for the same reason.

### 3.6 Decision engine (`decision/`) — the authority

Six modules: `cost_model` (rupee arithmetic), `threshold` (the derivation),
`policy` (the friction ladder), `reason_codes` (stable identifiers from SHAP),
`engine` (one order), `portfolio` (a whole book), `threshold_analysis` (the
sweep) and `simulation` (merchant what-ifs). Nothing here imports an LLM SDK,
the API, or the agent layer.


Inputs: a **calibrated** `RiskScore` and a `CostInputs`. Nothing else.

Given those, the output is a pure function. Same inputs, same `Decision`, every
machine, forever. A non-deterministic risk engine cannot be audited.

```
C_fp      = abandonment_on_friction × contribution_margin  + friction_support_cost
S_tp      = intervention_success_rate × rto_cost
threshold = C_fp / (C_fp + S_tp)
```

Worked example from the spec: `0.25 × 250 = 62.5`, `0.60 × 220 = 132`,
`62.5 / 194.5 = 0.3214`. **Not 0.5**, and it moves with the merchant's margin —
asserted in `test_merchant_economics_move_the_threshold`.

Band cut points are *multipliers on the derived threshold*, so the whole ladder
slides when the merchant's economics change. No absolute probability is hardcoded
anywhere in the policy. A band whose span collapses at a given threshold — a
thin-margin merchant leaves no room above 1.0 for three more rungs — is dropped
and **reported**, so a merchant is told "your economics leave no SEVERE tier"
rather than shown an empty one.

Boundaries are half-open `[lower, upper)`, so a probability exactly on a cut
point lands in the higher band and the flag rule is `p >= threshold` — matching
`confusion_at_threshold` in the evaluation harness. Two components disagreeing
about that one boundary would make the served flag rate differ from the measured
one, silently, for exactly the orders sitting on the line.

**The direction of the margin response is counter-intuitive, and this repository
had it backwards until Phase 6.** A *higher* contribution margin *raises* the
threshold, so a high-margin brand flags **less**. The margin is what a false
positive costs; a merchant with more to lose demands more certainty. The
merchant simulator made the contradiction visible and it is now asserted in
`test_a_higher_margin_raises_the_threshold`.

Failure posture: if the model artefact is missing, the engine raises. It does not
fall back to a default probability and does not pass an uncalibrated score
through to a threshold comparison.

### 3.7 Evaluation (`eval/`)

Built before any model exists, so the scoreboard cannot be bent around a result
after the fact.

Reporting rules are structural rather than aspirational:

- There is no `accuracy()` function in `metrics.py`, and that absence is
  deliberate.
- `EconomicResult` has a **required** `total_false_positive_cost_inr` field, so
  it cannot be netted away.
- Flag rate travels with precision, always.
- `PointEstimate` cannot exist without an interval.
- `report.py` raises rather than renders a report violating any rule in
  `evaluation.yaml` — including on the rendered text, where "precision quoted
  without a flag rate" is the rule that can actually be broken.
- Undefined is a state distinct from zero. A constant predictor has no ROC-AUC;
  it is carried as NaN, rendered as a dash, and serialised as JSON `null`,
  never as 0.5 ("measured, no better than chance") or 0.0 ("measured, terrible").
- Nothing writes a metric from a literal. Every number in
  [docs/ladder_results.md](docs/ladder_results.md) is computed from predictions
  and regenerated by `rto-sentinel train`.

The fairness audit asks whether **precision holds up in the tiers that get
flagged most** — not whether flag rates are equal, which they will not be and
should not be forced to be.

### 3.8 Agent layer (`agents/`)

Four jobs, each with a guardrail:

| Job | Guardrail |
|---|---|
| Reason-code phrasing | Given only feature names and contributions; output naming anything else is rejected |
| Confirmation drafting | Human-reviewed template; neutral-framing validator; never implies suspicion |
| Weekly digest | Figures come from SQL; the model writes prose and computes nothing |
| Address repair | Always a suggestion the customer accepts or rejects; never silently rewrites |

`provider.py` returns an `UnavailableProvider` when no key is configured. There
is no bundled key and **no canned fallback that imitates model output** — a
missing dependency must be visible, not disguised.

Rejection is a normal outcome, recorded as `grounded=False` with a reason, so the
rejection rate is measurable rather than invisible. A repaired hallucination is
still a hallucination that got close enough to pass.

### 3.9 Database (`db/`)

Five tables: `orders`, `order_outcomes`, `decisions`, `ops_overrides`,
`model_runs`.

**The label lives in its own table.** If `outcome` were a column on `orders`, the
convenient query would be the leaky one. Here the convenient query must reference
`resolved_at`, so the time constraint becomes the obvious thing to write.
`is_rto` is nullable because an immature order has no known outcome, and NULL is
the only honest representation of that.

**The decision log is append-only by construction.** There is no `updated_at`
column and `DecisionRepository` has no update method. An audit trail that can be
edited is not an audit trail.

**`ReadOnlyRepository` is what the agent layer receives.** It has no write
methods to call, so "the LLM must not modify a decision" is a fact about the type
it holds.

**Benchmark identifiers are unique per dataset run, not globally.** `order_id`,
`customer_hash` and `address_fingerprint` name a row inside one run: two
generator runs both number orders from `ORD-00000001` and both render some of the
same addresses. They were originally declared globally unique, which made a
database that could hold exactly one benchmark dataset — the second `seed-db`
failed on a unique-constraint violation, despite `dataset_runs` and `delete_run`
existing so that runs can coexist and be compared. Migration `4f1c2a7d8e30`
replaces each global index with a composite unique constraint on
`(dataset_run_id, identifier)` plus a **partial** unique index on the identifier
alone `WHERE dataset_run_id IS NULL`, so serving-path rows — real orders, part of
no benchmark — keep global uniqueness. The composite constraint alone would not
give them that: SQL treats NULLs as distinct.

### 3.9b Serving (`serving/`) — the composition layer

Every other layer is deliberately unable to reach the ones beside it: `features`
and `models` cannot import `db` (a model must be retrainable offline with no
server), `decision` cannot import either (it stays pure so the same inputs always
produce the same action), and `api.routers` cannot import an ML library. Those
constraints leave a gap — something has to assemble the pieces for a live
request — and this package is it.

```
database row → OrderFeatureService → ModelRegistry → calibrator
             → RiskScore → DecisionEngine → OrderAssessment
```

**`ModelRegistry` is a controlled loader, not a getter.** It verifies the
artefact's SHA-256 before unpickling, refuses an artefact whose card says
`calibration_method: null`, checks the card's feature fingerprint against the
pipeline the server is actually running, and caches the result behind a lock so
the served model is a stable, reportable version rather than whatever is on disk
at that instant. A mismatch is a 409, not a wrong number.

**`OrderFeatureService` reimplements nothing.** It reconstructs the *input frame*
from the database in the shape the generator produced and hands it to the same
`FeaturePipeline` training used. `test_serving_features_match_the_offline_pipeline`
compares the served row against the offline one field by field, because a serving
path that computes features slightly differently does not fail — the numbers just
move, and the model quietly stops being the model that was evaluated.

**The context frame is larger than one row, necessarily.** The feature families
recompute customer history and geography aggregates from data, so a single order
scored alone would look like a first-time customer in an unknown pincode. The
service loads the merchant's book up to that order's `ordered_at` and lets the
as-of machinery mask anything unresolved — safe precisely because that masking is
what the leakage suite tests. It costs about a second per order on the benchmark
and `docs/api.md` says so rather than hiding it.

### 3.10 API (`api/`)

Handlers marshal and delegate. `test_no_ml_logic_in_route_handlers` asserts no
router imports an ML library.

One error envelope for every failure. The `code` matters more than the status:
`MODEL_UNAVAILABLE` and `AGENT_UNAVAILABLE` are both 503 but mean opposite things
operationally, and a frontend that can only see the status would either
over-react to a missing sentence or under-react to a missing model.

Unimplemented endpoints return **501 `NOT_IMPLEMENTED`**, never plausible
placeholder data. A fabricated score would flow into a chart, a screenshot, and
then a claim.

### 3.10b Monitoring (`monitoring/`)

Drift measurement, kept deliberately thin and label-aware.

**Drift is not failure, and the types make it impossible to claim otherwise.** A
`DriftSignal` carries a distance and a severity band and has no field in which to
record a verdict. A `PerformanceDelta` records a measured change in quality and
can only be constructed where mature labels exist. When the current window has
not matured, the report says the question is unanswered rather than showing an
all-clear — which is the single most misleading thing a monitoring page could do.

The severities are `stable` / `watch` / `investigate`, not `pass` / `warn` /
`fail`. Indian e-commerce moves hard during festive season; COD share, order
values and category mix all shift for entirely ordinary reasons, and a monitor
that pages someone every Diwali gets muted by March.

### 3.11 Console (`console/`)

React + TypeScript + Vite. **Reads every number from the backend and computes
none.** A chart that can invent its own numbers is a picture, not a report.

One HTTP client; no component fetches on its own. That leaves nowhere for a
component to hardcode a fallback when a call fails.

Five screens: dashboard, order queue, order investigation, economic simulator,
evaluation. The type system carries the honesty rule — `AsyncState<T>` is a
discriminated union, so `data` does not exist until a request has succeeded and
there is no property to write `data ?? fallback` against. Formatters return an
em-dash for `null`, never a zero, and take no default arguments.

The simulator is the sharp case. Dragging a slider changes local state only;
recomputing sends the inputs to `POST /v1/economics/simulate` and the server
re-derives the threshold, re-resolves the bands, re-assigns every order in the
scored book and re-prices it. Doing that arithmetic in JavaScript would be
faster and would create a second implementation of the decision rule — the one
on screen, and the one nobody tested.

Full detail: [docs/console.md](docs/console.md).

---

## 4. Boundaries, enforced

`tests/architecture/test_layering.py` parses the import graph — including
`TYPE_CHECKING` blocks — and asserts:

| Rule | Test |
|---|---|
| `decision/` imports no LLM SDK and no `agents/` | `test_decision_layer_never_imports_an_llm` |
| `decision/` imports no database, web framework or HTTP client | `test_decision_layer_is_free_of_io` |
| `agents/` imports no `decision/`, `models/`, `features/`, `data/`, `eval/` or ML library | `test_agents_cannot_reach_the_decision_engine_or_models` |
| `agents/` reaches `serving/` only through `agent_tools`, the read-only tool registry | `test_agents_reach_serving_only_through_the_tool_registry` |
| Route handlers import no ML library | `test_no_ml_logic_in_route_handlers` |
| `data/`, `features/`, `models/`, `eval/`, `monitoring/` import no `api/` or `db/` | `test_ml_layers_do_not_import_the_web_or_database_layer` |
| `contracts/` imports nothing above itself | `test_contracts_depend_on_nothing_but_contracts` |
| Only `settings.py` reads `os.environ` | `test_only_settings_reads_the_environment` |

A type-only import of an LLM SDK into the decision engine would not execute — but
it would mean someone was reaching for it, and the point of a boundary is to
notice that.

---

## 5. Data flow

**Training (offline, no database, no server):**

```
generator.yaml ─▶ generate ─▶ validate ─▶ as-of features ─▶ split (temporal+grouped)
                                                                  │
                            ┌─────────────────────────────────────┤
                            ▼                     ▼               ▼
                          train              validation         test
                            │                     │           (SEALED,
                            ▼                     ▼            scored once)
                       fit rungs 0–5   ─▶  fit calibrator  ─▶  final report
                                             fix threshold
```

**Serving (online):**

```
POST /v1/score
   │
   ├─▶ contracts.OrderPayload      (validate; reject unhashed identity)
   ├─▶ features.FeaturePipeline    (as-of aggregates; refused columns rejected)
   ├─▶ models.RiskModel            (probability) ─▶ calibrator ─▶ RiskScore
   ├─▶ decision.DecisionEngine     (threshold ─▶ band ─▶ action ─▶ reason codes)
   ├─▶ db.DecisionRepository       (append to the log)
   └─▶ ScoreResponse
                     ⋯ optionally, and never blocking ⋯
        agents.write_explanation ─▶ Explanation (grounded, or not)
```

The dotted step can fail entirely without affecting anything above it.

---

## 6. Security boundaries

| Boundary | Rule |
|---|---|
| **Secrets** | Only `settings.py` reads the environment. Keys are `SecretStr`, absent from `repr()`. `.env` is git-ignored; `.env.example` documents variables with empty values. `test_no_credential_literals_in_repository` sweeps for key-shaped literals. |
| **PII** | `OrderPayload` and the `orders` table have no name, phone, email, gender or age column — not nullable ones, none. A column that does not exist cannot be populated later. |
| **Identity** | `customer_hash` is validated as a hex digest; a phone number posted there is rejected at the HTTP edge. |
| **Response hygiene** | `/readiness` reports the database URL with the password redacted and the API key only as a boolean. Validation errors are stripped of `input` before being returned, so a rejected identifier is not echoed back. |
| **CORS** | Restricted to configured origins. Never `*`. |
| **Container** | Runs as a non-root user; multi-stage build carries no toolchain into runtime. |
| **Database ports** | Bound to `127.0.0.1` in compose, never `0.0.0.0`. |
| **Agent capability** | Read-only tools. No tool writes a score, a decision, or a message to a customer. |

### External services

**Anthropic Claude API** (`https://api.anthropic.com`) is the only external
service, used by the four language jobs.

- Required variable: `ANTHROPIC_API_KEY`
- Optional: `RTO_LLM_MODEL`, `RTO_LLM_MAX_TOKENS`, `RTO_LLM_TIMEOUT_SECONDS`
- Hard switch: `RTO_AGENTS_ENABLED` (off by default; a key alone is not enough)

No key is bundled, no fake response is substituted, and the SDK is an optional
dependency (`pip install -e ".[agents]"`) so the core system installs without it.

---

## 7. Defense-only posture

RTO Sentinel consumes order metadata and emits a risk score, a recommended
action, and a human-readable explanation. It contains no component that generates
fraudulent behaviour, no adversarial or evasion-testing module, no synthesis of
realistic fake identities or payment instruments, and no capability that
transfers to attacking any system.

**The synthetic generator is the one component deserving scrutiny, so it is named
explicitly.** It is a labelled tabular sampler: order metadata plus a
probabilistic RTO label drawn from published aggregate base rates. It does not
produce working payment credentials, valid identity documents, deliverable
addresses, or anything usable outside this repository's evaluation harness.
Customer identifiers are opaque hashes; pincodes are synthetic six-digit
identifiers, not a map of real Indian postcodes. The code is short enough for a
reviewer to verify that in a minute.

---

## 8. Implementation status

| Component | Status |
|---|---|
| Configuration schemas, loader, fingerprint | **Implemented and tested** |
| Shared contracts + safety invariants | **Implemented and tested** |
| Settings, secret handling, redaction | **Implemented and tested** |
| Benchmark generator (v1.0.0) | **Implemented and tested** |
| Address rendering and observable signals | **Implemented and tested** |
| Data validation (10 check groups) | **Implemented and tested** |
| As-of joins + brute-force reference + leakage guard | **Implemented and tested** |
| Split assignment (pool-then-window) and test-set seal | **Implemented and tested** |
| Dataset artefacts (parquet + provenance JSON) | **Implemented and tested** |
| Database schema, 10 tables, Alembic migration | **Implemented; applied to PostgreSQL** |
| Dataset bulk loader and read-back queries | **Implemented and tested** |
| CLI: `config check`, `generate`, `validate`, `db upgrade`, `db stats`, `seed-db`, `features`, `split`, `train`, `evaluate` | **Implemented and exercised** |
| The four spec leakage tests | **Running and passing** (were skipped in Phase 1) |
| API app, error envelope, `/health`, `/readiness` | **Implemented and tested** |
| API contract for all 17 endpoints | **Published in OpenAPI**; 2 return 501 (fairness audit, address repair) |

| Architecture layering tests | **Implemented and passing** |
| Feature families, pipeline, dataset contract | **Implemented and tested** |
| Feature catalogue with per-feature availability | **Implemented and tested** |
| The seven Phase 3 leakage tests | **Running and passing** |
| Ladder rungs 0-4 behind one interface | **Implemented and tested** |
| Model artefacts (joblib + card + SHA-256) | **Implemented and tested** |
| Experiment records, ladder runner, results JSON | **Implemented and tested** |
| Cost model and threshold derivation | **Implemented and tested** |
| Metrics, economics, bootstrap, report, plots | **Implemented and tested** |
| Calibration: isotonic, Platt, identity, cross-validated selection | **Implemented and tested** |
| `CalibratedModel` wrapper and its artefact round-trip | **Implemented and tested** |
| Hyperparameter search with the 1-SE tie rule | **Implemented and tested** |
| Frozen selection manifest and sealed-set gating | **Implemented and tested** |
| Final-model evaluation, metrics JSON/CSV, plots | **Implemented and exercised** |
| Generated model card | **Implemented and tested** |
| Policy bands, friction ladder, decision engine | **Implemented and tested** |
| Reason codes from SHAP attributions | **Implemented and tested** |
| Portfolio economics (expected and realized) | **Implemented and tested** |
| Threshold sweep with stated selection methodology | **Implemented and tested** |
| Merchant simulation service and its API | **Implemented and tested** |
| Provenance taxonomy on every reported quantity | **Implemented and tested** |
| Generated economic evaluation report | **Implemented and exercised** |
| Serving layer: model registry, feature service, assessment | **Implemented and tested** |
| Serving repositories (orders, decision log, overrides) | **Implemented and tested** |
| Orders, risk, decision, override, monitoring endpoints | **Implemented and tested** |
| Evaluation endpoints reading frozen artefacts | **Implemented and tested** |
| End-to-end integration test (database → model → decision) | **Implemented and passing** |
| Console: dashboard, order queue, investigation, simulator, evaluation | **Implemented and tested** |
| Console: every displayed value sourced from a backend response | **Implemented and tested** (`OrderInvestigation.test.tsx` asserts the request URL and the rendered response) |
| Console: fairness, ablation and drift screen | **Implemented and tested** |
| Integration and failure-behaviour suite (Phase 11) | **Implemented and passing** — see `docs/phase11_report.md` |
| Frontend/backend path contract test | **Implemented and passing** — paths, methods, and every declared response field checked against the OpenAPI schema; field *types* are not |
| API-key authentication on every `/v1` route | **Implemented and tested**; refuses to start open in a deployed environment |
| Per-key sliding-window rate limiting | **Implemented and tested**; `memory` backend is per process, `database` shares one counter across workers |
| TLS termination and server-side key injection | **Implemented** as a compose overlay (`docker-compose.production.yml`) |
| Per-key `read`/`write` scopes | **Implemented and tested**; `read` is the default, `write` guards the two routes that change stored state |
| Request audit log | **Implemented and tested** — one line per `/v1` request, never the key |
| Key rotation, secret manager, per-route scopes | **Not implemented** — `docs/deployment.md` §6 |
| Real LLM round trip | **Never executed.** The transport is the one leg no test covers |
| One-command bootstrap (`scripts/bootstrap.sh`) | **Implemented and exercised end to end.** Run from scratch on 2026-09-05 it reproduced the same dataset run id, the same model hash and all 27 measured artefacts byte for byte, differing only in timestamps and wall-clock durations |
| Executable demonstration (`scripts/demo.sh`) | **Implemented and exercised** against the live system |
| Generated evaluation report | **Implemented** — `docs/evaluation_report.md` |
| Dataset run reproducibility from (seed, parameters) | **Verified**: re-seeding reproduced run `7b5ae86219ac7cafe45e7d51` exactly |
| Leave-one-family-out ablation, in net rupees | **Implemented, tested and run** — `docs/evaluation_report.md` §8b |
| Frontend response-shape contract test | **Implemented and passing** — 21 console interfaces checked against the OpenAPI schemas |
| Fairness, shift and drift endpoints | **Implemented and tested**; each returns 501 with its reason until its experiment is run |
| Agent tool contract, schemas and permission boundaries | **Implemented and tested** |
| Read-only application toolset (6 tools) | **Implemented and tested** |
| Grounding validators (4) | **Implemented and tested** |
| Agent audit trail | **Implemented and tested** |
| Anthropic provider, incl. tool use | **Implemented; unverified against the live API** |
| Risk investigation agent (tool loop) | **Implemented; unverified against the live API** |
| Confirmation writer, digest writer | **Implemented; unverified against the live API** |
| Address repair | **Deferred, with reasons** (`agents/address_repair.py`) |
| Cohort fairness audit, Wilson intervals, support gating | **Implemented, tested and run** — results in `docs/responsible_ai.md` |
| Sensitive-attribute refusal (hard error, not a skipped cohort) | **Implemented and tested** |
| Controlled distribution-shift study (9 perturbations, frozen model) | **Implemented, tested and run** |
| Drift monitoring: feature, prediction, rate and calibration | **Implemented, tested and run** |
| Responsible-AI report, generated from artefacts | **Implemented and generated** |
| Outcome loop: treated-vs-control intervention measurement | **Implemented and tested**; reports insufficient data until 200 orders accumulate on each arm |
| Override analytics by band | **Implemented and tested** |
| Feeding realised outcomes back into *training* | **Not implemented** |
| Live drift monitoring against production traffic | **Not implemented** — the drift module compares two windows of a stored book; wiring it to live traffic is deployment work |

Unimplemented functions raise `NotImplementedError` with the phase named. They do
not return placeholder values.

---

## 9. Decisions worth defending

**SQLAlchemy 2.0 declarative, not SQLModel.** SQLModel unifies the wire contract
and the storage schema. Here they are deliberately different things — see §3.2.

**No metrics framework, no MLflow.** A plain JSON experiment log plus the config
fingerprint gives full traceability. MLflow would be infrastructure to run and
explain for a benefit this project does not yet need.

**Recharts, and only for the console.** The reliability diagram, the PR curve and
the threshold sweep genuinely need a charting library. Nothing else does.

**Parquet, not CSV, for dataset artefacts.** A CSV round-trip loses exactly the
two dtypes that matter here: timezone-aware timestamps, and the difference
between a NULL label and the string `"None"`.

**Customer pools before temporal windows.** The alternative that satisfies both
split rules — earliest split wins, drop later orders — is badly biased toward
cold-start customers in validation and test. Measured, not assumed: 43% versus
87%. See §3.3 and `data/splits.py`.

**501 rather than stubbed responses.** Discussed in §3.10. This is the decision
most likely to make an early demo look worse and the project look honest. Phase 7
retired most of them by implementing the endpoints; the one that remains is
`/v1/evaluation/fairness`, which returns 501 because the audit has genuinely never
been run and a fabricated breakdown would be the most damaging fake in the API.

**The API references orders, it does not ingest them.** There is no endpoint that
accepts an arbitrary order object and scores it. Customer history and geography
aggregates are computed from the merchant's book as of the order's own timestamp,
so an unpersisted order has no history and would be scored as a first-time
customer in an unknown pincode — confidently, and wrongly, every time. Ingestion
is a separate concern and conflating the two is how a serving path starts lying.

**The model is checked before the order is looked up.** On a server with no
artefact both preconditions fail, and "no model is loaded" is the one an operator
can act on; reporting "no such order" would send them hunting a data problem that
is not there.

**The agent layer declares its tools; the composition layer implements them.**
`agents` is forbidden from importing `decision`, `models`, `features`, `data` or
`eval`, which is the mechanical form of "the LLM is downstream of the decision".
That rule would be worth little if the package simply imported the decision
engine to read from it — so `agents.tools` declares schemas, a protocol and a
permission boundary, and `serving.agent_tools` implements them. An agent receives
a toolset as an argument; it cannot construct one.

The narrow exception that follows — `serving` may import `agents.tools` and
`agents.audit` — is encoded in the layering tests along with its complement:
`serving` may **not** import an agent job or the provider. Without that half,
"serving may import agents" would quietly permit the composition layer to start
running language models.

**Retrieved values are reported; the model's prose is not.**
`RiskInvestigation` carries `probability`, `band`, `threshold` and
`model_version` copied from the tool results. A model asserting a different
number changes the sentence and nothing else. That is the structural reason an
agent cannot alter a risk decision, and it is asserted directly.

**Provenance travels with the number, not in a footnote.** An economic report
mixes measured metrics, merchant inputs, published figures, simulator parameters
and derived arithmetic, and on a dashboard they look identical. `Quantity` cannot
be constructed without a `Provenance`, and `ASSUMED_INTERVENTION` is its own
category — separate from `PUBLISHED` — because intervention effectiveness is the
number the rupee figures are most sensitive to and the one nobody here has
measured. See `contracts/provenance.py`.

**Expected economics need no labels; realized ones need them; both are reported.**
A calibrated probability *is* an expectation, so a merchant can price today's
unlabelled order book. That is entirely contingent on the calibration being good,
so where labels exist the realized figures are computed alongside and the gap
between them is reported as a calibration check.

**The graduated ladder is priced, not assumed to be worth it.** Under the
multipliers this project ships with, applying one action uniformly above the
threshold beats the ladder — see §9 and `docs/economics.md`. That result is
computed by `compare_ladder_against_uniform` and reported rather than buried.

**The ladder is ranked on net rupees, not PR-AUC.** `evaluation.yaml` declares
`net_inr_saved_per_1000_orders` as the primary metric and the schema refuses to
load if that changes. A model that ranks well and loses money has not earned
production; if a simpler rung wins on money, the simpler rung ships.

**Net savings are measured against doing nothing, and the false-negative term
cancels.** `net = TP × S_tp − FP × C_fp`. Missed RTOs cost the same whether or
not a model exists, so subtracting them from the intervention's net would
double-count a loss already inside the do-nothing baseline. An earlier
implementation did exactly that and inflated the headline roughly threefold; the
arithmetic is now hand-checked in
`test_net_savings_are_measured_against_doing_nothing`.

**LightGBM is not assumed to win, and on this benchmark it does not.** Rung 3
beats rung 4 on both PR-AUC and net rupees, reproducibly across seeds. The
mechanism was investigated rather than tuned away — see
[docs/ladder_results.md](docs/ladder_results.md) and README § Honest metrics.
Phase 5's capacity search closed most of that gap by shrinking the model rather
than growing it.

**Model selection uses a one-standard-error rule, not the highest number.** The
validation split is about two thousand orders, so the 95% interval on PR-AUC
spans roughly ±0.05 and a 0.005 gap between two candidates is noise. Every
candidate within one standard error of the best is treated as tied, and the
smallest model among those wins. The bias towards smaller is deliberate: the
failure the search exists to fix was excess capacity.

**The sealed set is opened by a separate command, gated on a frozen manifest.**
`rto-sentinel final` writes a `SelectionManifest` containing every decision and a
hash over those decisions; `rto-sentinel final-test` refuses to run without one,
refuses to run twice without `--again`, requires a written unseal reason, and
loads the frozen artefact rather than retraining. Re-rendering a report never
requires re-measuring: `rto-sentinel final-report` reads saved artefacts and
touches no split.
