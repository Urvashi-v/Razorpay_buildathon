# RTO Sentinel — Architecture

> A cost-calibrated return-risk scorer for Indian COD commerce — that knows what
> a false positive costs.

This document defines the components, their responsibilities, and the boundaries
between them. Where a boundary is enforced by a test rather than by convention,
the test is named. Where something is not built yet, it says so.

**Status: Phase 1 (architecture and foundations).** The data pipeline, models,
decision arithmetic, console UI and agent layer are scaffolded with fixed
interfaces and are not implemented. See [Implementation status](#implementation-status).

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
│   ├── data/                # generator, validation, as-of joins, splits
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
├── migrations/              # Alembic
├── tests/
│   ├── unit/                # contracts, config, settings, refused features
│   ├── architecture/        # the layering rules, mechanically enforced
│   ├── api/                 # health, contract surface
│   ├── db/                  # schema shape and privacy commitments
│   └── leakage/             # the four leakage tests from the spec
├── docker/                  # API and console images
└── scripts/                 # setup and check
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
| `generator.py` | Synthetic order sampler driven entirely by `generator.yaml` |
| `schema.py` | The canonical column list and `FORBIDDEN_IN_FEATURES` |
| `validation.py` | Structural checks on any order table |
| `asof.py` | As-of joins — the single reviewed implementation of the leakage rule |
| `splits.py` | Temporal + grouped split assignment, and the test-set seal |

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

Rungs 0–5 behind one interface, so the evaluation harness scores a do-nothing
baseline and a calibrated LightGBM identically and has no idea which is which.

`predict_proba` returns probabilities. **Nothing in this layer returns a band, an
action, a rupee figure or a threshold** — those belong to `decision/`, and
keeping them out of the interface is what stops a model quietly deciding policy.

`calibration.py` is a first-class component, not a postprocessing step: the
entire decision layer depends on the score being an honest probability. Isotonic
regression is fitted on **validation** — never train (calibrating to one's own
overfitting), never test (contaminating the seal).

### 3.6 Decision engine (`decision/`) — the authority

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
anywhere in the policy.

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
  `evaluation.yaml`.

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

**The decision log is append-only by construction.** There is no `updated_at`
column and `DecisionRepository` has no update method. An audit trail that can be
edited is not an audit trail.

**`ReadOnlyRepository` is what the agent layer receives.** It has no write
methods to call, so "the LLM must not modify a decision" is a fact about the type
it holds.

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

### 3.11 Console (`console/`)

React + TypeScript + Vite. **Reads every number from the backend and computes
none.** A chart that can invent its own numbers is a picture, not a report.

One HTTP client; no component fetches on its own. That leaves nowhere for a
component to hardcode a fallback when a call fails.

---

## 4. Boundaries, enforced

`tests/architecture/test_layering.py` parses the import graph — including
`TYPE_CHECKING` blocks — and asserts:

| Rule | Test |
|---|---|
| `decision/` imports no LLM SDK and no `agents/` | `test_decision_layer_never_imports_an_llm` |
| `decision/` imports no database, web framework or HTTP client | `test_decision_layer_is_free_of_io` |
| `agents/` imports no `decision/`, `models/`, `features/`, `data/`, `eval/` or ML library | `test_agents_cannot_reach_the_decision_engine_or_models` |
| Route handlers import no ML library | `test_no_ml_logic_in_route_handlers` |
| `data/`, `features/`, `models/`, `eval/` import no `api/` or `db/` | `test_ml_layers_do_not_import_the_web_or_database_layer` |
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
| Database schema (5 tables, constraints) | **Implemented and tested** (SQLite; no PostgreSQL run yet) |
| API app, error envelope, `/health`, `/readiness` | **Implemented and tested** |
| API contract for all 17 endpoints | **Published in OpenAPI**; 15 return 501 |
| Console shell + live backend status | **Implemented and tested** |
| Architecture layering tests | **Implemented and passing** |
| Data generator, validation, as-of, splits | Interfaces fixed — Phase 2 |
| Cost model, threshold derivation, policy, engine | Interfaces fixed — Phase 2 |
| Feature families and pipeline | Interfaces fixed — Phase 2 |
| Ladder rungs 0–5, calibration, artefacts | Interfaces fixed — Phase 3 |
| Metrics, economics, bootstrap, fairness, report | Interfaces fixed — Phase 4 |
| Repositories, initial Alembic revision | Interfaces fixed — Phase 4 |
| Console: queue, sliders, charts, fairness | Phase 4 |
| Agent layer (4 jobs + grounding) | Interfaces fixed — Phase 5 |
| Drift monitoring, outcome loop | Interfaces fixed — Phase 6 |

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

**No initial Alembic revision yet.** The schema is settled but unexercised
against live PostgreSQL. A migration history that begins with a correction is
worse than one that begins a step later. See `migrations/README.md`.

**501 rather than stubbed responses.** Discussed in §3.10. This is the decision
most likely to make an early demo look worse and the project look honest.
