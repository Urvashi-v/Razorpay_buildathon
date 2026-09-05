# RTO Sentinel

**A cost-calibrated return-to-origin risk system for Indian cash-on-delivery
commerce — one that knows what a false positive costs.**

Built for the Razorpay hiring challenge, Track 02 (AI Risk Manager).

---

> ## ⚠️ Read this before any number below
>
> **The labels are simulated.** Every outcome in this project was drawn from a
> causal process this repository wrote and documents in
> [docs/simulator.md](docs/simulator.md). They are not observations of real
> returns.
>
> Every metric here is therefore a statement **about that simulator**. A good
> number means the model learned the simulator. Whether the simulator resembles
> Indian COD commerce is a separate question this project does not claim to have
> settled.
>
> Its base rates are anchored to published figures ([docs/sources.md](docs/sources.md)),
> and it produces no working credentials, no valid identity documents and no
> deliverable addresses.

---

## 1. The problem

An Indian D2C merchant ships a ₹1,200 cash-on-delivery order. Nobody answers the
door. Three days later the parcel comes back. The merchant has now paid freight
twice, a repack, a QC pass and a support call, and has sold nothing.

This happens to roughly **a quarter of COD orders**. COD is about 60–65% of
Indian e-commerce, so it is not an edge case — it is the dominant failure mode
of the dominant payment method.

The naive fix is to block COD on risky-looking orders. That fails for a reason
worth stating precisely: **the cost of being wrong is asymmetric and nobody
measures it.** Block a good customer and you lose the margin, the repeat
business and the goodwill. Most systems optimise for catching returns and treat
the customers they wrongly frictioned as invisible.

## 2. Why RTO, and why this framing

RTO is a **risk-management** problem, not a classification problem, because the
useful output is not "will this return?" but "**is intervening worth it?**" —
and that depends on economics the model has never seen.

The threshold falls out of the arithmetic:

```
threshold = C_fp / (C_fp + S_tp)

  C_fp = abandonment_on_friction × contribution_margin + support_cost
  S_tp = intervention_success_rate × rto_cost
```

For the default merchant profile this gives **0.3481**, not 0.5. And it moves in
a direction most people guess backwards:

The three configured profiles, with the thresholds the API actually derives:

| Profile | Margin | RTO cost | Threshold | SEVERE begins at |
|---|---:|---:|---:|---:|
| `thin_margin_reseller` | ₹90 | ₹180 | **0.2612** | 0.6269 |
| `mid_margin_d2c` (default) | ₹250 | ₹220 | **0.3481** | 0.8356 |
| `high_margin_beauty` | ₹520 | ₹240 | **0.3944** | 0.9466 |

**A higher margin *raises* the threshold, so a high-margin merchant flags less.**
A false positive costs more when the order you drove away was worth more. Push
the margin to ₹1,800 in the simulator and the threshold reaches 0.7763 with a
flag rate of 0.1% — recomputed server-side, never in the browser.

This also explains a measured property of the system: **SEVERE never fires at the
default profile.** It begins at 0.8356 and the highest-scoring order in the book
is 0.8213. The band a customer lands in depends on the merchant's economics as
much as on the model.

## 3. Product overview

A merchant console backed by a scoring API. Six things it does:

1. **Scores an order** through the real pipeline on request — database →
   features → calibrated model → decision engine.
2. **Derives the operating threshold** from that merchant's economics rather
   than using a fixed cut-off.
3. **Applies graduated friction** — a four-rung ladder from no action to
   prepaid-only, with stable reason codes and an appeal path on every rung.
4. **Simulates** — change the margin or the RTO cost and the backend re-derives
   the threshold, re-bands every order and re-prices the book.
5. **Explains** — a read-only investigation agent that retrieves its own
   evidence through six controlled tools and is structurally unable to change
   what the system decided.
6. **Audits itself** — cohort fairness, a controlled distribution-shift study,
   and drift monitoring that never reports "moved" as "broken".

## 4. Architecture

Two flows, kept apart by import rules that fail the build:

```
DECISION FLOW
  console → API → serving → features → LightGBM → Platt calibrator
                                                        ↓
                                            deterministic decision engine
                                          (threshold, bands, reason codes)
                                                        ↓
                                                  API → console

EXPLANATION FLOW
  console → agent → 6 read-only tools → real application data → PostgreSQL
                 ↘ Anthropic Messages API ↗
                          ↓
              grounding validators → audit trail → console
```

The agent **describes** decisions; it cannot make or influence one. Four
mechanisms enforce that, each with a test — see
[docs/architecture.md](docs/architecture.md) for the diagrams and
[ARCHITECTURE.md](ARCHITECTURE.md) for the full module inventory.

## 5. Data strategy

No public dataset carries per-order RTO labels with the customer history and
address quality this problem needs, and scraping one would be both unreliable
and outside what a defensive risk project should do. So the benchmark is
**generated by a documented causal simulator**:

- Latent customer traits (reliability, address-writing quality, prepaid
  affinity) that are **never exposed as features** — this is the irreducible
  error that makes the task realistic rather than a rule waiting to be
  reverse-engineered.
- A per-order logit built from ten documented drivers, then a per-payment-method
  intercept solved by bisection so the marginal RTO rates hit their published
  targets.
- Deliberate noise: a per-order Gaussian shock, a latent per-pincode effect, and
  a 0.4% symmetric label-flip standing in for courier miscoding.

Anchors: ~23% blended RTO, ~26% COD vs <2% prepaid, 62% COD share
([docs/sources.md](docs/sources.md)). Full mechanism:
[docs/simulator.md](docs/simulator.md).

## 6. Synthetic-data disclaimer

Stated once at the top of this file, again on every generated report, again on
every API response that carries a metric, and again in the console's provenance
footer. It appears in those four places because it is the sentence most likely
to be dropped when a result is quoted onward.

**These are controlled benchmark experiments. They are not evidence of
production performance, production fairness or production robustness.**

## 7. Feature engineering

**54 features across six families**, every one computed *as-of the order's own
timestamp*:

| Family | Examples | Why it might be wrong |
|---|---|---|
| `customer_history` | prior RTO rate, order value vs own median | Strongest honest signal; cold-start for new customers |
| `address_quality` | token count, house number present, city/pincode consistency | Text signals only — latent quality stays hidden |
| `order_shape` | COD flag, discount depth, item count, category | Cheap and stable |
| `session_intent` | time-to-checkout, sessions before purchase, cart edits | Weakest family; most likely to be dropped |
| `geography_route` | smoothed pincode rate, tier, courier lane | **Highest fairness risk** |
| `temporal` | hour, late-night flag, sale day, days since last order | Seasonal-shopper confound |

**Leakage prevention is the load-bearing part.** Seven explicit tests
([tests/leakage/](tests/leakage/)) assert that no feature can see its own
outcome, that as-of joins never read forward in time, and that a brute-force
reference implementation agrees with the vectorised one row for row.

The riskiest feature in the project is the smoothed per-pincode RTO rate: a raw
pincode rate is an income and region proxy. Three guards — Bayesian shrinkage, a
minimum support threshold below which it is NaN, and as-of computation. Full
catalogue: [docs/features.md](docs/features.md).

## 8. ML pipeline

A **baseline ladder**, because "our model gets 0.48 PR-AUC" means nothing
without knowing what a merchant could build in an afternoon:

| Rung | Model | PR-AUC (validation) | Train−val gap | Net ₹/1k |
|---:|---|---:|---:|---:|
| 0 | do nothing | 0.191 | −0.023 | ₹0 |
| 1 | blanket COD block | 0.234 | −0.032 | ₹1,228 |
| 2 | pincode blocklist | 0.196 | +0.018 | ₹−808 |
| 3 | logistic regression | 0.483 | −0.043 | ₹3,544 |
| 4 | LightGBM | 0.443 | **+0.519** | ₹3,298 |

**Rung 3 beat rung 4**, and the +0.519 train−val gap says why: the untuned
LightGBM had memorised the training window. That was investigated rather than
tuned away — the simulator draws labels from a linear logit combination, which
is exactly the shape logistic regression fits. The Phase 5 capacity search cut
the gap from +0.519 to +0.012.

Selection uses a **one-standard-error rule**: among candidates within one SE of
the best, take the smallest capacity. Ties break by trees × leaves.

### What each feature family is worth

Leave-one-family-out, retrained per arm, measured in **net rupees** rather than
AUC, on validation only:

| Family removed | Net ₹/1k | Δ vs full | 95% interval | Reading |
|---|---:|---:|---:|---|
| *(full model)* | ₹5,169 | — | — | reference |
| `order_shape` | ₹856 | −4,313 | [−6,163, −2,510] | **earns its place** |
| `geography_route` | ₹4,126 | −1,043 | [−1,948, −88] | **earns its place** |
| `customer_history` | ₹4,412 | −757 | [−1,879, +472] | not established |
| `session_intent` | ₹4,935 | −234 | [−1,063, +575] | not established |
| `address_quality` | ₹4,996 | −173 | [−1,095, +700] | not established |

Two things worth reading carefully. **`geography_route` — the highest
fairness-risk family — does pay for itself, but its interval clears zero by only
₹88.** That is the justification its fairness cost requires, and it is a marginal
one.

And **`session_intent` removal *improved* PR-AUC (+0.009) while costing money.**
That is precisely why this project ranks families on rupees rather than on AUC.

"Not established" is not "worthless": leave-one-out measures what a family adds
*once every other family is present*, and overlapping signal hides individual
value.

## 9. Calibration

The decision layer compares a probability against a threshold. If the score is
not an honest probability, that comparison is arithmetic on a number that
denotes nothing, and every rupee figure downstream is fiction. So calibration is
a headline metric here.

Three candidates — isotonic, Platt-on-logit, identity — selected by K-fold CV
*inside* validation, with a **dual gate**: ECE must improve by a margin **and**
the Brier score must not worsen. The Brier veto exists because binned ECE can
"improve" through binning noise while the probabilities themselves get worse.

Platt was selected. And the honest result:

| | Validation | Sealed test |
|---|---:|---:|
| ECE, calibrated | 0.0136 | **0.0290** |
| ECE, uncalibrated | 0.0285 | **0.0175** |

**Calibration helped on validation and hurt on the sealed set.** The mapping was
fitted on validation and did not transfer. Reported because it is exactly the
kind of result that quietly disappears from a write-up.

## 10. Economic decisioning

Deterministic, and the LLM cannot reach it. The engine:

1. Derives the threshold from `CostInputs`.
2. Resolves band boundaries as multiples of that threshold.
3. Assigns the band, action, reason codes, appeal path and control-holdout flag.

Reported separately and never netted away: net ₹/1,000, **false-positive cost**,
flag rate, intervention rate, expected orders affected. `EconomicResult` keeps
false-positive cost as a required field with nowhere to hide it.

Provenance is typed on every quantity: `MEASURED`, `MERCHANT_INPUT`,
`PUBLISHED`, `SIMULATED`, `ASSUMED_INTERVENTION`, `DERIVED`. Full derivation:
[docs/economics.md](docs/economics.md).

## 11. Intervention logic

| Band | Range (default profile) | Action | Customer-visible | Appeal |
|---|---|---|---|---|
| LOW | p < 0.3481 | none | no | n/a |
| ELEVATED | 0.3481 – 0.5570 | prepaid nudge | yes | yes |
| HIGH | 0.5570 – 0.8356 | confirmation required | yes | yes |
| SEVERE | ≥ 0.8356 | prepaid only | yes | yes |

Three properties the config cannot express away: **no silent hard block**
(`PolicyConfig` rejects `hard_block_allowed: true`), **every band has an appeal
path** (`Decision` rejects `appeal_available=False`), and **a 2% control holdout**
of flagged orders receives no friction — the only way true precision stays
measurable once the system starts acting.

**Measured, and worth stating: the graduated ladder loses to uniform action on
this benchmark, and SEVERE never fires** — no order in the book exceeds 0.8356.
Both are in [docs/ladder_results.md](docs/ladder_results.md).

## 12. AI agents

Real Anthropic tool-use loops over six read-only application tools. The agents
**may** retrieve, summarise and explain. They **may not** compute a probability,
choose a threshold, override the engine, invent evidence, or approve or block
anything.

Four enforcement mechanisms, each tested:

1. `agents/` cannot import `decision`, `models`, `features`, `data` or `eval`.
2. `agents/` reaches `serving/` **only** through the tool registry.
3. The package cannot name a repository, `session_scope`, `.commit(` or
   `session.add(`.
4. `probability`, `band`, `threshold` and `model_version` are copied from tool
   results — the model's prose is never parsed for numbers.

Four grounding validators reject rather than repair: a fabricated driver, a
figure that was not computed, evidence never retrieved, or accusatory customer
copy all fail closed.

**Requires `ANTHROPIC_API_KEY`. Without it the endpoints return 501/503 naming
the variable — there is no scripted fallback.** See §19 and
[docs/agents.md](docs/agents.md).

## 13. Fairness

Audited across **operational cohorts only**: pincode tier, order-value band,
customer-history depth, payment method.

**No sensitive characteristic is examined, inferred or approximated.** There is
no gender, religion, caste, ethnicity, age or income in this data — not
withheld, not present — and `eval/fairness.py` refuses by name any cohort
matching a sensitive token, as a hard error rather than a skipped cohort.

Sealed-set result: **the disparity review did not trigger.** Tier-3 is flagged
~2.5× as often as tier-1 **and with better precision** (0.389 vs 0.327), so the
model is not transferring cost onto tier-3 beyond what its accuracy justifies.
The most-flagged group came **71% of the way** to the precision-drop trigger —
a margin to re-check at the next retrain, not a clean pass.

Every rate carries a Wilson interval; thin groups are shown but excluded from
the comparison in both directions. [docs/responsible_ai.md](docs/responsible_ai.md).

## 14. Distribution shift

Nine named perturbations of the generator — COD share, RTO base rate, category
mix, customer mix, geography mix — with the model **frozen and not retrained**
and the threshold held fixed.

**Read ranking lift, not raw PR-AUC.** A random ranker scores PR-AUC equal to
the positive rate, so raw PR-AUC *rises* when the world gets riskier. The
sharpest finding:

| Environment | RTO rate | PR-AUC | Lift | ECE | Net ₹/1k |
|---|---:|---:|---:|---:|---:|
| `reference` | 16.7% | 0.430 | 2.57× | 0.025 | ₹2,276 |
| `rto_base_rate_down` | 9.6% | 0.299 | **3.10×** | **0.083** | **₹−2,130** |

Ranking *improved*; calibration more than tripled its error and the economics
turned **negative**. Ranking survived, the fixed operating point did not.

## 15. Evaluation

Full results: **[docs/evaluation_report.md](docs/evaluation_report.md)** —
generated from artefacts, never hand-written.

| | Validation<br>*selection-contaminated* | Sealed test<br>*the honest read* |
|---|---:|---:|
| Orders | 2,034 | 1,698 |
| PR-AUC | 0.484 [0.439, 0.536] | 0.365 [0.315, 0.426] |
| ROC-AUC | 0.806 | 0.781 |
| ECE | 0.0136 | 0.0290 |
| Brier | 0.1239 | 0.1162 |
| Recall @ P80 | 0.036 | **—** (unattainable) |
| Flag rate | 19.1% | 15.9% |
| Precision / Recall | 0.482 / 0.482 | 0.370 / 0.376 |
| Confusion TP/FP/FN/TN | 187/201/201/1445 | 100/170/166/1262 |
| False-positive cost | ₹14,170 | ₹11,985 |
| **Net ₹/1,000 orders** | ₹5,169 [₹3,279, ₹7,173] | **₹716 [₹−1,009, ₹2,603]** |

> **The sealed-set interval crosses zero.** On 1,698 orders this measurement
> cannot distinguish the model from doing nothing. The point estimate is
> positive; the evidence does not establish it.

The test split was opened **once**, after model selection, calibration and
threshold methodology were frozen in a content-hashed manifest. Reading it
requires an explicit `--unseal-reason`.

## 16. Setup

**Requirements:** Python ≥ 3.11 (developed on 3.12), Node ≥ 20, Docker.

```bash
git clone <this-repo> && cd rto-sentinel
cp .env.example .env      # then set POSTGRES_PASSWORD
./scripts/bootstrap.sh
```

`bootstrap.sh` does everything: virtualenv, dependencies, config validation,
PostgreSQL, migrations, dataset generation and load, the baseline ladder, the
final model and calibration, the sealed evaluation, economics, fairness, the
shift study, drift, and all four generated reports. It is idempotent — the
generator and trainer are seeded, so re-running reproduces the same dataset run
id and model version.

**Use `127.0.0.1`, not `localhost`, in `RTO_DATABASE_URL`.** `localhost`
resolves to `::1` first and a container published on IPv4 is not listening
there; libpq falls back only after a connect timeout that measured ~130 s on
Windows.

<details>
<summary>Step by step, if you would rather not run the script</summary>

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m rto_sentinel.cli config check
docker compose up -d db
.venv/Scripts/python.exe -m rto_sentinel.cli db upgrade
.venv/Scripts/python.exe -m rto_sentinel.cli seed-db --seed 7 --orders 60000 \
    --customers 20000 --start-date 2025-09-01 --end-date 2026-02-27
.venv/Scripts/python.exe -m rto_sentinel.cli train  --seed 7 --orders 60000 --customers 20000
.venv/Scripts/python.exe -m rto_sentinel.cli final  --seed 7 --orders 60000 --customers 20000
.venv/Scripts/python.exe -m rto_sentinel.cli final-test --seed 7 --orders 60000 --customers 20000 \
    --unseal-reason "selection frozen in the manifest"
.venv/Scripts/python.exe -m rto_sentinel.cli economics
.venv/Scripts/python.exe -m rto_sentinel.cli fairness --split validation
.venv/Scripts/python.exe -m rto_sentinel.cli fairness --split test
.venv/Scripts/python.exe -m rto_sentinel.cli shift --n-orders 9000
.venv/Scripts/python.exe -m rto_sentinel.cli monitor --split validation
.venv/Scripts/python.exe -m rto_sentinel.cli responsible-report
.venv/Scripts/python.exe -m rto_sentinel.cli evaluation-report
cd console && npm install
```

`--customers 20000` is the number **requested**. The generator writes fewer
(activity weights are clipped); `dataset_run.json` records both, because the
requested figure is what the run id hashes.

</details>

## 17. Running the project

Two processes, two terminals:

```bash
.venv/Scripts/python.exe -m uvicorn rto_sentinel.api.main:app --port 8000
```

```bash
cd console && npm run dev
```

| | |
|---|---|
| Console | <http://localhost:5173> |
| API docs | <http://localhost:8000/docs> |
| Readiness | <http://localhost:8000/readiness> |

Then walk the demonstration against the live system:

```bash
./scripts/demo.sh
```

Every figure it prints is computed during the run — see
[docs/demo.md](docs/demo.md).

With no model trained, the console reports **"model: not ready"** and every risk
panel shows that error. That is correct — the service says so rather than
inventing a probability.

## 18. Testing

```bash
./scripts/check.sh
```

Runs everything CI runs: ruff lint and format, mypy strict, configuration
validation, pytest, then the console's typecheck, lint, test and build.

| Suite | Tests |
|---|---:|
| `tests/unit` | 603 |
| `tests/api` | 120 |
| `tests/leakage` | 29 |
| `tests/db` | 27 |
| `tests/architecture` | 17 |
| **Backend** | **796** |
| Console (vitest) | 30 |

Two integration tests run the full chains: **database → model → decision engine
→ API**, and **agent → tools → real data → LLM → validated response**. The first
fails if the model artefact is unavailable rather than returning a fake
prediction.

## 19. Limitations

**This is not production-ready.** Specifically:

1. **Authentication, scopes, rate limiting, an access log and TLS are
   implemented** — API keys on every `/v1` route, `read`/`write` scopes with
   `read` as the default, a sliding per-key window that can share one counter
   across workers via PostgreSQL, one audit line per request, and TLS
   terminated by a reverse proxy ([docs/deployment.md](docs/deployment.md)).
   What remains there is real but smaller: key rotation is manual, scopes are
   two rather than per-route, and secrets come from the environment rather than
   a secret manager.
2. **No real LLM round trip is covered by any test.** The tool loop, grounding
   and audit are tested against real data with the transport scripted; the
   Anthropic HTTP call itself is never exercised.
3. **The sealed-set economic result is not statistically established** — the
   interval crosses zero.
4. **Intervention effectiveness and abandonment are assumptions**, never
   measured. Every rupee figure inherits that.
5. **~3 s to score one order**, because the as-of feature context is rebuilt per
   request. The model is 0.5% of that. Fine for investigation, not for checkout.
6. **Three feature families have no established economic contribution.** The
   ablation ran (§8b of the evaluation report): `order_shape` and
   `geography_route` pay for themselves; `address_quality`, `customer_history`
   and `session_intent` have intervals spanning zero. That is not evidence they
   are worthless — leave-one-out measures marginal contribution given every other
   family — but nothing here justifies their cost either.
7. **The outcome loop is implemented but has no data.** The treated-vs-control
   comparison runs against the decision log and today reports *"0 treated, 0
   control, remains the configured ASSUMPTION"* — this system has never operated
   on live traffic. `is_assumed` flips to False on its own once both arms reach
   200 matured orders. Until then the 60% intervention success rate is an
   assumption and every rupee figure inherits it.
8. **Fairness is operational, not demographic**, and on synthetic data.
9. **Integration tests run on SQLite**; PostgreSQL is exercised manually and by
   `seed-db`.
10. **Frontend response shapes are checked for field presence, not types.**
    Every field `console/src/types/api.ts` declares must exist in the OpenAPI
    schema, which catches the silent failure — a renamed field rendering as an
    em-dash. A `number` declared against a schema `string` still passes; that
    one surfaces loudly at the first render.

Full detail: [docs/phase11_report.md](docs/phase11_report.md).

## 20. Future work

In the order that would actually matter:

1. **Run the outcome loop on real traffic.** The measurement exists and is
   tested; it needs ~10,000 frictioned orders to accumulate 200 controls at the
   2% holdout rate. That single change replaces the largest assumption in every
   rupee figure with a number.
2. **A feature store** for as-of aggregates, taking scoring from ~3 s to
   checkout-viable.
3. **Separate the overlapping families.** The ablation showed three with
   unestablished marginal contributions; a grouped or forward-selection study
   would say whether they are redundant or genuinely idle.
4. **Recalibrate on a rolling window.** The shift study showed calibration is
   the first thing to break and the thing that breaks the economics.
5. **A real fairness process** — consented, legally reviewed — if this ever
   touched real customers. The current audit cannot answer that question and
   does not pretend to.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/deployment.md](docs/deployment.md) | **Running it for real.** API keys, rate limiting, TLS, and what deploying still does not solve |
| [docs/demo.md](docs/demo.md) | **The demonstration.** Four real orders, the thirteen steps, and what to say when the interval crosses zero |
| [docs/architecture.md](docs/architecture.md) | **The diagrams.** Decision flow, explanation flow, layer rules, request lifecycle |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full module inventory, every boundary test, implementation status |
| [docs/evaluation_report.md](docs/evaluation_report.md) | **All measured results.** Generated, never hand-written |
| [docs/model_card.md](docs/model_card.md) | Intended use, limitations, fairness, drift, every measured number |
| [docs/responsible_ai.md](docs/responsible_ai.md) | Cohort fairness, distribution shift, drift, and what none of it proves |
| [docs/economics.md](docs/economics.md) | Threshold derivation, friction ladder, sensitivity |
| [docs/simulator.md](docs/simulator.md) | **How the data is produced**, and what it can and cannot demonstrate |
| [docs/features.md](docs/features.md) | Every feature, its lookback, and why it is available at scoring time |
| [docs/splitting.md](docs/splitting.md) | The split protocol, and why random splitting would be dangerous |
| [docs/agents.md](docs/agents.md) | Tools, permission boundaries, grounding, audit |
| [docs/api.md](docs/api.md) | Endpoint reference with real request/response examples |
| [docs/console.md](docs/console.md) | The five screens and the rules they hold |
| [docs/phase11_report.md](docs/phase11_report.md) | Integration audit, security findings, measured latency |
| [docs/ladder_results.md](docs/ladder_results.md) | Measured baseline-ladder results |
| [docs/sources.md](docs/sources.md) | Every market figure, with its citation |
| [docs/adr/](docs/adr/) | Architecture decision records |

## Defence-only

This system reduces fraud and operational loss. It generates no working payment
credentials, no valid identity documents and no deliverable addresses, and
nothing here is usable outside this repository's own evaluation harness.

## Licence

Prepared for the Razorpay hiring challenge. Not licensed for production use —
see §19.
