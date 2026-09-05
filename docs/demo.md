# The demonstration

Run it:

```bash
./scripts/demo.sh
```

Every figure it prints is computed by the running system during the run. Nothing
is read from a fixture, and if the implementation breaks the script fails — which
is the point of it being a script rather than a document.

**Prerequisites:** `./scripts/bootstrap.sh` has been run, the API is on `:8000`,
and the console on `:5173` if you want to follow along visually.

If the API has keys configured (`RTO_API_KEYS`), set `RTO_API_KEY` to one of
them and the script sends it on every request. A **read** key is enough — the
demo scores, simulates and investigates, and never records a decision:

```bash
RTO_API_KEY=sk_live_... ./scripts/demo.sh
```

Locally the API runs open, so no key is needed. The console needs one only when
the API has them — set `RTO_API_KEY` before `npm run dev` and the Vite proxy
attaches it server-side, keeping it out of the browser.

---

## The four demonstration orders

Real rows in PostgreSQL from dataset run `7b5ae86219ac7cafe45e7d51` — the same
run the shipped model was trained on, reproducible from
`(seed 7, 60,000 orders, 20,000 customers requested, 2025-09-01 … 2026-02-27)`.

They were chosen by scoring the book through the **live** endpoint and taking one
order per band, not by picking pleasing numbers.

| Order | P(RTO) | Band | Action | Actual outcome |
|---|---:|---|---|---|
| `ORD-00048750` | 0.1054 | LOW | none | delivered ✓ |
| `ORD-00046230` | 0.5560 | ELEVATED | prepaid nudge | **rto** ✓ |
| `ORD-00043224` | 0.7789 | HIGH | confirmation required | **rto** ✓ |
| `ORD-00044422` | 0.8213 | SEVERE¹ | prepaid only | **rto** ✓ |
| `ORD-00047511` | 0.5539 | ELEVATED | prepaid nudge | delivered ✗ |

The last row is a **false positive**, included deliberately. A demo that shows
only correct predictions is showing a curated subset, and the whole argument of
this project is that the cost of being wrong is the thing nobody measures.

### ¹ Why SEVERE needs a second profile

At the default `mid_margin_d2c` profile, SEVERE begins at **0.8356**
(threshold 0.3481 × 2.4). The highest-scoring order in the entire book is
**0.8213**.

**SEVERE never fires at the default economics.** That is a measured property of
this model on this data, not a gap in the demo, and it is recorded in
[ladder_results.md](ladder_results.md).

The demo reaches SEVERE the only honest way: by changing the merchant's
economics. Under `thin_margin_reseller` the threshold falls to 0.2612, SEVERE
begins at 0.6269, and `ORD-00044422` lands in it. That is a genuine server-side
recalculation through `POST /v1/score`, and it happens to be the single most
instructive moment in the demo:

```
mid_margin_d2c        margin ₹250  ->  threshold 0.3481   p=0.8213   band HIGH    (confirmation_required)
thin_margin_reseller  margin ₹90   ->  threshold 0.2612   p=0.8213   band SEVERE  (prepaid_only)
```

**The probability did not move.** The model saw the same order and said the same
thing. The merchant's economics moved the threshold, and the band followed. That
is the product.

---

## The thirteen steps

| # | Step | What it exercises | Where |
|---|---|---|---|
| 1 | Merchant opens the dashboard | `/readiness`, `/v1/monitoring/model` | Console → Dashboard |
| 2 | Sees actual risk and economic metrics | `/v1/monitoring/data`, `/v1/evaluation/final` | Dashboard panels |
| 3 | Opens an order | `/v1/orders` filtered server-side | Order queue |
| 4 | Sees the calibrated probability | Full chain: DB → features → LightGBM → Platt | Investigate |
| 5 | Sees why it was flagged | Reason codes + TreeSHAP contributions | Investigate |
| 6 | Sees the economic decision | `C_fp`, `S_tp`, the threshold formula | Investigate |
| 7 | Changes merchant economics | Sliders, no client-side arithmetic | Simulator |
| 8 | Backend recalculates | `POST /v1/economics/simulate` re-prices the book | Simulator |
| 9 | Opens the investigation agent | `/v1/explanations/status` | Investigate → agent panel |
| 10 | Agent retrieves real evidence | Six read-only tools against live data | Agent panel |
| 11 | Agent explains the decision | Grounded response, or an honest refusal | Agent panel |
| 12 | Shows calibration and evaluation | Validation vs sealed test, side by side | Evaluation |
| 13 | Shows fairness and shift | Cohort audit, nine environments | Fairness & drift |

### Steps 9–11 without a credential

`ANTHROPIC_API_KEY` is not configured in this repository. The agent panel shows
the backend's own reason, names the two variables to set, and produces **no
explanation at all**.

That is the designed behaviour, not a broken demo. A canned sentence here would
be indistinguishable from a model-generated one to every consumer, and would
carry the authority of an "AI explanation" with nothing behind it. The reason
codes and SHAP contributions above it are unaffected — they come from the model,
not from a language model, and they are the record.

To enable it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export RTO_AGENTS_ENABLED=true
```

Then restart the API. `demo.sh` detects availability and adapts.

---

## Talking points that survive scrutiny

Four things worth saying out loud, because they are the ones a reviewer will
otherwise find themselves:

**1. The interval crosses zero.** Sealed-set net is ₹716 per 1,000 orders with a
95% interval of ₹−1,009 to ₹2,603. On 1,698 orders this measurement cannot
distinguish the model from doing nothing. Say it before someone asks.

**2. Calibration did not transfer.** ECE improved on validation (0.0285 → 0.0136)
and got worse on the sealed set (0.0175 → 0.0290). The Platt mapping was selected
honestly by cross-validation and still failed to generalise.

**3. Logistic regression beat LightGBM on the first pass**, and the +0.519
train−val gap says why. That was investigated rather than tuned away: the
simulator draws labels from a linear logit combination.

**4. The shift study found a failure the ranking metrics miss.** When the RTO
base rate falls, ranking lift *improves* (2.57× → 3.10×) while calibration error
more than triples and the economics turn **negative** (−₹2,130 per 1,000).
Ranking survived the shift; the fixed operating point did not.

---

## If something fails during the demo

Fix the implementation. There is no fallback path, no fixture mode and no
scripted response anywhere in this system — by design, and asserted by
`test_no_scripted_responder_ships_in_the_product`.

Most likely causes, in order:

| Symptom | Cause | Fix |
|---|---|---|
| One panel spins, others render | Database unreachable | `docker compose up -d db`; check `RTO_DATABASE_URL` uses `127.0.0.1`, not `localhost` |
| Every risk panel shows an error | No model artefact | `rto-sentinel final` |
| Evaluation screens 404 | No evaluation artefacts | `rto-sentinel final-test --unseal-reason "..."` |
| Fairness or drift screen 501 | Experiment not run | `rto-sentinel fairness` / `shift` / `monitor` |
| Agent panel disabled | No API key | Expected — see above |
| Order not found | Database holds a different dataset run | Re-run `scripts/bootstrap.sh` |
| Everything returns 401 | The API has keys and the caller sent none | `RTO_API_KEY=... ./scripts/demo.sh`, and set the same before `npm run dev` |
| One request returns 403 | A read key on a write route | Expected unless you are recording a decision; that needs a `name:secret:write` key |
| Everything returns 429 | Rate limit hit | Wait, or raise `RTO_RATE_LIMIT_PER_MINUTE` |
