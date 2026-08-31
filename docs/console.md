# The merchant console

A React client for the RTO Sentinel API. It has one job: show what the backend
says, and be visibly honest about everything it does not know.

**It computes nothing.** No probability, no threshold, no band, no rupee total is
derived in the browser. Every number on screen arrived in a response body. This
is not a style preference — it is what keeps the console from becoming a second,
untested implementation of the decision rule that disagrees with the one serving
production traffic.

---

## The five screens

| Screen | What it reads | The claim it supports |
|---|---|---|
| **Dashboard** | `/v1/monitoring/data`, `/v1/monitoring/model`, `/v1/evaluation/final?split=test` | Row counts, the artefact actually loaded, and the sealed-set economics |
| **Order queue** | `/v1/orders` with filters as query parameters | Real rows; the total is the *filtered* total |
| **Investigation** | `/v1/orders/{id}/risk` | One order scored end to end by the real pipeline |
| **Simulator** | `/v1/economics/cost-profiles`, `POST /v1/economics/simulate` | The backend re-derives the threshold and re-prices the book |
| **Evaluation** | `/v1/evaluation/final`, `/v1/evaluation/ladder` | Validation and sealed test kept in separate columns |
| **Fairness & drift** | `/v1/evaluation/fairness`, `/v1/evaluation/shift`, `/v1/monitoring/drift` | Cohort audit with intervals, the shift study, and drift that never reads as failure |

---

## Four rules, and where each is enforced

### 1. A missing value renders as an em-dash, never as zero

`components/format.ts` returns `—` for `null` and `undefined`. There is no
default parameter anywhere in it. A `0` on this screen is a measured zero.

The type system carries the same rule: `AsyncState<T>` is a discriminated union,
so `data` does not exist until `status === 'success'`. There is nothing to write
`data ?? fallback` against, because during a request there is no `data` property
to fall back from.

`is_rto: null` gets its own visible treatment — `pending`, not `delivered`.
Rendering an immature order as delivered is the single most effective way to make
a risk console optimistic, so `OutcomeCell` branches on `null` before it branches
on the boolean.

### 2. Economic arithmetic happens on the server

Moving a slider changes local state. Pressing **Recompute on the server** sends
the inputs to `POST /v1/economics/simulate`, and the backend re-derives the
threshold, re-resolves every band boundary, re-assigns every order in the scored
book and re-prices the result.

Computing `C_fp / (C_fp + S_tp)` in JavaScript as the user drags would be
snappier. It would also mean two implementations of the decision rule, and the
one on screen would be the one nobody tested. The visible cost is a real loading
state while the server recomputes. That is the honest trade.

The direction this exposes is counter-intuitive and worth checking by hand:
raising the contribution margin from ₹250 to ₹1,800 moves the threshold from
0.348 to 0.776 and the flag rate from 19.1% to 0.1%. **A higher margin flags
less**, because a false positive costs more when the order you drive away was
worth more. `0.25 × 1800 + 8 = 458`, `0.6 × 220 = 132`, `458 / (458 + 132) =
0.776`.

### 3. A failed request shows the failure

`ErrorState` renders the backend's own error code and message. There is no
retry-until-it-looks-fine, no cached previous value presented as current, and no
canned text.

This matters most on the agent panel. When the language layer is unconfigured the
panel says so, quotes the backend's reason, names the environment variables, and
points at the reason codes — which come from the model, not from a language
model, and are the artefact of record. It never substitutes prose of its own for
prose the backend did not produce.

The panel also renders `probability`, `band`, `threshold` and `model_version`
from the response's structured fields, never scraped from the model's summary
text. That is what makes it structurally impossible for a language model to
change what this screen reports about a decision.

### 4. Validation and sealed test are never averaged together

The evaluation table has two columns, headed *selection-contaminated* and *the
honest read*. Hyperparameters were chosen on validation and the shipped
calibrator was refitted on it; the sealed test set was opened once, after every
choice was frozen. Presenting a single "accuracy" number would merge a figure the
model was tuned against with one it was not.

Where a metric was not computed for a split — `Recall @ precision 80%` on the
sealed set — the cell is an em-dash.

---

## Testing

`OrderInvestigation.test.tsx` covers the chain the spec asks for: render the
order page → the client issues a request → the backend's risk response comes
back → the returned values are what appears on screen.

It stubs `fetch` rather than the endpoint module, deliberately. Stubbing
`endpoints.ts` would leave URL construction — the query parameters, the encoding
of the order id — untested, which is exactly where a client silently asks for the
wrong thing.

```bash
cd console && npm run test
```

Typecheck, lint, test and build all run from `scripts/check.sh`.

---

### 5. A thin cohort never looks like a solid one

On the fairness screen, a group below the minimum support is shown — suppressing
thin groups would hide exactly what an audit exists to look at — but the row is
de-emphasised and carries a "too thin to read" marker. Opacity alone would fail
anyone who cannot perceive it, so the marker is text.

Every rate is printed with its Wilson interval, so a reader sees the precision of
the estimate at the same moment they see the estimate. A cohort table without
intervals invites comparing 0.44 against 0.47 as though the difference were real.

### 6. Drift is shown as distance, never as failure

The distances table and the labelled comparisons are separate sections with
separate headings. When the current window has no matured outcomes, a banner says
so before anything else and the labelled section says "this is the honest empty
state, not a passing result". The severity badges read `stable` / `watch` /
`investigate` and are deliberately not a red-amber-green palette.

The shift table leads with **PR-AUC lift**, not raw PR-AUC. Raw PR-AUC is present
but muted, because a random ranker scores PR-AUC equal to the positive rate — so
raw PR-AUC rises when the base rate rises, and leading with it would report the
arithmetic of the base rate as a property of the model.

---

## Not built

**Nothing on this console is a stub.** Each screen either shows backend data or
shows the backend's reason for not having any.

Where an experiment has not been run, the endpoint returns 501 with its reason
and the screen renders that reason rather than an empty table — an empty fairness
table reads as "we checked and found nothing", which is a different and much
worse claim than "we have not checked".

---

## Development note

If a panel that reads the database hangs while panels that do not render fine,
check the database URL before suspecting the client. `localhost` resolves to
`::1` first, and a container published as `127.0.0.1:5442->5432` is not listening
there; libpq falls back to IPv4 only after a connect timeout that ran ~130
seconds on Windows. Use `127.0.0.1` in `RTO_DATABASE_URL`. The engine now sets an
explicit `connect_timeout` (`db/session.py`), so this fails fast rather than
hanging, and `tests/db/test_engine_config.py` keeps it that way.
