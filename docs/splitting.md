# Splitting, and why random splitting would be dangerous

**Short version:** a random split of transactions would let this model read the
future, memorise individuals, and report a score several points better than
anything it could achieve in production. It is the single most common way a risk
model produces a fake number.

---

## What a random split actually does here

Take 120,000 orders spanning 180 days, shuffle them, and hold out 20%. It looks
neutral. It is not, for three separate reasons, each of which independently
invalidates the result.

### 1. It leaks the future into the past

RTO risk is not stationary. Festival cycles, sale events, courier changes and
seasonal shopping all move the base rate. A random split puts orders from day 175
into training and orders from day 5 into test, so the model learns from a period
it would not have had, and is graded on a period it has already seen.

More concretely: with a random split the model can learn "November behaves like
this" from November rows in training and then be graded on other November rows in
test. In production it would have had to learn November from October.

The temporal split forces the honest question: **given only the past, how well
does this generalise forward?**

### 2. It lets the model memorise individuals rather than learn behaviour

The strongest signal in this problem is a customer's own return history. With a
random split, a customer with eight orders typically has six in training and two
in test.

The model does not need to learn *why* that person returns things. It learns
*that this person* returns things, and is then graded on the same person. That
transfers to no new customer, which is exactly the population a merchant most
needs a score for.

The customer-disjoint constraint forces the model to generalise across people.

### 3. It makes as-of discipline unenforceable

With a random split, "was this feature available at scoring time?" has no stable
answer, because there is no shared notion of *now*. A feature can be leak-free
relative to one row and leaking relative to another, and no single check
distinguishes them.

With a temporal split, every training row precedes every validation row, which
precedes every test row. "Available at the time" becomes a testable property -
and the seven leakage tests in `tests/leakage/` test it.

---

## What this project does instead

Three rules, from SPEC section 03, implemented together in
`src/rto_sentinel/data/splits.py`.

| Rule | Implementation |
|---|---|
| **Temporal** | Train days 1-126, validation 127-147, test 148-180 |
| **Grouped** | No `customer_hash` in more than one split |
| **Sealed** | Test scored exactly once, after the threshold is fixed on validation |

### The conflict between rules 1 and 2, and how it is resolved

A customer who orders in both the training window and the test window violates
rule 2. There are three ways out and only one is sound.

**Move their orders into their earliest split.** Wrong. It drags later orders
backwards into training, putting future rows in the training set - breaking rule 1
to satisfy rule 2, and reintroducing exactly the leakage the temporal split
exists to prevent.

**Assign each customer to their earliest split and drop their later orders.**
Satisfies both rules. This was the first implementation, and it is *badly
biased* - which matters because the bias is invisible in the split counts.
Measured on a 20,000-order sample:

| Split | First-time customers |
|---|---|
| train | 43% |
| validation | **87%** |
| test | **88%** |

Validation and test end up composed almost entirely of customers who had no
orders in the training window, because any customer who *did* appear there was
removed. A threshold fitted on that is a cold-start threshold, and a test score
measured on it is a cold-start score wearing a general-performance label.

**Partition customers into disjoint pools first, then apply the temporal window
within each pool.** What this project does. Pool assignment is a deterministic
SHA-256 of the customer identifier and a fixed salt - independent of behaviour,
reproducible, identical on every machine. Every split therefore keeps the
population's new-versus-repeat mix.

### What it costs

Roughly half the dataset falls outside the three modelling splits, marked
`excluded_group_protocol`. A customer in the validation pool contributes only
their orders inside the validation window.

That cost is real, it is reported in `SplitAssignment` rather than hidden, and it
is why the default dataset is large. Pool shares are deliberately *not* equal to
the window shares - the validation window is only 21 days wide, so its pool is
enlarged to keep the split sizes comparable:

```yaml
pool_shares: {train: 0.55, validation: 0.27, test: 0.18}
```

### What is *not* lost

A dropped row still contributed to its customer's as-of history. A
validation-window order retains the prior-order counts earned by that customer's
earlier, excluded orders - which is correct, because at serving time the merchant
genuinely does know that history.

### The residual difference, which is real and should be there

After pooling, later windows still contain more returning customers:

| Split | First-time customers | Mean prior orders |
|---|---|---|
| train | 38% | 1.6 |
| validation | 13% | 4.6 |
| test | 11% | 5.6 |

This is genuine temporal maturation - a merchant's customer base does accumulate
history over 180 days - and every temporal split has it. It points the *opposite*
way to the selection bias it replaced, which is how you can tell the two apart.

It is also why the new-versus-returning cohort breakdown is a required part of
every evaluation: a model that looks good overall may be carried entirely by the
returning-customer majority in the later windows.

---

## Why the test set is sealed in code

`ModelingDataset.test` raises `TestSetAccessError`. Reaching it requires
`unseal_test(reason=...)` with a written justification, which is recorded on the
dataset object.

This is not paranoia about malice. It is about the 2am accident: a quick
`dataset.test` in a notebook to "just check", which contaminates the one number
the whole submission rests on and leaves no trace it happened.

An honour system has a perfect record of failing under deadline pressure. A
property that raises does not.

The dataset's `describe()` deliberately *does* report the test split's row count
and date range without unsealing it. Those are not outcomes. The label is what
stays sealed.

---

## Feature computation and splits

The feature pipeline is handed the **whole** order table, including test rows.
That looks alarming and is the correct, safer choice.

Features are computed as-of each row's own order time. A training row from day 40
sees only outcomes resolved before day 40, regardless of what else is in the
frame - the as-of machinery does not care whether a later row is labelled "test".

Filtering to a split *before* computing features would be actively worse: a
validation-window order would lose the customer history it genuinely had, and the
model would train on a customer who looks like a first-time buyer when the
merchant knew perfectly well they were not. That is a distribution shift
introduced in the name of safety.

`test_test_set_isolation` verifies the claim rather than asserting it. It builds
features on the full frame, builds them again on a frame truncated to the
training window, and checks every training row is identical. If any feature
aggregated globally - a target-encoded pincode rate computed over the whole
dataset, say - the truncated build would differ and the test fails.

---

## The seven leakage tests

In `tests/leakage/`, all running against real generated data.

| Test | What it proves |
|---|---|
| `test_no_future_outcome_features` | No feature changes when future outcomes are hidden. Run at three cutoffs, over all 54 features at once. |
| `test_label_maturity` | Immature orders carry a NULL label, never an optimistic "delivered", and never enter a modelling split. |
| `test_temporal_ordering` | Every training order strictly precedes every validation order, which precedes every test order. |
| `test_test_set_isolation` | Training-row features are byte-identical whether or not test rows are present. |
| `test_feature_timestamp_integrity` | Each feature's declared observation point matches its behaviour - checked in **both** directions. |
| `test_duplicate_or_near_duplicate_leakage` | No order, and no indistinguishable near-duplicate, spans two splits. |
| `test_customer_history_cutoff` | History features match a brute-force nested scan with the timestamp comparison written out. |

### The method: rewind, rebuild, compare

Most of these work the same way. Pick a cutoff. Rebuild the feature matrix on a
frame where every order that had not resolved by that cutoff has its outcome and
resolution timestamp blanked - the world exactly as it looked at that instant.
Compare, for rows ordered before the cutoff, against the matrix built on full
data. Anything that changed was reading the future.

**Rewinding rather than deleting matters.** An earlier version removed unresolved
rows instead of blanking their outcomes, which silently excluded from the
comparison exactly the rows most likely to leak.

**A single cutoff only exposes leaks that straddle it** - the consuming row must
be before it and the leaked outcome after it - so every rewind test runs at
several.

### The guards can fail

Two tests exist purely to prove the checks are not decoration:

- `test_the_rewind_check_can_actually_fail` injects a deliberately leaky feature
  (an expanding mean over orders *placed* earlier rather than *resolved* earlier)
  and asserts the comparison catches it.
- `test_the_duplicate_check_can_actually_fail` injects a duplicated row across
  two splits and asserts the fingerprint check notices.

A guard nobody has tried to break is a guard that passes everything.
