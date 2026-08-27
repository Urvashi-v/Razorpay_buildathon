# ADR 0001 — The LLM is downstream of the decision, never inside it

**Status:** Accepted
**Date:** 2026-08-27

## Context

RTO Sentinel is an "AI Risk Manager" submission, and there is real pressure —
competitive and rhetorical — to put a language model somewhere prominent in the
risk path. The obvious temptation is to have an LLM read an order and produce a
risk assessment, or to have it adjudicate borderline cases.

## Decision

The LLM is downstream of the decision. It describes decisions that have already
been made. It never produces a probability, chooses a threshold, selects a band,
picks an action, or modifies a prediction.

## Rationale

An LLM has four properties that disqualify it from producing a risk decision that
must be auditable:

1. **Slow.** Checkout scoring has a sub-100ms budget.
2. **Expensive.** Per-order inference at commerce volume is not viable.
3. **Uncalibrated.** The entire expected-value layer depends on the score being an
   honest probability. A model that cannot tell you its own reliability cannot
   feed a threshold comparison.
4. **Non-deterministic.** A non-deterministic risk engine cannot be audited, and
   an unauditable risk engine cannot be deployed by a payments company.

The fourth is decisive. A merchant disputing a decision, a regulator reviewing
disparate impact, and an engineer debugging a bad week all need the same thing:
the ability to replay a logged decision and get the same answer. The decision
engine is a pure function of `(calibrated probability, cost inputs)` precisely so
that replay is possible.

## What this costs

The demo is less flashy. "An agent decides whether to block the order" is a more
arresting sentence than "the agent writes one sentence explaining arithmetic that
has already happened". This ADR accepts that cost.

## Consequences

- `rto_sentinel.decision` imports nothing from `rto_sentinel.agents`, and
  `tests/architecture/test_layering.py` fails if that changes.
- Agent outputs are typed as `contracts.explanation.*`, none of which carries a
  probability, threshold, band or action. The boundary is in the type system, not
  only in a review comment.
- The agent layer receives `ReadOnlyRepository`, which has no write methods.
- If every LLM call fails, the system scores, thresholds and acts unchanged. Only
  the wording degrades.
- Grounding validators reject rather than repair. A repaired hallucination is a
  hallucination that got close enough to pass.
