# The agent layer

Four language jobs sit downstream of the risk system. None of them can score an
order, choose a threshold, or change a decision — and that is enforced by the
dependency graph, not by a convention.

**Current state: the language layer is not configured.** No `ANTHROPIC_API_KEY`
is set, so every agent endpoint reports itself unavailable with a reason. The
risk system is unaffected: scoring, calibration, the decision engine and every
endpoint outside `/v1/explanations` work exactly as they did. See
[Configuring it](#configuring-it).

---

## The architectural rule, and how it is enforced

The LLM is downstream of the decision, never inside it. Three mechanisms, in
increasing order of how hard they are to defeat:

**1. The package cannot import the decision engine.**
`tests/architecture/test_layering.py` forbids `agents` from importing `decision`,
`models`, `features`, `data` or `eval`. There is no route from an agent to the
code that produces a probability.

**2. The tools are the whole capability surface.** Six tools, every one a `get_`.
No tool writes, scores, re-bands, approves, blocks or sends. The list is
published at `GET /v1/explanations/tools` so the claim can be checked rather than
believed.

**3. The retrieved values are reported, not the model's prose.**
`RiskInvestigation` carries `probability`, `band`, `threshold` and
`model_version` copied from the tool results. A model that writes "this is
actually low risk at 5%" changes the sentence and nothing else — the fields still
say 0.5719 and HIGH. `test_retrieved_values_are_reported_not_the_models_prose`
asserts exactly that.

Because the agent layer declares the tool *contract* but cannot import the
decision engine, the concrete toolset lives in `serving.agent_tools`, which is
already allowed to compose those layers. An agent receives a toolset as an
argument; it cannot construct one.

---

## The tools

| Tool | Returns | Boundary |
|---|---|---|
| `get_order` | Value, payment method, category, courier, pincode **tier**, address-quality signals, outcome if resolved | Read-only. No name, phone or address text. |
| `get_customer_history` | Prior order count, prior RTO count and rate, recent prior orders | Read-only. **Strictly as-of** the order's timestamp. |
| `get_risk_prediction` | The calibrated probability, with model and feature versions | Read-only. Cannot compute or adjust a probability. |
| `get_model_explanation` | SHAP attributions, reason codes, and `permitted_features` | Read-only. `permitted_features` is the allow-list the validator enforces. |
| `get_economic_decision` | Band, action, threshold, and the economics behind it | Read-only. Cannot choose a threshold or change a band. |
| `get_relevant_order_events` | The delivery timeline in sequence | Read-only. |

### Two design decisions worth stating

**Absence is a value, not an exception.** Every tool returns `found` and, when
false, a `reason` written for the model: *"no order 'ORD-99999999' exists in the
database… Do not describe this order; say the evidence is unavailable."* That is
what lets the agent distinguish "the evidence says no" from "the evidence is
missing" — a distinction a language model will otherwise paper over.

**Customer history is read as-of the order.** Only orders that had already
*resolved* before this order's timestamp are returned — the same cutoff the
model's features use. Without it an agent could cite history the model never had,
producing a sentence that is true about the customer and wrong about the
decision.

---

## What the agents do

### 1. Risk investigation — `POST /v1/explanations/{order_id}/investigate`

A real tool-use loop. The model is given the six tools and no pre-loaded context;
it decides what to fetch, and the loop executes those calls against the live
database and model. Nothing is stuffed into the prompt in advance, because a
prompt full of pre-fetched context is one where nobody can tell which evidence
the model actually used.

Three outcomes, and no fourth:

1. **Grounded explanation** — evidence retrieved, output validated.
2. **Insufficient evidence** — `sufficient_evidence: false` with what is missing.
   This is a success. An honest "I cannot tell you" is the correct answer.
3. **Rejected** — the validator refused it; the reason codes stand without prose.

### 2. Confirmation message — `POST /v1/explanations/{order_id}/confirmation`

Customer-facing copy for a frictioned order. The action comes from the decision
engine; the agent only chooses words. A LOW-band order has no confirmation to
draft and the endpoint says so rather than messaging a customer the policy chose
not to contact.

Rejected copy falls back to a **human-reviewed template**, labelled as such
(`llm_model: "template"`, `grounded: false`). The worst case is slightly less
tailored wording, never a customer told they are suspected of something.

### 3. Merchant digest — `GET /v1/explanations/digest`

SQL computes every number; the model writes sentences around them. A digest with
an invented figure is rejected and the figures table is returned without prose. A
wrong number is therefore a bug in a query — findable — rather than a
hallucination.

### 4. Address repair — **deferred**

`POST /v1/explanations/address-repair` returns **501** with its reasoning. Three
reasons, each sufficient alone:

1. The benchmark addresses are synthetic strings; a suggester would be repairing
   text unrelated to real Indian addresses.
2. The agent layer is denied raw address text by design — a model that also
   drafts customer-facing copy should not hold delivery addresses.
3. Correctness needs a postal reference dataset this project does not have, and a
   confidently wrong "corrected" address is worse than a flagged incomplete one.

The shipped behaviour is to ask the customer to confirm their own address.
Nothing here rewrites one.

---

## Grounding

Four validators, all of which **reject rather than repair**. A repaired
hallucination is still a hallucination that got close enough to pass.

| Validator | Catches |
|---|---|
| `validate_feature_grounding` | Risk drivers this system has no feature for — chargebacks, credit scores, device fingerprints, protected attributes |
| `validate_figure_grounding` | Numbers in a digest that were not computed and handed to the model |
| `validate_neutral_framing` | Customer copy that discloses risk assessment or implies suspicion |
| `validate_evidence_references` | Claimed support from evidence no tool returned |

### What these checks are, and are not

They are a **blocklist over a known vocabulary**, not a proof of truthfulness.
`validate_feature_grounding` catches a sentence citing a driver this system does
not have, because those are the fabrications that matter and they are nameable in
advance. It cannot catch a fluent sentence that misdescribes a driver it *was*
given.

That limit is why the **reason codes, not the prose, are the artefact of record**.
The sentence is a convenience laid over them.

One implementation note that turned out to matter: terms are matched at **word
boundaries**, not as substrings. "age" is a protected attribute and also lives
inside "average" — a validator that rejects "above the merchant's average" gets
switched off within a week, and then it protects nothing.

---

## Audit

Every run records: agent type, request, subject id, provider, model, timestamps,
duration, every tool invoked with its arguments and whether it found anything,
LLM turn count, token usage, the grounding verdict, and the final structured
output. Failed runs record their error.

Deliberately **not** recorded: the API key (it never reaches this layer), raw
prompts (reconstructible from agent type and subject), and tool outputs (the
database is already the record).

`test_the_audit_record_carries_no_secret` asserts the first of those.

---

## Configuring it

| Variable | Current | Needed |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | a real key |
| `RTO_AGENTS_ENABLED` | `false` | `true` |

Both, in `.env` at the repository root (git-ignored). A key alone is not enough —
`LLMSettings.enabled` requires both, so a key sitting in an environment cannot
switch the language layer on by accident.

The `anthropic` SDK is declared in `pyproject.toml` but **not installed in the
venv**: `pip install anthropic`. A missing SDK is reported as an unavailable
language layer, not an import error at startup.

`ANTHROPIC_BASE_URL` is honoured if set, for a gateway or proxy.

```bash
curl -s localhost:8000/v1/explanations/status
```

```json
{
  "available": false,
  "reason": "RTO_AGENTS_ENABLED is false",
  "provider": "anthropic",
  "model": "unavailable",
  "required_environment_variable": "ANTHROPIC_API_KEY",
  "enable_switch": "RTO_AGENTS_ENABLED",
  "tools": ["get_order", "get_customer_history", "get_risk_prediction",
            "get_model_explanation", "get_economic_decision", "get_relevant_order_events"],
  "note": "The risk system does not depend on this layer..."
}
```

---

## There is no fallback, and that is the point

When the language layer is unavailable, `/investigate` returns **503 with the
reason**. It does not return a scripted explanation, a templated sentence, or
anything else shaped like model output.

The test suite drives the agents through a recording double, and that double
lives in `tests/` — never in `src/`.
`test_no_scripted_responder_ships_in_the_product` greps the shipped package for
it. A double in a test proves the orchestration works; a double in the product
hides that it does not.

The endpoints that *can* degrade do so visibly: a confirmation falls back to a
reviewed template with `grounded: false`, and a digest returns its figures with
no prose. In both cases the caller can tell from the response which happened.
