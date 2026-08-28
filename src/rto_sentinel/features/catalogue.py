"""Rendering the feature dictionary.

``docs/features.md`` is generated from :class:`FeatureSpec` declarations rather
than written by hand. Hand-written feature documentation is wrong within a month
- someone adds a column, someone else changes a lookback window, and the document
quietly becomes fiction that a reviewer nonetheless trusts.

Generating it means the documentation cannot drift from the code, because it *is*
the code. ``tests/unit/test_feature_catalogue.py`` asserts the committed file
matches what the current declarations would produce, so a stale copy fails CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.features.spec import FeatureSet

# The markdown table rows below exceed the code line limit. They are generated
# output, not code, and breaking them would break the rendered table.
HEADER = """# Feature dictionary

**Generated from code. Do not edit by hand.**

Regenerate with:

```bash
rto-sentinel features docs
```

Every feature below is declared in `src/rto_sentinel/features/` as a
`FeatureSpec`, which carries its own source columns, lookback window, observation
point and availability. The pipeline refuses to emit any feature whose
availability is not `at_order_time`, and
`tests/leakage/test_feature_leakage.py` checks the declarations against the data
itself - a feature that claims to be outcome-independent and behaves otherwise
fails a test.

## How to read the observation point

| Value | Meaning | Knowable at checkout? |
|---|---|---|
| `order_payload` | Comes straight off the order being scored | Yes, trivially |
| `customer_record` | Fixed when the account was created | Yes |
| `prior_orders_placed` | Over orders **placed** earlier | Yes - the merchant saw them |
| `prior_orders_resolved` | Over orders that **came back** earlier | Only outcomes that arrived |
| `population_resolved` | Aggregated across all customers' resolved orders | Yes, same caveat |
| `post_order` | Not knowable at scoring time | **No** - refused by the pipeline |

The distinction between `prior_orders_placed` and `prior_orders_resolved` is the
one that matters. "Orders placed in the last 30 days" is knowable instantly.
"Orders returned in the last 30 days" is knowable only for orders that had
actually come back - an order placed on day 40 that returns on day 47 is invisible
to an order placed on day 42. Confusing the two is the most common leak in this
class of problem, so the specs record which clock each window runs on and a
validator refuses an outcome window keyed on placement time.

"""

FOOTER = """
## What is deliberately absent

Four groups of features are refused outright, with the reasons recorded in
`config/features.yaml`:

- **Name-derived features of any kind.** Religion, caste and region inference
  from names is a live harm in Indian systems. Names are hashed for identity
  only, never featurised.
- **Raw pincode as a categorical.** With enough trees this becomes a redlining
  machine. Only smoothed, shrunk aggregates with a minimum support threshold.
- **Gender, age, or anything inferable from them.** No lift worth the exposure.
  Note that `cust_account_age_days` is account *tenure*, not customer age; it is
  listed as an explicit, justified exception in the config.
- **Cross-merchant behaviour.** A consent question this project cannot resolve
  responsibly.

Matching is by whole token, not substring. Substring matching was tried first and
was unusable - the pattern `age` matched `cust_account_age_days` and
`session_product_page_seconds` - and a check that cries wolf gets switched off,
which is worse than no check at all.
"""


def render_markdown(feature_set: FeatureSet, *, feature_version: str, fingerprint: str) -> str:
    """Render the full feature dictionary."""
    lines = [HEADER.rstrip(), ""]
    lines.append(f"**Feature version:** `{feature_version}`  ")
    lines.append(f"**Feature-set fingerprint:** `{fingerprint}`  ")
    lines.append(
        f"**Total features:** {len(feature_set)} across {len(feature_set.families)} families"
    )
    lines.append("")

    for family in feature_set.families:
        subset = feature_set.by_family(family)
        lines.append(f"## `{family}` — {len(subset)} features")
        lines.append("")
        lines.append("| Feature | Type | Observation point | Lookback | Available | Description |")
        lines.append("|---|---|---|---|---|---|")
        for spec in subset:
            lookback = str(spec.lookback) if spec.lookback else "—"
            available = "yes" if spec.is_available_at_prediction_time else "**NO**"
            lines.append(
                f"| `{spec.name}` | {spec.dtype} | `{spec.observation_point}` | {lookback} "
                f"| {available} | {spec.description} |"
            )
        lines.append("")
        lines.append("<details><summary>Source columns and risk notes</summary>")
        lines.append("")
        for spec in subset:
            lines.append(f"**`{spec.name}`**")
            lines.append("")
            lines.append(f"- Source columns: {', '.join(f'`{c}`' for c in spec.source_columns)}")
            if spec.monotonic:
                lines.append(f"- Monotonic constraint: {spec.monotonic}")
            if spec.expected_null_share:
                lines.append(f"- Expected null share: ~{spec.expected_null_share:.0%}")
            lines.append(f"- Risk: {spec.risk_note}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(FOOTER.strip())
    lines.append("")
    return "\n".join(lines)
