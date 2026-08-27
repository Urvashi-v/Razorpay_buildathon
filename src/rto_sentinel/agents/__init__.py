"""The AI agent layer - downstream assistance only.

THE ARCHITECTURAL PRINCIPLE (SPEC section 08)
=============================================
The LLM is downstream of the decision, never inside it. If every LLM call fails,
the system still scores orders, still applies the correct threshold, and still
takes the right action - it just explains itself less gracefully.

Anything else makes the risk engine non-deterministic, and a non-deterministic
risk engine cannot be audited.

WHAT THIS PACKAGE MAY NOT DO
----------------------------
* generate a risk probability
* choose or adjust the economic threshold
* override, modify or re-run a decision from the engine
* fabricate evidence, a figure, or a feature
* approve or block an order, silently or otherwise

These are not conventions. ``rto_sentinel.decision`` does not import this
package, ``tests/architecture/test_layering.py`` asserts that it never will, and
the only types this package can construct are the grounded, describe-only ones in
``rto_sentinel.contracts.explanation`` - none of which carries a probability, a
threshold, a band, or an action.

THE FOUR JOBS
-------------
1. :mod:`~rto_sentinel.agents.reason_code_writer` - phrase SHAP reason codes.
2. :mod:`~rto_sentinel.agents.confirmation_writer` - draft customer confirmations.
3. :mod:`~rto_sentinel.agents.digest_writer` - prose around SQL-computed figures.
4. :mod:`~rto_sentinel.agents.address_repair` - suggest an address correction.
"""

from rto_sentinel.agents.address_repair import suggest_repair
from rto_sentinel.agents.confirmation_writer import draft_confirmation
from rto_sentinel.agents.digest_writer import write_digest
from rto_sentinel.agents.grounding import (
    GroundingVerdict,
    validate_feature_grounding,
    validate_figure_grounding,
    validate_neutral_framing,
)
from rto_sentinel.agents.provider import (
    AgentUnavailableError,
    AnthropicProvider,
    LLMProvider,
    UnavailableProvider,
    get_provider,
)
from rto_sentinel.agents.reason_code_writer import write_explanation
from rto_sentinel.agents.tools import AgentToolset, DigestFigures

__all__ = [
    "AgentToolset",
    "AgentUnavailableError",
    "AnthropicProvider",
    "DigestFigures",
    "GroundingVerdict",
    "LLMProvider",
    "UnavailableProvider",
    "draft_confirmation",
    "get_provider",
    "suggest_repair",
    "validate_feature_grounding",
    "validate_figure_grounding",
    "validate_neutral_framing",
    "write_digest",
    "write_explanation",
]
