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
4. :mod:`~rto_sentinel.agents.investigator` - the risk investigation agent, which
   runs a real tool loop over read-only application tools.

:mod:`~rto_sentinel.agents.address_repair` is DEFERRED, with the reasons written
out in that module rather than a stub that fabricates a correction.
"""

from rto_sentinel.agents.address_repair import (
    DEFERRAL_REASON,
    AddressRepairDeferred,
    suggest_address_repair,
)
from rto_sentinel.agents.audit import (
    AgentAuditRecord,
    AuditBuilder,
    AuditLog,
    ToolInvocation,
)
from rto_sentinel.agents.confirmation_writer import ConfirmationWriter
from rto_sentinel.agents.digest_writer import DigestWriter
from rto_sentinel.agents.grounding import (
    ACCUSATORY_TERMS,
    FABRICATED_DRIVERS,
    GroundingVerdict,
    validate_evidence_references,
    validate_feature_grounding,
    validate_figure_grounding,
    validate_neutral_framing,
)
from rto_sentinel.agents.investigator import (
    InvestigationError,
    RiskInvestigation,
    RiskInvestigationAgent,
)
from rto_sentinel.agents.provider import (
    API_KEY_VARIABLE,
    ENABLE_VARIABLE,
    AgentUnavailableError,
    AnthropicProvider,
    Completion,
    LLMProvider,
    ToolCall,
    UnavailableProvider,
    get_provider,
)
from rto_sentinel.agents.reason_code_writer import write_explanation
from rto_sentinel.agents.tools import (
    TOOL_SPECS,
    TOOLS_BY_NAME,
    AgentToolset,
    CustomerHistoryRef,
    DigestRef,
    OrderRef,
    ToolSpec,
    anthropic_tool_definitions,
)

__all__ = [
    "ACCUSATORY_TERMS",
    "API_KEY_VARIABLE",
    "DEFERRAL_REASON",
    "ENABLE_VARIABLE",
    "FABRICATED_DRIVERS",
    "TOOLS_BY_NAME",
    "TOOL_SPECS",
    "AddressRepairDeferred",
    "AgentAuditRecord",
    "AgentToolset",
    "AgentUnavailableError",
    "AnthropicProvider",
    "AuditBuilder",
    "AuditLog",
    "Completion",
    "ConfirmationWriter",
    "CustomerHistoryRef",
    "DigestRef",
    "DigestWriter",
    "GroundingVerdict",
    "InvestigationError",
    "LLMProvider",
    "OrderRef",
    "RiskInvestigation",
    "RiskInvestigationAgent",
    "ToolCall",
    "ToolInvocation",
    "ToolSpec",
    "UnavailableProvider",
    "anthropic_tool_definitions",
    "get_provider",
    "suggest_address_repair",
    "validate_evidence_references",
    "validate_feature_grounding",
    "validate_figure_grounding",
    "validate_neutral_framing",
    "write_explanation",
]
