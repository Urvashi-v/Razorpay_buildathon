"""What every agent run records, so a generated sentence can be accounted for.

WHY AN AUDIT TRAIL FOR PROSE
============================
The decision log already records what happened to an order and why, in reason
codes that can be counted and filtered. This records something different: which
agent ran, what it was asked, which tools it actually called, what came back, and
what it produced. Not because the prose is authoritative - it is not - but
because when an ops associate acts on an explanation and the explanation was
wrong, "which evidence did it actually look at" is the first question and the
only one a transcript can answer.

WHAT IS DELIBERATELY NOT IN HERE
================================
**No secrets.** The API key never reaches this module; the provider reads it once
and holds it. The recorded ``provider`` and ``model`` are identifiers, not
credentials.

**No raw prompt text by default.** The system prompt is a template and the user
turn is a question about an order; both are reconstructible from
``agent_type`` and ``subject_id``. Storing generated prose *is* useful - it is
the artefact someone acted on - so the final output is kept.

**No tool outputs.** Only tool names, their arguments and whether evidence was
found. A full transcript of every tool result would duplicate the database into
a log file, and the database is already the record.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger("rto_sentinel.agents.audit")


class ToolInvocation(BaseModel):
    """One tool call an agent made, and whether it found anything."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    found: bool
    reason: str | None = Field(default=None, description="Why nothing was found, when nothing was")
    duration_ms: float = Field(ge=0.0)
    error: str | None = Field(
        default=None, description="Exception type and message, never a stack trace"
    )


class AgentAuditRecord(BaseModel):
    """The complete account of one agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_type: str = Field(max_length=64)
    request: str = Field(max_length=2000, description="What the agent was asked")
    subject_id: str | None = Field(
        default=None, max_length=64, description="Order, merchant or other entity in question"
    )
    dataset_run_id: str | None = Field(default=None, max_length=64)

    provider: str = Field(max_length=32, description="Identifier, never a credential")
    model: str = Field(max_length=64)

    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0.0)

    tools_invoked: list[ToolInvocation] = Field(default_factory=list)
    llm_turns: int = Field(default=0, ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None

    grounded: bool | None = Field(default=None, description="Null when no generation was produced")
    rejection_reason: str | None = None
    output: dict[str, Any] | None = Field(
        default=None, description="The final structured output, when there was one"
    )
    error: str | None = Field(
        default=None, description="Why the run failed. Present when output is null."
    )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(call.tool for call in self.tools_invoked)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.output is not None

    def summary(self) -> str:
        """One line, for a log or a console."""
        state = "ok" if self.succeeded else "failed"
        if self.succeeded and self.grounded is False:
            state = "rejected"
        return (
            f"{self.agent_type} {state} subject={self.subject_id} "
            f"tools={list(self.tool_names)} turns={self.llm_turns} "
            f"{self.duration_ms:.0f}ms"
        )


class AuditLog:
    """Where agent runs are recorded.

    Writes structured JSON to a logger rather than to the database. Agent runs
    are diagnostic, high-volume and not part of the decision record - putting
    them in the same store as decisions would blur the line between "what the
    system did" and "what it said about it", and the first of those is the one
    that must stay clean.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or LOGGER
        self._records: list[AgentAuditRecord] = []

    def record(self, entry: AgentAuditRecord) -> AgentAuditRecord:
        self._records.append(entry)
        self._logger.info(
            entry.summary(),
            extra={"agent_audit": json.loads(entry.model_dump_json())},
        )
        return entry

    @property
    def records(self) -> tuple[AgentAuditRecord, ...]:
        """Runs recorded in this process. Diagnostic, not a durable store."""
        return tuple(self._records)

    def last(self) -> AgentAuditRecord | None:
        return self._records[-1] if self._records else None


class AuditBuilder:
    """Accumulates an audit record while a run is in progress."""

    def __init__(
        self, *, agent_type: str, request: str, provider: str, model: str, subject_id: str | None
    ) -> None:
        self.agent_type = agent_type
        self.request = request
        self.provider = provider
        self.model = model
        self.subject_id = subject_id
        self.dataset_run_id: str | None = None
        self.started_at = datetime.now(UTC)
        self.tools: list[ToolInvocation] = []
        self.llm_turns = 0
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def add_tool(self, invocation: ToolInvocation) -> None:
        self.tools.append(invocation)

    def add_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens

    def finish(
        self,
        *,
        output: dict[str, Any] | None = None,
        grounded: bool | None = None,
        rejection_reason: str | None = None,
        error: str | None = None,
    ) -> AgentAuditRecord:
        finished = datetime.now(UTC)
        return AgentAuditRecord(
            agent_type=self.agent_type,
            request=self.request[:2000],
            subject_id=self.subject_id,
            dataset_run_id=self.dataset_run_id,
            provider=self.provider,
            model=self.model,
            started_at=self.started_at,
            finished_at=finished,
            duration_ms=(finished - self.started_at).total_seconds() * 1000.0,
            tools_invoked=list(self.tools),
            llm_turns=self.llm_turns,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            grounded=grounded,
            rejection_reason=rejection_reason,
            output=output,
            error=error,
        )
