/**
 * The investigation agent, and what happens when it is not available.
 *
 * THE RULE THIS COMPONENT EXISTS TO HOLD
 * ======================================
 * If the backend fails, this shows the failure. There is no canned sentence, no
 * "explanation temporarily unavailable, here is a summary anyway", and no
 * client-side text that could be mistaken for model output. When the language
 * layer is off the panel says so, quotes the backend's reason, and points at the
 * reason codes on the risk screen - which are the artefact of record and do not
 * depend on any language model.
 *
 * WHAT IS DISPLAYED FROM WHERE
 * ============================
 * `probability`, `band`, `threshold` and `model_version` in the response are
 * copied by the backend from tool results, not parsed out of the model's prose.
 * This component renders those fields. It never scrapes a number out of
 * `summary`, which is what makes it structurally impossible for a language model
 * to change what this screen reports about a decision.
 */

import { useState } from 'react';

import { fetchAgentStatus, investigateOrder } from '@/api/endpoints';
import {
  ErrorState,
  LoadingState,
  Panel,
  RiskBadge,
} from '@/components/primitives';
import { useAction, useApi } from '@/hooks/useApi';
import type { AgentAuditRecord, RiskInvestigation } from '@/types/api';

const DEFAULT_QUESTION = 'Why did this order receive its risk level?';

export default function AgentPanel({ orderId }: { orderId: string }): JSX.Element {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const status = useApi((signal) => fetchAgentStatus(signal), []);
  const investigation = useAction(
    (id: string, prompt: string, signal: AbortSignal) =>
      investigateOrder(id, prompt, undefined, signal),
  );

  const available = status.status === 'success' && status.data.available;

  return (
    <Panel
      title="Ask the investigation agent"
      subtitle="The agent retrieves its own evidence through read-only tools. It cannot score, re-band or override anything."
    >
      {status.status === 'loading' ? <LoadingState label="Checking the language layer…" /> : null}
      {status.status === 'error' ? (
        <ErrorState error={status.error} onRetry={status.reload} context="checking agent status" />
      ) : null}

      {status.status === 'success' && !status.data.available ? (
        <div className="notice notice--warning" role="status">
          <p>
            <strong>The language layer is not configured.</strong> {status.data.reason}
          </p>
          <p>
            Set <code>{status.data.required_environment_variable}</code> and{' '}
            <code>{status.data.enable_switch}</code> to enable it. {status.data.note}
          </p>
          <p>
            The reason codes and feature contributions above are unaffected — they come from the
            model, not from a language model, and they are the record.
          </p>
        </div>
      ) : null}

      <form
        className="filters"
        onSubmit={(event) => {
          event.preventDefault();
          void investigation.run(orderId, question);
        }}
      >
        <div className="field field--grow">
          <label htmlFor="agent-question">Question</label>
          <input
            id="agent-question"
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={!available}
          />
        </div>
        <button
          type="submit"
          className="button"
          disabled={!available || investigation.state.status === 'loading'}
        >
          {investigation.state.status === 'loading' ? 'Investigating…' : 'Ask'}
        </button>
      </form>

      {investigation.state.status === 'loading' ? (
        <LoadingState label="The agent is retrieving evidence and composing an answer…" />
      ) : null}

      {investigation.state.status === 'error' ? (
        <>
          <ErrorState error={investigation.state.error} context="running the investigation" />
          <p className="state__detail">
            No explanation is shown because none was produced. The reason codes above remain
            available.
          </p>
        </>
      ) : null}

      {investigation.state.status === 'success' ? (
        <Answer
          investigation={investigation.state.data.investigation}
          audit={investigation.state.data.audit}
          question={question}
        />
      ) : null}
    </Panel>
  );
}

function Answer({
  investigation,
  audit,
  question,
}: {
  investigation: RiskInvestigation;
  audit: AgentAuditRecord;
  question: string;
}): JSX.Element {
  return (
    <div className="agent-answer">
      <blockquote className="agent-question">{question}</blockquote>

      {!investigation.grounded ? (
        <div className="notice notice--warning" role="alert">
          <p>
            <strong>This answer was rejected by the grounding validator.</strong>{' '}
            {investigation.rejection_reason}
          </p>
          <p>The prose is withheld. The reason codes below came from the model, not the agent.</p>
        </div>
      ) : null}

      {!investigation.sufficient_evidence ? (
        <div className="notice" role="status">
          <p>
            <strong>The agent reported insufficient evidence.</strong> That is an answer, not a
            failure.
          </p>
        </div>
      ) : null}

      {investigation.grounded ? <p className="agent-summary">{investigation.summary}</p> : null}

      {investigation.uncertainty ? (
        <p className="agent-uncertainty">
          <strong>Could not establish:</strong> {investigation.uncertainty}
        </p>
      ) : null}

      <div className="agent-facts">
        <h4 className="subheading">Retrieved facts — not the model's prose</h4>
        <p className="state__detail">
          These are copied from the tool results by the backend. A model writing a different number
          in its summary cannot change them.
        </p>
        <ul className="fact-list">
          {investigation.band ? (
            <li>
              <RiskBadge
                band={investigation.band}
                probability={investigation.probability}
                threshold={investigation.threshold}
              />
            </li>
          ) : null}
          {investigation.model_version ? (
            <li>
              Model <code>{investigation.model_version}</code>
            </li>
          ) : null}
          {investigation.reason_codes.map((code) => (
            <li key={code}>
              <code>{code}</code>
            </li>
          ))}
        </ul>
      </div>

      {investigation.key_drivers.length > 0 ? (
        <div>
          <h4 className="subheading">Drivers the agent named</h4>
          <ul className="code-list">
            {investigation.key_drivers.map((driver) => (
              <li key={driver}>
                <code>{driver}</code>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {investigation.caveats.length > 0 ? (
        <ul className="caveats">
          {investigation.caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      ) : null}

      <details className="details">
        <summary>
          Evidence trail — {audit.tools_invoked.length} tool call
          {audit.tools_invoked.length === 1 ? '' : 's'}, {audit.llm_turns} model turn
          {audit.llm_turns === 1 ? '' : 's'}
        </summary>
        <table className="table">
          <caption className="visually-hidden">Tools the agent invoked</caption>
          <thead>
            <tr>
              <th scope="col">Tool</th>
              <th scope="col">Found</th>
              <th scope="col">Duration</th>
              <th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            {audit.tools_invoked.map((call, index) => (
              <tr key={`${call.tool}-${index}`}>
                <th scope="row">
                  <code>{call.tool}</code>
                </th>
                <td>{call.found ? 'yes' : 'no'}</td>
                <td className="numeric">{Math.round(call.duration_ms)} ms</td>
                <td>{call.error ?? call.reason ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="state__detail">
          Model <code>{audit.model}</code> via {audit.provider}, {Math.round(audit.duration_ms)} ms
          total
          {audit.input_tokens !== null && audit.output_tokens !== null
            ? `, ${audit.input_tokens} in / ${audit.output_tokens} out tokens`
            : ''}
          . Generated {investigation.generated_at.slice(0, 16).replace('T', ' ')}.
        </p>
      </details>
    </div>
  );
}
