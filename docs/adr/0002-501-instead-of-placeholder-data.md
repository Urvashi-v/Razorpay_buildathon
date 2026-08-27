# ADR 0002 — Unimplemented endpoints return 501, never placeholder data

**Status:** Accepted
**Date:** 2026-08-27

## Context

Phase 1 publishes the full API contract for 17 endpoints so the console can be
built against a real schema. Fifteen of them have no implementation yet.

There are two ways to handle that: return representative sample data so the UI can
be developed and demonstrated, or return an explicit error.

## Decision

Every unimplemented endpoint returns `501` with
`{"error": {"code": "NOT_IMPLEMENTED", "detail": {"phase": "Phase N"}}}`. No
endpoint returns synthesised sample data. Unimplemented Python functions raise
`NotImplementedError` naming the phase; they do not return defaults.

## Rationale

A fabricated number does not stay in the API. It flows into a chart, the chart
into a screenshot, the screenshot into a slide, and the slide into a claim. By
then nobody remembers which numbers were real.

This project's entire argument is that its numbers are checkable. A demo
containing one invented figure undermines that argument more than an obviously
incomplete demo does.

There is a practical benefit too: a frontend developer wiring against a 501
discovers the gap in seconds. One wiring against plausible sample data discovers
it much later, after building layout and logic around a shape that may not survive
contact with the real response.

## Consequences

- The Phase 1 console shows a status panel and a phase list rather than a
  dashboard. This is the honest state of the project.
- `tests/api/test_contract_surface.py::test_scoring_endpoint_does_not_fabricate_a_score`
  asserts that a 501 response carries no `probability` field.
- The distinction between `NOT_IMPLEMENTED` (not built), `MODEL_UNAVAILABLE`
  (built, but no artefact loaded) and `AGENT_UNAVAILABLE` (optional layer off) is
  preserved in the error code, so the console can respond differently to each.
- `/readiness` reports `model: not ready` on a fresh checkout. That is correct
  behaviour, not a bug to be papered over.
