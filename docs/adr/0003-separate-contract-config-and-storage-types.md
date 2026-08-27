# ADR 0003 — Three type hierarchies, not one

**Status:** Accepted
**Date:** 2026-08-27

## Context

The project needs typed models for three things that look similar: the API wire
format, the YAML configuration, and the database schema. SQLModel exists precisely
to unify the first and third, and a single set of Pydantic models could plausibly
serve all three.

## Decision

Three separate hierarchies:

| Package | Purpose | Library |
|---|---|---|
| `rto_sentinel.contracts` | API wire format | Pydantic |
| `rto_sentinel.configuration.schemas` | YAML shape | Pydantic, frozen |
| `rto_sentinel.db.models` | Storage schema | SQLAlchemy 2.0 declarative |

## Rationale

They change for different reasons and on different schedules:

- A wire contract changes when the frontend needs a different shape. That should
  not force a database migration.
- A storage schema gains columns for auditing and provenance —
  `config_fingerprint`, `engine_version`, `created_at`. Those should not appear in
  a public API response.
- Configuration changes when someone tunes a parameter. It should never touch
  either of the others.

Unifying them would make every change in any of those three categories a change in
all three. The coupling is not hypothetical: `decisions` carries
`config_fingerprint` and `engine_version` for replay, which no API consumer needs,
and `OrderPayload` carries a nested `items` list that the flat `orders` table
stores as a count.

There is also a privacy dimension. `OrderPayload` and the `orders` table both
deliberately lack a customer-name column. Keeping them separate means that
commitment is asserted twice, in two independent tests, rather than once.

## Consequences

- Mapping code is required between layers. That is the cost, and it is small and
  explicit: repositories do the mapping in one place.
- `tests/db/test_schema.py` and `tests/unit/test_contracts.py` each independently
  assert the privacy commitments.
- SQLAlchemy 2.0 declarative with `Mapped[...]` gives full mypy-strict coverage of
  the storage layer without SQLModel's runtime coupling.
