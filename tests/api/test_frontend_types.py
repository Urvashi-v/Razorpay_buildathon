"""The console's hand-written TypeScript interfaces match the API's response models.

WHY THIS EXISTS
===============
`test_frontend_contract.py` proves every path the console requests exists and is
served over the right method. It explicitly does not check response *shapes*, and
that gap was recorded as unresolved in the Phase 11 report.

This closes it. `console/src/types/api.ts` is written by hand, and `tsc`
type-checks the console against those interfaces rather than against the server -
so a backend field renamed from `net_inr_per_1000_orders` to `net_per_1000` type-
checks clean on both sides and renders an em-dash in production. The failure is
silent, and the console's own tests cannot catch it because they stub `fetch` and
return whatever the stub was told to return.

HOW IT WORKS
============
The OpenAPI schema is the backend's own description of what it returns. The
TypeScript interfaces are parsed with a small regex reader - not a full TS parser,
because the file is hand-written in a consistent style and a real parser would be
a dependency and a maintenance burden out of proportion to the check.

For each interface that maps to a schema, every field the console declares must
exist in the schema. The reverse is deliberately NOT asserted: a console that
ignores fields it does not need is fine, and requiring it to declare all of them
would make every backend addition a frontend chore.

WHAT THIS STILL DOES NOT CHECK
==============================
Types. `probability: number` against a schema saying `string` would pass here.
Field *presence* is where the silent failures live - a renamed or removed field
renders as an em-dash - and a type mismatch surfaces loudly at the first render.
Recorded rather than implied away.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from rto_sentinel.api.main import create_app
from rto_sentinel.settings import REPO_ROOT

TYPES_FILE = REPO_ROOT / "console" / "src" / "types" / "api.ts"

#: Console interface -> the OpenAPI component schema it mirrors.
#:
#: Written out rather than inferred by name. Several console interfaces are
#: deliberate subsets or renames (`FairnessResponse` mirrors a response model
#: defined inside a router, `CohortResult` mirrors a contract type), and guessing
#: the mapping would silently skip the ones that did not match.
MAPPING: dict[str, str] = {
    "OrderSummary": "OrderSummary",
    "RiskAssessmentResponse": "RiskAssessmentResponse",
    "FeatureContribution": "FeatureContribution",
    "CostInputs": "CostInputs",
    "SimulationResult": "SimulationResult",
    "ThresholdDerivation": "ThresholdDerivation",
    "DataStatusResponse": "DataStatusResponse",
    "ModelStatusResponse": "ModelStatusResponse",
    "FinalModelResponse": "FinalModelResponse",
    "LadderResponse": "LadderResponse",
    "AgentStatusResponse": "AgentStatusResponse",
    "CohortResult": "CohortResult",
    "FairnessAudit": "FairnessAudit",
    "FairnessResponse": "FairnessResponse",
    "ShiftResult": "ShiftResult",
    "ShiftStudy": "ShiftStudy",
    "DriftSignal": "DriftSignal",
    "DriftReport": "DriftReport",
    "PerformanceDelta": "PerformanceDelta",
    "WindowSummary": "WindowSummary",
    "EnvironmentSpec": "EnvironmentSpec",
    # `AblationArm` / `AblationStudy` are served as a raw dict from the saved
    # artefact rather than through a response model, so they have no OpenAPI
    # schema to check against. Covered instead by `tests/unit/test_ablation.py`,
    # which asserts the fields the artefact writer emits.
}

_INTERFACE = re.compile(r"export interface (\w+)\s*\{(.*?)\n\}", re.DOTALL)

#: A property line: optional leading comment lines are skipped by the caller.
#: Captures the name, ignoring an optional `?` and everything after the colon.
_FIELD = re.compile(r"^\s{2}(\w+)\??\s*:")


def console_interfaces() -> dict[str, set[str]]:
    """Interface name -> the field names it declares."""
    source = TYPES_FILE.read_text(encoding="utf-8")
    found: dict[str, set[str]] = {}
    for name, body in _INTERFACE.findall(source):
        fields = set()
        for line in body.splitlines():
            # Skip comment bodies, which can contain colons.
            if line.lstrip().startswith(("*", "/", "//")):
                continue
            match = _FIELD.match(line)
            if match:
                fields.add(match.group(1))
        found[name] = fields
    return found


@pytest.fixture(scope="module")
def schemas() -> dict[str, set[str]]:
    """Schema name -> its property names, from the API's own OpenAPI document."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        document = client.get("/openapi.json").json()
    return {
        name: set((schema.get("properties") or {}).keys())
        for name, schema in document["components"]["schemas"].items()
    }


def test_the_types_file_is_where_it_is_expected_to_be() -> None:
    """A guard on the guard: a moved file must not pass vacuously."""
    assert TYPES_FILE.is_file(), f"console types not found at {TYPES_FILE}"
    interfaces = console_interfaces()
    assert len(interfaces) > 15, f"only parsed {len(interfaces)} interfaces - reader broken?"


def test_every_mapped_interface_has_a_matching_schema(schemas: dict[str, set[str]]) -> None:
    """The mapping itself must stay true as the API evolves.

    A schema renamed on the backend would otherwise make its entry unresolvable,
    and the field check below would skip it - passing while checking nothing.
    """
    missing = sorted(schema for schema in MAPPING.values() if schema not in schemas)
    assert not missing, (
        "these OpenAPI schemas no longer exist, so the interfaces mapped to them are "
        f"unchecked: {missing}"
    )


def test_every_mapped_interface_is_parsed_from_the_console(
    schemas: dict[str, set[str]],
) -> None:
    interfaces = console_interfaces()
    missing = sorted(name for name in MAPPING if name not in interfaces)
    assert not missing, f"these console interfaces were not found in api.ts: {missing}"


@pytest.mark.parametrize(("interface", "schema_name"), sorted(MAPPING.items()))
def test_console_fields_exist_on_the_backend_model(
    schemas: dict[str, set[str]], interface: str, schema_name: str
) -> None:
    """No console interface may declare a field the API does not return.

    This is the failure that ships green: `tsc` checks the console against these
    interfaces, the console's tests stub `fetch`, and the first person to see the
    em-dash is a merchant.
    """
    declared = console_interfaces()[interface]
    served = schemas[schema_name]
    phantom = sorted(declared - served)

    assert not phantom, (
        f"`{interface}` in console/src/types/api.ts declares field(s) the API's "
        f"`{schema_name}` does not return: {phantom}. Either the backend renamed "
        "them and the console was not updated, or the console invented them - and "
        "either way they will render as em-dashes."
    )
