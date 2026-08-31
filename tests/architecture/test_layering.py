"""The layering rules, enforced mechanically.

ARCHITECTURE.md states which layer may depend on which. A document nobody runs
decays within a fortnight, so these tests parse the import graph and assert it.

The rule that matters most is the LLM boundary. ``rto_sentinel.decision`` is the
authority on what happens to an order, and it must remain a pure function of a
calibrated probability and a set of merchant cost inputs. If a future change
imports the agent layer into the decision layer - even under ``TYPE_CHECKING``,
even "temporarily" - this test fails and says why.

Imports are collected from the AST, including those inside ``if TYPE_CHECKING``
blocks. A type-only import of an LLM SDK into the decision engine would not
execute, but it would mean someone was reaching for it, and the point of a
boundary is to notice that.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

#: Third-party libraries that indicate a layer is doing something it should not.
ML_LIBRARIES = frozenset({"lightgbm", "shap", "sklearn", "scipy", "numpy", "pandas"})
LLM_LIBRARIES = frozenset({"anthropic", "openai", "langchain", "llama_index", "transformers"})
WEB_LIBRARIES = frozenset({"fastapi", "starlette", "uvicorn"})
DB_LIBRARIES = frozenset({"sqlalchemy", "alembic", "psycopg"})


def _iter_modules(source_root: Path) -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted(source_root.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_names(tree: ast.Module) -> set[str]:
    """Every module name imported anywhere in the file, TYPE_CHECKING included."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _top_level(name: str) -> str:
    return name.split(".", 1)[0]


def _layer_of(path: Path, source_root: Path) -> str:
    """The layer a file belongs to, e.g. ``decision`` or ``api.routers``."""
    relative = path.relative_to(source_root)
    if len(relative.parts) == 1:
        return relative.stem
    if relative.parts[0] == "api" and len(relative.parts) > 2:
        return "api.routers"
    return relative.parts[0]


def _violations(
    source_root: Path,
    layer: str,
    *,
    forbidden_internal: frozenset[str] = frozenset(),
    forbidden_external: frozenset[str] = frozenset(),
) -> list[str]:
    found: list[str] = []
    for path, tree in _iter_modules(source_root):
        if _layer_of(path, source_root) != layer:
            continue
        for imported in _imported_names(tree):
            if imported.startswith("rto_sentinel."):
                sub = imported.removeprefix("rto_sentinel.").split(".", 1)[0]
                if sub in forbidden_internal:
                    found.append(f"{path.name} imports rto_sentinel.{sub}")
            elif _top_level(imported) in forbidden_external:
                found.append(f"{path.name} imports {imported}")
    return found


# ---------------------------------------------------------------------------
# The critical architectural rule
# ---------------------------------------------------------------------------


def test_decision_layer_never_imports_an_llm(source_root: Path) -> None:
    """The LLM may not reach the decision engine, by any route.

    SPEC section 08: the LLM is downstream of the decision, never inside it.
    Anything else makes the risk engine non-deterministic, and a
    non-deterministic risk engine cannot be audited.
    """
    found = _violations(
        source_root,
        "decision",
        forbidden_internal=frozenset({"agents"}),
        forbidden_external=LLM_LIBRARIES,
    )
    assert not found, (
        "The decision engine must never depend on an LLM. It converts a calibrated "
        "probability into an action deterministically. " + str(found)
    )


def test_decision_layer_is_free_of_io(source_root: Path) -> None:
    """No database, no HTTP framework, no network inside the decision engine.

    Sub-100ms with no external calls, and the same inputs always produce the
    same decision - on any machine, at any time.
    """
    found = _violations(
        source_root,
        "decision",
        forbidden_internal=frozenset({"api", "db"}),
        forbidden_external=(
            DB_LIBRARIES | WEB_LIBRARIES | frozenset({"requests", "httpx", "socket"})
        ),
    )
    assert not found, f"The decision engine must stay pure and I/O-free: {found}"


def test_agents_cannot_reach_the_decision_engine_or_models(source_root: Path) -> None:
    """The language layer describes decisions; it cannot make or influence them.

    The agent layer may import ``contracts`` - it has to, to accept a decision as
    input - but it may not import ``decision`` or ``models``, which is where a
    probability, a threshold or an action could actually be produced.
    """
    found = _violations(
        source_root,
        "agents",
        forbidden_internal=frozenset({"decision", "models", "features", "data", "eval"}),
        forbidden_external=ML_LIBRARIES,
    )
    assert not found, (
        "The agent layer must not be able to generate a score, choose a threshold, "
        f"or modify a prediction: {found}"
    )


#: The one module in `serving` that the agent layer may reach for.
#:
#: `serving` is not on the forbidden-imports list because the agent layer has to
#: get its evidence from somewhere, and `serving.agent_tools` is that somewhere -
#: a registry of six read-only `get_` tools with an `invoke` that dispatches by
#: name and refuses anything not registered.
#:
#: But "not forbidden" is not the same as "bounded". Nothing stopped a later
#: edit adding `from rto_sentinel.serving.scoring import score` to an agent
#: module, which would hand the language layer the very capability every other
#: rule here exists to withhold. This narrows the door to the one module.
AGENT_SERVING_ENTRYPOINT = "agent_tools"


def test_agents_reach_serving_only_through_the_tool_registry(source_root: Path) -> None:
    """The agent layer may import `serving.agent_tools`, and nothing else in serving.

    `serving` also contains `scoring`, `assessment`, `features` and
    `model_registry` - a probability, a decision, a feature matrix and a loaded
    booster respectively. Any one of those imported into `agents` would defeat
    the separation the other tests assert, without tripping any of them.
    """
    offenders: list[str] = []
    for path in sorted((source_root / "agents").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if "rto_sentinel.serving" not in stripped or stripped.startswith("#"):
                continue
            if f"rto_sentinel.serving.{AGENT_SERVING_ENTRYPOINT}" in stripped:
                continue
            offenders.append(f"{path.name}:{number}: {stripped}")

    assert not offenders, (
        "The agent layer may only reach serving through "
        f"`serving.{AGENT_SERVING_ENTRYPOINT}`, the read-only tool registry. "
        f"Found: {offenders}"
    )


# ---------------------------------------------------------------------------
# Separation of concerns
# ---------------------------------------------------------------------------


def test_the_agent_layer_holds_no_write_capability(source_root: Path) -> None:
    """Agents describe decisions. They cannot reach anything that changes one.

    The import ban in `test_agents_cannot_reach_the_decision_engine_or_models` is
    the first half. This is the second: even the names the package mentions must
    not include a write path, because a toolset is only as read-only as the
    methods it can reach for.

    `.append(` is deliberately NOT on this list. Python lists have that method
    and the agent package builds lists everywhere, so matching it produces four
    false positives and no true one - a check that has to be ignored to pass is
    worse than no check.
    """
    banned = (
        "DecisionLogRepository",
        "OpsOverrideRepository",
        "DatasetRepository",
        "ServingRepository",
        "session_scope",
        ".commit(",
        ".flush(",
        "session.add(",
    )
    offenders: list[str] = []
    for path in (source_root / "agents").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {name}" for name in banned if name in source]

    assert not offenders, (
        "the agent package must not name a write-capable repository or a session "
        f"mutation: {offenders}"
    )


def test_no_ml_logic_in_route_handlers(source_root: Path) -> None:
    """Routers marshal and delegate. ML belongs in the pipeline, not in a handler."""
    found = _violations(source_root, "api.routers", forbidden_external=ML_LIBRARIES)
    assert not found, f"ML libraries must not be imported by a route handler: {found}"


def test_ml_layers_do_not_import_the_web_or_database_layer(source_root: Path) -> None:
    """The pipeline must be runnable without a server or a database.

    Training reads files and writes artefacts. If the ML layer needed the API or
    the database, a model could not be retrained offline - and the pipeline would
    stop being reproducible from config plus seed alone.

    `monitoring` is held to the same rule. Drift is computed over two frames; it
    must not care whether they came from a parquet file, a database or a live
    stream, because the moment it does, the arithmetic can only be tested against
    a running service.
    """
    for layer in ("data", "features", "models", "eval", "monitoring"):
        found = _violations(
            source_root,
            layer,
            forbidden_internal=frozenset({"api", "db"}),
            forbidden_external=WEB_LIBRARIES | DB_LIBRARIES,
        )
        assert not found, f"Layer {layer!r} must not depend on the web or database layer: {found}"


#: Modules in `agents` that are pure contract - schemas and record types with no
#: behaviour, no I/O and no provider. `serving` implements the tool interface, so
#: it must import these; importing them gives a language model no path to
#: anything, because there is nothing behind them to reach.
AGENT_CONTRACT_MODULES = frozenset({"tools", "audit"})


def test_the_agent_contract_modules_really_are_contract_only(source_root: Path) -> None:
    """The exemption below is only safe if these modules hold no behaviour.

    Checked rather than asserted: if someone puts a provider call or a database
    query into `agents/tools.py`, the exemption that lets `serving` import it
    stops being harmless and this test fails first.
    """
    offenders: list[str] = []
    for name in sorted(AGENT_CONTRACT_MODULES):
        path = source_root / "agents" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _imported_names(tree):
            top = _top_level(imported)
            if top in LLM_LIBRARIES | DB_LIBRARIES | WEB_LIBRARIES | ML_LIBRARIES:
                offenders.append(f"{name}.py imports {imported}")
            if imported.startswith("rto_sentinel."):
                sub = imported.removeprefix("rto_sentinel.").split(".", 1)[0]
                if sub not in {"contracts", "agents"}:
                    offenders.append(f"{name}.py imports rto_sentinel.{sub}")

    assert not offenders, (
        "agents.tools and agents.audit must stay pure contract, because `serving` is "
        f"allowed to import them: {offenders}"
    )


def test_the_serving_layer_composes_but_is_not_composed(source_root: Path) -> None:
    """The composition layer may reach down; nothing above it may reach in.

    ``serving`` exists to put the database, the feature pipeline, the model
    artefact and the decision engine together for a live request - which is why
    it is allowed to import all four when no other layer is. What it may not
    import is ``api`` (composition must not depend on its caller, or the
    pipeline becomes untestable without a server).

    THE ONE EXCEPTION, AND WHY IT IS NARROW
    ---------------------------------------
    ``serving.agent_tools`` implements the tool interface the agent layer
    declares, so it imports ``agents.tools`` and ``agents.audit``. That is the
    implementation depending on its interface, which is the right direction and
    the only one available: the agent package is forbidden from importing the
    decision engine, so it cannot build its own toolset.

    What ``serving`` may still not import is an agent *job* - the investigator,
    the writers, or the provider. Composing an agent is the API's business.
    Without that half of the rule, "serving may import agents" would quietly
    permit the composition layer to start running language models.
    """
    found = _violations(
        source_root,
        "serving",
        forbidden_internal=frozenset({"api"}),
        forbidden_external=LLM_LIBRARIES | WEB_LIBRARIES,
    )

    forbidden_agent_modules = ("investigator", "confirmation_writer", "digest_writer", "provider")
    for path, tree in _iter_modules(source_root):
        if _layer_of(path, source_root) != "serving":
            continue
        for imported in _imported_names(tree):
            if imported.startswith("rto_sentinel.agents."):
                module = imported.removeprefix("rto_sentinel.agents.").split(".", 1)[0]
                if module in forbidden_agent_modules:
                    found.append(f"{path.name} imports an agent job: {imported}")

    assert not found, (
        "The serving layer composes the pipeline; it must not depend on the web "
        f"layer, or run a language model itself: {found}"
    )


def test_route_handlers_do_not_build_sql(source_root: Path) -> None:
    """Queries live in repositories, so the same question has one answer.

    A handler that assembles a query is a handler that will eventually assemble a
    slightly different one for the same question, and the two will disagree in a
    way nobody notices until the numbers do.
    """
    found = _violations(
        source_root,
        "api.routers",
        forbidden_external=frozenset({"psycopg", "psycopg2", "alembic"}),
    )
    assert not found, f"route handlers must not reach for a database driver: {found}"


def test_contracts_depend_on_nothing_but_contracts(source_root: Path) -> None:
    """Shared types are the base of the graph and must import nothing above them."""
    found = _violations(
        source_root,
        "contracts",
        forbidden_internal=frozenset(
            {"api", "db", "agents", "decision", "models", "features", "data", "eval", "monitoring"}
        ),
        forbidden_external=ML_LIBRARIES | WEB_LIBRARIES | DB_LIBRARIES | LLM_LIBRARIES,
    )
    assert not found, f"contracts must sit at the bottom of the dependency graph: {found}"


def test_only_settings_reads_the_environment(source_root: Path) -> None:
    """One module owns configuration from the environment.

    Scattered ``os.environ`` reads are how a service ends up with behaviour that
    depends on a variable nobody documented, and how a secret ends up in a log
    line. ``settings.py`` reads the environment; everything else is handed values.
    """
    offenders: list[str] = []
    for path, tree in _iter_modules(source_root):
        if path.name == "settings.py":
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr in {"environ", "getenv"}
            ):
                offenders.append(f"{path.relative_to(source_root)}:{node.lineno}")
    assert not offenders, f"Only settings.py may read the environment; found reads in: {offenders}"


def test_every_package_has_an_init(source_root: Path) -> None:
    """No implicit namespace packages - they break editable installs subtly."""
    missing = [
        str(directory.relative_to(source_root))
        for directory in source_root.rglob("*")
        if directory.is_dir()
        and not directory.name.startswith("__")
        and any(directory.glob("*.py"))
        and not (directory / "__init__.py").is_file()
    ]
    assert not missing, f"packages missing __init__.py: {missing}"
