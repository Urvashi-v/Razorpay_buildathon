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


# ---------------------------------------------------------------------------
# Separation of concerns
# ---------------------------------------------------------------------------


def test_no_ml_logic_in_route_handlers(source_root: Path) -> None:
    """Routers marshal and delegate. ML belongs in the pipeline, not in a handler."""
    found = _violations(source_root, "api.routers", forbidden_external=ML_LIBRARIES)
    assert not found, f"ML libraries must not be imported by a route handler: {found}"


def test_ml_layers_do_not_import_the_web_or_database_layer(source_root: Path) -> None:
    """The pipeline must be runnable without a server or a database.

    Training reads files and writes artefacts. If the ML layer needed the API or
    the database, a model could not be retrained offline - and the pipeline would
    stop being reproducible from config plus seed alone.
    """
    for layer in ("data", "features", "models", "eval"):
        found = _violations(
            source_root,
            layer,
            forbidden_internal=frozenset({"api", "db"}),
            forbidden_external=WEB_LIBRARIES | DB_LIBRARIES,
        )
        assert not found, f"Layer {layer!r} must not depend on the web or database layer: {found}"


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
