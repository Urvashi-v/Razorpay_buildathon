"""No credential ever gets committed to this repository.

Two checks:

* **No key-shaped literal anywhere in tracked source or config.** A regex sweep
  for the recognisable prefixes of common credential formats.
* **``.env.example`` documents variables without supplying values.** The example
  file must name every variable the system reads, and must not carry a real
  value for any secret. A placeholder that looks like a working key is worse than
  no placeholder at all: someone copies it and wonders why authentication fails,
  or worse, it *is* real.

These are cheap and they catch the mistake that matters - a key pasted in during
debugging and forgotten.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

#: Recognisable credential prefixes. Not exhaustive - a heuristic that catches
#: the accident, not an adversary.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "Private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

SCANNED_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ts",
    ".tsx",
    ".js",
    ".md",
    ".example",
}
SKIPPED_DIRS = {
    ".venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
}


def _tracked_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        if path.suffix in SCANNED_SUFFIXES or path.name == ".env.example":
            files.append(path)
    return files


def test_no_credential_literals_in_repository(repo_root: Path) -> None:
    """No file in the repository contains a key-shaped literal."""
    findings: list[str] = []
    for path in _tracked_files(repo_root):
        # This test file necessarily contains the patterns themselves.
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - binary file
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label} in {path.relative_to(repo_root)}")
    assert not findings, f"Credential-shaped literals found: {findings}"


def test_env_example_exists_and_has_no_secret_values(repo_root: Path) -> None:
    """``.env.example`` names the secret variables but supplies none of them."""
    example = repo_root / ".env.example"
    assert example.is_file(), ".env.example must exist so setup does not require guesswork"

    secret_vars = {"ANTHROPIC_API_KEY"}
    values: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()

    for name in secret_vars:
        assert name in values, f".env.example must document {name}"
        assert values[name] == "", (
            f".env.example must leave {name} empty. A placeholder that looks like a key "
            "gets copied into a real .env and fails confusingly, or turns out to be real."
        )


def test_env_file_is_gitignored(repo_root: Path) -> None:
    """A real ``.env`` can never be committed by accident."""
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "\n.env\n" in f"\n{gitignore}", ".gitignore must ignore .env"
    assert "!.env.example" in gitignore, ".gitignore must still allow .env.example"
