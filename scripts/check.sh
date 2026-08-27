#!/usr/bin/env bash
# Everything CI runs, in one command. Run this before pushing.
#
#   ./scripts/check.sh
#
# Fails on the first problem so the output stays readable.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"

echo "==> ruff (lint)"
"$PY" -m ruff check .

echo "==> ruff (format)"
"$PY" -m ruff format --check .

echo "==> mypy (strict)"
"$PY" -m mypy

echo "==> configuration validation"
"$PY" -m rto_sentinel.cli config check

echo "==> pytest"
"$PY" -m pytest -q

if [ -d console/node_modules ]; then
  echo "==> console: typecheck"
  (cd console && npm run typecheck)
  echo "==> console: lint"
  (cd console && npm run lint)
  echo "==> console: test"
  (cd console && npm run test)
  echo "==> console: build"
  (cd console && npm run build)
else
  echo "!! console/node_modules missing - skipping frontend checks."
  echo "   Run: cd console && npm install"
fi

echo
echo "All checks passed."
