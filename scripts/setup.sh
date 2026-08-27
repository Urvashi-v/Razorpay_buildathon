#!/usr/bin/env bash
# First-time developer setup.
#
#   ./scripts/setup.sh
#
# Creates the virtualenv, installs dependencies, and copies .env.example to .env
# if it does not exist. It never writes a credential.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtualenv"
[ -d .venv ] || python -m venv .venv

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"

echo "==> Installing Python dependencies"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -e ".[dev]"

echo "==> Installing console dependencies"
(cd console && npm install)

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
  echo
  echo "   .env created. Edit it before running against PostgreSQL:"
  echo "     - set POSTGRES_PASSWORD to something of your own"
  echo "     - leave ANTHROPIC_API_KEY empty unless you want the language layer"
else
  echo "==> .env already exists; leaving it alone"
fi

echo
echo "Setup complete. Next:"
echo "  ./scripts/check.sh          # run every check"
echo "  rto-sentinel serve          # start the API"
echo "  cd console && npm run dev   # start the console"
