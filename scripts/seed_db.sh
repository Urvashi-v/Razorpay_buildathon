#!/usr/bin/env bash
# Create the schema, generate a dataset, validate it, and load it into PostgreSQL.
#
#   ./scripts/seed_db.sh                       # defaults from config/generator.yaml
#   ./scripts/seed_db.sh --orders 12000 --customers 4000 --lenient
#
# Requires a running database. Start one with:  docker compose up -d db
#
# Every argument is forwarded to `rto-sentinel seed-db`, which runs:
#   1. alembic migrations to head
#   2. generation (seed, version, params recorded on the dataset run)
#   3. validation - and REFUSES to load if it fails
#   4. bulk load into PostgreSQL
#   5. verification, by querying the row counts back out
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"

if [ ! -f .env ]; then
  echo "!! .env not found. Copy .env.example to .env and set POSTGRES_PASSWORD." >&2
  exit 1
fi

exec "$PY" -m rto_sentinel.cli seed-db "$@"
