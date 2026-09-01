#!/usr/bin/env bash
# =============================================================================
# The whole system, from an empty checkout to a running demo. One command.
#
#   ./scripts/bootstrap.sh
#
# WHAT THIS IS FOR
# ----------------
# Every step below can be run by hand and each is documented in the README. This
# script exists so that "does it reproduce?" has a single answer that anyone can
# check, rather than a list of commands that has to be followed correctly.
#
# WHAT IT GUARANTEES
# ------------------
# The dataset, the model and the database all come from ONE dataset run. That is
# not decoration: the model's calibration and every artefact under artifacts/
# describe a specific run id, and a database holding a different run means the
# order ids in the evaluation cannot be looked up in the console. Reproducing the
# run from (seed, parameters) is what keeps them the same.
#
# WHAT IT DOES NOT DO
# -------------------
# It does not start the API or the console - those are foreground processes and
# the script tells you the two commands at the end. It does not configure the
# language layer, which needs a credential this repository does not ship.
#
# Idempotent: re-running regenerates the same dataset run id and retrains to the
# same model version, because the generator and the trainer are both seeded.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"

# --- the canonical run --------------------------------------------------------
#
# These four numbers ARE the reproducibility contract. The generator derives its
# run id from them, so changing any one produces a different dataset run and a
# different model version - which is correct, and is why they are pinned here
# rather than left to whatever the config default happens to be.
SEED=7
ORDERS=60000
# The number of customers REQUESTED, which is what the run id hashes. The
# generator writes fewer (activity weights are clipped, so some drawn customers
# receive no orders); `dataset_run.json` records both, and confusing the two is
# how a "reproduction" quietly produces a different run id.
CUSTOMERS=20000
START_DATE=2025-09-01
END_DATE=2026-02-27

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# --- 1. environment -----------------------------------------------------------
step "1/9  Python environment"
if [ ! -x "$PY" ]; then
  python -m venv .venv
  PY=".venv/Scripts/python.exe"
  [ -x "$PY" ] || PY=".venv/bin/python"
fi
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e ".[dev]"
"$PY" -c "import rto_sentinel; print('   rto_sentinel', rto_sentinel.__version__)"

# --- 2. configuration ---------------------------------------------------------
step "2/9  Configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "   wrote .env from .env.example - set POSTGRES_PASSWORD before continuing"
  echo "   (the compose file refuses to start without it, rather than defaulting to blank)"
  exit 1
fi
"$PY" -m rto_sentinel.cli config check

# --- 3. database --------------------------------------------------------------
step "3/9  PostgreSQL"
docker compose up -d db
# Wait for the healthcheck rather than sleeping a fixed amount: a fixed sleep is
# either too short on a cold start or wasted time on a warm one.
for _ in $(seq 1 60); do
  if docker compose ps db | grep -q healthy; then break; fi
  sleep 2
done
docker compose ps db | grep -q healthy || { echo "   database did not become healthy"; exit 1; }

# --- 4. migrations ------------------------------------------------------------
step "4/9  Migrations"
"$PY" -m rto_sentinel.cli db upgrade

# --- 5. dataset + database load ----------------------------------------------
# `seed-db` generates, validates and loads in one pass. It regenerates rather
# than importing a parquet, which is the point: if the generator has drifted, the
# run id changes and the mismatch is loud instead of silent.
step "5/9  Generate the benchmark dataset and load it (seed $SEED, $ORDERS orders)"
"$PY" -m rto_sentinel.cli seed-db \
  --seed "$SEED" --orders "$ORDERS" --customers "$CUSTOMERS" \
  --start-date "$START_DATE" --end-date "$END_DATE"

# --- 6. features + baselines --------------------------------------------------
step "6/9  Baseline ladder (features, splits, rungs 0-4)"
"$PY" -m rto_sentinel.cli train \
  --seed "$SEED" --orders "$ORDERS" --customers "$CUSTOMERS" \
  --start-date "$START_DATE" --end-date "$END_DATE"

# --- 7. final model + calibration --------------------------------------------
# Reads train and validation only. The sealed test split is not touched here.
step "7/9  Final model: hyperparameter search, calibration, frozen manifest"
"$PY" -m rto_sentinel.cli final \
  --seed "$SEED" --orders "$ORDERS" --customers "$CUSTOMERS" \
  --start-date "$START_DATE" --end-date "$END_DATE"

# --- 8. the single sealed evaluation ------------------------------------------
# Requires a frozen manifest and a stated reason. This is the only command in the
# repository that opens the test split.
step "8/9  Sealed test-set evaluation (opens the test split, once)"
"$PY" -m rto_sentinel.cli final-test \
  --unseal-reason "bootstrap: model, calibration and threshold methodology frozen in the manifest" \
  --seed "$SEED" --orders "$ORDERS" --customers "$CUSTOMERS" \
  --start-date "$START_DATE" --end-date "$END_DATE"

# --- 9. economics, fairness, robustness, monitoring, reports -------------------
step "9/9  Economics, fairness, distribution shift, drift, reports"
"$PY" -m rto_sentinel.cli economics
"$PY" -m rto_sentinel.cli fairness --split validation
"$PY" -m rto_sentinel.cli fairness --split test
"$PY" -m rto_sentinel.cli shift --n-orders 9000
"$PY" -m rto_sentinel.cli monitor --split validation
"$PY" -m rto_sentinel.cli responsible-report

# --- console dependencies -----------------------------------------------------
if [ -d console ] && [ ! -d console/node_modules ]; then
  step "Console dependencies"
  (cd console && npm install --silent)
fi

cat <<'DONE'

==============================================================================
Bootstrap complete. Everything below came from one seeded dataset run.

Start the two processes, in two terminals:

    .venv/Scripts/python.exe -m uvicorn rto_sentinel.api.main:app --port 8000
    cd console && npm run dev

    Console  http://localhost:5173
    API docs http://localhost:8000/docs

Generated reports:

    docs/model_card.md         the model, its measurements and its limits
    docs/economics.md          threshold derivation and the friction ladder
    docs/responsible_ai.md     fairness cohorts, shift study, drift
    docs/evaluation_report.md  the consolidated measured results

The language layer is OFF and no agent explanation will be produced until
ANTHROPIC_API_KEY and RTO_AGENTS_ENABLED=true are set. That is deliberate: the
system refuses rather than inventing a model response.
==============================================================================
DONE
