# Create the schema, generate a dataset, validate it, and load it into PostgreSQL.
#
#   .\scripts\seed_db.ps1
#   .\scripts\seed_db.ps1 --orders 12000 --customers 4000 --lenient
#
# Requires a running database:  docker compose up -d db
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".env")) {
    Write-Error ".env not found. Copy .env.example to .env and set POSTGRES_PASSWORD."
}

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m rto_sentinel.cli seed-db @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
