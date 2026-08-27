# First-time developer setup (Windows).
#
#   .\scripts\setup.ps1
#
# Creates the virtualenv, installs Python and Node dependencies, and copies
# .env.example to .env if it does not exist. It never writes a credential.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "==> Creating virtualenv" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }

Write-Host "==> Installing Python dependencies" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Write-Host "==> Installing console dependencies" -ForegroundColor Cyan
Push-Location console
try { npm install } finally { Pop-Location }

if (-not (Test-Path ".env")) {
    Write-Host "==> Creating .env from .env.example" -ForegroundColor Cyan
    Copy-Item .env.example .env
    Write-Host ""
    Write-Host "   .env created. Edit it before running against PostgreSQL:" -ForegroundColor Yellow
    Write-Host "     - set POSTGRES_PASSWORD to something of your own"
    Write-Host "     - leave ANTHROPIC_API_KEY empty unless you want the language layer"
} else {
    Write-Host "==> .env already exists; leaving it alone" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Setup complete. Next:" -ForegroundColor Green
Write-Host "  .\scripts\check.ps1                 # run every check"
Write-Host "  .\.venv\Scripts\rto-sentinel serve   # start the API"
Write-Host "  cd console; npm run dev             # start the console"
