# Everything CI runs, in one command (Windows).
#
#   .\scripts\check.ps1
#
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Invoke-Step "ruff (lint)"                { & $py -m ruff check . }
Invoke-Step "ruff (format)"              { & $py -m ruff format --check . }
Invoke-Step "mypy (strict)"              { & $py -m mypy }
Invoke-Step "configuration validation"   { & $py -m rto_sentinel.cli config check }
Invoke-Step "pytest"                     { & $py -m pytest -q }

if (Test-Path "console\node_modules") {
    Push-Location console
    try {
        Invoke-Step "console: typecheck" { npm run typecheck }
        Invoke-Step "console: lint"      { npm run lint }
        Invoke-Step "console: test"      { npm run test }
        Invoke-Step "console: build"     { npm run build }
    } finally { Pop-Location }
} else {
    Write-Host "!! console\node_modules missing - skipping frontend checks." -ForegroundColor Yellow
    Write-Host "   Run: cd console; npm install"
}

Write-Host ""
Write-Host "All checks passed." -ForegroundColor Green
