param(
    [ValidateRange(1024,65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
# Keep Python and the Paddle subprocess Unicode-safe on Windows.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$EnvFile = Join-Path $Project '.env'
if (-not (Test-Path $Python)) { throw 'Run scripts/setup.ps1 first.' }
if (-not (Test-Path $EnvFile)) { throw 'Missing .env. Run setup or copy .env.example and set OCR_API_KEY.' }
Push-Location $Project
try {
    & $Python (Join-Path $Project '..\dino-sam-hezar-colab\scripts\verify_install.py') (Join-Path $Project '..\dino-sam-hezar-colab')
    if ($LASTEXITCODE -ne 0) { throw 'Installed OCR source is stale. Rerun scripts/setup.ps1.' }
    Write-Host "Starting OCR service. Once ready, open http://127.0.0.1:$Port/docs"
    # One process owns one set of GPU models. Never enable reload or extra workers.
    & $Python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --workers 1 --env-file $EnvFile --no-access-log
    if ($LASTEXITCODE -ne 0) { throw "Service exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
