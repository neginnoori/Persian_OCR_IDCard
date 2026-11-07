$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
Push-Location $Project
try {
    & py -3.11 (Join-Path $PSScriptRoot 'setup_windows.py')
    if ($LASTEXITCODE -ne 0) { throw "Setup failed (exit $LASTEXITCODE). See the first error above." }
} finally {
    Pop-Location
}
