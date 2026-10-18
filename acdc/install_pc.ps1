param([string]$PythonPath = 'D:/conda/python.exe')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath 'acdc/.venv/Scripts/python.exe')) {
        & uv venv acdc/.venv --python $PythonPath
        if ($LASTEXITCODE -ne 0) { throw 'ACDC virtual environment creation failed' }
    }
    & uv pip sync --python acdc/.venv/Scripts/python.exe acdc/requirements-pc.lock
    if ($LASTEXITCODE -ne 0) { throw 'ACDC dependency installation failed' }
} finally { Pop-Location }
