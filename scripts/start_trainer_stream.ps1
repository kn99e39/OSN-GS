$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

& $python -m osn_gs.interop.trainer_ws_server --host "127.0.0.1" --port 8080 @args

exit $LASTEXITCODE
