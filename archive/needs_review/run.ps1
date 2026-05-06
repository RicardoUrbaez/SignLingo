$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    throw 'Missing venv\Scripts\python.exe. Create the virtual environment and install requirements first.'
}

Set-Location $projectRoot
& $pythonExe 'app.py'