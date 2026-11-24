# Use the project's existing environment; no activation or global PATH changes.
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Error 'Project environment is missing. Follow the fresh-install steps in README.md.'
    exit 1
}
& $projectPython -u (Join-Path $PSScriptRoot 'main.py')
exit $LASTEXITCODE
