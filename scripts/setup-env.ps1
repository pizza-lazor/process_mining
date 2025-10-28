param(
    [string]$PythonExe = "python",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$bootstrapScript = Join-Path $projectRoot "scripts/bootstrap_env.py"

if (-not (Test-Path $bootstrapScript)) {
    throw "Bootstrap script not found at $bootstrapScript"
}

Push-Location $projectRoot

try {
    Write-Host "Bootstrapping virtual environment via $bootstrapScript"
    $argsList = @($bootstrapScript, "--python", $PythonExe)
    if ($Force.IsPresent) {
        $argsList += "--force"
    }

    & $PythonExe @argsList

    Write-Host ""
    Write-Host "Done. Activate the environment with:"
    Write-Host "    .\.venv\Scripts\Activate.ps1"
    Write-Host "Then run:"
    Write-Host "    python pyqt_app.py"
    Write-Host ""
}
finally {
    Pop-Location
}
