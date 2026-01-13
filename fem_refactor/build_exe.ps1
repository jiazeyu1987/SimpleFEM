$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "python not found in PATH" }

& python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { & python -m pip install pyinstaller }

$name = "SimpleFEM_ROI_Daemon"
$entry = "fem_refactor\simple_roi_daemon.py"
$data = "fem_refactor\simple_fem_config.json;fem_refactor"

& python -m PyInstaller --noconfirm --clean --onefile --console --name $name --add-data $data $entry

Write-Output ("dist\{0}.exe" -f $name)

