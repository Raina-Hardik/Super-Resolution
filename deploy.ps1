# Super-Resolution Deployment Script for Windows
# This script sets up and deploys the Super-Resolution application

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "prod")]
    [string]$Mode = "dev",
    
    [Parameter(Mandatory=$false)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Colors
$ColorGreen = "Green"
$ColorYellow = "Yellow"
$ColorRed = "Red"

# Configuration
$AppName = "super-resolution"
$VenvDir = "venv"
$ScriptDir = $PSScriptRoot

Set-Location $ScriptDir

Write-Host "Super-Resolution Deployment Script" -ForegroundColor $ColorGreen
Write-Host "====================================" -ForegroundColor $ColorGreen
Write-Host ""

# Check for uv (fast package manager)
$UseUv = $false
try {
    uv --version | Out-Null
    Write-Host "✓ Found uv (Astral's fast Python package manager)" -ForegroundColor $ColorGreen
    $UseUv = $true
}
catch {
    Write-Host "uv not found, using pip. Install uv for faster dependency management:" -ForegroundColor $ColorYellow
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor $ColorYellow
}

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor $ColorYellow
try {
    $PythonVersion = (python --version 2>&1) -replace "Python ", ""
    Write-Host "✓ Found Python $PythonVersion" -ForegroundColor $ColorGreen
}
catch {
    Write-Host "Error: Python is not installed!" -ForegroundColor $ColorRed
    Write-Host "Please install Python 3.11 or higher from https://www.python.org/" -ForegroundColor $ColorRed
    exit 1
}

# Setup environment based on available tools
if ($UseUv) {
    # Using uv - much faster!
    Write-Host "Setting up environment with uv..." -ForegroundColor $ColorYellow
    
    # Create venv with uv if it doesn't exist
    if (-not (Test-Path $VenvDir)) {
        uv venv $VenvDir
        Write-Host "✓ Virtual environment created with uv" -ForegroundColor $ColorGreen
    }
    else {
        Write-Host "✓ Virtual environment exists" -ForegroundColor $ColorGreen
    }
    
    # Activate virtual environment
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        & $ActivateScript
    }
    else {
        Write-Host "Error: Cannot activate virtual environment!" -ForegroundColor $ColorRed
        exit 1
    }
    
    # Install dependencies with uv (much faster than pip)
    Write-Host "Installing dependencies with uv..." -ForegroundColor $ColorYellow
    uv pip install -e .
    Write-Host "✓ Dependencies installed" -ForegroundColor $ColorGreen
}
else {
    # Using traditional pip
    Write-Host "Checking pip installation..." -ForegroundColor $ColorYellow
    try {
        python -m pip --version | Out-Null
        Write-Host "✓ pip is installed" -ForegroundColor $ColorGreen
    }
    catch {
        Write-Host "Error: pip is not installed!" -ForegroundColor $ColorRed
        exit 1
    }

    # Create virtual environment
    Write-Host "Setting up virtual environment..." -ForegroundColor $ColorYellow
    if (-not (Test-Path $VenvDir)) {
        python -m venv $VenvDir
        Write-Host "✓ Virtual environment created" -ForegroundColor $ColorGreen
    }
    else {
        Write-Host "✓ Virtual environment exists" -ForegroundColor $ColorGreen
    }

    # Activate virtual environment
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        & $ActivateScript
    }
    else {
        Write-Host "Error: Cannot activate virtual environment!" -ForegroundColor $ColorRed
        exit 1
    }

    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor $ColorYellow
    python -m pip install --upgrade pip setuptools wheel | Out-Null

    # Install dependencies
    Write-Host "Installing dependencies..." -ForegroundColor $ColorYellow
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor $ColorGreen
}

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor $ColorYellow
@("uploads", "output", "Data") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor $ColorGreen

# Start application
Write-Host ""
if ($Mode -eq "dev") {
    Write-Host "Starting in Development mode..." -ForegroundColor $ColorGreen
    Write-Host ""
    Write-Host "Application will be available at: http://localhost:$Port" -ForegroundColor $ColorYellow
    Write-Host "Press Ctrl+C to stop" -ForegroundColor $ColorYellow
    Write-Host ""
    
    uvicorn app:app --host 0.0.0.0 --port $Port --reload
}
elseif ($Mode -eq "prod") {
    Write-Host "Starting in Production mode..." -ForegroundColor $ColorGreen
    Write-Host ""
    
    # Ask for process manager
    Write-Host "Select process manager:"
    Write-Host "1) Run in foreground"
    Write-Host "2) Run as Windows Service (requires admin)"
    Write-Host "3) Run in background (PowerShell job)"
    $Choice = Read-Host "Enter choice [1-3]"
    
    switch ($Choice) {
        "1" {
            Write-Host ""
            Write-Host "Application will be available at: http://localhost:$Port" -ForegroundColor $ColorYellow
            Write-Host "Press Ctrl+C to stop" -ForegroundColor $ColorYellow
            Write-Host ""
            
            uvicorn app:app --host 0.0.0.0 --port $Port --workers 4
        }
        "2" {
            Write-Host ""
            Write-Host "Creating Windows Service..." -ForegroundColor $ColorYellow
            Write-Host "Note: This requires NSSM (Non-Sucking Service Manager)" -ForegroundColor $ColorYellow
            Write-Host "Download from: https://nssm.cc/download" -ForegroundColor $ColorYellow
            Write-Host ""
            
            $NssmPath = Read-Host "Enter path to nssm.exe (or press Enter to skip)"
            
            if ($NssmPath -and (Test-Path $NssmPath)) {
                $UvicornPath = Join-Path $VenvDir "Scripts\uvicorn.exe"
                
                & $NssmPath install $AppName $UvicornPath "app:app --host 0.0.0.0 --port $Port --workers 4"
                & $NssmPath set $AppName AppDirectory $ScriptDir
                & $NssmPath set $AppName DisplayName "Super-Resolution API"
                & $NssmPath set $AppName Description "Image Super-Resolution FastAPI Application"
                & $NssmPath set $AppName Start SERVICE_AUTO_START
                & $NssmPath start $AppName
                
                Write-Host "✓ Service created and started" -ForegroundColor $ColorGreen
                Write-Host ""
                Write-Host "Service commands:"
                Write-Host "  Status:  $NssmPath status $AppName"
                Write-Host "  Stop:    $NssmPath stop $AppName"
                Write-Host "  Restart: $NssmPath restart $AppName"
                Write-Host "  Remove:  $NssmPath remove $AppName confirm"
            }
            else {
                Write-Host "Skipping service installation" -ForegroundColor $ColorYellow
            }
        }
        "3" {
            Write-Host ""
            Write-Host "Starting in background..." -ForegroundColor $ColorYellow
            
            $Job = Start-Job -ScriptBlock {
                param($Dir, $Port)
                Set-Location $Dir
                & "$Dir\$VenvDir\Scripts\Activate.ps1"
                uvicorn app:app --host 0.0.0.0 --port $Port --workers 4
            } -ArgumentList $ScriptDir, $Port
            
            Write-Host "✓ Application started in background (Job ID: $($Job.Id))" -ForegroundColor $ColorGreen
            Write-Host ""
            Write-Host "Job commands:"
            Write-Host "  Status:  Get-Job -Id $($Job.Id)"
            Write-Host "  Logs:    Receive-Job -Id $($Job.Id) -Keep"
            Write-Host "  Stop:    Stop-Job -Id $($Job.Id); Remove-Job -Id $($Job.Id)"
        }
        default {
            Write-Host "Invalid choice" -ForegroundColor $ColorRed
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor $ColorGreen
Write-Host ""
Write-Host "Quick Start Guide:" -ForegroundColor $ColorYellow
Write-Host "  Web Interface: http://localhost:$Port" -ForegroundColor $ColorYellow
Write-Host "  API Docs:      http://localhost:$Port/docs" -ForegroundColor $ColorYellow
Write-Host "  Health Check:  http://localhost:$Port/health" -ForegroundColor $ColorYellow
