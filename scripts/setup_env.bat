@echo off
echo =====================================================================
echo 🛠️  NVIDIA NEMOTRON REASONING PIPELINE: WINDOWS ENVIRONMENT SETUP
echo =====================================================================
echo.

cd /d "%~dp0.."

:: Check if python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python was not found in your system PATH.
    echo Please install Python 3.10+ and select "Add Python to PATH" during installation.
    exit /b 1
)

:: Create virtual environment
if not exist .venv (
    echo 📦 Creating Python Virtual Environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ❌ ERROR: Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo ✨ Virtual environment (.venv) already exists.
)

:: Activate and install dependencies
echo ⚡ Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo 🚀 Upgrading pip...
python -m pip install --upgrade pip

echo 📦 Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to install dependencies.
    exit /b 1
)

echo.
echo =====================================================================
echo ✅ ENVIRONMENT SETUP SUCCESSFUL!
echo =====================================================================
echo.
echo To activate the virtual environment in PowerShell/CMD:
echo.
echo   CMD:        .venv\Scripts\activate.bat
echo   PowerShell: .venv\Scripts\Activate.ps1
echo.
echo To run unit tests to verify the setup:
echo.
echo   pytest
echo.
echo To run the full pipeline in debug/prep mode:
echo.
echo   python scripts/run_pipeline.py --phase prep --limit 5
echo.
echo =====================================================================
pause
