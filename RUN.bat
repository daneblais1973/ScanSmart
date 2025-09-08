@echo off
echo ========================================
echo FINANCIAL CATALYST SCANNER - WINDOWS SETUP
echo ========================================
echo.
echo Installing Financial Intelligence Platform...
echo Features: AI Analysis, Real-time Monitoring, ML Training
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python detected - proceeding with installation...
echo.

REM Run the Python launcher
python run_local.py

pause