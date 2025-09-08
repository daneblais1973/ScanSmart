@echo off
cd /d "%~dp0"
echo ========================================
echo STOPPING FINANCIAL CATALYST SCANNER
echo ========================================
echo.

echo 🛑 Shutting down Streamlit application...

REM Kill all Python processes running Streamlit
taskkill /f /im python.exe /fi "WINDOWTITLE eq streamlit*" 2>nul
taskkill /f /im python.exe /fi "COMMANDLINE eq *streamlit*" 2>nul

REM Alternative method - kill processes using port 5000
echo 🔍 Checking for processes on port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do (
    echo 🛑 Stopping process ID: %%a
    taskkill /f /pid %%a 2>nul
)

REM Kill any remaining Streamlit processes
wmic process where "commandline like '%streamlit%' and commandline like '%app.py%'" delete 2>nul

echo.
echo ✅ Financial Catalyst Scanner stopped successfully!
echo 📊 Application is no longer running on localhost:5000
echo.
echo Press any key to exit...
pause >nul