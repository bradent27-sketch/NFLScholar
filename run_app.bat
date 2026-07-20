@echo off
rem NFL Scholar one-click launcher - double-click this to start the app.
rem Starts the local Streamlit server and opens your browser automatically.
rem (A single-file .exe was evaluated and deliberately not done: Streamlit
rem is a web server, so an .exe would still just do exactly this, via a
rem fragile 500MB+ frozen bundle that breaks on library upgrades.)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on this machine. Install it from python.org
    echo ^(check "Add python.exe to PATH" during install^), then run this again.
    pause
    exit /b 1
)

python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Streamlit failed to start. If it's not installed, run:
    echo     pip install -r requirements.txt
    pause
)
