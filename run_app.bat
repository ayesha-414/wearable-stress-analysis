@echo off
REM =====================================================================
REM  run_app.bat — NeuroTwin AI Streamlit Dashboard Runner
REM =====================================================================

echo.
echo ============================================================
echo   NEUROTWIN AI: Physiological Digital Twin & Stress Analysis
echo ============================================================
echo.

setlocal enabledelayedexpansion
set "PYTHONPATH=C:\big data prj\.venv\Lib\site-packages;C:\Users\AYESHA\AppData\Local\Programs\Python\Python313\Lib\site-packages;%CD%"

REM Check for Python in default user path or system PATH
if exist "C:\Users\AYESHA\AppData\Local\Programs\Python\Python313\python.exe" (
    set "PY_CMD=C:\Users\AYESHA\AppData\Local\Programs\Python\Python313\python.exe"
) else (
    set "PY_CMD=python"
)

echo Starting NeuroTwin Dashboard on http://localhost:8501 ...
echo.
"%PY_CMD%" -m streamlit run dashboard\app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)
