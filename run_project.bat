@echo off
REM =====================================================================
REM  run_project.bat  —  Wearable Stress Analysis  One-Click Runner
REM  Runs the full pipeline: load → EDA → features → stats → models
REM =====================================================================

echo.
echo ============================================================
echo  Wearable Stress Analysis — Full Pipeline
echo ============================================================
echo.

python scripts\01_load_and_verify.py
if errorlevel 1 (echo [ERROR] Step 1 failed. & exit /b 1)
echo [OK] Step 1 complete.

python scripts\02_eda.py
if errorlevel 1 (echo [ERROR] Step 2 failed. & exit /b 1)
echo [OK] Step 2 complete.

python scripts\03_feature_engineering.py
if errorlevel 1 (echo [ERROR] Step 3 failed. & exit /b 1)
echo [OK] Step 3 complete.

python scripts\04_stress_analysis.py
if errorlevel 1 (echo [ERROR] Step 4 failed. & exit /b 1)
echo [OK] Step 4 complete.

python scripts\05_model_evaluation.py
if errorlevel 1 (echo [ERROR] Step 5 failed. & exit /b 1)
echo [OK] Step 5 complete.

echo.
echo ============================================================
echo  Pipeline complete. Outputs in data/ and reports/figures/
echo ============================================================
echo.

REM Run tests
echo Running tests...
python -m pytest tests\ -v --tb=short 2>&1
echo.
pause
