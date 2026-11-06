# =====================================================================
# run_app.ps1 - NeuroTwin AI Streamlit Dashboard Runner (PowerShell)
# =====================================================================

$env:PYTHONPATH = "C:\big data prj\.venv\Lib\site-packages;${PSScriptRoot}"

$pythonExe = "C:\Users\AYESHA\AppData\Local\Programs\Python\Python313\python.exe"

if (-not (Test-Path $pythonExe)) {
    # Fallback to python in PATH
    $pythonExe = "python"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NEUROTWIN AI: Physiological Digital Shadow & Stress Analysis" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching NeuroTwin Research Dashboard..." -ForegroundColor Green
Write-Host "Local URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""

& $pythonExe -m streamlit run "$PSScriptRoot\dashboard\app.py"

