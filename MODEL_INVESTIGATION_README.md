# Model investigation — diagnosis phase (additions only)

Copy these files into the project root (same folders). Nothing existing is modified or deleted.

Run (PowerShell, project root):
    Copy-Item -Recurse . "..\NeuroTwin_full_project_BACKUP_$(Get-Date -Format yyyyMMdd_HHmm)"
    python scripts\11_environment_fingerprint.py
    python scripts\10_model_diagnosis.py
    python -m pytest tests -q

Read first: docs\neurotwin\MODEL_INVESTIGATION_DIAGNOSIS_AND_PLAN.md
