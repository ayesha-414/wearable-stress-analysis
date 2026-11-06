#!/usr/bin/env bash
# =====================================================================
#  run_app.sh — NeuroTwin AI Streamlit Dashboard Runner (Linux/macOS)
# =====================================================================

set -e

echo ""
echo "============================================================"
echo "  NEUROTWIN AI: Physiological Digital Shadow & Stress Analysis"
echo "============================================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:."

if command -v streamlit &> /dev/null; then
    streamlit run dashboard/app.py
elif command -v python3 &> /dev/null; then
    python3 -m streamlit run dashboard/app.py
else
    python -m streamlit run dashboard/app.py
fi

