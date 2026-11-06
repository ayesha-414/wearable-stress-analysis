"""Page 6: Model Evaluation & Transparency — NeuroTwin Aurora.

Presents:
- 6-stage Journey Ribbon (at 'Learn')
- Side-by-side v1 (original protocol mapping) vs v2 (corrected mapping) comparison cards
- Programmatic Decision Rule (§9.4) audit panel and honest RETAIN_BASELINE verdict
- Interactive 'Inspect any participant' LOSO explorer across all 35 participants
- Verified empirical metric tables (no fabricated numbers)
- 'What this model cannot do' limitation banner
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.components_aurora import (
    render_decision_rule_panel,
    render_journey_ribbon,
    render_v1_v2_comparison,
    render_what_model_cannot_do,
)
from src.neurotwin import DISCLAIMER
from src.neurotwin.estimator import StressEstimator


def render_page_evaluation(data_dict: dict | None = None):
    """Render the Model Evaluation & Transparency page."""
    st.caption(f"🔬 {DISCLAIMER}")
    render_journey_ribbon("Learn")

    st.markdown("## 🔬 Evaluation & Model Transparency")
    st.markdown(
        "All metrics and comparisons shown here are strictly derived from **Leave-One-Subject-Out (LOSO)** "
        "cross-validation. Every estimate is produced by a model that was never trained on that participant."
    )

    # 1. Prominent 'What This Model Cannot Do' Banner
    render_what_model_cannot_do()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Side-by-Side V1 vs V2 Comparison Cards
    render_v1_v2_comparison()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Decision Rule (§9.4) Verdict Panel
    render_decision_rule_panel()

    st.markdown("---")

    # 4. Interactive Participant Explorer (Inspect Any Participant)
    st.markdown("### 👤 Inspect Any Participant")
    st.caption("Review out-of-sample predictions and ground-truth self-ratings for any of the 35 participants.")

    v2_preds_path = Path("data/processed/model_investigation/v2/benchmark_predictions_primary_B.csv")
    if not v2_preds_path.is_file():
        v2_preds_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_investigation" / "v2" / "benchmark_predictions_primary_B.csv"
    if v2_preds_path.is_file():
        df_preds = pd.read_csv(v2_preds_path)
        all_subjects = sorted(df_preds["subject"].unique())
        selected_sub = st.selectbox("Choose Participant to Inspect:", all_subjects, index=0)

        sub_windows = df_preds[df_preds["subject"] == selected_sub].copy()
        if not sub_windows.empty:
            sub_windows["Agreement"] = sub_windows["y_true"] == sub_windows["y_pred"]
            sub_windows["Agreement_Icon"] = sub_windows["Agreement"].apply(lambda a: "✅ Match" if a else "⚠️ Discrepancy")
            sub_windows["Observed_Rating"] = sub_windows["y_true"].apply(lambda y: ">= 5 (High)" if y == 1 else "< 5 (Low)")
            sub_windows["Model_Estimate"] = sub_windows["y_pred"].apply(lambda y: "High Stress" if y == 1 else "Low Stress")

            display_cols = ["stage", "Observed_Rating", "Model_Estimate", "Agreement_Icon"]
            st.dataframe(
                sub_windows[display_cols].rename(
                    columns={
                        "stage": "Protocol Stage",
                        "Observed_Rating": "Self-Reported Rating",
                        "Model_Estimate": "LOSO Model Estimate",
                        "Agreement_Icon": "Agreement",
                    }
                ),
                use_container_width=True,
            )
            match_rate = sub_windows["Agreement"].mean()
            st.caption(f"Participant {selected_sub}: {match_rate:.0%} agreement across {len(sub_windows)} recorded stages.")
    else:
        st.info("V2 benchmark predictions file available under data/processed/model_investigation/v2/.")

    st.markdown("---")

    # 5. Full Benchmark Results Table
    st.markdown("### 📋 Primary Benchmark Results (Design B — Duration-Matched 30 s Windows)")
    v2_results_path = Path("data/processed/model_investigation/v2/benchmark_results_primary_B.csv")
    if v2_results_path.is_file():
        df_res = pd.read_csv(v2_results_path)
        metric_cols = [c for c in ["model", "accuracy", "balanced_accuracy", "roc_auc", "brier_score", "brier_skill_score"] if c in df_res.columns]
        st.dataframe(df_res[metric_cols].round(4), use_container_width=True)
    else:
        st.info("Generating or loading benchmark summary tables.")

