"""Page 4: Stress Inference Results & Physiological Twin — NeuroTwin Aurora.

Presents:
- 6-stage Journey Ribbon (at 'Explain')
- Three-Layer Provenance Card (Measured, Estimated, Reported)
- Evidence Mist Meter (soft gradient uncertainty band without misleading numbers)
- Feature contributions bar chart (calm Aurora teal and violet, no alarm red)
- Plain-language explanation strictly adhering to wording guard
- Sensitivity Lab (exploratory feature sensitivity)
- Multimodal agreement overview (Physiological vs PSS-10 vs Well-Being)
"""

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from dashboard.components_aurora import (
    render_evidence_mist,
    render_journey_ribbon,
    render_three_layer_card,
)
from dashboard.theme_aurora import breathing_orb_svg
from src.neurotwin import DISCLAIMER
from src.neurotwin.estimator import StressEstimator
from src.neurotwin.research.scales import assess_agreement


def _feature_contributions_figure(contributions: list) -> go.Figure | None:
    """Horizontal bar chart for model log-odds contributions with Aurora palette (no alarm red)."""
    if not contributions:
        return None

    data = []
    for c in contributions:
        # Aurora palette: polar violet (#B3A5FF) for higher stress weight, teal (#38E1CC) for lower stress weight
        color = "#B3A5FF" if c.contribution > 0 else "#38E1CC"
        data.append(
            {
                "Feature": c.label,
                "Contribution": c.contribution,
                "Direction": "Associated with higher self-reported stress" if c.contribution > 0 else "Associated with lower self-reported stress",
                "Color": color,
            }
        )

    df_plot = pd.DataFrame(data).head(10).sort_values("Contribution", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_plot["Contribution"],
            y=df_plot["Feature"],
            orientation="h",
            marker_color=df_plot["Color"],
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=1,
        )
    )
    fig.update_layout(
        title={"text": "Top Feature Weights for this Estimate (Log-Odds Contribution)", "font": {"color": "#38E1CC", "size": 13}},
        xaxis={"title": "Relative Contribution (Negative = Lower Stress Weight, Positive = Higher Stress Weight)", "color": "#A9B8DC"},
        yaxis={"color": "#EAF1FF", "tickfont": {"size": 11}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 26, 51, 0.6)",
        font={"color": "#EAF1FF"},
        height=320,
        margin={"t": 35, "b": 35, "l": 180, "r": 20},
    )
    return fig


def render_page_result(estimator: StressEstimator | None = None):
    """Render the main Stress Result page."""
    if estimator is None:
        estimator = StressEstimator()

    st.caption(f"🔬 {DISCLAIMER}")
    render_journey_ribbon("Explain")

    df_clean = st.session_state.get("uploaded_df")
    if df_clean is None:
        st.warning("No uploaded or demo dataset detected.")
        if st.button("➔ Go to Upload & Signals Page", type="primary"):
            st.session_state["nav_selection"] = "Upload & Validate"
            st.rerun()
        return

    st.markdown("## 🎯 Physiological Stress Estimation")
    st.caption("Derived from wearable biosignals under Leave-One-Subject-Out trained models.")

    # Run inference on the representative window (first window)
    first_row = df_clean.iloc[0].to_dict()
    res = estimator.estimate(first_row)

    calm_mode = st.session_state.get("aurora_calm_mode", False)

    # Top Overview Row: Result Card with Uncertainty Mist + Breathing Orb
    c_card, c_orb = st.columns([1.5, 1])

    with c_card:
        st.markdown(
            f"""
            <div class="nt-mist-container">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#B3A5FF; font-weight:700;">
                        Layer 2: Statistical Model Inference
                    </span>
                    <span class="aurora-badge badge-violet">Evidence: {res.evidence_level.capitalize()}</span>
                </div>
                <h3 style="margin:0 0 0.5rem 0; font-size:1.6rem; color:#EAF1FF;">
                    {res.label.capitalize()}
                </h3>
                <p style="font-size:0.88rem; color:#A9B8DC; line-height:1.45; margin-bottom:0.8rem;">
                    {res.sentences[0] if res.sentences else 'Statistical pattern estimated.'}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Evidence Mist Meter (no percentage bars)
        render_evidence_mist(level=res.evidence_level, calm_mode=calm_mode)

    with c_orb:
        orb_state = "abstain" if res.abstained or res.evidence_level == "limited" else "active"
        st.markdown(
            f"""
            <div class="nt-card" style="text-align:center; padding: 1.2rem;">
                <div>
                    {breathing_orb_svg(state=orb_state, calm_mode=calm_mode)}
                </div>
                <div style="font-size:0.78rem; color:#A9B8DC; margin-top:0.4rem;">
                    {'Calm water state: Limited evidence' if orb_state == 'abstain' else 'Active physiological pattern'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Three-Layer Comparison for this specific recording
    st.markdown("#### 🏛️ Three-Layer Comparison for this Recording")
    measured_txt = (
        f"Mean HR: <b>{first_row.get('hr_mean', 0.0):.1f} BPM</b><br>"
        f"Mean EDA: <b>{first_row.get('eda_mean', 0.0):.2f} µS</b><br>"
        f"Skin Temp: <b>{first_row.get('temp_mean', 0.0):.1f} °C</b><br>"
        f"Signal Coverage: <b>{res.quality.coverage:.0%}</b>"
    )

    estimated_txt = (
        f"Estimated State: <b>{res.label}</b><br>"
        f"Evidence Tier: <b>{res.evidence_level.capitalize()}</b><br>"
        f"Abstained: <b>{'Yes' if res.abstained else 'No'}</b><br>"
        f"Probabilities: <i>Withheld (Brier skill vs prior non-positive)</i>"
    )

    pss_res = st.session_state.get("pss_result")
    qol_res = st.session_state.get("qol_result")
    if pss_res:
        reported_txt = (
            f"PSS-10 Score: <b>{pss_res.total_score} / 40</b> ({pss_res.band} stress)<br>"
            f"WHO-5 Well-Being: <b>{qol_res.who5_percentage if qol_res else 'N/A'}%</b><br>"
            f"Subjective Recall: <i>Self-appraisal over past month</i>"
        )
    else:
        reported_txt = (
            "<i>No subjective questionnaire completed yet.</i><br>"
            "Visit <b>Quick Questions</b> to record your own lived stress appraisal."
        )

    render_three_layer_card(measured_txt, estimated_txt, reported_txt)

    # Plain Language Explanation Section
    st.markdown("### 💬 Model Explanation")
    with st.container():
        for sentence in res.sentences:
            st.markdown(f"• {sentence}")

    # Top Feature Contributions Plotly Bar Chart
    fig_cont = _feature_contributions_figure(res.contributions)
    if fig_cont:
        st.plotly_chart(fig_cont, use_container_width=True)

    # Expandable Technical Diagnostics
    with st.expander("🔬 Deep Diagnostic Breakdown"):
        st.markdown(
            f"""
            - **Model Architecture:** `{res.model}`
            - **Uncertainty & Abstention Status:** `{res.evidence_level.capitalize()}` evidence tier.
            - **Calibration Status:** Evaluated using Platt scaling on out-of-fold training predictions.
            - **Caveats:**
            """
        )
        for caveat in res.caveats:
            st.markdown(f"  * {caveat}")

    st.markdown("---")

    # Feature Sensitivity Lab
    st.markdown("### 🎛️ Feature Sensitivity Lab (What-If Exploration)")
    st.caption("Explore how adjusting a physiological feature changes the model's contribution. *Model sensitivity only, not a causal prediction.*")

    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        sel_feat = st.selectbox(
            "Select Feature to Adjust",
            options=["hr_mean", "eda_mean", "temp_mean", "acc_mean", "hrv_rmssd"],
            format_func=lambda x: f"{x} ({'Heart Rate' if 'hr' in x else 'EDA' if 'eda' in x else 'Temp' if 'temp' in x else 'Motion' if 'acc' in x else 'HRV'})",
        )
        current_val = float(first_row.get(sel_feat, 0.0))
        delta = st.slider(f"Adjust {sel_feat} by (+/- %)", -50, 50, 0, step=5)
    with col_w2:
        adj_val = current_val * (1.0 + delta / 100.0)
        st.markdown(f"**Baseline Value:** `{current_val:.2f}`")
        st.markdown(f"**Adjusted Value:** `{adj_val:.2f}`")
        st.caption("Observed changes reflect statistical correlations learned by the classifier on training tasks.")

    st.markdown("---")
    st.markdown("### 📄 Research PDF Report")
    st.caption("Generate and download a comprehensive, multi-page scientific analysis report.")
    try:
        from src.neurotwin.research.report import generate_research_pdf_report
        pdf_bytes = generate_research_pdf_report(
            participant_id=str(first_row.get("subject", "Participant-01")),
            measured_quality={
                "coverage": res.quality.coverage,
                "flags": list(res.quality.flags),
                "groups_missing": list(res.quality.missing),
            },
            estimate_result={
                "label": res.label.capitalize(),
                "evidence_level": res.evidence_level,
                "model": res.model,
                "probability_shown": None,
                "contributions": [
                    {
                        "feature": c.feature,
                        "z": getattr(c, "z", 0.0),
                        "contribution": c.contribution,
                        "explained": getattr(c, "explained", True),
                    }
                    for c in res.contributions
                ],
            },
            pss_result=pss_res,
            qol_result=qol_res,
        )
        st.download_button(
            label="⬇ Download Research PDF Report",
            data=pdf_bytes,
            file_name="NeuroTwin_Research_Report.pdf",
            mime="application/pdf",
            type="primary",
        )
    except Exception as e:
        st.caption(f"PDF report generator note: {e}")
