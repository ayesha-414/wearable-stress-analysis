"""Page 8: Quality of Life, Stress Scale & SEM Architecture.

Presents:
- PSS-10 Psychometric Profile (score 0–40, band, normative percentile)
- WHO-5 Well-Being Index (0–100) & 4-Domain QoL spider/bar chart
- One-Factor Confirmatory Factor Model (CFA) loadings & Cronbach's alpha
- Conceptual Structural Equation Model (SEM) path architecture (explicitly labeled 'conceptual, not fitted')
- SVM Score vs PSS-10 Agreement Analysis (physiological state vs 30-day cognitive recall)
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from dashboard.theme import metric_card
from src.neurotwin.research.scales import (
    calculate_one_factor_cfa,
    compute_svm_model_score,
    get_sem_path_model,
)


def _render_sem_diagram() -> str:
    """Return responsive SVG for the conceptual SEM path architecture."""
    return """
    <svg width="100%" height="220" viewBox="0 0 900 220" xmlns="http://www.w3.org/2000/svg" style="background:rgba(12,21,39,0.7); border:1px solid rgba(56,189,248,0.2); border-radius:12px; margin:1rem 0;">
      <defs>
        <marker id="sem-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#00f2fe"/>
        </marker>
      </defs>

      <!-- Label: Conceptual Notice -->
      <rect x="20" y="12" width="220" height="24" rx="4" fill="rgba(245,158,11,0.15)" stroke="#f59e0b" stroke-width="1"/>
      <text x="130" y="28" fill="#f59e0b" font-family="Inter, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">CONCEPTUAL MODEL (NOT FITTED)</text>

      <!-- Node 1: Sensor Signals (Observed) -->
      <rect x="30" y="65" width="170" height="110" rx="8" fill="rgba(15,27,51,0.9)" stroke="#38bdf8" stroke-width="1.8"/>
      <text x="115" y="95" fill="#38bdf8" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Sensor Signals</text>
      <text x="115" y="115" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">HR, HRV, EDA, TEMP, ACC</text>
      <text x="115" y="135" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">[Measured Layer]</text>
      <text x="115" y="155" fill="#64748b" font-family="Inter, sans-serif" font-size="9" text-anchor="middle">Empatica E4 Wristband</text>

      <!-- Path 1 -> 2 -->
      <line x1="200" y1="120" x2="255" y2="120" stroke="#00f2fe" stroke-width="2" marker-end="url(#sem-arrow)"/>
      <text x="228" y="112" fill="#00f2fe" font-family="Inter, sans-serif" font-size="9" text-anchor="middle">Autonomic</text>

      <!-- Node 2: Estimated Stress (Latent/Classifier) -->
      <ellipse cx="340" cy="120" rx="80" ry="50" fill="rgba(15,27,51,0.9)" stroke="#00f2fe" stroke-width="1.8"/>
      <text x="340" y="112" fill="#00f2fe" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Estimated Stress</text>
      <text x="340" y="130" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Momentary State</text>
      <text x="340" y="146" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">[Estimated Layer]</text>

      <!-- Path 2 -> 3 -->
      <line x1="420" y1="120" x2="475" y2="120" stroke="#00f2fe" stroke-width="2" marker-end="url(#sem-arrow)"/>
      <text x="448" y="112" fill="#00f2fe" font-family="Inter, sans-serif" font-size="9" text-anchor="middle">Appraisal</text>

      <!-- Node 3: Perceived Stress (Reported) -->
      <ellipse cx="560" cy="120" rx="80" ry="50" fill="rgba(15,27,51,0.9)" stroke="#f59e0b" stroke-width="1.8"/>
      <text x="560" y="112" fill="#f59e0b" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Perceived Stress</text>
      <text x="560" y="130" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">PSS-10 (0–40)</text>
      <text x="560" y="146" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">[Reported Layer]</text>

      <!-- Path 3 -> 4 -->
      <line x1="640" y1="120" x2="695" y2="120" stroke="#00f2fe" stroke-width="2" marker-end="url(#sem-arrow)"/>
      <text x="668" y="112" fill="#00f2fe" font-family="Inter, sans-serif" font-size="9" text-anchor="middle">Negative Path</text>

      <!-- Node 4: Quality of Life (Reported Outcome) -->
      <rect x="700" y="65" width="170" height="110" rx="8" fill="rgba(15,27,51,0.9)" stroke="#00d4aa" stroke-width="1.8"/>
      <text x="785" y="95" fill="#00d4aa" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Quality of Life</text>
      <text x="785" y="115" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">WHO-5 & WHOQOL</text>
      <text x="785" y="135" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">[Reported Layer]</text>
      <text x="785" y="155" fill="#64748b" font-family="Inter, sans-serif" font-size="9" text-anchor="middle">Distal Outcome Anchor</text>

      <!-- Direct Path 2 -> 4 (Curved at top) -->
      <path d="M 340,70 Q 560,30 785,65" fill="none" stroke="#64748b" stroke-dasharray="4,4" stroke-width="1.5" marker-end="url(#sem-arrow)"/>
      <text x="560" y="44" fill="#94a3b8" font-family="Inter, sans-serif" font-size="9" text-anchor="middle">Direct Somatic Impact (Fatigue / Energy)</text>
    </svg>
    """


def render_page_qol_scale():
    """Render the Quality of Life and Psychometric Scales page."""
    st.markdown("## 🌿 Quality of Life & Stress Scale Architecture")
    st.markdown(
        """
        Integrates subjective psychometric instruments with physiological data under a unified
        structural framework. Emphasizes the critical scientific separation between **Estimated**
        physiological states and **Reported** cognitive appraisals.
        """
    )

    pss_res = st.session_state.get("pss_result")
    qol_res = st.session_state.get("qol_result")
    uploaded_df = st.session_state.get("uploaded_df")

    # --- Section 1: Psychometric Assessment Cards ---
    st.markdown("### 📊 Your Psychometric Profile")
    if pss_res:
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(metric_card("PSS-10 Total", f"{pss_res.total_score} / 40", f"{pss_res.band} Stress"), unsafe_allow_html=True)
        with p2:
            st.markdown(metric_card("Norm Percentile", f"{pss_res.norm_percentile}th", "General Adult Norms"), unsafe_allow_html=True)
        with p3:
            who_val = f"{qol_res.who5_percentage:.0f} / 100" if qol_res else "N/A"
            st.markdown(metric_card("WHO-5 Well-Being", who_val, "Quality of Life"), unsafe_allow_html=True)
        with p4:
            qol_status = qol_res.wellbeing_level if qol_res else "N/A"
            st.markdown(metric_card("Well-Being Tier", qol_status, "Cutoff <= 50"), unsafe_allow_html=True)

        st.caption(f"ℹ️ {pss_res.band_description}")
    else:
        st.info("💡 You have not completed the Quick Questions yet. Take 2 minutes on the Quick Questions page to populate your psychometric profile.")

    st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

    # --- Section 2: One-Factor Confirmatory Factor Analysis (CFA) ---
    st.markdown("### 🧩 One-Factor Confirmatory Model of Perceived Stress (PSS-10)")
    st.markdown(
        r"""
        **Scientific Notice:** A factor model evaluates the latent covariance structure across thousands of subjects.
        Below are the standard standardized factor loadings ($\lambda$) from the normative psychometric validation
        by Taylor (2015) ($N=2,387$, $\alpha = 0.865$). If you completed the questions, your individual score is mapped
        as a standardized factor estimate.
        """
    )

    cfa_data = calculate_one_factor_cfa(pss_res.scored_items if pss_res else None)

    col_load, col_stat = st.columns([1.5, 1])

    with col_load:
        loadings_df = pd.DataFrame(
            [
                {"Item": f"Item {k}", "Loading (λ)": v}
                for k, v in cfa_data["factor_loadings"].items()
            ]
        )
        fig_load = px.bar(
            loadings_df,
            x="Item",
            y="Loading (λ)",
            title="Standardized Factor Loadings on General Perceived Stress Factor",
            color="Loading (λ)",
            color_continuous_scale="Teal",
            range_y=[0, 1.0],
        )
        fig_load.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 18, 34, 0.6)",
            font={"color": "#f8fafc"},
            height=320,
        )
        st.plotly_chart(fig_load, use_container_width=True)

    with col_stat:
        st.markdown(
            f"""
            <div class="nt-card" style="padding:1.2rem;">
                <h4 style="color:#38bdf8; margin-top:0;">Model Psychometrics (Taylor 2015)</h4>
                <ul style="font-size:0.85rem; color:#cbd5e1; line-height:1.6; padding-left:1.2rem;">
                    <li><b>Latent Factor:</b> General Perceived Stress</li>
                    <li><b>Internal Consistency:</b> Cronbach's α = <b>{cfa_data['cronbach_alpha']}</b></li>
                    <li><b>Variance Explained:</b> <b>{cfa_data['variance_explained']:.1%}</b></li>
                    <li><b>Fit Index (CFI):</b> 0.942 (good fit &gt; 0.90)</li>
                    <li><b>RMSEA:</b> 0.058 (close fit &lt; 0.06)</li>
                    <li><b>Your Standardized Z-Score:</b> {cfa_data['user_latent_score_z'] if cfa_data['user_latent_score_z'] is not None else 'N/A'}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

    # --- Section 3: Structured Equation Model (SEM) Path View ---
    st.markdown("### 🗺️ Structural Equation Model (SEM) Architecture")
    st.markdown(
        """
        The SEM below illustrates the theoretical path relationships from raw sensor features, through
        momentary autonomic state estimation, to retrospective cognitive appraisal and distal quality of life.
        """
    )
    st.markdown(_render_sem_diagram(), unsafe_allow_html=True)

    sem_info = get_sem_path_model()
    st.caption(f"⚠️ **Integrity Note:** {sem_info['disclaimer']}")

    st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

    # --- Section 4: SVM Model Score vs PSS-10 Agreement ---
    st.markdown("### ⚖️ SVM Model Score vs 30-Day PSS-10 Score")
    st.markdown(
        """
        Comparison between the **SVM-RBF physiological classifier score** (momentary autonomic activation)
        and the **PSS-10 score** (30-day cognitive self-report).
        """
    )

    if uploaded_df is not None:
        first_row = uploaded_df.iloc[0].to_dict()
        svm_score, svm_label = compute_svm_model_score(first_row)

        c_svm, c_pss = st.columns(2)
        with c_svm:
            st.markdown(metric_card("SVM Model Score", f"{svm_score:.1%}", f"Predicted: {svm_label} Stress"), unsafe_allow_html=True)
            st.caption("Derived from instantaneous heart rate, EDA and temperature biosignals.")
        with c_pss:
            pss_val = f"{pss_res.total_score} / 40" if pss_res else "Not completed"
            pss_band = pss_res.band if pss_res else "N/A"
            st.markdown(metric_card("30-Day PSS Score", pss_val, f"Classification: {pss_band}"), unsafe_allow_html=True)
            st.caption("Derived from subjective cognitive self-appraisal over the past month.")

        st.markdown(
            """
            <div class="nt-card" style="margin-top:0.8rem;">
                <div style="font-weight:600; color:#38bdf8; font-size:0.9rem;">Scientific Concordance Insight</div>
                <div style="font-size:0.84rem; color:#cbd5e1; line-height:1.5; margin-top:0.3rem;">
                    Autonomic arousal and cognitive appraisal measure distinct dimensions of the human stress response.
                    A person can be physiologically aroused while exercising or highly engaged at work without experiencing
                    negative psychological distress. Conversely, mental rumination and worry can occur while seated quietly
                    with resting heart rates.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload data to view the live SVM model score.")
