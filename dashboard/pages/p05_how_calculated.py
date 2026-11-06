"""Page 5: How Stress Is Calculated — The Research Core.

Presents:
- Complete research-grade pipeline diagram (Raw E4 signals -> Preprocessing -> 23 Features -> Standardization -> LOSO -> Label)
- Which features were used / not used table (all 23 used; hrv_lf_hf_ratio marked suppressed from explanations)
- 23 Expandable feature cards with:
  * Name, group, unit
  * Exact mathematical formula rendered with LaTeX (st.latex)
  * Physiological rationale (ANS sympathetic vs parasympathetic dynamics)
  * Evidence validity label (Matches standard / Custom derived / Heuristic approximation / Uncertain)
  * Known limitations & discrepancies
  * Verified citation with clickable DOI link
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import evidence_badge
from src.neurotwin.research.formulas import PHYSIOLOGICAL_FEATURES


def _render_pipeline_svg() -> str:
    """Return responsive SVG showing the end-to-end signal processing and modeling pipeline."""
    return """
    <svg width="100%" height="160" viewBox="0 0 950 160" xmlns="http://www.w3.org/2000/svg" style="background:rgba(12,21,39,0.7); border:1px solid rgba(56,189,248,0.2); border-radius:12px; margin:1rem 0;">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#38bdf8"/>
        </marker>
      </defs>
      
      <!-- Stage 1: Raw Signals -->
      <rect x="20" y="30" width="145" height="100" rx="8" fill="rgba(15,27,51,0.9)" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="92" y="55" fill="#38bdf8" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Empatica E4 Raw</text>
      <text x="92" y="75" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">EDA (4Hz) | HR (1Hz)</text>
      <text x="92" y="92" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">BVP (64Hz) | IBI</text>
      <text x="92" y="110" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">TEMP (4Hz) | ACC (32Hz)</text>

      <line x1="165" y1="80" x2="195" y2="80" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Stage 2: Cleaning & Segmentation -->
      <rect x="200" y="30" width="145" height="100" rx="8" fill="rgba(15,27,51,0.9)" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="272" y="55" fill="#38bdf8" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Preprocessing</text>
      <text x="272" y="75" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Artifact Filtering</text>
      <text x="272" y="92" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">1 Hz Resampling</text>
      <text x="272" y="110" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Protocol Stage Windows</text>

      <line x1="345" y1="80" x2="375" y2="80" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Stage 3: Feature Extraction -->
      <rect x="380" y="30" width="145" height="100" rx="8" fill="rgba(15,27,51,0.9)" stroke="#00f2fe" stroke-width="1.5"/>
      <text x="452" y="55" fill="#00f2fe" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">23 Features</text>
      <text x="452" y="75" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">HRV (RMSSD, SDNN)</text>
      <text x="452" y="92" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">EDA SCL & SCR Proxy</text>
      <text x="452" y="110" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Thermal & ACC Activity</text>

      <line x1="525" y1="80" x2="555" y2="80" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Stage 4: Standardization & Classifier -->
      <rect x="560" y="30" width="145" height="100" rx="8" fill="rgba(15,27,51,0.9)" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="632" y="55" fill="#38bdf8" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Classification</text>
      <text x="632" y="75" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Median Imputation</text>
      <text x="632" y="92" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">StandardScaler (z)</text>
      <text x="632" y="110" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">LogReg / SVM / RF</text>

      <line x1="705" y1="80" x2="735" y2="80" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Stage 5: Evaluation & Output -->
      <rect x="740" y="30" width="180" height="100" rx="8" fill="rgba(15,27,51,0.9)" stroke="#00d4aa" stroke-width="1.5"/>
      <text x="830" y="55" fill="#00d4aa" font-family="Inter, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">LOSO & Decision Gate</text>
      <text x="830" y="75" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">35 Subject-Indep Folds</text>
      <text x="830" y="92" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Platt Calibration Audit</text>
      <text x="830" y="110" fill="#94a3b8" font-family="Inter, sans-serif" font-size="10" text-anchor="middle">Stress Label (Low / High)</text>
    </svg>
    """


def render_page_how_calculated():
    """Render the feature formula validation and physiology reference page."""
    st.markdown("## 🔬 How Stress Is Calculated: Scientific Core & Mathematical Formulas")
    st.markdown(
        """
        In medical and psychological research, transparent equations, units, and physiological mechanisms are vital.
        Below is the exact processing pipeline and mathematical formulation for all **23 physiological features**
        extracted from the Empatica E4 wristband in `src/features.py`.
        """
    )

    # Pipeline Diagram
    st.markdown("### 📡 End-to-End Physiological Processing Architecture")
    st.markdown(_render_pipeline_svg(), unsafe_allow_html=True)

    st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

    # Which Features Were Used / Not Used Table
    st.markdown("### 📋 Feature Selection & Model Usage Status")
    st.markdown(
        """
        All 23 features are supplied to the classifiers. However, in accordance with research ethics and
        Stage 1 formula validation, features with known scaling discrepancies are suppressed from user-facing explanations.
        """
    )

    usage_rows = []
    for feat_id, feat in PHYSIOLOGICAL_FEATURES.items():
        status = "Used in Model & Explained" if feat.explained_in_model else "Used in Model (Suppressed from Explanations)"
        usage_rows.append(
            {
                "Feature Name": feat.name,
                "Group": feat.group,
                "Unit": feat.unit,
                "Evidence Quality": feat.evidence_label,
                "Status": status,
            }
        )
    usage_df = pd.DataFrame(usage_rows)
    st.dataframe(usage_df, use_container_width=True)

    st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

    # Filter by feature group
    st.markdown("### 📐 Mathematical Formulation & Physiological Justification Cards")

    groups = ["All Groups", "Heart Rate", "Electrodermal Activity", "Skin Temperature", "Movement / Activity", "Heart Rate Variability"]
    selected_group = st.selectbox("Filter by physiological domain:", groups, index=0)

    for feat_id, feat in PHYSIOLOGICAL_FEATURES.items():
        if selected_group != "All Groups" and feat.group != selected_group:
            continue

        badge = evidence_badge(feat.evidence_label)

        with st.expander(f"{feat.display_name} (`{feat.name}`) — {feat.group}", expanded=(selected_group != "All Groups")):
            st.markdown(f"**Group:** {feat.group} &nbsp;|&nbsp; **Unit:** `{feat.unit}` &nbsp;|&nbsp; {badge}", unsafe_allow_html=True)

            st.markdown("**Mathematical Formula:**")
            st.latex(feat.formula_latex)

            st.markdown(f"**🧠 Physiological Mechanism:**\n{feat.physiological_reasoning}")

            st.markdown(f"**⚠️ Scientific Limitations:**\n{feat.limitations}")

            st.markdown(
                f"**📚 Verified Reference:**\n"
                f"{feat.reference_citation} &nbsp; "
                f"[Clickable DOI Link]({feat.reference_doi})"
            )

