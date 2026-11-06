"""Page 2: Signals & Features Constellation — NeuroTwin Aurora.

Presents:
- 6-stage Journey Ribbon (at 'Detect')
- Sensor signal constellation grouped by physiological channel (EDA, HR, TEMP, ACC, HRV)
- Deterministic mini SVG bars and units
- Data quality, coverage assessment, and motion/temperature flags
- CSV feature uploader and demo participant loader
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.components_aurora import render_journey_ribbon, render_signal_constellation
from src.models import FEATURE_COLS
from src.neurotwin import DISCLAIMER
from src.neurotwin.estimator import StressEstimator


def render_page_signals(validate_fn=None, get_model_fn=None):
    """Render the Signals & Features Constellation page."""
    st.caption(f"🔬 {DISCLAIMER}")
    render_journey_ribbon("Detect")

    st.markdown("## 📡 Signals & Features Constellation")
    st.markdown(
        "Explore the 23 physiological features extracted from wearable sensors (Empatica E4), "
        "grouped into five biological channels. View signal quality, missingness, and motion flags."
    )

    # Top action bar: download template & sample
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        template_df = pd.DataFrame(columns=FEATURE_COLS)
        st.download_button(
            label="📄 Download 23-Feature CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="neurotwin_feature_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_t2:
        sample_path = Path("sample_upload.csv")
        if sample_path.is_file():
            st.download_button(
                label="🧪 Download Verified Sample CSV",
                data=sample_path.read_bytes(),
                file_name="sample_upload.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # File uploader or load demo
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a feature CSV (23 physiological columns)",
        type=["csv"],
        help="Upload a CSV with the 23 standard columns.",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["uploaded_df"] = df
            st.session_state["demo_active"] = False
            st.success(f"Loaded {len(df)} feature windows from {uploaded.name}")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

    df_active = st.session_state.get("uploaded_df")
    if df_active is None:
        # Fall back to sample_upload if present or empty
        sample_path = Path("sample_upload.csv")
        if sample_path.is_file():
            df_active = pd.read_csv(sample_path)
            st.session_state["uploaded_df"] = df_active

    if df_active is not None and not df_active.empty:
        st.markdown("---")
        st.markdown("#### 🔍 Physiological Constellation Viewer")
        window_idx = st.slider("Select Window Index", 0, max(0, len(df_active) - 1), 0)
        row = df_active.iloc[window_idx].to_dict()

        # Render signal constellation
        render_signal_constellation(row)

        # Signal Quality Assessment
        est = StressEstimator()
        quality = est.assess_quality(row)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1:
            cov_pct = int(quality.coverage * 100)
            badge_type = "badge-teal" if cov_pct >= 80 else "badge-gold"
            st.markdown(
                f"<div class='nt-card'>"
                f"<div class='layer-tag'>Feature Coverage</div>"
                f"<div style='font-size:1.6rem; font-weight:700;'>{cov_pct}%</div>"
                f"<span class='aurora-badge {badge_type}'>{'Sufficient' if not quality.insufficient else 'Low Coverage'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with q2:
            flags_str = ", ".join(quality.flags) if quality.flags else "None (Signals Steady)"
            st.markdown(
                f"<div class='nt-card'>"
                f"<div class='layer-tag'>Artifact & Motion Flags</div>"
                f"<div style='font-size:0.95rem; font-weight:600; margin-top:0.4rem;'>{flags_str}</div>"
                f"<div style='font-size:0.75rem; color:#A9B8DC;'>Movement & thermal flags</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with q3:
            missing_grps = ", ".join(quality.groups_missing) if quality.groups_missing else "None (All 5 Channels Present)"
            st.markdown(
                f"<div class='nt-card'>"
                f"<div class='layer-tag'>Missing Channels</div>"
                f"<div style='font-size:0.95rem; font-weight:600; margin-top:0.4rem;'>{missing_grps}</div>"
                f"<div style='font-size:0.75rem; color:#A9B8DC;'>Required for full appraisal</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Upload a feature CSV or click 'Use Demo Participant' on Home to explore the signal constellation.")

