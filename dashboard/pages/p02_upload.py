"""Page 2: Upload & Sensor Data Validation.

Implements:
- Drag-and-drop feature CSV upload
- Rigorous schema validation via validate_uploaded_csv
- Comprehensive Data-Quality Assessment Card (coverage %, missing per group, motion flags)
- Downloadable 23-feature template and sample upload CSV
- Graceful error handling for missing columns, non-numeric values, or corrupted files
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.theme import metric_card
from src.models import FEATURE_COLS
from src.neurotwin.estimator import GROUPS, StressEstimator


def render_page_upload(validate_fn, get_model_fn):
    """Render the Upload and Validation interface."""
    st.markdown("## 📤 Upload & Validate Wearable Features")
    st.markdown(
        """
        Upload a model-ready feature CSV derived from your wearable recording (Empatica E4 or equivalent).
        The file must contain the **23 standardized physiological feature columns** describing heart rate,
        electrodermal activity, skin temperature, accelerometry, and heart-rate variability.
        """
    )

    # Top helper actions (download template & sample)
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        template_df = pd.DataFrame(columns=FEATURE_COLS)
        st.download_button(
            label="📄 Download 23-Feature CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="neurotwin_feature_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Empty CSV with the 23 exact feature column names.",
        )
    with col_t2:
        sample_path = Path("sample_upload.csv")
        if not sample_path.is_file():
            sample_path = Path(__file__).resolve().parents[2] / "sample_upload.csv"
        if sample_path.is_file():
            sample_bytes = sample_path.read_bytes()
            st.download_button(
                label="🧪 Download Verified Sample CSV (10 Windows)",
                data=sample_bytes,
                file_name="sample_upload.csv",
                mime="text/csv",
                use_container_width=True,
                help="Pre-packaged valid feature recording ready for instant testing.",
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # File uploader widget
    uploaded = st.file_uploader(
        "Drop your physiological feature CSV here",
        type=["csv"],
        help=f"Accepts CSV with {len(FEATURE_COLS)} feature columns. Missing values will be imputed by median.",
    )

    df_raw = None
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to parse CSV: {e}. Please ensure it is a valid comma-separated text file.")
            return
    elif st.session_state.get("uploaded_df") is not None:
        df_raw = st.session_state.get("raw_uploaded_df", st.session_state.get("uploaded_df"))
        if st.session_state.get("demo_active", False):
            st.info("ℹ️ Currently analyzing loaded **Demo Participant** data.")
        else:
            st.info("ℹ️ Currently analyzing loaded dataset.")

    if df_raw is not None:
        st.markdown(f"**Loaded Dataset:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

        # Preview table
        with st.expander("🔍 Preview First 5 Feature Rows", expanded=False):
            st.dataframe(df_raw.head(5), use_container_width=True)

        # Validate with the model's exact target columns
        try:
            pipeline, expected_cols, train_info = get_model_fn()
        except Exception:
            pipeline, expected_cols, train_info = None, FEATURE_COLS, {}

        valid, msg, df_clean = validate_fn(df_raw, expected_cols=expected_cols)

        if not valid:
            st.error(f"⚠️ Validation Failed:\n\n{msg}")
            st.info("Tip: You can download the 23-Feature CSV Template above to inspect the expected column headers.")
            return

        st.success(f"✅ {msg}")
        st.session_state["uploaded_df"] = df_clean
        st.session_state["raw_uploaded_df"] = df_raw

        # --- Data Quality Assessment Card ---
        st.markdown("### 🛡️ Measured Layer: Data Quality Card")
        estimator = StressEstimator()

        first_row = df_clean.iloc[0].to_dict()
        q = estimator.assess_quality(first_row)

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            cov_pct = f"{q.coverage:.0%}"
            st.markdown(metric_card("Coverage", cov_pct, f"{len(q.missing)} missing cols"), unsafe_allow_html=True)
        with q2:
            motion_status = "High Movement" if "movement_high" in q.flags else "Rest / Stable"
            st.markdown(metric_card("Movement Flag", motion_status, "ACC 95th %ile check"), unsafe_allow_html=True)
        with q3:
            temp_status = "Unusual" if "skin_temperature_unusual" in q.flags else "Normal Range"
            st.markdown(metric_card("Skin Temp Flag", temp_status, "30°C - 36°C typical"), unsafe_allow_html=True)
        with q4:
            eval_gate = "Insufficient" if q.insufficient else "Passed Gate"
            st.markdown(metric_card("Quality Gate", eval_gate, "Minimum signal standards"), unsafe_allow_html=True)

        # Missingness per signal group
        st.markdown("##### Feature Group Integrity Breakdown")
        grp_data = []
        for g_name, g_cols in GROUPS.items():
            sub = df_clean[[c for c in g_cols if c in df_clean.columns]]
            miss_pct = (sub.isnull().sum().sum() / max(1, sub.size)) * 100.0
            status = "100% Intact" if miss_pct == 0 else f"{miss_pct:.1f}% Missing"
            grp_data.append({"Signal Group": g_name.title(), "Columns": len(g_cols), "Status": status})

        st.dataframe(pd.DataFrame(grp_data), use_container_width=True)

        if q.flags:
            st.warning("⚠️ **Active Sensor Quality Flags:** " + ", ".join(q.flags))
            if "movement_high" in q.flags:
                st.caption(
                    "Physical movement increases heart rate and electrodermal arousal independently of mental stress."
                )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Navigation CTAs
        c_next1, c_next2 = st.columns(2)
        with c_next1:
            if st.button("Proceed to Quick Questions (Recommended) ➔", type="primary", use_container_width=True):
                st.session_state["nav_selection"] = "Quick Questions"
                st.rerun()
        with c_next2:
            if st.button("Skip to Stress Results Directly ➔", use_container_width=True):
                st.session_state["nav_selection"] = "Stress Result"
                st.rerun()

