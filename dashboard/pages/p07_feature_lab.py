"""Page 7: Feature Lab & Biosignal Exploration.

Presents:
- Global feature-importance bar chart (feature_importance.csv) with physiological group colors
- Physiological correlation matrix heatmap
- Per-feature distributions separated by stress class (Low vs High)
- Feature-group contribution donut chart
- Stress trajectory across protocol stages (Baseline -> Stroop -> Rest -> Subtraction)
- 'Compare My Upload vs Dataset' percentile overlay
- Physiological feature selection justification section
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import metric_card
from src.neurotwin.estimator import GROUPS
from src.neurotwin.research.formulas import PHYSIOLOGICAL_FEATURES


def render_page_feature_lab(data_dict: dict):
    """Render the interactive Feature Lab page."""
    st.markdown("## 🧬 Feature Lab: Signal Distributions & Physiological Importance")
    st.markdown(
        """
        Explore the statistical properties, inter-feature correlations, and relative predictive power
        of the 23 physiological descriptors extracted from the Empatica E4 wristband.
        """
    )

    fi_df = data_dict.get("feature_imp")
    raw_df = data_dict.get("raw_features")

    tab_imp, tab_dist, tab_traj, tab_compare = st.tabs(
        ["1. Feature Importance & Groups", "2. Correlation & Distributions", "3. Protocol Trajectories", "4. Compare Upload vs Dataset"]
    )

    # --- TAB 1: Feature Importance ---
    with tab_imp:
        st.markdown("### 🏆 Global Feature Importance (Random Forest)")
        if fi_df is not None:
            # Merge with physiological group
            fi = fi_df.copy()
            feature_col = "feature" if "feature" in fi.columns else fi.columns[0]
            val_col = "importance" if "importance" in fi.columns else fi.select_dtypes(include="number").columns[0]

            def get_group(f_name):
                for g, cols in GROUPS.items():
                    if f_name in cols or f_name.replace("hr_diff_rms", "hr_rmssd_approx") in cols:
                        return g.title()
                return "Other"

            fi["Group"] = fi[feature_col].apply(get_group)
            fi = fi.sort_values(val_col, ascending=True).tail(15)

            col_chart, col_donut = st.columns([1.5, 1])

            with col_chart:
                fig_fi = px.bar(
                    fi,
                    x=val_col,
                    y=feature_col,
                    color="Group",
                    orientation="h",
                    color_discrete_sequence=["#00f2fe", "#00d4aa", "#f59e0b", "#f43f5e", "#a855f7"],
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10, 18, 34, 0.6)",
                    font={"color": "#f8fafc"},
                    height=420,
                    margin={"t": 30, "b": 30, "l": 150, "r": 20},
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            with col_donut:
                st.markdown("#### Importance by Modality")
                grp_sum = fi.groupby("Group")[val_col].sum().reset_index()
                fig_donut = px.pie(
                    grp_sum,
                    names="Group",
                    values=val_col,
                    hole=0.55,
                    color_discrete_sequence=["#00f2fe", "#00d4aa", "#f59e0b", "#f43f5e", "#a855f7"],
                )
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f8fafc"},
                    height=380,
                    margin={"t": 20, "b": 20, "l": 20, "r": 20},
                )
                st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Feature importance data not found.")

    # --- TAB 2: Correlation & Distributions ---
    with tab_dist:
        st.markdown("### 🔗 Physiological Feature Correlation Heatmap")
        if raw_df is not None:
            num_cols = [c for c in raw_df.select_dtypes(include="number").columns if c != "stress_level"]
            default_heatmap = [c for c in ["hr_mean", "hr_std", "eda_mean", "eda_slope", "temp_mean", "acc_mean", "hrv_rmssd", "hrv_sdnn"] if c in num_cols]
            sel_cols = st.multiselect("Select features for correlation matrix:", num_cols, default=default_heatmap)

            if len(sel_cols) >= 2:
                corr = raw_df[sel_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto",
                )
                fig_corr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10, 18, 34, 0.6)",
                    font={"color": "#f8fafc"},
                    height=450,
                    margin={"t": 30, "b": 30, "l": 40, "r": 20},
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # Feature distribution by stress class
            st.markdown("### 🎻 Feature Distribution by Stress Class (Low vs High)")
            feat_to_inspect = st.selectbox("Pick a feature to inspect distributions:", default_heatmap, index=0)
            if "stress_level" in raw_df.columns:
                plot_raw = raw_df.copy()
                plot_raw["Class"] = np.where(plot_raw["stress_level"] >= 5.0, "High Stress (>=5)", "Low Stress (<5)")
                fig_box = px.box(
                    plot_raw,
                    x="Class",
                    y=feat_to_inspect,
                    color="Class",
                    points="outliers",
                    color_discrete_sequence=["#00d4aa", "#f43f5e"],
                )
                fig_box.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10, 18, 34, 0.6)",
                    font={"color": "#f8fafc"},
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Processed features not available.")

    # --- TAB 3: Protocol Trajectories ---
    with tab_traj:
        st.markdown("### 📈 Stress Trajectory Across Protocol Stages")
        st.caption("Average self-reported stress and physiological response across the experimental protocol.")

        stage_stats_path = Path("data/processed/stress_stage_stats.csv")
        if stage_stats_path.is_file():
            stage_df = pd.read_csv(stage_stats_path)
            st.dataframe(stage_df, use_container_width=True)

            if "stage" in stage_df.columns and "mean" in stage_df.columns:
                fig_traj = px.line(
                    stage_df,
                    x="stage",
                    y="mean",
                    markers=True,
                    title="Mean Stress Level Progression Across Protocol Tasks",
                    color_discrete_sequence=["#00f2fe"],
                )
                fig_traj.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10, 18, 34, 0.6)",
                    font={"color": "#f8fafc"},
                    yaxis={"title": "Reported Stress (1–10 Scale)"},
                    height=350,
                )
                st.plotly_chart(fig_traj, use_container_width=True)
        else:
            fig_traj_png = Path("reports/figures/04_stress_trajectory_v1.png")
            if fig_traj_png.is_file():
                st.image(str(fig_traj_png), caption="Protocol Stress Trajectory (Reports)", use_container_width=True)

    # --- TAB 4: Compare Upload vs Dataset ---
    with tab_compare:
        st.markdown("### 📊 Compare Uploaded Features vs Research Population")
        uploaded_df = st.session_state.get("uploaded_df")

        if uploaded_df is not None and raw_df is not None:
            user_means = uploaded_df.mean()
            comp_rows = []

            for col in uploaded_df.columns:
                if col in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[col]):
                    val = float(user_means[col])
                    pop_series = raw_df[col].dropna()
                    pct = float((pop_series < val).mean() * 100.0)
                    pop_mean = float(pop_series.mean())
                    pop_std = float(pop_series.std()) if pop_series.std() > 0 else 1.0
                    z = (val - pop_mean) / pop_std

                    comp_rows.append(
                        {
                            "Feature": col,
                            "Your Mean Value": round(val, 3),
                            "Dataset Mean": round(pop_mean, 3),
                            "Z-Score": round(z, 2),
                            "Percentile": round(pct, 1),
                        }
                    )

            df_comp = pd.DataFrame(comp_rows)
            st.dataframe(df_comp, use_container_width=True)

            fig_pct = px.bar(
                df_comp.head(12),
                x="Percentile",
                y="Feature",
                orientation="h",
                color="Percentile",
                color_continuous_scale="Viridis",
                range_x=[0, 100],
            )
            fig_pct.add_vline(x=50, line_dash="dash", line_color="#cbd5e1", annotation_text="Median (50th %ile)")
            fig_pct.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 18, 34, 0.6)",
                font={"color": "#f8fafc"},
                height=400,
            )
            st.plotly_chart(fig_pct, use_container_width=True)
        else:
            st.info("💡 Upload your CSV or activate the Demo Participant to compare your features against the research dataset.")

    st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

    # Physiological Feature Selection Justification
    st.markdown("### 🔬 Physiological Rationale for Feature Selection")
    st.markdown(
        """
        - **Heart Rate (HR):** Acute sympathetic arousal triggers $\\beta_1$-adrenergic sinoatrial excitation, increasing cardiac frequency (`hr_mean`).
        - **Heart Rate Variability (HRV):** High-frequency beat variations (`hrv_rmssd`, `hrv_pnn50`) directly mirror vagal parasympathetic tone. Stress causes vagal withdrawal, leading to marked reductions in RMSSD.
        - **Electrodermal Activity (EDA):** Governed solely by sympathetic cholinergic innervation of eccrine sweat glands. Phasic burst count (`eda_scr_count`) and tonic level (`eda_scl_mean`) track sympathetic burst frequency without parasympathetic opposition.
        - **Skin Temperature (TEMP):** Peripheral sympathetic vasoconstriction diverts cutaneous blood flow toward core skeletal muscles, driving downward slopes (`temp_slope`).
        - **Movement (ACC):** Crucial confounder filter. Physical movement increases cardiac and electrodermal metrics; monitoring acceleration ensures metabolic demand is not misclassified as psychological distress.
        """
    )

