"""Page 1: Project Home & Orientation — NeuroTwin Aurora.

Presents:
- Project title & non-diagnostic mission statement
- 6-stage Journey Ribbon (Detect -> Explain -> Ask -> Understand -> Recommend -> Learn)
- Breathing Orb centerpiece (decorative SVG, still when abstaining)
- Three-Layer provenance card (Measured, Estimated, Reported)
- Empirical dataset summary cards
- Quick action CTA buttons: 'Upload your CSV' and 'Use demo participant'
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.components_aurora import render_journey_ribbon, render_three_layer_card
from dashboard.theme_prism import render_flow_strip, render_hologram_twin, render_ticker
from src.neurotwin import DISCLAIMER


def render_page_home():
    """Render the Aurora Home overview page."""
    st.caption(f"🔬 {DISCLAIMER}")

    # Journey progress ribbon
    render_journey_ribbon("Detect")

    # Top Hero Section
    col_hero_text, col_hero_visual = st.columns([1.5, 1])

    with col_hero_text:
        st.markdown(
            """
            <div style="margin-bottom:0.8rem;">
                <span class="aurora-badge badge-teal">Research Prototype</span>
                <span class="aurora-badge badge-violet">Leave-One-Subject-Out Validated</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<span class='prism-title'>NEUROTWIN AI</span>", unsafe_allow_html=True)
        st.markdown(
            "<div class='prism-sub'>Physiological Digital Twin &amp; <b>Adaptive Stress Management</b></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **NeuroTwin AI** is an explainable personal physiological digital twin designed to model acute
            physiological responses from wearable biosignals and connect them with self-appraised stress and
            well-being.

            Rather than treating stress as an alarm or a single numeric percentage, NeuroTwin embraces
            uncertainty and separates what was **Measured** on hardware, what was **Estimated** by a statistical model,
            and what was **Reported** in your own voice.
            """
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("📤 Upload Your CSV", type="primary", use_container_width=True):
                st.session_state["nav_selection"] = "Upload & Validate"
                st.rerun()
        with col_btn2:
            if st.button("🧪 Use Demo Participant", use_container_width=True):
                demo_path = Path("sample_upload.csv")
                if not demo_path.is_file():
                    demo_path = Path(__file__).resolve().parents[2] / "sample_upload.csv"
                if demo_path.is_file():
                    demo_df = pd.read_csv(demo_path)
                    st.session_state["uploaded_df"] = demo_df
                    st.session_state["demo_active"] = True
                    st.session_state["nav_selection"] = "Upload & Validate"
                    st.rerun()

    with col_hero_visual:
        calm_mode = st.session_state.get("aurora_calm_mode", False)
        theme = st.session_state.get("aurora_theme", "dark")
        render_hologram_twin(calm_mode=calm_mode, theme=theme)

    render_ticker()
    render_flow_strip()

    st.markdown("---")

    # Three-Layer Provenance Section
    st.markdown("#### 🏛️ Three-Layer Architecture")
    st.markdown(
        "To prevent mistaking statistical inferences for physical measurements or medical diagnoses, "
        "every finding is strictly grouped into one of three layers:"
    )

    render_three_layer_card(
        measured_content=(
            "<b>Hardware Sensor Signals:</b> Heart rate (HR), electrodermal activity (EDA), "
            "skin temperature, accelerometry motion, and beat intervals recorded on an Empatica E4 wristband."
        ),
        estimated_content=(
            "<b>Hypothetical Statistical Inference:</b> Machine-learning pattern classification trained on "
            "laboratory tasks. Subject to uncertainty mist; probabilities are hidden unless calibration beats the training prior."
        ),
        reported_content=(
            "<b>Human Self-Appraisal:</b> The person's own subjective check-in ratings, PSS-10 perceived "
            "stress, WHO-5 well-being indices, and personal coping notes."
        ),
    )

    # Scientific Snapshot Metric Cards
    st.markdown("#### 📊 Empirical Dataset Snapshot (PhysioNet v1.0.1)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            "<div class='nt-card' style='text-align:center;'>"
            "<div class='layer-tag'>Cohort</div>"
            "<div style='font-size:1.8rem; font-weight:700; color:#EAF1FF;'>35</div>"
            "<div style='font-size:0.75rem; color:#A9B8DC;'>Healthy volunteers (18–30)</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='nt-card' style='text-align:center;'>"
            "<div class='layer-tag'>Windows</div>"
            "<div style='font-size:1.8rem; font-weight:700; color:#EAF1FF;'>263</div>"
            "<div style='font-size:0.75rem; color:#A9B8DC;'>Standard protocol segments</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='nt-card' style='text-align:center;'>"
            "<div class='layer-tag'>Lower Stress</div>"
            "<div style='font-size:1.8rem; font-weight:700; color:#7BE3A5;'>162</div>"
            "<div style='font-size:0.75rem; color:#A9B8DC;'>≈ 61.6% (majority class)</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            "<div class='nt-card' style='text-align:center;'>"
            "<div class='layer-tag'>Higher Stress</div>"
            "<div style='font-size:1.8rem; font-weight:700; color:#FFB38A;'>101</div>"
            "<div style='font-size:0.75rem; color:#A9B8DC;'>≈ 38.4% (minority class)</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Honest evaluation notice
    st.markdown(
        """
        <div class="nt-card" style="border-left: 3px solid #38E1CC; margin-top: 1.2rem;">
            <div style="font-size:0.9rem; font-weight:600; color:#38E1CC; margin-bottom:0.3rem;">
                Scientific Integrity & Abstention Policy
            </div>
            <div style="font-size:0.83rem; color:#A9B8DC; line-height:1.5;">
                NeuroTwin operates under strict Leave-One-Subject-Out (LOSO) cross-validation.
                Our honest benchmark: Logistic Regression achieves an accuracy of <b>0.570</b> vs a naive majority guessing
                baseline of <b>0.616</b> (AUC 0.484, v2 window-B protocol). When physiological evidence is incomplete or ambiguous,
                the system deliberately abstains rather than producing unwarranted estimates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
