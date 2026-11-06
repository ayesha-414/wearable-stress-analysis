"""Page 7: Research Background, Methods & Verified References — NeuroTwin Aurora.

Presents:
- Scientific background & provenance boundaries
- Verified citations with DOIs from docs/neurotwin/model_investigation_references.csv
- Ground truth definitions and protocol details
- Scientific integrity and software verification statements
"""

import streamlit as st

from dashboard.components_aurora import render_what_model_cannot_do
from src.neurotwin import DISCLAIMER


def render_page_about():
    """Render the About, Methods & Verified References page."""
    st.caption(f"🔬 {DISCLAIMER}")

    st.markdown("## 📚 Research Background, Methods & Citations")
    st.markdown(
        "**NeuroTwin AI** is an MSc Software Systems research prototype investigating whether acute "
        "physiological stress can be reliably, explainably, and honestly modeled from consumer wearable sensors."
    )

    # Limitation reminder
    render_what_model_cannot_do()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    tab_meth, tab_refs = st.tabs(["1. Scientific Methodology", "2. Verified Literature & DOIs"])

    with tab_meth:
        st.markdown("#### Experimental Design & Evaluation Protocol")
        st.markdown(
            """
            - **Dataset:** PhysioNet Wearable Device Dataset v1.0.1 (Empatica E4, 35 modelled participants, ages 18–30).
            - **Sensors:** 4-channel photoplethysmography (HR 1 Hz), Electrodermal Activity (EDA 4 Hz), Skin Temperature (TEMP 4 Hz), 3-axis Accelerometry (ACC 32 Hz).
            - **Ground Truth Target:** Post-stage subjective self-report on a 1–10 Likert scale, dichotomized at threshold $\\ge 5.0$.
            - **Validation Rigor:** Strict Leave-One-Subject-Out (LOSO) cross-validation. All scalers, imputers, feature selectors, and Platt calibrators are fitted exclusively on training subjects.
            - **Duration Standardization:** Primary analysis uses duration-matched 30 s windows anchored at stage start (Variant B) to eliminate duration artifacts.
            """
        )

    with tab_refs:
        st.markdown("#### Key Literature (DOIs Verified)")
        st.caption("All citations are indexed in `docs/neurotwin/model_investigation_references.csv`.")

        refs = [
            ("Saeb et al. (2017)", "The need to approximate the use-case in clinical machine learning.", "GigaScience 6(5):gix019", "https://doi.org/10.1093/gigascience/gix019"),
            ("Little et al. (2017)", "Using and understanding cross-validation strategies.", "GigaScience 6(5):gix020", "https://doi.org/10.1093/gigascience/gix020"),
            ("Varoquaux (2018)", "Cross-validation failure: small sample sizes lead to large error bars.", "NeuroImage 180:68-77", "https://doi.org/10.1016/j.neuroimage.2017.06.061"),
            ("Vabalas et al. (2019)", "Machine learning algorithm validation with a limited sample size.", "PLoS ONE 14(11):e0224365", "https://doi.org/10.1371/journal.pone.0224365"),
            ("Van Calster et al. (2019)", "Calibration: the Achilles heel of predictive analytics.", "BMC Med 17:230", "https://doi.org/10.1186/s12916-019-1466-7"),
            ("El-Yaniv & Wiener (2010)", "On the foundations of noise-free selective classification.", "J Mach Learn Res 11:1605-1641", "https://jmlr.org/papers/v11/el-yaniv10a.html"),
            ("Field & Welsh (2007)", "Bootstrapping clustered data.", "J R Stat Soc B 69(3):369-390", "https://doi.org/10.1111/j.1467-9868.2007.00593.x"),
            ("Schmidt et al. (2018)", "Introducing WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection.", "ICMI 2018", "https://doi.org/10.1145/3242969.3242985"),
        ]

        for auth, title, ven, url in refs:
            st.markdown(
                f"""
                <div class="nt-card" style="padding: 0.8rem 1rem; margin-bottom: 0.6rem;">
                    <b>{auth}</b> — <i>{title}</i><br>
                    <span style="font-size:0.8rem; color:#A9B8DC;">{ven}</span> &nbsp;|&nbsp;
                    <a href="{url}" target="_blank" style="font-size:0.8rem; color:#38E1CC; text-decoration:none;">View Publication ↗</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

