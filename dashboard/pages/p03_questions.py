"""Page 3: Quick Questions (Psychometrics & Quality of Life) — NeuroTwin Aurora.

Implements:
- 6-stage Journey Ribbon (at 'Ask')
- Perceived Stress Scale (PSS-10, Cohen et al. 1983) with reverse items (4, 5, 7, 8)
- WHO-5 Well-Being Index (0–100 scale, Topp et al. 2015)
- Clear provenance separation: all inputs are strictly Reported layer data
- Local SQLite storage with user consent gating
- Skippable flow with instant navigation
"""

import streamlit as st

from dashboard.components_aurora import render_journey_ribbon
from src.neurotwin import DISCLAIMER
from src.neurotwin.research.scales import (
    PSS_ITEMS,
    PSS_OPTIONS,
    WHO5_ITEMS,
    WHO5_OPTIONS,
    save_questionnaire_to_store,
    score_pss10,
    score_qol,
)
from src.neurotwin.ui import active_store


def render_page_questions():
    """Render the Quick Questions psychometric evaluation page."""
    st.caption(f"🔬 {DISCLAIMER}")
    render_journey_ribbon("Ask")

    st.markdown("## 📝 Quick Questions: Subjective Stress & Well-Being")
    st.markdown(
        """
        Wearable sensors reflect momentary autonomic shifts (such as sweat gland activity or heart-rate variations),
        whereas lived psychological strain depends on cognitive perception. Completing these two validated
        instruments helps ground biosignal patterns against your subjective experience.

        *All questions are completely optional and skippable. Your answers are stored locally as **Reported**
        layer data and are never conflated with raw sensor measurements.*
        """
    )

    tab_pss, tab_who5 = st.tabs(["1. Perceived Stress Scale (PSS-10)", "2. WHO-5 Well-Being Index (QoL)"])

    pss_answers = {}
    with tab_pss:
        st.markdown("#### PSS-10 (Past 30 Days Recall)")
        st.caption(
            "The questions in this scale ask about your feelings and thoughts during the last month. "
            "For each question, indicate how often you felt that way. Citing Cohen, Kamarck & Mermelstein (1983)."
        )

        for item in PSS_ITEMS:
            idx = item["id"]
            rev_label = " *(reverse-scored)*" if item["reverse"] else ""
            st.markdown(f"**{idx}. {item['text']}**{rev_label}")
            val = st.radio(
                f"Item {idx} response",
                options=[opt[0] for opt in PSS_OPTIONS],
                format_func=lambda x: dict(PSS_OPTIONS)[x],
                index=1 if item["reverse"] else 2,
                key=f"pss_q_{idx}",
                horizontal=True,
                label_visibility="collapsed",
            )
            pss_answers[idx] = val
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    who5_answers = {}
    with tab_who5:
        st.markdown("#### WHO-5 Well-Being Index (Past 2 Weeks)")
        st.caption(
            "Please indicate for each of the five statements which is closest to how you have been feeling over the last two weeks. "
            "Scores range from 0 to 100, where higher indicates better psychological well-being. Citing Topp et al. (2015)."
        )

        for item in WHO5_ITEMS:
            idx = item["id"]
            st.markdown(f"**{idx}. {item['text']}**")
            val = st.radio(
                f"WHO-5 Item {idx}",
                options=[opt[0] for opt in WHO5_OPTIONS],
                format_func=lambda x: dict(WHO5_OPTIONS)[x],
                index=2,
                key=f"who5_q_{idx}",
                horizontal=True,
                label_visibility="collapsed",
            )
            who5_answers[idx] = val
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Consent and submission
    st.markdown("##### 🔒 Privacy & Local Consent")
    consent_check = st.checkbox(
        "I consent to store these questionnaire responses locally on my machine in the Reported database table.",
        value=True,
        help="NeuroTwin stores data exclusively in a local SQLite file. No data ever leaves your computer.",
    )

    col_sub1, col_sub2 = st.columns([1.2, 1])
    with col_sub1:
        if st.button("💾 Save Answers & Calculate Results", type="primary", use_container_width=True):
            pss_res = score_pss10(pss_answers)
            qol_res = score_qol(who5_answers)

            st.session_state["pss_result"] = pss_res
            st.session_state["qol_result"] = qol_res

            if consent_check:
                try:
                    store = active_store()
                    save_questionnaire_to_store(store, "local_user", pss_res, qol_res)
                    st.toast("Questionnaire responses stored locally in Reported table.", icon="💾")
                except Exception as e:
                    st.warning(f"Note on storage: {e}")

            st.session_state["nav_selection"] = "Stress Result"
            st.rerun()

    with col_sub2:
        if st.button("⏭️ Skip Questionnaire & View Results", use_container_width=True):
            st.session_state["pss_result"] = None
            st.session_state["qol_result"] = None
            st.session_state["nav_selection"] = "Stress Result"
            st.rerun()
