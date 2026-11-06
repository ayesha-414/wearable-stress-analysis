"""Page 5: Adaptive Recommendations & Guided Interventions — NeuroTwin Aurora.

Presents:
- 6-stage Journey Ribbon (at 'Recommend')
- Guided Breathing Visual (Inhale 4s, Hold 4s, Exhale 6s) with safety notice
- Transparent, rule-based adaptive recommendation cards
- 'Why this was suggested' expandable drawer
- Real-time feedback buttons (updating the twin's lightweight preference store)
- Non-clinical disclaimer
"""

import streamlit as st

from dashboard.components_aurora import (
    render_breathing_guide_widget,
    render_journey_ribbon,
)
from src.neurotwin import DISCLAIMER
from src.neurotwin.catalogue import load_catalogue
from src.neurotwin.recommender import Context, recommend
from src.neurotwin.ui import active_store


def render_page_interventions():
    """Render the Adaptive Recommendations and Guided Interventions page."""
    st.caption(f"🔬 {DISCLAIMER}")
    render_journey_ribbon("Recommend")

    st.markdown("## 🌿 Adaptive Suggestions & Coping Support")
    st.markdown(
        "Explore gentle, transparent coping suggestions matched to your reported context and "
        "physiological state. Every suggestion is rule-based and adapts to your personal feedback."
    )

    calm_mode = st.session_state.get("aurora_calm_mode", False)

    # 1. Guided Paced Breathing Visual
    with st.container():
        render_breathing_guide_widget(calm_mode=calm_mode)

    st.markdown("---")

    # 2. Adaptive Recommendation Cards
    st.markdown("### 💡 Context-Aware Suggestions")
    st.caption("Suggestions are sorted by category match and your prior feedback. They do not constitute clinical interventions.")

    # Context filters
    c_flt1, c_flt2 = st.columns(2)
    with c_flt1:
        current_mood = st.selectbox(
            "Current feeling",
            options=["overwhelmed", "tense", "tired", "low", "calm_ok", "other"],
            format_func=lambda x: {
                "overwhelmed": "Feeling overwhelmed",
                "tense": "Feeling physically tense",
                "tired": "Feeling tired or drained",
                "low": "Feeling down or low",
                "calm_ok": "Feeling calm or okay",
                "other": "Other feeling",
            }.get(x, x),
            index=1,
        )
    with c_flt2:
        reason_focus = st.selectbox(
            "Primary context",
            options=["work", "studies", "sleep", "relationships", "health", "unknown"],
            format_func=lambda x: x.capitalize(),
            index=0,
        )

    store = active_store()
    user_id = "local_user"
    prefs = store.get_preferences(user_id) if hasattr(store, "get_preferences") else {}

    cat = load_catalogue()
    ctx = Context(mood=current_mood, reason=reason_focus)
    suggestions = recommend(cat, ctx, preferences=prefs)

    for i, s in enumerate(suggestions[:4]):
        item = s.item
        st.markdown(
            f"""
            <div class="rec-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; font-size:1.05rem; color:#EAF1FF;">{item.title}</div>
                    <span class="aurora-badge badge-teal">{item.category.replace('_', ' ').capitalize()}</span>
                </div>
                <div style="font-size:0.88rem; color:#A9B8DC; margin:0.4rem 0 0.6rem 0;">
                    {item.summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Why suggested & feedback controls
        with st.expander(f"ℹ️ Why this was suggested & details for '{item.title}'"):
            st.markdown(f"**Description:** {item.detail}")
            st.markdown("**Transparent Match Reasons:**")
            for r in s.reasons:
                st.markdown(f"• {r}")

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            st.markdown("**Was this suggestion helpful?**")
            fb_col1, fb_col2, _ = st.columns([1, 1, 3])
            with fb_col1:
                if st.button("👍 Yes, helpful", key=f"fb_yes_{item.id}_{i}"):
                    try:
                        store.record_feedback(user_id, item.id, helped="yes", note="helpful in session")
                        st.toast("Thank you. The twin noted your positive preference.", icon="✨")
                    except Exception as e:
                        st.info(f"Feedback logged: {e}")
            with fb_col2:
                if st.button("👎 Not for me", key=f"fb_no_{item.id}_{i}"):
                    try:
                        store.record_feedback(user_id, item.id, helped="no", note="not helpful in session")
                        st.toast("Preference updated. This suggestion will be downweighted.", icon="👌")
                    except Exception as e:
                        st.info(f"Feedback logged: {e}")

    st.markdown("---")
    st.caption("🔒 Preferences are stored exclusively in your local session or local SQLite database per your consent settings.")

