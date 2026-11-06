"""Streamlit pages for the NeuroTwin support journey:

    Detect -> Explain -> Ask -> Understand -> Recommend -> Learn

Ten pages, all in supportive, non-diagnostic language. Every page keeps three things apart:
MEASURED (what the sensor recorded), ESTIMATED (what the model output) and REPORTED (what the
person said). Demo windows come from a laboratory dataset and are labelled as such.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models import STRESS_THRESHOLD

from . import DISCLAIMER, config, patterns, safety, service
from .catalogue import load_catalogue
from .demo import DEMO_NOTICE, DemoData
from .dialogue import CheckInSession, NODES
from .estimator import FEATURE_LABELS, GROUPS, StressEstimator
from .profile import BASELINE_NOTE, build_profile
from .recommender import Context
from .store import ConsentError, Store

USER_ID = "local_user"
JOURNEY = "Detect → Explain → Ask → Understand → Recommend → Learn"
SUPPORT_PAGES = [
    "Home & Status", "Signals & Features", "Explain the Estimate", "Check-in",
    "Possible Stressors", "Recommendations", "Twin Profile", "History & Patterns",
    "Coping Feedback", "Evaluation & Transparency",
]
REASON_LABELS = {"work": "Work", "studies": "Studies", "relationships": "Relationships",
                 "health": "Health", "finances": "Money", "sleep": "Lack of sleep",
                 "other": "Something else", "unknown": "I don't know"}
MOOD_LABELS = {"calm_ok": "Calm or okay", "tense": "Tense or uneasy", "overwhelmed": "Overwhelmed",
               "tired": "Tired or drained", "low": "Low or down", "other": "Something else",
               "skip": "I'd rather not say"}


# ============================================================================ state & stores
@st.cache_resource(show_spinner="Preparing the estimator…")
def get_estimator() -> StressEstimator:
    return StressEstimator()


def get_demo() -> DemoData:
    return DemoData(get_estimator())


@st.cache_data(show_spinner=False)
def _cached_simulation() -> dict:
    from . import simulation
    return simulation.run()


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("nt_mem_store", Store(":memory:"))
    ss.setdefault("nt_region", None)
    ss.setdefault("nt_declined_items", set())
    ss.setdefault("nt_declined_tags", set())


def active_store() -> Store:
    """Session-only memory store by default; on-disk store only with explicit consent."""
    ss = st.session_state
    persist = bool(ss.get("nt_persist", False))
    ft = bool(ss.get("nt_c_free_text", False))
    if persist:
        if "nt_disk_store" not in ss:
            ss["nt_disk_store"] = Store(config.PROFILE_DB)
        store = ss["nt_disk_store"]
        store.ensure_user(USER_ID, ss["nt_region"])
        store.set_consent(
            USER_ID, store_checkins=bool(ss.get("nt_c_checkins", False)),
            store_stressors=bool(ss.get("nt_c_stressors", False)), store_free_text=ft,
            store_feedback=bool(ss.get("nt_c_feedback", False)),
            prompts_enabled=bool(ss.get("nt_c_prompts", False)))
    else:
        store = ss["nt_mem_store"]
        store.ensure_user(USER_ID, ss["nt_region"])
        store.set_consent(USER_ID, store_checkins=True, store_stressors=True, store_feedback=True,
                          store_free_text=ft, prompts_enabled=False)
    return store


def _banner() -> None:
    st.caption(f"🔬 {DISCLAIMER}")


def _layer_header(icon: str, title: str, caption: str) -> None:
    st.markdown(f"#### {icon} {title}")
    st.caption(caption)


# ============================================================================ sidebar
def render_sidebar_support() -> None:
    """Always-visible help control plus privacy settings (added to the sidebar)."""
    _init_state()
    with st.sidebar:
        with st.expander("🆘 Talk to someone now", expanded=False):
            st.write("If you are in danger or thinking about harming yourself, please reach out to a real "
                     "person now.")
            render_resources(st.session_state["nt_region"])
        with st.expander("🔒 Privacy & consent", expanded=False):
            regions = ["Not set"] + safety.available_regions()
            cur = st.session_state["nt_region"]
            choice = st.selectbox("Your region (for support numbers)", regions,
                                  index=regions.index(cur) if cur in regions else 0, key="nt_region_sel")
            st.session_state["nt_region"] = None if choice == "Not set" else choice
            st.checkbox("Allow typing free text (never analysed; screened only for safety words)",
                        key="nt_c_free_text", value=False)
            persist = st.checkbox("Save my data on this device", key="nt_persist", value=False,
                                  help="Off = session-only: everything is discarded when you close the page.")
            if persist:
                st.checkbox("Save check-ins", key="nt_c_checkins", value=False)
                st.checkbox("Save reasons I choose", key="nt_c_stressors", value=False)
                st.checkbox("Save my feedback", key="nt_c_feedback", value=False)
                st.checkbox("Allow gentle prompts (limited, opt-in)", key="nt_c_prompts", value=False)
                st.caption("Stored locally in data/neurotwin_profile.db, unencrypted. Prototype only.")
            else:
                st.caption("Session-only mode: nothing is written to disk.")


def render_resources(region: str | None) -> None:
    info = safety.get_resources(region)
    if info["needs_region"]:
        st.info("Choose your region under Privacy & consent to see local numbers.")
    for r in info["resources"]:
        line = f"**{r['name']}**"
        if r.get("phone"):
            line += f" — {r['phone']}"
        st.markdown(line)
        if r.get("notes"):
            st.caption(r["notes"])
        if r.get("source_url"):
            st.caption(f"[Source]({r['source_url']})")
    if info["unverified"]:
        st.caption("These details have not yet been verified by the project owner. Please double-check them.")
    st.caption(safety.SUPPORT_NOTE)


# ============================================================================ window picker
def _select_window(demo: DemoData, key: str):
    st.caption(f"ℹ️ {DEMO_NOTICE}")
    subs = demo.participants()
    sel = st.session_state.get("nt_sel", (subs[0], 0))
    c1, c2 = st.columns(2)
    subject = c1.selectbox("Demo participant", subs, index=subs.index(sel[0]) if sel[0] in subs else 0,
                           key=f"{key}_sub")
    wins = demo.windows(subject)
    labels = [f"{r.stage}" for r in wins.itertuples()]
    widx = min(sel[1] if sel[0] == subject else 0, len(labels) - 1)
    stage = c2.selectbox("Recording window", labels, index=widx, key=f"{key}_win")
    st.session_state["nt_sel"] = (subject, labels.index(stage))
    index = wins.index[labels.index(stage)]
    return index, subject, stage


# ============================================================================ 1 Home
def page_home() -> None:
    _init_state()
    st.subheader("🏠 Home — where things stand")
    _banner()
    st.markdown(f"**Your journey:** {JOURNEY}")
    est, demo = get_estimator(), get_demo()
    index, subject, stage = _select_window(demo, "home")
    res = est.estimate_for_window(index)
    c1, c2, c3 = st.columns(3)
    with c1:
        _layer_header("📡", "Measured", "What the wristband recorded")
        st.metric("Signal coverage", f"{res.quality.coverage:.0%}")
        if res.quality.flags:
            for f in res.quality.flags:
                st.caption("• " + f.replace("_", " "))
        else:
            st.caption("No quality flags.")
    with c2:
        _layer_header("🤖", "Estimated", "What the model output (it can be wrong)")
        if res.abstained:
            st.warning("Not enough reliable signal to say anything about stress for this window.")
        else:
            st.write(f"The model sees **{res.label}**.")
            st.write(f"Evidence level: **{res.evidence_level}**")
            if res.probability_shown is None:
                st.caption("No percentage is shown, because the model's probabilities were not reliable on "
                           "unseen participants.")
        if res.caveats:
            st.caption(res.caveats[0])
    with c3:
        _layer_header("🗣️", "Reported", "What a person said")
        lab = demo.lab_report(index)
        if lab["rating"] is not None:
            st.write(f"The volunteer rated their stress **{lab['rating']:.0f}/10** at this point in the "
                     "laboratory protocol.")
        rep = [r for r in active_store().list_reported(USER_ID) if r["kind"] == "stress_rating"]
        if rep:
            st.write(f"Your latest check-in rating: **{rep[-1]['value']}/10**")
        else:
            st.caption("You have not shared a rating in this app yet.")
    st.divider()
    st.info("These three are different things and are never merged. The estimate cannot tell you why your "
            "signals changed, and it is not a measure of how you feel. Open **Check-in** from the sidebar "
            "whenever you would like to talk it through.")


# ============================================================================ 2 Signals & features
def page_signals() -> None:
    _init_state()
    st.subheader("📈 Signals & Features")
    _banner()
    est, demo = get_estimator(), get_demo()
    index, subject, stage = _select_window(demo, "sig")
    vals = demo.measured(index)
    base = demo.baseline(subject)
    st.caption("Measured features for this window. 'Your baseline' is the same participant's own recorded "
               "baseline window (descriptive only).")
    st.caption(BASELINE_NOTE)
    rows = []
    for g, cols in GROUPS.items():
        for c in cols:
            v = vals.get(c)
            b = base["values"].get(c) if base.get("available") else None
            rows.append({"Group": g, "Feature": FEATURE_LABELS.get(c, c), "Value": v, "Your baseline": b,
                         "Difference": (None if v is None or b is None else v - b)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    q = est.assess_quality(vals)
    st.markdown("**Quality check**")
    st.write(f"{q.coverage:.0%} of features measured." +
             (f" Missing: {', '.join(FEATURE_LABELS.get(m, m) for m in q.missing)}." if q.missing else ""))
    for f in q.flags:
        st.caption("• " + f.replace("_", " "))
    st.caption("Note: 'beat-to-beat heart-rate change (HR-difference RMS)' is computed from the heart-rate "
               "series and is not the same as conventional beat-interval RMSSD. The LF/HF value is under review "
               "(see feature_formula_validation.md).")


# ============================================================================ 3 Explain
def page_explain() -> None:
    _init_state()
    st.subheader("🔍 Explain the Estimate")
    _banner()
    est, demo = get_estimator(), get_demo()
    index, subject, stage = _select_window(demo, "exp")
    res = est.estimate_for_window(index)
    if res.abstained:
        st.warning("The model is not giving an estimate for this window.")
        for r in res.quality.reasons:
            st.write("• " + r)
    else:
        for s in res.sentences:
            st.write(s)
        usable = [c for c in res.contributions if c.explained][:8]
        if usable:
            fig = go.Figure(go.Bar(
                x=[c.contribution for c in usable][::-1], y=[c.label for c in usable][::-1],
                orientation="h",
                marker_color=["#e07a5f" if c.contribution > 0 else "#3d85c6" for c in usable][::-1]))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                              title="Model weights for this window (toward 'higher' ← 0 → toward 'lower')",
                              xaxis_title="Contribution (log-odds)")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### Please keep in mind")
    for c in res.caveats:
        st.write("• " + c)
    st.markdown("##### Other things that can look like this in the signals")
    st.write("Physical activity or movement, a warm or cold environment, caffeine, illness, excitement, "
             "a loose sensor, and normal day-to-day differences between people.")
    st.markdown("##### Association is not cause")
    st.write("The model learned statistical patterns in a small laboratory dataset. It cannot tell you why a "
             "signal changed, and it does not know what is happening in your life.")
    st.divider()
    st.markdown("##### Does this match your experience?")
    c1, c2, c3 = st.columns(3)
    for col, (val, label) in zip((c1, c2, c3), (("matches", "It matches"), ("doesnt_match", "It doesn't match"),
                                                ("not_sure", "Not sure"))):
        if col.button(label, key=f"agree_{val}"):
            try:
                active_store().add_reported(USER_ID, "estimate_agreement", val, session_id="explain-page")
                st.success("Thanks. Your answer is saved as something you reported. It does not change the "
                           "model.")
            except ConsentError:
                st.info("Not saved (consent is off).")


# ============================================================================ 4 Check-in
def _new_session(trigger: str) -> None:
    ss = st.session_state
    ss["nt_session"] = CheckInSession(trigger, allow_free_text=bool(ss.get("nt_c_free_text", False)))
    for k in ("nt_shown", "nt_chosen", "nt_saved_session"):
        ss.pop(k, None)


def _finalize(s: CheckInSession) -> None:
    ss = st.session_state
    if ss.get("nt_saved_session") == s.id:
        return
    store = active_store()
    shown = ss.get("nt_shown", [])
    summary = service.save_checkin(store, USER_ID, s, shown)
    chosen = ss.get("nt_chosen")
    if chosen and s.feedback and not s.crisis:
        diff = s.feedback.get("different")
        fut = s.feedback.get("future")
        service.record_feedback(
            store, USER_ID, chosen["item"], s.feedback.get("helped"), s.feedback.get("feel_after"),
            None if diff not in ("yes", "no") else diff == "yes",
            None if fut not in ("yes", "no") else fut == "yes",
            rec_id=summary["rec_ids"].get(chosen["item"].id))
    ss["nt_saved_session"] = s.id
    ss["nt_last_summary"] = summary


def _render_suggestion(sg, key: str) -> None:
    it = sg.item
    st.markdown(f"**{it.title}**")
    st.write(it.text)
    if it.link:
        st.markdown(f"[More]({it.link})")
    with st.expander("Evidence, cautions and why this was suggested"):
        st.write("**Evidence:** " + it.evidence_note)
        st.write("**Please note:** " + it.contraindications)
        for src in it.sources:
            st.markdown(f"- [{src['name']}]({src['url']}) (source level {src['level']})")
        st.write("**Why it was suggested:** " + "; ".join(sg.reasons))
    st.caption("Optional. You can skip this, and it is not a promise that it will help.")


def page_checkin() -> None:
    _init_state()
    ss = st.session_state
    st.subheader("💬 Check-in")
    _banner()
    s: CheckInSession | None = ss.get("nt_session")

    if s is None:
        st.write("A short, optional conversation. You can skip any question, stop at any time, and nothing "
                 "is stored unless your consent settings allow it.")
        demo_trigger = st.checkbox("Demo: pretend the estimate triggered this check-in", value=False)
        if st.button("Start a check-in", type="primary"):
            _new_session("estimate" if demo_trigger else "user")
            st.rerun()
        st.info("Check-ins are never started automatically unless you opt in to prompts, and with the "
                "current model's accuracy the app does not interrupt you.")
        return

    # persistent controls
    cA, cB, _ = st.columns([1, 1, 3])
    if not s.finished:
        if cA.button("🆘 I need urgent help", key="urgent"):
            s.urgent()
            st.rerun()
        if cB.button("End check-in", key="decline"):
            s.decline()
            st.rerun()

    if s.crisis:
        _finalize(s)
        st.error(s.crisis_message())
        render_resources(ss["nt_region"])
        if st.button("Close", key="close_crisis"):
            ss["nt_session"] = None
            st.rerun()
        return
    if s.declined:
        _finalize(s)
        st.info("No problem. We have stopped here. You can come back any time.")
        if st.button("Close", key="close_declined"):
            ss["nt_session"] = None
            st.rerun()
        return

    node = s.node
    if node.id == "feeling":
        st.write(s.intro())

    if node.id == "recommend":
        _render_recommend_step(s)
        return
    if node.id == "close":
        _finalize(s)
        st.success(NODES["close"].prompt)
        summary = ss.get("nt_last_summary", {})
        if summary.get("skipped_kinds"):
            st.caption("Not saved because of your consent settings: " + ", ".join(summary["skipped_kinds"]))
        if st.button("Done", key="done"):
            ss["nt_session"] = None
            st.rerun()
        return

    st.markdown(f"**{node.prompt}**")
    if node.rating:
        val = st.slider("0 = not at all, 10 = extremely", 0, 10, 5, key=f"{s.id}-{node.id}-slider")
        c1, c2 = st.columns(2)
        if c1.button("Continue", key=f"{s.id}-{node.id}-go"):
            s.answer(rating=val)
            st.rerun()
        if c2.button("Skip", key=f"{s.id}-{node.id}-skip"):
            s.answer(option="skip")
            st.rerun()
        return
    text = None
    if node.free_text and s.allow_free_text:
        text = st.text_input("In your own words (optional)", key=f"{s.id}-{node.id}-text",
                             help="Checked only for safety words. Never analysed or used to guess a cause.")
    cols = st.columns(min(4, max(1, len(node.options))))
    for i, o in enumerate(node.options):
        if cols[i % len(cols)].button(o.label, key=f"{s.id}-{node.id}-{o.id}"):
            s.answer(option=o.id, text=text or None)
            st.rerun()
    if node.free_text and text and st.button("Use my own words and continue", key=f"{s.id}-{node.id}-own"):
        s.answer(option="other", text=text)
        st.rerun()
    if not node.options:
        if st.button("Continue", key=f"{s.id}-{node.id}-cont"):
            s.answer(option="skip")
            st.rerun()


def _render_recommend_step(s: CheckInSession) -> None:
    ss = st.session_state
    st.markdown(f"**{NODES['recommend'].prompt}**")
    if "nt_shown" not in ss:
        flags = []
        try:
            est, demo = get_estimator(), get_demo()
            sub, widx = ss.get("nt_sel", (demo.participants()[0], 0))
            wins = demo.windows(sub)
            flags = est.estimate_for_window(wins.index[widx]).quality.flags
        except Exception:
            pass
        ctx = service.context_from_session(
            s, recent_exertion="movement_high" in flags,
            declined_items=ss["nt_declined_items"], declined_tags=ss["nt_declined_tags"])
        ss["nt_shown"] = service.preview_recommendations(active_store(), USER_ID, ctx)
    shown = ss["nt_shown"]
    if not shown:
        st.info("I have no suggestions to offer right now, and that is okay.")
    for i, sg in enumerate(shown):
        _render_suggestion(sg, f"r{i}")
        if st.button("Not for me", key=f"nfm-{s.id}-{i}"):
            ss["nt_declined_items"].add(sg.item.id)
            ss.pop("nt_shown")
            st.rerun()
    st.divider()
    names = ["I didn't try any"] + [sg.item.title for sg in shown]
    pick = st.selectbox("Did you try one of these?", names, key=f"pick-{s.id}")
    if st.button("Continue", key=f"cont-{s.id}"):
        if pick == "I didn't try any" or not shown:
            ss["nt_chosen"] = None
            s.node_id = "close"
        else:
            ss["nt_chosen"] = {"item": shown[names.index(pick) - 1].item}
            s.node_id = "fb_helped"
        st.rerun()


# ============================================================================ 5 Possible stressors
def page_stressors() -> None:
    _init_state()
    st.subheader("🧭 Possible Stressors")
    _banner()
    st.info("This page only holds what **you** choose to tell us. The system never works out a stressor "
            "from your wearable signals, and your own explanation is always kept separate from the model's "
            "estimate.")
    store = active_store()
    with st.form("add_reason", clear_on_submit=True):
        reason = st.selectbox("Something that might be adding to how things feel (optional)",
                              list(REASON_LABELS), format_func=lambda k: REASON_LABELS[k])
        own = st.text_input("In your own words (optional)") if st.session_state.get("nt_c_free_text") else None
        sleep = st.number_input("Hours of sleep last night (optional, 0 = skip)", 0.0, 16.0, 0.0, 0.5)
        sub = st.form_submit_button("Save")
    if sub:
        if own and safety.screen_text(own):
            st.error(safety.CRISIS_MESSAGE)
            render_resources(st.session_state["nt_region"])
        else:
            for kind, val in (("reason", reason), ("free_text", own or None),
                              ("sleep_hours", sleep or None)):
                if val is None:
                    continue
                try:
                    store.add_reported(USER_ID, kind, val, session_id="manual")
                except ConsentError:
                    st.caption(f"'{kind}' was not saved (consent is off).")
            st.success("Saved as something you reported.")
    st.markdown("##### What you have told us")
    rows = [r for r in store.list_reported(USER_ID, ("reason", "free_text", "sleep_hours"))]
    if not rows:
        st.caption("Nothing yet. You never have to share this.")
    else:
        df = pd.DataFrame([{"id": r["id"], "when": r["ts"][:16], "type": r["kind"], "value": r["value"]}
                           for r in rows])
        st.dataframe(df, hide_index=True, use_container_width=True)
        pick = st.selectbox("Correct or delete an entry", df["id"].tolist(), key="fix_pick")
        c1, c2 = st.columns(2)
        newv = c1.selectbox("Change reason to", list(REASON_LABELS), format_func=lambda k: REASON_LABELS[k],
                            key="fix_val")
        if c1.button("Correct this entry"):
            kind = next(r["kind"] for r in rows if r["id"] == pick)
            if kind == "reason":
                store.correct_reported(USER_ID, int(pick), newv)
                st.rerun()
            else:
                st.caption("Only reasons can be corrected here. You can delete other entries.")
        if c2.button("Delete this entry"):
            store.delete_reported(USER_ID, int(pick))
            st.rerun()
    st.markdown("##### Observations from your own reports")
    pats = patterns.all_patterns(store.list_reported(USER_ID), store.list_feedback(USER_ID))
    if pats["reasons"]:
        for p in pats["reasons"]:
            st.write(p["text"])
    else:
        st.caption("Not enough repeated reports yet to say anything. " + pats["min_rules"])


# ============================================================================ 6 Recommendations
def page_recommendations() -> None:
    _init_state()
    ss = st.session_state
    st.subheader("🌿 Recommendation Centre")
    _banner()
    st.caption("Optional ideas, chosen by transparent rules from what you tell us and what you have found "
               "helpful before. None of them is a treatment, and none is guaranteed to help.")
    c1, c2, c3 = st.columns(3)
    mood = c1.selectbox("How are you feeling?", list(MOOD_LABELS), format_func=lambda k: MOOD_LABELS[k],
                        index=list(MOOD_LABELS).index("skip"), key="rc_mood")
    reason = c2.selectbox("What is it about? (optional)", ["skip"] + list(REASON_LABELS),
                          format_func=lambda k: "I'd rather not say" if k == "skip" else REASON_LABELS[k],
                          key="rc_reason")
    rating = c3.slider("Stress right now (optional)", 0, 10, 5, key="rc_rating")
    d1, d2 = st.columns(2)
    if d1.checkbox("Please don't suggest breathing exercises", key="rc_nobreath"):
        ss["nt_declined_tags"].add("breathing_declined")
    else:
        ss["nt_declined_tags"].discard("breathing_declined")
    if d2.checkbox("Please don't suggest movement", key="rc_nomove"):
        ss["nt_declined_tags"].add("movement_declined")
    else:
        ss["nt_declined_tags"].discard("movement_declined")
    if st.button("Show suggestions", type="primary", key="rc_show"):
        ctx = Context(mood=mood, reason=reason, stress_rating=rating, wants="suggestion",
                      declined_items=frozenset(ss["nt_declined_items"]),
                      declined_tags=frozenset(ss["nt_declined_tags"]))
        ss["rc_shown"] = service.preview_recommendations(active_store(), USER_ID, ctx)
    shown = ss.get("rc_shown")
    if shown is not None and not shown:
        st.info("I have no suggestions to offer with these settings, and that is okay.")
    for i, sg in enumerate(shown or []):
        with st.container(border=True):
            _render_suggestion(sg, f"rc{i}")
            a, b = st.columns(2)
            if a.button("I'll try this", key=f"rc_try_{i}"):
                try:
                    active_store().add_recommendation(USER_ID, sg.item.id, i + 1, sg.reasons, "rec-page")
                    st.success("Noted. Tell us how it went on the Coping Feedback page.")
                except ConsentError:
                    st.info("Not logged (consent is off).")
            if b.button("Not for me", key=f"rc_no_{i}"):
                ss["nt_declined_items"].add(sg.item.id)
                ss["rc_shown"] = [x for x in shown if x.item.id != sg.item.id]
                st.rerun()
    st.divider()
    st.markdown("##### If things feel heavy")
    st.write("If distress feels persistent or severe, please consider talking to someone you trust or a health "
             "professional. This app cannot replace that.")
    render_resources(ss["nt_region"])


# ============================================================================ 7 Twin profile
def page_profile() -> None:
    _init_state()
    st.subheader("🧬 Twin Profile")
    _banner()
    st.caption("A personal digital shadow with a feedback loop: sensing flows into it, and your feedback "
               "changes which suggestions come first. It is not a validated digital twin and it does not "
               "forecast anything.")
    est, demo = get_estimator(), get_demo()
    sub, _ = st.session_state.get("nt_sel", (demo.participants()[0], 0))
    base = demo.baseline(sub)
    store = active_store()
    prof = build_profile(store, USER_ID, base)
    st.markdown("##### 📡 Measured baseline (demo participant)")
    st.caption(DEMO_NOTICE)
    if base.get("available"):
        df = pd.DataFrame([{"Feature": FEATURE_LABELS.get(k, k), "Baseline": v}
                           for k, v in base["values"].items()])
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.write(base.get("note"))
    st.caption(prof["baseline_note"])
    st.markdown("##### 🗣️ What you have reported")
    st.json(prof["reported"])
    st.markdown("##### 🌿 Coping preferences learned from your feedback")
    c1, c2 = st.columns(2)
    c1.write("Preferred" if prof["preferred_coping"] else "No preferences learned yet.")
    for p in prof["preferred_coping"]:
        c1.write(f"• {p['title']} ({p['score']:+.1f})")
    if prof["avoided_coping"]:
        c2.write("Not wanted")
        for p in prof["avoided_coping"]:
            c2.write(f"• {p['title']}")
    st.markdown("##### 🔎 Observations from your own reports")
    obs = prof["observations"]
    items = obs["reasons"] + obs["sleep"] + obs["coping"]
    if items:
        for o in items:
            st.write(o["text"])
    else:
        st.caption("Nothing to report yet. " + obs["min_rules"])
    st.divider()
    st.markdown("##### Your data, your control")
    export = store.export_user(USER_ID)
    st.download_button("Download everything held about me (JSON)", json.dumps(export, indent=2, default=str),
                       file_name="neurotwin_my_data.json")
    if st.button("Forget what the system learned about my preferences"):
        store.reset_preferences(USER_ID)
        st.rerun()
    confirm = st.checkbox("I understand this deletes all my data in this app", key="del_confirm")
    if st.button("Delete all my data", disabled=not confirm):
        store.delete_user(USER_ID)
        st.session_state.pop("nt_shown", None)
        st.rerun()


# ============================================================================ 8 History & patterns
def page_history() -> None:
    _init_state()
    st.subheader("🗓️ History & Patterns")
    _banner()
    store = active_store()
    rep = store.list_reported(USER_ID)
    ratings = [(r["ts"], r["value"]) for r in rep if r["kind"] == "stress_rating"]
    if not ratings:
        st.info("No check-ins yet. When you share ratings they appear here. Everything on this page comes "
                "from your own reports, not from the sensor.")
    else:
        df = pd.DataFrame(ratings, columns=["when", "your rating"])
        df["when"] = pd.to_datetime(df["when"])
        st.line_chart(df.set_index("when"))
    obs = patterns.all_patterns(rep, store.list_feedback(USER_ID))
    st.markdown("##### Observations")
    shown = obs["reasons"] + obs["sleep"] + obs["coping"]
    if shown:
        for o in shown:
            st.write("• " + o["text"])
    else:
        st.caption("Not enough repeated reports yet. " + obs["min_rules"])
    st.caption("These describe what you reported. They do not show that anything caused a change in your "
               "body or mood.")


# ============================================================================ 9 Coping feedback
def page_feedback() -> None:
    _init_state()
    st.subheader("🔁 Coping Feedback")
    _banner()
    store = active_store()
    cat = load_catalogue()
    recs = store.list_recommendations(USER_ID)
    answered = {f["rec_id"] for f in store.list_feedback(USER_ID)}
    pending = [r for r in recs if r["id"] not in answered]
    if not pending:
        st.info("Nothing waiting for feedback. After you try a suggestion, come back and tell us how it went.")
    else:
        pick = st.selectbox("Which suggestion?", pending,
                            format_func=lambda r: f"{cat[r['item_id']].title} ({r['ts'][:16]})")
        helped = st.radio("Did this help?", ["yes", "somewhat", "no", "skipped"],
                          format_func={"yes": "Yes", "somewhat": "A little", "no": "Not really",
                                       "skipped": "Skip"}.get, horizontal=True)
        feel = st.slider("How do you feel now? (0 = not at all stressed, 10 = extremely)", 0, 10, 5)
        diff = st.radio("Would you like to try something different?", ["skip", "yes", "no"], horizontal=True)
        fut = st.radio("Would you like this type of suggestion in future?", ["skip", "yes", "no"],
                       horizontal=True)
        if st.button("Save feedback", type="primary"):
            out = service.record_feedback(
                store, USER_ID, cat[pick["item_id"]], helped, feel,
                None if diff == "skip" else diff == "yes", None if fut == "skip" else fut == "yes",
                rec_id=pick["id"])
            st.success("Thank you." if out["stored"] else
                       "Not saved (feedback consent is off), so nothing was learned.")
            st.rerun()
    st.markdown("##### How suggestions adapt")
    st.write("Your answers nudge a small, visible score for each idea and each type of idea. Higher scores come "
             "first; ideas you say you do not want are not shown again. This is simple rule-based adaptation. "
             "It is not machine learning, and we have not tested whether it helps people.")
    prefs = store.get_preferences(USER_ID)
    if prefs:
        st.dataframe(pd.DataFrame(sorted(prefs.items()), columns=["item / type", "score"]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No preferences stored yet.")


# ============================================================================ 10 Evaluation & transparency
def page_evaluation() -> None:
    _init_state()
    st.subheader("📊 Evaluation & Transparency")
    _banner()
    st.markdown("##### Part A — what has been evaluated (participant-independent, one laboratory dataset)")
    try:
        st.dataframe(pd.read_csv(config.PROCESSED_DIR / "model_comparison_scores.csv"),
                     hide_index=True, use_container_width=True)
    except Exception:
        st.caption("model_comparison_scores.csv not found.")
    rep_path = config.ESTIMATOR_REPORT
    # Use the saved audit if present; otherwise compute it (a few seconds, nothing is written).
    rep = (json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.is_file()
           else get_estimator().calibration_report())
    if rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (logistic regression)", f"{rep['pooled_accuracy']:.3f}")
        c2.metric("Majority-class baseline", f"{rep['majority_baseline_accuracy']:.3f}")
        c3.metric("AUC (held-out participants)", f"{rep['pooled_auc']:.3f}")
        c4.metric("Calibration acceptable?", "No" if not rep["calibration_ok"] else "Yes")
        st.write(f"Brier score: recalibrated {rep['brier_recalibrated']}, always-guess-the-prior "
                 f"{rep['brier_training_prior']}. Expected calibration error: {rep['ece_recalibrated']}.")
        st.caption(rep["probability_policy"])
        st.dataframe(pd.DataFrame(rep["reliability_raw"]), hide_index=True, use_container_width=True)
        st.caption("Reliability of the raw model on unseen participants: in a well-calibrated model the "
                   "observed rate would rise with the predicted probability.")
    st.warning("No classifier beat the always-guess-the-majority baseline on accuracy. The app therefore "
               "caps evidence at 'limited', shows no probabilities, and does not interrupt you.")

    # Added in A6 integration: Model investigation evidence comparing v1 vs v2
    try:
        from dashboard.components_aurora import render_v1_v2_comparison, render_decision_rule_panel
        render_v1_v2_comparison()
        render_decision_rule_panel()
    except Exception:
        pass

    st.markdown("##### What the dataset is, and is not")
    st.write("36 healthy volunteers aged 18–30 (35 modelled), an Empatica E4 wristband, and structured "
             "laboratory stress tasks. Stress is a single verbal 1–10 self-rating per stage, thresholded at "
             f"{STRESS_THRESHOLD:g} for the model. There are no questionnaires, sleep records, real-life "
             "stressors, long-term histories or intervention outcomes in the data.")
    st.markdown("##### Part B — what is only verified as software, not evaluated for effect")
    sim_path = config.SIMULATION_REPORT
    st.write("The conversation, recommendations, feedback and profile work as specified in automated tests. "
             "Whether they help real people has **not** been studied.")
    sim = json.loads(sim_path.read_text(encoding="utf-8")) if sim_path.is_file() else _cached_simulation()
    if sim:
        st.caption("⚠️ " + sim["label"])
        st.write(f"In a scripted simulation with {sim['n_simulated_users']} invented users, the adapted "
                 f"ordering matched the invented preferences more often ({sim['adaptive_mean']} vs "
                 f"{sim['static_mean']}). This checks the code, not real-world benefit.")
    st.markdown("##### Part C — future validation")
    st.write("Longitudinal studies with consenting users, repeated measurements, and outcome assessment, "
             "under ethics approval.")
    st.markdown("##### What this app does not claim")
    for t in ("It does not diagnose or screen for any condition.",
              "It does not know why your signals changed or what is causing any stress.",
              "It does not show that any suggestion reduces stress.",
              "A personal baseline is not a clinical normal range.",
              "It is not a validated digital twin and does not replace professional support."):
        st.write("• " + t)


PAGE_FUNCS = {
    "Home & Status": page_home,
    "Signals & Features": page_signals,
    "Explain the Estimate": page_explain,
    "Check-in": page_checkin,
    "Possible Stressors": page_stressors,
    "Recommendations": page_recommendations,
    "Twin Profile": page_profile,
    "History & Patterns": page_history,
    "Coping Feedback": page_feedback,
    "Evaluation & Transparency": page_evaluation,
}
