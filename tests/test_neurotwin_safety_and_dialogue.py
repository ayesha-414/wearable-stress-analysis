"""Safety rules, crisis routing, prompt discipline and the scripted dialogue."""
from datetime import datetime, timedelta

import pytest

from src.neurotwin import safety, service
from src.neurotwin.dialogue import NODES, CheckInSession
from src.neurotwin.language import assert_clean, find_forbidden
from src.neurotwin.store import Store


# ------------------------------------------------------------------ crisis screen
@pytest.mark.parametrize("text", [
    "I want to die", "I'm thinking about suicide", "i might hurt myself", "I want to end my life",
    "I don't want to live anymore", "thinking of self-harm", "better off dead"])
def test_crisis_phrases_detected(text):
    assert safety.screen_text(text)


@pytest.mark.parametrize("text", ["I have an exam on Monday", "work deadlines are heavy", "", None,
                                  "my heart is racing before the presentation"])
def test_ordinary_text_not_flagged(text):
    assert not safety.screen_text(text)


def test_resources_have_no_default_region():
    info = safety.get_resources(None)
    assert info["region"] == "GENERIC" and info["needs_region"] is True


def test_resources_are_marked_unverified_until_owner_verifies():
    for region in (None, "IN"):
        assert safety.get_resources(region)["unverified"] is True


def test_known_region_lists_its_resources_plus_generic():
    names = " ".join(r["name"] for r in safety.get_resources("IN")["resources"])
    assert "Tele-MANAS" in names and "trust" in names


def test_unknown_region_falls_back_safely():
    assert safety.get_resources("ZZ")["region"] == "GENERIC"


def test_every_resource_entry_has_required_fields():
    for region, data in safety.load_resources().items():
        for r in data["resources"]:
            assert {"name", "kind", "notes", "verified_on"} <= set(r), (region, r)


# ------------------------------------------------------------------ prompt policy
NOW = datetime(2026, 11, 2, 14, 0)


def test_prompts_are_off_by_default():
    ok, why = safety.PromptPolicy().should_prompt(NOW, "moderate")
    assert not ok and "off" in why.lower()


def test_prompt_never_on_weak_or_missing_evidence():
    p = safety.PromptPolicy(enabled=True)
    for level in ("limited", "insufficient", None):
        assert p.should_prompt(NOW, level)[0] is False
    assert p.should_prompt(NOW, "moderate", abstained=True)[0] is False


def test_prompt_allowed_only_when_everything_lines_up():
    assert safety.PromptPolicy(enabled=True).should_prompt(NOW, "moderate")[0] is True


def test_quiet_hours_cover_midnight():
    p = safety.PromptPolicy(enabled=True)
    assert p.should_prompt(datetime(2026, 11, 2, 23, 30), "moderate")[0] is False
    assert p.should_prompt(datetime(2026, 11, 2, 6, 0), "moderate")[0] is False


def test_interval_cap_and_snooze():
    p = safety.PromptPolicy(enabled=True)
    p.record(NOW - timedelta(minutes=30))
    assert p.should_prompt(NOW, "moderate")[0] is False                 # too soon
    p2 = safety.PromptPolicy(enabled=True, daily_cap=2)
    p2.record(datetime(2026, 11, 2, 8)); p2.record(datetime(2026, 11, 2, 11))
    assert p2.should_prompt(NOW, "moderate")[0] is False                # daily cap
    p3 = safety.PromptPolicy(enabled=True)
    p3.snooze(NOW)
    assert p3.should_prompt(NOW + timedelta(hours=1), "moderate")[0] is False


# ------------------------------------------------------------------ dialogue
def test_every_prompt_and_option_uses_clean_wording():
    for node in NODES.values():
        assert not find_forbidden(node.prompt), node.id
        for o in node.options:
            assert not find_forbidden(o.label)


def test_every_question_can_be_skipped_or_declined():
    ask = [n for n in NODES.values() if n.kind in ("mood", "reason", "similar_before", "estimate_agreement")]
    for n in ask:
        assert "skip" in {o.id for o in n.options}, n.id
    s = CheckInSession(); s.decline()
    assert s.finished and s.declined and s.items == []


def test_skipping_records_nothing_and_never_blocks():
    s = CheckInSession()
    for _ in range(12):
        if s.finished or s.node_id == "recommend":
            break
        s.answer(option="skip")
    assert s.items == [] or all(k not in ("reason", "mood") for k, _ in s.items)


def test_nothing_right_now_ends_the_conversation():
    s = CheckInSession(); s.answer("calm_ok"); s.answer(rating=2)
    assert s.answer("nothing") == "close"


def test_user_can_say_they_do_not_know_the_reason():
    s = CheckInSession(); s.answer("tense"); s.answer(rating=4); s.answer("talk")
    assert s.node_id == "reason"
    s.answer("unknown")
    assert s.value("reason") == "unknown"


def test_agreement_question_only_when_estimate_triggered_it():
    for trigger, expect in (("user", "recommend"), ("estimate", "agreement")):
        s = CheckInSession(trigger); s.answer("tense"); s.answer(rating=4); s.answer("talk")
        s.answer("skip"); nxt = s.answer("yes")
        assert nxt == expect, trigger


def test_estimate_intro_says_estimate_and_may_be_wrong():
    text = CheckInSession("estimate").intro().lower()
    assert "estimate" in text and "wrong" in text


def test_rating_is_clamped_to_0_10():
    s = CheckInSession(); s.answer("tense"); s.answer(rating=99)
    assert s.value("stress_rating") == 10


def test_invalid_option_rejected():
    s = CheckInSession()
    with pytest.raises(ValueError):
        s.answer("not_an_option")


def test_free_text_ignored_unless_user_allowed_it():
    for allow, expected in ((False, None), (True, "exam on Monday")):
        s = CheckInSession(allow_free_text=allow)
        s.answer("tense"); s.answer(rating=4); s.answer("talk"); s.answer("studies", text="exam on Monday")
        assert s.value("free_text") == expected


def test_crisis_text_stops_flow_and_is_never_kept():
    s = CheckInSession(allow_free_text=True)
    s.answer("tense"); s.answer(rating=4); s.answer("talk")
    s.answer("other", text="I want to die")
    assert s.crisis and s.finished
    assert all("die" not in str(v) for _, v in s.items)
    assert "emergency" in s.crisis_message().lower()
    before = list(s.items)
    s.answer("studies")                                         # further answers are ignored
    assert s.items == before


def test_urgent_button_works_at_every_step():
    for step in ("feeling", "rating", "wants", "reason"):
        s = CheckInSession(); s.node_id = step; s.urgent()
        assert s.crisis and s.finished


def test_crisis_session_gets_no_recommendations_and_stores_no_crisis_text():
    s = CheckInSession(allow_free_text=True)
    s.answer("overwhelmed"); s.answer(rating=9); s.answer("suggestion"); s.answer("other", text="I want to kill myself")
    ctx = service.context_from_session(s)
    from src.neurotwin.recommender import recommend
    assert recommend(ctx) == []
    st = Store(":memory:"); st.ensure_user("u")
    st.set_consent("u", store_checkins=True, store_stressors=True, store_free_text=True, store_feedback=True)
    service.save_checkin(st, "u", s, [])
    assert not any("kill" in str(r["value"]) for r in st.list_reported("u"))


def test_save_checkin_respects_consent_and_reports_what_was_skipped():
    s = CheckInSession(); s.answer("tense"); s.answer(rating=4); s.answer("talk"); s.answer("studies")
    st = Store(":memory:"); st.ensure_user("u"); st.set_consent("u", store_checkins=True)
    out = service.save_checkin(st, "u", s, [])
    kinds = {r["kind"] for r in st.list_reported("u")}
    assert "mood" in kinds and "reason" not in kinds and "reason" in out["skipped_kinds"]
