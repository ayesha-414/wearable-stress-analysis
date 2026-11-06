"""Catalogue discipline, recommender rules, preference adaptation and pattern reporting."""
import pytest

from src.neurotwin import config, patterns
from src.neurotwin.catalogue import load_catalogue, validate
from src.neurotwin.language import find_forbidden
from src.neurotwin.recommender import Context, recommend, update_preferences

CAT = load_catalogue()


# ------------------------------------------------------------------ catalogue
def test_catalogue_is_valid_and_clean():
    assert validate(CAT) == []


def test_every_item_has_source_evidence_and_caution():
    for it in CAT.values():
        assert it.sources and all(s["url"].startswith("http") for s in it.sources), it.id
        assert it.evidence_note.strip() and it.contraindications.strip(), it.id
        assert all(s["level"] in ("A", "B") for s in it.sources)


def test_no_item_promises_an_effect():
    for it in CAT.values():
        assert find_forbidden(" ".join([it.title, it.text, it.evidence_note, it.contraindications])) == []


def test_every_category_has_at_least_one_item():
    import json
    cats = json.load(open(config.CATALOGUE_JSON, encoding="utf-8"))["categories"]
    for c in cats:
        assert any(c in it.categories for it in CAT.values()), c


def test_breathing_item_is_slow_and_has_safety_text():
    b = CAT["slow_breathing"]
    assert "do not hold your breath" in b.text.lower()
    assert "dizzy" in b.contraindications.lower()
    assert "fast" in b.evidence_note.lower()


def test_who_items_point_to_the_who_guide():
    for k in ("who_grounding", "who_notice_name", "who_kindness", "who_making_room"):
        assert any("who.int" in s["url"] for s in CAT[k].sources)


# ------------------------------------------------------------------ recommender rules
def ids(ctx, prefs=None):
    return [s.item.id for s in recommend(ctx, prefs)]


def test_crisis_means_no_suggestions_at_all():
    assert recommend(Context(crisis=True, mood="overwhelmed", stress_rating=10)) == []


def test_user_who_wants_nothing_gets_nothing():
    assert recommend(Context(wants="nothing", mood="tense")) == []


def test_never_more_than_max_suggestions():
    assert len(recommend(Context(mood="overwhelmed", reason="studies", stress_rating=9))) <= config.MAX_SUGGESTIONS


def test_sleep_reason_gives_sleep_items_first():
    got = ids(Context(reason="sleep"))
    assert got and all(CAT[i].categories and "sleep" in CAT[i].categories for i in got)


def test_study_reason_gives_study_items():
    assert "split_task" in ids(Context(reason="studies"))


def test_recent_exertion_excludes_movement():
    assert "gentle_movement" not in ids(Context(mood="tense", reason="work", recent_exertion=True))
    assert "gentle_movement" in [s.item.id for s in recommend(Context(mood="tense"), max_items=10)]


def test_declined_tags_and_items_are_respected():
    assert "slow_breathing" not in [s.item.id for s in recommend(
        Context(mood="overwhelmed", declined_tags=frozenset({"breathing_declined"})), max_items=10)]
    assert "split_task" not in ids(Context(reason="studies", declined_items=frozenset({"split_task"})))


def test_high_distress_always_offers_a_human_support_option():
    for ctx in (Context(mood="overwhelmed"), Context(stress_rating=9, reason="studies")):
        got = [s.item.id for s in recommend(ctx)]
        assert {"talk_to_someone", "professional_support"} & set(got), ctx


def test_cold_start_is_deterministic():
    c = Context(mood="tense", reason="work")
    assert ids(c) == ids(c)


def test_every_suggestion_explains_itself():
    for s in recommend(Context(mood="tense", reason="work")):
        assert s.reasons and any("base priority" in r for r in s.reasons)


def test_repeat_penalty_rotates_suggestions():
    first = ids(Context(reason="sleep"))[0]
    again = ids(Context(reason="sleep", recent_item_ids=(first, first)))
    assert again[0] != first


# ------------------------------------------------------------------ adaptation
def test_helpful_feedback_raises_and_unhelpful_lowers_score():
    up = update_preferences({}, CAT["who_grounding"], "yes")
    down = update_preferences({}, CAT["who_grounding"], "no")
    assert up["item:who_grounding"] > 0 > down["item:who_grounding"]


def test_adaptation_changes_the_ranking_in_the_expected_direction():
    ctx = Context(mood="overwhelmed")
    before = ids(ctx)
    prefs = {}
    for _ in range(3):
        prefs = update_preferences(prefs, CAT[before[0]], "no")
    after = ids(ctx, prefs)
    assert after[0] != before[0] or before[0] not in after


def test_preferences_are_bounded():
    p = {}
    for _ in range(50):
        p = update_preferences(p, CAT["who_grounding"], "yes", want_future=True)
    assert max(p.values()) <= config.PREF_BOUND
    q = {}
    for _ in range(50):
        q = update_preferences(q, CAT["who_grounding"], "no")
    assert min(q.values()) >= -config.PREF_BOUND


def test_do_not_want_again_excludes_item_from_future():
    p = update_preferences({}, CAT["slow_breathing"], "no", want_future=False)
    assert "slow_breathing" not in [s.item.id for s in recommend(Context(mood="overwhelmed"), p, max_items=10)]


def test_user_can_change_their_mind():
    p = update_preferences({}, CAT["slow_breathing"], "no", want_future=False)
    p["item:slow_breathing"] = 0.0                       # e.g. after 'forget what you learned'
    assert "slow_breathing" in [s.item.id for s in recommend(Context(mood="overwhelmed"), p, max_items=10)]


def test_skipped_feedback_changes_nothing():
    assert update_preferences({}, CAT["who_grounding"], "skipped")["item:who_grounding"] == 0.0


def test_update_does_not_mutate_input():
    p = {"item:x": 1.0}
    update_preferences(p, CAT["who_grounding"], "yes")
    assert p == {"item:x": 1.0}


# ------------------------------------------------------------------ patterns
def rep(kind, value, day):
    return {"kind": kind, "value": value, "ts": f"2026-11-{day:02d}T10:00:00"}


def test_no_pattern_below_repeat_threshold():
    assert patterns.reason_patterns([rep("reason", "studies", 1), rep("reason", "studies", 2)]) == []


def test_same_day_repeats_are_not_a_pattern():
    assert patterns.reason_patterns([rep("reason", "studies", 1)] * 5) == []


def test_pattern_reported_after_enough_repeats_on_different_days():
    out = patterns.reason_patterns([rep("reason", "studies", d) for d in (1, 2, 3)])
    assert len(out) == 1 and out[0]["n"] == 3 and out[0]["layer"] == "hypothesis"
    assert "reported" in out[0]["text"].lower()


def test_pattern_wording_never_claims_cause():
    out = patterns.reason_patterns([rep("reason", "studies", d) for d in (1, 2, 3)])
    t = out[0]["text"].lower()
    assert "does not show" in t and "causes" in t           # states the limit explicitly
    assert find_forbidden(out[0]["text"]) == []


def test_sleep_comparison_needs_enough_days_in_both_groups():
    few = [rep("sleep_hours", 5, d) for d in range(1, 4)] + [rep("stress_rating", 7, d) for d in range(1, 4)]
    assert patterns.sleep_comparison(few) is None
    many = []
    for d in range(1, 6):
        many += [rep("sleep_hours", 5, d), rep("stress_rating", 7, d)]
    for d in range(6, 11):
        many += [rep("sleep_hours", 8, d), rep("stress_rating", 3, d)]
    out = patterns.sleep_comparison(many)
    assert out and out["mean_short"] == 7.0 and out["mean_longer"] == 3.0
    assert "does not show" in out["text"].lower() and find_forbidden(out["text"]) == []


def test_coping_helpfulness_needs_enough_answers():
    fb = lambda h: {"item_id": "who_grounding", "helped": h}
    assert patterns.coping_helpfulness([fb("yes"), fb("yes")]) == []
    out = patterns.coping_helpfulness([fb("yes"), fb("yes"), fb("no")])
    assert out and out[0]["yes"] == 2 and "does not show" in out[0]["text"]


def test_skipped_feedback_not_counted_as_an_answer():
    fb = [{"item_id": "a", "helped": "skipped"}] * 5
    assert patterns.coping_helpfulness(fb) == []


def test_reports_without_repeated_data_give_empty_patterns():
    out = patterns.all_patterns([], [])
    assert out["reasons"] == [] and out["sleep"] == [] and out["coping"] == []
