"""The personal 'digital twin' profile (view-model).

Terminology: with the National Academies definition used in the digital-twin literature
(personalised, dynamically updated, predictive, decision-informing, two-way), this is best
described as a PERSONAL DIGITAL SHADOW WITH A FEEDBACK LOOP FOR RECOMMENDATIONS:
sensing flows one way into the profile; the person's feedback changes which suggestions are
ranked first; it estimates a current state rather than forecasting. It is not claimed to be
a validated digital twin. See docs/neurotwin/RESEARCH_OBJECTIVES.md.

The profile keeps the layers apart: ``measured``, ``estimated``, ``reported`` and
``observations`` (computed only from the person's own reports).
"""
from __future__ import annotations

from collections import Counter

from . import patterns
from .catalogue import load_catalogue
from .store import Store

BASELINE_NOTE = (
    "A baseline here is what was measured for you at one time. It is not a clinical reference "
    "range and not necessarily your healthy or typical state."
)


def preferred_coping(prefs: dict[str, float], top: int = 5) -> list[dict]:
    cat = load_catalogue()
    rows = [(k.split(":", 1)[1], v) for k, v in prefs.items() if k.startswith("item:") and v > 0]
    rows.sort(key=lambda kv: (-kv[1], kv[0]))
    return [{"item_id": i, "title": cat[i].title if i in cat else i, "score": round(v, 2)}
            for i, v in rows[:top]]


def avoided_coping(prefs: dict[str, float]) -> list[dict]:
    cat = load_catalogue()
    rows = [(k.split(":", 1)[1], v) for k, v in prefs.items() if k.startswith("item:") and v < 0]
    rows.sort(key=lambda kv: (kv[1], kv[0]))
    return [{"item_id": i, "title": cat[i].title if i in cat else i, "score": round(v, 2)} for i, v in rows]


def build_profile(store: Store, user_id: str, baseline: dict | None = None,
                  recent_n: int = 5) -> dict:
    reported = store.list_reported(user_id)
    feedback = store.list_feedback(user_id)
    prefs = store.get_preferences(user_id)
    estimates = store.list_estimates(user_id)
    ratings = [(r["ts"], r["value"]) for r in reported if r["kind"] == "stress_rating"]
    reasons = Counter(r["value"] for r in reported if r["kind"] == "reason")
    return {
        "user_id": user_id,
        "consent": store.get_consent(user_id),
        "measured_baseline": baseline or {"available": False,
                                          "note": "No baseline recorded for this profile."},
        "baseline_note": BASELINE_NOTE,
        "estimated_recent": [
            {"ts": e["ts"], "label": e["result"].get("label"),
             "evidence_level": e["result"].get("evidence_level"),
             "probability_shown": e["result"].get("probability_shown")}
            for e in estimates[-recent_n:]],
        "reported": {
            "n_checkins": len({r.get("session_id") for r in reported if r.get("session_id")}),
            "stress_ratings": ratings[-recent_n:],
            "stressors": dict(reasons),
        },
        "recommendations_shown": len(store.list_recommendations(user_id)),
        "feedback_given": len(feedback),
        "preferred_coping": preferred_coping(prefs),
        "avoided_coping": avoided_coping(prefs),
        "observations": patterns.all_patterns(reported, feedback),
        "labels": {"layer_note": ("Measured = what the sensor recorded. Estimated = what the model output. "
                                  "Reported = what you told us. Observations are computed only from your own reports.")},
    }
