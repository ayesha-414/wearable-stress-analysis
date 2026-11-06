"""Cautious pattern reporting from a person's OWN reports.

Rules
-----
* Nothing is reported without repeated observations (``config.MIN_*``).
* Statements describe what the person REPORTED. They never say that something CAUSED a
  physiological change or a feeling, and they never use the wearable data.
* Output items have layer 'hypothesis': an observation about the person's own log,
  kept apart from the Reported records they are computed from.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean

from . import config

REASON_PHRASE = {
    "work": "work-related stress", "studies": "study-related stress",
    "relationships": "stress related to relationships", "health": "stress related to health",
    "finances": "stress related to finances", "sleep": "sleep difficulties",
    "other": "stress for another reason",
}


def _day(ts: str) -> str:
    return str(ts)[:10]


def reason_patterns(reported: list[dict]) -> list[dict]:
    """Repeated self-reported reasons. ``reported`` = rows from Store.list_reported."""
    by_reason: dict[str, list[str]] = defaultdict(list)
    for r in reported:
        if r["kind"] == "reason" and r["value"] in REASON_PHRASE:
            by_reason[r["value"]].append(r["ts"])
    out = []
    for reason, stamps in by_reason.items():
        days = {_day(t) for t in stamps}
        if len(stamps) >= config.MIN_PATTERN_REPEATS and len(days) >= config.MIN_PATTERN_DISTINCT_DAYS:
            out.append({
                "layer": "hypothesis", "kind": "reason_repeat", "reason": reason,
                "n": len(stamps), "days": len(days),
                "text": (f"You have reported {REASON_PHRASE[reason]} on {len(stamps)} occasions "
                         f"across {len(days)} days. This describes your own check-ins only; "
                         "it does not show what causes any change in your body or mood."),
            })
    return sorted(out, key=lambda d: -d["n"])


def sleep_comparison(reported: list[dict], threshold_hours: float = 6.0) -> dict | None:
    """Compare the person's own stress ratings on days with short vs longer logged sleep.
    Only if both groups have enough observations. Purely descriptive; no inference."""
    sleep_by_day, ratings_by_day = {}, defaultdict(list)
    for r in reported:
        if r["kind"] == "sleep_hours" and isinstance(r["value"], (int, float)):
            sleep_by_day[_day(r["ts"])] = float(r["value"])
        elif r["kind"] == "stress_rating" and isinstance(r["value"], (int, float)):
            ratings_by_day[_day(r["ts"])].append(float(r["value"]))
    short, longer = [], []
    for d, hours in sleep_by_day.items():
        if d in ratings_by_day:
            (short if hours < threshold_hours else longer).append(mean(ratings_by_day[d]))
    k = config.MIN_GROUP_FOR_COMPARISON
    if len(short) >= k and len(longer) >= k:
        return {
            "layer": "hypothesis", "kind": "sleep_vs_rating", "n_short": len(short), "n_longer": len(longer),
            "mean_short": round(mean(short), 2), "mean_longer": round(mean(longer), 2),
            "text": (f"On days when you logged under {threshold_hours:g} hours of sleep, your own stress "
                     f"ratings averaged {mean(short):.1f} (n={len(short)} days), compared with "
                     f"{mean(longer):.1f} (n={len(longer)} days) otherwise. This describes your log only "
                     "and does not show that sleep affects your stress."),
        }
    return None


def coping_helpfulness(feedback: list[dict]) -> list[dict]:
    """Per-suggestion share of 'helped' answers, only with enough answered feedback."""
    by_item: dict[str, list[str]] = defaultdict(list)
    for f in feedback:
        if f.get("helped") in ("yes", "somewhat", "no"):
            by_item[f["item_id"]].append(f["helped"])
    out = []
    for item, answers in by_item.items():
        if len(answers) >= config.MIN_HELP_OBSERVATIONS:
            yes = sum(a == "yes" for a in answers)
            some = sum(a == "somewhat" for a in answers)
            out.append({
                "layer": "hypothesis", "kind": "coping_helpfulness", "item_id": item,
                "n": len(answers), "yes": yes, "somewhat": some,
                "text": (f"You said '{item.replace('_', ' ')}' helped {yes} time(s) and helped "
                         f"somewhat {some} time(s) out of {len(answers)}. This is your own feedback; "
                         "it does not show that the suggestion changed your stress."),
            })
    return sorted(out, key=lambda d: (-(d["yes"] + 0.5 * d["somewhat"]) / d["n"], d["item_id"]))


def all_patterns(reported: list[dict], feedback: list[dict]) -> dict:
    sleep = sleep_comparison(reported)
    return {
        "reasons": reason_patterns(reported),
        "sleep": [sleep] if sleep else [],
        "coping": coping_helpfulness(feedback),
        "min_rules": (f"A pattern is shown only after at least {config.MIN_PATTERN_REPEATS} reports on "
                      f"{config.MIN_PATTERN_DISTINCT_DAYS}+ different days (sleep comparison: "
                      f"{config.MIN_GROUP_FOR_COMPARISON}+ days per group; helpfulness: "
                      f"{config.MIN_HELP_OBSERVATIONS}+ answers)."),
    }
