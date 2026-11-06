"""Transparent, rule-based recommender with simple preference adaptation.

Not machine learning, not reinforcement learning. Each suggestion is scored by a short,
fully visible sum (base priority + category match + stored preference + repeat penalty)
and the reasons are returned with it. Hard safety exclusions are applied FIRST.

Adaptation (``update_preferences``) changes bounded per-item and per-category scores from
the person's own feedback. It makes no claim that the adapted ordering helps anyone;
effectiveness is untested (see docs/neurotwin/).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .catalogue import Item, load_catalogue

REASON_TO_CATEGORIES = {
    "work": ["study_work_pressure"], "studies": ["study_work_pressure"],
    "relationships": ["emotional_distress"], "health": ["emotional_distress"],
    "finances": ["emotional_distress"], "sleep": ["sleep"],
    "other": ["general"], "unknown": ["general", "immediate_overwhelm"], "skip": ["general"],
}
MOOD_TO_CATEGORIES = {
    "overwhelmed": ["immediate_overwhelm"], "tense": ["physical_tension", "immediate_overwhelm"],
    "tired": ["sleep", "physical_tension"], "low": ["emotional_distress"],
    "calm_ok": ["general"], "other": ["general"], "skip": [],
}


@dataclass
class Context:
    crisis: bool = False
    mood: str | None = None
    stress_rating: int | None = None
    reason: str | None = None
    wants: str | None = None               # talk | exercise | suggestion | nothing
    recent_exertion: bool = False          # e.g. movement flag from the estimator
    declined_items: frozenset = frozenset()
    declined_tags: frozenset = frozenset() # e.g. {"breathing_declined", "movement_declined"}
    recent_item_ids: tuple = ()            # most recent suggestions (for repeat penalty)


@dataclass
class Suggestion:
    item: Item
    score: float
    reasons: list[str] = field(default_factory=list)


def _item_pref_key(item_id: str) -> str:
    return f"item:{item_id}"


def _cat_pref_key(cat: str) -> str:
    return f"cat:{cat}"


def blocked_keys(prefs: dict[str, float]) -> set[str]:
    """Items/categories the user said they do not want in future (score pinned very low)."""
    return {k for k, v in prefs.items() if v <= -config.PREF_BOUND}


def recommend(ctx: Context, prefs: dict[str, float] | None = None,
              catalogue: dict[str, Item] | None = None,
              max_items: int = config.MAX_SUGGESTIONS) -> list[Suggestion]:
    """Return up to ``max_items`` suggestions, best first. Empty if none is appropriate."""
    prefs = prefs or {}
    cat = catalogue or load_catalogue()

    # ---- hard rules, applied first -------------------------------------------------
    if ctx.crisis:
        return []                                  # crisis path: no exercises, no suggestions
    if ctx.wants == "nothing":
        return []

    # ---- which categories are relevant ---------------------------------------------
    wanted: dict[str, float] = {}
    for c in REASON_TO_CATEGORIES.get(ctx.reason or "", []):
        wanted[c] = max(wanted.get(c, 0), 1.0)
    for c in MOOD_TO_CATEGORIES.get(ctx.mood or "", []):
        wanted[c] = max(wanted.get(c, 0), 0.8)
    if (ctx.stress_rating is not None and ctx.stress_rating >= 7) or ctx.mood == "overwhelmed":
        wanted["immediate_overwhelm"] = max(wanted.get("immediate_overwhelm", 0), 1.0)
        wanted["emotional_distress"] = max(wanted.get("emotional_distress", 0), 0.6)
    if not wanted:
        wanted = {"general": 0.5}

    blocked = blocked_keys(prefs)
    suggestions: list[Suggestion] = []
    tags = set(ctx.declined_tags)
    if ctx.recent_exertion:
        tags.add("recent_exertion")

    for it in cat.values():
        if it.id in ctx.declined_items or _item_pref_key(it.id) in blocked:
            continue
        if set(it.excluded_when) & tags:
            continue
        matched = [c for c in it.categories if c in wanted]
        if not matched:
            continue
        if all(_cat_pref_key(c) in blocked for c in it.categories):
            continue
        reasons = []
        score = it.priority
        reasons.append(f"base priority {it.priority:+.2f}")
        best_cat = max(wanted[c] for c in matched)
        score += best_cat
        reasons.append(f"matches what you shared ({', '.join(matched)}) {best_cat:+.2f}")
        ip = prefs.get(_item_pref_key(it.id), 0.0)
        if ip:
            score += 0.5 * ip
            reasons.append(f"your earlier feedback on this exercise {0.5 * ip:+.2f}")
        cp = max((prefs.get(_cat_pref_key(c), 0.0) for c in it.categories), default=0.0)
        cn = min((prefs.get(_cat_pref_key(c), 0.0) for c in it.categories), default=0.0)
        cpref = cp if abs(cp) >= abs(cn) else cn
        if cpref:
            score += 0.25 * cpref
            reasons.append(f"your feedback on this kind of suggestion {0.25 * cpref:+.2f}")
        repeats = sum(1 for r in ctx.recent_item_ids if r == it.id)
        if repeats:
            score -= config.REPEAT_PENALTY * repeats
            reasons.append(f"suggested recently {-config.REPEAT_PENALTY * repeats:+.2f}")
        suggestions.append(Suggestion(it, round(score, 3), reasons))

    # deterministic order: score desc, then id
    suggestions.sort(key=lambda s: (-s.score, s.item.id))
    out = suggestions[:max_items]

    # high distress: make sure a human-support option is visible even if it ranks lower
    high = (ctx.stress_rating is not None and ctx.stress_rating >= 8) or ctx.mood == "overwhelmed"
    if high and not any(s.item.id in ("talk_to_someone", "professional_support") for s in out):
        for s in suggestions:
            if s.item.id in ("talk_to_someone", "professional_support"):
                s.reasons.append("shown because you reported high distress")
                out = out[:-1] + [s] if len(out) >= max_items else out + [s]
                break
    return out


def update_preferences(prefs: dict[str, float], item: Item, helped: str | None,
                       want_future: bool | None = None) -> dict[str, float]:
    """Return new preference scores after feedback. Bounded, transparent, reversible
    (new feedback moves scores back). ``want_future=False`` pins the item to the lower
    bound, which excludes it from future suggestions until the user changes their mind."""
    new = dict(prefs)
    delta = config.PREF_DELTAS.get(helped or "skipped", 0.0)
    ik = _item_pref_key(item.id)
    new[ik] = _clip(new.get(ik, 0.0) + delta)
    for c in item.categories:
        ck = _cat_pref_key(c)
        new[ck] = _clip(new.get(ck, 0.0) + 0.5 * delta)
    if want_future is True:
        new[ik] = _clip(new[ik] + config.FUTURE_YES_BONUS)
    elif want_future is False:
        new[ik] = -config.PREF_BOUND
    return new


def reset_preferences() -> dict[str, float]:
    return {}


def _clip(x: float) -> float:
    return max(-config.PREF_BOUND, min(config.PREF_BOUND, x))
