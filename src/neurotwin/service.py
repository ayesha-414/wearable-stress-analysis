"""Orchestration between dialogue, recommender and store (no Streamlit here, so it is testable)."""
from __future__ import annotations

from .catalogue import Item, load_catalogue
from .dialogue import CheckInSession
from .recommender import Context, Suggestion, recommend, update_preferences
from .store import ConsentError, Store


def context_from_session(session: CheckInSession, recent_exertion: bool = False,
                         declined_items=(), declined_tags=(), recent_item_ids=()) -> Context:
    return Context(
        crisis=session.crisis,
        mood=session.value("mood"),
        stress_rating=session.value("stress_rating"),
        reason=session.value("reason"),
        wants=session.value("wants"),
        recent_exertion=recent_exertion,
        declined_items=frozenset(declined_items),
        declined_tags=frozenset(declined_tags),
        recent_item_ids=tuple(recent_item_ids),
    )


def save_checkin(store: Store, user_id: str, session: CheckInSession,
                 shown: list[Suggestion] | None = None) -> dict:
    """Write what the person REPORTED (and the suggestions SHOWN) subject to consent.
    Crisis text is never in session.items, so it is never stored."""
    saved, skipped = [], []
    for kind, value in session.items:
        try:
            saved.append(store.add_reported(user_id, kind, value, session_id=session.id))
        except ConsentError:
            skipped.append(kind)
    rec_ids: dict[str, int] = {}
    for rank, s in enumerate(shown or [], start=1):
        try:
            rec_ids[s.item.id] = store.add_recommendation(user_id, s.item.id, rank, s.reasons, session.id)
        except ConsentError:
            skipped.append("recommendations")
            break
    return {"saved_reported_ids": saved, "skipped_kinds": sorted(set(skipped)), "rec_ids": rec_ids}


def record_feedback(store: Store, user_id: str, item: Item, helped: str | None,
                    feel_after: int | None = None, want_different: bool | None = None,
                    want_future: bool | None = None, rec_id: int | None = None) -> dict:
    """Store the person's feedback and adapt preferences (both need feedback consent)."""
    prefs = store.get_preferences(user_id)
    new = update_preferences(prefs, item, helped, want_future)
    stored = True
    try:
        if rec_id is not None:
            store.add_feedback(user_id, rec_id, helped, feel_after, want_different, want_future)
        for k, v in new.items():
            if prefs.get(k) != v:
                store.set_preference(user_id, k, v)
    except ConsentError:
        stored = False
        new = prefs                               # no consent -> no learning, no storage
    return {"stored": stored, "preferences": new}


def preview_recommendations(store: Store, user_id: str, ctx: Context) -> list[Suggestion]:
    prefs = store.get_preferences(user_id)
    recent = [r["item_id"] for r in store.list_recommendations(user_id)][-5:]
    ctx.recent_item_ids = tuple(recent)
    return recommend(ctx, prefs, load_catalogue())
