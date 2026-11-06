"""Scripted SIMULATED users: SOFTWARE VERIFICATION ONLY.

These simulations check that the feedback-adaptation code behaves as specified (it moves
suggestions toward what a simulated user was set up to like, and respects exclusions). The
simulated users' 'preferences' are invented by this script, so the result says nothing about
whether adaptation helps real people. It is never to be presented as evidence of
effectiveness, and simulated records must never be written to a real user's profile.
"""
from __future__ import annotations

import random
from statistics import mean, stdev

from . import config
from .catalogue import load_catalogue
from .recommender import Context, recommend, update_preferences

LABEL = "SIMULATED USERS - software verification only; not evidence of effectiveness."
REASONS = ["work", "studies", "relationships", "health", "finances", "sleep", "other", "unknown"]
MOODS = ["overwhelmed", "tense", "tired", "low", "calm_ok"]


def _make_user(rng: random.Random, item_ids: list[str]) -> dict[str, float]:
    return {i: rng.choice([0.05, 0.2, 0.5, 0.8, 0.9]) for i in item_ids}


def _session_context(rng: random.Random) -> Context:
    return Context(mood=rng.choice(MOODS), reason=rng.choice(REASONS), stress_rating=rng.randint(3, 8))


def run(n_users: int = 40, rounds: int = 30, seed: int = 7) -> dict:
    cat = load_catalogue()
    ids = sorted(cat)
    results = {"static": [], "adaptive": []}
    for u in range(n_users):
        rng_user = random.Random(f"{seed}-user-{u}")
        like = _make_user(rng_user, ids)
        for strategy in ("static", "adaptive"):
            rng = random.Random(f"{seed}-ctx-{u}")          # identical contexts for both strategies
            rng_fb = random.Random(f"{seed}-fb-{u}")
            prefs: dict[str, float] = {}
            late_yes = []
            recent: list[str] = []
            for r in range(rounds):
                ctx = _session_context(rng)
                ctx.recent_item_ids = tuple(recent[-3:])
                sugg = recommend(ctx, prefs if strategy == "adaptive" else {}, cat)
                if not sugg:
                    continue
                top = sugg[0].item
                recent.append(top.id)
                yes = rng_fb.random() < like[top.id]
                if r >= rounds - 10:
                    late_yes.append(1.0 if yes else 0.0)
                if strategy == "adaptive":
                    prefs = update_preferences(prefs, top, "yes" if yes else "no")
            results[strategy].append(mean(late_yes) if late_yes else 0.0)
    diffs = [a - s for a, s in zip(results["adaptive"], results["static"])]
    se = stdev(diffs) / (len(diffs) ** 0.5) if len(diffs) > 1 else 0.0
    return {
        "label": LABEL, "n_simulated_users": n_users, "rounds_per_user": rounds, "seed": seed,
        "metric": "share of simulated 'helped' answers to the top suggestion in the last 10 rounds",
        "static_mean": round(mean(results["static"]), 3),
        "adaptive_mean": round(mean(results["adaptive"]), 3),
        "paired_difference_mean": round(mean(diffs), 3),
        "paired_difference_95ci": [round(mean(diffs) - 1.96 * se, 3), round(mean(diffs) + 1.96 * se, 3)],
        "caveat": ("The simulated users' preferences were invented by the script, so this only verifies "
                   "that the adaptation code does what it is specified to do."),
    }
