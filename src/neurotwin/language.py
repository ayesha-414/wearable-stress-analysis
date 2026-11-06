"""Wording guard.

Every user-facing string produced by this package can be scanned with
``find_forbidden``. The patterns encode the project's claims register
(see docs/neurotwin/CLAIMS_REGISTER.md): no diagnosis, no causal claims from
sensors, no efficacy promises, no 'normal range' language for personal baselines.
"""
from __future__ import annotations

import re

FORBIDDEN_PATTERNS = [
    r"\byou are (?:a )?stressed\b",
    r"\byou(?:'re| are) (?:clearly |definitely )?(?:anxious|depressed)\b",
    r"\byou have (?:stress|anxiety|depression|a disorder)\b",
    r"\bstressed (?:person|individual)\b",
    r"\byour stress is (?:caused|due)\b",
    r"\bcaused by your\b",
    r"\bbecause you have an? (?:exam|deadline|test)\b",
    r"\bwill (?:reduce|lower|cure|fix|remove|relieve) your stress\b",
    r"\bguarantee[sd]?\b",
    r"\bclinically (?:validated|proven|tested)\b",
    r"\bproven to\b",
    r"\bnormal range\b",
    r"\bhealthy range\b",
    r"\byour normal\b",
    r"\bwe diagnos(?:e|ed)\b",
    r"\bthis (?:diagnoses|diagnoses you)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]


def find_forbidden(text: str) -> list[str]:
    """Return the forbidden patterns that match ``text`` (empty list = clean)."""
    return [p.pattern for p in _COMPILED if p.search(text or "")]


def assert_clean(text: str) -> str:
    bad = find_forbidden(text)
    if bad:
        raise ValueError(f"Forbidden wording {bad!r} in: {text!r}")
    return text
