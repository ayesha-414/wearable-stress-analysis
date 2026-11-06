"""Safety layer: crisis screening, region-configurable resources, prompt policy.

IMPORTANT LIMITS
* ``screen_text`` is a conservative keyword BACKSTOP. It will miss many ways of
  expressing distress (false negatives) and can fire on harmless text (false
  positives). It must never be the only route to help: every page also shows a
  persistent "talk to someone now" control.
* Resource details are data (``data/crisis_resources.json``) and must be verified
  by the project owner. There is deliberately no default region.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import config

# Deliberately small and conservative. Extend with supervisor/clinical advice.
_CRISIS_PATTERNS = [
    r"\bkill (?:myself|me)\b",
    r"\bend (?:my|it all|my own) life\b",
    r"\bsuicid\w*",
    r"\bwant(?:ed)? to die\b",
    r"\bdon'?t want to (?:live|be alive|exist)\b",
    r"\bno reason to (?:live|go on)\b",
    r"\bhurt(?:ing)? myself\b",
    r"\bharm(?:ing)? myself\b",
    r"\bself[- ]?harm\w*",
    r"\bcut(?:ting)? myself\b",
    r"\bbetter off (?:dead|without me)\b",
    r"\btake my own life\b",
    r"\boverdos\w*",
]
_CRISIS_RE = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]

CRISIS_MESSAGE = (
    "Thank you for telling me. What you are feeling matters, and you deserve support "
    "from a real person right now. I am not able to help with this safely, so I am "
    "going to stop the exercises and suggestions. If you might act on thoughts of harming "
    "yourself, or someone is in immediate danger, please contact your local emergency "
    "services now, or reach out to someone you trust and stay with them."
)
SUPPORT_NOTE = (
    "NeuroTwin is a research prototype and cannot replace a mental health professional "
    "or emergency help."
)


def screen_text(text: str | None) -> bool:
    """True if ``text`` contains a crisis phrase. Backstop only (see module docstring)."""
    if not text:
        return False
    return any(rx.search(text) for rx in _CRISIS_RE)


def load_resources(path=None) -> dict:
    with open(path or config.CRISIS_RESOURCES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_resources(region: str | None, resources: dict | None = None) -> dict:
    """Resources for ``region``; GENERIC if unset/unknown. Never raises.

    Returns {"region", "name", "resources": [...], "needs_region": bool,
             "unverified": bool}. ``unverified`` is True while any entry has
    verified_on == null (the project owner has not yet checked it).
    """
    res = resources if resources is not None else load_resources()
    key = (region or "").upper()
    chosen = res.get(key)
    needs_region = chosen is None
    if chosen is None:
        key, chosen = "GENERIC", res["GENERIC"]
    items = list(chosen["resources"])
    if key != "GENERIC":
        items += list(res["GENERIC"]["resources"])
    return {
        "region": key,
        "name": chosen["name"],
        "resources": items,
        "needs_region": needs_region,
        "unverified": any(not r.get("verified_on") for r in items),
    }


def available_regions(resources: dict | None = None) -> list[str]:
    res = resources if resources is not None else load_resources()
    return [k for k in res if k != "GENERIC"]


# ---------------------------------------------------------------- prompt policy
@dataclass
class PromptPolicy:
    """Opt-in, rate-limited prompting. Defaults are conservative."""
    enabled: bool = False                       # opt-in: OFF until the user turns it on
    min_interval_min: int = config.PROMPT_MIN_INTERVAL_MIN
    daily_cap: int = config.PROMPT_DAILY_CAP
    quiet_start_hour: int = config.QUIET_HOURS[0]
    quiet_end_hour: int = config.QUIET_HOURS[1]
    snoozed_until: datetime | None = None
    history: list[datetime] = field(default_factory=list)

    def in_quiet_hours(self, now: datetime) -> bool:
        h = now.hour
        if self.quiet_start_hour <= self.quiet_end_hour:
            return self.quiet_start_hour <= h < self.quiet_end_hour
        return h >= self.quiet_start_hour or h < self.quiet_end_hour

    def snooze(self, now: datetime, minutes: int = 240) -> None:
        self.snoozed_until = now + timedelta(minutes=minutes)

    def record(self, now: datetime) -> None:
        self.history.append(now)

    def should_prompt(self, now: datetime, evidence_level: str | None, abstained: bool = False):
        """Return (allowed, reason). Never prompts on weak evidence."""
        if not self.enabled:
            return False, "Prompts are off (opt-in)."
        if abstained or evidence_level != "moderate":
            return False, "Evidence is not strong enough to interrupt you, so no prompt."
        if self.in_quiet_hours(now):
            return False, "Quiet hours."
        if self.snoozed_until and now < self.snoozed_until:
            return False, "Snoozed."
        today = [t for t in self.history if t.date() == now.date()]
        if len(today) >= self.daily_cap:
            return False, "Daily prompt limit reached."
        if self.history and (now - max(self.history)) < timedelta(minutes=self.min_interval_min):
            return False, "Too soon after the last prompt."
        return True, "OK"
