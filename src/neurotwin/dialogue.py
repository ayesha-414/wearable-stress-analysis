"""Scripted, consent-aware conversational check-in.

Deliberately NOT a generative language model: every prompt is written in advance,
reviewable and testable. The conversation is user-led and never pressures disclosure:
every step has 'prefer not to say / skip', the user can decline at any time, correct the
system's interpretation, and say they only want to talk.

Free text, if the user chooses to type, is screened for crisis phrases (a backstop, see
safety.py) and is NEVER analysed or used to infer a stressor. Free text is only offered if
the user has consented to storing it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from . import safety
from .language import assert_clean


@dataclass(frozen=True)
class Option:
    id: str
    label: str


@dataclass(frozen=True)
class Node:
    id: str
    prompt: str
    options: tuple[Option, ...] = ()
    kind: str | None = None            # reported kind this node records
    rating: bool = False               # 0-10 slider with 'skip'
    free_text: bool = False            # optional typed answer (only if consented)
    next: dict = field(default_factory=dict)   # option id -> node id
    default_next: str = "close"


SKIP = Option("skip", "I'd rather not say")


def _n(id, prompt, options=(), **kw):
    assert_clean(prompt)
    for o in options:
        assert_clean(o.label)
    return Node(id, prompt, tuple(options), **kw)


NODES: dict[str, Node] = {n.id: n for n in [
    _n("feeling", "How are you feeling right now? There are no right or wrong answers.",
       [Option("calm_ok", "Calm or okay"), Option("tense", "Tense or uneasy"),
        Option("overwhelmed", "Overwhelmed"), Option("tired", "Tired or drained"),
        Option("low", "Low or down"), Option("other", "Something else"), SKIP],
       kind="mood", default_next="rating"),
    _n("rating", "If you would like to, how stressed do you feel right now, from 0 (not at all) to 10 (extremely)?",
       kind="stress_rating", rating=True, default_next="wants"),
    _n("wants", "What would be most helpful right now?",
       [Option("talk", "Talk it through"), Option("exercise", "A short exercise"),
        Option("suggestion", "A practical suggestion"), Option("nothing", "Nothing right now")],
       kind="wants",
       next={"talk": "reason", "exercise": "recommend", "suggestion": "reason", "nothing": "close"},
       default_next="close"),
    _n("reason", "Has something happened recently that might be adding to this? You can choose one, "
                 "tell me in your own words, say you are not sure, or skip.",
       [Option("work", "Work"), Option("studies", "Studies"), Option("relationships", "Relationships"),
        Option("health", "Health"), Option("finances", "Money"), Option("sleep", "Lack of sleep"),
        Option("other", "Something else"), Option("unknown", "I don't know"), SKIP],
       kind="reason", free_text=True, default_next="similar_before"),
    _n("similar_before", "Have you felt something like this before?",
       [Option("yes", "Yes"), Option("no", "No"), Option("not_sure", "Not sure"), SKIP],
       kind="similar_before", default_next="agreement"),
    _n("agreement", "The wristband estimate may not match how you feel. How well does it match your experience?",
       [Option("matches", "It matches"), Option("doesnt_match", "It doesn't match"),
        Option("not_sure", "Not sure"), SKIP],
       kind="estimate_agreement", default_next="recommend"),
    _n("recommend", "Thank you. Here are a few optional ideas. You can skip any of them.",
       default_next="fb_helped"),
    _n("fb_helped", "Did that help?",
       [Option("yes", "Yes"), Option("somewhat", "A little"), Option("no", "Not really"),
        Option("skipped", "Skip")], default_next="fb_feel"),
    _n("fb_feel", "How do you feel now, from 0 (not at all stressed) to 10 (extremely stressed)?",
       rating=True, default_next="fb_different"),
    _n("fb_different", "Would you like to try something different?",
       [Option("yes", "Yes"), Option("no", "No"), Option("skipped", "Skip")], default_next="fb_future"),
    _n("fb_future", "Would you like this type of suggestion again in future?",
       [Option("yes", "Yes"), Option("no", "No"), Option("skipped", "Skip")], default_next="close"),
    _n("close", "Thank you for checking in. You can come back any time. "
                "Remember this is a research prototype and not a replacement for professional support."),
]}

INTRO_ESTIMATE = ("I noticed a pattern in your wristband signals that the model associates with higher "
                  "self-reported stress in a laboratory dataset. This is only an estimate and may be wrong. ")
INTRO_USER = "Thanks for checking in. "


class CheckInSession:
    """One conversation. Collects REPORTED items; never writes anything itself."""

    def __init__(self, trigger: str = "user", allow_free_text: bool = False):
        assert trigger in ("user", "estimate")
        self.id = uuid.uuid4().hex[:12]
        self.trigger = trigger
        self.allow_free_text = allow_free_text
        self.node_id = "feeling"
        self.items: list[tuple[str, object]] = []      # (kind, value) reported by the person
        self.feedback: dict = {}
        self.crisis = False
        self.declined = False

    # -------------------------------------------------------------- presentation
    def intro(self) -> str:
        return (INTRO_ESTIMATE if self.trigger == "estimate" else INTRO_USER)

    @property
    def node(self) -> Node:
        return NODES[self.node_id]

    @property
    def finished(self) -> bool:
        return self.crisis or self.declined or self.node_id == "close"

    def crisis_message(self) -> str:
        return safety.CRISIS_MESSAGE

    # -------------------------------------------------------------- actions
    def urgent(self) -> None:
        """Person pressed 'I need urgent help' (available at every step)."""
        self.crisis = True

    def decline(self) -> None:
        """Person ends the conversation. Nothing further is asked."""
        self.declined = True

    def answer(self, option: str | None = None, text: str | None = None,
               rating: int | None = None) -> str:
        """Record an answer and move on. Returns the next node id."""
        if self.finished:
            return self.node_id
        node = self.node
        # crisis screen on any typed text (backstop)
        if text and safety.screen_text(text):
            self.crisis = True
            return self.node_id
        if node.free_text and text and self.allow_free_text:
            self.items.append(("free_text", text.strip()))
        if node.rating:
            if rating is not None:
                r = max(0, min(10, int(rating)))
                if node.kind:
                    self.items.append((node.kind, r))
                else:
                    self.feedback["feel_after"] = r
            return self._advance(node, "rating")
        if option is None or option == "skip":
            return self._advance(node, option or "skip")
        valid = {o.id for o in node.options}
        if option not in valid:
            raise ValueError(f"Unknown option {option!r} for step {node.id!r}")
        if node.kind:
            self.items.append((node.kind, option))
        elif node.id.startswith("fb_"):
            self.feedback[node.id[3:]] = option
        return self._advance(node, option)

    def _advance(self, node: Node, option: str) -> str:
        nxt = node.next.get(option, node.default_next)
        # estimate-agreement only makes sense if the estimate triggered the conversation
        if nxt == "agreement" and self.trigger != "estimate":
            nxt = "recommend"
        self.node_id = nxt
        return nxt

    # -------------------------------------------------------------- derived values
    def value(self, kind: str):
        for k, v in reversed(self.items):
            if k == kind:
                return v
        return None
