"""Loader and validator for the sourced coping catalogue."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config
from .language import find_forbidden

REQUIRED = ("id", "title", "categories", "text", "sources", "evidence_note", "contraindications")


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    categories: tuple[str, ...]
    text: str
    sources: tuple[dict, ...]
    evidence_note: str
    contraindications: str
    priority: float = 0.5
    excluded_when: tuple[str, ...] = ()
    link: str | None = None


def load_catalogue(path=None) -> dict[str, Item]:
    with open(path or config.CATALOGUE_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    items = {}
    for d in raw["items"]:
        missing = [k for k in REQUIRED if not d.get(k)]
        if missing:
            raise ValueError(f"Catalogue item {d.get('id')!r} is missing {missing}")
        items[d["id"]] = Item(
            id=d["id"], title=d["title"], categories=tuple(d["categories"]), text=d["text"],
            sources=tuple(d["sources"]), evidence_note=d["evidence_note"],
            contraindications=d["contraindications"], priority=float(d.get("priority", 0.5)),
            excluded_when=tuple(d.get("excluded_when", ())), link=d.get("link"))
    return items


def validate(items: dict[str, Item]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    for it in items.values():
        text = " ".join([it.title, it.text, it.evidence_note, it.contraindications])
        if find_forbidden(text):
            problems.append(f"{it.id}: forbidden wording {find_forbidden(text)}")
        if not any(s.get("url") for s in it.sources):
            problems.append(f"{it.id}: no source URL")
    return problems
