"""NeuroTwin theme injection and styling utilities."""

from pathlib import Path
import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent / "neurotwin.css"


def inject_neurotwin_theme() -> None:
    """Inject the official NeuroTwin CSS stylesheet into the active Streamlit app."""
    if _CSS_PATH.is_file():
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def metric_card(title: str, value: str, subtitle: str = "") -> str:
    """Return HTML string for a themed metric card."""
    sub = f"<div class='sub'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div class='metric-card'>"
        f"<h4>{title}</h4>"
        f"<p>{value}</p>"
        f"{sub}"
        f"</div>"
    )


def evidence_badge(label: str) -> str:
    """Return HTML badge for an evidence classification."""
    clean = label.strip()
    cls_map = {
        "Matches standard": "nt-badge-standard",
        "Custom derived": "nt-badge-custom",
        "Heuristic approximation": "nt-badge-heuristic",
        "Uncertain": "nt-badge-uncertain",
    }
    cls = cls_map.get(clean, "nt-badge-custom")
    return f"<span class='nt-badge {cls}'>{clean}</span>"

