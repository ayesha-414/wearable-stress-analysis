"""
dashboard/theme_aurora.py
=========================
NeuroTwin "Aurora" theme tokens, WCAG AA contrast validation, CSS generator,
and pure local SVG components (Breathing Orb, Evidence Mist, Guided Breathing).

Design Principles:
- Dark default ("Aurora Night"): deep night #0A0F1F meeting polar aurora accents.
- Light mode ("Dawn"): warm parchment #FBF7F1 with darkened accents for >= 4.5:1 contrast.
- Visual Honesty: no alarm colors (no red for stress), no percentage rings or gauges.
- Uncertainty Mist: backdrop filter and radial gradients that visually communicate limited evidence.
- Accessibility: WCAG AA contrast (>= 4.5:1), prefers-reduced-motion, and instant Calm Mode toggle.
- Offline & Local: strict system font stacks, zero external HTTP/CDN requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import streamlit as st


# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaletteTokens:
    name: str
    bg: str
    surface_1: str
    surface_2: str
    border: str
    text_primary: str
    text_muted: str
    teal: str       # Primary: fresh, non-clinical (Measured)
    violet: str     # Secondary: reflection, inference (Estimated)
    peach: str      # Tertiary: human, warm (Reported)
    gold: str       # Soft gold: attention, caveats, crisis help
    sage: str       # Soft sage: low-stress, baseline, recovery


AURORA_NIGHT = PaletteTokens(
    name="aurora_night",
    bg="#0A0F1F",
    surface_1="#121A33",
    surface_2="#1B2547",
    border="rgba(56, 225, 204, 0.20)",
    text_primary="#EAF1FF",
    text_muted="#A9B8DC",
    teal="#38E1CC",
    violet="#B3A5FF",
    peach="#FFB38A",
    gold="#FFD98A",
    sage="#7BE3A5",
)

AURORA_DAWN = PaletteTokens(
    name="aurora_dawn",
    bg="#FBF7F1",
    surface_1="#FFFFFF",
    surface_2="#F1EAE0",
    border="rgba(11, 122, 112, 0.22)",
    text_primary="#1D2440",
    text_muted="#4A5578",
    teal="#0B7A70",
    violet="#5642D1",
    peach="#A8461F",
    gold="#7A5A00",
    sage="#1F7A45",
)


# ---------------------------------------------------------------------------
# WCAG AA Contrast Calculation (Automated)
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert hex string (e.g. #0A0F1F or 0A0F1F) to normalized RGB."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex colour: {hex_str}")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def _srgb_to_linear(c: float) -> float:
    """Convert sRGB component to linear component for relative luminance."""
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_str: str) -> float:
    """Calculate relative luminance per WCAG 2.1 specs."""
    r, g, b = _hex_to_rgb(hex_str)
    rl = _srgb_to_linear(r)
    gl = _srgb_to_linear(g)
    bl = _srgb_to_linear(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def contrast_ratio(hex1: str, hex2: str) -> float:
    """Calculate contrast ratio between two hex colours per WCAG 2.1 (1.0 to 21.0)."""
    l1 = relative_luminance(hex1)
    l2 = relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def verify_wcag_aa_compliance(palette: PaletteTokens) -> dict[str, float]:
    """Verify that body text and accents meet WCAG AA standards (>= 4.5:1 for text, >= 3.0:1 for large/UI)."""
    results = {
        "text_primary_vs_bg": contrast_ratio(palette.text_primary, palette.bg),
        "text_primary_vs_surface1": contrast_ratio(palette.text_primary, palette.surface_1),
        "text_primary_vs_surface2": contrast_ratio(palette.text_primary, palette.surface_2),
        "text_muted_vs_bg": contrast_ratio(palette.text_muted, palette.bg),
        "text_muted_vs_surface1": contrast_ratio(palette.text_muted, palette.surface_1),
        "text_muted_vs_surface2": contrast_ratio(palette.text_muted, palette.surface_2),
        "teal_vs_bg": contrast_ratio(palette.teal, palette.bg),
        "violet_vs_bg": contrast_ratio(palette.violet, palette.bg),
        "peach_vs_bg": contrast_ratio(palette.peach, palette.bg),
        "gold_vs_bg": contrast_ratio(palette.gold, palette.bg),
        "sage_vs_bg": contrast_ratio(palette.sage, palette.bg),
    }
    return results


# ---------------------------------------------------------------------------
# CSS Generator
# ---------------------------------------------------------------------------

def generate_aurora_css(theme: str = "dark", calm_mode: bool = False) -> str:
    """Generate comprehensive Aurora CSS with dark/light themes and Calm Mode / reduced-motion rules."""
    p = AURORA_NIGHT if theme == "dark" else AURORA_DAWN

    animation_disabled_css = """
        * {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
        }
        .aurora-mesh-bg, .breathing-orb, .pulse-ring, .heartbeat, .aurora-node-glow {
            animation: none !important;
        }
    """

    return f"""
    /* =====================================================================
       NeuroTwin Aurora Theme ({p.name})
       Generated with pure CSS & local system typography. Zero external CDNs.
       ===================================================================== */

    :root {{
        --nt-bg: {p.bg};
        --nt-surface-1: {p.surface_1};
        --nt-surface-2: {p.surface_2};
        --nt-border: {p.border};
        --nt-text-primary: {p.text_primary};
        --nt-text-muted: {p.text_muted};
        --nt-teal: {p.teal};
        --nt-violet: {p.violet};
        --nt-peach: {p.peach};
        --nt-gold: {p.gold};
        --nt-sage: {p.sage};
        --nt-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        --nt-font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    }}

    /* Base application background and container */
    .stApp {{
        background-color: var(--nt-bg) !important;
        color: var(--nt-text-primary) !important;
        font-family: var(--nt-font-sans) !important;
    }}

    /* Background Aurora mesh drift (aria-hidden, very slow 45s cycle, low opacity) */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        opacity: {0.0 if calm_mode else (0.08 if theme == "dark" else 0.04)};
        background: radial-gradient(circle at 20% 25%, {p.teal} 0%, transparent 45%),
                    radial-gradient(circle at 80% 30%, {p.violet} 0%, transparent 50%),
                    radial-gradient(circle at 50% 80%, {p.peach} 0%, transparent 45%);
        background-size: 200% 200%;
        animation: auroraDrift 45s ease-in-out infinite alternate;
    }}

    @keyframes auroraDrift {{
        0% {{ background-position: 0% 0%; }}
        50% {{ background-position: 100% 100%; }}
        100% {{ background-position: 0% 0%; }}
    }}

    /* Headers and typography */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--nt-text-primary) !important;
        font-family: var(--nt-font-sans) !important;
        letter-spacing: -0.015em;
        font-weight: 600;
    }}

    p, span, label, div {{
        color: var(--nt-text-primary);
        font-family: var(--nt-font-sans);
    }}

    small, .stCaption, .nt-muted {{
        color: var(--nt-text-muted) !important;
    }}

    code, pre, .katex {{
        font-family: var(--nt-font-mono) !important;
    }}

    /* Sidebar Aurora styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--nt-surface-1) !important;
        border-right: 1px solid var(--nt-border) !important;
    }}

    /* Crisis help persistent gold button */
    .nt-crisis-button {{
        display: block;
        width: 100%;
        background: linear-gradient(135deg, rgba(255, 217, 138, 0.15), rgba(255, 217, 138, 0.05));
        border: 1px solid var(--nt-gold);
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        color: var(--nt-gold) !important;
        font-weight: 600;
        font-size: 0.85rem;
        text-align: center;
        text-decoration: none;
        margin-bottom: 0.8rem;
        transition: background 0.2s ease;
    }}
    .nt-crisis-button:hover {{
        background: rgba(255, 217, 138, 0.22);
        color: var(--nt-gold) !important;
    }}

    /* Three-Layer Provenance Cards */
    .nt-card {{
        background: var(--nt-surface-1);
        border: 1px solid var(--nt-border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        position: relative;
    }}

    /* Layer 1: Measured (crisp cyan border, physical recording) */
    .nt-card-measured {{
        background: var(--nt-surface-1);
        border: 1.5px solid var(--nt-teal);
        border-radius: 12px;
        padding: 1.2rem;
        position: relative;
    }}
    .nt-card-measured .layer-tag {{
        color: var(--nt-teal);
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }}

    /* Layer 2: Estimated (dashed violet border, uncertainty mist backdrop) */
    .nt-card-estimated {{
        background: var(--nt-surface-1);
        border: 1.5px dashed var(--nt-violet);
        border-radius: 12px;
        padding: 1.2rem;
        position: relative;
        backdrop-filter: blur(8px);
    }}
    .nt-card-estimated::after {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 12px;
        pointer-events: none;
        background: radial-gradient(circle at 50% 50%, rgba(179, 165, 255, 0.06), transparent 70%);
    }}
    .nt-card-estimated .layer-tag {{
        color: var(--nt-violet);
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }}

    /* Layer 3: Reported (warm peach solid border, human voice) */
    .nt-card-reported {{
        background: var(--nt-surface-1);
        border: 1.5px solid var(--nt-peach);
        border-radius: 12px;
        padding: 1.2rem;
        position: relative;
    }}
    .nt-card-reported .layer-tag {{
        color: var(--nt-peach);
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }}

    /* Uncertainty Mist Container */
    .nt-mist-container {{
        background: radial-gradient(ellipse at center, rgba(179, 165, 255, 0.08) 0%, rgba(10, 15, 31, 0) 70%);
        border: 1px solid rgba(179, 165, 255, 0.25);
        border-radius: 14px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(6px);
    }}

    /* Journey Ribbon */
    .journey-ribbon {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--nt-surface-1);
        border: 1px solid var(--nt-border);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-bottom: 1.2rem;
        overflow-x: auto;
    }}
    .journey-node {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--nt-text-muted);
        white-space: nowrap;
    }}
    .journey-node.active {{
        color: var(--nt-teal);
    }}
    .journey-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--nt-text-muted);
    }}
    .journey-node.active .journey-dot {{
        background-color: var(--nt-teal);
        box-shadow: 0 0 8px var(--nt-teal);
    }}
    .journey-arrow {{
        color: var(--nt-border);
        font-size: 0.75rem;
    }}

    /* Check-in Speech Bubbles */
    .chat-bubble-agent {{
        background: var(--nt-surface-2);
        border: 1px solid var(--nt-border);
        border-radius: 14px 14px 14px 2px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        max-width: 85%;
        color: var(--nt-text-primary);
        font-size: 0.9rem;
    }}
    .chat-bubble-user {{
        background: linear-gradient(135deg, rgba(255, 179, 138, 0.2), rgba(255, 179, 138, 0.1));
        border: 1px solid var(--nt-peach);
        border-radius: 14px 14px 2px 14px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        margin-left: auto;
        max-width: 85%;
        color: var(--nt-text-primary);
        font-size: 0.9rem;
    }}

    /* Adaptive Recommendation Cards */
    .rec-card {{
        background: var(--nt-surface-1);
        border: 1px solid var(--nt-border);
        border-left: 4px solid var(--nt-teal);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }}

    /* Badges */
    .aurora-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .badge-teal {{
        background: rgba(56, 225, 204, 0.15);
        color: var(--nt-teal);
        border: 1px solid var(--nt-teal);
    }}
    .badge-violet {{
        background: rgba(179, 165, 255, 0.15);
        color: var(--nt-violet);
        border: 1px solid var(--nt-violet);
    }}
    .badge-peach {{
        background: rgba(255, 179, 138, 0.15);
        color: var(--nt-peach);
        border: 1px solid var(--nt-peach);
    }}
    .badge-gold {{
        background: rgba(255, 217, 138, 0.15);
        color: var(--nt-gold);
        border: 1px solid var(--nt-gold);
    }}
    .badge-sage {{
        background: rgba(123, 227, 165, 0.15);
        color: var(--nt-sage);
        border: 1px solid var(--nt-sage);
    }}

    /* Accessibility: Respect prefers-reduced-motion everywhere */
    @media (prefers-reduced-motion: reduce) {{
        {animation_disabled_css}
    }}

    /* Calm Mode Instant Class Override */
    .calm-mode, body.calm-mode, .stApp.calm-mode {{
        {animation_disabled_css}
    }}
    """ + (animation_disabled_css if calm_mode else "")


def inject_aurora_theme(theme: str = "dark", calm_mode: bool = False) -> None:
    """Inject the generated Aurora stylesheet directly into the Streamlit app container."""
    css = generate_aurora_css(theme=theme, calm_mode=calm_mode)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)



def compact_html(html: str) -> str:
    """Strip indentation and blank lines so st.markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in html.splitlines() if line.strip())


def _compact(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*a, **k):
        return compact_html(fn(*a, **k))

    return wrapper

# ---------------------------------------------------------------------------
# Pure SVG Generators (Breathing Orb, Mist Meter, Guided Breathing)
# ---------------------------------------------------------------------------

@_compact
def breathing_orb_svg(state: str = "idle", calm_mode: bool = False) -> str:
    """
    Generate responsive SVG for the Breathing Orb centerpiece.
    - 6-second expansion/contraction animation (4s inhale, 2s exhale wave).
    - When evidence is limited or abstaining: still and cool ('calm water').
    - Never encodes stress level. Decorative only (aria-hidden).
    """
    is_still = calm_mode or state in ("abstain", "insufficient", "still")
    anim_style = "animation: none;" if is_still else "animation: orbBreathe 6s ease-in-out infinite;"

    return f"""
    <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" role="presentation">
        <defs>
            <radialGradient id="auroraOrbGrad" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#38E1CC" stop-opacity="0.85" />
                <stop offset="45%" stop-color="#B3A5FF" stop-opacity="0.45" />
                <stop offset="80%" stop-color="#121A33" stop-opacity="0.1" />
                <stop offset="100%" stop-color="#0A0F1F" stop-opacity="0" />
            </radialGradient>
            <filter id="orbGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="8" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>
        <style>
            @keyframes orbBreathe {{
                0%   {{ transform: scale(0.88); opacity: 0.70; }}
                50%  {{ transform: scale(1.06); opacity: 0.95; }}
                100% {{ transform: scale(0.88); opacity: 0.70; }}
            }}
            .breathing-orb {{
                transform-origin: 100px 100px;
                {anim_style}
            }}
        </style>
        <circle cx="100" cy="100" r="85" fill="url(#auroraOrbGrad)" class="breathing-orb" filter="url(#orbGlow)" />
        <circle cx="100" cy="100" r="48" fill="none" stroke="#38E1CC" stroke-width="1.2" opacity="0.6" stroke-dasharray="3 3" />
        <circle cx="100" cy="100" r="26" fill="#38E1CC" opacity="0.35" class="breathing-orb" />
        <circle cx="100" cy="100" r="4" fill="#EAF1FF" opacity="0.9" />
    </svg>
    """


@_compact
def evidence_mist_meter_svg(level: str = "limited", calm_mode: bool = False) -> str:
    """
    Generate the Evidence Mist Meter SVG.
    Replaces percentage bars with a soft blurred gradient band whose width
    corresponds to the uncertainty interval, not a misleading point estimate.
    """
    level_lower = level.strip().lower()
    if level_lower == "insufficient":
        label_text = "Insufficient Evidence (Abstained)"
        band_width = 30
        band_color = "#A9B8DC"
        opacity = "0.4"
    elif level_lower == "limited":
        label_text = "Limited Evidence (Uncertain)"
        band_width = 70
        band_color = "#B3A5FF"
        opacity = "0.65"
    else:
        label_text = "Moderate Evidence (Laboratory Bound)"
        band_width = 130
        band_color = "#38E1CC"
        opacity = "0.85"

    return f"""
    <svg width="100%" height="42" viewBox="0 0 320 42" xmlns="http://www.w3.org/2000/svg" aria-label="Evidence meter: {label_text}">
        <defs>
            <linearGradient id="mistGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="{band_color}" stop-opacity="0.1" />
                <stop offset="50%" stop-color="{band_color}" stop-opacity="{opacity}" />
                <stop offset="100%" stop-color="{band_color}" stop-opacity="0.1" />
            </linearGradient>
            <filter id="mistBlur" x="-10%" y="-30%" width="120%" height="160%">
                <feGaussianBlur stdDeviation="3" />
            </filter>
        </defs>
        <!-- Background track -->
        <rect x="10" y="8" width="300" height="12" rx="6" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.12)" stroke-width="1" />
        <!-- Soft mist band representing uncertainty span -->
        <rect x="20" y="7" width="{band_width}" height="14" rx="7" fill="url(#mistGrad)" filter="url(#mistBlur)" />
        <rect x="20" y="9" width="{band_width}" height="10" rx="5" fill="{band_color}" opacity="{float(opacity)*0.7:.2f}" />
        <!-- Text label -->
        <text x="12" y="34" font-family="-apple-system, sans-serif" font-size="11" fill="{band_color}" font-weight="600">{label_text}</text>
    </svg>
    """


@_compact
def guided_breathing_svg(calm_mode: bool = False) -> str:
    """
    Generate pure SVG visual for 2-minute paced breathing.
    Cycle: Inhale 4s (expand) -> Hold 4s (pause) -> Exhale 6s (contract).
    Includes clear stop notice and mandatory safety text.
    """
    anim_style = "animation: none;" if calm_mode else "animation: paceCycle 14s ease-in-out infinite;"

    return f"""
    <div style="text-align: center; padding: 1rem 0;">
        <svg width="220" height="220" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" aria-label="Paced breathing visual">
            <defs>
                <radialGradient id="breathGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#38E1CC" stop-opacity="0.7" />
                    <stop offset="70%" stop-color="#B3A5FF" stop-opacity="0.3" />
                    <stop offset="100%" stop-color="#0A0F1F" stop-opacity="0" />
                </radialGradient>
            </defs>
            <style>
                @keyframes paceCycle {{
                    0%   {{ transform: scale(0.65); }} /* Inhale start */
                    28%  {{ transform: scale(1.10); }} /* Inhale end (4s / 14s) */
                    57%  {{ transform: scale(1.10); }} /* Hold end (4s / 14s) */
                    100% {{ transform: scale(0.65); }} /* Exhale end (6s / 14s) */
                }}
                .breath-circle {{
                    transform-origin: 110px 110px;
                    {anim_style}
                }}
            </style>
            <circle cx="110" cy="110" r="95" fill="none" stroke="rgba(56, 225, 204, 0.25)" stroke-width="1.5" stroke-dasharray="4 4" />
            <circle cx="110" cy="110" r="75" fill="url(#breathGrad)" class="breath-circle" />
            <circle cx="110" cy="110" r="30" fill="none" stroke="#EAF1FF" stroke-width="1.5" opacity="0.8" />
            <text x="110" y="114" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#EAF1FF" font-weight="600">Breathe</text>
        </svg>
        <div style="font-size: 0.85rem; color: #FFD98A; margin-top: 0.5rem; font-weight: 500;">
            Safety notice: Stop if you feel dizzy or lightheaded; never hold your breath uncomfortably.
        </div>
    </div>
    """

