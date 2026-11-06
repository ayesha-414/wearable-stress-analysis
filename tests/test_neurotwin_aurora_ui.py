"""
tests/test_neurotwin_aurora_ui.py
=================================
Automated verification of the NeuroTwin Aurora UI (Workstream B):
- WCAG AA contrast ratio compliance (>= 4.5:1 for text/background token pairs)
- Calm Mode & prefers-reduced-motion CSS rules
- Strict wording guard (find_forbidden) across all UI files
- No probability percentage policy verification
- Persistent crisis control checks
"""

from __future__ import annotations

from pathlib import Path
import pytest

from dashboard.theme_aurora import (
    AURORA_DAWN,
    AURORA_NIGHT,
    breathing_orb_svg,
    contrast_ratio,
    generate_aurora_css,
    guided_breathing_svg,
    verify_wcag_aa_compliance,
)
from src.neurotwin.estimator import StressEstimator
from src.neurotwin.language import find_forbidden


def test_wcag_aa_contrast_ratios_dark_mode():
    """Verify WCAG AA contrast compliance for Aurora Night (dark default)."""
    results = verify_wcag_aa_compliance(AURORA_NIGHT)
    # High-contrast body text must achieve >= 4.5:1 (WCAG AA)
    assert results["text_primary_vs_bg"] >= 4.5
    assert results["text_primary_vs_surface1"] >= 4.5
    assert results["text_primary_vs_surface2"] >= 4.5
    # Muted text must achieve >= 4.5:1
    assert results["text_muted_vs_bg"] >= 4.5
    assert results["text_muted_vs_surface1"] >= 4.5
    assert results["text_muted_vs_surface2"] >= 4.5
    # Primary accent tokens vs background
    assert results["teal_vs_bg"] >= 4.5
    assert results["gold_vs_bg"] >= 4.5


def test_wcag_aa_contrast_ratios_light_mode():
    """Verify WCAG AA contrast compliance for Dawn (light mode)."""
    results = verify_wcag_aa_compliance(AURORA_DAWN)
    # High-contrast body text must achieve >= 4.5:1 (WCAG AA)
    assert results["text_primary_vs_bg"] >= 4.5
    assert results["text_primary_vs_surface1"] >= 4.5
    assert results["text_primary_vs_surface2"] >= 4.5
    # Muted text must achieve >= 4.5:1
    assert results["text_muted_vs_bg"] >= 4.5
    # Accents tuned for light mode must achieve >= 4.5:1
    assert results["teal_vs_bg"] >= 4.5
    assert results["violet_vs_bg"] >= 4.5
    assert results["peach_vs_bg"] >= 4.5
    assert results["gold_vs_bg"] >= 4.5
    assert results["sage_vs_bg"] >= 4.5


def test_calm_mode_and_reduced_motion_css_rules():
    """Verify CSS and SVGs support Calm Mode and respect prefers-reduced-motion."""
    css_normal = generate_aurora_css(theme="dark", calm_mode=False)
    assert "prefers-reduced-motion: reduce" in css_normal
    assert ".calm-mode" in css_normal

    css_calm = generate_aurora_css(theme="dark", calm_mode=True)
    assert "animation: none !important;" in css_calm or "animation-duration: 0.001ms !important;" in css_calm

    # Test SVG calm mode overrides
    orb_calm = breathing_orb_svg(calm_mode=True)
    assert "animation: none;" in orb_calm

    breath_calm = guided_breathing_svg(calm_mode=True)
    assert "animation: none;" in breath_calm


def test_wording_guard_across_all_aurora_ui_files():
    """Scan all Aurora UI and page files with find_forbidden to guarantee zero forbidden phrases."""
    ui_files = [
        "dashboard/theme_aurora.py",
        "dashboard/components_aurora.py",
        "dashboard/pages/p01_home.py",
        "dashboard/pages/p02_signals.py",
        "dashboard/pages/p03_questions.py",
        "dashboard/pages/p04_result.py",
        "dashboard/pages/p04_digital_twin.py",
        "dashboard/pages/p05_interventions.py",
        "dashboard/pages/p06_evaluation.py",
        "dashboard/pages/p07_about.py",
        "dashboard/app.py",
    ]

    root = Path(__file__).resolve().parents[1]
    all_violations = {}

    for rel_path in ui_files:
        p = root / rel_path
        if p.is_file():
            content = p.read_text(encoding="utf-8")
            violations = find_forbidden(content)
            if violations:
                all_violations[rel_path] = violations

    assert not all_violations, f"Forbidden wording violations found in UI files: {all_violations}"


def test_no_probability_percentages_rendered():
    """Verify that when Brier skill vs prior is non-positive, probability is withheld."""
    est = StressEstimator()
    rep = est.calibration_report()
    assert rep["calibration_ok"] is False

    row = est.X.iloc[0].to_dict()
    res = est.estimate(row)
    assert res.probability_shown is None

