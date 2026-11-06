"""
dashboard/components_aurora.py
==============================
Reusable Aurora visual and functional components for NeuroTwin AI:
- Journey Ribbon (Detect -> Explain -> Ask -> Understand -> Recommend -> Learn)
- Three-Layer Provenance Card (Measured, Estimated, Reported)
- Evidence Mist Meter
- Signals & Features Constellation (sensor grouping with units and mini sparkline bars)
- Guided Breathing Interactive Widget with safety notices
- Adaptive Recommendation Card with feedback and expandable reasoning
- V1 vs V2 Side-by-Side Model Transparency Panels
- §9.4 Programmatic Decision Rule Transparency Card
- What This Model Cannot Do limitation banner
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Any

from dashboard.theme_aurora import (
    breathing_orb_svg,
    evidence_mist_meter_svg,
    guided_breathing_svg,
)


JOURNEY_STEPS = ["Detect", "Explain", "Ask", "Understand", "Recommend", "Learn"]


def render_journey_ribbon(current_step: str = "Detect") -> None:
    """Render the 6-step Journey Ribbon showing where the user is in their twin journey."""
    cur = current_step.strip().capitalize()
    html_nodes = []

    for i, step in enumerate(JOURNEY_STEPS):
        is_active = (step.lower() == cur.lower())
        active_cls = "active" if is_active else ""
        html_nodes.append(
            f"<div class='journey-node {active_cls}'>"
            f"<span class='journey-dot'></span>"
            f"<span>{step}</span>"
            f"</div>"
        )
        if i < len(JOURNEY_STEPS) - 1:
            html_nodes.append("<span class='journey-arrow'>➔</span>")

    ribbon_html = f"<div class='journey-ribbon'>{''.join(html_nodes)}</div>"
    st.markdown(ribbon_html, unsafe_allow_html=True)


def render_three_layer_card(
    measured_content: str,
    estimated_content: str,
    reported_content: str,
) -> None:
    """
    Render the tactile Three-Layer comparison card:
    - Measured: Left, crisp cyan border, hardware raw signals.
    - Estimated: Middle, dashed violet border with uncertainty mist, explicitly hypothetical ML inference.
    - Reported: Right, warm peach solid border, participant's own subjective appraisals.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="nt-card-measured">
                <div class="layer-tag">1. Measured (Hardware Signals)</div>
                <div style="margin-top: 0.6rem; font-size: 0.88rem; line-height: 1.45;">
                    {measured_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="nt-card-estimated">
                <div class="layer-tag">2. Estimated (Statistical Model)</div>
                <div style="margin-top: 0.6rem; font-size: 0.88rem; line-height: 1.45;">
                    {estimated_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="nt-card-reported">
                <div class="layer-tag">3. Reported (Your Appraisals)</div>
                <div style="margin-top: 0.6rem; font-size: 0.88rem; line-height: 1.45;">
                    {reported_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_mist(level: str = "limited", calm_mode: bool = False) -> None:
    """Render the Evidence Mist meter replacing numeric percentage gauges."""
    svg = evidence_mist_meter_svg(level=level, calm_mode=calm_mode)
    st.markdown(svg, unsafe_allow_html=True)


def render_signal_constellation(row: dict[str, Any]) -> None:
    """
    Render features grouped by sensor with real units, mini SVG bars, and missing badges.
    Sensors: Electrodermal (EDA), Heart Rate (HR), Skin Temperature (TEMP), Motion (ACC), Beat Intervals (HRV).
    """
    sensors = {
        "⚡ Electrodermal (EDA)": [
            ("Average EDA", "eda_mean", "µS", 0.0, 15.0),
            ("EDA Variability", "eda_std", "µS", 0.0, 5.0),
            ("SCL Baseline Proxy", "eda_scl_mean", "µS", 0.0, 15.0),
            ("SCR Peaks", "eda_scr_count", "peaks", 0.0, 30.0),
        ],
        "❤️ Heart Rate (HR)": [
            ("Mean Heart Rate", "hr_mean", "BPM", 40.0, 160.0),
            ("Heart Rate Spread", "hr_std", "BPM", 0.0, 30.0),
            ("Heart Rate Range", "hr_range", "BPM", 0.0, 50.0),
            ("Beat Difference RMS", "hr_diff_rms", "BPM", 0.0, 25.0),
        ],
        "🌡️ Skin Temperature (TEMP)": [
            ("Mean Temperature", "temp_mean", "°C", 25.0, 38.0),
            ("Temperature Slope", "temp_slope", "°C/s", -0.1, 0.1),
            ("Temperature Spread", "temp_std", "°C", 0.0, 2.0),
        ],
        "🏃 Motion & Accelerometry (ACC)": [
            ("Mean Movement", "acc_mean", "g", 0.0, 100.0),
            ("Peak Movement", "acc_max", "g", 0.0, 200.0),
            ("Movement Spread", "acc_std", "g", 0.0, 50.0),
        ],
        "⏱️ Beat Intervals (HRV)": [
            ("Mean Beat Interval", "hrv_mean_ibi", "ms", 400.0, 1200.0),
            ("Interval Spread (SDNN)", "hrv_sdnn", "ms", 0.0, 150.0),
            ("Short-Term Spread (RMSSD)", "hrv_rmssd", "ms", 0.0, 150.0),
            ("Large Changes (pNN50)", "hrv_pnn50", "%", 0.0, 60.0),
        ],
    }

    cols = st.columns(len(sensors))
    for col, (sensor_name, feature_list) in zip(cols, sensors.items()):
        with col:
            st.markdown(f"<div style='font-weight:700; font-size:0.85rem; color:#38E1CC; margin-bottom:0.4rem;'>{sensor_name}</div>", unsafe_allow_html=True)
            for label, key, unit, min_val, max_val in feature_list:
                val = row.get(key, None)
                if val is None or pd.isna(val):
                    badge = "<span class='aurora-badge' style='background:rgba(169,184,220,0.15); color:#A9B8DC;'>unmeasured</span>"
                    st.markdown(f"<div style='font-size:0.8rem; margin-bottom:0.35rem;'>{label}: {badge}</div>", unsafe_allow_html=True)
                else:
                    v_float = float(val)
                    ratio = min(max((v_float - min_val) / (max_val - min_val + 1e-6), 0.05), 1.0)
                    bar_w = int(ratio * 70)
                    bar_svg = f"<svg width='70' height='6' style='vertical-align:middle;'><rect width='70' height='6' rx='3' fill='rgba(255,255,255,0.08)'/><rect width='{bar_w}' height='6' rx='3' fill='#38E1CC'/></svg>"
                    st.markdown(
                        f"<div style='font-size:0.8rem; margin-bottom:0.4rem; display:flex; justify-content:space-between; align-items:center;'>"
                        f"<span>{label}</span>"
                        f"<span><b>{v_float:.1f}</b> <small style='color:#A9B8DC;'>{unit}</small> {bar_svg}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


def render_breathing_guide_widget(calm_mode: bool = False, key_prefix: str = "breathing") -> None:
    """Render interactive 2-minute paced breathing visual with clear safety notices."""
    st.markdown("### 🫁 Guided Paced Breathing")
    st.markdown(
        "A steady, gentle breathing pattern (4 seconds in, 4 seconds hold, 6 seconds out) "
        "supports autonomic balance during cognitive or emotional strain."
    )

    svg = guided_breathing_svg(calm_mode=calm_mode)
    st.markdown(svg, unsafe_allow_html=True)


def render_v1_v2_comparison() -> None:
    """Render side-by-side transparent cards comparing v1 (original mapping) vs v2 (corrected mapping)."""
    st.markdown("### 🔬 Model Investigation: Original Mapping (v1) vs Corrected Mapping (v2)")
    st.caption("All comparisons evaluated under identical 35-participant Leave-One-Subject-Out (LOSO) cross-validation.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="nt-card" style="border-top: 3px solid #A9B8DC;">
                <div style="font-weight:700; font-size:1.05rem; color:#EAF1FF; margin-bottom:0.4rem;">
                    v1 — Original Protocol Mapping (Reference Baseline)
                </div>
                <div style="font-size:0.83rem; color:#A9B8DC; line-height:1.5;">
                    <p><b>Boundary mapping:</b> 1-off tag alignment (tasks & rests swapped for Stroop, TMCT, Opinion, Subtract).</p>
                    <p><b>Sensor alignment:</b> HR start lag (10s) left unshifted in legacy frame.</p>
                    <p><b>Window duration:</b> Variable 24s to 853s windows; strong duration artifacts (e.g. <i>temp_std</i> r = 0.79).</p>
                    <hr style="border-color:rgba(255,255,255,0.08); margin:0.6rem 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>Accuracy (Pooled):</span> <b>58.9%</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>Balanced Accuracy:</span> <b>52.6%</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>ROC-AUC:</span> <b>0.527</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>Majority Class Baseline:</span> <b>61.6%</b>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Brier Skill Score:</span> <b>Negative</b> (worse than prior)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="nt-card" style="border-top: 3px solid #38E1CC;">
                <div style="font-weight:700; font-size:1.05rem; color:#EAF1FF; margin-bottom:0.4rem;">
                    v2 — Corrected Notebook Mapping (Primary Window B)
                </div>
                <div style="font-size:0.83rem; color:#A9B8DC; line-height:1.5;">
                    <p><b>Boundary mapping:</b> Notebook-consistent stage intervals (100% V1, 94.4% V2 verified in audit).</p>
                    <p><b>Sensor alignment:</b> +10s HR timing alignment to EDA timeline.</p>
                    <p><b>Window duration:</b> Duration-matched 30s windows anchored at stage start (eliminates duration confounds).</p>
                    <hr style="border-color:rgba(255,255,255,0.08); margin:0.6rem 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>Accuracy (Pooled):</span> <b>57.0%</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>Balanced Accuracy:</span> <b>51.1%</b> (95% CI: 44.2%–58.1%)
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>ROC-AUC:</span> <b>0.484</b> (95% CI: 0.387–0.579)
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span>Majority Class Baseline:</span> <b>61.6%</b>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Brier Skill Score:</span> <b>-0.149</b> (calibrates to prior)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_decision_rule_panel() -> None:
    """Render the programmatic decision rule audit panel per §9.4 of the plan."""
    st.markdown(
        """
        <div class="nt-card" style="border-left: 4px solid #FFD98A;">
            <div style="font-weight:700; font-size:1.0rem; color:#FFD98A; margin-bottom:0.5rem;">
                Programmatic Decision Rule (§9.4) Audit Verdict: RETAIN BASELINE WITH ABSTENTION
            </div>
            <div style="font-size:0.85rem; color:#EAF1FF; line-height:1.5;">
                To guard against scientific confirmation bias, the model replacement criteria were pre-specified
                before running benchmark experiments. A candidate model could only replace the reference baseline
                if it satisfied all pre-registered gates:
                <ul style="margin-top:0.4rem; padding-left:1.2rem; color:#A9B8DC;">
                    <li><b>Gate (i) Discrimination:</b> 95% bootstrap lower bound for balanced accuracy and ROC-AUC must exceed 0.50 (chance). <i>[Failed: Best balanced accuracy CI lower bound was 0.442; AUC CI lower bound was 0.387].</i></li>
                    <li><b>Gate (ii) Superiority:</b> Must demonstrate higher accuracy than the naive majority guessing baseline (61.6%). <i>[Failed: Best candidate achieved 57.0%].</i></li>
                    <li><b>Gate (vi) Calibration:</b> Brier skill score relative to training prior must be strictly positive. <i>[Failed: Platt calibrator yields negative Brier skill score -0.149].</i></li>
                </ul>
                <b>Scientific Verdict:</b> We honestly report that wristband physiological features alone are insufficient to separate
                laboratory stress from non-stress in this cohort. The system maintains the baseline pipeline with transparent uncertainty
                abstention rather than claiming artificial gains.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_what_model_cannot_do() -> None:
    """Render prominent banner stating what the application does not claim."""
    st.markdown(
        """
        <div class="nt-card" style="border: 1px solid rgba(255, 217, 138, 0.4); background: rgba(255, 217, 138, 0.05);">
            <div style="font-weight: 700; color: #FFD98A; font-size: 0.95rem; margin-bottom: 0.3rem;">
                ⚠️ What NeuroTwin AI Does Not Claim
            </div>
            <div style="font-size: 0.83rem; color: #EAF1FF; line-height: 1.5;">
                • It does not diagnose, treat, or screen for any medical, psychiatric, or psychological condition.<br>
                • It does not determine why biosignals changed or identify the root cause of personal tension.<br>
                • It does not promise that any suggested relaxation or self-care activity will reduce stress.<br>
                • A personal recorded baseline represents a single observation period, not a clinical standard.<br>
                • It is an exploratory computational twin for educational and self-reflection purposes, never a substitute for qualified professional medical care.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

