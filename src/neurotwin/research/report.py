"""Research-grade PDF report generator for NeuroTwin.

Generates a formatted, multi-page scientific analysis report containing:
- Measured Layer (wearable data quality, coverage, motion flags)
- Estimated Layer (classifier prediction, evidence level, calibration notice, feature contributions)
- Reported Layer (PSS-10 score, WHO-5 well-being index, 4-domain QoL)
- Multimodal Agreement Meter (physiological vs cognitive convergence)
- Research Limitations & Clinical Disclaimers
- Verified Bibliography with DOIs
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpdf import FPDF

# Suppress fpdf2 transitional deprecation warnings on cell parameters
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fpdf")

from .formulas import PHYSIOLOGICAL_FEATURES
from .scales import PSS10Result, QoLResult


def _clean(text: str) -> str:
    """Sanitize unicode characters to ensure 100% compatibility with standard PDF fonts."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "—": "--",
        "–": "-",
        "≤": "<=",
        "≥": ">=",
        "·": "*",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "…": "...",
        "µ": "u",
        "°": " deg ",
        "±": "+/-",
        "²": "^2",
        "√": "sqrt",
        "Δ": "Delta ",
        "σ": "sigma",
        "μ": "mu",
        "β": "beta",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class NeuroTwinPDF(FPDF):
    """Custom FPDF subclass with consistent headers, footers, and brand palette."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 150, 180)
        self.cell(0, 8, _clean("NEUROTWIN AI  |  Research Stress Analysis & Physiological Twin Report"), border=0, ln=1, align="L")
        self.set_draw_color(0, 180, 216)
        self.set_line_width(0.3)
        self.line(16, 22, 194, 22)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 144, 156)
        page_str = f"Page {self.page_no()} of {{nb}}"
        self.cell(0, 8, _clean(f"Confidential & Research Only  |  Not a Medical Device  |  {page_str}"), border=0, align="C")


def generate_research_pdf_report(
    participant_id: str,
    measured_quality: Dict[str, Any],
    estimate_result: Dict[str, Any],
    pss_result: Optional[PSS10Result] = None,
    qol_result: Optional[QoLResult] = None,
    svm_score: Optional[float] = None,
    agreement_data: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Generate a research PDF report and return the binary content."""
    pdf = NeuroTwinPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Title Banner ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, _clean("Multimodal Stress Analysis Report"), ln=1, align="L")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta_text = _clean(f"Participant / Session: {participant_id}   |   Generated: {ts}   |   Sensor: Empatica E4 Wristband")
    pdf.cell(0, 5, meta_text, ln=1, align="L")
    pdf.ln(3)

    # --- Section 1: Measured Layer (Wearable Signal Quality) ---
    pdf.set_fill_color(240, 249, 255)
    pdf.set_draw_color(186, 230, 253)
    pdf.rect(16, pdf.get_y(), 178, 28, style="DF")
    pdf.set_y(pdf.get_y() + 2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(3, 105, 161)
    pdf.cell(0, 6, _clean("  1. MEASURED LAYER -- Sensor Signal Quality"), ln=1)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    cov = measured_quality.get("coverage", 1.0)
    cov_str = f"{cov:.0%}" if isinstance(cov, float) else str(cov)
    flags = measured_quality.get("flags", [])
    flags_str = ", ".join(flags) if flags else "None (optimal stability)"
    missing_groups = measured_quality.get("groups_missing", [])
    mg_str = ", ".join(missing_groups) if missing_groups else "None (all 5 groups present)"

    pdf.cell(58, 5, _clean(f"  Feature Coverage: {cov_str}"), ln=0)
    pdf.cell(60, 5, _clean(f"Missing Signal Groups: {mg_str}"), ln=0)
    pdf.cell(60, 5, _clean(f"Motion / Temp Flags: {flags_str}"), ln=1)

    input_rows = measured_quality.get("input_rows", 1)
    pdf.cell(0, 5, _clean(f"  Analyzed Windows: {input_rows} protocol segments  |  Sampling: EDA 4Hz, HR 1Hz, TEMP 4Hz, ACC 32Hz"), ln=1)
    pdf.ln(5)

    # --- Section 2: Estimated Layer (ML Classification & Uncertainty) ---
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(16, pdf.get_y(), 178, 38, style="DF")
    pdf.set_y(pdf.get_y() + 2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, _clean("  2. ESTIMATED LAYER -- Physiological Model Inference"), ln=1)

    label = estimate_result.get("label", "Not assessed")
    evidence = estimate_result.get("evidence_level", "limited").capitalize()
    model_name = estimate_result.get("model", "Logistic Regression / Random Forest")
    prob_shown = estimate_result.get("probability_shown")
    prob_str = f"{prob_shown:.1%}" if prob_shown is not None else "Suppressed (ECE / Calibration Rule)"

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(180, 20, 50) if "high" in label.lower() else pdf.set_text_color(16, 149, 110)
    pdf.cell(60, 6, _clean(f"  Predicted State: {label}"), ln=0)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(60, 6, _clean(f"Evidence Level: {evidence}"), ln=0)
    pdf.cell(58, 6, _clean(f"Calibrated P(Stress): {prob_str}"), ln=1)

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(
        174,
        4.2,
        _clean(
            f"  Model Architecture: {model_name}. Evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation.\n"
            "  Uncertainty Policy: Raw probabilities are strictly withheld when post-calibration Brier score does not exceed the "
            "training prior or ECE > 0.10. An ordinal evidence level is displayed instead."
        ),
    )
    pdf.ln(4)

    # Top Feature Contributions
    contribs = estimate_result.get("contributions", [])
    if contribs:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, _clean("Top Physiological Feature Influences:"), ln=1)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(50, 5, _clean("Feature"), border=1, fill=True)
        pdf.cell(40, 5, _clean("Physiological Group"), border=1, fill=True)
        pdf.cell(30, 5, _clean("Z-Score"), border=1, fill=True)
        pdf.cell(58, 5, _clean("Contribution Direction"), border=1, fill=True, ln=1)

        pdf.set_font("Helvetica", "", 8)
        for c in contribs[:5]:
            feat = c.get("feature", "")
            feat_obj = PHYSIOLOGICAL_FEATURES.get(feat)
            grp = feat_obj.group if feat_obj else "Physiological"
            z = c.get("z", 0.0)
            contrib = c.get("contribution", 0.0)
            direction = "Toward Higher Stress" if contrib > 0 else "Toward Lower Stress"
            if not c.get("explained", True):
                direction = "Suppressed (Under Review)"

            pdf.cell(50, 4.5, _clean(feat[:24]), border=1)
            pdf.cell(40, 4.5, _clean(grp[:20]), border=1)
            pdf.cell(30, 4.5, _clean(f"{z:+.2f} std"), border=1)
            pdf.cell(58, 4.5, _clean(direction), border=1, ln=1)
        pdf.ln(4)

    # --- Section 3: Reported Layer (Psychometrics) ---
    pdf.set_fill_color(254, 243, 199)
    pdf.set_draw_color(253, 230, 138)
    pdf.rect(16, pdf.get_y(), 178, 30, style="DF")
    pdf.set_y(pdf.get_y() + 2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(180, 83, 9)
    pdf.cell(0, 6, _clean("  3. REPORTED LAYER -- Subjective Questionnaires & Life Quality"), ln=1)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    if pss_result:
        pdf.cell(
            60,
            5,
            _clean(f"  PSS-10 Score: {pss_result.total_score} / 40"),
            ln=0,
        )
        pdf.cell(60, 5, _clean(f"Classification: {pss_result.band} Stress"), ln=0)
        pdf.cell(58, 5, _clean(f"Normative Percentile: {pss_result.norm_percentile}th"), ln=1)
    else:
        pdf.cell(0, 5, _clean("  PSS-10 Questionnaire: Skipped or not completed."), ln=1)

    if qol_result:
        pdf.cell(60, 5, _clean(f"  WHO-5 Well-Being: {qol_result.who5_percentage:.0f} / 100"), ln=0)
        pdf.cell(60, 5, _clean(f"Status: {qol_result.wellbeing_level}"), ln=0)
        pdf.cell(58, 5, _clean(f"Raw Total: {qol_result.who5_raw} / 25"), ln=1)
    else:
        pdf.cell(0, 5, _clean("  Quality of Life Index: Skipped."), ln=1)
    pdf.ln(5)

    # --- Section 4: Multimodal Agreement Analysis ---
    if agreement_data:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 6, _clean("Multimodal Agreement Analysis (Sensor vs Self-Report):"), ln=1)

        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(51, 65, 85)
        status = agreement_data.get("status", "Partial")
        explanation = agreement_data.get("explanation", "")
        pdf.multi_cell(
            178,
            4.2,
            _clean(
                f"Concordance Status: {status}\n{explanation}\n"
                "Note: Wristband sensors reflect real-time sympathetic/parasympathetic tone, whereas PSS-10 captures "
                "30-day cognitive self-appraisal. Divergence between sensor activation and retrospective appraisal is scientifically expected."
            ),
        )
        pdf.ln(3)

    # --- Section 5: Limitations & Ethical Governance ---
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(180, 20, 50)
    pdf.cell(0, 5, _clean("Scientific Limitations & Safety Disclaimer:"), ln=1)

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(71, 85, 105)
    disclaimer_text = (
        "1. NOT A MEDICAL DEVICE: This system is a research prototype. It does not diagnose, treat, or prevent any illness.\n"
        "2. POPULATION CONSTRAINTS: Trained on 35 healthy subjects (n=263 windows, ages 18-30) under laboratory stress protocols "
        "(Stroop test, mental arithmetic). Real-world generalizability has not been established.\n"
        "3. CRACK-FREE EVALUATION: Reported leave-one-subject-out accuracy is 57.0% (vs majority baseline 61.6%, AUC 0.539). "
        "No inflated numbers or data leakage are permitted.\n"
        "4. CRISIS SUPPORT: If you are experiencing emotional distress, reach out to professional services (Tele-MANAS: 14416 / 1-800-891-4416; NHS: 111)."
    )
    pdf.multi_cell(178, 3.5, _clean(disclaimer_text))
    pdf.ln(2)

    # --- Section 6: Verified References ---
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 5, _clean("Key References (DOIs Verified):"), ln=1)

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 116, 139)
    refs = [
        "1. Hongn et al. Wearable Device Dataset. PhysioNet 2025. doi:10.13026/he0v-tf17",
        "2. Cohen, Kamarck, Mermelstein. Perceived Stress Scale. J Health Soc Behav 1983;24:385-396. doi:10.2307/2136404",
        "3. Task Force of ESC/NASPE. Heart rate variability standards. Circulation 1996;93:1043-1065. doi:10.1161/01.CIR.93.5.1043",
        "4. Topp et al. The WHO-5 Well-Being Index. Psychother Psychosom 2015;84:167-176. doi:10.1159/000376585",
        "5. Boucsein et al. Publication recommendations for EDA. Psychophysiology 2012;49:1017-1034. doi:10.1111/j.1469-8986.2012.01384.x",
    ]
    for r in refs:
        pdf.cell(0, 3.5, _clean(r), ln=1)

    return bytes(pdf.output())

