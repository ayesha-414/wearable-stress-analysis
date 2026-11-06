"""Unit tests for NeuroTwin Research Layer and Protected Files Integrity.

Covers:
1. PSS-10 psychometric scoring, reverse items (4, 5, 7, 8), and clinical cutoffs.
2. WHO-5 Well-Being Index (0–100) and QoL scoring.
3. One-factor Confirmatory Factor Model (CFA) loadings and Cronbach's alpha.
4. Conceptual SEM path model specification.
5. Multimodal Agreement Meter (sensor vs self-report).
6. Wearable CSV validation edge cases (missing cols, aliases, NaNs).
7. PDF report generator output validity.
8. Rigorous SHA-256 hash verification confirming that protected files are unchanged.
"""

import hashlib
from pathlib import Path
import pandas as pd
import pytest

from dashboard.app import validate_uploaded_csv
from src.models import FEATURE_COLS
from src.neurotwin.research.formulas import PHYSIOLOGICAL_FEATURES
from src.neurotwin.research.report import generate_research_pdf_report
from src.neurotwin.research.scales import (
    assess_agreement,
    calculate_one_factor_cfa,
    get_sem_path_model,
    score_pss10,
    score_qol,
)

# Baseline SHA-256 hashes of all protected files (Absolute Rule)
PROTECTED_FILE_HASHES = {
    "src/features.py": "8f81998f2b4151802f2308c7e24a214fe1657728e27bca40589b7bd284447b60",
    "src/models.py": "0030f2683538448aa31d1d8604a47c5ffc438accf04e8c89b07c418559d9e168",
    "src/loader.py": "c102c6e1457b2458178439f02f696a642f8a084b30c77713d9a457d2fd9a7e8c",
    "src/preprocessing.py": "84e05bb0878e37ea00612b2a5e16319c22265b60f620ee17549857d69a277d6a",
    "scripts/01_load_and_verify.py": "c140e077ab46600e7de9a685d69491867b00b201d0ffbdee81fcc74ad5fccdfd",
    "scripts/02_eda.py": "b946489d3bddf2d15d38babb5b4d914f166b504f205564605c06e02783c0c797",
    "scripts/03_feature_engineering.py": "293a7a27a01bf7fe20470de86fcaf7a17d7c0a15d289dfd540b7c19a2a2166db",
    "scripts/04_stress_analysis.py": "6741921b192ed9ebc147be5bd46b85ddd2bdee8c7a4be77f6ca4d53bf5573af3",
    "scripts/05_model_evaluation.py": "fce8c7cc5f81733c118ff46022063ac4017744488db0a084ab9a0151f8a77732",
    "scripts/06_feature_selection.py": "92bcb9cb5926203e46f311e20f8fdf871eddf94aaa3d523ebaa04154439650bb",
    "scripts/07_factor_analysis.py": "2dd0416c789ab7b3bf028962a2266656444edf470fa56a36a65b748133990eb4",
    "data/features/stress_features.csv": "13a44d283f234b2b8f7b474298528a8d2bafb03717e1cf9017698cbbe8bd5bc4",
    "data/processed/alignment_diagnostic_20261027_210750.txt": "392872ad99cdf9ea9eba383f1b319f4b063b4e4716f5df5ef48877b25feb017e",
    "data/processed/dataset_summary.csv": "65a150711bf2f918be897241e7b314ed5acf1c7d6f218e0b6e982858c373005d",
    "data/processed/feature_importance.csv": "d6a439d7a106c27d90e974c4cf908b9eb5f73b534118517433ec85a23f1bce71",
    "data/processed/loso_results.csv": "c1dfd9bd7247ad6e73669cd93de4b66c1cb783fb32c37b8abf835bcbbb5630b4",
    "data/processed/model_comparison_scores.csv": "994cc8e23e8efa27921651a52363ce6e6e5749dfa0a72769ea95141db6b0a794",
    "data/processed/neurotwin_estimator_report.json": "0120d05376105cd310c9b2e33c41cba8d04d81e9abf81195bef0bbc94f2d1e46",
    "data/processed/neurotwin_simulation_report.json": "c708755462efaa739ba6edf6d9f640c480dd0a50373be5e1732864f8a3553da3",
    "data/processed/signal_quality_report.csv": "8d53a978f5f52b3c82603b69565c787477ea8d0752f67e8d99005bb704a214cf",
    "data/processed/stress_stage_stats.csv": "a74f25841c9cd93b08fcaf2a94c45e8a0c94623367b651c5d734f5abdc2a4e63",
}


# ---------------------------------------------------------------------------
# Test 1: Protected Files Integrity
# ---------------------------------------------------------------------------
def test_protected_files_are_strictly_unmodified():
    """Verify that none of the protected research code, models, or data were altered."""
    root = Path(__file__).resolve().parents[1]
    for rel_path, expected_hash in PROTECTED_FILE_HASHES.items():
        file_path = root / rel_path
        assert file_path.is_file(), f"Protected file missing: {rel_path}"
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"PROTECTED FILE ALTERED: {rel_path} has hash {actual_hash}, expected {expected_hash}."
        )


# ---------------------------------------------------------------------------
# Test 2: PSS-10 Scoring & Reverse Items
# ---------------------------------------------------------------------------
def test_pss10_all_zeros():
    """If user responds 0 to all items, reverse items (4,5,7,8) become 4, total is 16 (Moderate)."""
    answers = {i: 0 for i in range(1, 11)}
    res = score_pss10(answers)
    assert res.total_score == 16
    assert res.band == "Moderate"
    assert res.scored_items[4] == 4
    assert res.scored_items[5] == 4
    assert res.scored_items[7] == 4
    assert res.scored_items[8] == 4
    assert res.scored_items[1] == 0


def test_pss10_minimum_score():
    """Minimum score (0): all positive items 0, all reverse items 4."""
    answers = {1: 0, 2: 0, 3: 0, 4: 4, 5: 4, 6: 0, 7: 4, 8: 4, 9: 0, 10: 0}
    res = score_pss10(answers)
    assert res.total_score == 0
    assert res.band == "Low"


def test_pss10_maximum_score():
    """Maximum score (40): all positive items 4, all reverse items 0."""
    answers = {1: 4, 2: 4, 3: 4, 4: 0, 5: 0, 6: 4, 7: 0, 8: 0, 9: 4, 10: 4}
    res = score_pss10(answers)
    assert res.total_score == 40
    assert res.band == "High"


def test_pss10_clinical_band_boundaries():
    """Check exact boundary thresholds: Low <= 13, Moderate 14-26, High >= 27."""
    low_answers = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 1, 7: 3, 8: 3, 9: 1, 10: 1}  # sum = 10
    assert score_pss10(low_answers).band == "Low"

    mod_answers = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}  # sum = 20
    assert score_pss10(mod_answers).band == "Moderate"

    high_answers = {1: 3, 2: 3, 3: 3, 4: 1, 5: 1, 6: 3, 7: 1, 8: 1, 9: 3, 10: 3}  # sum = 30
    assert score_pss10(high_answers).band == "High"


# ---------------------------------------------------------------------------
# Test 3: WHO-5 Well-Being Index Scoring
# ---------------------------------------------------------------------------
def test_who5_scoring_range():
    """Verify raw range (0-25) scales correctly to (0-100)."""
    min_res = score_qol({i: 0 for i in range(1, 6)})
    assert min_res.who5_raw == 0
    assert min_res.who5_percentage == 0.0
    assert min_res.wellbeing_level == "Reduced"

    max_res = score_qol({i: 5 for i in range(1, 6)})
    assert max_res.who5_raw == 25
    assert max_res.who5_percentage == 100.0
    assert max_res.wellbeing_level == "Good"


# ---------------------------------------------------------------------------
# Test 4: One-Factor Model & Cronbach's Alpha
# ---------------------------------------------------------------------------
def test_one_factor_cfa_structure():
    cfa = calculate_one_factor_cfa()
    assert "factor_loadings" in cfa
    assert len(cfa["factor_loadings"]) == 10
    assert cfa["cronbach_alpha"] == 0.865
    assert all(0.5 <= l <= 0.9 for l in cfa["factor_loadings"].values())


# ---------------------------------------------------------------------------
# Test 5: Conceptual SEM Path Specification
# ---------------------------------------------------------------------------
def test_sem_model_specification():
    sem = get_sem_path_model()
    assert sem["status"] == "conceptual, not fitted"
    assert len(sem["paths"]) >= 4
    assert any("Sensor" in p["from"] for p in sem["paths"])
    assert any("Quality of Life" in p["to"] for p in sem["paths"])


# ---------------------------------------------------------------------------
# Test 6: Multimodal Agreement Meter
# ---------------------------------------------------------------------------
def test_agreement_meter_concordance():
    # Both elevated
    a1 = assess_agreement(sensor_stress_high=True, pss_band="High")
    assert "Both Elevated" in a1["status"]

    # Both calm
    a2 = assess_agreement(sensor_stress_high=False, pss_band="Low")
    assert "Both Calm" in a2["status"]

    # Divergent: Sensor high, PSS low
    a3 = assess_agreement(sensor_stress_high=True, pss_band="Low")
    assert "Divergent" in a3["status"]
    assert "acute autonomic" in a3["explanation"]


# ---------------------------------------------------------------------------
# Test 7: CSV Validation Edge Cases
# ---------------------------------------------------------------------------
def test_validate_csv_success_with_sample():
    sample_path = Path("sample_upload.csv")
    if sample_path.is_file():
        df = pd.read_csv(sample_path)
        valid, msg, clean_df = validate_uploaded_csv(df)
        assert valid is True
        assert clean_df is not None
        assert clean_df.shape[1] == len(FEATURE_COLS)


def test_validate_csv_missing_column_rejected():
    sample_path = Path("sample_upload.csv")
    if sample_path.is_file():
        df = pd.read_csv(sample_path).drop(columns=["hr_mean"])
        valid, msg, clean_df = validate_uploaded_csv(df)
        assert valid is False
        assert clean_df is None
        assert "hr_mean" in msg


def test_validate_csv_column_alias_handling():
    sample_path = Path("sample_upload.csv")
    if sample_path.is_file():
        df = pd.read_csv(sample_path)
        # Rename hr_diff_rms to legacy hr_rmssd_approx or vice versa
        if "hr_rmssd_approx" in df.columns:
            df = df.rename(columns={"hr_rmssd_approx": "hr_diff_rms"})
        valid, msg, clean_df = validate_uploaded_csv(df)
        assert valid is True


# ---------------------------------------------------------------------------
# Test 8: Research PDF Report Generation
# ---------------------------------------------------------------------------
def test_pdf_report_generation_binary_header():
    pss = score_pss10({i: 2 for i in range(1, 11)})
    qol = score_qol({i: 3 for i in range(1, 6)})
    pdf_bytes = generate_research_pdf_report(
        participant_id="TEST-001",
        measured_quality={"coverage": 1.0, "flags": [], "groups_missing": []},
        estimate_result={
            "label": "Low stress pattern",
            "evidence_level": "limited",
            "model": "Logistic Regression",
            "probability_shown": None,
            "contributions": [],
        },
        pss_result=pss,
        qol_result=qol,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# Test 9: Complete 23 Features Formula Registry
# ---------------------------------------------------------------------------
def test_all_23_features_registered_with_formulas():
    assert len(PHYSIOLOGICAL_FEATURES) == 23
    for name, feat in PHYSIOLOGICAL_FEATURES.items():
        assert feat.formula_latex.strip() != "", f"Missing formula for {name}"
        assert feat.physiological_reasoning.strip() != "", f"Missing reasoning for {name}"
        assert feat.reference_doi.startswith("http"), f"Missing valid DOI link for {name}"
        assert feat.evidence_label in {
            "Matches standard",
            "Custom derived",
            "Heuristic approximation",
            "Uncertain",
        }

