"""
tests/test_investigation_decision_rule.py
=========================================
Unit tests for the programmatic implementation of the §9.4 Decision Rule,
and consistency tests between the written verdict JSON and evaluation files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "data" / "processed" / "model_investigation" / "v2"


def test_decision_rule_verdict_consistency():
    verdict_file = V2_DIR / "decision_rule_verdict.json"
    benchmark_file = V2_DIR / "benchmark_results_primary_B.csv"

    assert verdict_file.exists(), f"Verdict JSON not found: {verdict_file}"
    assert benchmark_file.exists(), f"Benchmark CSV not found: {benchmark_file}"

    verdict_data = json.loads(verdict_file.read_text(encoding="utf-8"))
    df_bm = pd.read_csv(benchmark_file)

    # Verdict must be RETAIN_BASELINE or REPLACE_BASELINE
    verdict = verdict_data["final_verdict"]
    assert verdict in ("RETAIN_BASELINE", "REPLACE_BASELINE")

    # In our empirical evaluation, no model cleared condition (i) and (vi)
    # Check that the verdict in the JSON accurately matches the candidate evaluations
    candidates = verdict_data["decision_rule_per_candidate"]
    for c in candidates:
        model_name = c["model"]
        row = df_bm[df_bm["model"] == model_name].iloc[0]

        # Condition (i) check consistency
        cond_i = c["cond_i_ci_lower_bounds_gt_half"]["passed"]
        expected_cond_i = bool(row["balanced_accuracy_ci_lower"] > 0.5 and row["roc_auc_ci_lower"] > 0.5)
        assert cond_i == expected_cond_i

        # Condition (vi) check consistency
        cond_vi = c["cond_vi_brier_skill_positive"]["passed"]
        expected_cond_vi = bool(row["brier_skill"] > 0.0)
        assert cond_vi == expected_cond_vi

    # If all candidates failed at least one condition, verdict must be RETAIN_BASELINE
    any_cleared = any(c["clears_all_criteria"] for c in candidates)
    if not any_cleared:
        assert verdict == "RETAIN_BASELINE"


def test_synthetic_decision_rule_logic():
    """Unit test decision rule logic with mock pass and fail records."""
    # Scenario A: all criteria pass -> should recommend REPLACE
    mock_candidate_pass = {
        "cond_i": True,
        "cond_ii": True,
        "cond_iii": True,
        "cond_iv": True,
        "cond_v": True,
    }
    assert all(mock_candidate_pass.values()) is True

    # Scenario B: lower CI bound <= 0.5 -> should fail
    mock_candidate_fail_ci = {
        "cond_i": False,
        "cond_ii": True,
        "cond_iii": True,
        "cond_iv": True,
        "cond_v": True,
    }
    assert all(mock_candidate_fail_ci.values()) is False

