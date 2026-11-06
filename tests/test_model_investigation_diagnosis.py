"""Regression tests that DOCUMENT defects found in the model investigation (diagnosis phase).

Nothing here changes production code. Where the current behaviour contradicts the intended or
documented behaviour, a test marked xfail(strict=True) states the expectation; it will fail loudly
(XPASS) once the defect is fixed, forcing the marker to be removed on purpose.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import preprocessing as pp  # noqa: E402

T0 = 1_000_000_000.0


def _cleaned_hr():
    """EDA starts at T0; the HR file starts 10 s later (measured in all 37 real recordings).
    HR row k has value 60 + k and was recorded at EDA-time k + 10."""
    n = 150
    eda = pd.DataFrame({"utc_ts": T0 + np.arange(n * 4) / 4.0, "EDA": np.ones(n * 4)})
    hr = pd.DataFrame({"utc_ts": T0 + 10 + np.arange(n), "HR": 60 + np.arange(n)})
    out = pp.clean_subject_signals({"EDA": eda, "HR": hr, "TEMP": None, "BVP": None, "ACC": None,
                                    "IBI": None, "tags": None})
    return out["HR"]


def test_CHARACTERISATION_hr_cleaning_leaves_values_unshifted():
    """Current behaviour: reindex(index - 10) looks values up by label, so nothing moves."""
    h = _cleaned_hr()
    for t in (20, 60, 100):
        assert h.get(t) == 60 + t               # value from row t, i.e. recorded at EDA-time t + 10


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT (diagnosis RC2): HR is read 10 s later than the EDA-based stage boundaries; "
    "src/preprocessing.py reindex(index - 10) does not shift the series."))
def test_EXPECTED_hr_is_aligned_to_eda_time():
    """Intended behaviour: the value placed at EDA-time t was recorded at EDA-time t."""
    h = _cleaned_hr()
    for t in (20, 60, 100):
        assert h.get(t) == 60 + (t - 10)


# Chronological order of stages in the dataset's own self-report files (Stress_Level_v1/v2.csv column
# order), as reported by the raw-data alignment diagnostic (section 5b).
CHRONO = {
    "V1": ["Baseline", "Stroop", "First Rest", "TMCT", "Second Rest", "Real Opinion", "Opposite Opinion", "Subtract"],
    "V2": ["Baseline", "TMCT", "First Rest", "Real Opinion", "Opposite Opinion", "Second Rest", "Subtract"],
}


def _project_order(stage_map):
    return [name for name, _ in sorted(stage_map.items(), key=lambda kv: kv[1][0])]


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT (diagnosis RC1): the project's tag-index stage map does not follow the chronological "
    "order of the self-report files; the alignment diagnostic found zero overlap with the dataset "
    "authors' task windows for most stages."))
@pytest.mark.parametrize("cohort,stage_map", [("V1", pp.STRESS_V1_STAGES), ("V2", pp.STRESS_V2_STAGES)])
def test_EXPECTED_stage_map_follows_chronological_self_report_order(cohort, stage_map):
    assert _project_order(stage_map) == CHRONO[cohort]


def test_project_stage_windows_are_disjoint_by_tag_interval():
    """No two project windows share a tag interval (adjacent windows only share a boundary second)."""
    for m in (pp.STRESS_V1_STAGES, pp.STRESS_V2_STAGES):
        spans = sorted(m.values())
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0


def test_diagnosis_baseline_matches_the_saved_logistic_regression_result():
    path = ROOT / "data" / "processed" / "model_investigation" / "diagnosis" / "diagnosis_results.json"
    if not path.is_file():
        pytest.skip("run scripts/10_model_diagnosis.py first")
    res = json.loads(path.read_text(encoding="utf-8"))
    saved = pd.read_csv(ROOT / "data" / "processed" / "model_comparison_scores.csv").set_index("Classifier")
    row = [i for i in saved.index if str(i).upper().startswith("LOG")][0]
    assert res["D_baseline_existing_logreg"]["pooled"]["accuracy"] == pytest.approx(float(saved.loc[row, "Accuracy"]), abs=1e-3)
    assert res["n_participants"] == 35 and res["n_windows"] == 263
