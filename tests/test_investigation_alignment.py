"""
tests/test_investigation_alignment.py
=====================================
Unit tests for sensor alignment, sign convention, and stage map plausibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.investigation.alignment import align_hr_to_eda, clean_subject_signals_v2
from src.investigation.stage_maps import (
    EXPECTED_DURATIONS_V1,
    EXPECTED_DURATIONS_V2,
    STRESS_V1_STAGES_V2,
    STRESS_V2_STAGES_V2,
    get_stage_map_v2,
)

T0 = 1_000_000_000.0


def test_hr_alignment_synthetic_timing_sign_convention():
    """
    Synthetic series with known ground truth timing:
    - EDA starts at T0
    - HR starts at T0 + 10s (measured across all 37 recordings)
    - HR row k has value 60 + k and was recorded at EDA-time 10 + k

    Verify:
    - Values at EDA time t < 10 are NaN
    - The value placed at EDA time t is 60 + (t - 10) for all t >= 10
    """
    n = 120
    eda = pd.DataFrame({"utc_ts": T0 + np.arange(n * 4) / 4.0, "EDA": np.ones(n * 4)})
    hr = pd.DataFrame({"utc_ts": T0 + 10 + np.arange(n), "HR": 60 + np.arange(n)})

    out = clean_subject_signals_v2({
        "EDA": eda,
        "HR": hr,
        "TEMP": None,
        "BVP": None,
        "ACC": None,
        "IBI": None,
        "tags": None,
    })
    h = out["HR"]
    assert h is not None

    # Before t = 10, HR should be NaN
    for t in (0, 3, 7, 9):
        val = h.get(t)
        assert pd.isna(val) or val is None

    # At t >= 10, HR should match the exact value recorded at that second
    for t in (10, 25, 60, 100):
        expected_val = 60 + (t - 10)
        assert h.get(t) == expected_val


def test_stage_maps_follow_chronological_self_report_order():
    """Verify that stage maps in investigation follow chronological self-report order."""
    chrono_v1 = ["Baseline", "Stroop", "First Rest", "TMCT", "Second Rest", "Real Opinion", "Opposite Opinion", "Subtract"]
    chrono_v2 = ["Baseline", "TMCT", "First Rest", "Real Opinion", "Opposite Opinion", "Second Rest", "Subtract"]

    order_v1 = [name for name, _ in sorted(STRESS_V1_STAGES_V2.items(), key=lambda kv: kv[1][0])]
    order_v2 = [name for name, _ in sorted(STRESS_V2_STAGES_V2.items(), key=lambda kv: kv[1][0])]

    assert order_v1 == chrono_v1
    assert order_v2 == chrono_v2


def test_stage_maps_windows_are_disjoint():
    """Verify that no two stages overlap in raw tag intervals."""
    for stage_map in (STRESS_V1_STAGES_V2, STRESS_V2_STAGES_V2):
        spans = sorted(stage_map.values())
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0


def test_stage_duration_plausibility_ranges():
    """Test that stage map expected durations are clinically and experimentally plausible."""
    # V1 Stroop and TMCT are cognitive tasks of 1-3 minutes
    assert EXPECTED_DURATIONS_V1["Stroop"][0] >= 60.0
    assert EXPECTED_DURATIONS_V1["TMCT"][0] >= 60.0

    # Opinion and Subtract tasks are short ~30 s trials
    for task in ["Real Opinion", "Opposite Opinion", "Subtract"]:
        assert EXPECTED_DURATIONS_V1[task][0] >= 20.0
        assert EXPECTED_DURATIONS_V1[task][1] <= 60.0

    # V2 tasks
    assert EXPECTED_DURATIONS_V2["TMCT"][0] >= 120.0
    for task in ["Real Opinion", "Opposite Opinion", "Subtract"]:
        assert EXPECTED_DURATIONS_V2[task][0] >= 20.0
        assert EXPECTED_DURATIONS_V2[task][1] <= 70.0

