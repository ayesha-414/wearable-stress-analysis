"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    to_relative_seconds,
    resample_to_1hz,
    clean_eda,
    clean_hr,
    clean_temp,
    moving_average_acc,
    get_stage_map,
    STRESS_V1_STAGES,
    STRESS_V2_STAGES,
)


class TestRelativeSeconds:
    def test_first_value_is_zero(self):
        df = pd.DataFrame({"utc_ts": [1000.0, 1001.0, 1002.0]})
        result = to_relative_seconds(df)
        assert result["utc_ts"].iloc[0] == 0.0

    def test_differences_preserved(self):
        df = pd.DataFrame({"utc_ts": [1000.0, 1001.5, 1003.0]})
        result = to_relative_seconds(df)
        assert result["utc_ts"].iloc[1] == pytest.approx(1.5)
        assert result["utc_ts"].iloc[2] == pytest.approx(3.0)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"utc_ts": [1000.0, 1001.0]})
        original_val = df["utc_ts"].iloc[0]
        to_relative_seconds(df)
        assert df["utc_ts"].iloc[0] == original_val


class TestResampleTo1Hz:
    def test_output_length(self):
        # 10 seconds at 4 Hz = 40 rows
        start = 0.0
        fs = 4.0
        n = 40
        times = start + np.arange(n) / fs
        vals = np.ones(n) * 5.0
        df = pd.DataFrame({"utc_ts": times, "EDA": vals})
        result = resample_to_1hz(df, "EDA")
        assert len(result) == 10

    def test_mean_preserved(self):
        times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        vals = [4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0]
        df = pd.DataFrame({"utc_ts": times, "EDA": vals})
        result = resample_to_1hz(df, "EDA")
        assert result.iloc[0] == pytest.approx(4.0)
        assert result.iloc[1] == pytest.approx(8.0)


class TestCleanEDA:
    def test_clips_negative(self):
        s = pd.Series([-1.0, 0.0, 2.0, 5.0])
        result = clean_eda(s)
        assert result.min() >= 0.0

    def test_clips_above_60(self):
        s = pd.Series([10.0, 61.0, 100.0, 200.0])
        result = clean_eda(s)
        assert result.max() <= 60.0

    def test_valid_values_unchanged(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = clean_eda(s)
        pd.testing.assert_series_equal(result, s)


class TestCleanHR:
    def test_clips_below_30(self):
        s = pd.Series([20.0, 25.0, 70.0])
        result = clean_hr(s)
        assert result.min() >= 30.0

    def test_clips_above_220(self):
        s = pd.Series([70.0, 250.0, 300.0])
        result = clean_hr(s)
        assert result.max() <= 220.0


class TestCleanTemp:
    def test_clips_above_42(self):
        s = pd.Series([35.0, 43.0, 50.0])
        result = clean_temp(s)
        assert result.max() <= 42.0

    def test_clips_below_20(self):
        s = pd.Series([18.0, 35.0])
        result = clean_temp(s)
        assert result.min() >= 20.0


class TestMovingAverageAcc:
    def test_returns_series(self):
        n = 96  # 3 windows of 32
        df = pd.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n),
            "z": np.random.randn(n),
        })
        result = moving_average_acc(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_zero_movement(self):
        df = pd.DataFrame({"x": [0.0] * 32, "y": [0.0] * 32, "z": [0.0] * 32})
        result = moving_average_acc(df)
        assert result.iloc[0] == pytest.approx(0.0)


class TestGetStageMap:
    def test_s_cohort_gets_v1(self):
        stage_map = get_stage_map("STRESS", "S01")
        assert "Baseline" in stage_map
        assert stage_map == STRESS_V1_STAGES

    def test_f_cohort_gets_v2(self):
        stage_map = get_stage_map("STRESS", "f01")
        assert "Baseline" in stage_map
        assert stage_map == STRESS_V2_STAGES

    def test_unknown_activity_returns_empty(self):
        stage_map = get_stage_map("UNKNOWN", "S01")
        assert stage_map == {}
