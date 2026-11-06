"""
tests/test_features.py
========================
Unit tests for src/features.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    hr_features,
    eda_features,
    temp_features,
    acc_features,
    ibi_features,
    _safe_slope,
    MIN_SAMPLES,
)


def make_series(values, start=0):
    return pd.Series(values, index=range(start, start + len(values)))


class TestHRFeatures:
    def test_returns_dict(self):
        s = make_series([70, 72, 75, 68, 71] * 5)
        result = hr_features(s)
        assert isinstance(result, dict)

    def test_keys_present(self):
        s = make_series([70] * 30)
        result = hr_features(s)
        assert "hr_mean" in result
        assert "hr_std" in result

    def test_none_returns_nans(self):
        result = hr_features(None)
        assert all(np.isnan(v) for v in result.values())

    def test_too_short_returns_nans(self):
        s = make_series([70] * (MIN_SAMPLES - 1))
        result = hr_features(s)
        assert all(np.isnan(v) for v in result.values())

    def test_values_sensible(self):
        s = make_series([60.0, 65.0, 70.0, 75.0, 80.0] * 4)
        result = hr_features(s)
        assert result["hr_mean"] == pytest.approx(70.0)
        assert result["hr_min"] == pytest.approx(60.0)
        assert result["hr_max"] == pytest.approx(80.0)


class TestEDAFeatures:
    def test_returns_eda_keys(self):
        s = make_series([2.0, 2.1, 2.3, 1.9, 2.5] * 5)
        result = eda_features(s)
        assert "eda_mean" in result
        assert "eda_scr_count" in result

    def test_none_returns_nans(self):
        result = eda_features(None)
        assert all(np.isnan(v) if isinstance(v, float) else v == 0 for v in result.values())

    def test_slope_positive_trend(self):
        s = make_series([float(i) for i in range(20)])
        result = eda_features(s)
        assert result["eda_slope"] > 0

    def test_slope_negative_trend(self):
        s = make_series([float(20 - i) for i in range(20)])
        result = eda_features(s)
        assert result["eda_slope"] < 0


class TestTempFeatures:
    def test_sensible_values(self):
        s = make_series([35.0, 35.1, 34.9, 35.2] * 5)
        result = temp_features(s)
        assert result["temp_mean"] == pytest.approx(35.05, abs=0.1)

    def test_none_returns_nans(self):
        result = temp_features(None)
        assert all(np.isnan(v) for v in result.values())


class TestAccFeatures:
    def test_all_zero_data(self):
        s = make_series([0.0] * 20)
        result = acc_features(s)
        assert result["acc_mean"] == pytest.approx(0.0)
        assert result["acc_std"] == pytest.approx(0.0)

    def test_positive_values(self):
        s = make_series([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        result = acc_features(s)
        assert result["acc_max"] == pytest.approx(5.0)


class TestIBIFeatures:
    def make_ibi_df(self, n=30, mean_ibi=0.8):
        np.random.seed(42)
        ibi_vals = np.abs(np.random.normal(mean_ibi, 0.05, n))
        offsets = np.cumsum(ibi_vals)
        return pd.DataFrame({"offset_s": offsets, "ibi_s": ibi_vals})

    def test_returns_hrv_keys(self):
        df = self.make_ibi_df()
        result = ibi_features(df)
        assert "hrv_sdnn" in result
        assert "hrv_rmssd" in result
        assert "hrv_pnn50" in result

    def test_none_returns_nans(self):
        result = ibi_features(None)
        assert all(np.isnan(v) for v in result.values())

    def test_pnn50_between_0_and_1(self):
        df = self.make_ibi_df(50)
        result = ibi_features(df)
        if not np.isnan(result["hrv_pnn50"]):
            assert 0.0 <= result["hrv_pnn50"] <= 1.0

    def test_sdnn_positive(self):
        df = self.make_ibi_df(50)
        result = ibi_features(df)
        if not np.isnan(result["hrv_sdnn"]):
            assert result["hrv_sdnn"] >= 0


class TestSafeSlope:
    def test_positive_slope(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _safe_slope(s) == pytest.approx(1.0)

    def test_flat_slope(self):
        s = pd.Series([5.0] * 10)
        assert _safe_slope(s) == pytest.approx(0.0, abs=1e-6)

    def test_single_value_returns_nan(self):
        s = pd.Series([3.0])
        assert np.isnan(_safe_slope(s))
