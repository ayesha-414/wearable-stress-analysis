"""
Independent numerical reference tests for physiological feature calculations.

Expected values are computed from first-principles arithmetic in this file.
They are not obtained by calling the production helpers being tested
(_safe_slope, _rolling_tonic, _scr_peak_count, or numpy/pandas wrappers
inside src.features).

Passing these tests shows that the current code matches the stated arithmetic
on synthetic inputs. It does not establish physiological validity on wearable
recordings.

The accelerometer activity index is produced in src.preprocessing.moving_average_acc
before acc_features() summarises it. Both stages are checked here because
acc_mean / acc_std / acc_max are not Euclidean acceleration-magnitude statistics.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    MIN_SAMPLES,
    acc_features,
    eda_features,
    hr_features,
    ibi_features,
    temp_features,
)
from src.preprocessing import moving_average_acc


# ---------------------------------------------------------------------------
# Independent gold-standard arithmetic (not production code)
# ---------------------------------------------------------------------------

def _mean(values) -> float:
    xs = [float(v) for v in values]
    return sum(xs) / len(xs)


def _sample_std(values) -> float:
    """Bessel-corrected standard deviation (divisor n-1)."""
    xs = [float(v) for v in values]
    n = len(xs)
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _population_std(values) -> float:
    """Population standard deviation (divisor n)."""
    xs = [float(v) for v in values]
    n = len(xs)
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / n)


def _ols_slope_vs_sample_index(values) -> float:
    """Ordinary least-squares slope of y against x = 0, 1, ..., n-1."""
    ys = [float(v) for v in values]
    n = len(ys)
    xs = list(range(n))
    x_bar = _mean(xs)
    y_bar = _mean(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    den = sum((x - x_bar) ** 2 for x in xs)
    return num / den


def _rms(values) -> float:
    xs = [float(v) for v in values]
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def _successive_diffs(values) -> list[float]:
    xs = [float(v) for v in values]
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def _centered_rolling_mean(values, window: int = 60) -> list[float]:
    """
    Independent centred rolling mean.

    At index i the window is [i - floor((W-1)/2), i + ceil((W-1)/2)],
    clipped to the series. At least one sample is required.

    For W=60 this is [i-29, i+30]. When n <= 30 the window covers the
    whole series at every index, so the rolling mean equals the global mean.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    left = (window - 1) // 2
    right = window // 2
    out = []
    for i in range(n):
        start = max(0, i - left)
        end = min(n, i + right + 1)
        out.append(_mean(xs[start:end]))
    return out


def _rising_edge_count(binary_flags) -> int:
    flags = [int(v) for v in binary_flags]
    return sum(1 for i in range(1, len(flags)) if flags[i] == 1 and flags[i - 1] == 0)


def _make_series(values) -> pd.Series:
    return pd.Series([float(v) for v in values])


def _ibi_frame(ibi_s, offsets=None) -> pd.DataFrame:
    ibi = [float(v) for v in ibi_s]
    if offsets is None:
        offsets = []
        t = 0.0
        for interval in ibi:
            t += interval
            offsets.append(t)
    return pd.DataFrame({"offset_s": offsets, "ibi_s": ibi})


def _assert_all_nan(result: dict) -> None:
    assert result, "expected a result dictionary"
    for key, value in result.items():
        assert isinstance(value, float) and math.isnan(value), (
            f"{key} should be NaN for invalid/short input, got {value!r}"
        )


# Canonical IBI sequence (seconds). Ten intervals so MIN_SAMPLES is met.
# Successive differences in milliseconds:
#   +50, -50, -50, +60, -50, +50, -20, +50, -50
# Absolute values: 50, 50, 50, 60, 50, 50, 20, 50, 50
# Task Force NN50 / pNN50 uses differences *more than* 50 ms, i.e. > 50.
# Only the +60 ms step qualifies, so pNN50 = 1/9.
IBI_S = [0.80, 0.85, 0.80, 0.75, 0.81, 0.76, 0.81, 0.79, 0.84, 0.79]


# =============================================================================
# A. IBI-derived HRV
# =============================================================================

class TestIBIReferenceValues:
    def test_mean_ibi_is_arithmetic_mean_of_intervals_in_seconds(self):
        expected = _mean(IBI_S)
        result = ibi_features(_ibi_frame(IBI_S))
        assert result["hrv_mean_ibi"] == pytest.approx(expected, rel=0, abs=1e-12)
        # Mean is of IBI in seconds, not milliseconds.
        assert result["hrv_mean_ibi"] < 2.0

    def test_sdnn_uses_sample_std_of_ibi_in_milliseconds(self):
        ibi_ms = [x * 1000.0 for x in IBI_S]
        expected_sample = _sample_std(ibi_ms)
        expected_population = _population_std(ibi_ms)
        result = ibi_features(_ibi_frame(IBI_S))
        assert result["hrv_sdnn"] == pytest.approx(expected_sample, rel=0, abs=1e-9)
        # Guard: implementation must not use the population divisor n.
        assert result["hrv_sdnn"] != pytest.approx(expected_population, abs=1e-9)
        # SDNN is reported in ms, not seconds.
        assert result["hrv_sdnn"] > 1.0

    def test_rmssd_is_rms_of_successive_ibi_differences_in_milliseconds(self):
        ibi_ms = [x * 1000.0 for x in IBI_S]
        diffs_ms = _successive_diffs(ibi_ms)
        expected = _rms(diffs_ms)
        result = ibi_features(_ibi_frame(IBI_S))
        assert result["hrv_rmssd"] == pytest.approx(expected, rel=0, abs=1e-9)
        assert len(diffs_ms) == len(IBI_S) - 1

    def test_pnn50_uses_strict_greater_than_50_ms(self):
        ibi_ms = [x * 1000.0 for x in IBI_S]
        diffs_ms = _successive_diffs(ibi_ms)
        n_gt = sum(1 for d in diffs_ms if abs(d) > 50.0)
        n_ge = sum(1 for d in diffs_ms if abs(d) >= 50.0)
        # Sequence is constructed so these two conventions disagree.
        assert n_gt == 1
        assert n_ge == 8
        expected = n_gt / len(diffs_ms)
        result = ibi_features(_ibi_frame(IBI_S))
        assert result["hrv_pnn50"] == pytest.approx(expected, rel=0, abs=1e-12)
        assert result["hrv_pnn50"] != pytest.approx(n_ge / len(diffs_ms), abs=1e-12)

    def test_pnn50_is_a_proportion_not_a_percentage(self):
        result = ibi_features(_ibi_frame(IBI_S))
        assert 0.0 <= result["hrv_pnn50"] <= 1.0

    def test_insufficient_intervals_return_nan(self):
        short = IBI_S[: MIN_SAMPLES - 1]
        assert len(short) == 9
        _assert_all_nan(ibi_features(_ibi_frame(short)))

    def test_none_and_non_dataframe_return_nan(self):
        _assert_all_nan(ibi_features(None))
        _assert_all_nan(ibi_features([0.8] * 20))  # type: ignore[arg-type]

    def test_nan_intervals_are_dropped_before_the_length_check(self):
        ibi = IBI_S + [float("nan")]
        result = ibi_features(_ibi_frame(ibi))
        # 10 finite intervals remain after dropna, so time-domain HRV is computed.
        assert result["hrv_mean_ibi"] == pytest.approx(_mean(IBI_S), abs=1e-12)

    def test_too_few_finite_intervals_after_dropping_nan_return_nan(self):
        ibi = [0.8] * 5 + [float("nan")] * 10
        _assert_all_nan(ibi_features(_ibi_frame(ibi)))


class TestIBIFrequencyDomainNotIndependentlyGold:
    def test_lf_hf_is_not_assigned_a_hand_calculated_gold_value(self):
        """
        hrv_lf_hf_ratio uses a normalised Lomb–Scargle periodogram averaged
        over LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz). That ratio is not
        uniquely determined by a short arithmetic example, so no numeric
        gold value is asserted here.
        """
        result = ibi_features(_ibi_frame(IBI_S))
        assert "hrv_lf_hf_ratio" in result
        # A finite number may be returned, but it is not independently verified.
        value = result["hrv_lf_hf_ratio"]
        assert math.isnan(value) or math.isfinite(value)


# =============================================================================
# B. hr_diff_rms (not IBI RMSSD)
# =============================================================================

class TestHRDiffRms:
    def test_rms_of_successive_1hz_hr_differences(self):
        # Alternating 60 / 70 bpm: nine successive differences of ±10 bpm.
        hr = [60.0, 70.0] * 5
        diffs = _successive_diffs(hr)
        expected = _rms(diffs)  # sqrt(mean(100)) = 10 bpm
        assert expected == pytest.approx(10.0, abs=1e-12)
        result = hr_features(_make_series(hr))
        assert result["hr_diff_rms"] == pytest.approx(expected, rel=0, abs=1e-12)

    def test_hr_diff_rms_is_not_ibi_rmssd(self):
        hr = [60.0, 70.0] * 5
        hr_result = hr_features(_make_series(hr))
        hrv_result = ibi_features(_ibi_frame(IBI_S))
        assert hr_result["hr_diff_rms"] == pytest.approx(10.0, abs=1e-12)
        # IBI RMSSD is in milliseconds and uses a different input series.
        assert hrv_result["hrv_rmssd"] != pytest.approx(hr_result["hr_diff_rms"], abs=1.0)
        assert hr_result["hr_diff_rms"] < 20  # bpm-scale
        assert hrv_result["hrv_rmssd"] > 20  # ms-scale for this IBI sequence

    def test_constant_hr_gives_zero_diff_rms(self):
        hr = [72.0] * 12
        result = hr_features(_make_series(hr))
        assert result["hr_diff_rms"] == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# C. EDA
# =============================================================================

class TestEDAReferenceValues:
    def test_mean_std_range_and_slope_on_an_arithmetic_ramp(self):
        # y = 2 + 0.5 * t, t = 0..11  (12 samples, 1 Hz index)
        values = [2.0 + 0.5 * t for t in range(12)]
        result = eda_features(_make_series(values))
        assert result["eda_mean"] == pytest.approx(_mean(values), abs=1e-12)
        assert result["eda_std"] == pytest.approx(_sample_std(values), abs=1e-12)
        assert result["eda_range"] == pytest.approx(max(values) - min(values), abs=1e-12)
        assert result["eda_slope"] == pytest.approx(_ols_slope_vs_sample_index(values), abs=1e-12)
        assert result["eda_slope"] == pytest.approx(0.5, abs=1e-12)

    def test_constant_signal_zero_variation_and_zero_scr(self):
        values = [1.7] * 12
        result = eda_features(_make_series(values))
        assert result["eda_mean"] == pytest.approx(1.7, abs=1e-12)
        assert result["eda_std"] == pytest.approx(0.0, abs=1e-12)
        assert result["eda_range"] == pytest.approx(0.0, abs=1e-12)
        assert result["eda_slope"] == pytest.approx(0.0, abs=1e-12)
        assert result["eda_scl_mean"] == pytest.approx(1.7, abs=1e-12)
        assert result["eda_scr_count"] == 0

    def test_scl_proxy_equals_global_mean_when_series_shorter_than_window(self):
        values = [1.0, 1.1, 1.2, 1.0, 0.9, 1.05, 1.15, 0.95, 1.08, 1.02, 0.98, 1.11]
        tonic = _centered_rolling_mean(values, window=60)
        assert all(v == pytest.approx(_mean(values), abs=1e-12) for v in tonic)
        result = eda_features(_make_series(values))
        assert result["eda_scl_mean"] == pytest.approx(_mean(tonic), abs=1e-12)
        assert result["eda_scl_mean"] == pytest.approx(_mean(values), abs=1e-12)

    def test_scr_count_is_phasic_threshold_rising_edges_not_a_peak_detector(self):
        """
        Heuristic under test: phasic = EDA − rolling-mean tonic;
        count 0→1 crossings of (phasic > 0.02).

        This is not a validated SCR peak detector (no amplitude, width,
        or recovery criteria).
        """
        # n=12 << window 60, so tonic is the global mean at every sample.
        values = [1.0] * 5 + [2.0, 2.0] + [1.0] * 5
        tonic = _centered_rolling_mean(values, window=60)
        phasic = [v - t for v, t in zip(values, tonic)]
        above = [1 if p > 0.02 else 0 for p in phasic]
        expected_count = _rising_edge_count(above)
        assert expected_count == 1
        result = eda_features(_make_series(values))
        assert result["eda_scr_count"] == expected_count

    def test_scr_threshold_separates_phasic_just_below_and_just_above_0_02(self):
        """
        Replaces the earlier exact-boundary test, which had a TEST-DESIGN flaw:
        it built a phasic peak of "exactly" 0.02 from 1.0 + 0.02*12/11, but in
        IEEE-754 that evaluates to 0.020000000000000018 (1.7e-17 above 0.02), so
        the production comparison `> 0.02` was correctly True and the test
        wrongly expected 0. Production code was behaving correctly.

        Derivation (n = 12 < 60, so tonic = global mean at every sample):
          values = [b]*11 + [b + d]
          tonic  = b + d/12
          phasic(last) = d - d/12 = 11*d/12      -> choose d = target*12/11
          phasic(other) = -d/12                  -> far below the threshold
        Targets 0.019 and 0.021 sit 1e-3 either side of 0.02, about 1e13 times
        larger than floating-point noise, so the expected counts are unambiguous:
        0.019 -> 0 SCR, 0.021 -> 1 SCR (rising edge at the last sample).

        Strictness AT exactly 0.02 cannot be built through arithmetic on the
        signal and is tested in tests/test_feature_reference_stage1.py using a
        zero tonic, where 0.02 - 0.0 is exact.
        """
        n = 12
        baseline = 1.0
        for target_phasic, expected_count in [(0.019, 0), (0.021, 1)]:
            delta = target_phasic * n / (n - 1)
            values = [baseline] * (n - 1) + [baseline + delta]
            tonic = _centered_rolling_mean(values, window=60)
            phasic_last = values[-1] - tonic[-1]
            assert phasic_last == pytest.approx(target_phasic, abs=1e-12)
            above = [1 if (v - t) > 0.02 else 0 for v, t in zip(values, tonic)]
            assert _rising_edge_count(above) == expected_count
            result = eda_features(_make_series(values))
            assert result["eda_scr_count"] == expected_count, target_phasic

    def test_short_and_missing_eda_return_nan(self):
        _assert_all_nan(eda_features(None))
        _assert_all_nan(eda_features(_make_series([1.0] * (MIN_SAMPLES - 1))))


# =============================================================================
# D. Temperature
# =============================================================================

class TestTemperatureReferenceValues:
    def test_mean_std_and_slope(self):
        values = [32.0 + 0.25 * t for t in range(12)]
        result = temp_features(_make_series(values))
        assert result["temp_mean"] == pytest.approx(_mean(values), abs=1e-12)
        assert result["temp_std"] == pytest.approx(_sample_std(values), abs=1e-12)
        assert result["temp_slope"] == pytest.approx(_ols_slope_vs_sample_index(values), abs=1e-12)
        assert result["temp_slope"] == pytest.approx(0.25, abs=1e-12)

    def test_range_is_not_implemented(self):
        values = [34.0 + 0.1 * t for t in range(12)]
        result = temp_features(_make_series(values))
        assert "temp_range" not in result
        assert set(result) == {"temp_mean", "temp_std", "temp_slope"}

    def test_constant_temperature(self):
        values = [35.0] * 12
        result = temp_features(_make_series(values))
        assert result["temp_mean"] == pytest.approx(35.0, abs=1e-12)
        assert result["temp_std"] == pytest.approx(0.0, abs=1e-12)
        assert result["temp_slope"] == pytest.approx(0.0, abs=1e-12)

    def test_short_and_missing_temperature_return_nan(self):
        _assert_all_nan(temp_features(None))
        _assert_all_nan(temp_features(_make_series([35.0] * (MIN_SAMPLES - 1))))


# =============================================================================
# E. Accelerometer activity index, then summary statistics
# =============================================================================

class TestAccelerometerActivityIndex:
    def test_block_sum_of_max_axis_delta_then_ema(self):
        """
        moving_average_acc (window=32):
          for each 32-sample block,
            s = sum_i max(|dx_i|, |dy_i|, |dz_i|) relative to the previous
                sample (zeros before the first block),
            avg <- 0.9 * avg + 0.1 * (s / 32)
          emit avg (one value per block).

        Two blocks of constant (10, 0, 0):
          block 0 vs (0,0,0): s = 32 * 10, avg = 1.0
          block 1 vs (10,0,0): s = 0,       avg = 0.9
        """
        n = 64
        acc = pd.DataFrame({
            "x": [10.0] * n,
            "y": [0.0] * n,
            "z": [0.0] * n,
        })
        activity = moving_average_acc(acc)
        assert list(activity) == pytest.approx([1.0, 0.9], abs=1e-12)

    def test_activity_index_is_not_euclidean_magnitude(self):
        n = 32
        acc = pd.DataFrame({
            "x": [10.0] * n,
            "y": [0.0] * n,
            "z": [0.0] * n,
        })
        euclid = math.sqrt(10.0 ** 2 + 0.0 ** 2 + 0.0 ** 2)
        activity = moving_average_acc(acc)
        assert activity.iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert activity.iloc[0] != pytest.approx(euclid, abs=0.1)

    def test_acc_features_summarise_the_activity_index_not_raw_axes(self):
        # Ten one-second blocks of constant (10, 0, 0) after a zero previous vector.
        n_blocks = 12
        acc = pd.DataFrame({
            "x": [10.0] * (32 * n_blocks),
            "y": [0.0] * (32 * n_blocks),
            "z": [0.0] * (32 * n_blocks),
        })
        activity = moving_average_acc(acc)
        expected_index = [1.0 * (0.9 ** k) for k in range(n_blocks)]
        assert list(activity) == pytest.approx(expected_index, abs=1e-12)

        result = acc_features(activity)
        assert result["acc_mean"] == pytest.approx(_mean(expected_index), abs=1e-12)
        assert result["acc_std"] == pytest.approx(_sample_std(expected_index), abs=1e-12)
        assert result["acc_max"] == pytest.approx(max(expected_index), abs=1e-12)

    def test_acc_features_on_a_known_activity_series(self):
        values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        result = acc_features(_make_series(values))
        assert result["acc_mean"] == pytest.approx(_mean(values), abs=1e-12)
        assert result["acc_std"] == pytest.approx(_sample_std(values), abs=1e-12)
        assert result["acc_max"] == pytest.approx(5.5, abs=1e-12)

    def test_short_and_missing_acc_return_nan(self):
        _assert_all_nan(acc_features(None))
        _assert_all_nan(acc_features(_make_series([1.0] * (MIN_SAMPLES - 1))))


# =============================================================================
# F. HR summary statistics, missing data, constants, short signals
# =============================================================================

class TestHRReferenceValues:
    def test_mean_sample_std_min_max_range(self):
        values = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0]
        result = hr_features(_make_series(values))
        assert result["hr_mean"] == pytest.approx(_mean(values), abs=1e-12)
        assert result["hr_std"] == pytest.approx(_sample_std(values), abs=1e-12)
        assert result["hr_std"] != pytest.approx(_population_std(values), abs=1e-9)
        assert result["hr_min"] == pytest.approx(60.0, abs=1e-12)
        assert result["hr_max"] == pytest.approx(82.0, abs=1e-12)
        assert result["hr_range"] == pytest.approx(22.0, abs=1e-12)

    def test_nan_samples_are_dropped_before_statistics(self):
        values = [70.0] * 5 + [float("nan")] + [80.0] * 5
        finite = [70.0] * 5 + [80.0] * 5
        result = hr_features(_make_series(values))
        assert result["hr_mean"] == pytest.approx(_mean(finite), abs=1e-12)
        assert result["hr_min"] == pytest.approx(70.0, abs=1e-12)
        assert result["hr_max"] == pytest.approx(80.0, abs=1e-12)

    def test_constant_hr(self):
        values = [75.0] * 12
        result = hr_features(_make_series(values))
        assert result["hr_mean"] == pytest.approx(75.0, abs=1e-12)
        assert result["hr_std"] == pytest.approx(0.0, abs=1e-12)
        assert result["hr_range"] == pytest.approx(0.0, abs=1e-12)
        assert result["hr_diff_rms"] == pytest.approx(0.0, abs=1e-12)

    def test_none_empty_and_short_hr_return_nan(self):
        _assert_all_nan(hr_features(None))
        _assert_all_nan(hr_features(_make_series([])))
        _assert_all_nan(hr_features(_make_series([70.0] * (MIN_SAMPLES - 1))))

    def test_exactly_min_samples_is_computed(self):
        values = [60.0 + t for t in range(MIN_SAMPLES)]
        result = hr_features(_make_series(values))
        assert result["hr_mean"] == pytest.approx(_mean(values), abs=1e-12)
        assert not math.isnan(result["hr_diff_rms"])
