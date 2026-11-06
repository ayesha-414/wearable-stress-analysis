"""
Stage 1 review tests for src/features.py  (Requirement 1, NeuroTwin AI).

Scope
-----
These tests ADD evidence about four behaviours that the first set of reference
tests (test_feature_reference_values.py) left open:

  A. SCR threshold: exact-boundary strictness and first-sample behaviour.
  B. LF/HF: averaging Lomb-Scargle ordinates versus integrating band power.
  C. IBI time-domain HRV (RMSSD, pNN50) when beats are dropped or rejected.

Rules followed
--------------
* Production code is NOT modified and NOT used to compute expected values.
  Every expected number is derived in this file from first-principles
  arithmetic (see each docstring for the derivation).
* Tests that merely CHARACTERISE current behaviour say so explicitly. A passing
  characterisation test does not mean the behaviour is scientifically correct.
* Where behaviour contradicts a recognised definition, a separate test marked
  xfail(strict=True) states the expected-by-the-standard behaviour. It will
  start failing loudly (XPASS) once production code is corrected, which
  forces the marker to be removed deliberately.
* Self-contained: no imports from other test modules.

Passing these tests does not establish physiological validity on real
wearable recordings.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lombscargle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import _scr_peak_count, eda_features, ibi_features  # noqa: E402


# ---------------------------------------------------------------------------
# Independent helpers (not production code)
# ---------------------------------------------------------------------------

def _mean(values) -> float:
    xs = [float(v) for v in values]
    return sum(xs) / len(xs)


def _sample_std(values) -> float:
    xs = [float(v) for v in values]
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _rms(values) -> float:
    xs = [float(v) for v in values]
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def _successive_diffs(values) -> list[float]:
    xs = [float(v) for v in values]
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def _centered_rolling_mean(values, window: int = 60) -> list[float]:
    """Centred rolling mean; window at i is [i-(W-1)//2, i+W//2], clipped."""
    xs = [float(v) for v in values]
    n = len(xs)
    left, right = (window - 1) // 2, window // 2
    return [_mean(xs[max(0, i - left): min(n, i + right + 1)]) for i in range(n)]


def _flags(values, threshold: float = 0.02) -> list[int]:
    tonic = _centered_rolling_mean(values, window=60)
    return [1 if (v - t) > threshold else 0 for v, t in zip(values, tonic)]


def _rising_edges_excluding_first(flags) -> int:
    """Count 0 -> 1 transitions between consecutive samples (index >= 1)."""
    return sum(1 for i in range(1, len(flags)) if flags[i] == 1 and flags[i - 1] == 0)


def _make_series(values) -> pd.Series:
    return pd.Series([float(v) for v in values])


def _ibi_frame(ibi_s, offsets=None) -> pd.DataFrame:
    """offset_s = time of each beat (cumulative sum of intervals), as in E4 IBI.csv."""
    ibi = [float(v) for v in ibi_s]
    if offsets is None:
        offsets, t = [], 0.0
        for interval in ibi:
            t += interval
            offsets.append(t)
    return pd.DataFrame({"offset_s": offsets, "ibi_s": ibi})


# ===========================================================================
# A. SCR threshold behaviour
# ===========================================================================

class TestSCRThresholdStrictness:
    """
    Why this test exists
    --------------------
    The earlier boundary test tried to build a phasic value of "exactly" 0.02
    out of arithmetic on 1.0 and 1.0 + 0.02*12/11. In IEEE-754 that value came
    out as 0.020000000000000018 (1.7e-17 above 0.02), so `> 0.02` was correctly
    True. That was a TEST-DESIGN flaw, not a production defect.

    To test strictness honestly we need an input whose phasic value is exactly
    the double nearest 0.02. Subtracting a tonic of exactly 0.0 is exact in
    IEEE-754, so `eda - tonic == 0.02` bit-for-bit, and the comparison
    `0.02 > 0.02` is False. We therefore call the helper `_scr_peak_count`
    directly with a zero tonic. Expected values below are hand-specified.
    """

    @staticmethod
    def _count(values) -> int:
        eda = _make_series(values)
        tonic = _make_series([0.0] * len(values))
        return _scr_peak_count(eda, tonic)

    def test_phasic_exactly_equal_to_threshold_is_not_counted(self):
        # phasic = [0, 0, 0.02, 0, 0] - 0 -> the peak equals the threshold.
        assert self._count([0.0, 0.0, 0.02, 0.0, 0.0]) == 0

    def test_phasic_just_above_threshold_is_counted(self):
        # 0.02 + 1e-9 is representable and strictly greater than 0.02.
        assert self._count([0.0, 0.0, 0.02 + 1e-9, 0.0, 0.0]) == 1

    def test_phasic_just_below_threshold_is_not_counted(self):
        assert self._count([0.0, 0.0, 0.02 - 1e-9, 0.0, 0.0]) == 0

    def test_two_separate_excursions_count_twice(self):
        # flags: 0 1 0 1 0 -> two 0->1 edges.
        assert self._count([0.0, 0.5, 0.0, 0.5, 0.0]) == 2

    def test_sustained_excursion_counts_once(self):
        # flags: 0 1 1 1 0 -> a single 0->1 edge, not three.
        assert self._count([0.0, 0.5, 0.5, 0.5, 0.0]) == 1


class TestSCRFirstSampleBehaviour:
    """
    Production logic (src/features.py::_scr_peak_count):

        above = (phasic > threshold).astype(int)
        edges = above.diff().clip(lower=0).sum()

    Series.diff() is NaN at index 0, and NaN is skipped by sum(). Therefore a
    window whose FIRST sample is already above threshold gets no edge for that
    initial high state. An excursion is only counted when a 0 -> 1 transition
    is observed inside the window.

    SCIENTIFIC STATUS: uncertain, not a clear defect. If the window opens
    mid-response, the response onset happened before the window, so not
    counting it is defensible; counting it would be defensible too. The tests
    below document the current behaviour so that any future change is a
    deliberate choice.

    All signals have n = 12 (> MIN_SAMPLES, <= 30) so the 60-sample centred
    rolling mean covers the whole series: tonic equals the global mean at
    every index, which makes the expected flags easy to derive by hand.
    """

    def test_first_sample_high_is_not_counted_as_an_edge(self):
        # values: [2, 1 x 11]; tonic = (2 + 11) / 12 = 13/12 = 1.08333
        # phasic[0] = 2 - 1.08333 = +0.9167 (> 0.02)  -> flag 1
        # phasic[1..] = 1 - 1.08333 = -0.0833         -> flag 0
        # flags = [1, 0, 0, ...]; there is no 0 -> 1 transition => 0 edges.
        values = [2.0] + [1.0] * 11
        flags = _flags(values)
        assert flags[0] == 1 and sum(flags) == 1
        expected = _rising_edges_excluding_first(flags)
        assert expected == 0
        assert eda_features(_make_series(values))["eda_scr_count"] == expected

    def test_counting_initial_high_state_would_give_a_different_answer(self):
        # Alternative convention (NOT production): also count an initial 1 as
        # the start of a response. Shows the choice is not cosmetic.
        values = [2.0] + [1.0] * 11
        flags = _flags(values)
        alternative = _rising_edges_excluding_first(flags) + flags[0]
        assert alternative == 1
        production = eda_features(_make_series(values))["eda_scr_count"]
        assert production != alternative

    def test_only_the_second_of_two_excursions_is_counted_when_first_is_at_start(self):
        # values = [2,1,1,1,2] + [1]*7, sum = 14, tonic = 14/12 = 1.16667
        # phasic(2) = +0.8333 -> flag 1; phasic(1) = -0.1667 -> flag 0
        # flags = [1,0,0,0,1,0,0,0,0,0,0,0] -> one 0->1 edge (index 4).
        values = [2.0, 1.0, 1.0, 1.0, 2.0] + [1.0] * 7
        flags = _flags(values)
        assert flags == [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        expected = _rising_edges_excluding_first(flags)
        assert expected == 1
        assert eda_features(_make_series(values))["eda_scr_count"] == expected

    def test_excursion_starting_after_first_sample_is_counted(self):
        # values = [1, 2] + [1]*10 -> flags = [0,1,0,...] -> one edge at index 1.
        values = [1.0, 2.0] + [1.0] * 10
        flags = _flags(values)
        assert flags[:3] == [0, 1, 0]
        expected = _rising_edges_excluding_first(flags)
        assert expected == 1
        assert eda_features(_make_series(values))["eda_scr_count"] == expected

    def test_excursion_on_the_last_sample_is_counted(self):
        # values = [1]*11 + [2] -> flags = [0,...,0,1] -> edge at final index.
        values = [1.0] * 11 + [2.0]
        flags = _flags(values)
        assert flags[-1] == 1 and sum(flags) == 1
        expected = _rising_edges_excluding_first(flags)
        assert expected == 1
        assert eda_features(_make_series(values))["eda_scr_count"] == expected


# ===========================================================================
# B. LF/HF: mean of Lomb-Scargle ordinates versus integrated band power
# ===========================================================================
#
# Production (src/features.py::ibi_features), paraphrased:
#     freqs = np.linspace(0.003, 0.4, 500)
#     pgram = lombscargle(t, ibi - mean, 2*pi*freqs, normalize=True)
#     lf = pgram[(freqs >= 0.04) & (freqs <= 0.15)].mean()
#     hf = pgram[(freqs >  0.15) & (freqs <= 0.40)].mean()
#     lf_hf = lf / hf
#
# The conventional LF/HF is a ratio of band POWERS (area under the spectrum
# in each band; Task Force 1996; Shaffer & Ginsberg 2017: "Power is the
# signal energy found within a frequency band").
#
# On a uniform frequency grid with spacing df:
#     band power  ~=  df * sum(P over band)
#     band mean   =   sum(P over band) / N_band
# so   mean_LF/mean_HF  =  (sum_LF/sum_HF) * (N_HF/N_LF)
# i.e. the production ratio is the band-power ratio multiplied by the ratio
# of the number of grid points in the two bands. This is an algebraic
# identity and holds for ANY spectrum.

# Grid constants copied from production so the reference uses the same grid.
_F_LO, _F_HI, _N_GRID = 0.003, 0.4, 500
_FREQS = np.linspace(_F_LO, _F_HI, _N_GRID)
_LF_MASK = (_FREQS >= 0.04) & (_FREQS <= 0.15)
_HF_MASK = (_FREQS > 0.15) & (_FREQS <= 0.40)


def _grid_point_counts_exact() -> tuple[int, int]:
    """
    Count grid points per band with exact rational arithmetic (no floats).

    f_i = 0.003 + 0.397 * i / 499,  i = 0..499
    LF: 0.04 <= f_i <= 0.15  =>  i in [ceil(0.037*499/0.397), floor(0.147*499/0.397)]
    HF: 0.15 <  f_i <= 0.40  =>  i in [floor(0.147*499/0.397)+1, 499]
    """
    step = Fraction("0.397") / 499
    lf_lo = math.ceil((Fraction("0.04") - Fraction("0.003")) / step)
    lf_hi = math.floor((Fraction("0.15") - Fraction("0.003")) / step)
    hf_lo = lf_hi + 1                       # strictly above 0.15 (no tie: 184.77 is not an integer)
    hf_hi = 499                              # f_499 = 0.4 exactly
    return lf_hi - lf_lo + 1, hf_hi - hf_lo + 1


def _lomb_scargle_power(t, y, freqs_hz) -> np.ndarray:
    """
    Independent Lomb-Scargle periodogram (Lomb 1976; Scargle 1982), mean-centred.

        P(w) = 0.5 * [ (sum y cos w(t-tau))^2 / sum cos^2 w(t-tau)
                     + (sum y sin w(t-tau))^2 / sum sin^2 w(t-tau) ]
        tan(2 w tau) = sum sin 2wt / sum cos 2wt
    Returned in un-normalised units; any constant normalisation cancels in a ratio.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float) - np.mean(y)
    out = np.empty(len(freqs_hz))
    for k, f in enumerate(freqs_hz):
        w = 2.0 * np.pi * f
        tau = np.arctan2(np.sum(np.sin(2 * w * t)), np.sum(np.cos(2 * w * t))) / (2 * w)
        c, s = np.cos(w * (t - tau)), np.sin(w * (t - tau))
        out[k] = 0.5 * ((y @ c) ** 2 / (c @ c) + (y @ s) ** 2 / (s @ s))
    return out


def _synthetic_ibi(duration_s=300.0, base=0.8, a_lf=0.03, a_hf=0.03, f_lf=0.10, f_hf=0.25):
    """
    Deterministic IBI series: baseline 0.8 s plus a 0.10 Hz (LF) and a 0.25 Hz (HF)
    sinusoid of EQUAL amplitude, sampled at the beat times themselves.
    Returns (beat_times_s, ibi_s).
    """
    t, times, ibi = 0.0, [], []
    while t < duration_s:
        v = base + a_lf * math.sin(2 * math.pi * f_lf * t) + a_hf * math.sin(2 * math.pi * f_hf * t)
        t += v
        times.append(t)
        ibi.append(v)
    return np.array(times), np.array(ibi)


def _band_power_ratio_sum(p) -> float:
    """Rectangle rule on the uniform grid: ratio of sums (df cancels)."""
    return float(p[_LF_MASK].sum() / p[_HF_MASK].sum())


def _band_power_ratio_trapezoid(p) -> float:
    """Trapezoid rule on the uniform grid (own implementation, numpy-version independent)."""
    df = _FREQS[1] - _FREQS[0]

    def _trap(y):
        return float(np.sum((y[:-1] + y[1:]) / 2.0) * df)

    return _trap(p[_LF_MASK]) / _trap(p[_HF_MASK])


@pytest.fixture(scope="module")
def equal_power_case():
    t, ibi = _synthetic_ibi()
    p = _lomb_scargle_power(t, ibi, _FREQS)
    prod = ibi_features(_ibi_frame(ibi, offsets=t))["hrv_lf_hf_ratio"]
    return {"t": t, "ibi": ibi, "p": p, "production": prod}


class TestLfHfBandAveragingVersusBandPower:
    def test_grid_point_counts_by_exact_arithmetic(self):
        n_lf, n_hf = _grid_point_counts_exact()
        assert (n_lf, n_hf) == (138, 315)
        # The float-mask version used below agrees with the exact count.
        assert int(_LF_MASK.sum()) == n_lf
        assert int(_HF_MASK.sum()) == n_hf

    def test_independent_periodogram_matches_scipy_up_to_a_constant(self, equal_power_case):
        # Validates the REFERENCE implementation. SciPy normalize=True divides by a
        # frequency-independent constant, so P_indep / P_scipy must be flat.
        t, ibi, p = equal_power_case["t"], equal_power_case["ibi"], equal_power_case["p"]
        p_scipy = lombscargle(t, ibi - ibi.mean(), 2 * np.pi * _FREQS, normalize=True)
        ratio = p / p_scipy
        assert (ratio.max() - ratio.min()) / ratio.mean() < 1e-6

    def test_independent_periodogram_peaks_at_the_two_injected_frequencies(self, equal_power_case):
        p = equal_power_case["p"]
        lf_peak = _FREQS[_LF_MASK][np.argmax(p[_LF_MASK])]
        hf_peak = _FREQS[_HF_MASK][np.argmax(p[_HF_MASK])]
        assert lf_peak == pytest.approx(0.10, abs=0.005)
        assert hf_peak == pytest.approx(0.25, abs=0.005)

    def test_production_ratio_equals_ratio_of_band_means_of_independent_spectrum(self, equal_power_case):
        # Pins down exactly WHAT production computes, using the independent spectrum.
        p = equal_power_case["p"]
        expected = float(p[_LF_MASK].mean() / p[_HF_MASK].mean())
        assert equal_power_case["production"] == pytest.approx(expected, rel=1e-6)

    def test_ratio_of_means_is_ratio_of_sums_times_grid_point_ratio(self, equal_power_case):
        # Algebraic identity (any spectrum): mean-ratio = sum-ratio * N_HF / N_LF.
        p = equal_power_case["p"]
        n_lf, n_hf = _grid_point_counts_exact()
        mean_ratio = float(p[_LF_MASK].mean() / p[_HF_MASK].mean())
        assert mean_ratio == pytest.approx(_band_power_ratio_sum(p) * n_hf / n_lf, rel=1e-12)
        assert n_hf / n_lf == pytest.approx(2.2826086956521738, rel=1e-12)

    def test_identity_holds_for_an_arbitrary_synthetic_spectrum(self):
        # Same identity on a spectrum that has nothing to do with HRV: P(f) = 1 + f.
        p = 1.0 + _FREQS
        n_lf, n_hf = _grid_point_counts_exact()
        assert float(p[_LF_MASK].mean() / p[_HF_MASK].mean()) == pytest.approx(
            _band_power_ratio_sum(p) * n_hf / n_lf, rel=1e-12)

    def test_flat_spectrum_has_true_power_ratio_equal_to_bandwidth_ratio_but_mean_ratio_one(self):
        # P(f) = 1 everywhere. Band powers are proportional to bandwidths:
        #   LF width 0.15-0.04 = 0.11 Hz, HF width 0.40-0.15 = 0.25 Hz
        #   => true power ratio LF/HF = 0.11 / 0.25 = 0.44 (grid gives 138/315 = 0.438).
        # Mean-of-ordinates ratio is 1.0 regardless. Shows the two quantities differ
        # by the bandwidth ratio even for a trivial spectrum.
        p = np.ones_like(_FREQS)
        assert float(p[_LF_MASK].mean() / p[_HF_MASK].mean()) == pytest.approx(1.0, abs=1e-12)
        assert _band_power_ratio_sum(p) == pytest.approx(138 / 315, rel=1e-12)
        assert _band_power_ratio_sum(p) == pytest.approx(0.11 / 0.25, rel=0.01)

    def test_equal_power_sinusoids_true_ratio_is_about_one(self, equal_power_case):
        # Two sinusoids of equal amplitude A carry equal power A^2/2, so the
        # conventional band-power ratio should be ~1 (up to spectral leakage and
        # irregular sampling). Tolerance +/-5% was fixed before inspecting the result.
        p = equal_power_case["p"]
        assert 0.95 < _band_power_ratio_sum(p) < 1.05
        assert 0.95 < _band_power_ratio_trapezoid(p) < 1.05

    def test_CHARACTERISATION_production_ratio_is_inflated_by_the_grid_point_ratio(self, equal_power_case):
        """
        Characterises CURRENT behaviour (passes today; says nothing about being correct).
        For equal-power LF and HF components production returns ~2.28 rather than ~1.
        """
        p = equal_power_case["p"]
        n_lf, n_hf = _grid_point_counts_exact()
        production = equal_power_case["production"]
        assert production == pytest.approx(_band_power_ratio_sum(p) * n_hf / n_lf, rel=1e-6)
        assert production > 2.0
        assert production / _band_power_ratio_sum(p) == pytest.approx(315 / 138, rel=1e-6)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN DISCREPANCY (not yet fixed, awaiting approval): hrv_lf_hf_ratio averages "
            "periodogram ordinates per band instead of integrating band power, inflating the "
            "ratio by N_HF/N_LF = 315/138 ~= 2.28 on the current frequency grid."
        ),
    )
    def test_EXPECTED_BY_STANDARD_production_ratio_equals_band_power_ratio(self, equal_power_case):
        # Expected-by-the-standard behaviour: equal-power components -> LF/HF ~ 1.
        assert equal_power_case["production"] == pytest.approx(1.0, rel=0.10)


# ===========================================================================
# C. IBI time-domain HRV when beats are dropped / rejected
# ===========================================================================
#
# What production does (src/features.py::ibi_features):
#   * drops rows with NaN ibi_s, then
#   * differences CONSECUTIVE ROWS of ibi_s (np.diff), regardless of offset_s.
#
# Dataset definition (PhysioNet Wearable Device Dataset README): IBI.csv column 1
# is the time of the detected beat, column 2 is the interval "from the previous
# beat". Task Force 1996 defines RMSSD/NN50 on ADJACENT NN intervals.
#
# UNCERTAINTY (cannot be resolved without raw recordings): when the E4 rejects a
# beat, we do not know whether the next row holds the long interval spanning the
# rejected beat (scenario "missed beat") or an interval measured from the last
# accepted beat. Both scenarios are exercised below. No claim is made about how
# often either occurs in the real data.
#
# Synthetic series used for the "dropped row" scenarios:
#   ibi_k = 0.70 + 0.03*k  seconds, k = 0..14  (15 rows)
#   every successive difference is exactly +30 ms
#   => clean RMSSD = 30 ms, clean pNN50 = 0 (30 ms is not > 50 ms).

_DRIFT_IBI = [0.70 + 0.03 * k for k in range(15)]
_DROP_INDEX = 7


def _gap_aware_diffs_ms(offsets, ibi_s, tol_s=1e-6) -> list[float]:
    """
    HYPOTHETICAL alternative (not production, not asserted to be the right
    method): keep a successive difference only when the two rows are adjacent
    beats, i.e. when offset[k] - offset[k-1] equals ibi_s[k] within `tol_s`.
    """
    diffs = []
    for k in range(1, len(ibi_s)):
        if abs((offsets[k] - offsets[k - 1]) - ibi_s[k]) <= tol_s:
            diffs.append((ibi_s[k] - ibi_s[k - 1]) * 1000.0)
    return diffs


def _pnn50(diffs_ms) -> float:
    return sum(1 for d in diffs_ms if abs(d) > 50.0) / len(diffs_ms)


class TestIBIDroppedBeats:
    def test_clean_drift_series_reference_values(self):
        diffs = [d * 1000.0 for d in _successive_diffs(_DRIFT_IBI)]
        assert len(diffs) == 14 and all(d == pytest.approx(30.0, abs=1e-6) for d in diffs)
        result = ibi_features(_ibi_frame(_DRIFT_IBI))
        assert result["hrv_rmssd"] == pytest.approx(30.0, abs=1e-6)
        assert result["hrv_pnn50"] == 0.0

    def test_dropped_row_creates_a_double_width_difference_that_production_uses(self):
        """
        Remove row 7, keep the timestamps of the remaining beats (a beat that was
        never reported). Rows 6 and 8 are now neighbours in the file but two beats
        apart in time.

        Expected, by hand:
          diffs (ms) = 12 x 30  +  1 x 60   (13 differences; 60 = ibi_8 - ibi_6)
          RMSSD = sqrt((12*30^2 + 60^2) / 13) = sqrt(14400/13) = 33.2820 ms
          pNN50 = 1/13 = 0.076923          (only the 60 ms step exceeds 50 ms)
        Clean series: RMSSD 30, pNN50 0.
        """
        full_offsets = _ibi_frame(_DRIFT_IBI)["offset_s"].tolist()
        ibi = [v for i, v in enumerate(_DRIFT_IBI) if i != _DROP_INDEX]
        offsets = [v for i, v in enumerate(full_offsets) if i != _DROP_INDEX]

        adjacent_row_diffs = [d * 1000.0 for d in _successive_diffs(ibi)]
        assert len(adjacent_row_diffs) == 13
        assert sorted(round(d, 6) for d in adjacent_row_diffs) == [30.0] * 12 + [60.0]

        expected_rmssd = math.sqrt((12 * 30.0 ** 2 + 60.0 ** 2) / 13)
        expected_pnn50 = 1 / 13
        assert expected_rmssd == pytest.approx(33.28201, abs=1e-5)

        result = ibi_features(_ibi_frame(ibi, offsets=offsets))
        assert result["hrv_rmssd"] == pytest.approx(expected_rmssd, abs=1e-6)
        assert result["hrv_pnn50"] == pytest.approx(expected_pnn50, abs=1e-12)

        # The offsets expose the gap: across it, the time step equals ibi_6->8 spans two beats.
        gap_dt = offsets[_DROP_INDEX] - offsets[_DROP_INDEX - 1]
        assert gap_dt == pytest.approx(_DRIFT_IBI[_DROP_INDEX] + _DRIFT_IBI[_DROP_INDEX + 1], abs=1e-9)
        assert gap_dt > 1.5 * ibi[_DROP_INDEX]  # clearly more than one interval

    def test_hypothetical_gap_aware_reference_would_recover_clean_values(self):
        # NOT production. Shows the size of the effect: skipping the non-adjacent
        # pair gives 12 diffs of 30 ms -> RMSSD 30 ms and pNN50 0.
        full_offsets = _ibi_frame(_DRIFT_IBI)["offset_s"].tolist()
        ibi = [v for i, v in enumerate(_DRIFT_IBI) if i != _DROP_INDEX]
        offsets = [v for i, v in enumerate(full_offsets) if i != _DROP_INDEX]
        diffs = _gap_aware_diffs_ms(offsets, ibi)
        assert len(diffs) == 12
        assert _rms(diffs) == pytest.approx(30.0, abs=1e-6)
        assert _pnn50(diffs) == 0.0

        production = ibi_features(_ibi_frame(ibi, offsets=offsets))
        assert production["hrv_rmssd"] - _rms(diffs) == pytest.approx(3.28201, abs=1e-5)  # +3.28 ms (+10.9%)
        assert production["hrv_pnn50"] - _pnn50(diffs) == pytest.approx(1 / 13, abs=1e-12)

    def test_nan_row_is_treated_exactly_like_a_removed_row(self):
        # dropna() removes the NaN row, so rows 6 and 8 become neighbours.
        ibi_nan = list(_DRIFT_IBI)
        ibi_nan[_DROP_INDEX] = float("nan")
        removed = [v for i, v in enumerate(_DRIFT_IBI) if i != _DROP_INDEX]
        expected_rmssd = _rms([d * 1000.0 for d in _successive_diffs(removed)])
        result = ibi_features(_ibi_frame(ibi_nan))
        assert result["hrv_rmssd"] == pytest.approx(expected_rmssd, abs=1e-6)
        assert result["hrv_rmssd"] == pytest.approx(math.sqrt(14400 / 13), abs=1e-6)
        assert result["hrv_pnn50"] == pytest.approx(1 / 13, abs=1e-12)

    def test_time_domain_features_ignore_beat_timestamps(self):
        # Insert a 30 s hole in beat times (every offset after row 7 shifted +30 s)
        # while leaving ibi_s untouched. Time-domain HRV must be unchanged because
        # production never looks at offset_s for these features.
        base = _ibi_frame(_DRIFT_IBI)
        shifted = base.copy()
        shifted.loc[_DROP_INDEX:, "offset_s"] += 30.0
        a, b = ibi_features(base), ibi_features(shifted)
        for key in ("hrv_mean_ibi", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"):
            assert b[key] == pytest.approx(a[key], abs=1e-12), key
        assert a["hrv_mean_ibi"] == pytest.approx(_mean(_DRIFT_IBI), abs=1e-12)
        assert a["hrv_sdnn"] == pytest.approx(_sample_std([v * 1000 for v in _DRIFT_IBI]), abs=1e-9)

    def test_long_interval_from_a_missed_beat_is_not_filtered(self):
        """
        Scenario: the algorithm reports one interval spanning a missed beat
        (twice the normal 0.8 s).   ibi = [0.8]*7 + [1.6] + [0.8]*7   (15 rows)

        Differences (ms): 0 x6, +800, -800, 0 x6   -> 14 differences
          RMSSD = sqrt((800^2 + 800^2)/14) = 800/sqrt(7) = 302.372 ms
          NN50  = 2 of 14 -> pNN50 = 2/14 = 0.142857
        The underlying rhythm has zero variability (clean RMSSD = 0, pNN50 = 0), so
        every unit of the result is produced by the single artefact: production
        applies no ectopic/artefact rejection.
        """
        ibi = [0.8] * 7 + [1.6] + [0.8] * 7
        diffs = [d * 1000.0 for d in _successive_diffs(ibi)]
        assert len(diffs) == 14
        expected_rmssd = _rms(diffs)
        assert expected_rmssd == pytest.approx(800.0 / math.sqrt(7), abs=1e-9)
        assert expected_rmssd == pytest.approx(302.3716, abs=1e-4)

        clean = ibi_features(_ibi_frame([0.8] * 15))
        assert clean["hrv_rmssd"] == pytest.approx(0.0, abs=1e-9)
        assert clean["hrv_pnn50"] == 0.0

        result = ibi_features(_ibi_frame(ibi))
        assert result["hrv_rmssd"] == pytest.approx(expected_rmssd, abs=1e-6)
        assert result["hrv_pnn50"] == pytest.approx(2 / 14, abs=1e-12)

    def test_pnn50_denominator_is_number_of_pairs_not_number_of_intervals(self):
        # Convention check. Task Force Table 1 wording (as reproduced in secondary
        # sources): "NN50 count divided by the total number of all NN intervals"
        # (denominator N). Production divides by the number of successive PAIRS
        # (N - 1). Both conventions appear in the literature/software; they
        # differ by a factor (N-1)/N. For the 15-row missed-beat series:
        #   production 2/14 = 0.142857   versus   2/15 = 0.133333
        ibi = [0.8] * 7 + [1.6] + [0.8] * 7
        result = ibi_features(_ibi_frame(ibi))
        assert result["hrv_pnn50"] == pytest.approx(2 / 14, abs=1e-12)
        assert result["hrv_pnn50"] != pytest.approx(2 / 15, abs=1e-6)
