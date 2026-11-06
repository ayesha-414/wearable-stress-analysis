"""
src/features.py
===============
Physiological feature extraction from cleaned wearable signals.

Features extracted per stage window
-------------------------------------
HR  -> mean, std, min, max, range, rmssd_approx
EDA -> mean, std, slope, range, scl_mean, scr_count
TEMP-> mean, std, slope
ACC -> mean, std, max  (activity magnitude)
IBI -> mean_ibi, sdnn, rmssd, pnn50, lf_hf_ratio
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

logger = logging.getLogger(__name__)

MIN_SAMPLES = 10

# Feature columns used throughout the pipeline (updated name)
FEATURE_COLS = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_diff_rms",
    "eda_mean", "eda_std", "eda_slope", "eda_range", "eda_scl_mean", "eda_scr_count",
    "temp_mean", "temp_std", "temp_slope",
    "acc_mean", "acc_std", "acc_max",
    "hrv_mean_ibi", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf_ratio",
]





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_slope(series: pd.Series) -> float:
    """Linear regression slope."""
    y = series.dropna().values
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    if np.std(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _rolling_tonic(eda: pd.Series, window_s: int = 60) -> pd.Series:
    """Approximate SCL (tonic) as rolling mean."""
    return eda.rolling(window=window_s, min_periods=1, center=True).mean()


def _scr_peak_count(eda: pd.Series, tonic: pd.Series, threshold: float = 0.02) -> int:
    """Count phasic SCR rising edges above tonic + threshold."""
    phasic = eda - tonic
    above = (phasic > threshold).astype(int)
    edges = above.diff().clip(lower=0).sum()
    return int(edges)


# ---------------------------------------------------------------------------
# Per-signal feature extractors
# ---------------------------------------------------------------------------

def hr_features(series: Optional[pd.Series]) -> dict:
    prefix = "hr_"
    nans = {f"{prefix}{k}": float("nan") for k in
            ["mean", "std", "min", "max", "range", "diff_rms"]}
    if series is None or len(series.dropna()) < MIN_SAMPLES:
        return nans
    s = series.dropna()
    diff = s.diff().dropna()
    rmssd = float(np.sqrt(np.mean(diff.values ** 2))) if len(diff) > 1 else float("nan")
    return {
        "hr_mean":         float(s.mean()),
        "hr_std":          float(s.std()),
        "hr_min":          float(s.min()),
        "hr_max":          float(s.max()),
        "hr_range":        float(s.max() - s.min()),
        "hr_diff_rms":     rmssd,  # RMS of consecutive 1-Hz HR differences (not IBI RMSSD)
    }


def eda_features(series: Optional[pd.Series]) -> dict:
    prefix = "eda_"
    nans = {f"{prefix}{k}": float("nan") for k in
            ["mean", "std", "slope", "range", "scl_mean", "scr_count"]}
    if series is None or len(series.dropna()) < MIN_SAMPLES:
        return nans
    s = series.dropna()
    tonic = _rolling_tonic(s)
    return {
        "eda_mean":      float(s.mean()),
        "eda_std":       float(s.std()),
        "eda_slope":     _safe_slope(s),
        "eda_range":     float(s.max() - s.min()),
        "eda_scl_mean":  float(tonic.mean()),
        "eda_scr_count": _scr_peak_count(s, tonic),
    }


def temp_features(series: Optional[pd.Series]) -> dict:
    nans = {k: float("nan") for k in ["temp_mean", "temp_std", "temp_slope"]}
    if series is None or len(series.dropna()) < MIN_SAMPLES:
        return nans
    s = series.dropna()
    return {
        "temp_mean":  float(s.mean()),
        "temp_std":   float(s.std()),
        "temp_slope": _safe_slope(s),
    }


def acc_features(series: Optional[pd.Series]) -> dict:
    nans = {k: float("nan") for k in ["acc_mean", "acc_std", "acc_max"]}
    if series is None or len(series.dropna()) < MIN_SAMPLES:
        return nans
    s = series.dropna()
    return {
        "acc_mean": float(s.mean()),
        "acc_std":  float(s.std()),
        "acc_max":  float(s.max()),
    }


def ibi_features(ibi_df: Optional[pd.DataFrame]) -> dict:
    """HRV features from IBI DataFrame (columns: offset_s, ibi_s)."""
    nans = {k: float("nan") for k in
            ["hrv_mean_ibi", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf_ratio"]}
    if ibi_df is None or not isinstance(ibi_df, pd.DataFrame):
        return nans
    df = ibi_df.dropna(subset=["ibi_s"])
    if len(df) < MIN_SAMPLES:
        return nans

    ibi = df["ibi_s"].values.astype(float)
    ibi_ms = ibi * 1000.0

    succ = np.diff(ibi_ms)
    rmssd = float(np.sqrt(np.mean(succ ** 2))) if len(succ) > 0 else float("nan")
    pnn50 = float(np.mean(np.abs(succ) > 50)) if len(succ) > 0 else float("nan")

    lf_hf = float("nan")
    try:
        t = df["offset_s"].values.astype(float)
        ibi_c = ibi - ibi.mean()
        freqs = np.linspace(0.003, 0.4, 500)
        omegas = 2 * np.pi * freqs
        pgram = lombscargle(t, ibi_c, omegas, normalize=True)
        lf = pgram[(freqs >= 0.04) & (freqs <= 0.15)].mean()
        hf = pgram[(freqs > 0.15) & (freqs <= 0.4)].mean()
        lf_hf = float(lf / hf) if hf > 0 else float("nan")
    except Exception as exc:
        logger.debug("LF/HF failed: %s", exc)

    return {
        "hrv_mean_ibi":    float(np.mean(ibi)),
        "hrv_sdnn":        float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else float("nan"),
        "hrv_rmssd":       rmssd,
        "hrv_pnn50":       pnn50,
        "hrv_lf_hf_ratio": lf_hf,
    }


# ---------------------------------------------------------------------------
# Combined per-window extractor
# ---------------------------------------------------------------------------

def extract_features(stage_signals: dict, activity: str,
                     subject_id: str, stage_name: str) -> dict:
    """Extract all features from one stage window."""
    row: dict = {
        "subject":  subject_id,
        "activity": activity,
        "stage":    stage_name,
        "cohort":   "V2" if subject_id.startswith("f") else "V1",
    }
    row.update(hr_features(stage_signals.get("HR")))
    row.update(eda_features(stage_signals.get("EDA")))
    row.update(temp_features(stage_signals.get("TEMP")))
    row.update(acc_features(stage_signals.get("ACC_mag")))
    row.update(ibi_features(stage_signals.get("IBI_raw")))
    return row


# ---------------------------------------------------------------------------
# Full dataset feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(all_data: dict, stress_v1: pd.DataFrame,
                         stress_v2: pd.DataFrame,
                         activity: str = "STRESS") -> pd.DataFrame:
    """
    Extract features for all subjects in an activity and attach stress labels.

    Returns DataFrame with one row per (subject, stage).
    """
    from src.preprocessing import clean_subject_signals, segment_by_stage

    rows = []
    subjects = all_data.get(activity, {})

    for subject_id, signals in subjects.items():
        try:
            cleaned = clean_subject_signals(signals)
            segments = segment_by_stage(cleaned, activity, subject_id)

            for stage_name, stage_sigs in segments.items():
                row = extract_features(stage_sigs, activity, subject_id, stage_name)

                if activity == "STRESS":
                    cohort_df = stress_v1 if subject_id.startswith("S") else stress_v2
                    sl = float("nan")
                    if subject_id in cohort_df.index and stage_name in cohort_df.columns:
                        sl = float(cohort_df.loc[subject_id, stage_name])
                    row["stress_level"] = sl

                rows.append(row)
        except Exception as exc:
            logger.warning("Feature extraction failed %s/%s: %s",
                           activity, subject_id, exc)

    return pd.DataFrame(rows)
