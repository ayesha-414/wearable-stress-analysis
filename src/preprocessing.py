"""
src/preprocessing.py
====================
Signal cleaning, resampling, and stage segmentation for the wearable dataset.

Steps applied per subject
--------------------------
1. Convert UTC timestamps -> relative seconds (t=0 at signal start)
2. Resample all regular signals to 1 Hz (uniform grid for feature extraction)
3. Remove physiological outliers (clip EDA < 0, HR outside [30, 220])
4. Linear interpolation for short gaps (<= 5 s)
5. Extract per-stage segments using tags.csv event marks
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage definitions (tag index slices per protocol version)
# ---------------------------------------------------------------------------

# V1 stress protocol -- 12+ tags -> 8 stages
STRESS_V1_STAGES = {
    "Baseline":         (0, 2),
    "Stroop":           (3, 4),
    "First Rest":       (2, 3),
    "TMCT":             (5, 6),
    "Second Rest":      (4, 5),
    "Real Opinion":     (7, 8),
    "Opposite Opinion": (9, 10),
    "Subtract":         (11, 12),
}

# V2 stress protocol -- 9 tags -> 7 stages
STRESS_V2_STAGES = {
    "Baseline":         (0, 1),
    "TMCT":             (2, 3),
    "First Rest":       (1, 2),
    "Real Opinion":     (4, 5),
    "Opposite Opinion": (6, 7),
    "Second Rest":      (3, 4),
    "Subtract":         (7, 8),
}

AEROBIC_V1_STAGES = {
    "Rest":   (0, 1),
    "60rpm":  (1, 2),
    "65rpm":  (2, 3),
    "70rpm":  (3, 4),
    "75rpm":  (4, 5),
    "80rpm":  (5, 6),
    "85rpm":  (6, 7),
    "90rpm":  (7, 8),
    "95rpm":  (8, 9),
    "100rpm": (9, 10),
    "105rpm": (10, 11),
    "110rpm": (11, 12),
}

AEROBIC_V2_STAGES = {
    "Rest":     (0, 1),
    "60rpm":    (1, 2),
    "70rpm":    (2, 3),
    "80rpm":    (3, 4),
    "90rpm":    (4, 5),
    "100rpm":   (5, 6),
    "110rpm":   (6, 7),
    "Recovery": (7, 8),
}

ANAEROBIC_V1_STAGES = {
    "Rest":    (0, 1),
    "Sprint1": (1, 2),
    "Rest2":   (2, 3),
    "Sprint2": (3, 4),
    "Rest3":   (4, 5),
    "Sprint3": (5, 6),
    "Rest4":   (6, 7),
}

ANAEROBIC_V2_STAGES = {
    "Rest":    (0, 1),
    "Sprint1": (1, 2),
    "Rest2":   (2, 3),
    "Sprint2": (3, 4),
    "Rest3":   (4, 5),
    "Sprint3": (5, 6),
    "Rest4":   (6, 7),
    "Sprint4": (7, 8),
    "Rest5":   (8, 9),
    "Sprint5": (9, 10),
    "Rest6":   (10, 11),
}

STAGE_MAPS = {
    "STRESS":    {"S": STRESS_V1_STAGES,   "f": STRESS_V2_STAGES},
    "AEROBIC":   {"S": AEROBIC_V1_STAGES,  "f": AEROBIC_V2_STAGES},
    "ANAEROBIC": {"S": ANAEROBIC_V1_STAGES, "f": ANAEROBIC_V2_STAGES},
}


# ---------------------------------------------------------------------------
# Core cleaning helpers
# ---------------------------------------------------------------------------

def to_relative_seconds(df: pd.DataFrame, utc_col: str = "utc_ts") -> pd.DataFrame:
    """Shift UTC timestamps so t=0 is session start (non-destructive copy)."""
    df = df.copy()
    df[utc_col] = df[utc_col] - df[utc_col].iloc[0]
    return df


def resample_to_1hz(df: pd.DataFrame, signal_col: str,
                    utc_col: str = "utc_ts") -> pd.Series:
    """
    Resample a regular-sampled signal to 1 Hz using mean aggregation.
    Returns a Series indexed by integer seconds from 0 (relative time).
    """
    df = df.copy()
    t0 = float(df[utc_col].iloc[0])
    df["second"] = (df[utc_col].astype(float) - t0).astype(int)
    resampled = df.groupby("second")[signal_col].mean()
    return resampled


def clean_eda(series: pd.Series) -> pd.Series:
    """Clip EDA to [0, 60] uS and interpolate short gaps."""
    s = series.clip(lower=0, upper=60)
    s = s.interpolate(method="linear", limit=5)
    return s


def clean_hr(series: pd.Series) -> pd.Series:
    """Clip HR to physiological range [30, 220] bpm."""
    s = series.clip(lower=30, upper=220)
    s = s.interpolate(method="linear", limit=5)
    return s


def clean_temp(series: pd.Series) -> pd.Series:
    """Clip skin TEMP to [20, 42] deg C."""
    s = series.clip(lower=20, upper=42)
    s = s.interpolate(method="linear", limit=5)
    return s


def moving_average_acc(acc_df: pd.DataFrame, window: int = 32) -> pd.Series:
    """
    Replicate the original notebook's activity index from ACC.
    For each 32-sample window: sum of max(|Dx|, |Dy|, |Dz|), EMA smoothed.
    Returns a Series of activity magnitude at ~1 sample/second.
    """
    vals = acc_df[["x", "y", "z"]].values.astype(float)
    avg = 0.0
    results = []
    prev = np.zeros(3)
    for i in range(0, len(vals), window):
        buf = vals[i:i + window]
        s = np.max(np.abs(buf - prev), axis=1).sum()
        avg = avg * 0.9 + (s / window) * 0.1
        results.append(avg)
        if len(buf) > 0:
            prev = buf[-1]
    return pd.Series(results)


# ---------------------------------------------------------------------------
# Subject-level cleaning pipeline
# ---------------------------------------------------------------------------

def clean_subject_signals(signals: dict) -> dict:
    """
    Apply full cleaning pipeline to one subject's raw signals.

    Parameters
    ----------
    signals : dict  (output of loader.load_subject or _load_split_session)

    Returns
    -------
    dict with keys:
      HR, EDA, TEMP, BVP  -> cleaned 1-Hz pd.Series (index = seconds from 0)
      ACC_mag              -> activity magnitude pd.Series
      IBI_raw              -> raw IBI DataFrame
      tags_s               -> list[float] of tag times relative to EDA start
      session_start_utc    -> float Unix timestamp of EDA session start
    """
    cleaned: dict = {}

    # Determine session start from EDA
    eda_df = signals.get("EDA")
    session_start_utc = None
    if eda_df is not None and isinstance(eda_df, pd.DataFrame) and len(eda_df) > 0:
        session_start_utc = float(eda_df["utc_ts"].iloc[0])
    cleaned["session_start_utc"] = session_start_utc

    # HR
    hr_df = signals.get("HR")
    if hr_df is not None and isinstance(hr_df, pd.DataFrame) and len(hr_df) > 0:
        cleaned["HR"] = clean_hr(resample_to_1hz(hr_df, "HR"))
    else:
        cleaned["HR"] = None

    # EDA
    if eda_df is not None and isinstance(eda_df, pd.DataFrame) and len(eda_df) > 0:
        cleaned["EDA"] = clean_eda(resample_to_1hz(eda_df, "EDA"))
    else:
        cleaned["EDA"] = None

    # TEMP
    temp_df = signals.get("TEMP")
    if temp_df is not None and isinstance(temp_df, pd.DataFrame) and len(temp_df) > 0:
        cleaned["TEMP"] = clean_temp(resample_to_1hz(temp_df, "TEMP"))
    else:
        cleaned["TEMP"] = None

    # BVP
    bvp_df = signals.get("BVP")
    if bvp_df is not None and isinstance(bvp_df, pd.DataFrame) and len(bvp_df) > 0:
        cleaned["BVP"] = resample_to_1hz(bvp_df, "BVP")
    else:
        cleaned["BVP"] = None

    # ACC activity magnitude
    acc_df = signals.get("ACC")
    if acc_df is not None and isinstance(acc_df, pd.DataFrame) and len(acc_df) > 0:
        cleaned["ACC_mag"] = moving_average_acc(acc_df)
    else:
        cleaned["ACC_mag"] = None

    # IBI raw (kept for HRV)
    cleaned["IBI_raw"] = signals.get("IBI")

    # tags: align to EDA session start using utc_ts
    tags_df = signals.get("tags")
    if (tags_df is not None and isinstance(tags_df, pd.DataFrame)
            and len(tags_df) > 0 and "utc_ts" in tags_df.columns):
        if session_start_utc is not None:
            tags_rel = (tags_df["utc_ts"].astype(float) - session_start_utc).tolist()
        else:
            tags_rel = tags_df.get("time_s", pd.Series([], dtype=float)).tolist()
        cleaned["tags_s"] = [float(t) for t in tags_rel]
    else:
        cleaned["tags_s"] = []

    return cleaned


# ---------------------------------------------------------------------------
# Stage segmentation
# ---------------------------------------------------------------------------

def get_stage_map(activity: str, subject_id: str) -> dict:
    """Return the correct stage map for this subject/activity combination."""
    cohort_key = "f" if subject_id.startswith("f") else "S"
    return STAGE_MAPS.get(activity, {}).get(cohort_key, {})


def segment_by_stage(cleaned: dict, activity: str, subject_id: str) -> dict:
    """
    Slice each cleaned signal into per-stage windows using tag timestamps.

    Returns
    -------
    dict: stage_name -> {signal_name -> pd.Series slice or DataFrame}
    """
    tags = cleaned.get("tags_s", [])
    stage_map = get_stage_map(activity, subject_id)
    segments: dict = {}

    for stage_name, (t_start_idx, t_end_idx) in stage_map.items():
        if t_start_idx >= len(tags) or t_end_idx > len(tags):
            continue

        t0 = int(tags[t_start_idx])
        t1 = int(tags[t_end_idx]) if t_end_idx < len(tags) else None

        stage_sigs: dict = {}

        for sig in ("HR", "EDA", "TEMP", "BVP", "ACC_mag"):
            series = cleaned.get(sig)
            if series is None:
                stage_sigs[sig] = None
                continue
            idx = series.index
            mask = idx >= t0
            if t1 is not None:
                mask = mask & (idx <= t1)
            stage_sigs[sig] = series[mask]

        # IBI within window -- uses offset_s column
        ibi_raw = cleaned.get("IBI_raw")
        if (ibi_raw is not None and isinstance(ibi_raw, pd.DataFrame)
                and "offset_s" in ibi_raw.columns):
            mask = ibi_raw["offset_s"] >= t0
            if t1 is not None:
                mask = mask & (ibi_raw["offset_s"] < t1)
            stage_sigs["IBI_raw"] = ibi_raw[mask].copy()
        else:
            stage_sigs["IBI_raw"] = None

        segments[stage_name] = stage_sigs

    return segments
