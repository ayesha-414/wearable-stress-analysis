"""
src/loader.py
=============
Portable signal loader for the PhysioNet Wearable Device Dataset v1.0.1.

Reads all Empatica E4 CSV files (ACC, BVP, EDA, HR, IBI, TEMP, tags) for every
subject across STRESS, AEROBIC, and ANAEROBIC activity folders.

Key design decisions
--------------------
- Paths are resolved relative to the project root (no hardcoded user paths).
- Each signal file is returned as a DataFrame with a proper DatetimeIndex.
- IBI is kept as irregular time-series; all others are resampled to 1 Hz for
  downstream feature extraction.
- Known data constraints (split sessions, duplicated data, empty files) are
  handled explicitly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Wearable_Dataset"
STRESS_L_V1 = PROJECT_ROOT / "Stress_Level_v1.csv"
STRESS_L_V2 = PROJECT_ROOT / "Stress_Level_v2.csv"
SUBJECT_INFO = PROJECT_ROOT / "subject-info.csv"

ACTIVITIES = ["STRESS", "AEROBIC", "ANAEROBIC"]

# Known duplicate-data start rows (0-indexed data rows, AFTER header strip)
DUPLICATED_SAMPLES = {
    "STRESS/S02": {
        "ACC": 49544,   # row 49,545 in original file (1-indexed)
        "BVP": 99090,
        "EDA": 6194,
        "TEMP": 6194,
    }
}

# Subjects where specific sensors are invalid
INVALID_SENSORS = {
    "STRESS/f07": ["BVP", "HR", "IBI", "TEMP"],
}

# Split sessions that must be concatenated
SPLIT_SESSIONS: dict[str, list[str]] = {
    "STRESS/f14":      ["f14_a", "f14_b"],
    "AEROBIC/S11":     ["S11_a", "S11_b"],
    "ANAEROBIC/S16":   ["S16_a", "S16_b"],
}

# Known empty files
EMPTY_FILES = {"ANAEROBIC/S01/IBI"}

SIGNAL_FILES = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP", "tags"]


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _parse_utc(val: str) -> float:
    """
    Parse a UTC value that may be either:
      - A datetime string: '2013-02-20 17:55:19'  → Unix float
      - Already a float/int Unix timestamp: 1361382919.0
    Returns seconds as float (Unix epoch).
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    try:
        return pd.Timestamp(str(val)).timestamp()
    except Exception:
        return 0.0


def _read_empatica_csv(path: Path, signal_name: str) -> Optional[pd.DataFrame]:
    """
    Read a single Empatica E4 CSV file and return a DataFrame.

    The dataset has UTC timestamps pre-converted from Unix to datetime strings
    (e.g. '2013-02-20 17:55:19') as noted in the README.

    Row 0 → UTC datetime string for session start
    Row 1 → Sampling frequency in Hz  (for regular signals)
    Row 2+ → Signal values

    IBI: col0 = offset in seconds (float), col1 = interval in seconds.
    ACC: three columns (x, y, z) at 32 Hz.
    tags: each row is a UTC datetime string.
    """
    if not path.exists():
        logger.debug("File not found: %s", path)
        return None

    if signal_name == "tags":
        try:
            df = pd.read_csv(path, header=None, names=["utc_raw"])
            if df.empty:
                return None
            # Convert datetime strings to Unix seconds
            utc_vals = df["utc_raw"].apply(_parse_utc)
            df["utc_ts"] = utc_vals
            df["time_s"] = utc_vals - utc_vals.iloc[0]
            return df
        except pd.errors.EmptyDataError:
            return None

    if signal_name == "IBI":
        try:
            raw_head = pd.read_csv(path, header=None, nrows=1)
            start_utc = _parse_utc(raw_head.iloc[0, 0])
            # IBI rows after header: col0=offset_s, col1=ibi_s
            # Some files have only 1 header row (no fs row)
            df = pd.read_csv(path, header=None, skiprows=1,
                             names=["offset_s", "ibi_s"])
            # Drop rows where offset_s is not float-parseable (extra text lines)
            df = df[pd.to_numeric(df["offset_s"], errors="coerce").notna()].copy()
            df["offset_s"] = df["offset_s"].astype(float)
            df["ibi_s"] = df["ibi_s"].astype(float)
            if df.empty:
                return None
            df["utc_ts"] = start_utc + df["offset_s"]
            return df
        except (pd.errors.EmptyDataError, ValueError, IndexError):
            return None

    try:
        raw = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError:
        return None

    start_utc = _parse_utc(raw.iloc[0, 0])
    fs = float(raw.iloc[1, 0])
    data = raw.iloc[2:].reset_index(drop=True)

    # Convert all columns to float
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.astype(float)

    n = len(data)
    times = start_utc + np.arange(n) / fs

    if signal_name == "ACC":
        data.columns = list(range(data.shape[1]))
        data = data.rename(columns={0: "x", 1: "y", 2: "z"})
    else:
        data.columns = [signal_name] + [f"_extra_{i}" for i in range(data.shape[1] - 1)]

    data.insert(0, "utc_ts", times)
    data["fs"] = fs
    return data


def _truncate_duplicates(df: pd.DataFrame, signal_name: str, key: str) -> pd.DataFrame:
    """Remove known duplicated tails from a DataFrame."""
    dup_map = DUPLICATED_SAMPLES.get(key, {})
    cut = dup_map.get(signal_name)
    if cut is not None and len(df) > cut:
        df = df.iloc[:cut].copy()
        logger.warning("Truncated duplicate tail: %s / %s at row %d", key, signal_name, cut)
    return df


# ---------------------------------------------------------------------------
# Subject-level loader
# ---------------------------------------------------------------------------

def load_subject(activity: str, subject_id: str) -> dict[str, Optional[pd.DataFrame]]:
    """
    Load all signal files for one subject/activity.

    Returns
    -------
    dict mapping signal name → DataFrame (or None if unavailable/invalid)
    """
    key = f"{activity}/{subject_id}"
    folder = DATASET_DIR / activity / subject_id
    signals: dict[str, Optional[pd.DataFrame]] = {}

    invalid = INVALID_SENSORS.get(key, [])

    for sig in SIGNAL_FILES:
        if sig in invalid:
            logger.info("Skipping invalid sensor %s for %s", sig, key)
            signals[sig] = None
            continue

        path = folder / f"{sig}.csv"
        df = _read_empatica_csv(path, sig)

        if df is not None and key in DUPLICATED_SAMPLES:
            df = _truncate_duplicates(df, sig, key)

        signals[sig] = df

    return signals


def _load_split_session(activity: str, canonical_id: str, parts: list[str]) -> dict:
    """Concatenate split sessions into a single signal dict."""
    all_parts: list[dict] = []
    for part_id in parts:
        folder = DATASET_DIR / activity / part_id
        if not folder.exists():
            logger.warning("Split part not found: %s / %s", activity, part_id)
            continue
        part_signals = {}
        for sig in SIGNAL_FILES:
            df = _read_empatica_csv(folder / f"{sig}.csv", sig)
            part_signals[sig] = df
        all_parts.append(part_signals)

    if not all_parts:
        return {s: None for s in SIGNAL_FILES}

    merged: dict[str, Optional[pd.DataFrame]] = {}
    for sig in SIGNAL_FILES:
        parts_dfs = [p[sig] for p in all_parts if p.get(sig) is not None]
        if not parts_dfs:
            merged[sig] = None
        elif len(parts_dfs) == 1:
            merged[sig] = parts_dfs[0].copy()
        else:
            merged[sig] = pd.concat(parts_dfs, ignore_index=True)
    return merged


# ---------------------------------------------------------------------------
# Full dataset loader
# ---------------------------------------------------------------------------

def load_all(activities: list[str] = ACTIVITIES) -> dict[str, dict[str, dict]]:
    """
    Load the entire dataset.

    Returns
    -------
    data[activity][subject_id][signal_name] → DataFrame or None
    """
    data: dict[str, dict[str, dict]] = {}

    for activity in activities:
        act_dir = DATASET_DIR / activity
        if not act_dir.exists():
            logger.warning("Activity folder not found: %s", act_dir)
            continue

        data[activity] = {}

        # Identify canonical subjects (skip split-part folders like S11_a)
        all_folders = sorted(f.name for f in act_dir.iterdir() if f.is_dir())
        split_parts: set[str] = set()
        for key, parts in SPLIT_SESSIONS.items():
            act_key, _ = key.split("/", 1)
            if act_key == activity:
                split_parts.update(parts)

        canonical_subjects = [f for f in all_folders if f not in split_parts]

        for subject_id in canonical_subjects:
            key = f"{activity}/{subject_id}"
            if key in SPLIT_SESSIONS:
                logger.info("Loading split session: %s", key)
                signals = _load_split_session(activity, subject_id, SPLIT_SESSIONS[key])
            else:
                signals = load_subject(activity, subject_id)

            data[activity][subject_id] = signals
            logger.debug("Loaded: %s / %s", activity, subject_id)

    return data


# ---------------------------------------------------------------------------
# Metadata loaders
# ---------------------------------------------------------------------------

def load_stress_levels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stress_v1, stress_v2) DataFrames."""
    v1 = pd.read_csv(STRESS_L_V1, index_col=0)
    v2 = pd.read_csv(STRESS_L_V2, index_col=0)
    return v1, v2


def load_subject_info() -> pd.DataFrame:
    """Return subject demographics DataFrame."""
    df = pd.read_csv(SUBJECT_INFO, index_col=0)
    # Drop trailing empty rows
    df = df[df.index.notna() & (df.index.str.strip() != "")]
    # Drop reference rows at bottom
    df = df[df.index.str.match(r'^[Sf]\d+')].copy()
    return df


def get_subject_ids() -> dict[str, list[str]]:
    """Return {activity: [subject_ids]} with split parts replaced by canonical names."""
    ids: dict[str, list[str]] = {}
    for activity in ACTIVITIES:
        act_dir = DATASET_DIR / activity
        if not act_dir.exists():
            ids[activity] = []
            continue
        split_parts: set[str] = set()
        canonical_splits: set[str] = set()
        for key, parts in SPLIT_SESSIONS.items():
            act_key, canonical_id = key.split("/", 1)
            if act_key == activity:
                split_parts.update(parts)
                canonical_splits.add(canonical_id)
        subjects = set(
            f.name for f in act_dir.iterdir()
            if f.is_dir() and f.name not in split_parts
        )
        subjects.update(canonical_splits)
        ids[activity] = sorted(subjects)
    return ids
