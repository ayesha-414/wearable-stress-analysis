"""
src/investigation/features_v2.py
================================
Feature extraction for the NeuroTwin model investigation.

Calls existing feature extraction routines in src/features.py unchanged.
Adds normalized per-minute rate features (e.g. eda_scr_per_min) as extra columns,
without replacing or altering any of the 23 original feature columns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features import extract_features

logger = logging.getLogger(__name__)


def extract_features_v2(
    stage_signals: dict,
    window_meta: dict,
    activity: str,
    subject_id: str,
    stage_name: str,
    stress_level: Optional[float] = None,
) -> dict:
    """
    Extract canonical 23 features using existing src/features functions unchanged,
    then add per-minute rate columns and window tracking metadata.
    """
    # 1. Call existing feature extractor unchanged
    row = extract_features(stage_signals, activity, subject_id, stage_name)

    # 2. Attach window metadata
    row["window_start_s"] = window_meta.get("window_start_s", np.nan)
    row["window_end_s"] = window_meta.get("window_end_s", np.nan)
    row["window_duration_s"] = window_meta.get("window_duration_s", np.nan)
    row["stage_duration_s"] = window_meta.get("stage_duration_s", np.nan)
    row["variant"] = window_meta.get("variant", "")

    # 3. Add per-minute rate feature(s) without replacing count features
    duration_s = row["window_duration_s"]
    scr_count = row.get("eda_scr_count")
    if duration_s is not None and not np.isnan(duration_s) and duration_s > 0 and scr_count is not None and not np.isnan(scr_count):
        row["eda_scr_per_min"] = float(scr_count) / (float(duration_s) / 60.0)
    else:
        row["eda_scr_per_min"] = np.nan

    # 4. Attach ground truth label
    row["stress_level"] = stress_level if stress_level is not None else np.nan
    if stress_level is not None and not np.isnan(stress_level):
        row["stress_binary"] = int(stress_level >= 5.0)
    else:
        row["stress_binary"] = np.nan

    return row

