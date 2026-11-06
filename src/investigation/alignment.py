"""
src/investigation/alignment.py
==============================
Physiological sensor alignment and cleaning with verified temporal synchronization.

SIGN CONVENTION FOR HR ALIGNMENT:
---------------------------------
1. The Empatica E4 device starts recording EDA at session UTC timestamp T0.
2. The HR recording starts exactly 10 seconds later at T0 + 10 (verified across all 37 recordings).
3. HR row k is recorded at UTC timestamp T0 + 10 + k, corresponding to EDA-relative second (10 + k).
4. Therefore, the physiological value recorded at EDA-relative second t (for t >= 10) corresponds to
   HR row (t - 10).
5. In this implementation:
       hr_resampled.index = hr_resampled.index + 10
   When queried at integer second t >= 10, hr_resampled[t] returns the HR measurement taken at
   EDA second t (row t - 10). For seconds 0 <= t < 10, no HR measurement exists (represented as NaN).
6. Contrast with legacy defect in src/preprocessing.py:
   The legacy code ran `hr.reindex(hr.index - 10)`. Because reindex looks up by index label,
   it left the value from HR row t sitting at index t (which corresponds to measurement taken at t + 10).
   The new function clean_subject_signals_v2 fixes this without modifying legacy files.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing import clean_eda, clean_hr, clean_temp, moving_average_acc, resample_to_1hz

logger = logging.getLogger(__name__)


def align_hr_to_eda(hr_df: pd.DataFrame, max_eda_seconds: Optional[int] = None) -> pd.Series:
    """
    Resample HR to 1 Hz and align to EDA time base (where t=0 is EDA session start).

    HR starts 10 s after EDA. HR row k is placed at index 10 + k.
    For 0 <= t < 10, values are NaN.
    """
    if hr_df is None or len(hr_df) == 0:
        return pd.Series([], dtype=float)

    hr_1hz = clean_hr(resample_to_1hz(hr_df, "HR"))
    if len(hr_1hz) == 0:
        return pd.Series([], dtype=float)

    # Shift indices so that HR row 0 is at EDA-relative second 10
    aligned_index = hr_1hz.index + 10
    hr_aligned = pd.Series(hr_1hz.values, index=aligned_index, name="HR")

    if max_eda_seconds is not None and max_eda_seconds > 0:
        full_index = pd.RangeIndex(0, max_eda_seconds + 1)
        hr_aligned = hr_aligned.reindex(full_index, fill_value=np.nan)

    return hr_aligned


def clean_subject_signals_v2(signals: dict) -> dict:
    """
    Apply cleaned 1-Hz resampling and corrected HR alignment to one subject's signals.

    Parameters
    ----------
    signals : dict containing DataFrames for 'EDA', 'HR', 'TEMP', 'BVP', 'ACC', 'IBI', 'tags'

    Returns
    -------
    dict with:
        HR, EDA, TEMP, BVP, ACC_mag -> pd.Series indexed by seconds from EDA start
        IBI_raw -> DataFrame
        tags_s -> list of tag timestamps relative to EDA start
        session_start_utc -> float Unix timestamp of EDA start
    """
    cleaned: dict = {}

    eda_df = signals.get("EDA")
    session_start_utc = None
    eda_max_sec = 0

    if eda_df is not None and isinstance(eda_df, pd.DataFrame) and len(eda_df) > 0:
        session_start_utc = float(eda_df["utc_ts"].iloc[0])
        cleaned["EDA"] = clean_eda(resample_to_1hz(eda_df, "EDA"))
        eda_max_sec = int(cleaned["EDA"].index[-1]) if len(cleaned["EDA"]) > 0 else 0
    else:
        cleaned["EDA"] = None

    cleaned["session_start_utc"] = session_start_utc

    # HR with corrected +10s alignment to EDA time base
    hr_df = signals.get("HR")
    if hr_df is not None and isinstance(hr_df, pd.DataFrame) and len(hr_df) > 0:
        cleaned["HR"] = align_hr_to_eda(hr_df, max_eda_seconds=eda_max_sec)
    else:
        cleaned["HR"] = None

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

    cleaned["IBI_raw"] = signals.get("IBI")

    # Tags relative to EDA session start
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

