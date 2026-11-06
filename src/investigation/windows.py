"""
src/investigation/windows.py
============================
Window extraction logic for the NeuroTwin model investigation.

Supports three pre-specified window variants:
- Variant A   : Full corrected stage window [stage_start, stage_end]
- Variant B   : Primary design: 30 s window anchored at stage start [stage_start, min(stage_end, stage_start + 30)]
- Variant Bend: Sensitivity design: 30 s window anchored at stage end [max(stage_start, stage_end - 30), stage_end]

Records exact start, end, and duration for every window to guarantee auditability.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.investigation.stage_maps import get_stage_map_v2

logger = logging.getLogger(__name__)

WindowVariant = Literal["A", "B", "Bend"]


def segment_by_stage_v2(
    cleaned: dict,
    activity: str,
    subject_id: str,
    variant: WindowVariant = "B",
) -> dict[str, tuple[dict, dict]]:
    """
    Slice cleaned signals into stage windows according to the specified window variant.

    Parameters
    ----------
    cleaned : dict from clean_subject_signals_v2
    activity : str (e.g. "STRESS")
    subject_id : str (e.g. "S01", "f02")
    variant : {"A", "B", "Bend"}

    Returns
    -------
    dict mapping stage_name -> (window_signals_dict, window_metadata_dict)
    where window_metadata_dict contains:
        window_start_s, window_end_s, window_duration_s, stage_duration_s, variant
    """
    tags = cleaned.get("tags_s", [])
    stage_map = get_stage_map_v2(activity, subject_id)
    segments: dict[str, tuple[dict, dict]] = {}

    for stage_name, (t_start_idx, t_end_idx) in stage_map.items():
        if t_start_idx >= len(tags) or t_end_idx > len(tags):
            continue

        stage_t0 = float(tags[t_start_idx])
        stage_t1 = float(tags[t_end_idx])
        stage_dur = stage_t1 - stage_t0

        if variant == "A":
            w_start = stage_t0
            w_end = stage_t1
        elif variant == "B":
            w_start = stage_t0
            w_end = min(stage_t1, stage_t0 + 30.0)
        elif variant == "Bend":
            w_start = max(stage_t0, stage_t1 - 30.0)
            w_end = stage_t1
        else:
            raise ValueError(f"Unknown window variant: {variant}")

        w_dur = max(0.0, w_end - w_start)

        meta = {
            "window_start_s": round(w_start, 2),
            "window_end_s": round(w_end, 2),
            "window_duration_s": round(w_dur, 2),
            "stage_duration_s": round(stage_dur, 2),
            "variant": variant,
        }

        stage_sigs: dict = {}
        t0_int = int(round(w_start))
        t1_int = int(round(w_end))

        for sig in ("HR", "EDA", "TEMP", "BVP", "ACC_mag"):
            series = cleaned.get(sig)
            if series is None:
                stage_sigs[sig] = None
                continue
            idx = series.index
            mask = (idx >= t0_int) & (idx <= t1_int)
            stage_sigs[sig] = series[mask]

        ibi_raw = cleaned.get("IBI_raw")
        session_start_utc = cleaned.get("session_start_utc")
        if (ibi_raw is not None and isinstance(ibi_raw, pd.DataFrame)
                and "utc_ts" in ibi_raw.columns
                and session_start_utc is not None):
            t0_utc = session_start_utc + w_start
            t1_utc = session_start_utc + w_end
            mask = (ibi_raw["utc_ts"] >= t0_utc) & (ibi_raw["utc_ts"] < t1_utc)
            stage_sigs["IBI_raw"] = ibi_raw[mask].copy()
        elif (ibi_raw is not None and isinstance(ibi_raw, pd.DataFrame)
              and "offset_s" in ibi_raw.columns):
            mask = (ibi_raw["offset_s"] >= w_start) & (ibi_raw["offset_s"] < w_end)
            stage_sigs["IBI_raw"] = ibi_raw[mask].copy()
        else:
            stage_sigs["IBI_raw"] = None

        segments[stage_name] = (stage_sigs, meta)

    return segments

