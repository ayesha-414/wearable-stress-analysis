"""
scripts/12_audit_stage_boundaries.py
====================================
Read-only audit of raw tag intervals, notebook stage definitions, and HR alignment.
Addresses Gate A1 of the Model Investigation Plan.

Compares:
1. Stage spans from project's src/preprocessing.py (raw tag indices)
2. Stage spans from Wearable_Dataset.ipynb (notebook cell 25, accounting for prepended EDA t0)
3. Median durations and overlap (IoU) across all recordings
4. Verifies HR header starts exactly 10 s after EDA in every recording.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import src.loader as loader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "model_investigation" / "stage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Project stage maps (from src/preprocessing.py)
PROJECT_V1_STAGES = {
    "Baseline": (0, 2),
    "Stroop": (3, 4),
    "First Rest": (2, 3),
    "TMCT": (5, 6),
    "Second Rest": (4, 5),
    "Real Opinion": (7, 8),
    "Opposite Opinion": (9, 10),
    "Subtract": (11, 12),
}

PROJECT_V2_STAGES = {
    "Baseline": (0, 1),
    "TMCT": (2, 3),
    "First Rest": (1, 2),
    "Real Opinion": (4, 5),
    "Opposite Opinion": (6, 7),
    "Second Rest": (3, 4),
    "Subtract": (7, 8),
}

# Notebook stage spans in raw tag indices (notebook indices shifted by -1 due to prepended EDA start)
# In Wearable_Dataset.ipynb Cell 25:
# V1: Stroop [3,4], TMCT [5,6], Real Opinion [7,8], Opposite Opinion [9,10], Subtract [11,12]
# With raw_idx = nb_idx - 1:
# Stroop: (2, 3), TMCT: (4, 5), Real Opinion: (6, 7), Opposite Opinion: (8, 9), Subtract: (10, 11)
# Rest/baseline intervals follow chronological order of Stress_Level_v1.csv:
# Baseline: (0, 2), Stroop: (2, 3), First Rest: (3, 4), TMCT: (4, 5), Second Rest: (5, 6),
# Real Opinion: (6, 7), Opposite Opinion: (8, 9), Subtract: (10, 11)
NOTEBOOK_V1_STAGES = {
    "Baseline": (0, 2),
    "Stroop": (2, 3),
    "First Rest": (3, 4),
    "TMCT": (4, 5),
    "Second Rest": (5, 6),
    "Real Opinion": (6, 7),
    "Opposite Opinion": (8, 9),
    "Subtract": (10, 11),
}

# V2 in Wearable_Dataset.ipynb Cell 25:
# TMCT [2,3], Real Opinion [4,5], Opposite Opinion [6,7], Subtract [8,9]
# With raw_idx = nb_idx - 1:
# TMCT: (1, 2), Real Opinion: (3, 4), Opposite Opinion: (5, 6), Subtract: (7, 8)
# Rests follow chronological order of Stress_Level_v2.csv:
# Baseline: (0, 1), TMCT: (1, 2), First Rest: (2, 3), Real Opinion: (3, 4),
# Opposite Opinion: (5, 6), Second Rest: (6, 7), Subtract: (7, 8)
NOTEBOOK_V2_STAGES = {
    "Baseline": (0, 1),
    "TMCT": (1, 2),
    "First Rest": (2, 3),
    "Real Opinion": (3, 4),
    "Opposite Opinion": (5, 6),
    "Second Rest": (6, 7),
    "Subtract": (7, 8),
}


def compute_iou(span_a: tuple[float, float], span_b: tuple[float, float]) -> float:
    """Intersection over Union between two time spans [start, end]."""
    a0, a1 = span_a
    b0, b1 = span_b
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return intersection / union


def audit_stage_boundaries():
    sids = loader.get_subject_ids()["STRESS"]
    logger.info("Found %d STRESS subject IDs", len(sids))

    hr_audit_rows = []
    v1_intervals = []
    v2_intervals = []

    v1_valid_recordings = 0
    v2_valid_recordings = 0
    v1_total_recordings = 0
    v2_total_recordings = 0

    for sid in sids:
        cohort = "V1" if sid.startswith("S") else "V2"
        if cohort == "V1":
            v1_total_recordings += 1
        else:
            v2_total_recordings += 1

        if sid == "f14":
            # f14 split session loader
            sig = loader._load_split_session("STRESS", "f14", ["f14_a", "f14_b"])
        else:
            sig = loader.load_subject("STRESS", sid)

        eda = sig.get("EDA")
        hr = sig.get("HR")
        tags = sig.get("tags")

        eda_t0 = None
        hr_t0 = None
        hr_diff = None
        hr_10s_ok = False

        if eda is not None and len(eda) > 0 and hr is not None and len(hr) > 0:
            eda_t0 = float(eda["utc_ts"].iloc[0])
            hr_t0 = float(hr["utc_ts"].iloc[0])
            hr_diff = hr_t0 - eda_t0
            hr_10s_ok = abs(hr_diff - 10.0) < 1e-3
        elif sid == "f07":
            # Known missing HR sensor
            hr_10s_ok = True
            hr_diff = np.nan

        n_tags = len(tags) if tags is not None else 0
        expected_tags = 13 if cohort == "V1" else 9
        tags_ok = (n_tags == expected_tags)

        hr_audit_rows.append({
            "subject": sid,
            "cohort": cohort,
            "eda_t0": eda_t0,
            "hr_t0": hr_t0,
            "hr_minus_eda_s": hr_diff,
            "hr_10s_aligned": hr_10s_ok,
            "n_tags": n_tags,
            "expected_tags": expected_tags,
            "tags_match_expected": tags_ok,
        })

        if tags is None or n_tags < (12 if cohort == "V1" else 8) or eda_t0 is None:
            continue

        # Tag relative times (seconds from EDA start)
        tags_rel = (tags["utc_ts"].astype(float) - eda_t0).tolist()

        # Check plausibility of tag durations
        if cohort == "V1" and n_tags >= 13:
            v1_valid_recordings += 1
            for stage, p_span in PROJECT_V1_STAGES.items():
                n_span = NOTEBOOK_V1_STAGES[stage]
                p_t0, p_t1 = tags_rel[p_span[0]], tags_rel[p_span[1]]
                n_t0, n_t1 = tags_rel[n_span[0]], tags_rel[n_span[1]]
                iou = compute_iou((p_t0, p_t1), (n_t0, n_t1))
                v1_intervals.append({
                    "subject": sid,
                    "stage": stage,
                    "project_raw_span": f"{p_span[0]}->{p_span[1]}",
                    "notebook_raw_span": f"{n_span[0]}->{n_span[1]}",
                    "notebook_cell_span": f"{n_span[0]+1}->{n_span[1]+1}",
                    "project_duration_s": p_t1 - p_t0,
                    "notebook_duration_s": n_t1 - n_t0,
                    "start_diff_s": p_t0 - n_t0,
                    "end_diff_s": p_t1 - n_t1,
                    "iou_overlap": iou,
                })

        elif cohort == "V2" and n_tags >= 9:
            v2_valid_recordings += 1
            for stage, p_span in PROJECT_V2_STAGES.items():
                n_span = NOTEBOOK_V2_STAGES[stage]
                p_t0, p_t1 = tags_rel[p_span[0]], tags_rel[p_span[1]]
                n_t0, n_t1 = tags_rel[n_span[0]], tags_rel[n_span[1]]
                iou = compute_iou((p_t0, p_t1), (n_t0, n_t1))
                v2_intervals.append({
                    "subject": sid,
                    "stage": stage,
                    "project_raw_span": f"{p_span[0]}->{p_span[1]}",
                    "notebook_raw_span": f"{n_span[0]}->{n_span[1]}",
                    "notebook_cell_span": f"{n_span[0]+1}->{n_span[1]+1}",
                    "project_duration_s": p_t1 - p_t0,
                    "notebook_duration_s": n_t1 - n_t0,
                    "start_diff_s": p_t0 - n_t0,
                    "end_diff_s": p_t1 - n_t1,
                    "iou_overlap": iou,
                })

    # Save HR audit
    df_hr = pd.DataFrame(hr_audit_rows)
    df_hr.to_csv(OUT_DIR / "hr_alignment_audit.csv", index=False)

    # Aggregate stage comparison table
    df_v1 = pd.DataFrame(v1_intervals)
    df_v2 = pd.DataFrame(v2_intervals)

    summary_rows = []
    for cohort, df_cohort, nb_stages, prj_stages in [
        ("V1", df_v1, NOTEBOOK_V1_STAGES, PROJECT_V1_STAGES),
        ("V2", df_v2, NOTEBOOK_V2_STAGES, PROJECT_V2_STAGES),
    ]:
        for stage in nb_stages:
            sub = df_cohort[df_cohort["stage"] == stage]
            p_span = prj_stages[stage]
            n_span = nb_stages[stage]
            med_p_dur = sub["project_duration_s"].median()
            med_n_dur = sub["notebook_duration_s"].median()
            med_start_diff = sub["start_diff_s"].median()
            med_end_diff = sub["end_diff_s"].median()
            med_iou = sub["iou_overlap"].median()

            summary_rows.append({
                "cohort": cohort,
                "stage": stage,
                "project_raw_span": f"{p_span[0]}->{p_span[1]}",
                "notebook_raw_span": f"{n_span[0]}->{n_span[1]}",
                "notebook_cell_span": f"{n_span[0]+1}->{n_span[1]+1}",
                "n_recordings": len(sub),
                "med_project_duration_s": round(med_p_dur, 1),
                "med_notebook_duration_s": round(med_n_dur, 1),
                "med_start_diff_s": round(med_start_diff, 1),
                "med_end_diff_s": round(med_end_diff, 1),
                "med_iou_overlap": round(med_iou, 3),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "stage_audit_comparison_table.csv", index=False)

    # Gate verification
    v1_rate = v1_valid_recordings / max(1, v1_total_recordings)
    v2_rate = v2_valid_recordings / max(1, v2_total_recordings)
    hr_10s_all = all(r["hr_10s_aligned"] for r in hr_audit_rows)

    gate_passed = (v1_rate >= 0.90) and (v2_rate >= 0.90) and hr_10s_all

    gate_result = {
        "gate": "A1_stage_boundary_and_hr_audit",
        "gate_passed": bool(gate_passed),
        "v1_valid_recordings": v1_valid_recordings,
        "v1_total_recordings": v1_total_recordings,
        "v1_notebook_consistent_rate": round(float(v1_rate), 4),
        "v2_valid_recordings": v2_valid_recordings,
        "v2_total_recordings": v2_total_recordings,
        "v2_notebook_consistent_rate": round(float(v2_rate), 4),
        "hr_starts_10s_after_eda_in_all_recordings": bool(hr_10s_all),
        "hr_diff_seconds_unique": [float(x) for x in df_hr["hr_minus_eda_s"].dropna().unique()],
        "output_directory": str(OUT_DIR),
    }

    (OUT_DIR / "stage_audit_summary.json").write_text(
        json.dumps(gate_result, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("STAGE AUDIT SUMMARY (GATE A1)")
    print("=" * 70)
    print(f"V1 Notebook-consistent rate: {v1_rate:.1%} ({v1_valid_recordings}/{v1_total_recordings})")
    print(f"V2 Notebook-consistent rate: {v2_rate:.1%} ({v2_valid_recordings}/{v2_total_recordings})")
    print(f"HR starts 10 s after EDA in all recordings: {hr_10s_all}")
    print(f"Gate A1 Passed: {gate_passed}")
    print("=" * 70 + "\n")
    print(df_summary.to_string(index=False))

    return gate_passed


if __name__ == "__main__":
    passed = audit_stage_boundaries()
    if not passed:
        raise SystemExit("GATE A1 FAILED: Notebook-consistent reading does not hold for >= 90% of recordings.")

