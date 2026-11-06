"""
scripts/13_build_features_v2.py
===============================
Extracts features using the new investigation pipeline:
1. Notebook-consistent stage maps (V1 and V2)
2. Corrected HR-to-EDA alignment (+10 s)
3. Window variants:
   - Variant A   : Full corrected stage window
   - Variant B   : Primary design: 30 s from stage start
   - Variant Bend: Sensitivity design: 30 s from stage end
   - Optional    : --include-f14 for f14 sensitivity analysis

Outputs:
  data/features_v2/stress_features_v2_A.csv
  data/features_v2/stress_features_v2_B.csv (primary)
  data/features_v2/stress_features_v2_Bend.csv
  data/features_v2/stress_features_v2_B_f14_sensitivity.csv (if --include-f14)

Refuses to overwrite existing files unless explicitly flagged.
"""


from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import src.loader as loader
from src.investigation.alignment import clean_subject_signals_v2
from src.investigation.features_v2 import extract_features_v2
from src.investigation.windows import segment_by_stage_v2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "features_v2"


def build_features_for_variant(
    variant: str,
    include_f14: bool = False,
    force: bool = False,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if include_f14:
        out_path = OUT_DIR / f"stress_features_v2_{variant}_f14_sensitivity.csv"
    else:
        out_path = OUT_DIR / f"stress_features_v2_{variant}.csv"

    if out_path.exists() and not force:
        raise FileExistsError(
            f"Destination file already exists: {out_path}. "
            "Refusing to overwrite without --force flag."
        )

    logger.info("Building features for variant %s (include_f14=%s) -> %s",
                variant, include_f14, out_path.name)

    stress_v1, stress_v2 = loader.load_stress_levels()
    all_sids = loader.get_subject_ids()["STRESS"]

    if include_f14:
        subjects_to_process = all_sids
    else:
        subjects_to_process = [s for s in all_sids if s != "f14"]

    rows = []

    for sid in subjects_to_process:
        if sid == "f14":
            sig = loader._load_split_session("STRESS", "f14", ["f14_a", "f14_b"])
        else:
            sig = loader.load_subject("STRESS", sid)

        cleaned = clean_subject_signals_v2(sig)
        segments = segment_by_stage_v2(cleaned, "STRESS", sid, variant=variant)

        cohort_df = stress_v1 if sid.startswith("S") else stress_v2

        for stage_name, (stage_sigs, meta) in segments.items():
            sl = float("nan")
            if sid in cohort_df.index and stage_name in cohort_df.columns:
                sl = float(cohort_df.loc[sid, stage_name])

            row = extract_features_v2(
                stage_signals=stage_sigs,
                window_meta=meta,
                activity="STRESS",
                subject_id=sid,
                stage_name=stage_name,
                stress_level=sl,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows (%d participants) to %s", len(df), df["subject"].nunique(), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build features v2 for model investigation.")
    parser.add_argument("--include-f14", action="store_true", help="Include f14 split session as sensitivity analysis.")
    parser.add_argument("--variants", nargs="+", default=["A", "B", "Bend"], help="Window variants to build (A, B, Bend).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if present.")
    args = parser.parse_args()

    for var in args.variants:
        build_features_for_variant(var, include_f14=False, force=args.force)

    if args.include_f14:
        build_features_for_variant("B", include_f14=True, force=args.force)


if __name__ == "__main__":
    main()

