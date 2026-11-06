"""
scripts/01_load_and_verify.py
==============================
Step 1: Load all signals, verify data integrity, and save a summary CSV.

Outputs
-------
data/processed/dataset_summary.csv   — per-subject signal availability
logs/01_load_verify.log
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import (
    ACTIVITIES,
    load_all,
    load_stress_levels,
    load_subject_info,
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "01_load_verify.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("STEP 1 — Load & Verify Dataset")
    logger.info("=" * 60)

    # --- Load all sensor data ---
    logger.info("Loading all sensor signals...")
    data = load_all()

    # --- Build per-subject summary ---
    rows = []
    for activity in ACTIVITIES:
        subjects = data.get(activity, {})
        logger.info("  %s: %d subjects", activity, len(subjects))
        for subject_id, signals in subjects.items():
            row = {"activity": activity, "subject": subject_id}
            for sig_name, df in signals.items():
                if sig_name == "tags_s":
                    row["tags_count"] = len(df) if df is not None else 0
                    continue
                if df is None:
                    row[f"{sig_name}_rows"] = 0
                elif isinstance(df, pd.DataFrame):
                    row[f"{sig_name}_rows"] = len(df)
                else:
                    row[f"{sig_name}_rows"] = len(df) if hasattr(df, "__len__") else -1
            rows.append(row)

    summary_df = pd.DataFrame(rows).fillna(0)

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Saved dataset summary: %s (%d rows)", out_path, len(summary_df))

    # --- Load metadata ---
    sl_v1, sl_v2 = load_stress_levels()
    logger.info("Stress Level V1 shape: %s", sl_v1.shape)
    logger.info("Stress Level V2 shape: %s", sl_v2.shape)

    subj_info = load_subject_info()
    logger.info("Subject info shape: %s", subj_info.shape)

    # --- Print quick stats ---
    total_subjects = sum(len(data.get(a, {})) for a in ACTIVITIES)
    logger.info("Total subject-activity pairs loaded: %d", total_subjects)

    missing = summary_df[summary_df["HR_rows"] == 0]
    logger.info("Missing HR signals: %d entries", len(missing))

    logger.info("STEP 1 COMPLETE.")
    return summary_df


if __name__ == "__main__":
    main()
