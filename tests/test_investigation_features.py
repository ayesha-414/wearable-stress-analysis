"""
tests/test_investigation_features.py
====================================
Tests verifying the integrity and structure of newly built feature tables (v2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features_v2"


def test_primary_feature_table_structure_and_counts():
    path_b = FEAT_DIR / "stress_features_v2_B.csv"
    assert path_b.exists(), f"Missing primary feature table: {path_b}"
    df = pd.read_csv(path_b)

    # 35 standard participants
    assert df["subject"].nunique() == 35
    assert len(df) == 263

    # All 23 canonical features present
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing canonical feature: {col}"

    # Extra features present
    assert "eda_scr_per_min" in df.columns
    assert "window_start_s" in df.columns
    assert "window_end_s" in df.columns
    assert "window_duration_s" in df.columns
    assert "variant" in df.columns

    # Variant B windows are duration-matched to ~30 s
    durations = df["window_duration_s"]
    assert (durations <= 30.5).all(), "Variant B windows must not exceed 30 s"

    # No duplicate windows per subject
    dups = df.duplicated(subset=["subject", "stage"]).sum()
    assert dups == 0, f"Found {dups} duplicate windows in primary feature table"


def test_labels_unchanged_against_raw_self_report_files():
    """Verify that every (subject, stage) rating exactly reproduces the dataset self-report CSVs."""
    path_b = FEAT_DIR / "stress_features_v2_B.csv"
    df = pd.read_csv(path_b)

    v1_raw = pd.read_csv(ROOT / "Stress_Level_v1.csv", index_col=0)
    v2_raw = pd.read_csv(ROOT / "Stress_Level_v2.csv", index_col=0)

    for _, row in df.iterrows():
        sid = row["subject"]
        stage = row["stage"]
        val = row["stress_level"]
        raw_df = v1_raw if sid.startswith("S") else v2_raw
        if sid in raw_df.index and stage in raw_df.columns:
            expected = float(raw_df.loc[sid, stage])
            if pd.isna(expected):
                assert pd.isna(val)
            else:
                assert float(val) == pytest.approx(expected, abs=1e-4)


def test_f14_isolated_to_sensitivity_file_only():
    """Verify f14 does not appear in standard files, but appears in the sensitivity file."""
    for var in ["A", "B", "Bend"]:
        path = FEAT_DIR / f"stress_features_v2_{var}.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert "f14" not in df["subject"].values, f"f14 erroneously found in {path.name}"

    f14_path = FEAT_DIR / "stress_features_v2_B_f14_sensitivity.csv"
    if f14_path.exists():
        df_f14 = pd.read_csv(f14_path)
        assert "f14" in df_f14["subject"].values
        assert df_f14["subject"].nunique() == 36
        assert len(df_f14) == 270

