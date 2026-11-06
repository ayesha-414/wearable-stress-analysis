"""
tests/test_loader.py
=====================
Unit tests for src/loader.py
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import (
    PROJECT_ROOT,
    DATASET_DIR,
    ACTIVITIES,
    SPLIT_SESSIONS,
    get_subject_ids,
    load_stress_levels,
    load_subject_info,
    _read_empatica_csv,
)


class TestProjectPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists(), f"Project root not found: {PROJECT_ROOT}"

    def test_dataset_dir_exists(self):
        assert DATASET_DIR.exists(), f"Dataset dir not found: {DATASET_DIR}"

    def test_activity_folders_exist(self):
        for act in ACTIVITIES:
            assert (DATASET_DIR / act).exists(), f"Missing: {act}"


class TestSubjectIds:
    def test_stress_subjects_loaded(self):
        ids = get_subject_ids()
        assert len(ids["STRESS"]) > 0, "No STRESS subjects found"

    def test_split_parts_excluded(self):
        ids = get_subject_ids()
        # f14_a and f14_b should not appear as top-level subjects
        assert "f14_a" not in ids.get("STRESS", [])
        assert "f14_b" not in ids.get("STRESS", [])
        # f14 should appear (merged)
        assert "f14" in ids.get("STRESS", [])

    def test_s12_aerobic_missing(self):
        """S12 did not perform aerobic protocol."""
        ids = get_subject_ids()
        assert "S12" not in ids.get("AEROBIC", []), "S12 should have no aerobic data"

    def test_s11_aerobic_split(self):
        """S11 aerobic is split; split parts should not appear in IDs."""
        ids = get_subject_ids()
        assert "S11_a" not in ids.get("AEROBIC", [])
        assert "S11" in ids.get("AEROBIC", [])

    def test_s16_anaerobic_split(self):
        ids = get_subject_ids()
        assert "S16_a" not in ids.get("ANAEROBIC", [])
        assert "S16" in ids.get("ANAEROBIC", [])


class TestMetadataLoaders:
    def test_stress_v1_shape(self):
        v1, _ = load_stress_levels()
        assert v1.shape[0] == 18, f"Expected 18 subjects in V1, got {v1.shape[0]}"
        assert v1.shape[1] >= 7, "Expected at least 7 protocol stages in V1"

    def test_stress_v2_shape(self):
        _, v2 = load_stress_levels()
        assert v2.shape[0] == 18, f"Expected 18 subjects in V2, got {v2.shape[0]}"
        assert v2.shape[1] >= 6

    def test_stress_values_in_range(self):
        v1, v2 = load_stress_levels()
        assert v1.min().min() >= 0, "Stress below 0"
        assert v1.max().max() <= 10, "Stress above 10"
        assert v2.min().min() >= 0
        assert v2.max().max() <= 10

    def test_subject_info_has_s_cohort(self):
        df = load_subject_info()
        s_subjects = [i for i in df.index if str(i).startswith("S")]
        assert len(s_subjects) > 0

    def test_subject_info_has_f_cohort(self):
        df = load_subject_info()
        f_subjects = [i for i in df.index if str(i).startswith("f")]
        assert len(f_subjects) > 0


class TestSignalReader:
    def test_eda_readable(self):
        path = DATASET_DIR / "STRESS" / "S01" / "EDA.csv"
        df = _read_empatica_csv(path, "EDA")
        assert df is not None, "EDA.csv could not be read"
        assert "EDA" in df.columns
        assert "utc_ts" in df.columns
        assert len(df) > 100

    def test_hr_readable(self):
        path = DATASET_DIR / "STRESS" / "S01" / "HR.csv"
        df = _read_empatica_csv(path, "HR")
        assert df is not None
        assert "HR" in df.columns

    def test_ibi_readable(self):
        path = DATASET_DIR / "STRESS" / "S01" / "IBI.csv"
        df = _read_empatica_csv(path, "IBI")
        assert df is not None
        assert "ibi_s" in df.columns

    def test_tags_readable(self):
        path = DATASET_DIR / "STRESS" / "S01" / "tags.csv"
        df = _read_empatica_csv(path, "tags")
        assert df is not None
        assert "time_s" in df.columns

    def test_acc_has_xyz(self):
        path = DATASET_DIR / "STRESS" / "S01" / "ACC.csv"
        df = _read_empatica_csv(path, "ACC")
        assert df is not None
        for col in ["x", "y", "z"]:
            assert col in df.columns

    def test_nonexistent_file_returns_none(self):
        path = DATASET_DIR / "STRESS" / "S999" / "EDA.csv"
        df = _read_empatica_csv(path, "EDA")
        assert df is None
