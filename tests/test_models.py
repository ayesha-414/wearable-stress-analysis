"""
tests/test_models.py
=====================
Unit tests for src/models.py — classification pipeline,
LOSO cross-validation, metrics, and data preparation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import (
    FEATURE_COLS,
    STRESS_THRESHOLD,
    _build_model,
    _metrics,
    loso_cross_validate,
    majority_class_baseline,
    prepare_classification_data,
    feature_importance,
    compare_classifiers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n_subjects=6, stages_per_subject=4, seed=42):
    """Synthesise a small feature DataFrame for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_subjects):
        sid = f"T{i:02d}"
        for j in range(stages_per_subject):
            row = {c: rng.randn() for c in FEATURE_COLS}
            row["subject"] = sid
            row["activity"] = "STRESS"
            row["stage"] = f"stage_{j}"
            # Alternate high / low stress to guarantee both classes
            row["stress_level"] = 8.0 if (i + j) % 2 == 0 else 2.0
            rows.append(row)
    return pd.DataFrame(rows)


class TestPrepareClassificationData:
    def test_returns_correct_shapes(self):
        df = _make_feature_df()
        X, y, groups = prepare_classification_data(df)
        assert X.shape[0] == y.shape[0] == groups.shape[0]
        assert X.shape[1] > 0

    def test_binary_labels(self):
        df = _make_feature_df()
        _, y, _ = prepare_classification_data(df)
        assert set(y.unique()).issubset({0, 1})

    def test_threshold_applied_correctly(self):
        df = _make_feature_df()
        _, y, _ = prepare_classification_data(df, stress_threshold=STRESS_THRESHOLD)
        # stress_level 8.0 >= 5.0 → 1,  stress_level 2.0 < 5.0 → 0
        assert (y == 1).sum() > 0
        assert (y == 0).sum() > 0

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_classification_data(df)

    def test_raises_on_empty_valid_rows(self):
        df = _make_feature_df()
        df["stress_level"] = np.nan
        with pytest.raises(ValueError, match="No rows with valid stress"):
            prepare_classification_data(df)

    def test_legacy_alias_handled(self):
        """The CSV may contain hr_rmssd_approx instead of hr_diff_rms."""
        df = _make_feature_df()
        if "hr_diff_rms" in df.columns:
            df = df.rename(columns={"hr_diff_rms": "hr_rmssd_approx"})
        X, _, _ = prepare_classification_data(df)
        assert "hr_diff_rms" in X.columns

    def test_index_alignment(self):
        df = _make_feature_df()
        X, y, groups = prepare_classification_data(df)
        assert list(X.index) == list(y.index) == list(groups.index)


class TestBuildModel:
    @pytest.mark.parametrize("name", ["rf", "svm", "logreg", "extratrees"])
    def test_builds_pipeline(self, name):
        pipe = _build_model(name)
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            _build_model("unknown_model")


class TestMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        m = _metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = _metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_empty_returns_nans(self):
        m = _metrics([], [])
        assert all(np.isnan(v) for v in m.values())


class TestLOSOCrossValidate:
    def test_runs_without_error(self):
        df = _make_feature_df(n_subjects=4, stages_per_subject=4)
        X, y, groups = prepare_classification_data(df)
        result = loso_cross_validate(X, y, groups, classifier="rf")
        assert "overall_accuracy" in result
        assert "fold_scores" in result

    def test_fold_count_matches_subjects(self):
        df = _make_feature_df(n_subjects=5, stages_per_subject=4)
        X, y, groups = prepare_classification_data(df)
        result = loso_cross_validate(X, y, groups, classifier="rf")
        # Some folds may be skipped if single-class training
        assert result["n_folds_valid"] <= groups.nunique()

    def test_predictions_binary(self):
        df = _make_feature_df(n_subjects=4, stages_per_subject=6)
        X, y, groups = prepare_classification_data(df)
        result = loso_cross_validate(X, y, groups, classifier="logreg")
        cm = result["confusion_matrix"]
        assert cm.shape == (2, 2)


class TestMajorityBaseline:
    def test_runs_without_error(self):
        df = _make_feature_df(n_subjects=4, stages_per_subject=4)
        _, y, groups = prepare_classification_data(df)
        result = majority_class_baseline(y, groups)
        assert "overall_accuracy" in result

    def test_never_predicts_positive_class_when_negative_majority(self):
        # Create heavily imbalanced data (all low stress)
        y = pd.Series([0] * 20)
        groups = pd.Series([f"S{i}" for i in range(4)] * 5)
        result = majority_class_baseline(y, groups)
        # All predictions should be 0
        assert result["high_stress_recall"] == 0.0


class TestFeatureImportance:
    def test_returns_series(self):
        df = _make_feature_df()
        X, y, _ = prepare_classification_data(df)
        imp = feature_importance(X, y)
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]

    def test_values_sum_to_one(self):
        df = _make_feature_df()
        X, y, _ = prepare_classification_data(df)
        imp = feature_importance(X, y)
        assert imp.sum() == pytest.approx(1.0, abs=0.01)


class TestCompareClassifiers:
    def test_returns_dataframe(self):
        df = _make_feature_df(n_subjects=4, stages_per_subject=6)
        X, y, groups = prepare_classification_data(df)
        result = compare_classifiers(X, y, groups)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # RF, SVM, LogReg, ExtraTrees, Majority
