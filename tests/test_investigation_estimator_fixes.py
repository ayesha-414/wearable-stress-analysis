"""
tests/test_investigation_estimator_fixes.py
============================================
Unit tests verifying the A6 integration fixes in src/neurotwin/estimator.py:
(a) movement/temperature percentiles from training participants only
(b) row alignment between X and its metadata does not assume continuous index
(c) evidence ceiling from whole-sample audit and no probabilities shown when Brier skill <= 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neurotwin import config
from src.neurotwin.estimator import StressEstimator


def test_percentiles_computed_from_training_participants_only():
    """Verify that a subject's outlier values do not affect their own training bounds."""
    df = pd.read_csv(config.FEATURES_CSV)
    subj = sorted(df["subject"].unique())[0]

    est_clean = StressEstimator(df.copy())
    clean_stats = est_clean._stats_for_subject(subj)

    # Corrupt subject's temp_mean with an extreme value
    df_corrupted = df.copy()
    mask = df_corrupted["subject"] == subj
    df_corrupted.loc[mask, "temp_mean"] = 999.0
    est_corrupted = StressEstimator(df_corrupted)

    corrupted_stats_for_subj = est_corrupted._stats_for_subject(subj)

    # For the held-out subject, training stats MUST NOT be influenced by their own outlier
    assert corrupted_stats_for_subj["temp_hi"] == pytest.approx(clean_stats["temp_hi"], rel=1e-5)
    assert corrupted_stats_for_subj["temp_hi"] < 100.0


def test_metadata_alignment_with_filtered_or_non_continuous_index():
    """Verify that self.meta preserves row alignment with self.X even if raw features have gaps."""
    df = pd.read_csv(config.FEATURES_CSV).copy()
    # Artificially set some rows to NaN stress_level and drop some indices
    df.loc[5:10, "stress_level"] = np.nan
    df_shuffled = df.sample(frac=1.0, random_state=42)  # non-sequential index

    est = StressEstimator(df_shuffled)
    assert len(est.X) == len(est.meta)
    assert len(est.y) == len(est.meta)
    assert len(est.groups) == len(est.meta)
    # Check that subjects match exactly between groups and meta
    pd.testing.assert_series_equal(est.groups, est.meta["subject"].reset_index(drop=True), check_names=False)


def test_probabilities_withheld_when_calibration_not_demonstrated():
    """Verify that probability_shown is strictly None when calibration does not beat prior."""
    est = StressEstimator(pd.read_csv(config.FEATURES_CSV))
    cal = est.calibration_report()
    assert cal["calibration_ok"] is False

    for idx in range(min(10, len(est.X))):
        res = est.estimate_for_window(idx)
        assert res.probability_shown is None
        assert res.evidence_level in ("limited", "insufficient")

