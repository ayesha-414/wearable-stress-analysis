"""
tests/test_investigation_leakage.py
===================================
Tests verifying absence of data leakage in the evaluation framework.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import prepare_classification_data

_loso_mod = importlib.import_module("scripts.14_loso_benchmark")
evaluate_loso_model = _loso_mod.evaluate_loso_model

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_B = ROOT / "data" / "features_v2" / "stress_features_v2_B.csv"


@pytest.fixture(scope="module")
def primary_dataset():
    if not PRIMARY_B.exists():
        pytest.skip("stress_features_v2_B.csv not generated yet")
    df = pd.read_csv(PRIMARY_B)
    X, y, groups = prepare_classification_data(df)
    meta = df.loc[X.index, [c for c in ("subject", "stage", "cohort") if c in df.columns]].copy()
    return X, y, groups, meta


def test_fold_uniqueness_each_participant_held_out_exactly_once(primary_dataset):
    """Verify that every participant appears in the predictions exactly as many times as their windows."""
    X, y, groups, meta = primary_dataset
    cfg = {"type": "logreg_fixed"}
    _, df_preds = evaluate_loso_model("TestLogReg", X, y, groups, meta, cfg)

    unique_held_out = df_preds["subject"].unique()
    assert len(unique_held_out) == groups.nunique()
    assert set(unique_held_out) == set(groups.unique())

    # Window count per subject in predictions matches ground truth
    for s in unique_held_out:
        assert (df_preds["subject"] == s).sum() == (groups == s).sum()


def test_leakage_flipping_held_out_participant_labels_cannot_change_their_estimate(primary_dataset):
    """
    Leakage test: Flipping the ground truth labels of a held-out test participant
    MUST NOT change their predicted probabilities or predicted classes.
    """
    X, y, groups, meta = primary_dataset
    cfg = {"type": "logreg_fixed"}

    target_sub = sorted(groups.unique())[0]

    # Baseline evaluation
    _, df_p1 = evaluate_loso_model("LogReg", X, y, groups, meta, cfg)
    p1 = df_p1[df_p1["subject"] == target_sub]["y_prob"].values

    # Invert target participant's labels
    y_flipped = y.copy()
    mask = (groups == target_sub)
    y_flipped[mask] = 1 - y_flipped[mask]

    _, df_p2 = evaluate_loso_model("LogReg", X, y_flipped, groups, meta, cfg)
    p2 = df_p2[df_p2["subject"] == target_sub]["y_prob"].values

    np.testing.assert_allclose(p1, p2, atol=1e-7, err_msg="Leakage detected: flipping test labels altered predictions!")


def test_baseline_window_excluded_from_test_set_when_normalising(primary_dataset):
    """
    Verify test-exclusive baseline normalisation:
    A participant's baseline window is used strictly for normalisation and is
    EXCLUDED from that participant's evaluation test set.
    """
    X, y, groups, meta = primary_dataset
    cfg = {"type": "logreg_fixed", "baseline_relative": True}
    _, df_preds = evaluate_loso_model("TestNorm", X, y, groups, meta, cfg)

    # In df_preds, NO window should have stage == 'Baseline'
    assert "Baseline" not in df_preds["stage"].values
    assert len(df_preds) == len(X) - groups.nunique()  # exactly 35 baseline windows dropped
