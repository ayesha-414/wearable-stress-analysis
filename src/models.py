"""
src/models.py
=============
Machine-learning pipeline for wearable stress classification.

Approach
--------
- Binary classification: high stress (≥ 5) vs low stress (< 5)
- Leave-One-Subject-Out (LOSO) cross-validation to avoid data leakage
- Classifiers: Random Forest, SVM (RBF kernel)
- Evaluation: accuracy, F1 (macro), confusion matrix, per-subject scores

LOSO is critical because multiple windows from the same subject appear in the
dataset. Standard k-fold would allow subject-level data leakage.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

STRESS_THRESHOLD = 5.0   # ≥5 → high stress; <5 → low stress

FEATURE_COLS = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_rmssd_approx",
    "eda_mean", "eda_std", "eda_slope", "eda_range", "eda_scl_mean", "eda_scr_count",
    "temp_mean", "temp_std", "temp_slope",
    "acc_mean", "acc_std", "acc_max",
    "hrv_mean_ibi", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf_ratio",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_classification_data(
    features_df: pd.DataFrame,
    stress_threshold: float = STRESS_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Filter, binarize labels, and return X, y, groups (subject IDs).

    Returns
    -------
    X : feature DataFrame
    y : binary Series (0=low, 1=high stress)
    groups : Series of subject IDs for LOSO splitting
    """
    df = features_df.copy()
    df = df[df["stress_level"].notna()].copy()

    # Keep only feature columns present in data
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()

    y = (df["stress_level"] >= stress_threshold).astype(int)
    y.name = "label"

    groups = df["subject"]

    return X, y, groups


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def loso_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    classifier: str = "rf",
    n_estimators: int = 100,
) -> dict:
    """
    Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    X         : feature matrix
    y         : binary labels
    groups    : subject ID per row
    classifier: 'rf' (Random Forest) or 'svm'

    Returns
    -------
    dict with per-fold scores, confusion matrix, classification report
    """
    unique_subjects = groups.unique()
    all_preds, all_true = [], []
    fold_scores = []

    for subject in unique_subjects:
        test_mask = groups == subject
        train_mask = ~test_mask

        X_train = X[train_mask].fillna(X[train_mask].median())
        X_test = X[test_mask].fillna(X[train_mask].median())
        y_train = y[train_mask]
        y_test = y[test_mask]

        if y_train.nunique() < 2:
            logger.debug("Skipping fold %s: only one class in training", subject)
            continue
        if len(X_test) == 0:
            continue

        if classifier == "rf":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    class_weight="balanced",
                )),
            ])
        else:  # svm
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                            class_weight="balanced", random_state=42)),
            ])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_acc = accuracy_score(y_test, preds)
        fold_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        fold_scores.append({
            "subject": subject,
            "n_test": len(y_test),
            "accuracy": fold_acc,
            "f1_macro": fold_f1,
        })

        all_preds.extend(preds)
        all_true.extend(y_test)

    overall_acc = accuracy_score(all_true, all_preds) if all_true else float("nan")
    overall_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0) if all_true else float("nan")
    cm = confusion_matrix(all_true, all_preds) if all_true else np.array([[]])
    report = classification_report(all_true, all_preds,
                                   target_names=["Low Stress", "High Stress"],
                                   zero_division=0) if all_true else ""

    return {
        "classifier":    classifier.upper(),
        "n_subjects":    len(unique_subjects),
        "n_folds_valid": len(fold_scores),
        "overall_accuracy": overall_acc,
        "overall_f1_macro": overall_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "fold_scores": pd.DataFrame(fold_scores),
    }


def compare_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Run LOSO for both RF and SVM and return comparison DataFrame."""
    results = []
    for clf_name in ["rf", "svm"]:
        logger.info("Running LOSO for %s ...", clf_name.upper())
        res = loso_cross_validate(X, y, groups, classifier=clf_name)
        results.append({
            "Classifier":  res["classifier"],
            "N Subjects":  res["n_subjects"],
            "Valid Folds": res["n_folds_valid"],
            "Accuracy":    round(res["overall_accuracy"], 4),
            "F1 Macro":    round(res["overall_f1_macro"], 4),
        })
    return pd.DataFrame(results)


def feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
) -> pd.Series:
    """
    Train a single RF on all data and return feature importances.
    (Used for visualization only — not for evaluation.)
    """
    X_clean = X.fillna(X.median())
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                 class_weight="balanced")
    clf.fit(X_clean, y)
    return pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
