
"""Leakage-aware subject-independent stress classification using LOSO CV."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

STRESS_THRESHOLD = 5.0

# Use one canonical HR difference feature name.
# Older CSV files may contain hr_rmssd_approx instead.
FEATURE_COLS = [
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_range",
    "hr_diff_rms",
    "eda_mean",
    "eda_std",
    "eda_slope",
    "eda_range",
    "eda_scl_mean",
    "eda_scr_count",
    "temp_mean",
    "temp_std",
    "temp_slope",
    "acc_mean",
    "acc_std",
    "acc_max",
    "hrv_mean_ibi",
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    "hrv_lf_hf_ratio",
]

METRIC_NAMES = [
    "accuracy",
    "f1_macro",
    "balanced_accuracy",
    "high_stress_precision",
    "high_stress_recall",
]


def prepare_classification_data(
    features_df: pd.DataFrame,
    stress_threshold: float = STRESS_THRESHOLD,
):
    """
    Return numeric physiological features, binary labels, and subject groups.

    Label 0: stress_level below threshold.
    Label 1: stress_level at or above threshold.

    Rows without a stress label or subject ID are excluded.
    """
    required = {"stress_level", "subject"}
    missing = required - set(features_df.columns)

    if missing:
        raise ValueError(
            f"Feature table is missing required columns: {sorted(missing)}"
        )

    df = features_df.loc[
        features_df["stress_level"].notna()
        & features_df["subject"].notna()
    ].copy()

    if df.empty:
        raise ValueError("No rows with valid stress labels and subject IDs.")

    # Prefer the newer feature if available; otherwise use the legacy name.
    available = [
        col for col in FEATURE_COLS
        if col in df.columns
    ]

    if "hr_diff_rms" in df.columns:
        available = [
            col for col in available if col != "hr_rmssd_approx"
        ]
    elif "hr_rmssd_approx" in df.columns:
        available = [
            "hr_diff_rms" if col == "hr_rmssd_approx" else col
            for col in available
        ]

    # Resolve source column names separately from canonical output names.
    source_columns = []
    output_columns = []

    for col in FEATURE_COLS:
        if col == "hr_diff_rms":
            if "hr_diff_rms" in df.columns:
                source_columns.append("hr_diff_rms")
                output_columns.append("hr_diff_rms")
            elif "hr_rmssd_approx" in df.columns:
                source_columns.append("hr_rmssd_approx")
                output_columns.append("hr_diff_rms")
        elif col == "hr_rmssd_approx":
            # Do not include the legacy alias a second time.
            continue
        elif col in df.columns:
            source_columns.append(col)
            output_columns.append(col)

    if not source_columns:
        raise ValueError(
            "No recognised physiological feature columns were found."
        )

    X = df[source_columns].copy()
    X.columns = output_columns
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns that contain no usable values.
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("All selected physiological features are missing.")

    stress_values = pd.to_numeric(df["stress_level"], errors="coerce")

    valid = stress_values.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = (
        stress_values.loc[valid] >= stress_threshold
    ).astype(int).reset_index(drop=True)
    groups = df.loc[valid, "subject"].reset_index(drop=True)

    # Avoid accidental index misalignment between X, y, and groups.
    return X, y.rename("label"), groups.rename("subject")


def _build_model(
    classifier: str,
    n_estimators: int = 100,
) -> Pipeline:
    """Build a pipeline so imputation and scaling are fitted per training fold."""
    name = classifier.lower().strip()

    if name in {"rf", "random forest", "random_forest"}:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
        )
        needs_scaling = False

    elif name in {"svm", "rbf-svm", "rbf_svm"}:
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )
        needs_scaling = True

    elif name in {
        "logistic regression",
        "logistic_regression",
        "lr",
        "logreg",
    }:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        needs_scaling = True

    elif name in {"extra trees", "extra_trees", "et", "extratrees"}:
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
        )
        needs_scaling = False

    else:
        raise ValueError(
            f"Unknown classifier {classifier!r}. Choose "
            "'rf', 'svm', 'logistic_regression', or 'extra_trees'."
        )

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if needs_scaling:
        steps.append(("scaler", StandardScaler()))

    steps.append(("clf", clf))
    return Pipeline(steps)


def _metrics(y_true, y_pred) -> dict:
    """Calculate metrics with high stress represented by label 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {name: float("nan") for name in METRIC_NAMES}

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "high_stress_precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "high_stress_recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


def _empty_metrics() -> dict:
    return {name: float("nan") for name in METRIC_NAMES}


def _summarize_predictions(
    classifier: str,
    groups: pd.Series,
    fold_scores: list,
    all_true: list,
    all_preds: list,
) -> dict:
    """Return aggregate metrics and the existing LOSO result structure."""
    overall = _metrics(all_true, all_preds) if all_true else _empty_metrics()

    cm = (
        confusion_matrix(all_true, all_preds, labels=[0, 1])
        if all_true
        else np.zeros((2, 2), dtype=int)
    )

    report = (
        classification_report(
            all_true,
            all_preds,
            labels=[0, 1],
            target_names=["Low Stress", "High Stress"],
            zero_division=0,
        )
        if all_true
        else ""
    )

    return {
        "classifier": classifier.upper(),
        "n_subjects": int(groups.nunique()),
        "n_folds_valid": len(fold_scores),
        "n_samples_evaluated": len(all_true),
        "overall_accuracy": overall["accuracy"],
        "overall_f1_macro": overall["f1_macro"],
        "overall_balanced_accuracy": overall["balanced_accuracy"],
        "high_stress_precision": overall["high_stress_precision"],
        "high_stress_recall": overall["high_stress_recall"],
        "confusion_matrix": cm,
        "classification_report": report,
        "fold_scores": pd.DataFrame(fold_scores),
    }


def loso_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    classifier: str = "rf",
    n_estimators: int = 100,
) -> dict:
    """
    Evaluate a classifier using Leave-One-Subject-Out cross-validation.

    Each test subject is excluded from model training. Imputation, scaling,
    and classifier fitting occur using training subjects only.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must have equal row counts.")

    valid = y.notna() & groups.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    groups = groups.loc[valid].reset_index(drop=True)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 (low) and 1 (high).")

    fold_scores = []
    all_preds = []
    all_true = []

    unique_subjects = groups.unique()

    for subject in unique_subjects:
        test_mask = groups == subject
        train_mask = ~test_mask

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        if y_train.empty or y_test.empty:
            continue



        if y_train.nunique() < 2 or len(y_test) == 0:
            logger.warning(
                "Skipping subject %s: empty test fold or training "
                "fold has fewer than two classes.",
                subject,
            )
            continue

        model = _build_model(classifier, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        scores = _metrics(y_test, preds)
        fold_scores.append({
            "subject": subject,
            "n_test": len(y_test),
            **scores,
        })

        all_true.extend(y_test.tolist())
        all_preds.extend(preds.tolist())

    return _summarize_predictions(
        classifier, groups, fold_scores, all_true, all_preds
    )


def majority_class_baseline(
    y: pd.Series,
    groups: pd.Series,
) -> dict:
    """
    Leakage-free LOSO baseline.

    Predict the majority label in the training subjects for each fold.
    The held-out subject's labels are used only for evaluation, never to
    choose the prediction.
    """
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    if len(y) != len(groups):
        raise ValueError("y and groups must have equal row counts.")

    valid = y.notna() & groups.notna()
    y = y.loc[valid].astype(int).reset_index(drop=True)
    groups = groups.loc[valid].reset_index(drop=True)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 (low) and 1 (high).")

    fold_scores = []
    all_preds = []
    all_true = []

    for subject in groups.unique():
        test_mask = groups == subject
        train_mask = ~test_mask

        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        if y_train.empty or y_test.empty:
            continue

        # Crucial: majority class is calculated from training labels only.
        majority = int(y_train.value_counts().idxmax())
        preds = np.full(len(y_test), majority, dtype=int)

        fold_scores.append({
            "subject": subject,
            "n_test": len(y_test),
            **_metrics(y_test, preds),
        })

        all_true.extend(y_test.tolist())
        all_preds.extend(preds.tolist())

    return _summarize_predictions(
        "MAJORITY BASELINE",
        groups,
        fold_scores,
        all_true,
        all_preds,
    )


def compare_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """
    Run LOSO for RF, SVM, Logistic Regression, Extra Trees and Majority baseline.
    Returns a DataFrame with overall metrics for each classifier.
    """
    results = []

    for clf_name in ["rf", "svm", "logreg", "extratrees", "baseline"]:
        logger.info("Running LOSO for %s ...", clf_name.upper())
        if clf_name == "baseline":
            res = majority_class_baseline(y, groups)
            # Ensure consistent key name for classifier
            res["classifier"] = "MAJORITY"
        else:
            res = loso_cross_validate(X, y, groups, classifier=clf_name)

        results.append({
            "Classifier": res["classifier"],
            "N Subjects": res["n_subjects"],
            "Valid Folds": res["n_folds_valid"],
            "Samples Evaluated": res["n_samples_evaluated"],
            "Accuracy": res["overall_accuracy"],
            "Macro F1": res["overall_f1_macro"],
            "Balanced Accuracy": res["overall_balanced_accuracy"],
            "High-Stress Precision": res["high_stress_precision"],
            "High-Stress Recall": res["high_stress_recall"],
        })

    return pd.DataFrame(results)


def feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
) -> pd.Series:
    """
    Fit a descriptive Random Forest on all supplied data.

    This is descriptive only; do not use it as an unbiased estimate of
    held-out model performance or as test-fold feature selection.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError("X and y must have equal row counts.")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_clean, y)

    return pd.Series(
        clf.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)
