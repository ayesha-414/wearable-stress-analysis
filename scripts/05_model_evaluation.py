"""
scripts/05_model_evaluation.py
================================
Step 5: LOSO classification of high vs low stress.

- Load precomputed features from data/features/stress_features.csv
- Binary label: high stress (≥5) vs low stress (<5)
- LOSO cross-validation for Random Forest and SVM
- Save results, confusion matrices, feature importance

Outputs
-------
reports/figures/05_confusion_matrix_rf.png
reports/figures/05_confusion_matrix_svm.png
reports/figures/05_feature_importance.png
reports/figures/05_loso_fold_scores.png
reports/figures/05_classifier_comparison.png
data/processed/loso_results.csv
data/processed/feature_importance.csv
logs/05_model_evaluation.log
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import (
    compare_classifiers,
    feature_importance,
    loso_cross_validate,
    prepare_classification_data,
)

LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "reports" / "figures"
DATA_DIR = ROOT / "data" / "processed"
for d in [LOG_DIR, FIG_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "05_model_evaluation.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def plot_confusion_matrix(cm: np.ndarray, clf_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Stress", "High Stress"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {clf_name} (LOSO)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / f"05_confusion_matrix_{clf_name.lower()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_feature_importance(importances: pd.Series):
    top_n = min(20, len(importances))
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", top_n)
    bars = ax.barh(range(top_n), top.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top Feature Importances — Random Forest", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "05_feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_fold_scores(rf_res: dict, svm_res: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("LOSO Per-Subject Fold Scores", fontsize=13, fontweight="bold")

    for ax, res, clf_name in [(axes[0], rf_res, "Random Forest"), (axes[1], svm_res, "SVM")]:
        df = res["fold_scores"]
        if df.empty:
            ax.text(0.5, 0.5, "No results", transform=ax.transAxes, ha="center")
            continue
        x = range(len(df))
        ax.bar(x, df["accuracy"], color="#4C72B0", alpha=0.7, label="Accuracy")
        ax.plot(x, df["f1_macro"], color="#C44E52", marker="o", linewidth=1.5,
                label="F1 Macro")
        ax.axhline(res["overall_accuracy"], color="#4C72B0", linestyle="--",
                   linewidth=1.5, label=f"Mean Acc={res['overall_accuracy']:.2f}")
        ax.axhline(res["overall_f1_macro"], color="#C44E52", linestyle="--",
                   linewidth=1.5, label=f"Mean F1={res['overall_f1_macro']:.2f}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["subject"], rotation=45, ha="right", fontsize=7)
        ax.set_title(clf_name)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "05_loso_fold_scores.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_classifier_comparison(comparison_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Classifier Comparison (LOSO)", fontsize=13, fontweight="bold")

    for ax, metric, color in [
        (axes[0], "Accuracy", "#4C72B0"),
        (axes[1], "F1 Macro", "#C44E52"),
    ]:
        bars = ax.bar(comparison_df["Classifier"], comparison_df[metric],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(metric)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric)

    plt.tight_layout()
    path = FIG_DIR / "05_classifier_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def main():
    logger.info("=" * 60)
    logger.info("STEP 5 — Model Evaluation (LOSO)")
    logger.info("=" * 60)

    feat_path = ROOT / "data" / "features" / "stress_features.csv"
    if not feat_path.exists():
        logger.error("Features file not found: %s — run script 03 first", feat_path)
        sys.exit(1)

    feat_df = pd.read_csv(feat_path)
    logger.info("Loaded features: %s rows × %s columns", *feat_df.shape)

    X, y, groups = prepare_classification_data(feat_df)
    logger.info("Samples: %d | Low stress: %d | High stress: %d",
                len(y), (y == 0).sum(), (y == 1).sum())
    logger.info("Unique subjects: %d", groups.nunique())

    if len(y) < 10:
        logger.error("Insufficient labeled samples for evaluation. Aborting.")
        sys.exit(1)

    # --- Random Forest ---
    logger.info("Running LOSO — Random Forest...")
    rf_res = loso_cross_validate(X, y, groups, classifier="rf")
    logger.info("RF Accuracy: %.4f | F1 Macro: %.4f",
                rf_res["overall_accuracy"], rf_res["overall_f1_macro"])
    logger.info("\nClassification Report:\n%s", rf_res["classification_report"])

    # --- SVM ---
    logger.info("Running LOSO — SVM...")
    svm_res = loso_cross_validate(X, y, groups, classifier="svm")
    logger.info("SVM Accuracy: %.4f | F1 Macro: %.4f",
                svm_res["overall_accuracy"], svm_res["overall_f1_macro"])
    logger.info("\nClassification Report:\n%s", svm_res["classification_report"])

    # --- Comparison ---
    comparison_df = compare_classifiers(X, y, groups)
    logger.info("Classifier Comparison:\n%s", comparison_df.to_string(index=False))
    comparison_df.to_csv(DATA_DIR / "loso_results.csv", index=False)

    # --- Feature Importance ---
    logger.info("Computing feature importances...")
    try:
        importances = feature_importance(X, y)
        importances.to_csv(DATA_DIR / "feature_importance.csv", header=["importance"])
        logger.info("Top 5 features:\n%s", importances.head().to_string())
        plot_feature_importance(importances)
    except Exception as e:
        logger.warning("Feature importance failed: %s", e)

    # --- Plots ---
    logger.info("Generating plots...")
    plot_confusion_matrix(rf_res["confusion_matrix"], "RF")
    plot_confusion_matrix(svm_res["confusion_matrix"], "SVM")
    plot_fold_scores(rf_res, svm_res)
    plot_classifier_comparison(comparison_df)

    logger.info("STEP 5 COMPLETE.")
    return comparison_df


if __name__ == "__main__":
    main()
