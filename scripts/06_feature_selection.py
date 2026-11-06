"""
scripts/06_feature_selection.py
================================
Step 6: LOSO-aware feature selection experiment.

Compares the full 23-feature model against a reduced subset selected
within each training fold.  Selection uses mutual-information scoring
per fold, and the final subset is determined by selection frequency
(features chosen in ≥50% of folds).

Outputs
-------
data/processed/feature_selection_frequency.csv
data/processed/feature_selection_comparison.csv
reports/figures/06_selection_frequency.png
reports/figures/06_selection_comparison.png
logs/06_feature_selection.log
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import (
    FEATURE_COLS,
    _build_model,
    _metrics,
    loso_cross_validate,
    majority_class_baseline,
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
        logging.FileHandler(LOG_DIR / "06_feature_selection.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def mi_feature_ranking(X_train: pd.DataFrame, y_train: pd.Series,
                       n_select: int = 10) -> list[str]:
    """Rank features by mutual information and return the top n_select names."""
    imp = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
    mi = mutual_info_classif(X_clean, y_train, random_state=42, n_neighbors=5)
    ranked = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    return ranked.head(n_select).index.tolist()


def loso_with_selection(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                        classifier: str = "rf", n_select: int = 10):
    """LOSO CV where feature selection is performed inside each fold.

    Returns
    -------
    dict with overall metrics, fold scores, and per-fold selected features.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    valid = y.notna() & groups.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    groups = groups.loc[valid].reset_index(drop=True)

    fold_scores = []
    all_preds = []
    all_true = []
    selection_log = []  # list of dicts: {subject, selected_features}

    for subject in groups.unique():
        test_mask = groups == subject
        train_mask = ~test_mask

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        if y_train.empty or y_test.empty:
            continue
        if y_train.nunique() < 2 or len(y_test) == 0:
            continue

        # --- Feature selection on training fold only ---
        selected = mi_feature_ranking(X_train, y_train, n_select=n_select)
        selection_log.append({"subject": subject, "selected": selected})

        model = _build_model(classifier)
        model.fit(X_train[selected], y_train)
        preds = model.predict(X_test[selected])

        scores = _metrics(y_test, preds)
        fold_scores.append({"subject": subject, "n_test": len(y_test), **scores})
        all_true.extend(y_test.tolist())
        all_preds.extend(preds.tolist())

    overall = _metrics(all_true, all_preds) if all_true else {n: float("nan") for n in _metrics([], []).keys()}

    return {
        "classifier": classifier.upper(),
        "n_select": n_select,
        "n_folds_valid": len(fold_scores),
        "n_samples_evaluated": len(all_true),
        **overall,
        "fold_scores": pd.DataFrame(fold_scores),
        "selection_log": selection_log,
    }


def compute_selection_frequency(selection_log: list[dict],
                                all_features: list[str]) -> pd.Series:
    """Fraction of folds in which each feature was selected."""
    counts = pd.Series(0, index=all_features, dtype=float)
    for entry in selection_log:
        for feat in entry["selected"]:
            if feat in counts.index:
                counts[feat] += 1
    return (counts / len(selection_log)).sort_values(ascending=False)


def main():
    logger.info("=" * 60)
    logger.info("STEP 6 — LOSO Feature Selection Experiment")
    logger.info("=" * 60)

    feat_path = ROOT / "data" / "features" / "stress_features.csv"
    if not feat_path.exists():
        logger.error("Features file not found: %s", feat_path)
        sys.exit(1)

    feat_df = pd.read_csv(feat_path)
    X, y, groups = prepare_classification_data(feat_df)
    logger.info("Samples: %d | Features: %d | Subjects: %d",
                len(y), X.shape[1], groups.nunique())

    n_select = 10  # top-10 by mutual information per fold

    # ── Full model baseline (all 23 features) ───────────────────────
    logger.info("Running LOSO with all %d features (RF)...", X.shape[1])
    full_res = loso_cross_validate(X, y, groups, classifier="rf")
    logger.info("Full-feature RF: Acc=%.4f  F1=%.4f  BalAcc=%.4f",
                full_res["overall_accuracy"],
                full_res["overall_f1_macro"],
                full_res["overall_balanced_accuracy"])

    # ── Feature-selected model ──────────────────────────────────────
    logger.info("Running LOSO with MI-selected top-%d features (RF)...", n_select)
    sel_res = loso_with_selection(X, y, groups, classifier="rf", n_select=n_select)
    logger.info("Selected-feature RF: Acc=%.4f  F1=%.4f  BalAcc=%.4f",
                sel_res["accuracy"],
                sel_res["f1_macro"],
                sel_res["balanced_accuracy"])

    # Also run with LogReg (best full-feature performer)
    logger.info("Running LOSO with all features (LogReg)...")
    full_lr = loso_cross_validate(X, y, groups, classifier="logreg")
    logger.info("Full-feature LogReg: Acc=%.4f  F1=%.4f  BalAcc=%.4f",
                full_lr["overall_accuracy"],
                full_lr["overall_f1_macro"],
                full_lr["overall_balanced_accuracy"])

    logger.info("Running LOSO with MI-selected top-%d features (LogReg)...", n_select)
    sel_lr = loso_with_selection(X, y, groups, classifier="logreg", n_select=n_select)
    logger.info("Selected-feature LogReg: Acc=%.4f  F1=%.4f  BalAcc=%.4f",
                sel_lr["accuracy"],
                sel_lr["f1_macro"],
                sel_lr["balanced_accuracy"])

    # ── Selection frequency ─────────────────────────────────────────
    freq = compute_selection_frequency(sel_res["selection_log"], list(X.columns))
    freq.to_csv(DATA_DIR / "feature_selection_frequency.csv", header=["frequency"])
    logger.info("Selection frequency (top 10):\n%s", freq.head(10).to_string())

    # ── Comparison table ────────────────────────────────────────────
    comparison = pd.DataFrame([
        {"Model": "RF (23 features)", "Accuracy": full_res["overall_accuracy"],
         "Macro F1": full_res["overall_f1_macro"],
         "Balanced Accuracy": full_res["overall_balanced_accuracy"]},
        {"Model": f"RF (top-{n_select} MI)", "Accuracy": sel_res["accuracy"],
         "Macro F1": sel_res["f1_macro"],
         "Balanced Accuracy": sel_res["balanced_accuracy"]},
        {"Model": "LogReg (23 features)", "Accuracy": full_lr["overall_accuracy"],
         "Macro F1": full_lr["overall_f1_macro"],
         "Balanced Accuracy": full_lr["overall_balanced_accuracy"]},
        {"Model": f"LogReg (top-{n_select} MI)", "Accuracy": sel_lr["accuracy"],
         "Macro F1": sel_lr["f1_macro"],
         "Balanced Accuracy": sel_lr["balanced_accuracy"]},
    ])
    comparison.to_csv(DATA_DIR / "feature_selection_comparison.csv", index=False)
    logger.info("Comparison:\n%s", comparison.to_string(index=False))

    # ── Plots ───────────────────────────────────────────────────────
    # Selection frequency bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(freq))
    ax.barh(range(len(freq)), freq.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(freq)))
    ax.set_yticklabels(freq.index[::-1], fontsize=9)
    ax.set_xlabel("Selection Frequency (fraction of folds)")
    ax.set_title(f"Feature Selection Frequency — MI top-{n_select} per LOSO fold",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_selection_frequency.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", FIG_DIR / "06_selection_frequency.png")

    # Comparison grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_to_plot = ["Accuracy", "Macro F1", "Balanced Accuracy"]
    x = np.arange(len(comparison))
    width = 0.25
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, comparison[metric], width, label=metric, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(comparison["Model"], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Full vs Selected Features — LOSO Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_selection_comparison.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", FIG_DIR / "06_selection_comparison.png")

    logger.info("STEP 6 COMPLETE.")
    return comparison


if __name__ == "__main__":
    main()
