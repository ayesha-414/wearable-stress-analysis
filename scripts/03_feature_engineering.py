"""
scripts/03_feature_engineering.py
==================================
Step 3: Extract physiological features for all STRESS subjects
        and save to data/features/stress_features.csv.

Outputs
-------
data/features/stress_features.csv     — full feature table with stress labels
reports/figures/03_feature_distributions.png
reports/figures/03_feature_correlation_matrix.png
reports/figures/03_hrv_stress_boxplot.png
reports/figures/03_eda_stress_boxplot.png
logs/03_feature_engineering.log
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_all, load_stress_levels
from src.features import (
    FEATURE_COLS,
    extract_all_features,
)

LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "reports" / "figures"
FEAT_DIR = ROOT / "data" / "features"
for d in [LOG_DIR, FIG_DIR, FEAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "03_feature_engineering.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def plot_feature_distributions(df: pd.DataFrame):
    """Histograms for every feature column."""
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    n = len(available_feats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
    axes = axes.flatten()
    fig.suptitle("Feature Distributions (STRESS protocol)", fontsize=14, fontweight="bold")

    for i, feat in enumerate(available_feats):
        vals = df[feat].dropna()
        axes[i].hist(vals, bins=25, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[i].set_title(feat, fontsize=9)
        axes[i].set_ylabel("Count", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = FIG_DIR / "03_feature_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation heatmap of all features."""
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    corr = df[available_feats].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0,
        ax=ax, annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "03_feature_correlation_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_features_by_stress(df: pd.DataFrame, features: list[str], title_prefix: str, filename: str):
    """Box plots comparing feature distributions between low and high stress."""
    df2 = df.copy()
    df2["stress_label"] = df2["stress_level"].apply(
        lambda x: "High (≥5)" if x >= 5 else "Low (<5)" if pd.notna(x) else None
    )
    df2 = df2[df2["stress_label"].notna()].copy()

    available = [f for f in features if f in df2.columns]
    n = len(available)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    fig.suptitle(f"{title_prefix} — Low vs High Stress", fontsize=13, fontweight="bold")

    palette = {"Low (<5)": "#4C72B0", "High (≥5)": "#C44E52"}
    for i, feat in enumerate(available):
        ax = axes[i]
        groups = [df2[df2["stress_label"] == lab][feat].dropna().values
                  for lab in ["Low (<5)", "High (≥5)"]]
        ax.boxplot(groups, tick_labels=["Low (<5)", "High (>=5)"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel(feat)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def main():
    logger.info("=" * 60)
    logger.info("STEP 3 — Feature Engineering")
    logger.info("=" * 60)

    logger.info("Loading data...")
    data = load_all(activities=["STRESS"])
    sl_v1, sl_v2 = load_stress_levels()

    logger.info("Extracting features for all STRESS subjects...")
    feat_df = extract_all_features(data, sl_v1, sl_v2, activity="STRESS")

    logger.info("Feature table shape: %s", feat_df.shape)
    logger.info("Columns: %s", list(feat_df.columns))

    n_with_label = feat_df["stress_level"].notna().sum()
    logger.info("Rows with stress label: %d / %d", n_with_label, len(feat_df))

    # Save
    out = FEAT_DIR / "stress_features.csv"
    feat_df.to_csv(out, index=False)
    logger.info("Saved features: %s", out)

    # Summary stats
    available_feats = [c for c in FEATURE_COLS if c in feat_df.columns]
    logger.info("Feature completeness:\n%s",
                feat_df[available_feats].notna().mean().round(3).to_string())

    logger.info("Plotting feature distributions...")
    plot_feature_distributions(feat_df)

    logger.info("Plotting correlation matrix...")
    plot_correlation_matrix(feat_df)

    hrv_feats = ["hrv_mean_ibi", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf_ratio"]
    eda_feats = ["eda_mean", "eda_std", "eda_slope", "eda_scl_mean", "eda_scr_count"]

    logger.info("Plotting HRV features by stress level...")
    plot_features_by_stress(feat_df, hrv_feats, "HRV Features", "03_hrv_stress_boxplot.png")

    logger.info("Plotting EDA features by stress level...")
    plot_features_by_stress(feat_df, eda_feats, "EDA Features", "03_eda_stress_boxplot.png")

    logger.info("STEP 3 COMPLETE.")
    return feat_df


if __name__ == "__main__":
    main()
