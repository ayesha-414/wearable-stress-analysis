"""
scripts/04_stress_analysis.py
==============================
Step 4: Statistical stress analysis per protocol stage.

- Wilcoxon signed-rank test: baseline vs each stressed stage
- Effect size (Cohen's d) per stage
- Per-subject stress trajectory plots
- Stage-level heatmap of mean stress across subjects

Outputs
-------
reports/figures/04_stress_trajectory_v1.png
reports/figures/04_stress_trajectory_v2.png
reports/figures/04_stage_comparison_stats.png
reports/figures/04_stress_heatmap_v1.png
reports/figures/04_stress_heatmap_v2.png
data/processed/stress_stage_stats.csv
logs/04_stress_analysis.log
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_stress_levels, load_subject_info

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
        logging.FileHandler(LOG_DIR / "04_stress_analysis.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two independent groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 +
                          (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def stage_statistics(df: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    """
    Compute Wilcoxon test p-values and Cohen's d vs baseline for each stage.
    """
    rows = []
    baseline = df[baseline_col].dropna().values

    for col in df.columns:
        stage_vals = df[col].dropna().values
        if len(stage_vals) < 3:
            continue

        mean_val = float(np.mean(stage_vals))
        std_val = float(np.std(stage_vals, ddof=1))

        if col == baseline_col:
            p_val = float("nan")
            d_val = 0.0
        else:
            # Paired only if same subjects — align by index
            common = df[[baseline_col, col]].dropna()
            if len(common) >= 3:
                try:
                    stat, p_val = stats.wilcoxon(common[baseline_col].values,
                                                  common[col].values,
                                                  alternative="two-sided")
                except Exception:
                    p_val = float("nan")
            else:
                p_val = float("nan")
            d_val = cohens_d(stage_vals, baseline)

        rows.append({
            "stage":        col,
            "mean":         round(mean_val, 3),
            "std":          round(std_val, 3),
            "vs_baseline_p": round(p_val, 4) if not np.isnan(p_val) else float("nan"),
            "cohens_d":     round(d_val, 3),
            "significant":  (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    return pd.DataFrame(rows)


def plot_stress_trajectories(sl_v1: pd.DataFrame, sl_v2: pd.DataFrame):
    """Spaghetti plots of individual stress trajectories."""
    for version, df, title in [
        ("v1", sl_v1, "Stress Trajectories — Protocol V1 (S cohort, Male)"),
        ("v2", sl_v2, "Stress Trajectories — Protocol V2 (f cohort, Female)"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = sns.color_palette("tab20", len(df))

        for i, (idx, row) in enumerate(df.iterrows()):
            ax.plot(df.columns, row.values, marker="o", linewidth=1.5,
                    color=colors[i], alpha=0.75, label=idx)

        # Overlay mean
        mean_line = df.mean()
        ax.plot(df.columns, mean_line.values, color="black", linewidth=3,
                linestyle="--", label="Group Mean", zorder=10)

        ax.fill_between(
            range(len(df.columns)),
            (df.mean() - df.std()).values,
            (df.mean() + df.std()).values,
            alpha=0.12, color="black", label="+/-1 SD"
        )
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=30, ha="right")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Protocol Stage")
        ax.set_ylabel("Self-Reported Stress Level (0–10)")
        ax.set_ylim(0, 11)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)

        plt.tight_layout()
        path = FIG_DIR / f"04_stress_trajectory_{version}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)


def plot_stage_stats(stats_v1: pd.DataFrame, stats_v2: pd.DataFrame):
    """Bar chart of mean stress ± SD with significance markers."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Stage-Level Stress Statistics vs Baseline (Wilcoxon)", fontsize=13, fontweight="bold")

    for ax, stats_df, title in [(axes[0], stats_v1, "V1 (S cohort)"),
                                  (axes[1], stats_v2, "V2 (f cohort)")]:
        x = range(len(stats_df))
        colors = ["#C44E52" if sig else "#4C72B0" for sig in stats_df["significant"]]
        bars = ax.bar(x, stats_df["mean"], color=colors, alpha=0.85,
                      yerr=stats_df["std"], capsize=4, edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels(stats_df["stage"], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Mean Stress Level")
        ax.set_ylim(0, 12)
        ax.axhline(5, color="gray", linestyle=":", linewidth=1, label="Threshold (5)")

        for bar, row in zip(bars, stats_df.itertuples()):
            if row.significant:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        "*", ha="center", va="bottom", fontsize=14, color="black")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"d={row.cohens_d:.2f}", ha="center", va="bottom", fontsize=7, color="gray")

        red_patch = mpatches.Patch(color="#C44E52", alpha=0.85, label="p < 0.05")
        blue_patch = mpatches.Patch(color="#4C72B0", alpha=0.85, label="p ≥ 0.05")
        ax.legend(handles=[red_patch, blue_patch], fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "04_stage_comparison_stats.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_stress_heatmaps(sl_v1: pd.DataFrame, sl_v2: pd.DataFrame):
    """Heatmap: subject × stage stress levels."""
    for version, df in [("v1", sl_v1), ("v2", sl_v2)]:
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.45)))
        sns.heatmap(df, cmap="RdYlGn_r", ax=ax, annot=True, fmt=".1f",
                    annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                    vmin=0, vmax=10, cbar_kws={"label": "Stress Level (0–10)"})
        ax.set_title(f"Self-Reported Stress Heatmap — Protocol {version.upper()}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Protocol Stage")
        ax.set_ylabel("Subject")
        plt.tight_layout()
        path = FIG_DIR / f"04_stress_heatmap_{version}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)


def main():
    logger.info("=" * 60)
    logger.info("STEP 4 — Stress Analysis")
    logger.info("=" * 60)

    sl_v1, sl_v2 = load_stress_levels()

    logger.info("Computing stage statistics for V1...")
    stats_v1 = stage_statistics(sl_v1, "Baseline")
    logger.info("\n%s", stats_v1.to_string(index=False))

    logger.info("Computing stage statistics for V2...")
    stats_v2 = stage_statistics(sl_v2, "Baseline")
    logger.info("\n%s", stats_v2.to_string(index=False))

    # Save stats
    combined = pd.concat([
        stats_v1.assign(protocol="V1"),
        stats_v2.assign(protocol="V2"),
    ])
    out = DATA_DIR / "stress_stage_stats.csv"
    combined.to_csv(out, index=False)
    logger.info("Saved: %s", out)

    logger.info("Plotting stress trajectories...")
    plot_stress_trajectories(sl_v1, sl_v2)

    logger.info("Plotting stage statistics...")
    try:
        plot_stage_stats(stats_v1, stats_v2)
    except Exception as e:
        logger.warning("Stage stats plot failed: %s", e)

    logger.info("Plotting stress heatmaps...")
    plot_stress_heatmaps(sl_v1, sl_v2)

    logger.info("STEP 4 COMPLETE.")
    return stats_v1, stats_v2


if __name__ == "__main__":
    main()
