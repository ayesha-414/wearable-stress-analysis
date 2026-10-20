"""
scripts/02_eda.py
=================
Step 2: Exploratory Data Analysis — distributions, correlations,
        signal quality report, and self-reported stress visualizations.

Outputs
-------
reports/figures/02_stress_level_v1_all.png
reports/figures/02_stress_level_v2_all.png
reports/figures/02_stress_mean_by_stage.png
reports/figures/02_subject_demographics.png
reports/figures/02_signal_quality_heatmap.png
reports/figures/02_subject_signal_sample_S01.png
data/processed/signal_quality_report.csv
logs/02_eda.log
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_all, load_stress_levels, load_subject_info, ACTIVITIES

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "02_eda.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PALETTE = sns.color_palette("husl", 20)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_stress_levels(sl_v1: pd.DataFrame, sl_v2: pd.DataFrame):
    """Line plots of self-reported stress for all subjects."""
    for version, df in [("v1 (S01–S18)", sl_v1), ("v2 (f01–f18)", sl_v2)]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Self-Reported Stress Levels — {version}", fontsize=14, fontweight="bold")

        ax1 = axes[0]
        colors = sns.color_palette("tab20", len(df))
        for i, (idx, row) in enumerate(df.iterrows()):
            ax1.plot(df.columns, row.values, marker="o", label=idx, color=colors[i], linewidth=1.5)
        ax1.set_title("Individual Subjects")
        ax1.set_xlabel("Protocol Stage")
        ax1.set_ylabel("Stress Level (0–10)")
        ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)
        ax1.tick_params(axis="x", rotation=30)
        ax1.set_ylim(0, 11)

        ax2 = axes[1]
        mean_vals = df.mean()
        std_vals = df.std()
        ax2.bar(df.columns, mean_vals, color=sns.color_palette("mako", len(df.columns)),
                yerr=std_vals, capsize=4, alpha=0.85)
        ax2.set_title("Mean ± SD Across Subjects")
        ax2.set_xlabel("Protocol Stage")
        ax2.set_ylabel("Mean Stress Level")
        ax2.tick_params(axis="x", rotation=30)
        ax2.set_ylim(0, 11)

        plt.tight_layout()
        tag = "v1" if "v1" in version else "v2"
        path = FIG_DIR / f"02_stress_level_{tag}_all.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)


def plot_mean_stress_by_stage(sl_v1: pd.DataFrame, sl_v2: pd.DataFrame):
    """Combined view: mean stress per stage, both protocol versions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Mean Self-Reported Stress Level per Stage", fontsize=14, fontweight="bold")

    for ax, df, title in [(axes[0], sl_v1, "Protocol V1 (Male, S cohort)"),
                           (axes[1], sl_v2, "Protocol V2 (Female, f cohort)")]:
        mean_vals = df.mean()
        std_vals = df.std()
        bars = ax.bar(range(len(mean_vals)), mean_vals.values,
                      color=sns.color_palette("viridis", len(mean_vals)),
                      yerr=std_vals.values, capsize=5, alpha=0.9, edgecolor="white")
        ax.set_xticks(range(len(mean_vals)))
        ax.set_xticklabels(mean_vals.index, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Mean Stress Level")
        ax.set_ylim(0, 11)
        for bar, val in zip(bars, mean_vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "02_stress_mean_by_stage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_demographics(subj_info: pd.DataFrame):
    """Demographics bar/scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Subject Demographics", fontsize=14, fontweight="bold")

    # Age distribution
    age_data = pd.to_numeric(subj_info["Age"], errors="coerce").dropna()
    axes[0].hist(age_data, bins=10, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")

    # Gender split
    gender_counts = subj_info["Gender"].value_counts()
    axes[1].bar(gender_counts.index, gender_counts.values,
                color=["#55A868", "#C44E52"], edgecolor="white", alpha=0.9)
    axes[1].set_title("Gender Distribution")
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Count")

    # Physical activity regularity
    phys = subj_info["Does physical activity regularly?"].value_counts()
    axes[2].pie(phys.values, labels=phys.index, autopct="%1.0f%%",
                colors=["#4C72B0", "#DD8452"], startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[2].set_title("Regular Physical Activity")

    plt.tight_layout()
    path = FIG_DIR / "02_subject_demographics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_signal_quality_heatmap(summary_csv: Path):
    """Heatmap showing number of data rows per subject-signal combination."""
    df = pd.read_csv(summary_csv)
    stress_df = df[df["activity"] == "STRESS"].copy()
    stress_df = stress_df.set_index("subject")

    sig_cols = [c for c in stress_df.columns if c.endswith("_rows")]
    heat = stress_df[sig_cols].fillna(0)
    heat.columns = [c.replace("_rows", "") for c in heat.columns]

    # Normalize to 0–1 per signal for display
    heat_norm = heat.div(heat.max(axis=0)).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heat_norm, cmap="YlOrRd", ax=ax, linewidths=0.5,
                linecolor="white", cbar_kws={"label": "Relative Data Availability"})
    ax.set_title("Signal Data Availability per Subject (STRESS protocol)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Subject")
    plt.tight_layout()
    path = FIG_DIR / "02_signal_quality_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_sample_signals(data: dict):
    """Plot raw signals for one example subject (S01, STRESS)."""
    subject_id = "S01"
    signals = data.get("STRESS", {}).get(subject_id)
    if signals is None:
        logger.warning("S01/STRESS not found — skipping sample signal plot")
        return

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=False)
    fig.suptitle(f"Raw Signals — Subject {subject_id} (STRESS Protocol)", fontsize=14, fontweight="bold")

    signal_specs = [
        ("HR",   "Heart Rate (bpm)",         "#E64646"),
        ("EDA",  "EDA (µS)",                  "#2196F3"),
        ("TEMP", "Skin Temperature (°C)",     "#FF9800"),
        ("BVP",  "Blood Volume Pulse",        "#9C27B0"),
        ("ACC",  "Accelerometer (x,y,z)",     "#4CAF50"),
    ]

    for ax, (sig, ylabel, color) in zip(axes, signal_specs):
        df = signals.get(sig)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            ax.text(0.5, 0.5, f"{sig} not available", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            ax.set_ylabel(ylabel)
            continue

        if sig == "ACC":
            t = df["utc_ts"] - df["utc_ts"].iloc[0]
            ax.plot(t, df["x"].values, linewidth=0.5, alpha=0.7, label="x", color="#4CAF50")
            ax.plot(t, df["y"].values, linewidth=0.5, alpha=0.7, label="y", color="#FF5722")
            ax.plot(t, df["z"].values, linewidth=0.5, alpha=0.7, label="z", color="#2196F3")
            ax.legend(loc="upper right", fontsize=8)
        else:
            col = sig
            if col not in df.columns and len(df.columns) > 1:
                col = df.columns[1]
            t = df["utc_ts"] - df["utc_ts"].iloc[0]
            ax.plot(t, df[col].values, linewidth=0.8, color=color, alpha=0.85)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (seconds)")

    plt.tight_layout()
    path = FIG_DIR / "02_subject_signal_sample_S01.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def generate_signal_quality_report(data: dict) -> pd.DataFrame:
    """Create a CSV report of signal availability per subject/activity."""
    rows = []
    for activity in ACTIVITIES:
        for subject_id, signals in data.get(activity, {}).items():
            row = {"activity": activity, "subject": subject_id}
            for sig_name, df in signals.items():
                if sig_name == "tags_s":
                    row["tags_count"] = len(df) if df else 0
                    continue
                if df is None:
                    row[f"{sig_name}_available"] = False
                    row[f"{sig_name}_rows"] = 0
                elif isinstance(df, pd.DataFrame):
                    row[f"{sig_name}_available"] = len(df) > 0
                    row[f"{sig_name}_rows"] = len(df)
                else:
                    row[f"{sig_name}_available"] = True
                    row[f"{sig_name}_rows"] = -1
            rows.append(row)
    df = pd.DataFrame(rows)
    out = DATA_DIR / "signal_quality_report.csv"
    df.to_csv(out, index=False)
    logger.info("Saved signal quality report: %s", out)
    return df


def main():
    logger.info("=" * 60)
    logger.info("STEP 2 — Exploratory Data Analysis")
    logger.info("=" * 60)

    logger.info("Loading data...")
    data = load_all()
    sl_v1, sl_v2 = load_stress_levels()
    subj_info = load_subject_info()

    logger.info("Plotting stress level charts...")
    plot_stress_levels(sl_v1, sl_v2)
    plot_mean_stress_by_stage(sl_v1, sl_v2)

    logger.info("Plotting demographics...")
    plot_demographics(subj_info)

    logger.info("Generating signal quality report...")
    generate_signal_quality_report(data)

    summary_csv = ROOT / "data" / "processed" / "dataset_summary.csv"
    if summary_csv.exists():
        logger.info("Plotting signal quality heatmap...")
        plot_signal_quality_heatmap(summary_csv)
    else:
        logger.warning("dataset_summary.csv not found — run script 01 first")

    logger.info("Plotting sample signals...")
    plot_sample_signals(data)

    logger.info("STEP 2 COMPLETE.")


if __name__ == "__main__":
    main()
