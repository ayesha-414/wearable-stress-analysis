"""
scripts/07_factor_analysis.py
================================
Step 7: Factor analysis feasibility assessment.

Evaluates whether the 23 physiological features satisfy the statistical
prerequisites for factor analysis (KMO, Bartlett's).  If satisfied,
performs an exploratory PCA/FA and reports eigenvalues and loadings.
If not satisfied, documents why and reports a PCA decomposition as
the alternative.

Outputs
-------
data/processed/factor_analysis_report.csv
data/processed/pca_variance_explained.csv
data/processed/pca_loadings.csv
reports/figures/07_scree_plot.png
reports/figures/07_pca_loadings_heatmap.png
logs/07_factor_analysis.log
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import prepare_classification_data

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
        logging.FileHandler(LOG_DIR / "07_factor_analysis.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def kmo_sampling_adequacy(corr_matrix: np.ndarray) -> float:
    """Compute Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy.

    KMO = sum(r_ij^2) / [sum(r_ij^2) + sum(q_ij^2)]

    where r_ij are correlation matrix entries and q_ij are partial correlations.
    KMO >= 0.6 is generally considered acceptable for FA.
    """
    try:
        inv_corr = np.linalg.pinv(corr_matrix)
        # Partial correlation matrix
        d = np.diag(1.0 / np.sqrt(np.diag(inv_corr) + 1e-12))
        partial_corr = -d @ inv_corr @ d
        np.fill_diagonal(partial_corr, 0)
        np.fill_diagonal(corr_matrix, 0)

        sum_r2 = np.sum(corr_matrix ** 2)
        sum_q2 = np.sum(partial_corr ** 2)

        kmo = sum_r2 / (sum_r2 + sum_q2 + 1e-12)
        return float(kmo)
    except Exception as exc:
        logger.warning("KMO calculation failed: %s", exc)
        return float("nan")


def bartlett_sphericity_test(X: np.ndarray) -> tuple[float, float]:
    """Bartlett's test of sphericity.

    Tests whether the correlation matrix is an identity matrix.
    Returns (chi2, p_value).
    """
    from scipy.stats import chi2

    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    det = np.linalg.det(corr)
    if det <= 0:
        det = 1e-300  # avoid log(0)
    chi2_stat = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1.0 - chi2.cdf(chi2_stat, df)
    return float(chi2_stat), float(p_value)


def main():
    logger.info("=" * 60)
    logger.info("STEP 7 — Factor Analysis Feasibility Assessment")
    logger.info("=" * 60)

    feat_path = ROOT / "data" / "features" / "stress_features.csv"
    if not feat_path.exists():
        logger.error("Features file not found: %s", feat_path)
        sys.exit(1)

    feat_df = pd.read_csv(feat_path)
    X, y, groups = prepare_classification_data(feat_df)
    logger.info("Samples: %d | Features: %d", X.shape[0], X.shape[1])

    # ── Imputation & standardisation ────────────────────────────────
    imp = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X.columns)

    # ── Missing-data report ─────────────────────────────────────────
    nan_pct = X.isnull().mean() * 100
    logger.info("Missing data per feature:\n%s",
                nan_pct[nan_pct > 0].sort_values(ascending=False).to_string())

    # ── Correlation matrix ──────────────────────────────────────────
    corr = X_scaled.corr().values

    # ── KMO ─────────────────────────────────────────────────────────
    kmo = kmo_sampling_adequacy(corr.copy())
    logger.info("KMO sampling adequacy: %.4f", kmo)
    kmo_adequate = kmo >= 0.6

    # ── Bartlett's test ─────────────────────────────────────────────
    chi2_stat, p_val = bartlett_sphericity_test(X_scaled.values)
    logger.info("Bartlett's test: chi2=%.2f, p=%.2e", chi2_stat, p_val)
    bartlett_sig = p_val < 0.05

    # ── Decision ────────────────────────────────────────────────────
    fa_justified = kmo_adequate and bartlett_sig
    logger.info("Factor analysis justified: %s (KMO≥0.6: %s, Bartlett p<0.05: %s)",
                fa_justified, kmo_adequate, bartlett_sig)

    report_rows = [
        {"Test": "KMO Sampling Adequacy", "Value": f"{kmo:.4f}",
         "Threshold": "≥ 0.6", "Pass": str(kmo_adequate)},
        {"Test": "Bartlett's Sphericity χ²", "Value": f"{chi2_stat:.2f}",
         "Threshold": "p < 0.05", "Pass": str(bartlett_sig)},
        {"Test": "Bartlett's p-value", "Value": f"{p_val:.2e}",
         "Threshold": "< 0.05", "Pass": str(bartlett_sig)},
        {"Test": "Sample size (N)", "Value": str(X.shape[0]),
         "Threshold": "≥ 5× features = 115", "Pass": str(X.shape[0] >= 5 * X.shape[1])},
        {"Test": "N features", "Value": str(X.shape[1]),
         "Threshold": "—", "Pass": "—"},
        {"Test": "N/p ratio", "Value": f"{X.shape[0] / X.shape[1]:.1f}",
         "Threshold": "≥ 5–10", "Pass": str(X.shape[0] / X.shape[1] >= 5)},
        {"Test": "FA Justified", "Value": str(fa_justified),
         "Threshold": "Both KMO and Bartlett pass", "Pass": str(fa_justified)},
    ]
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(DATA_DIR / "factor_analysis_report.csv", index=False)
    logger.info("Report:\n%s", report_df.to_string(index=False))

    # ── PCA regardless (as alternative/exploratory) ─────────────────
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    var_explained = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Eigenvalue": pca.explained_variance_,
        "Variance_Explained": pca.explained_variance_ratio_,
        "Cumulative": np.cumsum(pca.explained_variance_ratio_),
    })
    var_explained.to_csv(DATA_DIR / "pca_variance_explained.csv", index=False)
    logger.info("PCA variance explained:\n%s", var_explained.head(10).to_string(index=False))

    # Kaiser criterion: components with eigenvalue > 1
    n_kaiser = int((pca.explained_variance_ > 1.0).sum())
    logger.info("Components with eigenvalue > 1 (Kaiser): %d", n_kaiser)

    # Report how many components to explain 80% variance
    cum80 = int((np.cumsum(pca.explained_variance_ratio_) < 0.80).sum()) + 1
    logger.info("Components for 80%% variance: %d", cum80)

    # Loadings for top n_kaiser components
    loadings = pd.DataFrame(
        pca.components_[:n_kaiser].T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_kaiser)],
    )
    loadings.to_csv(DATA_DIR / "pca_loadings.csv")
    logger.info("PCA loadings (top %d components):\n%s", n_kaiser,
                loadings.round(3).to_string())

    # ── Scree plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    n_comp = len(pca.explained_variance_)
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
           color="#4C72B0", alpha=0.7, label="Individual")
    ax.plot(range(1, n_comp + 1), np.cumsum(pca.explained_variance_ratio_) * 100,
            "o-", color="#C44E52", linewidth=2, label="Cumulative")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% threshold")
    ax.axvline(n_kaiser + 0.5, color="green", linestyle="--", linewidth=1,
               label=f"Kaiser cutoff (n={n_kaiser})")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Scree Plot — PCA on 23 Physiological Features", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_scree_plot.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", FIG_DIR / "07_scree_plot.png")

    # ── Loadings heatmap ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Loading"})
    ax.set_title(f"PCA Loadings — Top {n_kaiser} Components (Kaiser)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_pca_loadings_heatmap.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", FIG_DIR / "07_pca_loadings_heatmap.png")

    if not fa_justified:
        logger.info(
            "CONCLUSION: Factor analysis is NOT justified. "
            "KMO=%.4f %s the 0.6 threshold; Bartlett p=%.2e %s 0.05. "
            "PCA dimensionality reduction is provided as an alternative.",
            kmo, "meets" if kmo_adequate else "does not meet",
            p_val, "is below" if bartlett_sig else "is above",
        )
    else:
        logger.info(
            "CONCLUSION: Factor analysis prerequisites are met. "
            "KMO=%.4f, Bartlett p=%.2e. "
            "Consider running confirmatory FA using the %d Kaiser-criterion components as a starting point.",
            kmo, p_val, n_kaiser,
        )

    logger.info("STEP 7 COMPLETE.")
    return report_df


if __name__ == "__main__":
    main()
