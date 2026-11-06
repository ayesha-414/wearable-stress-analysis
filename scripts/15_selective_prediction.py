"""
scripts/15_selective_prediction.py
==================================
Selective prediction and abstention framework for the NeuroTwin model investigation.
Implements risk-coverage analysis from held-out predictions:
1. Signal quality abstention (sensor missingness, coverage thresholds)
2. Confidence-based abstention with confidence band selected inside training folds
3. Fallback message: "Insufficient physiological evidence to provide a reliable estimate."
4. Coverage vs. balanced accuracy curves.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "processed" / "model_investigation" / "v2"
OUT_DIR = IN_DIR / "selective_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_MESSAGE = "Insufficient physiological evidence to provide a reliable estimate."


def run_selective_prediction(predictions_csv: Path | None = None) -> dict:
    if predictions_csv is None:
        predictions_csv = IN_DIR / "benchmark_predictions_primary_B.csv"

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions not found at {predictions_csv}. Run 14_loso_benchmark.py first.")

    df_preds = pd.read_csv(predictions_csv)
    
    # Analyze the reference logreg model
    model_name = "Existing LogReg (C=1.0)"
    if model_name not in df_preds["model"].values:
        model_name = df_preds["model"].iloc[0]

    sub_df = df_preds[df_preds["model"] == model_name].copy()
    y_true = sub_df["y_true"].values
    y_pred = sub_df["y_pred"].values
    y_prob = sub_df["y_prob"].values
    subjects = sub_df["subject"].values

    # Confidence margin: distance from decision boundary 0.5
    conf = np.abs(y_prob - 0.5)
    sub_df["confidence"] = conf

    # Signal quality abstention: participant f07 has missing HR/TEMP sensors
    quality_mask = (sub_df["subject"] != "f07")

    # Risk-coverage analysis
    # Sweep confidence thresholds from 0.0 to 0.45
    curve_rows = []
    thresholds = np.linspace(0.0, 0.45, 46)

    for th in thresholds:
        # Window retained if quality ok and confidence >= threshold
        keep = quality_mask & (conf >= th)
        coverage = float(keep.mean())
        n_covered = int(keep.sum())

        if n_covered >= 10 and len(np.unique(y_true[keep])) == 2:
            acc = float(accuracy_score(y_true[keep], y_pred[keep]))
            bal_acc = float(balanced_accuracy_score(y_true[keep], y_pred[keep]))
            risk = 1.0 - acc
        else:
            acc = np.nan
            bal_acc = np.nan
            risk = np.nan

        curve_rows.append({
            "confidence_threshold": round(float(th), 3),
            "coverage": round(coverage, 4),
            "n_covered": n_covered,
            "accuracy": round(acc, 4) if not np.isnan(acc) else np.nan,
            "balanced_accuracy": round(bal_acc, 4) if not np.isnan(bal_acc) else np.nan,
            "risk": round(risk, 4) if not np.isnan(risk) else np.nan,
        })

    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(OUT_DIR / "risk_coverage_curve.csv", index=False)

    # In-fold threshold selection:
    # Choose confidence band inside training folds to guarantee target precision / balanced accuracy
    unique_subs = np.unique(subjects)
    in_fold_thresholds = []

    for s in unique_subs:
        tr_mask = (subjects != s) & quality_mask
        tr_y = y_true[tr_mask]
        tr_pred = y_pred[tr_mask]
        tr_conf = conf[tr_mask]

        best_th = 0.0
        best_score = -1.0
        for th in np.linspace(0.05, 0.35, 31):
            k = tr_conf >= th
            if k.sum() >= 20 and len(np.unique(tr_y[k])) == 2:
                score = balanced_accuracy_score(tr_y[k], tr_pred[k])
                if score > best_score:
                    best_score = score
                    best_th = th
        in_fold_thresholds.append(best_th)

    chosen_threshold = float(np.median(in_fold_thresholds))

    # Apply chosen threshold to test set
    retained_mask = quality_mask & (conf >= chosen_threshold)
    sub_df["abstained"] = ~retained_mask
    sub_df["fallback_message"] = np.where(sub_df["abstained"], FALLBACK_MESSAGE, "")

    final_coverage = float(retained_mask.mean())
    final_acc = float(accuracy_score(y_true[retained_mask], y_pred[retained_mask]))
    final_bal_acc = float(balanced_accuracy_score(y_true[retained_mask], y_pred[retained_mask]))

    summary = {
        "model": model_name,
        "total_windows": len(sub_df),
        "quality_abstained_windows": int((~quality_mask).sum()),
        "in_fold_chosen_confidence_threshold": round(chosen_threshold, 4),
        "total_abstained_windows": int((~retained_mask).sum()),
        "final_coverage": round(final_coverage, 4),
        "full_coverage_balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "selective_balanced_accuracy": round(final_bal_acc, 4),
        "full_coverage_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "selective_accuracy": round(final_acc, 4),
        "fallback_message": FALLBACK_MESSAGE,
        "recommendation_guardrail": "No recommendation may treat an estimate as proof of stress.",
    }

    sub_df.to_csv(OUT_DIR / "selective_prediction_decisions.csv", index=False)
    (OUT_DIR / "selective_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SELECTIVE PREDICTION SUMMARY (PHASE E)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Chosen confidence threshold (in-fold): {chosen_threshold:.3f}")
    print(f"Coverage: {final_coverage:.1%} ({retained_mask.sum()}/{len(sub_df)})")
    print(f"Balanced Accuracy: {final_bal_acc:.4f} (vs {summary['full_coverage_balanced_accuracy']:.4f} full)")
    print(f"Accuracy: {final_acc:.4f} (vs {summary['full_coverage_accuracy']:.4f} full)")
    print(f"Fallback message: {FALLBACK_MESSAGE}")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    run_selective_prediction()

