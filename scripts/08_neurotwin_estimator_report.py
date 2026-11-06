"""Write the NeuroTwin estimator audit (NEW file; no existing result is overwritten).

Usage:  python scripts/08_neurotwin_estimator_report.py
Output: data/processed/neurotwin_estimator_report.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neurotwin.estimator import StressEstimator  # noqa: E402


def main() -> None:
    est = StressEstimator()
    path = est.write_report()
    rep = est.calibration_report()
    print(f"Wrote {path}")
    for k in ("n_windows", "n_subjects", "pooled_accuracy", "majority_baseline_accuracy", "pooled_auc",
              "brier_recalibrated", "brier_training_prior", "ece_recalibrated", "calibration_ok"):
        print(f"  {k}: {rep[k]}")


if __name__ == "__main__":
    main()
