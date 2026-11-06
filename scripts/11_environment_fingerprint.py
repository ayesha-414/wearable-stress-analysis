"""Print an environment/code fingerprint so results can be compared across machines.

Read-only. Writes nothing. Usage (PowerShell, project root):
    python scripts\\11_environment_fingerprint.py
"""
import hashlib
import platform
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else "MISSING"


def main() -> None:
    print("python      :", sys.version.split()[0], platform.platform())
    for mod in ("numpy", "pandas", "scipy", "sklearn", "streamlit", "plotly", "pytest"):
        try:
            print(f"{mod:12s}:", __import__(mod).__version__)
        except Exception as exc:  # noqa: BLE001
            print(f"{mod:12s}: not importable ({exc.__class__.__name__})")
    print("\nfile fingerprints (md5 of LF-normalised bytes):")
    for rel in ("data/features/stress_features.csv", "src/preprocessing.py", "src/features.py", "src/models.py",
                "src/neurotwin/estimator.py", "data/processed/model_comparison_scores.csv",
                "data/processed/loso_results.csv"):
        print(f"  {md5(ROOT / rel)}  {rel}")
    from src.neurotwin.estimator import StressEstimator
    rep = StressEstimator().calibration_report()
    print("\nestimator audit recomputed on THIS machine:")
    for k in ("n_windows", "n_subjects", "pooled_accuracy", "majority_baseline_accuracy", "pooled_auc",
              "brier_raw", "brier_recalibrated", "brier_training_prior", "ece_raw", "ece_recalibrated",
              "calibration_ok"):
        print(f"  {k:28s} {rep[k]}")


if __name__ == "__main__":
    main()
