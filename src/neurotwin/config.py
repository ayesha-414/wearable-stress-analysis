"""Central, documented constants. Change values here, not in the modules."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "stress_features.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROFILE_DB = PROJECT_ROOT / "data" / "neurotwin_profile.db"      # created only on request
ESTIMATOR_REPORT = PROCESSED_DIR / "neurotwin_estimator_report.json"   # NEW file, nothing overwritten
SIMULATION_REPORT = PROCESSED_DIR / "neurotwin_simulation_report.json" # NEW file
DATA_DIR = Path(__file__).resolve().parent / "data"
CATALOGUE_JSON = DATA_DIR / "catalogue.json"
CRISIS_RESOURCES_JSON = DATA_DIR / "crisis_resources.json"

# --- estimator -----------------------------------------------------------
COVERAGE_MIN = 0.70            # fraction of the 23 features that must be measured
MARGIN_MODERATE = 0.10         # |p - 0.5| needed for "moderate" evidence
AUC_MIN_FOR_MODERATE = 0.65    # held-out (LOSO) AUC needed before ANY window may be called "moderate".
                               # A conservative project convention, declared in advance; not a published standard.
CAL_ECE_MAX = 0.10             # probability shown only if ECE <= this AND Brier beats prior
CAL_INNER_FOLDS = 5
MOVEMENT_PERCENTILE = 95.0     # acc_mean / acc_max above this percentile -> movement flag
# Held back from plain-language explanations until the LF/HF scaling discrepancy
# documented in feature_formula_validation.md is resolved. The model itself is unchanged.
SUPPRESSED_EXPLANATION_FEATURES = frozenset({"hrv_lf_hf_ratio"})

# --- patterns (only report with repeated observations) ---------------------
MIN_PATTERN_REPEATS = 3
MIN_PATTERN_DISTINCT_DAYS = 2
MIN_GROUP_FOR_COMPARISON = 5
MIN_HELP_OBSERVATIONS = 3

# --- recommender -----------------------------------------------------------
MAX_SUGGESTIONS = 3
PREF_BOUND = 3.0
PREF_DELTAS = {"yes": 1.0, "somewhat": 0.3, "no": -1.0, "skipped": 0.0}
FUTURE_YES_BONUS = 0.5
REPEAT_PENALTY = 0.4           # per recent repeat of the same item

# --- prompt policy (opt-in; conservative defaults) --------------------------
PROMPT_MIN_INTERVAL_MIN = 120
PROMPT_DAILY_CAP = 3
QUIET_HOURS = (22, 7)          # 22:00 -> 07:00
