"""Explainable, uncertainty-aware stress-state ESTIMATOR.

This module WRAPS the project's existing model builders (``src.models._build_model``,
``prepare_classification_data``, ``FEATURE_COLS``). It does not change features,
model definitions, hyper-parameters or any saved result.

What it adds
------------
* data-quality assessment (missing features, movement / skin-temperature flags)
* abstention: no estimate is offered when evidence is insufficient
* per-window feature contributions (logistic-regression coefficient x standardised value)
* probability calibration fitted INSIDE each training fold (Platt scaling on
  out-of-fold predictions), and a calibration audit on held-out participants
* a rule deciding whether a probability may be SHOWN at all (only if calibration is
  demonstrably better than the training prior); otherwise only an ordinal evidence
  level is shown

What it does NOT do
-------------------
* It does not say why a person is stressed. Contributions describe the MODEL.
* The target is "self-reported stress >= 5 on a 1-10 scale" (a thresholded self-rating
  from a laboratory protocol), not an objective stress measure.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.models import FEATURE_COLS, STRESS_THRESHOLD, _build_model, prepare_classification_data

from . import config

FEATURE_LABELS = {
    "hr_mean": "average heart rate", "hr_std": "heart-rate variability (spread)",
    "hr_min": "lowest heart rate", "hr_max": "highest heart rate",
    "hr_range": "heart-rate range", "hr_diff_rms": "beat-to-beat heart-rate change (HR-difference RMS)",
    "eda_mean": "average skin conductance (EDA)", "eda_std": "skin-conductance variability",
    "eda_slope": "skin-conductance trend", "eda_range": "skin-conductance range",
    "eda_scl_mean": "slow skin-conductance level (SCL proxy)",
    "eda_scr_count": "number of skin-conductance responses (SCR proxy)",
    "temp_mean": "average skin temperature", "temp_std": "skin-temperature variability",
    "temp_slope": "skin-temperature trend", "acc_mean": "average movement",
    "acc_std": "movement variability", "acc_max": "peak movement",
    "hrv_mean_ibi": "average beat interval", "hrv_sdnn": "beat-interval variability (SDNN)",
    "hrv_rmssd": "short-term beat-interval variability (RMSSD)",
    "hrv_pnn50": "share of large beat-interval changes (pNN50)",
    "hrv_lf_hf_ratio": "LF/HF ratio",
}
GROUPS = {
    "heart rate": [c for c in FEATURE_COLS if c.startswith("hr_")],
    "electrodermal": [c for c in FEATURE_COLS if c.startswith("eda_")],
    "skin temperature": [c for c in FEATURE_COLS if c.startswith("temp_")],
    "movement": [c for c in FEATURE_COLS if c.startswith("acc_")],
    "beat intervals": [c for c in FEATURE_COLS if c.startswith("hrv_")],
}
LABEL_HIGHER = "a pattern associated with higher self-reported stress"
LABEL_LOWER = "a pattern associated with lower self-reported stress"
LABEL_UNCLEAR = "no clear pattern"
LABEL_NOT_ASSESSED = "not assessed (evidence insufficient)"


# ------------------------------------------------------------------- result types
@dataclass
class Quality:
    coverage: float
    missing: list[str]
    groups_missing: list[str]
    flags: list[str]
    insufficient: bool
    reasons: list[str]


@dataclass
class Contribution:
    feature: str
    label: str
    contribution: float          # log-odds contribution relative to the training average
    z: float                     # standardised value (relative to training average)
    explained: bool              # False if the feature was not measured or is held back
    note: str = ""


@dataclass
class EstimateResult:
    layer: str                   # always "estimated"
    model: str
    label: str
    evidence_level: str          # insufficient | limited | moderate  (never "high")
    abstained: bool
    probability_shown: float | None
    probability_raw: float | None
    calibrated: bool
    quality: Quality
    contributions: list[Contribution] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------- helpers
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _ece(y, p, bins=5):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(y) * abs(conf - acc)
        table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                      "mean_predicted": round(conf, 3), "observed_rate": round(acc, 3)})
    return float(ece), table


def _brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


class StressEstimator:
    """Wrapper around the existing logistic-regression pipeline (default) or any
    classifier name accepted by ``src.models._build_model``."""

    def __init__(self, features: pd.DataFrame | None = None, classifier: str = "logreg"):
        if features is None:
            features = pd.read_csv(config.FEATURES_CSV)
        self.classifier = classifier
        self.raw = features

        # Fix (b): Row alignment between X and its metadata must not assume the index is unchanged.
        # Use the filtered frame that matches the valid stress_level mask in prepare_classification_data.
        stress_values = pd.to_numeric(features.get("stress_level", pd.Series(dtype=float)), errors="coerce")
        valid = stress_values.notna() if "stress_level" in features.columns else pd.Series(True, index=features.index)
        filtered = features.loc[valid].reset_index(drop=True)

        self.X, self.y, self.groups = prepare_classification_data(features)
        self.meta = filtered[[c for c in ("subject", "stage", "cohort") if c in filtered.columns]].copy()
        self.model_name = f"{classifier} (existing pipeline, unchanged)"
        self._full = None
        self._loso_cache: dict[str, tuple] = {}
        self._stats_cache: dict[str, dict] = {}
        self._calibration: dict | None = None
        # Default stats across available dataset
        self._stats = self._compute_stats(self.X)

    @staticmethod
    def _compute_stats(df: pd.DataFrame) -> dict:
        """Compute movement percentiles and temperature bounds."""
        acc_cols = [c for c in ("acc_mean", "acc_max") if c in df.columns]
        p95 = df[acc_cols].quantile(config.MOVEMENT_PERCENTILE / 100.0) if acc_cols else pd.Series(dtype=float)
        temp_lo = float(df["temp_mean"].quantile(0.05)) if "temp_mean" in df.columns else float("-inf")
        temp_hi = float(df["temp_mean"].quantile(0.95)) if "temp_mean" in df.columns else float("inf")
        return {
            "p95": p95,
            "temp_lo": temp_lo,
            "temp_hi": temp_hi,
        }

    def _stats_for_subject(self, subject: str) -> dict:
        """Fix (a): Movement and temperature percentiles from training participants only."""
        if subject not in self._stats_cache:
            keep = self.groups != subject
            self._stats_cache[subject] = self._compute_stats(self.X[keep])
        return self._stats_cache[subject]

    # ------------------------------------------------------------ fitting
    def _fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        """Fit pipeline and a Platt calibrator on out-of-fold training predictions."""
        pipe = _build_model(self.classifier)
        n_groups = groups.nunique()
        cv = GroupKFold(n_splits=min(config.CAL_INNER_FOLDS, n_groups))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof = cross_val_predict(clone(pipe), X, y, groups=groups, cv=cv,
                                    method="predict_proba")[:, 1]
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof), y)
            pipe.fit(X, y)
        return pipe, platt

    @property
    def full_model(self):
        if self._full is None:
            self._full = self._fit(self.X, self.y, self.groups)
        return self._full

    def _model_excluding(self, subject: str):
        if subject not in self._loso_cache:
            keep = self.groups != subject
            self._loso_cache[subject] = self._fit(self.X[keep], self.y[keep], self.groups[keep])
        return self._loso_cache[subject]

    # ------------------------------------------------------------ calibration audit
    def calibration_report(self, force: bool = False) -> dict:
        """Participant-independent audit (leave-one-subject-out) of raw vs recalibrated
        probabilities against the training prior. Calibrators are fitted inside each
        training fold only."""
        if self._calibration is not None and not force:
            return self._calibration
        ys, p_raw, p_cal, p_prior, subj = [], [], [], [], []
        for s in sorted(self.groups.unique()):
            te = self.groups == s
            pipe, platt = self._model_excluding(s)
            pr = pipe.predict_proba(self.X[te])[:, 1]
            ys += self.y[te].tolist()
            p_raw += pr.tolist()
            p_cal += platt.predict_proba(_logit(pr))[:, 1].tolist()
            p_prior += [float(self.y[~te].mean())] * int(te.sum())
            subj += [s] * int(te.sum())
        ys, p_raw, p_cal, p_prior = map(np.asarray, (ys, p_raw, p_cal, p_prior))
        ece_raw, tab_raw = _ece(ys, p_raw)
        ece_cal, tab_cal = _ece(ys, p_cal)
        pred = (p_raw >= 0.5).astype(int)
        acc = float((pred == ys).mean())
        maj_acc = float(max(ys.mean(), 1 - ys.mean()))
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(ys, p_raw))
        except Exception:
            auc = None
        rep = {
            "scope": ("Leave-one-subject-out on the existing feature table; calibrator fitted "
                      "inside each training fold. Target: self-reported stress >= "
                      f"{STRESS_THRESHOLD:g}. One laboratory dataset, ages 18-30."),
            "classifier": self.classifier,
            "n_windows": int(len(ys)), "n_subjects": int(len(set(subj))),
            "prevalence_high": round(float(ys.mean()), 4),
            "pooled_accuracy": round(acc, 4), "majority_baseline_accuracy": round(maj_acc, 4),
            "pooled_auc": None if auc is None else round(auc, 4),
            "brier_raw": round(_brier(ys, p_raw), 4),
            "brier_recalibrated": round(_brier(ys, p_cal), 4),
            "brier_training_prior": round(_brier(ys, p_prior), 4),
            "ece_raw": round(ece_raw, 4), "ece_recalibrated": round(ece_cal, 4),
            "reliability_raw": tab_raw, "reliability_recalibrated": tab_cal,
        }
        rep["calibration_ok"] = bool(rep["brier_recalibrated"] < rep["brier_training_prior"]
                                     and rep["ece_recalibrated"] <= config.CAL_ECE_MAX)
        rep["probability_policy"] = (
            "Probabilities are shown only if recalibrated Brier score beats the training prior "
            f"AND ECE <= {config.CAL_ECE_MAX}. Otherwise only an ordinal evidence level is shown."
        )
        self._calibration = rep
        return rep

    def write_report(self, path=None) -> Path:
        p = Path(path or config.ESTIMATOR_REPORT)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.calibration_report(), indent=2), encoding="utf-8")
        return p

    # ------------------------------------------------------------ quality
    def assess_quality(self, row, stats: dict | None = None) -> Quality:
        s = stats or self._stats
        vals = pd.Series({c: row.get(c, np.nan) for c in FEATURE_COLS}, dtype=float)
        missing = [c for c in FEATURE_COLS if pd.isna(vals[c])]
        coverage = 1.0 - len(missing) / len(FEATURE_COLS)
        groups_missing = [g for g, cols in GROUPS.items() if all(c in missing for c in cols)]
        flags, reasons = [], []
        insufficient = False
        if coverage < config.COVERAGE_MIN:
            insufficient = True
            reasons.append(f"Only {coverage:.0%} of the signal features could be measured.")
        for crit in ("heart rate", "electrodermal"):
            if crit in groups_missing:
                insufficient = True
                reasons.append(f"No usable {crit} features in this window.")
        if "beat intervals" in groups_missing:
            flags.append("beat_interval_features_missing")
        if "movement" in groups_missing:
            flags.append("movement_not_measured")
        else:
            p95 = s["p95"]
            if (vals["acc_mean"] > p95["acc_mean"]) or (vals["acc_max"] > p95["acc_max"]):
                flags.append("movement_high")
        t = vals.get("temp_mean", np.nan)
        if not pd.isna(t) and (t < s["temp_lo"] or t > s["temp_hi"]):
            flags.append("skin_temperature_unusual")
        return Quality(round(coverage, 3), missing, groups_missing, flags, insufficient, reasons)

    # ------------------------------------------------------------ estimate
    def estimate(self, row, _model=None, stats: dict | None = None) -> EstimateResult:
        """Estimate for one window. ``row`` is a dict/Series with the 23 features.
        Uses the model fitted on ALL data unless ``_model`` is given.
        If ``stats`` is provided, movement and temperature percentiles use those values
        (e.g., from training participants only)."""
        row = dict(row)
        q = self.assess_quality(row, stats=stats)
        caveats = [
            "This is an estimate from wristband signals, not a measurement of how you feel.",
            "Movement, heat, caffeine, illness, sensor fit and many other things can change these signals.",
            "The model describes statistical patterns; it cannot say why a signal changed.",
        ]
        if "movement_high" in q.flags:
            caveats.append("Movement was higher than in most recorded windows; physical activity "
                           "alone can change heart rate and skin conductance.")
        if "skin_temperature_unusual" in q.flags:
            caveats.append("Skin temperature was unusual for this dataset (for example a warm or "
                           "cold environment or loose sensor contact).")
        if "beat_interval_features_missing" in q.flags:
            caveats.append("Beat-interval features were unavailable, so evidence is capped at 'limited'.")
        if q.insufficient:
            return EstimateResult("estimated", self.model_name, LABEL_NOT_ASSESSED, "insufficient",
                                  True, None, None, False, q, [],
                                  q.reasons + ["No stress-related statement is made."], caveats)
        pipe, platt = _model or self.full_model
        Xrow = pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_COLS}])[list(self.X.columns)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_raw = float(pipe.predict_proba(Xrow)[0, 1])
            p_cal = float(platt.predict_proba(_logit([p_raw]))[0, 1])
        cal = self.calibration_report()
        # Strictly never show probabilities unless Brier skill vs prior is positive on held-out participants
        shown = round(p_cal, 3) if cal["calibration_ok"] else None
        margin = abs(p_raw - 0.5)
        if margin < config.MARGIN_MODERATE:
            level, label = "limited", LABEL_UNCLEAR
        else:
            level = "moderate"
            label = LABEL_HIGHER if p_raw >= 0.5 else LABEL_LOWER
        # Fix (c): Evidence ceiling from the whole-sample offline audit.
        # If the model discriminated poorly on unseen participants or its probabilities
        # were not calibrated (Brier skill <= 0), no window may be classified as "moderate" evidence.
        auc = cal.get("pooled_auc")
        ceiling_ok = bool(cal["calibration_ok"] and auc is not None and auc >= config.AUC_MIN_FOR_MODERATE)
        if not ceiling_ok:
            level = "limited"
            caveats.insert(0, (
                "On held-out participants this model separated higher from lower self-reported stress only "
                f"slightly better than chance (AUC {auc if auc is not None else 'n/a'}; accuracy "
                f"{cal['pooled_accuracy']:.2f} vs {cal['majority_baseline_accuracy']:.2f} for always guessing "
                "the majority class), so its estimates should not be relied on."))
        if "beat_interval_features_missing" in q.flags:
            level = "limited"
        contribs = self._contributions(pipe, Xrow, q)
        res = EstimateResult("estimated", self.model_name, label, level, False, shown,
                             round(p_raw, 3), bool(cal["calibration_ok"]), q, contribs, [], caveats)
        res.sentences = self.explain_sentences(res)
        return res

    def estimate_for_window(self, index) -> EstimateResult:
        """Out-of-sample estimate for a window of the research dataset: the model is
        fitted WITHOUT that window's participant (leave-one-subject-out).
        Movement and temperature percentiles are derived strictly from training participants."""
        subject = self.groups.loc[index]
        return self.estimate(
            self.X.loc[index].to_dict(),
            _model=self._model_excluding(subject),
            stats=self._stats_for_subject(subject),
        )

    # ------------------------------------------------------------ explanation
    def _contributions(self, pipe, Xrow: pd.DataFrame, q: Quality) -> list[Contribution]:
        if not hasattr(pipe[-1], "coef_"):
            return []
        z = pipe[:-1].transform(Xrow)[0]
        coef = pipe[-1].coef_[0]
        out = []
        for name, zi, ci in zip(Xrow.columns, z, coef):
            measured = name not in q.missing
            held = name in config.SUPPRESSED_EXPLANATION_FEATURES
            note = ("not measured in this window" if not measured else
                    "held back from explanations (scaling under review)" if held else "")
            out.append(Contribution(name, FEATURE_LABELS.get(name, name), float(ci * zi), float(zi),
                                    measured and not held, note))
        return sorted(out, key=lambda c: -abs(c.contribution))

    def explain_sentences(self, res: EstimateResult, top: int = 3) -> list[str]:
        if res.abstained:
            return list(res.caveats)
        usable = [c for c in res.contributions if c.explained][:top]
        s = [f"The model's estimate is {res.label} (evidence: {res.evidence_level})."]
        if res.evidence_level == "limited":
            s.append("The evidence is limited, so please treat this as uncertain.")
        for c in usable:
            toward = "higher" if c.contribution > 0 else "lower"
            rel = "above" if c.z > 0 else "below"
            s.append(f"The model weighted your {c.label} ({rel} the dataset average) "
                     f"toward '{toward}' self-reported stress.")
        held = [c for c in res.contributions if c.note.startswith("held back")]
        if held:
            s.append("One feature (LF/HF ratio) is not explained here while its calculation is under review.")
        s.append("These are the model's weights, not reasons why your signals changed.")
        return s

    # ------------------------------------------------------------ baseline (descriptive)
    @staticmethod
    def baseline_summary(df: pd.DataFrame, subject: str) -> dict:
        """Descriptive summary of one person's OWN recorded baseline window(s).
        A baseline is what was measured for that person at that time. It is not a
        clinical reference range and not necessarily a 'healthy' state."""
        d = df[(df["subject"] == subject) & (df["stage"].astype(str).str.lower() == "baseline")]
        cols = [c for c in FEATURE_COLS if c in d.columns]
        if d.empty:
            return {"available": False, "note": "No baseline window recorded for this participant."}
        return {"available": True, "n_windows": int(len(d)),
                "values": {c: (None if pd.isna(d[c].mean()) else float(d[c].mean())) for c in cols},
                "note": ("Measured at one time in a laboratory session. A personal baseline is not a "
                         "clinical norm and is not necessarily your healthy or typical state.")}
