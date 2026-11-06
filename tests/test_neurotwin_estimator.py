"""Estimator: leakage safety, abstention, evidence ceiling, calibration policy, wording."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.models import FEATURE_COLS, _build_model, prepare_classification_data
from src.neurotwin import config, simulation
from src.neurotwin.demo import DemoData
from src.neurotwin.estimator import (LABEL_NOT_ASSESSED, StressEstimator)
from src.neurotwin.language import find_forbidden

warnings.filterwarnings("ignore")
CSV = config.FEATURES_CSV


@pytest.fixture(scope="module")
def est():
    return StressEstimator(pd.read_csv(CSV))


def test_uses_all_23_existing_features_unchanged(est):
    assert len(FEATURE_COLS) == 23 and list(est.X.columns) == FEATURE_COLS


def test_loso_estimate_matches_independent_refit_without_that_subject(est):
    idx = est.X.index[40]
    subj = est.groups.loc[idx]
    keep = est.groups != subj
    pipe = _build_model("logreg").fit(est.X[keep], est.y[keep])
    expected = float(pipe.predict_proba(est.X.loc[[idx]])[0, 1])
    assert est.estimate_for_window(idx).probability_raw == pytest.approx(expected, abs=1e-3)


def test_flipping_a_heldout_subjects_labels_cannot_change_their_estimate(est):
    """Leakage test: a window's estimate must not depend on its own participant's labels."""
    idx = est.X.index[40]
    subj = est.groups.loc[idx]
    before = est.estimate_for_window(idx).probability_raw
    flipped = pd.read_csv(CSV)
    mask = flipped["subject"] == subj
    flipped.loc[mask, "stress_level"] = np.where(flipped.loc[mask, "stress_level"] >= 5, 1.0, 9.0)
    est2 = StressEstimator(flipped)
    after = est2.estimate_for_window(idx).probability_raw
    assert after == before


def test_calibration_audit_reproduces_saved_logistic_regression_accuracy(est):
    """Ties the new audit to the project's existing saved LOSO result."""
    saved = pd.read_csv(config.PROCESSED_DIR / "model_comparison_scores.csv").set_index("Classifier")
    row = [i for i in saved.index if str(i).upper().startswith("LOG")][0]
    assert est.calibration_report()["pooled_accuracy"] == pytest.approx(float(saved.loc[row, "Accuracy"]), abs=1e-3)


def test_majority_baseline_is_reported_and_current_model_does_not_beat_it(est):
    rep = est.calibration_report()
    assert rep["majority_baseline_accuracy"] == pytest.approx(0.616, abs=1e-3)
    assert rep["pooled_accuracy"] < rep["majority_baseline_accuracy"]


def test_all_audited_windows_come_from_35_participants_each_held_out_once(est):
    rep = est.calibration_report()
    assert rep["n_subjects"] == 35 and rep["n_windows"] == len(est.X)


def test_probabilities_hidden_when_calibration_not_demonstrated(est):
    rep = est.calibration_report()
    assert rep["calibration_ok"] is False
    for idx in est.X.index[::20]:
        r = est.estimate_for_window(idx)
        assert r.probability_shown is None and r.calibrated is False


def test_policy_would_show_probability_only_if_both_criteria_met(est):
    rep = est.calibration_report()
    expected = rep["brier_recalibrated"] < rep["brier_training_prior"] and rep["ece_recalibrated"] <= config.CAL_ECE_MAX
    assert rep["calibration_ok"] == expected


def test_evidence_is_never_moderate_or_high_with_current_audit(est):
    levels = {est.estimate_for_window(i).evidence_level for i in est.X.index}
    assert levels <= {"limited", "insufficient"}


def test_audit_caveat_is_shown_to_the_user(est):
    r = est.estimate_for_window(est.X.index[0])
    assert any("should not be relied on" in c for c in r.caveats)


def test_sensor_dropout_abstains_instead_of_guessing(est):
    row = {c: np.nan for c in FEATURE_COLS}
    r = est.estimate(row)
    assert r.abstained and r.evidence_level == "insufficient" and r.label == LABEL_NOT_ASSESSED
    assert r.probability_shown is None and r.contributions == []


def test_missing_heart_rate_group_abstains(est):
    row = est.X.iloc[5].to_dict()
    for c in FEATURE_COLS:
        if c.startswith("hr_"):
            row[c] = np.nan
    r = est.estimate(row)
    assert r.abstained and "heart rate" in " ".join(r.quality.reasons)


def test_windows_with_no_heart_rate_are_abstained_in_the_real_data(est):
    abst = [i for i in est.X.index if est.estimate_for_window(i).abstained]
    assert abst and set(est.groups.loc[abst]) == {"f07"}       # known sensor-cover issue


def test_missing_beat_intervals_flagged_and_evidence_capped(est):
    row = est.X.iloc[5].to_dict()
    for c in FEATURE_COLS:
        if c.startswith("hrv_"):
            row[c] = np.nan
    r = est.estimate(row)
    assert "beat_interval_features_missing" in r.quality.flags
    assert r.evidence_level in ("limited", "insufficient")


def test_high_movement_is_flagged_as_alternative_explanation(est):
    row = est.X.iloc[5].to_dict()
    row["acc_mean"] = float(est.X["acc_mean"].max() * 3)
    r = est.estimate(row)
    assert "movement_high" in r.quality.flags
    assert any("physical activity" in c.lower() for c in r.caveats)


def test_lf_hf_is_held_back_from_explanations_but_model_is_unchanged(est):
    for idx in est.X.index[::15]:
        r = est.estimate_for_window(idx)
        lf = [c for c in r.contributions if c.feature == "hrv_lf_hf_ratio"]
        assert lf and lf[0].explained is False
        assert not any("LF/HF" in s and "weighted" in s for s in r.sentences)
    assert "hrv_lf_hf_ratio" in FEATURE_COLS                  # still part of the unchanged model


def test_contributions_are_exactly_coefficient_times_standardised_value(est):
    idx = est.X.index[10]
    r = est.estimate_for_window(idx)
    pipe, _ = est._model_excluding(est.groups.loc[idx])
    z = pipe[:-1].transform(est.X.loc[[idx]])[0]
    coef = pipe[-1].coef_[0]
    got = {c.feature: c.contribution for c in r.contributions}
    for name, zi, ci in zip(FEATURE_COLS, z, coef):
        assert got[name] == pytest.approx(ci * zi, rel=1e-9, abs=1e-12)


def test_explanations_describe_model_not_cause(est):
    for idx in est.X.index[::7]:
        r = est.estimate_for_window(idx)
        if r.abstained:
            continue
        text = " ".join(r.sentences)
        assert "model" in text.lower() and "not reasons" in text.lower()


def test_no_forbidden_wording_in_any_estimate_text(est):
    for idx in est.X.index:
        r = est.estimate_for_window(idx)
        assert find_forbidden(" ".join(r.sentences + r.caveats + [r.label])) == [], idx


def test_result_layer_is_estimated(est):
    assert est.estimate_for_window(est.X.index[0]).layer == "estimated"


def test_baseline_summary_is_descriptive_not_a_norm(est):
    s = est.baseline_summary(est.raw, "S01")
    assert s["available"] and "not a clinical norm" in s["note"]
    assert find_forbidden(s["note"]) == []


def test_baseline_missing_is_reported_honestly(est):
    assert est.baseline_summary(est.raw, "nobody")["available"] is False


def test_report_file_matches_current_audit(est):
    path = config.ESTIMATOR_REPORT
    if not path.is_file():
        pytest.skip("run scripts/08_neurotwin_estimator_report.py first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    fresh = est.calibration_report()
    for k in ("pooled_accuracy", "pooled_auc", "brier_recalibrated", "ece_recalibrated", "calibration_ok"):
        assert saved[k] == fresh[k], k


# ------------------------------------------------------------------ demo adapter
def test_demo_reports_lab_rating_as_reported_not_measured(est):
    d = DemoData(est)
    lab = d.lab_report(est.X.index[0])
    assert lab["setting"].startswith("laboratory") and lab["rating"] is not None


# ------------------------------------------------------------------ simulation
def test_simulation_is_labelled_and_deterministic():
    a, b = simulation.run(n_users=10, rounds=20, seed=3), simulation.run(n_users=10, rounds=20, seed=3)
    assert a == b and a["label"].startswith("SIMULATED USERS")
    assert "not evidence" in a["label"].lower()


def test_simulation_checks_adaptation_code_moves_toward_invented_preferences():
    r = simulation.run(n_users=30, rounds=30, seed=11)
    assert r["adaptive_mean"] > r["static_mean"]
    assert "invented" in r["caveat"]


def test_simulation_never_touches_a_store():
    import pathlib
    text = pathlib.Path(simulation.__file__).read_text(encoding="utf-8")
    assert "store" not in text.lower().replace("restore", "")
