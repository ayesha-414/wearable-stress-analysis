"""Read-only diagnosis of the weak stress-classification performance.

This script CHANGES NOTHING in the existing pipeline. It reads the saved feature table, uses the
existing model builder unchanged, and writes NEW files under
    data/processed/model_investigation/diagnosis/

It reports (A) label/class structure, (B) stage signatures, (C) feature quality and
between-subject variance, (D) the existing baseline with participant-level resampling, (E) a
HYPOTHESIS-DRIVEN diagnostic of the stage-to-window mapping, and (F) how unstable ECE is.

IMPORTANT
* Everything is exploratory diagnosis. Nothing here selects or replaces a model.
* Section E tests ONE pre-specified hypothesis (documented in the report) with the existing,
  unchanged pipeline and the same participant-level folds for both conditions.
* The feature table has no timestamps, so window overlap and durations cannot be audited
  from it; those come from the raw-data alignment diagnostic.

Usage (PowerShell, from the project root):
    python scripts/10_model_diagnosis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import FEATURE_COLS, STRESS_THRESHOLD, _build_model, prepare_classification_data  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "data" / "processed" / "model_investigation" / "diagnosis"
SEED = 20261102
B = 2000


# ----------------------------------------------------------------------------- LOSO core
def loso_probs(X, y, g, model="logreg"):
    """Leave-one-subject-out probabilities with the EXISTING pipeline (fixed settings)."""
    out = pd.Series(index=X.index, dtype=float)
    base_pred = pd.Series(index=X.index, dtype=float)
    for s in sorted(g.unique()):
        te = g == s
        pipe = _build_model(model)
        pipe.fit(X[~te], y[~te])
        out[te] = pipe.predict_proba(X[te])[:, 1]
        base_pred[te] = float(y[~te].mean() >= 0.5)         # training-majority prediction
    return out, base_pred


def pooled_metrics(y, p, base):
    y, p, base = map(np.asarray, (y, p, base))
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return {"accuracy": (pred == y).mean(), "majority_accuracy": (base == y).mean(),
            "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
            "auc": auc, "brier": ((p - y) ** 2).mean()}


def cluster_bootstrap(y, p, base, groups, b=B, seed=SEED):
    """Resample PARTICIPANTS (not windows) with replacement; recompute pooled metrics."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(groups)))
    idx = {s: np.where(np.asarray(groups) == s)[0] for s in subs}
    y, p, base = map(np.asarray, (y, p, base))
    rows = []
    for _ in range(b):
        pick = rng.choice(subs, size=len(subs), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if len(set(y[ii])) < 2:
            continue
        m = pooled_metrics(y[ii], p[ii], base[ii])
        m["acc_minus_majority"] = m["accuracy"] - m["majority_accuracy"]
        rows.append(m)
    d = pd.DataFrame(rows)
    return {c: [float(d[c].quantile(.025)), float(d[c].quantile(.975))] for c in d.columns}


def per_subject(y, p, base, groups):
    rows = []
    for s in sorted(set(groups)):
        m = np.asarray(groups) == s
        ys, ps, bs = np.asarray(y)[m], np.asarray(p)[m], np.asarray(base)[m]
        acc, bacc = ((ps >= .5).astype(int) == ys).mean(), (bs == ys).mean()
        auc = roc_auc_score(ys, ps) if len(set(ys)) == 2 else np.nan
        rows.append({"subject": s, "n": int(m.sum()), "n_high": int(ys.sum()), "model_acc": acc,
                     "majority_acc": bacc, "auc": auc,
                     "verdict": "better" if acc > bacc else ("equal" if acc == bacc else "worse")})
    return pd.DataFrame(rows)


def ece(y, p, bins=5):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == k).sum() / len(y) * abs(p[idx == k].mean() - y[idx == k].mean())
                     for k in range(bins) if (idx == k).any()))


# ----------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "features" / "stress_features.csv")
    X, y, g = prepare_classification_data(raw)
    meta = raw.loc[X.index, ["subject", "stage", "cohort", "stress_level"]]
    res: dict = {"seed": SEED, "bootstrap_draws": B, "n_windows": int(len(X)),
                 "n_participants": int(g.nunique()), "threshold": STRESS_THRESHOLD}

    # ---------------- A. labels
    meta = meta.assign(high=(meta.stress_level >= STRESS_THRESHOLD).astype(int))
    per = meta.groupby("subject").high.agg(["sum", "count"])
    res["A_labels"] = {
        "high_overall": int(meta.high.sum()), "prevalence": round(float(meta.high.mean()), 4),
        "by_cohort": meta.groupby("cohort").high.agg(["sum", "count", "mean"]).round(3).to_dict("index"),
        "by_stage": meta.groupby("stage").high.agg(["sum", "count", "mean"]).round(3).to_dict("index"),
        "mean_rating_by_stage": meta.groupby(["cohort", "stage"]).stress_level.mean().round(2)
                                    .unstack(0).to_dict(),
        "subjects_never_high": int((per["sum"] == 0).sum()), "subjects_always_high": int((per["sum"] == per["count"]).sum()),
        "subjects_both_classes": int(((per["sum"] > 0) & (per["sum"] < per["count"])).sum()),
        "windows_per_subject": per["count"].value_counts().to_dict(),
        "rating_value_counts": meta.stress_level.value_counts().sort_index().to_dict(),
        "duplicate_feature_rows": int(X.round(10).duplicated().sum()),
    }
    meta.groupby(["cohort", "stage"]).agg(n=("high", "size"), mean_rating=("stress_level", "mean"),
                                          pct_high=("high", "mean")).round(3).to_csv(OUT / "A_label_by_stage.csv")
    per.assign(pct_high=per["sum"] / per["count"]).to_csv(OUT / "A_label_by_subject.csv")

    # ---------------- B. stage signatures (project stage NAMES; names are under review)
    sig = raw.groupby(["cohort", "stage"])[["hr_mean", "eda_mean", "temp_mean", "acc_mean", "stress_level"]].mean().round(3)
    sig.to_csv(OUT / "B_stage_signature_project_names.csv")
    res["B_stage_signature"] = {f"{c}|{s}": r for (c, s), r in sig.to_dict("index").items()}

    # ---------------- C. feature quality
    miss = X.isna().mean().round(3)
    sk = X.skew().round(2)
    rz = (X - X.median()) / (1.4826 * (X - X.median()).abs().median().replace(0, np.nan))
    sub_means = X.groupby(g).transform("mean")
    n_i = g.map(g.value_counts())
    # one-way random-effects ICC(1) (unbalanced-safe approximation using the average group size)
    k0 = (len(X) - (g.value_counts() ** 2).sum() / len(X)) / (g.nunique() - 1)
    icc = {}
    for c in X.columns:
        xc = X[c].dropna(); gg = g.loc[xc.index]
        gm = xc.groupby(gg).mean(); ng = xc.groupby(gg).size()
        msb = (ng * (gm - xc.mean()) ** 2).sum() / (len(gm) - 1)
        msw = ((xc - gg.map(gm)) ** 2).sum() / (len(xc) - len(gm))
        icc[c] = float((msb - msw) / (msb + (k0 - 1) * msw))
    corr = X.corr(method="spearman")
    pairs = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(X.columns)
             for b in X.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.9]
    fq = pd.DataFrame({"missing_frac": miss, "skew": sk, "robust_outliers_|z|>5": (rz.abs() > 5).sum(),
                       "icc1_between_subject": pd.Series(icc).round(3),
                       "cv": (X.std() / X.mean().abs()).round(3)})
    fq.to_csv(OUT / "C_feature_quality.csv")
    res["C_features"] = {"n_features": len(X.columns), "pairs_spearman_ge_0.9": pairs,
                         "median_icc1": round(float(np.median(list(icc.values()))), 3),
                         "icc1_ge_0.5": [c for c, v in icc.items() if v >= 0.5],
                         "missing_windows_by_feature_gt_0": miss[miss > 0].to_dict(),
                         "windows_missing_any_hrv": int(X[[c for c in X if c.startswith("hrv_")]].isna().any(axis=1).sum()),
                         "windows_missing_hr": int(X["hr_mean"].isna().sum()),
                         "events_per_parameter": round(float(y.sum() / X.shape[1]), 2)}
    # within-subject standardised high-vs-low effect (subjects with both classes), CI over subjects
    zs = (X - X.groupby(g).transform("mean")) / X.groupby(g).transform("std").replace(0, np.nan)
    eff = {}
    both = [s for s in g.unique() if 0 < y[g == s].sum() < (g == s).sum()]
    rng = np.random.default_rng(SEED)
    for c in X.columns:
        per_s = pd.Series({s: zs.loc[(g == s) & (y == 1), c].mean() - zs.loc[(g == s) & (y == 0), c].mean()
                           for s in both}).dropna()
        bs = [per_s.sample(len(per_s), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(1000)]
        eff[c] = {"mean_within_subject_z_diff": round(float(per_s.mean()), 3),
                  "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)], "n_subjects": int(len(per_s))}
    pd.DataFrame(eff).T.to_csv(OUT / "C_within_subject_effects.csv")
    res["C_within_subject_effects_n_subjects"] = len(both)
    res["C_features_with_ci_excluding_0"] = [c for c, v in eff.items() if v["ci95"][0] > 0 or v["ci95"][1] < 0]

    # ---------------- D. existing baseline, participant-level
    p, base = loso_probs(X, y, g)
    pm = pooled_metrics(y, p, base)
    ci = cluster_bootstrap(y, p, base, g.values)
    ps = per_subject(y, p, base, g.values)
    ps.round(3).to_csv(OUT / "D_per_subject_existing_logreg.csv", index=False)
    auc_s = ps.auc.dropna()
    res["D_baseline_existing_logreg"] = {
        "pooled": {k: round(float(v), 4) for k, v in pm.items()},
        "cluster_bootstrap_95ci": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci.items()},
        "per_subject": {"n_better_than_majority": int((ps.verdict == "better").sum()),
                        "n_equal": int((ps.verdict == "equal").sum()), "n_worse": int((ps.verdict == "worse").sum()),
                        "mean_acc": round(float(ps.model_acc.mean()), 3), "median_acc": round(float(ps.model_acc.median()), 3),
                        "sd_acc": round(float(ps.model_acc.std()), 3),
                        "n_with_auc": int(len(auc_s)), "mean_auc": round(float(auc_s.mean()), 3),
                        "median_auc": round(float(auc_s.median()), 3), "n_auc_above_0.5": int((auc_s > .5).sum())}}
    # dummy strategies with the same folds
    dummies = {}
    rs = np.random.default_rng(SEED)
    prior = {s: float(y[g != s].mean()) for s in g.unique()}
    pr = g.map(prior)
    dummies["most_frequent"] = float((base == y).mean())
    dummies["prior(predict_proba=training prevalence), Brier"] = float(((pr - y) ** 2).mean())
    dummies["stratified_random(mean of 200 draws)"] = float(np.mean([((rs.random(len(y)) < pr).astype(int) == y).mean() for _ in range(200)]))
    dummies["uniform_random(mean of 200 draws)"] = float(np.mean([((rs.random(len(y)) < .5).astype(int) == y).mean() for _ in range(200)]))
    res["D_dummy_accuracy"] = {k: round(v, 4) for k, v in dummies.items()}

    # ---------------- E. hypothesis-driven stage-mapping diagnostic
    # HYPOTHESIS H1 (pre-specified; evidence in the alignment diagnostic): the project's tag-index map is
    # shifted by one inter-tag interval for most stages, so each computed window belongs to a different
    # protocol step than its name says. Under the notebook-consistent reading:
    swap = {"V1": {"First Rest": "Stroop", "Stroop": "First Rest", "Second Rest": "TMCT", "TMCT": "Second Rest",
                   "Baseline": "Baseline"},
            "V2": {"First Rest": "TMCT", "TMCT": "First Rest", "Second Rest": "Real Opinion",
                   "Opposite Opinion": "Second Rest", "Subtract": "Subtract", "Baseline": "Baseline"}}
    # windows whose notebook-consistent counterpart has no feature row (V1 opinions/subtract; V2 'Real Opinion' gap
    # interval) are EXCLUDED, never guessed.
    rating = raw.set_index(["subject", "stage"]).stress_level.to_dict()
    sub = raw.copy()
    sub["true_stage"] = [swap[r.cohort].get(r.stage) for r in sub.itertuples()]
    sub = sub.dropna(subset=["true_stage"]).copy()
    sub["rating_true"] = [rating.get((r.subject, r.true_stage), np.nan) for r in sub.itertuples()]
    sub = sub.dropna(subset=["rating_true"])
    orig = sub.copy(); orig["stress_level"] = orig["stress_level"]
    fixed = sub.copy(); fixed["stress_level"] = fixed["rating_true"]
    Xo, yo, go = prepare_classification_data(orig)
    Xf, yf, gf = prepare_classification_data(fixed)
    common = Xo.index.intersection(Xf.index)
    Xo, yo, go, Xf, yf, gf = Xo.loc[common], yo.loc[common], go.loc[common], Xf.loc[common], yf.loc[common], gf.loc[common]
    po, bo = loso_probs(Xo, yo, go)
    pf, bf = loso_probs(Xf, yf, gf)
    mo, mf = pooled_metrics(yo, po, bo), pooled_metrics(yf, pf, bf)
    cio, cif = cluster_bootstrap(yo, po, bo, go.values), cluster_bootstrap(yf, pf, bf, gf.values)
    # paired participant-cluster bootstrap of the AUC difference (same resampled participants for both)
    rng = np.random.default_rng(SEED)
    subs = np.array(sorted(go.unique())); idx = {s: np.where(go.values == s)[0] for s in subs}
    dAUC = []
    for _ in range(B):
        ii = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(set(yo.values[ii])) < 2 or len(set(yf.values[ii])) < 2:
            continue
        dAUC.append(roc_auc_score(yf.values[ii], pf.values[ii]) - roc_auc_score(yo.values[ii], po.values[ii]))
    res["E_remapping_hypothesis_H1"] = {
        "n_windows_compared": int(len(common)), "n_participants": int(go.nunique()),
        "excluded_note": "windows whose notebook-consistent counterpart has no computed features were excluded",
        "prevalence_high_original_pairing": round(float(yo.mean()), 3), "prevalence_high_hypothesised_pairing": round(float(yf.mean()), 3),
        "original_pairing_same_windows": {k: round(float(v), 4) for k, v in mo.items()},
        "hypothesised_pairing_same_windows": {k: round(float(v), 4) for k, v in mf.items()},
        "ci95_original": {k: [round(a, 4), round(b, 4)] for k, (a, b) in cio.items() if k in ("auc", "accuracy", "acc_minus_majority")},
        "ci95_hypothesised": {k: [round(a, 4), round(b, 4)] for k, (a, b) in cif.items() if k in ("auc", "accuracy", "acc_minus_majority")},
        "paired_auc_difference_mean": round(float(np.mean(dAUC)), 4),
        "paired_auc_difference_ci95": [round(float(np.quantile(dAUC, .025)), 4), round(float(np.quantile(dAUC, .975)), 4)],
        "mean_rating_by_hypothesised_true_stage": sub.groupby(["cohort", "true_stage"]).rating_true.mean().round(2).unstack(0).to_dict(),
        "mean_hr_by_project_window_then_true_stage": sub.groupby(["cohort", "true_stage"]).hr_mean.mean().round(1).unstack(0).to_dict(),
        "caveat": ("Windows are still the project's (possibly shifted, HR 10 s late) windows; only the pairing of "
                   "window to label is changed. A proper test needs features recomputed from raw data."),
    }

    # ---------------- E2. objective-condition diagnostic on the SAME windows (task vs rest)
    # Same hypothesis H1, a cleaner target: is the window a protocol TASK or a REST/baseline? Evaluated with the
    # existing pipeline and the same participant folds under the project's names vs the hypothesised names.
    tasks = {"Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"}
    sub_pos = sub.reset_index(drop=True)          # prepare_classification_data re-indexes rows 0..n-1
    assert len(sub_pos) == len(Xo) and (sub_pos.loc[common, "subject"].values == go.loc[common].values).all(), \
        "row alignment between the remapped table and the model input is broken"
    sub2 = sub_pos.loc[common]
    y_name = sub2.stage.isin(tasks).astype(int); y_hyp = sub2.true_stage.isin(tasks).astype(int)
    pn, bn = loso_probs(Xo, y_name, go); ph, bh = loso_probs(Xo, y_hyp, go)
    mn, mh = pooled_metrics(y_name, pn, bn), pooled_metrics(y_hyp, ph, bh)
    dd = []
    rng2 = np.random.default_rng(SEED + 1)
    for _ in range(B):
        ii = np.concatenate([idx[s_] for s_ in rng2.choice(subs, len(subs), replace=True)])
        if len(set(y_name.values[ii])) < 2 or len(set(y_hyp.values[ii])) < 2:
            continue
        dd.append(roc_auc_score(y_hyp.values[ii], ph.values[ii]) - roc_auc_score(y_name.values[ii], pn.values[ii]))
    res["E2_task_vs_rest_diagnostic"] = {
        "n_windows": int(len(common)), "task_prevalence_by_project_names": round(float(y_name.mean()), 3),
        "task_prevalence_by_hypothesised_names": round(float(y_hyp.mean()), 3),
        "auc_by_project_names": round(float(mn["auc"]), 4), "auc_by_hypothesised_names": round(float(mh["auc"]), 4),
        "ci95_auc_project_names": [round(x, 4) for x in cluster_bootstrap(y_name, pn, bn, go.values)["auc"]],
        "ci95_auc_hypothesised_names": [round(x, 4) for x in cluster_bootstrap(y_hyp, ph, bh, go.values)["auc"]],
        "paired_auc_difference_mean": round(float(np.mean(dd)), 4),
        "paired_auc_difference_ci95": [round(float(np.quantile(dd, .025)), 4), round(float(np.quantile(dd, .975)), 4)],
        "balanced_accuracy_project_names": round(float(mn["balanced_accuracy"]), 4),
        "balanced_accuracy_hypothesised_names": round(float(mh["balanced_accuracy"]), 4),
        "note": "Objective protocol-condition target used ONLY as a diagnostic of the mapping hypothesis; it is not proposed as the model target.",
    }

    # ---------------- E3. window-duration confound
    # The feature table has no durations. These are STAGE-LEVEL MEDIAN durations (seconds) of the project's own windows,
    # transcribed from data/processed/alignment_diagnostic_*.txt section 5d (measured on the raw recordings):
    # project window = tag interval(s) of src/preprocessing.py; V1 Baseline spans two intervals (190 + 103 s).
    DUR = {("V1", "Baseline"): 293, ("V1", "First Rest"): 109, ("V1", "Stroop"): 326, ("V1", "Second Rest"): 137,
           ("V1", "TMCT"): 308, ("V1", "Real Opinion"): 34, ("V1", "Opposite Opinion"): 31, ("V1", "Subtract"): 34,
           ("V2", "Baseline"): 98, ("V2", "First Rest"): 261, ("V2", "TMCT"): 853, ("V2", "Second Rest"): 38,
           ("V2", "Real Opinion"): 24, ("V2", "Opposite Opinion"): 807, ("V2", "Subtract"): 38}
    logdur = pd.Series([np.log(DUR[(c, st)]) for c, st in zip(raw.cohort, raw.stage)], index=raw.index)
    Xd = pd.DataFrame({"log_duration": logdur.loc[X.index].values}, index=X.index)
    rho = {c: float(spearmanr(Xd["log_duration"], X[c], nan_policy="omit")[0]) for c in X.columns}
    pdur, bdur = loso_probs(Xd, y, g)
    mdur = pooled_metrics(y, pdur, bdur)
    # duration alone vs objective condition, on the E2 windows
    Xd2 = pd.DataFrame({"log_duration": logdur.loc[sub_pos.loc[common].index.map(lambda i: sub.index[i])].values}, index=common)
    pd_n, bd_n = loso_probs(Xd2, y_name, go); pd_h, bd_h = loso_probs(Xd2, y_hyp, go)
    res["E3_duration_confound"] = {
        "source_of_durations": "stage-level medians from the raw-data alignment diagnostic (not per-window)",
        "duration_range_s": [min(DUR.values()), max(DUR.values())],
        "spearman_logduration_vs_feature_top8": dict(sorted(((k, round(v, 3)) for k, v in rho.items()),
                                                            key=lambda kv: -abs(kv[1]))[:8]),
        "loso_auc_duration_only_predicting_self_report_label": round(float(mdur["auc"]), 4),
        "loso_auc_duration_only_predicting_task_vs_rest_project_names": round(float(pooled_metrics(y_name, pd_n, bd_n)["auc"]), 4),
        "loso_auc_duration_only_predicting_task_vs_rest_hypothesised_names": round(float(pooled_metrics(y_hyp, pd_h, bd_h)["auc"]), 4),
        "caveat": "A duration-only model is a diagnostic of confounding, not a proposed model.",
    }

    # ---------------- F. ECE instability
    def recal_ece(seed):
        r = np.random.default_rng(seed)
        ys, praw, pcal = [], [], []
        for s in sorted(g.unique()):
            te = g == s
            trn = g[~te]; subs_tr = trn.unique(); r.shuffle(subs_tr)
            fold = {sb: i % 5 for i, sb in enumerate(subs_tr)}
            fid = trn.map(fold).values
            oof = np.zeros((~te).sum())
            Xt, yt = X[~te], y[~te]
            for f in range(5):
                m = _build_model("logreg").fit(Xt[fid != f], yt[fid != f]); oof[fid == f] = m.predict_proba(Xt[fid == f])[:, 1]
            lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(lg(oof), yt)
            m = _build_model("logreg").fit(Xt, yt); q = m.predict_proba(X[te])[:, 1]
            ys += y[te].tolist(); praw += q.tolist(); pcal += pl.predict_proba(lg(q))[:, 1].tolist()
        ys, pcal = np.array(ys), np.array(pcal)
        return ece(ys, pcal), ece(ys, pcal, 10), float(((pcal - ys) ** 2).mean())
    runs = [recal_ece(SEED + i) for i in range(8)]
    res["F_ece_instability"] = {
        "n_random_inner_fold_assignments": 8,
        "recalibrated_ECE_5bins_range": [round(min(r[0] for r in runs), 4), round(max(r[0] for r in runs), 4)],
        "recalibrated_ECE_10bins_range": [round(min(r[1] for r in runs), 4), round(max(r[1] for r in runs), 4)],
        "recalibrated_Brier_range": [round(min(r[2] for r in runs), 4), round(max(r[2] for r in runs), 4)],
        "training_prior_Brier": round(float(((y.map({s: float(y[g != s].mean()) for s in g.unique()}) if False else g.map(prior)) - y).pow(2).mean()), 4),
    }

    (OUT / "diagnosis_results.json").write_text(json.dumps(res, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps({k: res[k] for k in ("E3_duration_confound",)},
                     indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
