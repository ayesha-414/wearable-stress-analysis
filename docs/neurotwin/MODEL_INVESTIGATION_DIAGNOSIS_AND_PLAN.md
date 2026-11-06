# Model investigation — diagnosis, research and proposed plan (for approval)

**Status:** Phases A and B and the plan for C–H. **No existing ML code, feature table, model or report was modified.** The final report (`MODEL_INVESTIGATION_AND_IMPROVEMENT_REPORT.md`) is written only after you approve the plan and the experiments are run.

## 0. What this is based on, and its limits

* I worked on my sandbox copy of the project (the zip I produced for you), **not** on `C:\Users\AYESHA\Downloads\NeuroTwin_full_project`, which I cannot reach. I have **no raw recordings**. The raw-data facts below come from the alignment diagnostic (`data/processed/alignment_diagnostic_*.txt`) that was run on your machine against the real files, plus the dataset authors' notebook in the repo. I could not re-verify them against the raw files.
* **Your numbers differ slightly from mine** on the same code: recalibrated Brier 0.2507 vs 0.2481 and ECE 0.0524 vs 0.1046 (accuracy 0.5703 and AUC 0.5393 are identical). Section 7 shows ECE moves between 0.061 and 0.106 depending only on how the inner folds are drawn, so this is expected; run `scripts\11_environment_fingerprint.py` so we can compare library versions and file hashes.
* Everything numeric below is reproducible with `scripts/10_model_diagnosis.py` (read-only; writes new files under `data/processed/model_investigation/diagnosis/`).
* Terminology: **verified** = demonstrated by a test or a measurement; **supported** = consistent with several independent lines of evidence; **hypothesis** = not yet tested.

## 1. Executive summary

The weak result is best described as **"no demonstrated skill"**, not "demonstrated harm": the 95% participant-level interval for the AUC is 0.41–0.66 and for accuracy minus the majority baseline it is −0.18 to +0.10, both including no effect. More importantly, I found **defects in the data pipeline that make the current features and labels hard to trust**. Fixing them must come before any model tuning.

| # | Root cause | Evidence strength | Does it explain the weak result? |
|---|---|---|---|
| **RC1** | **Stage-to-window mapping disagrees with the dataset authors' notebook** for 8 of the 9 task windows (median overlap 0.00). Durations, physiology and the self-report file order all point the same way (§3.1). | **Strong (supported)** | **Unknown.** A crude re-pairing test was inconclusive (ΔAUC +0.03, CI −0.09 to +0.15). Needs features recomputed from raw data. |
| **RC2** | **HR is read 10 s later than the stage boundaries.** `reindex(index − 10)` does not shift data (unit-tested). HR starts +10 s after EDA in all 37 recordings. | **Verified** | Unknown; matters most for the 30 s tasks. |
| **RC3** | **Window length varies 24–853 s** inside one feature table. Duration-sensitive features (`temp_std`, `hr_range`, `hr_std`, `eda_scr_count`) correlate with log duration at ρ = 0.53–0.79, and duration alone predicts task-vs-rest with AUC 0.82–0.90. | **Verified** (using stage-level medians) | Not for the self-report label (duration-only AUC 0.50), but it **contaminates any condition-related signal and any feature comparison**. |
| **RC4** | **The label is a thresholded single self-rating** with strong person effects: 9 of 35 participants never reach ≥5, 4 always do, only 22 have both classes. | **Verified** | Probably contributes; limits what any participant-independent model can reach. |
| **RC5** | **Very small effective sample:** 35 participants, 101 events, 23 features (4.4 events per parameter). | **Verified** | Yes: wide intervals; only large effects are detectable. |
| **RC6** | **Between-person variance dominates** many features (median ICC 0.43; skin temperature 0.93; EDA level 0.92). | **Verified** | Likely; motivates baseline-relative features, to be tested without leakage. |
| **RC7** | 19% of windows lack beat-interval features (median-imputed); participant `f07` lacks HR/temperature; `f14` is dropped (35 of 36 participants). | **Verified** | Minor. |
| **RC8** | Redundant features: 6 pairs with |ρ| ≥ 0.9 (e.g. `eda_mean` ≈ `eda_scl_mean`, ρ = 1.0). | **Verified** | Minor; worsens the events-per-parameter problem. |
| **RC9** | Calibration cannot be shown at this sample size; ECE alone is unstable. | **Verified** | Explains why probabilities must not be shown. |

**No leakage was found** in the leave-one-subject-out (LOSO) evaluation (§5). The model is therefore weak for data and design reasons, not because of a hidden evaluation flaw.

**Recommendation in one sentence:** keep the current baseline as the reference, resolve RC1–RC3 on a *new* feature table, then re-run a pre-specified, participant-level evaluation; if no model clears the criteria in §9.4, retain the transparent baseline with abstention.

## 2. Pipeline trace (raw recordings → evaluation)

| Step | Where | Finding |
|---|---|---|
| Load | `src/loader.py` | Subject list built from folders on disk; `f14_a`/`f14_b` are dropped and no merged `f14` is created (35 of 36 participants). |
| Per-signal time base | `src/preprocessing.py::resample_to_1hz` | Each signal is indexed from **its own first timestamp**. |
| HR "10 s alignment" | `clean_subject_signals` | `hr.reindex(hr.index − 10)` → **no shift** (RC2). |
| Stage windows | `segment_by_stage`, `STRESS_V1_STAGES` / `STRESS_V2_STAGES` | Tag-index intervals; HR/EDA/TEMP/BVP/ACC use a **closed** interval `[t0, t1]` (neighbouring windows share one boundary second), IBI uses `[t0, t1)` by UTC. |
| Features | `src/features.py` | Statistics computed over the whole window, whatever its length (RC3). |
| Labels | `src/models.py::prepare_classification_data` | `stress_level >= 5.0` from a single constant (`STRESS_THRESHOLD`), used consistently in code and dashboard. Rows without a rating are dropped and **the index is reset**. |
| Model | `_build_model` | `imputer → scaler → classifier` in one pipeline, fitted per fold; no tuning; threshold 0.5. |
| LOSO | `loso_cross_validate` | One fold per participant; verified each held out exactly once. |
| Calibration/uncertainty | `src/neurotwin/estimator.py` | Platt scaling on inner group-fold out-of-fold predictions inside each training fold. |

## 3. Dataset and label findings (Phase A-A)

* **Class balance:** 101 of 263 windows are high (0.384); V1 0.424 (61/144), V2 0.336 (40/119). The cohorts differ in protocol (V2 is remote, Stroop removed), so results should also be reported by cohort.
* **Per participant:** 7–8 windows each; 9 never high, 4 always high, 22 both classes. Pooled accuracy is therefore partly a between-person statement.
* **Rating values:** the rating counts include 0.0 and 0.5 (two windows each) although the dataset documents a 1–10 scale; half-point ratings also occur (1.5 ×2, 4.5 ×1, 5.5 ×2), the last two next to the threshold. Worth confirming with the source files.
* **Duplicates:** no duplicate feature rows. Windows are disjoint tag intervals except for a shared boundary second.
* **Participants:** `S02` appears once (the dataset notes a duplicated recording); `f07` has HR/temperature missing; `f14` is omitted. A merge is non-trivial (baseline in part A only; 91–102 s recording gap; per-part IBI offsets restart), so I will treat `f14` as a pre-specified **sensitivity analysis**, not a silent fix.

### 3.1 Stage-to-window mapping (RC1), the most important finding

Evidence, all from data already in the project:

1. **Notebook comparison** (diagnostic §5a): median window overlap (IoU) with the authors' notebook is **0.00** for V1 Stroop, TMCT, Real/Opposite Opinion and Subtract, and V2 TMCT, Real/Opposite Opinion; only V2 Subtract matches (1.00).
2. **Durations** (diagnostic §5d, median seconds). Project "V1 Stroop" = 326 s and "V1 TMCT" = 308 s, which fit the 5-minute **rests after** those tasks, while the notebook's task intervals are 109 s and 137 s. In V2 the project's "Opposite Opinion" window is **807 s** (an opinion task has a 30 s limit) and its "Second Rest" is 38 s.
3. **Chronological order:** the notebook-consistent reading reproduces the column order of `Stress_Level_v1/v2.csv` exactly for both cohorts; the project's tag-index order does not. (Tests: `test_EXPECTED_stage_map_follows_chronological_self_report_order`, marked as a known defect.)
4. **Physiology from your own feature table** (mean HR, bpm, by *project* stage name). V1: "Stroop" 75.0, "TMCT" 72.9 (lowest) but "First Rest" 81.7, "Second Rest" 83.1 (highest). V2: "TMCT" 78.6 vs "First Rest" 83.3. Windows named as tasks look like recovery; windows named as rests look like tasks.
5. **Sign of the within-person effect:** in the 22 participants with both classes, windows rated *higher* have *lower* HR (standardised difference −0.40, 95% CI −0.72 to −0.07) and longer beat intervals (+0.63, CI 0.29 to 0.93), the opposite of the expected direction.

What is **not** shown: that RC1 causes the weak performance. Re-pairing labels to the notebook-consistent windows for the 192 windows where a counterpart exists changed the AUC from 0.546 to 0.574 (paired difference +0.025, 95% CI −0.093 to +0.154): inconclusive. That test still uses the possibly shifted windows and omits the 71 windows that were never computed (V1 opinion/subtract windows, V2 "Opposite Opinion"), so only a recomputation from raw data can settle it.

## 4. Sensor and feature findings (Phase A-B)

* **HR offset (RC2):** unit test with a synthetic series of known timing: the cleaned HR value at EDA time *t* is the value recorded at *t + 10 s* (error +10 s for every sample tested). For 30–40 s windows this displaces a quarter to a third of the window. `tests/test_model_investigation_diagnosis.py` documents it (characterisation test + strict xfail of the intended behaviour).
* **Window length (RC3):** durations 24–853 s. Spearman correlation of log duration with `temp_std` 0.79, `hr_range` 0.74, `hr_std` 0.65, `eda_scr_count` 0.53. Duration alone (one number) yields LOSO AUC **0.82** for "task vs rest" by project names and **0.90** by notebook-consistent names, which is as good as or better than all 23 features (0.77, 0.87). So a high "task-vs-rest" score here **cannot** be taken as evidence of physiology. (Duration is *not* associated with the self-report label: AUC 0.496.)
* **Between-person variance (RC6):** median ICC(1) 0.43. ICC ≥ 0.5 for 11 of 23 features: `temp_mean` (0.93), `eda_mean` and `eda_scl_mean` (0.92), `eda_std`, `eda_range`, `hrv_mean_ibi` (0.66), `hr_mean` (0.64), `hr_min` (0.63), `hrv_sdnn`, `hrv_rmssd`, `hrv_pnn50`. By contrast `temp_std` (0.05) is almost entirely within-person (see `C_feature_quality.csv`).
* **Missingness:** beat-interval features missing in 51 windows (19.4%) and HR/temperature in 7 (participant `f07`); handled by in-fold median imputation.
* **Redundancy/skew:** 6 pairs with |ρ| ≥ 0.9 (`hr_mean`–`hr_min` 0.92, `hr_std`–`hr_range` 0.97, `eda_mean`–`eda_scl_mean` 1.00, `eda_std`–`eda_range` 0.99, `acc_std`–`acc_max` 0.96, `hrv_rmssd`–`hrv_pnn50` 0.92). Several features are strongly right-skewed (skewness: `hr_diff_rms` 10.4, `eda_slope` 8.4, `hr_std` 5.1, `eda_std` 4.5, `eda_range` 4.4).
* **Feature definitions** (from Stage 1 of the earlier audit): LF/HF averages ordinates instead of integrating power (×2.28 inflation); `hr_diff_rms` is not conventional RMSSD. Not changed.

## 5. Leakage audit (Phase A-C)

| Item | Verdict | Evidence |
|---|---|---|
| Subject leakage | None found | One fold per participant; each held out once (35 folds per classifier in `loso_results.csv`). |
| Window-overlap leakage | None across participants | Windows are disjoint tag intervals within a person, and all of a person's windows are held out together. Boundary seconds are shared inside a person only. |
| Preprocessing leakage | None | Imputer and scaler are inside the per-fold pipeline. |
| Feature-selection leakage | None in the evaluated model | The model uses all 23 features; `scripts/06_feature_selection.py` selects inside training folds (not yet executed). |
| Hyper-parameter leakage | None | Fixed settings; no tuning. |
| Calibration leakage | None | Platt calibrator fitted on out-of-fold training predictions only; a test flips a held-out participant's labels and confirms the estimate is unchanged. |
| Threshold leakage | None | Fixed 0.5. |
| Label leakage | None found | Label from the person's own rating; duration alone cannot predict it (AUC 0.496). |
| **Minor, in `src/neurotwin` (my code)** | **To fix with approval** | (a) Movement/temperature *flags* use percentiles over all participants' features (label-free, display only). (b) The "evidence ceiling" uses the all-participant audit, which is standard reporting but not a per-fold decision. (c) `StressEstimator.meta` assumes `X.index` equals the raw row index; true now (all rows labelled), but wrong if any unlabelled row were dropped because `prepare_classification_data` resets the index. |

## 6. Baseline results with participant-level uncertainty (Phase C preview)

Existing logistic regression, LOSO, 35 participants, 263 windows. Intervals use a **participant-cluster bootstrap** (2,000 draws; Field & Welsh 2007), not window resampling.

| Metric | Value | 95% interval |
|---|---|---|
| Accuracy | 0.570 | 0.481 – 0.654 |
| Majority-class accuracy | 0.616 | — |
| Accuracy − majority | −0.046 | −0.177 – +0.099 |
| Balanced accuracy | 0.560 | 0.468 – 0.647 |
| Sensitivity / specificity | 0.515 / 0.605 | — |
| AUC (pooled) | 0.539 | 0.408 – 0.660 |

Participant level: **10 better, 8 equal, 17 worse** than the majority baseline; per-participant accuracy mean 0.577, median 0.571, SD 0.267; of the 22 participants with both classes, mean AUC 0.510, median 0.483, and only 9 above 0.5. Dummy baselines on the same folds: most-frequent 0.616, stratified-random 0.516, uniform-random 0.500 (accuracy); training-prior Brier 0.2441.

## 7. Calibration (RC9)

Recalibrated Brier is **worse than the training-prior Brier in every one of 8 random inner-fold assignments** (0.2473–0.2505 vs 0.2441), so calibrated probabilities are not justified. ECE varies from 0.061 to 0.106 across those same 8 draws with identical data and model, so a single ECE value (including your 0.0524) is not a reliable basis for judging calibration, and **ECE must not be used to choose a calibration method**. Proper scoring (Brier skill against the prior, log loss) and reliability tables with sample counts are the criteria.

## 8. Literature (Phase B) and what each supports

Verified references are in `model_investigation_references.csv` (level A = record/abstract seen; C = preprint). Full-text reading is still needed before citing in the thesis. Statements marked **[inference]** are mine.

| Topic | Source | What it supports here |
|---|---|---|
| Subject-wise validation | Saeb 2017; Little 2017 | Subject-wise CV is the right protocol; record-wise can massively overestimate. Little et al. note clustered data raise issues beyond subject-wise CV. |
| Small samples | Varoquaux 2018; Vabalas 2019; Riley 2020 | CV error bars of about ±10% at ~100 samples and fold SE underestimates them; K-fold biased with small n; feature selection on pooled data biases more than tuning; ≥10 events per candidate parameter is a rule of thumb (here 4.4). |
| Selection bias | Ambroise & McLachlan 2002; Cawley & Talbot 2010 | Selection and tuning must sit inside the validation loop; model-selection over-fitting can equal between-algorithm differences, so limit the number of comparisons. |
| Uncertainty around metrics | Field & Welsh 2007 | Resample participants (clusters). |
| Calibration | Van Calster 2019; Niculescu-Mizil 2005 | Calibration needs adequate samples; poor calibration can mislead; tree ensembles distort probabilities. **[inference]** with 35 participants, calibration cannot be reliably demonstrated. |
| Selective prediction | El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017 | Reject option and risk–coverage curves are the established frame for abstention. Guarantees need sample sizes we lack, so only descriptive curves are justified. |
| Inter-subject variability / baselines | Lamichhane 2016; Alamudun 2012; Baseline-normalisation 2026 | Baseline-based normalisation is a recognised approach; **a 2026 study shows that including the baseline in the evaluation data inflates accuracy even with subject-independent partitions**, so any baseline-relative feature must be evaluated *test-exclusively*. |
| Label limitations | Schmidt 2018 (preprint); human-stress review (preprint) | Ground truth is usually subjective and person-dependent. **[inference]** a single thresholded rating is a weak target. |
| Validation practice in the field | Vos et al.; Sharma 2026 | Generalisation and reporting remain weak across the field. |

## 9. Proposed plan (needs your approval)

### 9.1 Stage 0: resolve the data defects (raw data, your machine)
1. You review the evidence in §3.1 and confirm the **notebook-consistent stage map** (Baseline; V1 Stroop, First Rest, TMCT, Second Rest, Real/Opposite Opinion, Subtract; V2 TMCT, First Rest, Real/Opposite Opinion, Second Rest, Subtract), using the raw tag intervals the authors' notebook uses.
2. Add a **new** preprocessing path (`src/investigation/`) with: corrected stage map, **HR shifted so that HR row k is placed at EDA time k + 10 s**, window start/end/duration saved, and no change to `src/preprocessing.py`.
3. Build a **new** feature table `data/features_v2/stress_features_v2.csv` in two variants, specified in advance: **(A)** full corrected windows; **(B, proposed primary)** duration-matched windows of 30 s anchored at the stage start, to remove the duration confound (sensitivity at the stage end). The 23 feature formulas stay as they are, and count-type features also get a per-minute version. The old table and every old result stay as the baseline.
4. `f14`: sensitivity analysis only (with/without a documented merge).

### 9.2 Stage 1: reproducible evaluation framework (Phase C)
Same participant-level LOSO folds for everything. Models: majority; `DummyClassifier` (prior, stratified, uniform); existing logistic regression; L2 logistic regression (C from {0.01, 0.1, 1, 10} chosen by **inner participant-level CV**); linear SVM; RBF SVM only if the inner CV supports it; random forest and Extra Trees with the existing fixed settings (no tuning). Report all metrics you listed, per class, confusion matrices, participant-level results, and participant-cluster bootstrap intervals; pooled and participant-level metrics reported separately; V1/V2 reported separately.

### 9.3 Stage 2: candidate improvements (Phase D), each justified and tested once
| Candidate | Addresses | Support | Expected benefit | Risk | Evaluable with 35 participants? |
|---|---|---|---|---|---|
| Corrected stage map + HR alignment + duration-matched windows | RC1–RC3 | Verified defects; Saeb 2017 (valid evaluation) | Possibly large; unknown | Changes feature values (new table, old kept) | Yes, directly |
| Regularised logistic regression | RC5 | Riley 2020 (events per parameter) | Less overfitting | Small gain | Limited |
| Pre-specified feature reduction (drop exact duplicates) and in-fold selection | RC5, RC8 | Ambroise 2002; Vabalas 2019 | Stability | Selection bias if done outside folds (it will not be) | Weakly |
| Baseline-relative features, **test-exclusive** (own baseline window is calibration only and is removed from the test set) | RC6 | Lamichhane 2016; baseline-normalisation 2026 | Reduces person offsets | Inflation if done test-inclusively (prevented) | Yes, with wide intervals |
| Robust scaling / log transforms for skewed features | skew | Standard practice | Small | Over-tuning | Limited |
| Class weighting, decision threshold from inner folds | prevalence 0.38 | Standard practice | Better balanced accuracy | Threshold optimism | Limited |
| Nested Platt calibration only | RC9 | Van Calster 2019 | Probabilities only if Brier skill > 0 | Likely to fail | Weakly |

### 9.4 Decision rule (fixed before running)
A new model is recommended only if, on the corrected data and with participant-level resampling: (i) pooled balanced accuracy and AUC have 95% interval lower bounds above 0.5; (ii) it beats the majority baseline on balanced accuracy, macro-F1 and AUC; (iii) a clear majority of the 22 two-class participants have AUC above 0.5 (one-sided binomial test); (iv) the paired improvement over the current logistic regression excludes zero; (v) leakage tests pass and the result reproduces across fixed seeds; (vi) probabilities are shown only if Brier skill versus the prior is positive in held-out participants; (vii) added complexity is justified. If not, **retain the baseline with abstention**.

### 9.5 Stage 3: selective prediction (Phase E)
Risk–coverage curves from held-out scores; abstention from signal quality and a confidence band chosen **inside** training folds; report coverage vs balanced accuracy. The message "Insufficient physiological evidence to provide a reliable estimate" is already the default for low-quality or unreliable windows; no recommendation will treat an estimate as proof of stress.

### 9.6 Rejected in advance
Deep models and large grids (Cawley 2010, small n); synthetic oversampling (imbalance is mild and synthetic samples add nothing real); per-person supervised fine-tuning (no legitimate per-person labelled training data in this protocol); test-inclusive baseline normalisation (inflates results); replacing the target by task-vs-rest (confounded by duration and a different construct); choosing thresholds or calibration on pooled held-out predictions or by ECE.

## 10. Files I intend to touch, and risks

**Add (new files only):** `src/investigation/` (stage map, HR alignment, duration-matched windows), `scripts/12_build_features_v2.py`, `scripts/13_loso_benchmark.py`, `scripts/14_selective_prediction.py`, tests for each, `data/features_v2/`, `data/processed/model_investigation/`, the final report.
**Modify only after separate approval:** `src/neurotwin/estimator.py` (fix the three minor items in §5 and optionally load a v2 model), the dashboard's Evaluation page text, `src/neurotwin/language`-related tests. **Not modified:** `src/models.py`, `src/features.py`, `src/preprocessing.py`, `src/loader.py`, saved v1 results.
**Risks:** v1 and v2 results will differ and must never be mixed (kept in separate directories with versioned file names); recomputing needs the raw data on your machine; any fix to RC1 changes what the dashboard's current evaluation numbers mean, so they will be labelled "v1 (original mapping)" until you decide.

## 11. What I need from you
1. Run the commands in §12 and send the output of `11_environment_fingerprint.py` (to reconcile library versions) and confirm your `src\preprocessing.py` has the same tag maps (the fingerprint prints its hash).
2. Review §3.1 and tell me whether you accept the notebook-consistent stage map as the working hypothesis to implement in Stage 0, or want to inspect the raw tag intervals yourself first.
3. Approve (or amend) the §9.4 decision rule and the proposed primary window design (B).
4. Tell me whether `f14` should be handled as a sensitivity analysis.

## 12. PowerShell commands (project root)

```powershell
cd C:\Users\AYESHA\Downloads\NeuroTwin_full_project

# 1. Backup before anything else
Copy-Item -Recurse . "..\NeuroTwin_full_project_BACKUP_$(Get-Date -Format yyyyMMdd_HHmm)"

# 2. Copy the new files from the overlay zip into the project (adds files only), then:
python scripts\11_environment_fingerprint.py
python scripts\10_model_diagnosis.py        # ~1 minute; writes data\processed\model_investigation\diagnosis\
python -m pytest tests -q
```
Expected: the new tests pass; 3 additional strict-xfail tests document RC1/RC2; nothing in your existing results changes.

## 13. Files added in this phase
`scripts/10_model_diagnosis.py`, `scripts/11_environment_fingerprint.py`, `tests/test_model_investigation_diagnosis.py`, `docs/neurotwin/MODEL_INVESTIGATION_DIAGNOSIS_AND_PLAN.md`, `docs/neurotwin/model_investigation_references.csv`, `data/processed/model_investigation/diagnosis/` (JSON + 4 CSVs). No existing file was modified.
