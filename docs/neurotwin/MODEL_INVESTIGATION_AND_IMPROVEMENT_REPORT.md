# NeuroTwin AI: Model Investigation and Improvement Report

**Project:** MSc Software Systems Project — NeuroTwin AI: An Explainable and Personalized Physiological Digital Twin for Stress Understanding and Adaptive Stress Management  
**Location:** `C:\Users\AYESHA\Downloads\NeuroTwin_full_project`  
**Date:** November 2026  
**Status:** Verification Gate Passed; Decision Rule Programmatically Evaluated; Baseline Retained with Selective Abstention; Aurora UI Integrated.  
**Reference Document:** `docs/neurotwin/MODEL_INVESTIGATION_DIAGNOSIS_AND_PLAN.md`

---

## 1. Executive Summary

This report delivers the comprehensive empirical investigation and algorithmic audit into the physiological stress classification pipeline of NeuroTwin AI, conducted strictly according to the pre-registered diagnosis and investigation plan.

### Core Scientific Findings:
1. **Pipeline and Data Defect Confirmation:**
   - **RC1 (Stage Boundary Desynchronization):** The legacy preprocessing path misaligned protocol stage tags by 1 due to an indexing discrepancy relative to the dataset authors' reference notebook (`Wearable_Dataset.ipynb`). In V1 recordings, tasks and rests for Stroop, TMCT, Opinion, and Subtract were systematically inverted.
   - **RC2 (Heart Rate Timing Lag):** Heart rate recording began exactly 10.0 seconds after EDA recording across 100% of dataset sessions (37/37). Legacy preprocessing applied an unshifted reindex that failed to align HR row $k$ with EDA time $k + 10$ s.
   - **RC3 (Window Duration Artifacts):** Legacy window durations varied from 24 to 853 seconds, creating severe duration confounding (e.g., `temp_std` correlation with $\log(\text{duration})$ of $r = 0.79$, `hr_range` $r = 0.74$).
2. **Corrected Pipeline (v2) Generation:**
   - A notebook-consistent, leakage-free preprocessing pipeline was implemented in a new package (`src/investigation/`).
   - Primary evaluation was standardized on **Variant B**: duration-matched 30-second windows anchored at stage start (263 windows across 35 participants). Sensitivity analyses were conducted on Variant A (full corrected windows), Variant Bend (last 30 seconds), and an f14-split sensitivity dataset.
3. **Model Benchmark & Programmatic Decision Rule §9.4:**
   - Fifteen models and pre-specified candidate improvements were evaluated under identical 35-participant Leave-One-Subject-Out (LOSO) cross-validation with 2,000 participant-cluster bootstrap resamples.
   - **Majority Class Guessing Baseline:** Accuracy = 0.6160 (61.6%).
   - **Existing Logistic Regression (Reference Baseline):** Accuracy = 0.5703, Balanced Accuracy = 0.5114 (95% CI [0.4420, 0.5834]), ROC-AUC = 0.4835 (95% CI [0.3869, 0.5884]), Brier Skill Score = -0.1487.
   - **Best Candidate Model (LogReg + In-Fold Feature Selection, $k=10$):** Accuracy = 0.5932, Balanced Accuracy = 0.5337 (95% CI [0.4528, 0.6175]), ROC-AUC = 0.5119 (95% CI [0.4082, 0.6215]), Brier Skill Score = -0.0777.
   - **Programmatic Decision Rule Outcome:** All candidate models failed Condition (i) (95% bootstrap lower bound for balanced accuracy and AUC must exceed 0.50) and Condition (vi) (Brier skill score relative to prior must be strictly positive).
   - **Final Verdict:** **RETAIN_BASELINE with selective prediction & abstention**. Rather than claiming artificial or brittle gains, the project honestly acknowledges the physiological limitations of consumer wristbands for laboratory stress discrimination.
4. **Selective Prediction & Abstention:**
   - In-fold confidence threshold selection yielded a coverage of 36.5% (96/263 windows).
   - On covered windows, balanced accuracy rose from 0.5114 to 0.6023 (accuracy 0.6354).
   - For all uncertain or noisy windows, the system reports: *"Insufficient physiological evidence to provide a reliable estimate."*
5. **NeuroTwin Aurora UI:**
   - The user interface was redesigned with the "Aurora" design language (Aurora Night dark default, Dawn light mode), achieving verified WCAG AA contrast ($\ge 4.5:1$).
   - Replaced alarm gauges with the Evidence Mist Meter, Breathing Orb, and Three-Layer Provenance architecture (`Measured`, `Estimated`, `Reported`).

---

## 2. Pipeline Audit & Root Causes (RC1–RC9)

The systematic audit of the original pipeline (`scripts/10_model_diagnosis.py` and `scripts/12_audit_stage_boundaries.py`) uncovered nine critical architectural root causes:

| Code | Defect / Mechanism | Empirical Observation in Baseline Audit | Resolution in v2 Investigation Pipeline |
| :--- | :--- | :--- | :--- |
| **RC1** | Stage boundary desynchronization | Reference notebook `Wearable_Dataset.ipynb` prepended EDA start time to raw tag list; legacy project code assumed raw tag index 0 was stage 0. Stroop, TMCT, Opinion, and Subtract tasks and rests were swapped. | Created `src/investigation/stage_maps.py` aligning with notebook cells 25 and 27. Verified notebook-consistent rate: 100% in V1, 94.4% in V2. |
| **RC2** | Heart rate time lag | Empatica E4 HR starts 10.0 s after EDA in 100% of sessions. Legacy `reindex(index - 10)` in `src/preprocessing.py` shifted index labels without moving array values. | Created `src/investigation/alignment.py` explicitly placing HR row $k$ at EDA second $10 + k$, with unit tests verifying synthetic timing shift. |
| **RC3** | Variable window duration confounding | Legacy windows ranged from 24 s to 853 s. Duration strongly correlated with features (`temp_std` $r=0.79$, `hr_range` $r=0.74$). | Primary Design B standardizes all windows to exactly 30 s anchored at stage start. |
| **RC4** | Feature collinearity & duplicate descriptors | Canonical feature table contained redundant descriptors (e.g. `hr_diff_rms` and `hrv_rmssd` correlation $r=0.999$). | Pre-specified duplicate-removal variant evaluated in LOSO benchmark. |
| **RC5** | Skewness in dispersion metrics | Non-normal, heavy-tailed distributions in EDA peaks and accelerometry variances. | Evaluated robust scaling (median/IQR) and log-transforms within training folds only. |
| **RC6** | Class imbalance confounding | Low stress constitutes 61.6% of protocol windows; models predicting the majority class achieved higher raw accuracy than balanced classifiers. | Pre-specified balanced accuracy and macro F1 as primary metrics alongside balanced class weighting. |
| **RC7** | Over-confidence & probability distortion | Raw logistic and SVM outputs produced uncalibrated confidence estimates ($ECE > 0.15$). | Nested Platt calibration evaluated within inner folds; probabilities withheld when Brier skill score $\le 0$. |
| **RC8** | Inter-subject physiological variability | Absolute physiological baselines vary widely across humans (e.g. resting HR 55 vs 85 BPM). | Evaluated test-exclusive baseline-relative normalisation (using a participant's baseline window only to normalise test windows, dropping the baseline from evaluation). |
| **RC9** | Silent handling of subject `f14` | Subject `f14` in V2 cohort experienced an interrupted session split into two recording folders. | Isolated to a documented sensitivity analysis (`stress_features_v2_B_f14_sensitivity.csv`), keeping canonical 35 subjects clean. |

---

## 3. Dataset and Label Findings

- **Cohort Characteristics:** 36 healthy adult volunteers (ages 18–30) recruited across two study cohorts: V1 (18 subjects) and V2 (18 subjects). Subject `f14` was split into two directories; 35 participants yielded complete continuous protocols.
- **Protocol Design:** Laboratory sessions comprised sequential resting baselines and standardized stress induction tasks:
  - *Stroop Color-Word Interference Task* (cognitive stressor)
  - *Trier Mental Challenge Task / TMCT* (mental arithmetic under evaluative stress)
  - *Public Speaking / Opinion Task* (social-evaluative threat)
  - *Cold Pressor / Stress Test* (physiological pain stressor)
- **Ground Truth Target:** Dichotomized self-reported stress score obtained verbally on a 1–10 Likert scale immediately following each protocol stage. Binary threshold $\ge 5.0$ defines the high-stress class (101 high-stress windows, 162 low-stress windows; prevalence = 38.4%).
- **Verification Gate A1 Results:**
  - V1 cohort notebook-consistent stage boundary match: **100.0% (18/18)**.
  - V2 cohort notebook-consistent stage boundary match: **94.4% (17/18)** (f14 split folder).
  - HR start offset: EDA start precedes HR start by **10.0 seconds in 37 of 37 recording directories (100.0%)**.
  - **Gate A1 Passed.**

---

## 4. Sensor and Feature Findings

1. **Duration Standardization:** Standardizing windows to 30 seconds (Design B) successfully decoupled feature values from window duration. Correlation between window duration and `temp_std` dropped from $0.79$ to $0.00$.
2. **SCR Frequency Normalisation:** In variable windows, raw SCR count (`eda_scr_count`) scaled with elapsed time. In `src/investigation/features_v2.py`, `eda_scr_per_min` was added as an additive feature while preserving all 23 canonical features.
3. **Heart Rate Variability Suppression:** Investigation confirmed that frequency-domain HRV (`hrv_lf_hf_ratio`) derived from photoplethysmography (PPG) over short 30-second windows possesses high estimation variance and is held back from user explanations.

---

## 5. Leakage Audit Findings

To guarantee scientific validity, the evaluation framework (`scripts/14_loso_benchmark.py`) enforces strict leakage controls:
- **Participant-Disjoint Folds:** Every evaluation fold holds out all windows for a given participant simultaneously. No participant's windows appear in both training and test partitions.
- **Nested Cross-Validation:** All imputation, robust scaling, feature selection, and probability calibration parameters are fitted exclusively on the $N-1$ training participants.
- **Test-Exclusive Baseline Normalisation:** When evaluating baseline-relative features, the participant's baseline window is used strictly as a normalisation anchor and is dropped from that participant's evaluation test set (preventing identity leakage).
- **Automated Verification:** `tests/test_investigation_leakage.py` systematically inverted labels of held-out participants and confirmed that test estimates remained mathematically unchanged (0.000000 divergence).

---

## 6. Literature Review & Verified References

All citations adhere strictly to verified sources catalogued in `docs/neurotwin/model_investigation_references.csv`:

1. **Cross-Validation & Small-Sample Inference:**
   - **Saeb et al. (2017):** *The need to approximate the use-case in clinical machine learning.* GigaScience 6(5):gix019. [DOI: 10.1093/gigascience/gix019](https://doi.org/10.1093/gigascience/gix019). Demonstrates that record-wise cross-validation overestimates accuracy in biomedical wearables; subject-independent partitioning is mandatory.
   - **Little et al. (2017):** *Using and understanding cross-validation strategies.* GigaScience 6(5):gix020. [DOI: 10.1093/gigascience/gix020](https://doi.org/10.1093/gigascience/gix020). Highlights persistent identity confounding even under subject-wise splits.
   - **Varoquaux (2018):** *Cross-validation failure: small sample sizes lead to large error bars.* NeuroImage 180:68–77. [DOI: 10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061). Shows that with $N \approx 100$ samples, 95% confidence intervals typically span $\pm 10\%$.
2. **Selection Bias & Normalisation:**
   - **Vabalas et al. (2019):** *Machine learning algorithm validation with a limited sample size.* PLoS ONE 14(11):e0224365. [DOI: 10.1371/journal.pone.0224365](https://doi.org/10.1371/journal.pone.0224365). Establishes that feature selection on pooled data introduces substantial optimistic bias; nested CV is essential.
   - **Ambroise & McLachlan (2002):** *Selection bias in gene extraction on the basis of microarray gene-expression data.* PNAS 99(10):6562–6566. [DOI: 10.1073/pnas.102102699](https://doi.org/10.1073/pnas.102102699). Foundational proof that feature filtering must be external to cross-validation loops.
   - **Frontiers in Digital Health (2026):** *Baseline normalization choices inflate classification performance in wearable health monitoring: quantification and mitigation strategies.* [DOI: 10.3389/fdgth.2026.1827279](https://doi.org/10.3389/fdgth.2026.1827279). Quantifies how test-inclusive baseline normalisation inflates accuracy even under subject-independent splits.
3. **Probability Calibration & Selective Prediction:**
   - **Van Calster et al. (2019):** *Calibration: the Achilles heel of predictive analytics.* BMC Med 17:230. [DOI: 10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7). Emphasizes that uncalibrated risk predictions mislead end users and that Brier skill score must beat the training prior.
   - **El-Yaniv & Wiener (2010):** *On the foundations of noise-free selective classification.* J Mach Learn Res 11:1605–1641. [URL](https://jmlr.org/papers/v11/el-yaniv10a.html). Provides theoretical foundation for risk-coverage curves and uncertainty abstention.
   - **Field & Welsh (2007):** *Bootstrapping clustered data.* J R Stat Soc B 69(3):369–390. [DOI: 10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x). Justifies participant-cluster bootstrap resampling for clustered experimental designs.

---

## 7. Baseline Results (v1 Reference)

Under the legacy dataset mapping (`data/features/stress_features.csv`, 263 windows), the reference models evaluated under Leave-One-Subject-Out cross-validation produced:

- **Majority Baseline:** Accuracy = 0.6160, Balanced Accuracy = 0.5000, AUC = 0.5000
- **Existing Logistic Regression (Reference Pipeline):**
  - Accuracy: **0.5894** (58.9%)
  - Balanced Accuracy: **0.5255** (52.6%)
  - ROC-AUC: **0.5273**
  - Macro F1: **0.5140**
  - Brier Skill Score: **Negative** (calibrates to prior)

---

## 8. Model Comparison on Primary Window B & Sensitivities

All models evaluated under 35-subject LOSO cross-validation with 2,000 participant-cluster bootstrap resamples on Primary Window B (duration-matched 30 s):

### Primary Benchmark Table (Design B — 30 s Windows):

| Model / Candidate Improvement | Accuracy | Balanced Accuracy (95% CI) | ROC-AUC (95% CI) | Macro F1 | Brier Skill Score | Verdict vs Majority |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.6160 | 0.5000 [0.5000, 0.5000] | 0.0830 [0.0457, 0.1407] | 0.3812 | -0.0319 | Reference |
| **Dummy (Prior)** | 0.6160 | 0.5000 [0.5000, 0.5000] | 0.0830 [0.0457, 0.1407] | 0.3812 | -0.0319 | Equal |
| **Dummy (Stratified)** | 0.5399 | 0.5017 [0.4624, 0.5435] | 0.5017 [0.4624, 0.5435] | 0.5004 | -0.9449 | Worse (-7.6%) |
| **Existing LogReg (C=1.0)** | **0.5703** | **0.5114 [0.4420, 0.5834]** | **0.4835 [0.3869, 0.5884]** | **0.5011** | **-0.1487** | **Worse (-4.6%)** |
| **L2 LogReg (Tuned C)** | 0.5779 | 0.4990 [0.4288, 0.5707] | 0.4504 [0.3489, 0.5588] | 0.4670 | -0.1387 | Worse (-3.8%) |
| **Linear SVM** | 0.5894 | 0.4970 [0.4522, 0.5524] | 0.4526 [0.3560, 0.5547] | 0.4424 | -0.0582 | Worse (-2.7%) |
| **Random Forest (Fixed)** | 0.5856 | 0.5238 [0.4491, 0.6094] | 0.5121 [0.3999, 0.6233] | 0.5122 | -0.0852 | Worse (-3.0%) |
| **Extra Trees (Fixed)** | 0.6084 | 0.5292 [0.4688, 0.5946] | 0.4491 [0.3164, 0.5932] | 0.5010 | -0.0726 | Worse (-0.8%) |
| **LogReg + Drop Duplicates** | 0.5627 | 0.5034 [0.4291, 0.5791] | 0.4926 [0.3963, 0.5973] | 0.4922 | -0.1315 | Worse (-5.3%) |
| **LogReg + In-Fold Feature Sel (k=10)** | **0.5932** | **0.5337 [0.4528, 0.6175]** | **0.5119 [0.4082, 0.6215]** | **0.5244** | **-0.0777** | **Worse (-2.3%)** |
| **LogReg + Robust Scale & Log Skew** | 0.5627 | 0.5015 [0.4293, 0.5762] | 0.4778 [0.3835, 0.5787] | 0.4888 | -0.1595 | Worse (-5.3%) |
| **LogReg + Balanced Class Weight** | 0.5095 | 0.5012 [0.4150, 0.5869] | 0.5014 [0.4023, 0.6036] | 0.4979 | -0.1852 | Worse (-10.7%) |
| **LogReg + Inner Threshold Selection** | 0.5285 | 0.4812 [0.3903, 0.5784] | 0.4835 [0.3869, 0.5884] | 0.4764 | -0.1487 | Worse (-8.8%) |
| **LogReg + Baseline Relative (Test-Excl)** | 0.5044 | 0.4500 [0.3612, 0.5394] | 0.3745 [0.2707, 0.4899] | 0.4368 | -0.3872 | Worse (-10.5%) |
| **LogReg + Nested Platt Calibration** | 0.6160 | 0.5000 [0.5000, 0.5000] | 0.2619 [0.1798, 0.3603] | 0.3812 | -0.0436 | Collapses to Prior |
| **RBF SVM** | 0.5970 | 0.5013 [0.4746, 0.5338] | 0.4103 [0.3062, 0.5271] | 0.4407 | -0.0727 | Worse (-1.9%) |

### Sensitivity Analyses:
- **Variant A (Full Corrected Windows, 263 rows):**
  - LogReg: Accuracy = **0.6236**, Balanced Accuracy = **0.5751** (95% CI [0.4890, 0.6648]), ROC-AUC = **0.5563** (95% CI [0.4402, 0.6739]), Brier Skill = -0.0785.
  - While full windows show marginal elevation, the 95% bootstrap lower bounds ($0.4890$ and $0.4402$) remain strictly below 0.50, and duration artifacts remain present.
- **Variant Bend (Last 30 s of Stage, 263 rows):**
  - LogReg: Accuracy = **0.5703**, Balanced Accuracy = **0.5002**, ROC-AUC = **0.4564**, Brier Skill = -0.1728.
- **Sensitivity f14 (f14 split folder included, 270 rows, 36 subjects):**
  - LogReg: Accuracy = **0.5667**, Balanced Accuracy = **0.5075**, ROC-AUC = **0.4866**, Brier Skill = -0.1383.

---

## 9. Calibration Analysis

- **Raw Model Brier Score:** 0.2717 (LogReg) vs Training Prior Brier Score of **0.2366**.
- **Brier Skill Score:** $\text{BSS} = 1 - \frac{0.2717}{0.2366} = -0.1487$.
- **Nested Platt Calibrator Outcome:** The Platt scaling intercept shifted predicted probabilities toward the sample prevalence ($0.384$), shrinking prediction spread and collapsing to the prior baseline without improving discrimination (AUC dropped to 0.2619).
- **Policy Enforcement:** Because the empirical Brier skill score is negative on held-out participants, the project's strict probability policy is activated: **numeric probability percentages are completely withheld from the user interface**, displaying only ordinal evidence levels (*Limited Evidence / Insufficient Evidence*).

---

## 10. Participant-Level Analysis

Participant-level evaluation reveals strong inter-subject heterogeneity:
- **Per-Participant Accuracy Distribution:** Mean = $0.5750$, Median = $0.6250$, SD = $0.2880$.
- **Participant Comparison Against Majority Guessing:**
  - Better than majority baseline: **2 participants**
  - Equal to majority baseline: **9 participants**
  - Worse than majority baseline: **24 participants**
- **Per-Participant ROC-AUC (Evaluated on the 22 participants with both classes present):**
  - Mean per-subject AUC: **0.5619**, Median: **0.5917**.
  - Participants with $\text{AUC} > 0.50$: **13 / 22** ($59.1\%$).
  - One-sided Binomial sign test against chance ($H_0: p \le 0.5$): $p = 0.2617$.
  - The participant-level advantage is not statistically significant ($p > 0.05$).

---

## 11. Improvements Implemented

1. **Notebook-Consistent Preprocessing Pipeline (`src/investigation/`):**
   - Corrected stage interval mapping (`stage_maps.py`).
   - Timing-aligned heart rate cleaning (`alignment.py`).
   - Duration-standardized window extraction (`windows.py`).
   - Non-destructive feature extensions (`features_v2.py`).
2. **Selective Prediction & Quality Abstention (`scripts/15_selective_prediction.py`):**
   - Implemented abstention filters based on signal coverage ($< 80\%$), missing critical groups (HR, EDA), motion saturation (`acc_mean` or `acc_max` $> 95$th percentile), and low confidence ($|p - 0.5| < 0.230$).
   - Coverage: $36.5\%$; selective balanced accuracy: $0.6023$; fallback message: *"Insufficient physiological evidence to provide a reliable estimate."*
3. **Estimator Integration Fixes (`src/neurotwin/estimator.py`):**
   - **Fix (a):** Movement and temperature bounds computed strictly from training participants during out-of-fold estimation (`_stats_for_subject`).
   - **Fix (b):** Metadata row alignment uses filtered DataFrame indices, preventing index misalignment.
   - **Fix (c):** Documented evidence ceiling derived from whole-sample audit.
4. **Aurora UI Design System (`dashboard/theme_aurora.py`, `dashboard/components_aurora.py`):**
   - Breathing Orb, Journey Ribbon, Three-Layer provenance cards, Evidence Mist meter, and WCAG AA compliance.

---

## 12. Improvements Rejected and Why

Under the pre-registered decision rule (§9.4), candidate algorithmic modifications were formally evaluated and rejected:

1. **Hyper-parameter Tuning ($C \in \{0.01, 0.1, 1, 10\}$):**
   - *Result:* Balanced accuracy dropped from 0.5114 to 0.4990; AUC dropped from 0.4835 to 0.4504. Regularization caused shrinkage toward majority class without improving discrimination.
2. **Tree Ensembles (Random Forest & Extra Trees):**
   - *Result:* RF achieved balanced accuracy 0.5238 (AUC 0.5121); Extra Trees achieved 0.5292 (AUC 0.4491). Both had negative Brier skill score ($-0.0852$ and $-0.0726$) and 95% bootstrap lower bounds extending down to $0.4491$ and $0.3164$.
3. **Class Weighting (`class_weight='balanced'`):**
   - *Result:* Overall accuracy collapsed from 0.5703 to 0.5095 (worse than tossing a coin relative to majority baseline).
4. **Threshold Tuning (Inner GroupKFold):**
   - *Result:* Balanced accuracy fell to 0.4812 due to threshold overfitting on small inner folds.
5. **Test-Exclusive Baseline-Relative Normalisation:**
   - *Result:* Balanced accuracy plummeted to 0.4500 (AUC 0.3745). Standardizing by a single 30-second resting baseline introduces severe ratio noise in short observational windows.

---

## 13. Final Recommendation

**Formal Verdict: RETAIN BASELINE WITH SELECTIVE ABSTENTION.**

Scientific integrity requires reporting that wristband physiological features alone are insufficient to reliably detect acute laboratory stress in this cohort under Leave-One-Subject-Out validation. 

Rather than adopting a complex, overfitted, or uncalibrated classifier, NeuroTwin retains the transparent logistic regression baseline with explicit selective abstention:
- When biosignals are ambiguous or uncalibrated, the twin pauses its estimate.
- No probability percentage is ever displayed.
- The interface emphasizes the person's own reported voice (`Reported` layer) over automated inferences.

---

## 14. Scientific Limitations

1. **Sample Size:** 35 young, healthy participants (ages 18–30) observed in a single laboratory protocol. Findings cannot be generalized to clinical populations, older adults, or chronic stress.
2. **Ground Truth Construct:** Target is a 1–10 post-stage verbal rating dichotomized at 5.0. It reflects immediate cognitive appraisal, not continuous cortisol, biochemical markers, or long-term psychiatric status.
3. **Sensor Form Factor:** Wrist-worn photoplethysmography and skin conductance are susceptible to motion artifacts, thermal fluctuations, and wrist placement pressure.
4. **Software Scope:** The interactive digital twin and recommendation components are verified as functional software, not validated clinical therapies.

---

## 15. PowerShell Reproduction Commands

To reproduce the entire investigation from a fresh PowerShell environment:

```powershell
# 1. Configure environment path
$env:PYTHONPATH = "C:\big data prj\.venv\Lib\site-packages;C:\Users\AYESHA\AppData\Local\Programs\Python\Python313\Lib\site-packages;."
$PYTHON = "C:\Users\AYESHA\AppData\Local\Programs\Python\Python313\python.exe"

# 2. Run environment fingerprint & model diagnosis
& $PYTHON scripts/11_environment_fingerprint.py
& $PYTHON scripts/10_model_diagnosis.py

# 3. Execute Gate A1 Stage Boundary Audit
& $PYTHON scripts/12_audit_stage_boundaries.py

# 4. Build v2 Feature Tables (Variants A, B, Bend, f14)
& $PYTHON scripts/13_build_features_v2.py
& $PYTHON scripts/13_build_features_v2.py --include-f14

# 5. Run LOSO Benchmark & Decision Rule Evaluation (2,000 bootstrap draws)
& $PYTHON scripts/14_loso_benchmark.py

# 6. Run Selective Prediction Risk-Coverage Analysis
& $PYTHON scripts/15_selective_prediction.py

# 7. Generate UI Visual Evidence Artifacts
& $PYTHON scripts/16_generate_ui_evidence.py

# 8. Execute Complete Pytest Test Suite
& $PYTHON -m pytest tests -v
```

---

## 16. Complete Changed-File List

### Protected Files (Verified 100% Byte-Identical to Backup):
- `src/models.py` (MD5: `91EE0DD34B138EF218EAAC4A0C97039C` — Untouched)
- `src/features.py` (MD5: `DF5C02FACAA8C904862F66A1C835AA9C` — Untouched)
- `src/preprocessing.py` (MD5: `B424B80A8FF3DA102AD0621123EE542D` — Untouched)
- `src/loader.py` (MD5: `538B6B1D6B87E46BBADB160E243D13C1` — Untouched)
- `data/features/stress_features.csv` (Untouched)
- `data/processed/model_comparison_scores.csv` (Untouched)
- `data/processed/loso_results.csv` (Untouched)
- `data/processed/feature_importance.csv` (Untouched)

### New Workstream A Pipeline & Investigation Files:
- `src/investigation/__init__.py`
- `src/investigation/stage_maps.py`
- `src/investigation/alignment.py`
- `src/investigation/windows.py`
- `src/investigation/features_v2.py`
- `scripts/12_audit_stage_boundaries.py`
- `scripts/13_build_features_v2.py`
- `scripts/14_loso_benchmark.py`
- `scripts/15_selective_prediction.py`
- `data/features_v2/stress_features_v2_A.csv`
- `data/features_v2/stress_features_v2_B.csv`
- `data/features_v2/stress_features_v2_Bend.csv`
- `data/features_v2/stress_features_v2_B_f14_sensitivity.csv`
- `data/processed/model_investigation/stage_audit/*`
- `data/processed/model_investigation/v2/*`
- `data/processed/model_investigation/v2/selective_prediction/*`

### Updated Integration & Support Files (per A6):
- `src/neurotwin/estimator.py` (Implemented fixes a, b, c)
- `src/neurotwin/ui.py` (Added v1 vs v2 transparency to evaluation page)

### New Workstream B UI Files:
- `dashboard/theme_aurora.py`
- `dashboard/components_aurora.py`
- `dashboard/pages/p01_home.py`
- `dashboard/pages/p02_signals.py`
- `dashboard/pages/p03_questions.py`
- `dashboard/pages/p04_result.py`
- `dashboard/pages/p04_digital_twin.py`
- `dashboard/pages/p05_interventions.py`
- `dashboard/pages/p06_evaluation.py`
- `dashboard/pages/p07_about.py`
- `dashboard/pages/__init__.py`
- `dashboard/app.py`
- `scripts/16_generate_ui_evidence.py`
- `reports/figures/ui/*`

### New Automated Unit Test Suites:
- `tests/test_investigation_alignment.py`
- `tests/test_investigation_features.py`
- `tests/test_investigation_leakage.py`
- `tests/test_investigation_decision_rule.py`
- `tests/test_investigation_estimator_fixes.py`
- `tests/test_neurotwin_aurora_ui.py`

