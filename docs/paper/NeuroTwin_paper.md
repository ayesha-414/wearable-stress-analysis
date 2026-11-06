# NeuroTwin AI: A Provenance-Aware Physiological Digital Shadow for Acute Stress Estimation and Multimodal Quality-of-Life Integration

**Ayesha et al.**  
*Department of Computer Science & Cognitive Systems Engineering*  
*Project NeuroTwin Research Group*  
November 2026

---

## Abstract

Wearable physiological sensing offers the promise of non-invasive, continuous stress assessment, yet existing machine learning prototypes frequently suffer from cross-subject data leakage, uncalibrated probability estimates, opaque feature definitions, and conflation of momentary autonomic arousal with chronic psychological distress. In this paper, we present **NeuroTwin AI**, a research-grade, provenance-aware physiological digital shadow architecture that couples wearable biosignal analysis with psychometric quality-of-life evaluation. The system strictly bifurcates its state representations into **Measured** (raw Empatica E4 sensor channels: EDA, HR, BVP, TEMP, ACC), **Estimated** (calibrated machine learning stress inferences with uncertainty-driven abstention), and **Reported** (subjective self-appraisals: PSS-10 and WHO-5 well-being indices) layers. We extract 23 physiological features across cardiac, autonomic electrodermal, peripheral thermal, and accelerometric modalities, documenting the exact mathematical formulation, physiological mechanism, evidence validity classification, and known instrumentation limitations for each descriptor. 

Under rigorous Leave-One-Subject-Out (LOSO) cross-validation across 35 participants ($N=263$ protocol segments) from the PhysioNet Wearable Device Dataset, our logistic regression pipeline achieves an honest pooled accuracy of $0.570$ ($\text{AUC} = 0.539$, Macro-F1 $= 0.557$) against a majority-class baseline of $0.616$. We demonstrate that post-hoc Platt recalibration reduces Expected Calibration Error (ECE) from $0.179$ to $0.052$, though out-of-fold Brier scores ($0.251$) do not surpass the uninformative training prior ($0.244$), triggering an automated safety abstention policy that suppresses raw pseudo-probabilities in favor of ordinal evidence tiers. Furthermore, we integrate a confirmatory one-factor measurement model of the Perceived Stress Scale (PSS-10, $\alpha = 0.865$) and a conceptual Structural Equation Model (SEM) bridging autonomic biosignals, perceived stress, and health-related quality of life. The complete framework runs locally via lightweight SQLite and Streamlit interfaces, embodying non-diagnostic, supportive communication protocols and immediate crisis resources.

**Index Terms**—Wearable computing, physiological stress detection, digital twin, explainable AI, heart rate variability (HRV), electrodermal activity (EDA), probability calibration, psychometrics, PSS-10, WHO-5.

---

## I. Introduction & Problem Statement

Acute psychological and environmental stress mobilizes the autonomic nervous system (ANS) through sympathetic adreno-medullary activation and parasympathetic vagal withdrawal. These homeostatic perturbations produce measurable shifts in peripheral biosignals, including heart rate acceleration, reduced beat-to-beat variability, elevated skin conductance, peripheral cutaneous vasoconstriction, and postural immobility. The emergence of research-grade wearable multi-sensor wristbands—such as the Empatica E4—has catalyzed extensive interest in automating stress state detection outside controlled clinical laboratories.

However, the contemporary affective computing literature reveals several persistent structural deficiencies:
1. **Inflated Laboratory Claims vs. Generalizability Failure:** Numerous studies report classification accuracies exceeding 90% on benchmark datasets such as WESAD (Schmidt et al., 2018). Yet, these figures frequently depend on random cross-validation splits that bleed intra-subject temporal dependencies between training and test sets. When evaluated under strict subject-independent protocols (Leave-One-Subject-Out; LOSO), reported performance declines sharply toward the majority baseline (Vos et al.; Sharma et al., 2026).
2. **Probability Distortion and Lack of Abstention:** Off-the-shelf supervised classifiers (e.g., Random Forests, Support Vector Machines) yield miscalibrated output scores that are routinely misinterpreted as true Bayesian posterior probabilities of stress (Niculescu-Mizil & Caruana, 2005). Systems rarely implement principled criteria to abstain from predicting when sensor coverage is compromised or model uncertainty is high.
3. **Formula Opacity and Metric Conflation:** Many implementations employ heuristic approximations (e.g., computing root-mean-square of successive 1 Hz heart rate differences and incorrectly labelling it as Task Force RMSSD, or computing periodogram averages rather than integrating band power for LF/HF) without documenting the divergence from physiological standards.
4. **Epistemic Conflation of Sensors and Lived Experience:** Engineering prototypes often treat momentary wearable predictions as an objective ground truth for whether an individual "is stressed," ignoring the critical distinction between acute autonomic physiological reactivity and retrospective psychological appraisal (such as the Perceived Stress Scale; Cohen et al., 1983).

To address these challenges, we present **NeuroTwin AI**, a research prototype and personal digital shadow (NASEM, 2024; Tudor et al., 2025) engineered to maintain strict provenance separation, transparent mathematical validation, honest evaluation against majority baselines, and seamless psychometric integration with health-related quality of life.

---

## II. Literature Review & Theoretical Foundations

### A. Wearable Biosignal Stress Detection
Wearable stress analysis relies on monitoring somatic markers of ANS reactivity. Photoplethysmography (PPG) sensors optically capture pulse waveforms (Blood Volume Pulse; BVP), yielding inter-beat intervals (IBI) from which heart rate variability (HRV) metrics are computed (Task Force, 1996; Shaffer & Ginsberg, 2017). Concurrently, skin conductance electrodes measure electrodermal activity (EDA) generated by sympathetic cholinergic innervation of eccrine sweat glands (Boucsein et al., 2012). While laboratory benchmarks like WESAD (Schmidt et al., 2018) established multi-sensor feasibility, systematic reviews by Vos et al. and Sharma et al. (2026) emphasize that cross-subject generalizability remains unresolved due to substantial inter-individual variability in baseline physiological tone, skin thickness, and psychological reactivity.

### B. Psychometrics: PSS-10 and One-Factor Latent Modeling
Somatic sensor measurements capture transient autonomic arousal over seconds or minutes, but clinical and psychological health is profoundly governed by perceived, chronic stress. The 10-item Perceived Stress Scale (PSS-10; Cohen, Kamarck & Mermelstein, 1983) is the most widely validated psychometric instrument for assessing the degree to which situations in one's life are appraised as unpredictable, uncontrollable, and overloading over the past 30 days. Psychometric evaluation by Taylor (2015) using confirmatory factor analysis (CFA) demonstrated that PSS-10 possesses robust internal consistency ($\alpha > 0.85$) and is structurally well-represented by a single general latent perceived stress factor, supporting an aggregated ordinal score (0–40) categorized into Low (0–13), Moderate (14–26), and High (27–40) stress tiers.

### C. Health-Related Quality of Life (WHO-5 & WHOQOL)
The World Health Organization-Five Well-Being Index (WHO-5; Topp et al., 2015) measures subjective psychological well-being over a two-week recall period on a standardized 0–100 scale. In psychosomatic medicine, prolonged perceived stress directly undermines vitality, sleep restoration, and emotional spirits. Integrating WHO-5 and the 4-domain WHOQOL-BREF framework (physical, psychological, social relationships, environment; WHOQOL Group, 1998) provides a distal outcome anchor for physiological monitoring.

### D. Structural Equation Modeling (SEM) Framework
Structural Equation Modeling (Bollen, 1989; Kline, 2016) provides a rigorous framework for specifying multivariate path relationships between observed physiological indicators, estimated latent stress states, self-reported cognitive appraisals, and ultimate quality-of-life outcomes. We incorporate this structural perspective into NeuroTwin to clarify the causal and correlational hierarchy of digital phenotyping.

### E. Explainable AI and Digital Twin Nomenclature
Attribution techniques such as SHAP (Lundberg & Lee, 2017) and standardized linear coefficients decompose model predictions into feature-level contributions. In wearable analytics, user evaluation of such explanations remains scarce (Abdelaal et al., 2024). Furthermore, adhering to the National Academies of Sciences, Engineering, and Medicine (NASEM, 2024) definitions, NeuroTwin is characterized scientifically as a **personal digital shadow with an adaptive recommendation loop** rather than a fully bidirectional, autonomous digital twin (Tudor et al., 2025).

---

## III. System Architecture & Provenance Separation

The architectural core of NeuroTwin AI enforces strict database-level constraints across three fundamental information layers:

```
  +-------------------------------------------------------------------------+
  |                             MEASURED LAYER                              |
  |  Raw Empatica E4 Channels: EDA (4Hz), HR (1Hz), BVP (64Hz), IBI,       |
  |  TEMP (4Hz), ACC (32Hz). Data Quality Veto & Motion Coverage Gate.      |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                             ESTIMATED LAYER                             |
  |  Preprocessed 23 Features -> Standardized ML Pipelines (LogReg, SVM,   |
  |  RF, ET) -> LOSO Validation -> Platt Calibrator -> Abstention Filter.   |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                             REPORTED LAYER                              |
  |  PSS-10 (0-40), WHO-5 (0-100), Momentary Mood & Stressor Attribution.  |
  |  Persisted in SQLite exclusively under explicit user consent.           |
  +-------------------------------------------------------------------------+
```

1. **Measured Layer:** Sensor-derived raw signals and deterministic summaries. Carries `provenance = 'sensor'` in the database schema. Includes missing-data checks, sensor-contact flags, and accelerometric motion thresholding ($> 95\text{th}$ percentile).
2. **Estimated Layer:** Outputs of mathematical pipelines and statistical classifiers. Carries `provenance = 'model'`. Outputs are explicitly qualified by evidence confidence levels and calibration gates. Model inferences never write into the user's reported state.
3. **Reported Layer:** Subjective, user-provided questionnaire responses, momentary check-ins, and coping feedback. Carries `provenance = 'user'`. Writes are strictly contingent on user consent flags (`store_checkins`, `store_stressors`, `store_feedback`) managed through local SQLite storage.

---

## IV. Dataset & Preprocessing Protocol

We utilize the **PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (version 1.0.1)** (Hongn et al., 2025; PhysioNet doi:10.13026/he0v-tf17). 

### A. Experimental Protocol
The dataset comprises physiological recordings from 36 healthy adult participants (aged 18–30) wearing an Empatica E4 wristband during a structured stress-induction protocol (Version 1: 12 button tags; Version 2: 9 button tags). Tasks included baseline resting conditions, the Stroop Color-Word interference test, mental arithmetic subtraction tasks, rapid opinion presentation, and seated recovery stages. Immediately before and after each task, participants recorded verbal self-reported stress levels on a 1–10 Likert scale.

### B. Preprocessing & Alignment
- **Heart Rate (HR):** Resampled to 1 Hz, checked for continuity, and clipped to $[30, 200]\text{ bpm}$.
- **Electrodermal Activity (EDA):** Resampled to 1 Hz, clipped to $[0, 60]\ \mu\text{S}$.
- **Skin Temperature (TEMP):** 4 Hz thermistor recordings averaged to 1 Hz.
- **Inter-Beat Intervals (IBI):** Peak-detected intervals in seconds from wrist PPG.
- **Acceleration (ACC):** 3-axis readings (32 Hz, unit $1/64\text{ g}$). Filtered using a 32-sample block moving-average of maximum delta excursions smoothed with a 0.1 exponential moving average (EMA).
- **Segmentation:** Signals were segmented into discrete task stage windows matching the protocol event tags. After quality exclusions, 35 subjects and $N=263$ feature windows were retained. Binary ground-truth labels were established at self-reported stress $\ge 5.0$ (High Stress: $n=101, \approx 38.4\%$; Low Stress: $n=162, \approx 61.6\%$).

---

## V. Feature Engineering & Mathematical Formulations

NeuroTwin extracts exactly 23 physiological features across five functional domains. In compliance with open-science transparency, every formula, physiological justification, evidence classification, and known limitation is explicitly documented below.

### A. Heart Rate Features (1 Hz Series)
1. **Mean Heart Rate (`hr_mean`):**
   $$\mu_{\text{HR}} = \frac{1}{N} \sum_{i=1}^{N} \text{HR}_i \quad [\text{bpm}]$$
   *Physiological Rationale:* Sympathetic stimulation and vagal withdrawal elevate sinoatrial pacing.  
   *Evidence Level:* Custom derived.  
   *Limitation:* Derived from 1 Hz PPG, not ECG R-peak timing.

2. **Heart Rate Standard Deviation (`hr_std`):**
   $$\sigma_{\text{HR}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\text{HR}_i - \mu_{\text{HR}})^2} \quad [\text{bpm}]$$
   *Physiological Rationale:* Quantifies overall dispersion of heart rate across the stage.  
   *Evidence Level:* Custom derived.  
   *Limitation:* Represents 1 Hz volatility; not to be confused with beat-to-beat SDNN.

3. **Minimum & Maximum Heart Rate (`hr_min`, `hr_max`):**
   $$\text{HR}_{\min} = \min_i (\text{HR}_i), \quad \text{HR}_{\max} = \max_i (\text{HR}_i) \quad [\text{bpm}]$$
   *Physiological Rationale:* Captures dynamic cardiac bounds and peak adrenergic tone.  
   *Evidence Level:* Custom derived.  
   *Limitation:* Susceptible to transient sensor movement spikes and clipping.

4. **Heart Rate Range (`hr_range`):**
   $$\Delta_{\text{HR}} = \text{HR}_{\max} - \text{HR}_{\min} \quad [\text{bpm}]$$

5. **Successive HR Difference RMS (`hr_diff_rms`):**
   $$\text{RMS}_{\Delta\text{HR}} = \sqrt{\frac{1}{K} \sum_{i=1}^{K} (\text{HR}_{i+1} - \text{HR}_i)^2} \quad [\text{bpm}]$$
   *Physiological Rationale:* Measures second-to-second heart rate volatility.  
   *Evidence Level:* Custom derived.  
   *Limitation:* Stored in legacy CSVs as `hr_rmssd_approx`. **It is not Task Force RMSSD**; it evaluates 1 Hz bpm differences rather than millisecond inter-beat intervals.

### B. Electrodermal Activity Features (1 Hz Series)
6. **Mean Skin Conductance (`eda_mean`):**
   $$\mu_{\text{EDA}} = \frac{1}{N} \sum_{i=1}^{N} \text{EDA}_i \quad [\mu\text{S}]$$
   *Physiological Rationale:* Sudomotor nerve burst activity increases eccrine sweat secretion via sympathetic cholinergic pathways (Boucsein et al., 2012).  
   *Evidence Level:* Custom derived.

7. **EDA Standard Deviation (`eda_std`):**
   $$\sigma_{\text{EDA}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\text{EDA}_i - \mu_{\text{EDA}})^2} \quad [\mu\text{S}]$$

8. **EDA Trend Slope (`eda_slope`):**
   $$\beta_{\text{EDA}} = \frac{\sum_{i=0}^{N-1} (i - \bar{i})(\text{EDA}_i - \bar{\text{EDA}})}{\sum_{i=0}^{N-1} (i - \bar{i})^2} \quad [\mu\text{S/s}]$$
   *Physiological Rationale:* Positive slope captures progressive sympathetic recruitment during acute cognitive demands.  
   *Evidence Level:* Custom derived (OLS against sample index).

9. **EDA Dynamic Range (`eda_range`):**
   $$\Delta_{\text{EDA}} = \text{EDA}_{\max} - \text{EDA}_{\min} \quad [\mu\text{S}]$$

10. **Slow Skin Conductance Level Proxy (`eda_scl_mean`):**
    $$\text{SCL}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{|W_i|} \sum_{j \in W_i} \text{EDA}_j \right), \quad W_i = [i-30, i+29] \quad [\mu\text{S}]$$
    *Evidence Level:* Heuristic approximation.  
    *Limitation:* Centered 60-sample rolling mean; collapses to global mean on short stages; does not execute convex optimization (cvxEDA; Greco et al., 2016).

11. **Skin Conductance Response Count Proxy (`eda_scr_count`):**
    $$\text{SCR}_{\text{count}} = \sum_{i=1}^{N-1} \mathbb{I}\left[(\text{EDA}_i - \text{tonic}_i > 0.02) \land (\text{EDA}_{i-1} - \text{tonic}_{i-1} \le 0.02)\right]$$
    *Evidence Level:* Heuristic approximation.  
    *Limitation:* Fixed $0.02\ \mu\text{S}$ threshold without peak morphology or rise-time filtering.

### C. Skin Temperature Features
12–14. **Mean, Std, and Slope (`temp_mean`, `temp_std`, `temp_slope`):**
$$\mu_{\text{TEMP}} = \frac{1}{N}\sum \text{TEMP}_i, \quad \sigma_{\text{TEMP}} = \sqrt{\frac{\sum (\text{TEMP}_i - \mu)^2}{N-1}}, \quad \beta_{\text{TEMP}} = \frac{\sum (i-\bar{i})(\text{TEMP}_i - \bar{T})}{\sum (i-\bar{i})^2}$$
*Physiological Rationale:* Cutaneous peripheral vasoconstriction during acute stress reduces skin surface temperature at the wrist.  
*Evidence Level:* Custom derived.  
*Limitation:* Strongly influenced by ambient thermal environment and clothing.

### D. Movement / Accelerometer Features
15–17. **Activity Index Mean, Std, and Max (`acc_mean`, `acc_std`, `acc_max`):**
$$\text{ACC}_{\text{EMA}}[k] = 0.9 \cdot \text{ACC}_{\text{EMA}}[k-1] + 0.1 \cdot \left(\frac{1}{32}\sum_{m=1}^{32} \max(|\Delta x_m|, |\Delta y_m|, |\Delta z_m|)\right)$$
*Physiological Rationale:* Monitors physical activity to prevent misattributing somatic exercise arousal to psychological stress.  
*Evidence Level:* Custom derived.  
*Limitation:* Block maximum delta index (1/64 g units), not Euclidean Norm Minus One (ENMO).

### E. Heart Rate Variability Features (IBI Series)
18. **Mean Inter-Beat Interval (`hrv_mean_ibi`):**
    $$\overline{\text{IBI}} = \frac{1}{N} \sum_{i=1}^{N} \text{IBI}_i \quad [\text{seconds}]$$
    *Evidence Level:* Matches standard.  
    *Limitation:* Unit is seconds; clinical standards quote milliseconds.

19. **SDNN (`hrv_sdnn`):**
    $$\text{SDNN} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\text{IBI}_i^{\text{ms}} - \overline{\text{IBI}}^{\text{ms}})^2} \quad [\text{ms}]$$
    *Evidence Level:* Matches standard (Task Force, 1996).  
    *Limitation:* Standard calls for 5-minute stationary recordings; protocol stages are $\approx 30\text{ s}$.

20. **RMSSD (`hrv_rmssd`):**
    $$\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (\text{IBI}_{i+1}^{\text{ms}} - \text{IBI}_i^{\text{ms}})^2} \quad [\text{ms}]$$
    *Physiological Rationale:* Primary index of parasympathetic (vagal) cardiac control. Stress induces rapid vagal withdrawal, lowering RMSSD.  
    *Evidence Level:* Matches standard (Task Force, 1996).

21. **pNN50 (`hrv_pnn50`):**
    $$\text{pNN50} = \frac{\sum_{i=1}^{N-1} \mathbb{I}\left[|\text{IBI}_{i+1}^{\text{ms}} - \text{IBI}_i^{\text{ms}}| > 50\text{ ms}\right]}{N-1}$$
    *Evidence Level:* Matches standard. Strict $> 50\text{ ms}$ threshold and $N-1$ divisor. Output is proportion $[0, 1]$.

22. **LF/HF Ratio (`hrv_lf_hf_ratio`):**
    $$\text{Ratio} = \frac{\frac{1}{|K_{\text{LF}}|}\sum_{k \in K_{\text{LF}}} P_{\text{LS}}(f_k)}{\frac{1}{|K_{\text{HF}}|}\sum_{m \in K_{\text{HF}}} P_{\text{LS}}(f_m)}, \quad \text{LF}\in[0.04, 0.15]\text{ Hz}, \text{HF}\in(0.15, 0.40]\text{ Hz}$$
    *Evidence Level:* **Uncertain / Discrepancy Documented.**  
    *Limitation:* Evaluates normalised Lomb-Scargle periodogram on mean-centered IBI. Slices contain 138 LF points and 315 HF points. Averages ordinates rather than integrating band power, which numerically inflates the calculated ratio by $N_{\text{HF}}/N_{\text{LF}} = 315/138 \approx 2.283$. Furthermore, 30-second windows are inadequate for reliable LF resolution (Shaffer & Ginsberg, 2017). Consequently, **this feature is retained in classifier models but strictly suppressed from end-user explanations**.

---

## VI. Machine Learning Methodology & Evaluation

### A. Evaluated Classifiers
We benchmarked five classifier pipelines, all incorporating median missing-value imputation and feature standardization:
1. **Logistic Regression (LogReg):** $L_2$-regularized linear model ($C=1.0$).
2. **Support Vector Machine (SVM-RBF):** Radial basis kernel ($C=1.0, \gamma=\text{scale}$).
3. **Random Forest (RF):** 100 estimators, balanced class weighting.
4. **Extra Trees (ET):** 100 randomized decision trees.
5. **Majority Baseline (Zero-R):** Always predicts the empirical majority class (Low Stress).

### B. Validation Protocol
Evaluation was executed under complete **Leave-One-Subject-Out (LOSO)** cross-validation across all 35 subjects ($K=35$ folds). In each fold, all windows of participant $s$ were strictly held out while models were trained on the remaining 34 participants. No participant-level data leaked between training and testing folds.

### C. Calibration & Abstention Policy
To assess probability reliability, Platt scaling (logistic calibration) was fitted exclusively on out-of-fold predictions inside each training fold:
$$\text{logit}(p) = \ln\left(\frac{p}{1-p}\right), \quad P(\text{Stress}=1) = \frac{1}{1 + \exp(- (a \cdot \text{logit}(p) + b))}$$
Calibration quality was audited via Expected Calibration Error (ECE) across 5 equal-width bins and Brier score:
$$\text{ECE} = \sum_{b=1}^{B} \frac{|I_b|}{N} \left| \text{acc}(I_b) - \text{conf}(I_b) \right|, \quad \text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2$$
**Abstention Gate:** A calibrated probability may only be displayed to a user if:
1. Recalibrated Brier score is demonstrably lower than the uninformative training prior ($\text{Brier}_{\text{cal}} < \text{Brier}_{\text{prior}}$), and
2. Recalibrated ECE $\le 0.10$.
Otherwise, numeric probabilities are withheld, and only an ordinal evidence tier (Limited / Insufficient) is shown.

---

## VII. Experimental Results

Table I presents the empirical results extracted from `model_comparison_scores.csv` under LOSO validation.

### TABLE I: Leave-One-Subject-Out Classifier Comparison ($N=263$ windows, 35 subjects)
| Classifier | Accuracy | Macro F1 | Balanced Acc | High-Stress Precision | High-Stress Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.5703** | **0.5568** | **0.5599** | **0.4483** | **0.5149** |
| **SVM (RBF)** | 0.5057 | 0.4795 | 0.4795 | 0.3592 | 0.3663 |
| **Extra Trees** | 0.4981 | 0.3940 | 0.4248 | 0.2075 | 0.1089 |
| **Random Forest** | 0.4791 | 0.4418 | 0.4429 | 0.3085 | 0.2871 |
| **Majority Baseline** | **0.6160** | 0.3812 | 0.5000 | 0.0000 | 0.0000 |

### TABLE II: Calibration Audit for Primary Estimator (`neurotwin_estimator_report.json`)
| Metric | Raw Value | Post-Recalibration | Reference Benchmark |
| :--- | :---: | :---: | :---: |
| **Pooled Accuracy** | 0.5703 | 0.5703 | Majority: 0.6160 |
| **Pooled ROC AUC** | 0.5393 | 0.5393 | Chance: 0.5000 |
| **Brier Score** | 0.2761 | 0.2507 | Training Prior: **0.2441** |
| **Expected Calibration Error (ECE)** | 0.1792 | **0.0524** | Threshold: $\le 0.10$ |

As shown in Table II, while Platt calibration successfully compressed ECE from 17.9% to 5.2%, the recalibrated Brier score ($0.2507$) failed to outperform the naive prior ($0.2441$). Consequently, **NeuroTwin's safety gate engaged**, correctly abstaining from presenting deceptive numeric confidence percentages to users.

---

## VIII. Multimodal Psychometric Integration & Agreement Analysis

To bridge biological state sensing and clinical psychological appraisal, NeuroTwin incorporates the Reported psychometric layer:

### A. PSS-10 Confirmatory Factor Model
Scoring of the PSS-10 reverses items 4, 5, 7, and 8:
$$\text{Score}_{\text{PSS}} = \sum_{i \in \{1,2,3,6,9,10\}} x_i + \sum_{j \in \{4,5,7,8\}} (4 - x_j) \quad \in [0, 40]$$
Confirmatory factor analysis based on normative psychometric cohorts ($N=2,387$; Taylor, 2015) indicates strong item loadings across all 10 indicators ($\lambda \in [0.54, 0.78]$, $\text{CFI} = 0.942$, $\text{RMSEA} = 0.058$) with high internal reliability ($\alpha = 0.865$).

### B. Quality-of-Life Anchor (WHO-5)
WHO-5 raw scores ($0–25$) are mapped to a standard $0–100$ scale:
$$\text{Score}_{\text{WHO-5}} = 4 \times \sum_{k=1}^{5} y_k$$
Scores $\le 50$ denote impaired well-being, while scores $< 28$ flag clinical depression risks (Topp et al., 2015).

### C. Conceptual SEM Path Architecture
The structural interaction between modalities is specified via a 4-node directed path model:
$$\text{Sensor Signals} \xrightarrow{\quad\text{autonomic}\quad} \text{Estimated State} \xrightarrow{\quad\text{appraisal}\quad} \text{Reported PSS-10} \xrightarrow{\quad\text{vitality}\quad} \text{Reported QoL (WHO-5)}$$
In accordance with scientific honesty, this SEM is presented as a **conceptual, non-fitted path model** because the laboratory dataset lacks concurrent multi-week longitudinal quality-of-life follow-up.

### D. Multimodal Agreement Analysis
We implement an automated Agreement Meter comparing momentary SVM/LogReg predictions against retrospective PSS-10 categories. Crucially, when an individual presents high physiological arousal (elevated HR/EDA) alongside low reported 30-day stress, NeuroTwin explains that physiological sensors detect acute metabolic and emotional activations, whereas PSS-10 reflects cumulative cognitive coping. Disagreement is treated as scientifically expected rather than an algorithmic error.

---

## IX. Discussion & Scientific Limitations

### A. Why Cross-Subject Accuracy is Modest ($57\%$)
Our subject-independent accuracy ($57.0\%$) is markedly lower than claims often published in machine learning venues ($> 90\%$). This disparity is explained by five objective physiological and methodological factors:
1. **Strict Subject-Independence:** Zero data leakage across folds.
2. **Small Cohort Size:** 35 subjects ($N=263$ windows) provides insufficient representation of the manifold of human autonomic diversity.
3. **Short Epoch Durations:** Task stages limited to $\approx 30\text{ s}$ truncate low-frequency HRV dynamics.
4. **Wrist PPG Noise:** Motion artifacts and peripheral vasomotor tone introduce noise absent in clinical ECG chest straps.
5. **Subjective Ground Truth:** Self-reported 1–10 Likert scales introduce cognitive reporting biases, baseline anchoring shifts, and subjective semantic drift.

### B. Ethical Governance & Non-Diagnostic Classification
NeuroTwin AI is strictly a research and educational prototype. It is not an FDA-cleared or CE-marked medical device. All interfaces enforce non-diagnostic language (e.g., "pattern associated with higher stress", avoiding "you are stressed"), offer immediate opt-out paths, and maintain persistent links to certified crisis resources (Tele-MANAS, NHS 111).

---

## X. Future Work & Conclusion

Future iterations will incorporate within-subject resting baseline normalization, longitudinal Ecological Momentary Assessment (EMA) over multiple weeks, and formal clinical validation of personalized intervention recommendations. In conclusion, NeuroTwin AI establishes a benchmark for research transparency in wearable computing by combining provenance separation, mathematically validated physiological feature extraction, honest majority-baseline evaluation, and principled uncertainty abstention.

---

## XI. Verified References

1. **Hongn2025-data:** Hongn A, Bosch F, Prado L, Bonomini P. Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (v1.0.1). *PhysioNet*; 2025. [doi:10.13026/he0v-tf17](https://doi.org/10.13026/he0v-tf17)
2. **Hongn2025-sci:** Hongn A, Bosch F, Prado LE, Ferrandez JM, Bonomini MP. Wearable Physiological Signals under Acute Stress and Exercise Conditions. *Sci Data* 2025;12. [doi:10.1038/s41597-025-04845-9](https://doi.org/10.1038/s41597-025-04845-9)
3. **Cohen1983-PSS:** Cohen S, Kamarck T, Mermelstein R. A global measure of perceived stress. *J Health Soc Behav* 1983;24(4):385–396. [doi:10.2307/2136404](https://doi.org/10.2307/2136404)
4. **Taylor2015-PSS-CFA:** Taylor JM. Psychometric analysis of the Ten-Item Perceived Stress Scale. *Psychol Assess* 2015;27(1):90–101. [doi:10.1037/a0038100](https://doi.org/10.1037/a0038100)
5. **Topp2015-WHO5:** Topp CW, Ostergaard SD, Sondergaard S, Bech P. The WHO-5 Well-Being Index: a systematic review of the literature. *Psychother Psychosom* 2015;84(3):167–176. [doi:10.1159/000376585](https://doi.org/10.1159/000376585)
6. **WHOQOL1998:** WHOQOL Group. Development of the World Health Organization WHOQOL-BREF quality of life assessment. *Psychol Med* 1998;28(3):551–558. [doi:10.1017/s0033291798006667](https://doi.org/10.1017/s0033291798006667)
7. **Kline2016-SEM:** Kline RB. *Principles and Practice of Structural Equation Modeling* (4th ed.). New York: Guilford Press; 2016.
8. **CortesVapnik1995-SVM:** Cortes C, Vapnik V. Support-vector networks. *Mach Learn* 1995;20(3):273–297. [doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
9. **TaskForce1996-HRV:** Task Force of the ESC and NASPE. Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation* 1996;93(5):1043–1065. [doi:10.1161/01.CIR.93.5.1043](https://doi.org/10.1161/01.CIR.93.5.1043)
10. **Shaffer2017-HRV:** Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. *Front Public Health* 2017;5:258. [doi:10.3389/fpubh.2017.00258](https://doi.org/10.3389/fpubh.2017.00258)
11. **Boucsein2012-EDA:** Boucsein W, et al. Publication recommendations for electrodermal measurements. *Psychophysiology* 2012;49(8):1017–1034. [doi:10.1111/j.1469-8986.2012.01384.x](https://doi.org/10.1111/j.1469-8986.2012.01384.x)
12. **Greco2016-cvxEDA:** Greco A, Valenza G, Lanata A, Scilingo EP, Citi L. cvxEDA: a convex optimization approach to electrodermal activity processing. *IEEE Trans Biomed Eng* 2016;63(4):797–804. [doi:10.1109/TBME.2015.2474131](https://doi.org/10.1109/TBME.2015.2474131)
13. **Lomb1976:** Lomb NR. Least-squares frequency analysis of unequally spaced data. *Astrophys Space Sci* 1976;39(2):447–462. [doi:10.1007/BF00648343](https://doi.org/10.1007/BF00648343)
14. **Scargle1982:** Scargle JD. Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data. *Astrophys J* 1982;263:835–853. [doi:10.1086/160554](https://doi.org/10.1086/160554)
15. **WESAD2018:** Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *Proc ACM ICMI* 2018. [doi:10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)
16. **NiculescuMizil2005:** Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. *Proc ICML* 2005:625–632. [doi:10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)
17. **Lundberg2017:** Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst 30 (NIPS 2017)*:4765–4774.
18. **Abdelaal2024:** Abdelaal Y, Aupetit M, Baggag A, Al-Thani D. Exploring the applications of explainability in wearable data analytics: systematic literature review. *J Med Internet Res* 2024;26:e53863. [doi:10.2196/53863](https://doi.org/10.2196/53863)
19. **Tudor2025:** Tudor BH, Shargo R, Gray GM, et al. A scoping review of human digital twins in healthcare applications and usage patterns. *npj Digit Med* 2025;8:587. [doi:10.1038/s41746-025-01910-w](https://doi.org/10.1038/s41746-025-01910-w)
20. **NASEM2024:** National Academies of Sciences, Engineering, and Medicine. *Foundational Research Gaps and Future Directions for Digital Twins*. Washington, DC: The National Academies Press; 2024.
21. **Sharma2026:** Sharma S, Janakiraman AK, Chen LK. Machine learning frameworks for wearable-based stress modeling in naturalistic settings: scoping review. *JMIR mHealth uHealth* 2026.
22. **Vos-SLR:** Vos G, Trinh K, Sarnyai Z, Rahimi Azghadi M. Generalizable machine learning for stress monitoring from wearable devices: a systematic literature review. *Int J Med Inform*. arXiv:2209.15137.
23. **TeleMANAS:** Ministry of Health and Family Welfare, India. Tele-MANAS national tele mental health helpline: 14416 or 1-800-891-4416. [https://telemanas.mohfw.gov.in](https://telemanas.mohfw.gov.in)

