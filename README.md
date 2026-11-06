# NeuroTwin AI: Wearable Physiological Stress Analysis & Research Platform

> A complete, research-grade physiological digital shadow and multimodal stress intelligence platform powered by the [PhysioNet Wearable Device Dataset v1.0.1](https://physionet.org/content/wearable-device-dataset/1.0.1/) (Empatica E4 multi-sensor wristband).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE.txt)
[![Dataset: ODC-By](https://img.shields.io/badge/Dataset-ODC--By-amber.svg)](https://physionet.org/content/wearable-device-dataset/1.0.1/)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-success.svg)](#disclaimer--medical-device-notice)

---

## 🌟 Product Overview

**NeuroTwin AI** transforms raw wearable biosignals into an interactive, transparent, and ethically governed stress analysis experience. Unlike generic templates or uncalibrated machine learning dashboards, NeuroTwin enforces strict architectural separation across three data layers:
1. **Measured Layer:** Continuous raw biosignals from the Empatica E4 (EDA 4Hz, HR 1Hz, BVP 64Hz, IBI, Skin Temp 4Hz, 3-Axis ACC 32Hz) with automatic motion and sensor decoupling quality vetoes.
2. **Estimated Layer:** Standardized machine learning inferences (Logistic Regression, SVM-RBF, Random Forest, Extra Trees) evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation with post-hoc probability calibration and uncertainty abstention gates.
3. **Reported Layer:** Subjective, cognitive psychometric assessments (Perceived Stress Scale PSS-10 and WHO-5 Well-Being Index) stored locally in SQLite exclusively with explicit user consent.

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           MEASURED LAYER                               │
  │  Raw Empatica E4: EDA (4Hz), HR (1Hz), BVP (64Hz), IBI, TEMP, ACC     │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                          ESTIMATED LAYER                               │
  │  23 Physiological Features ➔ LOSO Classifiers ➔ Platt Calibrator       │
  │  ➔ Uncertainty Abstention (No pseudo-probabilities if uncalibrated)    │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                          REPORTED LAYER                                │
  │  PSS-10 (0–40), WHO-5 (0–100), Coping Feedback (Consent-Gated SQLite)  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 🧭 Complete User Journey & Page Directory

NeuroTwin features a structured 9-stage **Primary Research Journey** plus a companion **Support Journey**:

### Primary Research Journey
1. **Home (`Home`):** Mission overview, interactive animated Digital Twin hero centerpiece, animated processing pipeline strip, dataset snapshot metrics, and one-click demo participant activation.
2. **Upload & Validate (`Upload & Validate`):** Drag-and-drop CSV validator checking all 23 feature schema columns, interactive Data Quality Card (% coverage, group integrity, motion flags), downloadable 23-feature template and sample dataset.
3. **Quick Questions (`Quick Questions`):** Optional psychometric intake featuring the **PSS-10** (10 items, reverse-scored items 4, 5, 7, 8; range 0–40) and **WHO-5 Well-Being Index** (0–100). Persisted locally under user consent.
4. **Stress Result (`Stress Result`):** Inferred stress state (Low vs High, threshold $\ge 5$), evidence confidence badge, semi-circle Stress Gauge, feature log-odds contribution chart, plain-language non-diagnostic explanations, **What-If sensitivity slider**, and **Multimodal Agreement Meter**. Includes one-click **Research PDF Report** generation.
5. **How Stress Is Calculated (`How Stress Is Calculated`):** Comprehensive research core detailing the exact mathematical formula in LaTeX, physiological ANS mechanism, evidence label, and verified DOI reference for each of the 23 features, accompanied by a full SVG pipeline diagram.
6. **Model Performance (`Model Performance`):** Real LOSO comparison table across 5 models, grouped bar chart with Majority Baseline reference line, ROC curve ($\text{AUC} = 0.539$), calibration reliability diagram, and the transparent *"Why Accuracy is Modest"* panel.
7. **Feature Lab (`Feature Lab`):** Global feature importance bar chart, correlation heatmap, distributions by stress class, protocol stage trajectories, and the *"Compare My Upload vs Research Population"* percentile overlay.
8. **Quality of Life & Stress Scale (`Quality of Life & Stress Scale`):** PSS-10 psychometric profile, one-factor confirmatory factor model (CFA, $\alpha = 0.865$) factor loadings, conceptual Structural Equation Model (SEM) path architecture, and continuous SVM model score comparison.
9. **Research Paper (`Research Paper`):** Complete in-app reader for the IEEE-style scientific paper (`docs/paper/NeuroTwin_paper.md`), downloadable Markdown/PDF, and full searchable Verified References library with clickable DOIs.

### Support Journey
Reachable via the sidebar Section selector:
- **Check-in**, **Possible Stressors**, **Recommendations**, **Twin Profile**, **History & Patterns**, **Coping Feedback**, and **Evaluation & Transparency**.

---

## 📊 Honest Empirical Performance (Real Numbers)

In accordance with strict scientific integrity, **no numbers are inflated or invented**. Evaluation was conducted under strict Leave-One-Subject-Out (LOSO) cross-validation on 35 healthy participants ($N=263$ stage windows) without temporal leakage:

| Classifier | Accuracy | Macro F1 | Balanced Accuracy | High-Stress Precision | High-Stress Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.5703** | **0.5568** | **0.5599** | **0.4483** | **0.5149** |
| **SVM (RBF Kernel)** | 0.5057 | 0.4795 | 0.4795 | 0.3592 | 0.3663 |
| **Extra Trees** | 0.4981 | 0.3940 | 0.4248 | 0.2075 | 0.1089 |
| **Random Forest** | 0.4791 | 0.4418 | 0.4429 | 0.3085 | 0.2871 |
| **Majority-Class Baseline** | **0.6160** | 0.3812 | 0.5000 | 0.0000 | 0.0000 |

### Calibration & Safety Abstention Gate
- **Pooled ROC AUC:** $0.5393$
- **Raw Brier Score:** $0.2761$ ➔ **Recalibrated Brier:** $0.2507$ (Training prior benchmark: **$0.2441$**)
- **Raw ECE:** $0.1792$ ➔ **Recalibrated ECE:** **$0.0524$**
- **Safety Gate Status:** Because post-calibration Brier ($0.2507$) does not outperform the uninformative training prior ($0.2441$), raw probabilities are **strictly suppressed** from user interfaces in favor of ordinal evidence levels (*Limited* / *Moderate*).

### Why Cross-Subject Accuracy is Modest (57%):
1. **Zero Data Leakage:** True subject-independent evaluation prevents models from memorizing individual resting baseline heart rates.
2. **Small Sample Size:** $n=35$ participants represents a small cross-section of autonomic diversity.
3. **Ultra-Short Windows:** Protocol stages lasted $\approx 30$ seconds (arithmetic/opinion), attenuating low-frequency spectral HRV.
4. **Wrist PPG Noise:** Motion artifacts and vasomotor peripheral tone create noise absent in clinical ECG chest straps.
5. **Subjective Ground Truth:** 1–10 Likert scales introduce cognitive self-report variance.

---

## 📐 The 23 Extracted Physiological Features

| Domain | Features | Key Formulation Highlights | Evidence Level |
| :--- | :--- | :--- | :--- |
| **Heart Rate** | `hr_mean`, `hr_std`, `hr_min`, `hr_max`, `hr_range`, `hr_diff_rms` | Resampled 1 Hz series; `hr_diff_rms` is RMS of successive 1 Hz HR differences (not Task Force RMSSD). | Custom derived |
| **Electrodermal** | `eda_mean`, `eda_std`, `eda_slope`, `eda_range`, `eda_scl_mean`, `eda_scr_count` | SCL proxy (60-sample rolling mean); SCR count proxy ($>0.02\ \mu\text{S}$ threshold crossing). | Custom / Heuristic |
| **Skin Temp** | `temp_mean`, `temp_std`, `temp_slope` | Thermistor 1 Hz series capturing peripheral sympathetic vasoconstriction. | Custom derived |
| **Movement** | `acc_mean`, `acc_std`, `acc_max` | 32-sample block maximum delta index smoothed by 0.1 EMA; motion artifact veto. | Custom derived |
| **HRV (IBI)** | `hrv_mean_ibi`, `hrv_sdnn`, `hrv_rmssd`, `hrv_pnn50`, `hrv_lf_hf_ratio` | Task Force millisecond NN formulas; Lomb-Scargle periodogram on mean-centered IBI. | Matches standard / Uncertain* |

*\*Note: `hrv_lf_hf_ratio` averages periodogram ordinates rather than integrating band power, inflating values by $\approx 2.283\times$; it is retained in classifier models but suppressed from explanations.*

---

## 🚀 Quickstart & How to Run

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/NeuroTwin_full_project.git
cd NeuroTwin_full_project
pip install -r requirements.txt
```

### 2. Launch the Application
#### Windows (One-Click Runner):
```cmd
run_app.bat
```

#### Linux / macOS:
```bash
chmod +x run_app.sh
./run_app.sh
```

#### Or directly via Streamlit:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite verifying feature formulas, model integrity, psychometric scoring, CSV validation, and protected file SHA-256 hashes:
```bash
python -m pytest tests/ -v
```
*(Note: Loader tests in `test_loader.py` require the raw multi-gigabyte PhysioNet E4 archives and are skipped when raw files are not locally mounted).*

---

## ⚖️ Disclaimer & Medical Device Notice

> **IMPORTANT:** NeuroTwin AI is an **educational and research prototype**. It is **NOT** a medical device, is not FDA-cleared or CE-marked, and does **NOT** provide medical diagnoses, clinical screening, or treatment recommendations. Predictions describe statistical patterns observed in laboratory datasets and should never replace consultation with a qualified healthcare or mental health professional.

### Immediate Crisis Support Resources:
- **India:** Tele-MANAS (24x7 National Tele Mental Health Helpline): **14416** or **1-800-891-4416**
- **United Kingdom:** NHS Mental Health Services: **111** (Free call, 24/7)
- **United States:** National Suicide and Crisis Lifeline: **988** (Free call or text, 24/7)
- **International:** Find local certified crisis support at [Befrienders Worldwide](https://www.befrienders.org/)

---

## 📄 Citation

If you use NeuroTwin AI in academic research, please cite:

```bibtex
@article{ayesha2026neurotwin,
  title={NeuroTwin AI: A Provenance-Aware Physiological Digital Shadow for Acute Stress Estimation and Multimodal Quality-of-Life Integration},
  author={Ayesha et al.},
  journal={Department of Computer Science & Cognitive Systems Engineering, Project NeuroTwin},
  year={2026}
}
```

Attribution: *Contains information from the Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (Hongn et al., 2025), made available under the Open Data Commons Attribution License (ODC-By).*
