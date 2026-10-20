# Wearable Stress Analysis

> Physiological stress detection using the [PhysioNet Wearable Device Dataset v1.0.1](https://physionet.org/content/wearable-device-dataset/1.0.1/) (Empatica E4 wristband recordings).

Contains information from the *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* which is made available under the **ODC Attribution License (ODC-By)**.  
Full licence text: [`LICENSE.txt`](LICENSE.txt)

---

## Project Overview

This project builds an end-to-end wearable biosignal analysis pipeline:

| Stage | Script | Output |
|-------|--------|--------|
| 1 · Load & Verify | `scripts/01_load_and_verify.py` | `data/processed/dataset_summary.csv` |
| 2 · EDA | `scripts/02_eda.py` | 6 figures in `reports/figures/` |
| 3 · Feature Engineering | `scripts/03_feature_engineering.py` | `data/features/stress_features.csv` |
| 4 · Stress Analysis | `scripts/04_stress_analysis.py` | Statistical CSVs + 5 figures |
| 5 · Model Evaluation | `scripts/05_model_evaluation.py` | LOSO results + confusion matrices |

Run everything with one command:
```
run_project.bat
```

---

## Dataset Structure

```
Wearable_Dataset/
├── STRESS/     37 subject folders (S01–S18, f01–f18)
├── AEROBIC/    31 subject folders
└── ANAEROBIC/  32 subject folders

Per subject: ACC.csv  BVP.csv  EDA.csv  HR.csv  IBI.csv  TEMP.csv  tags.csv
```

**36 participants** — 18 male (S cohort, Protocol V1) + 18 female (f cohort, Protocol V2)

---

## Signals

| Signal | Description | Sample Rate |
|--------|-------------|-------------|
| EDA | Electrodermal activity (µS) | 4 Hz |
| HR | Heart rate (bpm) | 1 Hz |
| BVP | Blood volume pulse | 64 Hz |
| IBI | Inter-beat intervals (s) | Irregular |
| TEMP | Skin temperature (°C) | 4 Hz |
| ACC | 3-axis accelerometer (±2g) | 32 Hz |
| tags | Button-press event markers | — |

---

## Features Extracted (per stage window)

- **HR**: mean, std, min, max, range, RMSSD approximation
- **EDA**: mean, std, slope, range, SCL (tonic), SCR peak count (phasic)
- **TEMP**: mean, std, slope
- **ACC**: activity magnitude (mean, std, max)
- **IBI/HRV**: mean IBI, SDNN, RMSSD, pNN50, LF/HF ratio (Lomb-Scargle)

---

## Models

- **Random Forest** and **SVM (RBF kernel)**
- Binary classification: high stress (≥5) vs low stress (<5)
- **Leave-One-Subject-Out (LOSO)** cross-validation

> ⚠️ Standard k-fold CV is not used — it would cause data leakage across subjects.

---

## Setup

```bash
pip install -r requirements.txt
```

### Run
```bash
# Windows one-click
run_project.bat

# Or step by step
python scripts/01_load_and_verify.py
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_stress_analysis.py
python scripts/05_model_evaluation.py
```

### Tests
```bash
python -m pytest tests/ -v
```

---

## Data Constraints Handled

| Subject | Issue | Handling |
|---------|-------|---------|
| S02/STRESS | Duplicated signal tails | Truncated at known row indices |
| f07/STRESS | PPG/TEMP sensors blocked | BVP, HR, IBI, TEMP marked None |
| f14/STRESS | Bluetooth split (a/b) | Sessions concatenated |
| S11/AEROBIC | Bluetooth split (a/b) | Sessions concatenated |
| S16/ANAEROBIC | Bluetooth split (a/b) | Sessions concatenated |
| S01/ANAEROBIC | Empty IBI file | IBI features set to NaN |
| S12 | No aerobic data | Subject excluded from AEROBIC |
| S03, S07 | Early aerobic stop | Used available data |
| S06 | Incomplete anaerobic | Used available sprints |

---

## Licence

- **Dataset**: ODC Attribution License (ODC-By) — see `LICENSE.txt`
- **Code**: MIT (this project's analysis code)

Attribution: *Contains information from the Wearable Device Dataset from Induced Stress and Structured Exercise Sessions, made available under the ODC Attribution License.*
