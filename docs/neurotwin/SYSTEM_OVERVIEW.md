# NeuroTwin AI — System Overview

**NeuroTwin AI: An Explainable and Personalized Physiological Digital Twin for Stress Understanding and Adaptive Stress Management** (research prototype; not a medical device).

The user journey: **Detect → Explain → Ask → Understand → Recommend → Learn.**

## 1. What was added, and what was left alone

| Left unchanged | Added (new) |
|---|---|
| `src/features.py`, `src/models.py`, `src/loader.py`, `src/preprocessing.py` | `src/neurotwin/` (the support layer) |
| All saved results (`data/features/stress_features.csv`, `data/processed/*.csv`) | `data/processed/neurotwin_estimator_report.json`, `neurotwin_simulation_report.json` (new files) |
| Existing dashboard pages (Overview … Methods) | Ten "Support journey" pages, reached from the sidebar section selector |
| The 23 features, their formulas and all classifiers | A wrapper that *uses* them: explanation, quality checks, abstention, calibration audit |

`dashboard/app.py` received a small additive edit (an import, a section selector, a router branch). Nothing else in it changed.

## 2. Three things that are never interchangeable

| Layer | Meaning | Where it lives |
|---|---|---|
| **Measured** | What the wristband recorded, and features computed from it | `measured` table; `Quality` objects |
| **Estimated** | What the model output (with evidence level, caveats) | `estimates` table; `EstimateResult.layer == "estimated"` |
| **Reported** | What the person said (mood, rating, reason, feedback) | `reported`, `feedback` tables |
| *Observations* | Statements computed **only** from the person's own reports | `patterns.py` (layer `hypothesis`) |

Enforcement: separate tables, a `CHECK(provenance = …)` constraint on each, no generic insert method, and tests that scan the code to confirm only `store.py` can write to those tables.

## 3. Module map (`src/neurotwin/`)

| Module | Role |
|---|---|
| `estimator.py` | Wraps the existing `logreg` pipeline. Quality flags, abstention, per-feature contributions, in-fold probability calibration, held-out audit, evidence ceiling |
| `dialogue.py` | Scripted, rule-based check-in (no generative model). Every step skippable; urgent-help and end buttons everywhere |
| `safety.py` | Crisis phrase backstop, region-configurable resources, opt-in rate-limited prompt policy |
| `store.py` | Local SQLite profile with consent gating, inspect/correct/delete, session-only (`:memory:`) mode |
| `catalogue.py`, `data/catalogue.json` | 14 coping items, each with sources, evidence note and caution (validated by tests) |
| `recommender.py` | Transparent rule-based scoring with visible reasons, hard exclusions, bounded preference updates |
| `patterns.py` | Pattern statements only after repeated observations |
| `profile.py` | Digital-twin profile view-model |
| `service.py` | Glue between dialogue, recommender and store (no UI code) |
| `simulation.py` | Simulated users for software verification **only** |
| `demo.py` | Presents windows of the research dataset as a labelled demo participant |
| `ui.py` | The ten Streamlit pages |
| `language.py` | Forbidden-wording guard used by tests |

## 4. The ten pages

Home & Status · Signals & Features · Explain the Estimate · Check-in · Possible Stressors · Recommendations · Twin Profile · History & Patterns · Coping Feedback · Evaluation & Transparency.

A persistent **"Talk to someone now"** control and the **Privacy & consent** panel sit in the sidebar on every page.

## 5. Important results that shape the design

From the held-out (leave-one-subject-out) audit of the logistic-regression pipeline (`data/processed/neurotwin_estimator_report.json`, 263 windows, 35 participants):

| Quantity | Value |
|---|---|
| Accuracy | 0.570 (reproduces the saved LOGREG row) |
| Majority-class baseline accuracy | 0.616 |
| AUC | 0.539 |
| Brier: recalibrated vs always-guess-the-prior | 0.248 vs 0.244 |
| Expected calibration error (recalibrated) | 0.105 (limit 0.10) |
| Calibration acceptable? | **No** |

Consequences, all implemented and tested:
* **No probability is ever shown** (the display rule needs Brier better than the prior *and* ECE ≤ 0.10).
* **Evidence is capped at "limited"** for every window (a window may be "moderate" only if held-out AUC ≥ 0.65 and calibration passes).
* **The app never interrupts the user** (prompts require "moderate" evidence and an explicit opt-in).
* 7 windows with no usable heart-rate data (participant `f07`, known sensor-cover issue) **abstain**.

## 6. Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests -q                                  # 16 loader tests need the raw PhysioNet files
python scripts/08_neurotwin_estimator_report.py            # writes the estimator audit (new file)
python scripts/09_neurotwin_simulation.py                  # writes the simulation report (new file)
streamlit run dashboard/app.py
```

Expected: new tests pass; `tests/test_loader.py` fails only if the raw recordings are absent.

## 7. Privacy and consent

* **Default: session-only.** Check-ins live in memory and disappear when the page closes. Nothing is written to disk.
* **Opt-in persistence** (Privacy & consent → "Save my data on this device") writes `data/neurotwin_profile.db`, with separate switches for check-ins, reasons, feedback and prompts. The database is unencrypted and unauthenticated, so it is for the prototype only; do not use with real users without a data-protection review.
* **Free text is off by default**, is checked only for safety words, and is never analysed or used to infer a stressor.
* The person can inspect/download, correct, delete individual entries, forget learned preferences, or delete everything.

## 8. Crisis and support resources

`src/neurotwin/data/crisis_resources.json` is region-configurable and has **no default region**. Entries carry `verified_on` dates, all currently `null`. The India entry (Tele-MANAS 14416 / 1-800-891-4416) was found in public sources and a parliamentary answer on 2026-11-01; the "112" emergency number is **not verified**. **The project owner must verify every entry before any use with real people.**
