# NeuroTwin AI — Research Objective, Contribution and Feasibility Assessment (Stage 1)

**Status:** objectives and feasibility from Stage 1, with the implementation status added in Section 13. Decisions D1–D8 were not individually answered; the assumptions used are in Section 13 and can be changed. All dataset figures below were computed from the uploaded `stress_features.csv` (263 windows, 35 participants) and the PhysioNet dataset page (v1.0.1). Literature claims refer to `literature_review.md`.

---

## 1. Central research objective (refined wording)

> **To design, implement and critically evaluate an explainable, uncertainty-aware wearable-based stress-state estimator, and to integrate it with a consent-based conversational self-report check-in and a transparent, preference-adaptive recommender of evidence-informed coping strategies, within a research prototype that keeps sensor measurements, model estimates and self-reports strictly distinct, so that individuals can reflect on their possible stress patterns and choose coping strategies.**
> The estimator's participant-independent performance is evaluated on an existing laboratory dataset. The conversational, recommendation and feedback components are verified technically and demonstrated; their effectiveness is left to future user studies.

**Why it is reworded.** Your wording ("support individuals in understanding their potential stress patterns and selecting appropriate coping strategies") is kept in spirit. The changes make the claims *evaluable*: what is evaluated (estimator), what is only verified/demonstrated (Part B), and what is future work (Part C).

## 2. Research questions (refined so each can be answered with available evidence)

| RQ | Wording | What can answer it | What it will **not** show |
|---|---|---|---|
| **RQ1** | How well do wrist-worn physiological features estimate *self-reported* stress level (thresholded at ≥5 on a 1–10 scale) in previously unseen participants, relative to a majority-class baseline, and how well-calibrated and robust are the estimates when signals are missing or poor? | Part A: leave-one-subject-out (LOSO) evaluation on the PhysioNet dataset, calibration nested in folds, missing-data stress tests. | Performance in daily life; performance for older or clinical groups. |
| **RQ2** | What explanations of a model estimate can be generated from physiological features, how faithful and stable are they, and how should they be worded to avoid causal or diagnostic misreading? | Offline faithfulness/stability checks on the model; documented wording rules; a heuristic design review. | Whether users *understand* or *trust* the explanations (needs a user study). |
| **RQ3** | How can a voluntary, structured self-report check-in complement the physiological estimate while keeping the two separate, and what can and cannot be concluded from combining them? | Data-model and interface design with enforced separation (tested in code); analysis of estimate-vs-self-report disagreement in the dataset. | The real causes of any person's stress; that stressors *cause* physiological changes. |
| **RQ4** | How can a transparent rule-based recommender use a user's stated preferences and feedback to adapt suggestions, and does it behave as specified (safety exclusions, cold start, adaptation) in scripted scenarios? | Automated tests and scripted simulations (explicitly labelled *software verification only*). | That any suggestion reduces stress, or that the adaptation improves outcomes. |
| **RQ5** | What are the technical, scientific and ethical limitations of a wearable-based digital twin for stress understanding and support? | A limitations register built across stages from audits (e.g. the `f14` split recording, LF/HF scaling, reproducibility difference), dataset constraints and the literature. | — |

## 3. Defensible contribution

1. **An integrated, open, reproducible prototype** that connects estimation, explanation, check-in conversation, recommendation and feedback in one auditable code base.
2. **An epistemic-separation design** (Section 4) enforced in the data model and tests.
3. **An honest evaluation case study.** Current results show that no classifier beats the majority-class baseline on accuracy (baseline 0.616; best model, logistic regression, 0.570; macro-F1 0.557 vs baseline 0.381). Reporting this plainly, and designing the system to *abstain* when evidence is weak, is itself a contribution.
4. **A safety scaffold** for wearable-triggered well-being prompts (consent, non-diagnostic language, crisis routing, prompt rate limits).

The project does **not** claim: higher accuracy than prior work; validated stress detection in daily life; an effective intervention; a validated digital twin.

## 4. The three-way separation, specified

| Layer | Meaning | Source | Example fields | Interface wording |
|---|---|---|---|---|
| **Measured** | What the sensor recorded and what was computed from it | Wearable signals, derived features, quality flags | HR, EDA, skin temperature, accelerometer, 23 features, missing-data and artefact flags | "Your wristband recorded …" |
| **Estimated** | What the model output | Model, with version and uncertainty | class, evidence level, calibrated probability *only if validated*, top contributing features | "The model's estimate is …; it weighted … most" |
| **Reported** | What the person says | User input, always optional | mood, 0–10 stress rating, chosen reason, free text, feedback | "You told us …" |
| *(Hypothesis)* | System-generated suggestion about a pattern | Pattern rules, only after enough repeated reports | "You have reported study-related stress on several occasions" | Marked as an observation of the person's own reports, never as a cause |

**Enforcement.** (i) Separate tables/namespaces with a `provenance` field; (ii) no code path writes model output into the Reported layer or user input into the Measured layer; (iii) a unit test that fails if one does; (iv) every screen labels which layer each statement comes from.

**Evidence from your data that the separation matters.** The model's label is the participant's own 1–10 rating thresholded at ≥5, not an objective stress measure and not the experimental condition. In the 263 windows:

| | self-report < 5 | self-report ≥ 5 |
|---|---|---|
| Rest/baseline windows (105) | 79 | 26 |
| Task windows (158) | 83 | 75 |

So a quarter of rest windows were rated ≥5 and more than half of task windows were rated <5. Also, **9 of 35 participants never reported ≥5, 4 reported ≥5 in every window, and only 22 have both classes.** **[Review's inference]** Individual use of the rating scale probably contributes to the label, which is a reason to be careful about calling the model a "stress detector".

## 5. Claim ladder (what each part may claim)

| Level | Claim | Status |
|---|---|---|
| **L0** | The software behaves as specified (unit/integration tests) | Achievable now |
| **L1** | Participant-independent offline performance on one laboratory dataset | Part A; partly done, audit items open |
| **L2** | Usability and acceptability with consenting users | Part C; requires ethics approval; **not done** |
| **L3** | Coping suggestions reduce stress | Part C; trial-level evidence; **out of MSc scope** |

Part B (conversation, recommender, feedback, profile) can reach **L0 only**. Any wording implying L2 or L3 must not appear in the dashboard, report or presentation.

## 6. Feasibility against the actual dataset and code

| # | Capability | Dataset/code support | Verdict | Notes and conditions |
|---|---|---|---|---|
| 1 | Stress-state estimate | 263 windows, 35 participants, 23 features | **Supported, weak** | Does not beat majority baseline on accuracy. The system must be built around abstention and evidence levels. |
| 2 | Calibrated confidence | Possible by nested calibration | **Conditional** | Only shown as a probability if reliability diagrams and Brier score on held-out participants support it; otherwise an ordinal evidence level. Random-forest result also differs between my re-run and the saved file (cause unconfirmed). |
| 3 | Feature-level explanation | Logistic regression is best and natively interpretable; SHAP possible | **Supported** | Explains the model, not physiology. LF/HF must not be explained until its scaling issue is resolved. |
| 4 | Missing/poor-quality flags | 51/263 windows lack IBI features (19%), 7 lack HR; `f07` cover not removed; accelerometer present | **Supported** | Rule: low quality → no high-stress statement. |
| 5 | Physical activity as alternative explanation | Accelerometer features exist; dataset also has aerobic (30) and anaerobic (31) exercise sessions | **Partly supported** | Needs new feature extraction from exercise recordings (to a *new* file) and raw data access. Heat cannot be identified (no ambient temperature); only skin-temperature shifts can be flagged. Requires your approval. |
| 6 | Baseline-normalised ("personalised") estimate | Each participant has a baseline and rest periods | **Within-session only** | Original Requirement 5; weaker than longitudinal personalisation. |
| 7 | Long-term personal baseline / longitudinal patterns | One session per participant (7–8 windows) | **Not supported** | Part B demo with clearly labelled synthetic or consented own data only. |
| 8 | Stressor exploration | Tasks are fixed experimental tasks; no stressor data | **Not supported by data** | Part B only, user-supplied, never inferred from physiology. |
| 9 | Recommendation learning | No intervention or outcome data | **Not supported by data** | Rule-based starting point sourced from literature; no learning claims. |
| 10 | Feedback adaptation | Needs user interaction | **Software verification only** | Scripted simulated users; labelled as such. |
| 11 | Questionnaire / SEM | README lists only age, weight, height and the single 1–10 rating | **Not justified** (pending my inspection of `subject-info.csv`) | Original Requirement 8; no PSS/STAI/PANAS exist to use. |
| 12 | Conversation | No dataset needed | **Supported** | Scripted, rule-based flow (see decision D3). |

## 7. How to describe "digital twin"

Using the definition in Tudor et al. (2025): a digital twin is personalised, dynamically updated, predictive and decision-informing, with two-way exchange. NeuroTwin meets some parts (personalised profile; decision-informing suggestions; feedback changes the recommender) but not others (estimation rather than forecasting; one-way sensing; limited verification/validation/uncertainty quantification). **Recommended wording:** *"a personal digital shadow with a feedback loop for recommendations"*, and a thesis paragraph stating which parts of the definition are met. Keep the project name.

## 8. Claims register

| Permitted | Not permitted |
|---|---|
| "The model's estimate is …; it weighted heart rate and electrodermal features most." | "You are stressed." / "You have stress." |
| "Your signals resemble the pattern associated, in a laboratory dataset, with higher self-reported stress." | "Your stress is caused by …" (from any sensor reading) |
| "You have reported study-related stress on several occasions." | "Studying causes your stress response." |
| "Some people find this exercise helpful; you can skip it." | "This will reduce your stress." |
| "Your usual readings (this week's baseline) …" | "Your normal / healthy range …" (a baseline is not a clinical norm) |
| "Evidence is limited because the signal quality is low." | Any high-stress statement when quality is low |
| "Research prototype; not a medical device." | "Clinically validated", "diagnoses", "treats" |

## 9. Safety and ethics commitments (to be implemented and tested in later stages)

1. **Non-diagnostic.** No diagnosis or labelling of disorders, no "stressed person" labels, no alarming alerts.
2. **Crisis routing.** A persistent "Talk to someone now" control on every page. If the person indicates immediate danger or self-harm, the scripted flow **stops**, offers no relaxation exercise, and shows human-support options. Free-text keyword screening is a *backstop with known false negatives*, never the only route. **Resource numbers are region-configurable and carry a "verified on" date; there is no default region until the owner sets and verifies one** (example found: India's Tele-MANAS, 14416 or 1-800-891-4416; support line, not an emergency service).
3. **User control.** Reject a prediction, correct an inferred item, decline any recommendation, pause prompts, inspect, edit, export and delete their own data.
4. **Consent and privacy.** Granular consent switches (store check-ins / stressor tags / feedback). Free text is **off by default**, never used for model inference, and never analysed by machine learning. Local storage for the prototype; not suitable for real deployment without a data-protection review.
5. **Prompt discipline.** Opt-in prompts, minimum interval, daily cap, quiet hours, snooze; no prompt when evidence is insufficient.
6. **Catalogue discipline.** Every recommendation item carries a source, an evidence note and a contraindication note. Items without a verified source are excluded. Initial content: WHO *Doing What Matters* grounding and related skills, slow gentle breathing (no breath-holding or forced breathing), brief break/task-splitting prompts, and "talk to someone you trust". Sleep-routine and movement items need a verified source before inclusion.
7. **Human oversight.** Supervisor and ethics review before any use with real participants; the prototype must not be presented as a replacement for professional support.

## 10. Dependencies on existing work, and revised roadmap

**Open items from earlier audit stages that block later work:**
- Requirement 1: decision on the LF/HF scaling discrepancy, plus renaming `hr_rmssd_approx`.
- Requirement 4: the missing participant `f14` (loader drops split recordings), the random-forest reproducibility difference, and incomplete per-class outputs.
- Requirements 2, 3, 5: feature selection, factor-analysis feasibility and baseline normalisation have not been run or tested.

| Stage | Work | Changes repository? | Approval gate |
|---|---|---|---|
| **1 (this)** | Literature review, objective, feasibility | No | Your review |
| **2** | Architecture, data model, data-flow, claims register in code form | New design docs and schema only | Approve architecture and decisions D1–D8 |
| **3a** | Finish Part A audit items (above) | Yes (tests, loader fix, new outputs; no overwrites) | Approve each scientific change |
| **3b** | Explainable, uncertainty-aware estimator | New module, new output files | Approve target(s) and calibration policy |
| **4** | Scripted conversational check-in | New module | Approve dialogue script and safety flow |
| **5** | Voluntary self-report and context store | New module and schema | Approve consent design |
| **6** | Rule-based recommender with sourced catalogue | New module | Approve catalogue and exclusions |
| **7** | Feedback and personal profile | New module | — |
| **8** | Dashboard integration | Yes (dashboard) | Approve page design |
| **9** | Testing, evaluation, documentation | Yes | — |
| **10** | Final report and presentation | Yes | — |

New code would live in a separate package (e.g. `src/neurotwin/`) so the existing research code stays untouched except where you approve changes.

## 11. Decisions I need from you

- **D1.** Accept the refined objective and RQ wording?
- **D2.** Accept the claim ladder and the "digital shadow with a feedback loop" terminology?
- **D3.** Scripted, rule-based dialogue with **no generative language model** (for safety and reproducibility)?
- **D4.** Part A target: keep "self-report ≥5" only, or also evaluate "task vs rest" as a second target (the two disagree substantially)?
- **D5.** May I extend feature extraction to the exercise sessions (new file, no overwrites; needs raw data) to support a physical-activity check?
- **D6.** Which region's crisis resources should the prototype use by default, and who will verify them?
- **D7.** Is an ethics route available for any user pilot (Part C)? If not, Part B stays demo-only.
- **D8.** Order of work: proceed with Stage 2 (documentation-only) now while you decide the earlier open Part A items, or finish Part A first?

## 12. Known limitations of this stage

- The literature search was targeted and non-systematic; findings come from abstracts and retrieved pages, not full texts.
- Several references are level B or C (see CSV) and are flagged.
- Dataset figures were computed from the uploaded feature file; I have not seen the raw recordings, `subject-info.csv`, or the self-rating CSVs.
- Crisis-resource details need owner verification before any real-user use.


## 13. Implementation status and assumptions (added after implementation)

**Assumptions applied for decisions D1–D8** (you asked me to proceed without changing the models, so I chose the most conservative options; tell me if you want any of them changed):

| Decision | Assumption used |
|---|---|
| D1, D2 | Refined objective, RQs, claim ladder and "personal digital shadow with a feedback loop" wording accepted |
| D3 | Scripted rule-based dialogue; **no generative language model** |
| D4 | Part A target unchanged: self-report ≥ 5 (no "task vs rest" target added; no model changes) |
| D5 | Not done: no feature extraction from exercise sessions (needs raw data). Physical activity is handled with movement/temperature flags and caveats only |
| D6 | **No default region.** India entry present but marked unverified; owner must verify |
| D7 | No ethics route assumed, so Part B is demo-only and verified as software; nothing is presented as evaluated with users |
| D8 | Implemented the new layer without touching the Part A items that remain open (see below) |

**What exists now**

| Research question | Evidence the project can now provide |
|---|---|
| RQ1 | Existing LOSO results plus the new held-out audit: accuracy 0.570 vs majority baseline 0.616, AUC 0.539, calibration not acceptable |
| RQ2 | Faithful per-window contributions (tested to equal coefficient × standardised value), wording rules enforced by tests. **Not** tested with users |
| RQ3 | Data model and interface that keep Measured / Estimated / Reported apart (database constraints + tests); analysis of rating-vs-condition disagreement in the dataset |
| RQ4 | Rule-based recommender and adaptation verified by automated tests and a **simulated-user** check (software verification only) |
| RQ5 | `LIMITATIONS_AND_FUTURE_WORK.md` |

**Still open from the earlier audit (untouched on purpose):** the LF/HF scaling decision (the feature is held back from explanations only), renaming `hr_rmssd_approx`, the missing participant `f14` and the README's "sessions concatenated" statement (inaccurate), the random-forest reproducibility difference, feature selection, factor analysis and baseline-normalisation experiments, and SEM feasibility (needs `subject-info.csv`).
