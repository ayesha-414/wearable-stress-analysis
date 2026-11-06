# NeuroTwin AI — Literature Review (Stage 1 draft)

**Status:** working draft prepared with AI assistance on 1 November 2026. Every reference is listed in `references_verified.csv` with a verification level. **The student must read the primary papers before citing them in the thesis.**

---

## 0. How this review was produced, and its limits

- **Method.** A targeted (non-systematic) search of the web, about 18 queries, roughly one per topic area. No PRISMA protocol, no database search, no formal quality appraisal. Papers were chosen for relevance and standing within each area. Absence of a paper here is therefore **not** evidence that it does not exist.
- **What was read.** For each paper I saw the bibliographic record and the abstract or page text returned by the search tool, not the full paper. Findings below summarise *what those pages state*. Numbers are quoted only if they appeared in a retrieved page.
- **Verification levels** (see CSV): **A** = record or abstract seen; **B** = seen only inside another source's reference list, not opened; **C** = preprint, not peer reviewed. B and C items are flagged in the text.
- **Convention.** Text marked **[Review's inference]** is my reasoning, not a finding of the cited work.

---

## 1. Wearable-based physiological stress detection

**What existing work does.** Laboratory datasets such as WESAD (15 participants; wrist and chest sensors; baseline, stress and amusement conditions; questionnaires) are the standard benchmark. The WESAD authors report accuracies of up to 93% for binary stress vs non-stress and up to 80% for three classes (Schmidt et al., 2018). The dataset used here (Hongn et al., 2025) is similar in spirit: 36 healthy volunteers aged 18–30 wearing an Empatica E4 during structured stress tasks, with a 1–10 verbal self-rating before and after each block, plus separate aerobic and anaerobic cycling sessions.

**Limitations reported by reviews.**
- A scoping review of naturalistic studies (Sharma et al., 2026) stresses poor standardisation of dataset and modelling reporting and the difficulty of real-world data collection. It cites a further scoping review (Bolpagni et al.) in which only 13 of 56 studies were in real-world settings (secondhand; not opened).
- A systematic review focused on generalisation (Vos et al.) concludes that the generalisation of models to unseen data still needs study and notes growing interest in person-specific rather than generic models.
- A 2026 **preprint (level C, not peer reviewed)** using 894 participants with wearable data and momentary self-reports reports that within-person models improved modestly while between-person models failed to generalise. Treat as a hypothesis, not evidence.

**[Review's inference]** High accuracy on small laboratory datasets does not establish real-world validity. For NeuroTwin this means: report participant-independent (leave-one-subject-out) results with a majority baseline, and never describe laboratory results as evidence about daily-life stress.

## 2. Explainable AI for physiological stress estimation

- SHAP assigns each feature an importance value for a *particular prediction* (Lundberg & Lee, 2017).
- A systematic review of explainability in wearable data analytics found that post hoc methods dominate, SHAP is prominent, outputs are usually graphs or reports, and **user evaluation of explanations is a recognised gap** (Abdelaal et al., 2024). A review of explainable AI for physiological signals makes a similar point that interpretability is often ignored in physiological modelling (Shen et al.; venue/year not confirmed).
- An applied study combining SHAP with gradient boosting on three wearable datasets reports that heart rate, RR intervals and HRV measures (notably LF/HF) were key predictors and that individual differences affected predictions (Mihirette et al.; venue/year not confirmed).

**[Review's inference]** A feature attribution explains *the model's behaviour*, not the person's physiology or the cause of their stress. The dashboard wording must say "the model weighted X", never "X shows you are stressed because …". Because the project's LF/HF feature has a known scaling discrepancy (see `feature_formula_validation.md`), any explanation that mentions LF/HF must be held back until that is resolved.

## 3. Personalised stress monitoring and individual baselines

- Taylor et al. (2020) and Jaques et al. (2017) argue that one-size-fits-all models are poorly suited to mood and stress because of individual differences, and use multitask learning to build per-person models that still borrow from the population. Their target is the **next-day** self-reported outcome, trained on **longitudinal** survey, wearable and phone data.
- The preprint noted above points the same way.

**[Review's inference]** Genuine personalisation in this literature needs repeated observations per person over days or weeks. The PhysioNet dataset gives each participant a single session of about 7–8 stages. At most, it can support a **within-session** test of baseline-normalised features (using each person's own baseline and rest periods). That is a weaker claim than "personalised model" and must be named as such.

## 4. Digital phenotyping

Digital phenotyping was introduced for smartphone-based, moment-by-moment behavioural quantification, with strong emphasis on privacy and on the need for raw data and statistical care (Onnela & Rauch, 2016; Onnela, 2021). **[Review's inference]** NeuroTwin's wearable-physiology focus is closer to "personal sensing" than to smartphone digital phenotyping; the thesis should use the term carefully and cite it for context, not as a description of the system.

## 5. Ecological Momentary Assessment (EMA)

EMA repeatedly samples current experience in real time and in natural environments, which reduces recall bias and increases ecological validity (Shiffman et al., 2008; Stone & Shiffman, 1994, level B). The same review lists reactivity and compliance as methodological concerns. **[Review's inference]** This motivates a *brief, optional, low-frequency* check-in design, and it supports using the user's momentary self-report as a distinct data stream rather than as a ground truth for the sensor.

## 6. Just-in-time adaptive interventions (JITAIs)

A JITAI is an intervention design that aims to provide the right type and amount of support at the right time by adapting to a person's changing internal and contextual state, while avoiding support that is interruptive or not beneficial (Nahum-Shani et al., 2018). The paper distinguishes system-triggered ("push") from user-accessed ("pull") components, and states that a major gap exists between technical capability to deliver JITAIs and research on developing and evaluating them. The formal component list (decision points, tailoring variables, intervention options, decision rules, proximal/distal outcomes) was **not visible in the pages I retrieved**; verify it in the full text before using the terminology.

**[Review's inference]** NeuroTwin fits the *push + pull* pattern: the prediction can trigger an *optional prompt*, and the user can also open the check-in at any time. A conservative decision rule (prompt only when evidence is sufficient and a minimum interval has passed) follows the "avoid interruptive support" principle.

## 7. Adaptive personalised recommendation

The most developed example is HeartSteps, where a reinforcement-learning algorithm decides five times a day whether to send an activity suggestion (Liao et al., 2020). It was designed using data from a prior 42-day study of 37 sedentary adults (HeartSteps V1), evaluated through a simulation built from that data, and the authors note that the treatment effect declined over time (non-stationarity). A related micro-randomised trial is cited in lecture material (Klasnja et al., 2019; level B).

**[Review's inference]** Reinforcement learning of this kind relies on prior trial data and simulation environments that NeuroTwin does not have. A **transparent rule-based recommender with simple per-user preference scores** is the defensible starting point. The thesis must not claim reinforcement learning, bandits or "learning" beyond what is implemented and evaluated.

## 8. Evidence-informed stress-management content

- **WHO "Doing What Matters in Times of Stress"** (WHO, 2020) is an illustrated self-help guide with five skills — grounding, unhooking, acting on values, being kind, making room — based on Acceptance and Commitment Therapy, with accompanying audio exercises, and designed to be used alone or with brief guidance.
- The main trial evidence is for **Self-Help Plus** (the WHO group course that includes the guide): a cluster-randomised trial in 694 South Sudanese refugee women with at least moderate distress found a small advantage over enhanced usual care at 3 months (d = −0.26; 95% CI of the primary coefficient −2.33 to −0.08) (Tol et al., 2020). The intervention was **facilitator-guided and group-based**.
- A meta-analysis of 12 RCTs (785 adults) found breathwork reduced self-reported stress, with most trials at moderate risk of bias; effects on anxiety (g = −0.32) and depression (g = −0.40) were similar (Fincham et al., 2023). A meta-analysis in *patient* populations found short-term benefits of pranayama, with adverse events more frequent for fast than for slow breathing techniques (Mütze et al., 2025).

**[Review's inference]** (a) The evidence base is for guided delivery in distressed populations; there is no evidence here that an unguided, chatbot-delivered version reduces stress in healthy students. (b) Outcomes are self-reported and short-term; **none of this evidence concerns wearable-measured physiology**. (c) Only slow, gentle breathing exercises should be offered, with no breath-holding or forceful breathing. The recommendation text can be sourced from the WHO guide (CC BY-NC-SA licence), but **no efficacy claim** may be made for the prototype.

## 9. Conversational agents for mental well-being

A randomised trial of a text-based CBT conversational agent (Woebot) in 70 university-age adults with depression or anxiety symptoms found greater symptom reduction than an information-only control over two weeks (Fitzpatrick et al., 2017). The study was unblinded, short, and the authors were affiliated with the company. A systematic review of AI CBT chatbots reports promising effects in ten studies (Farzan et al., 2024; lower-tier venue, cite cautiously). A systematic review of conversational agents in healthcare (Laranjo et al., 2018) is level B.

**[Review's inference]** Evidence for chatbots concerns *symptomatic* people using commercial CBT agents. It says nothing about a sensor-triggered check-in. For safety and reproducibility, the NeuroTwin dialogue should be a **rule-based, scripted flow** rather than a free-running generative model. This is a design decision for you to approve (see the feasibility document).

## 10. Human-in-the-loop learning and feedback

Interactive machine learning treats the user as a participant in a train–feedback–correct loop and argues for involving users throughout design (Amershi et al., 2014). "Explanatory debugging" proposes that explanations let users correct a model's behaviour (Kulesza et al., 2015; record only). **[Review's inference]** This supports letting users reject a prediction ("this isn't right"), correct an inferred label, and decline recommendations — and recording those responses *as the user's data, never as ground truth about their physiology*.

## 11. What does "digital twin" mean in health?

A scoping review (Tudor et al., 2025) adopts the National Academies definition: a digital twin must be **personalised, dynamically updated and predictive, and must inform decisions**, with a **bidirectional** exchange between the physical and virtual systems. It distinguishes a digital *model* (no automatic data exchange), a digital *shadow* (one-way flow) and a digital *twin* (two-way flow), and reports that only two of the reviewed studies mentioned verification, validation and uncertainty quantification (VVUQ). Other reviews (Katsoulakis et al., 2024; Kamel Boulos & Zhang, 2021) describe the concept and its ethical issues.

**[Review's inference]** Under these definitions the NeuroTwin prototype is best described as a **personal digital shadow with a feedback loop for recommendations**: one-way sensing into a profile, state *estimation* (not forecasting), user feedback altering recommendation order, and limited VVUQ. Calling it a validated "digital twin" would overclaim. The project name can stay; the thesis must define the term and state which parts of the definition are met.

## 12. Probability calibration and uncertainty

Tree ensembles such as random forests tend to produce distorted probabilities (Niculescu-Mizil & Caruana, 2005, as quoted in the scikit-learn documentation), which can be improved by post hoc calibration (sigmoid/Platt or isotonic), the latter needing more data. **[Review's inference]** With 263 windows from 35 participants, calibration must be fitted *inside each training fold* and judged by reliability diagrams and the Brier score on held-out participants. If calibration cannot be shown to work, the dashboard should present an ordinal **evidence level** (e.g. "limited / moderate evidence") rather than a percentage, and should abstain when quality is poor.

## 13. Ethics, safety and human support

The WHO guidance on ethics and governance of AI for health (2021; level B, via secondary summaries) sets six principles, including protecting human autonomy, promoting well-being and safety, and ensuring transparency, explainability and intelligibility. These match the project's requirements (user can reject predictions and decline advice; explanations; safety route). For crisis routing, regional resources differ: for example, India's Tele-MANAS is a free 24×7 mental-health helpline (14416 or 1-800-891-4416) run by the Ministry of Health and Family Welfare, and it is a support line, not an emergency service. **Resource numbers must be configurable by region and re-verified by the owner before any use with real people.**

---

## 14. Synthesis: the gap NeuroTwin can realistically address

**What the sources support:** (i) wearable stress classifiers look good in the laboratory but their generalisation and real-world validity are poorly established; (ii) explainability is usually offered as feature-attribution plots with little user evaluation; (iii) personalisation research relies on longitudinal data; (iv) JITAI design stresses timely, non-interruptive, adaptive support; (v) low-intensity self-help content has some trial evidence when guided; (vi) the digital-twin label is often applied loosely.

**What this targeted search did not find** (not a claim that none exists): an openly reproducible prototype that (1) keeps *sensor measurement*, *model estimate* and *self-report* explicitly separate throughout the interface and data model, (2) pairs each estimate with uncertainty and data-quality flags and can abstain, (3) uses a conversational check-in that does not infer stressors from physiology, (4) selects among evidence-informed coping suggestions using transparent, user-controlled preference rules, and (5) is accompanied by an honest participant-independent evaluation including a majority baseline.

**Defensible contribution.** NeuroTwin should be positioned as an **integrated, transparent, safety-aware research prototype and methodological case study**, not as a validated stress-management intervention and not as a claim of higher accuracy than prior work. Its Part A component (offline evaluation) can be assessed against existing data; its Part B component (conversation, recommendations, feedback) can be *verified as software* and *demonstrated*, but not shown to be effective without a user study (Part C).

---

## References
The full list with DOIs/URLs, what was actually seen, and verification level is in **`references_verified.csv`**. Author-year citations in this text map to the `key` column.
