# Claims register

Wording rules for the dashboard, report and presentation. `src/neurotwin/language.py` encodes the "not permitted" patterns and the tests scan every user-facing string.

| Permitted | Not permitted |
|---|---|
| "The model's estimate is a pattern associated with higher self-reported stress; evidence is limited." | "You are stressed." / "You have stress." / "stressed person" |
| "The model weighted your average heart rate toward 'higher'." | "Your heart rate shows you are stressed because …" |
| "You have reported study-related stress on several occasions." | "Studying causes your stress response." |
| "Some people find this exercise helpful. You can skip it." | "This will reduce your stress." / "proven to …" / "guaranteed" |
| "Your recorded baseline" | "Your normal range" / "healthy range" |
| "Evidence is limited because signal quality is low." | Any stress statement when quality is insufficient |
| "Research prototype; not a medical device." | "Clinically validated", "diagnoses", "treats" |
| "Accuracy 0.570 vs 0.616 for always guessing the majority class." | Any accuracy figure without the majority baseline |
| "In a simulation with invented users, the adapted ordering matched their invented preferences." | "Personalisation improves outcomes." |

## Claim ladder

| Level | Claim | Status |
|---|---|---|
| L0 | Software behaves as specified | Yes (automated tests) |
| L1 | Participant-independent offline performance on one laboratory dataset | Yes, and it is weak (see SYSTEM_OVERVIEW §5) |
| L2 | Usability / acceptability with consenting users | **Not done** (needs ethics approval) |
| L3 | Suggestions reduce stress | **Not done**; out of scope |

Part B (conversation, recommendations, feedback, profile) reaches **L0 only**.

## Terminology

With the National Academies definition used in the digital-twin literature (personalised, dynamically updated, predictive, decision-informing, two-way exchange), NeuroTwin is a **personal digital shadow with a feedback loop for recommendations**. It estimates a current state rather than forecasting, sensing is one-way, and verification/validation/uncertainty quantification is limited. It is not claimed to be a validated digital twin.
