# Limitations and future work

## Scientific
1. **The estimator is weak.** Held-out accuracy (0.570) is below always guessing the majority class (0.616); AUC is 0.539; probabilities are not calibrated. The app therefore caps evidence at "limited", shows no percentages and never interrupts the user.
2. **The label is a thresholded self-rating**, not objective stress. In the dataset, 26 of 105 rest/baseline windows were rated ≥5 and 83 of 158 task windows were rated <5; 9 of 35 participants never rated ≥5, 4 always did, and only 22 have both classes. Individual use of the scale probably contributes.
3. **One laboratory dataset**: 36 healthy volunteers aged 18–30, structured tasks, single verbal 1–10 rating per stage. No questionnaires, sleep, real-life stressors, long-term history or intervention outcomes. Findings should not be generalised to daily life, other ages or clinical groups.
4. **Explanations describe the model**, not physiology or causes. Alternative explanations (movement, heat, caffeine, illness, sensor fit) cannot be ruled out; only movement and skin-temperature flags exist, and no exertion classifier has been built.
5. **LF/HF** is held back from explanations because its band-averaging inflates the ratio by about 2.28× versus band-power integration (Stage 1 finding). The model itself still includes it.
6. **Personal baseline** is a single recorded laboratory window, not a longitudinal baseline and not a clinical norm. No personalised model has been trained.
7. **The recommender is rule-based.** Catalogue items rely on public-health advice and on WHO / meta-analytic sources; the strongest trial evidence (Self-Help Plus) concerns facilitator-guided group delivery in distressed refugee women, not an unguided app for healthy people. No efficacy is claimed.
8. **Adaptation is verified only with simulated users** whose preferences were invented. It shows the code works as specified, not that adaptation helps.

## Technical
* The profile database is unencrypted and has no authentication.
* Free-text crisis screening is a keyword backstop with false negatives.
* The Streamlit pages use `use_container_width`, which newer Streamlit versions deprecate (as does the existing dashboard code).
* `requirements.txt` previously lacked `plotly`, `scipy` and `pytest`, which the code imports; they are now listed.
* The README says `f14` sessions are "concatenated"; the loader actually drops `f14` (known audit item, not changed here).

## Ethical
* Crisis resources are unverified data and **must be verified by the owner** before any use with real people.
* No user study has been done. Any pilot needs ethics approval, a data-protection review and supervisor sign-off.

## Future work (Part C)
Longitudinal study with consenting users and repeated check-ins; comparison of static vs adaptive recommendations in a micro-randomised design; user evaluation of explanations; resolve the open Part A items (f14, LF/HF, baseline normalisation, feature selection); consider an exertion detector using the dataset's aerobic and anaerobic sessions.
