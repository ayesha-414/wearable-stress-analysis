# Physiological feature formula validation notes

This note records how each of the 23 pipeline features is currently computed in `src/features.py` (and, for ACC, `src/preprocessing.py`), how that relates to a recognised scientific definition where one exists, and what independent synthetic tests in `tests/test_feature_reference_values.py` and `tests/test_feature_reference_stage1.py` can and cannot show.

**Passing those tests does not prove physiological validity.** They only check that the implemented arithmetic matches independently calculated values on small, fully known inputs.

Production formulas were not changed in this stage. Stage 1 review findings (including one confirmed discrepancy) are in the section "Stage 1 review addendum" below.

---

## How to read the “match” column

| Label | Meaning |
|-------|---------|
| Matches standard | Arithmetic on the supplied samples follows the cited definition (units and divisor included). |
| Custom derived | Project-specific summary; no single clinical standard. |
| Heuristic approximation | Named after a standard construct but computed with a simplified proxy. |
| Uncertain / not independently verified | No hand-calculated gold value, or the method diverges from the usual definition in a way that is not fully characterised here. |

---

## Heart rate (from 1 Hz HR)

HR series are resampled to 1 Hz in preprocessing. `hr_features()` then uses pandas `mean` / `std` (Bessel-corrected, `ddof=1`), min, max, and range. Independent tests confirm sample rather than population standard deviation.

| Feature | Current implementation | Standard definition | Units | Match | Limitations | Reference |
|---------|------------------------|---------------------|-------|-------|-------------|-----------|
| `hr_mean` | Arithmetic mean of 1 Hz HR samples | Mean heart rate over the window | bpm | Custom derived (ordinary mean) | Not an IBI-derived mean heart rate. Dropna before the length check (`MIN_SAMPLES = 10`). | Descriptive statistic; no HRV-specific standard. |
| `hr_std` | Sample SD (`n−1`) of 1 Hz HR | SD of HR samples | bpm | Custom derived | Not SDNN (SDNN is SD of NN intervals in ms). | — |
| `hr_min` | Minimum 1 Hz HR | Minimum of the series | bpm | Custom derived | Sensitive to residual artefacts after clipping in `clean_hr`. | — |
| `hr_max` | Maximum 1 Hz HR | Maximum of the series | bpm | Custom derived | Same artefact sensitivity. | — |
| `hr_range` | `max − min` | Range of the series | bpm | Custom derived | Same. | — |
| `hr_diff_rms` | RMS of successive **1 Hz HR** differences: `sqrt(mean(ΔHR²))` | Not a Task Force HRV metric. Task Force RMSSD is RMS of successive **NN interval** differences in ms. | bpm | Custom derived | Must not be interpreted as RMSSD. Saved CSV may still use the legacy column name `hr_rmssd_approx`. Computed only when at least two successive differences exist (`len(diff) > 1`). | Task Force of the ESC and NASPE (1996), *Circulation* 93:1043–1065. [doi:10.1161/01.CIR.93.5.1043](https://doi.org/10.1161/01.CIR.93.5.1043) |

---

## Heart-rate variability (from IBI)

`ibi_features()` expects a DataFrame with `ibi_s` (interval duration in **seconds**) and `offset_s` (time of the beat, seconds). Intervals are converted to milliseconds for SDNN, RMSSD, and pNN50. Rows with NaN `ibi_s` are dropped; fewer than 10 remaining intervals yields all-NaN.

| Feature | Current implementation | Standard definition | Units | Match | Limitations | Reference |
|---------|------------------------|---------------------|-------|-------|-------------|-----------|
| `hrv_mean_ibi` | `mean(ibi_s)` | Mean NN interval | **seconds** in code (often reported as ms) | Matches the mean-NN idea; **unit is seconds, not ms** | Empatica IBI is PPG-derived, not ECG NN. No ectopic-beat rejection. | Task Force (1996) as above; Shaffer & Ginsberg (2017) *Front. Public Health* 5:258. [doi:10.3389/fpubh.2017.00258](https://doi.org/10.3389/fpubh.2017.00258) |
| `hrv_sdnn` | Sample SD (`ddof=1`) of IBI in ms | SDNN: standard deviation of all NN intervals | ms | Matches the usual SDNN **sample-SD** convention on the supplied intervals | Requires a sufficiently long, stationary NN series in clinical use (often 5 min or 24 h). Window here is a protocol **stage**, not a fixed 5 min. PPG IBI ≠ ECG NN. | Task Force (1996), Table 1. |
| `hrv_rmssd` | `sqrt(mean(ΔNN²))` with ΔNN in ms and `mean` over `N−1` differences | RMSSD: square root of the mean of the squares of successive NN differences | ms | Matches the Task Force wording (“mean of the sum of the squares of differences between adjacent NN intervals”) | Some software apply a Bessel correction to the differences; this code does not. No NN50-style artefact filtering. | Task Force (1996), Table 1. |
| `hrv_pnn50` | Proportion of successive \|ΔNN\| **> 50 ms** | pNN50: NN50 divided by a count of NN intervals, where NN50 is pairs of adjacent NN intervals **differing by more than 50 ms**. Task Force Table 1, as quoted by secondary sources, divides by *the total number of all NN intervals* (N); other software divides by the number of pairs (N−1). Production uses **N−1**. | proportion in `[0, 1]`, not a percentage | Matches the **> 50 ms** convention, not `≥ 50 ms`. Independent tests use a sequence where the two conventions disagree (1/9 vs 8/9). | Reported as a fraction, not ×100. Short stages make the estimate noisy. The N vs N−1 choice changes the value by a factor (N−1)/N (e.g. 2/14 vs 2/15 for 15 intervals); it is a convention difference, not an error. | Task Force (1996), Table 1 (NN50 / pNN50). Primary text not read; wording verified through secondary sources only (see verification table). |
| `hrv_lf_hf_ratio` | Mean of a **normalised Lomb–Scargle** periodogram on mean-centred IBI, LF = 0.04–0.15 Hz inclusive, HF = (0.15, 0.4] Hz, then LF/HF | LF/HF from PSD of an interpolated, resampled NN tachogram (FFT or AR), integrating band power | dimensionless | **Confirmed numerical discrepancy with the conventional definition (Stage 1).** Band edges follow common HRV practice, but the code averages periodogram ordinates instead of integrating band power. On the current 500-point grid (138 LF points, 315 HF points) this inflates the ratio by exactly N_HF/N_LF = 315/138 ≈ 2.283. For two equal-amplitude sinusoids (equal power) the band-power ratio is 0.9998, production returns 2.2822. IBI is not interpolated to even sampling (Lomb–Scargle handles uneven times, which is a method difference, not an error). | VLF is not reported. Failure of Lomb–Scargle yields NaN. Because the factor is constant for a fixed grid, the feature is (up to trapezoid-vs-sum endpoint effects) a rescaling of the band-power ratio, so scale-invariant models (tree ensembles; standardised SVM/logistic regression) are not materially affected, but absolute values must not be compared with published LF/HF norms. Stage windows are short (the dataset states the opinion and subtraction tasks had 30 s limits); ultra-short-term studies summarised in Shaffer & Ginsberg (2017, Table 4) report minimum epochs from roughly 20 s to 180 s depending on study and metric (LF/HF about 180 s in one ECG study of healthy students), so LF/HF from ~30 s stages is of doubtful reliability regardless of the formula. Those figures come from ECG studies, not wrist PPG. | Task Force (1996) frequency-domain section; Shaffer & Ginsberg (2017) Table 2 (LF/HF = ratio of LF to HF *power*); Lomb (1976) *Astrophys. Space Sci.* 39:447–462; Scargle (1982) *Astrophys. J.* 263:835–853; SciPy `lombscargle`. Tests: `tests/test_feature_reference_stage1.py::TestLfHfBandAveragingVersusBandPower`. |

---

## Electrodermal activity

EDA is resampled to 1 Hz and clipped to [0, 60] µS before features. SCL/SCR names follow electrodermal terminology but the estimators are simplified.

| Feature | Current implementation | Standard definition | Units | Match | Limitations | Reference |
|---------|------------------------|---------------------|-------|-------|-------------|-----------|
| `eda_mean` | Sample mean of 1 Hz EDA | Mean skin conductance | µS | Custom derived | Not decomposed into tonic/phasic. | Boucsein et al. (2012) *Psychophysiology* 49:1017–1034. [doi:10.1111/j.1469-8986.2012.01384.x](https://doi.org/10.1111/j.1469-8986.2012.01384.x) |
| `eda_std` | Sample SD | SD of skin conductance | µS | Custom derived | Confounds tonic drift and phasic bursts. | Boucsein et al. (2012) |
| `eda_range` | `max − min` | Range | µS | Custom derived | Artefact-sensitive. | — |
| `eda_slope` | OLS slope vs **sample index** (`np.polyfit`, x = 0..n−1) | Linear trend of SCL | µS per sample (≈ µS/s at 1 Hz) | Custom derived | Not time-stamped regression if the index is not 1 Hz after segmentation. | — |
| `eda_scl_mean` | Mean of a centred rolling mean, window = 60 samples, `min_periods=1` | SCL / tonic component from a validated decomposition (e.g. low-pass or convex optimization) | µS | Heuristic approximation | Window is 60 **samples**. At 1 Hz that is 60 s, but the window is not converted from seconds explicitly. For stages shorter than 60 s the “tonic” series is nearly the global mean. | Boucsein et al. (2012); Greco et al. (2016) cvxEDA, *IEEE TBME* 63:797–804. [doi:10.1109/TBME.2015.2474131](https://doi.org/10.1109/TBME.2015.2474131) |
| `eda_scr_count` | Count of **rising edges** where `(EDA − tonic) > 0.02` | Discrete SCRs with minimum amplitude (often 0.01–0.05 µS), rise-time and recovery criteria | count | Heuristic approximation **not** a validated SCR peak detector | No peak interpolation, no duration/latency, no overlapping-SCR separation. Threshold is **strict > 0.02 µS**. Independent tests document this heuristic only. A window whose first sample is already above threshold records no event for it (the first difference is NaN and is skipped); this is documented, scientifically debatable, and not changed. | No primary source verified for the 0.02 µS value. Boucsein et al. (2012) is a publication-reporting recommendation, not an SCR scoring standard; the "0.01–0.05 µS" convention was corroborated only by secondary software documentation. Candidate primary sources to check: Boucsein (2012) *Electrodermal Activity* (2nd ed., Springer); Dawson, Schell & Filion (2007) in *Handbook of Psychophysiology*. |

---

## Skin temperature

| Feature | Current implementation | Standard definition | Units | Match | Limitations | Reference |
|---------|------------------------|---------------------|-------|-------|-------------|-----------|
| `temp_mean` | Sample mean | Mean skin temperature | °C | Custom derived | Wrist temperature, not core temperature. | Empatica E4 TEMP is a thermistor on the wrist; see device documentation. |
| `temp_std` | Sample SD | SD of skin temperature | °C | Custom derived | — | — |
| `temp_slope` | OLS slope vs sample index | Linear trend | °C per sample | Custom derived | Same index-vs-time caveat as EDA slope. | — |
| *(no `temp_range`)* | Not implemented | Range is not extracted | — | — | Range exists for HR and EDA only. | — |

---

## Accelerometer / activity

Empatica ACC is 3-axis, 32 Hz, units of 1/64 g. Features are **not** mean/std/max of Euclidean acceleration magnitude.

Pipeline:

1. `moving_average_acc()` (`src/preprocessing.py`): for each block of 32 samples, sum `max(|Δx|, |Δy|, |Δz|)` relative to the previous sample (zeros before the first block), then exponential smoothing `avg ← 0.9·avg + 0.1·(s/32)`. One value per block (~1 Hz).
2. `acc_features()`: sample mean, sample SD, and max of that 1-D activity index.

| Feature | Current implementation | Standard definition | Units | Match | Limitations | Reference |
|---------|------------------------|---------------------|-------|-------|-------------|-----------|
| `acc_mean` | Mean of the activity index | No single clinical standard. Related to actigraphy / movement counts, not ENMO or Euclidean SVM. | arbitrary (smoothed max-axis delta per sample) | Custom derived | Independent tests show a constant (10,0,0) block yields `1.0`, not Euclidean 10. Scale depends on Empatica integer units and the 0.1 EMA. | Empatica E4 ACC description (PhysioNet Wearable Device Dataset data dictionary); activity index copied from the dataset notebook. |
| `acc_std` | Sample SD of the activity index | — | same | Custom derived | — | — |
| `acc_max` | Max of the activity index | — | same | Custom derived | — | — |

---

## Features not given an independent numeric gold value in this stage

- **`hrv_lf_hf_ratio`**: still has no hand-calculated gold value for its absolute level (Lomb–Scargle output is method-dependent). Stage 1 added an independent Lomb–Scargle reference and showed that band averaging ≠ band-power integration (see addendum).
- **SCR physiology**: tests only reproduce the threshold-crossing heuristic.
- **Real wearable validity**: no comparison to ECG NN, cvxEDA, or published E4 HRV libraries.

---

## Implementation details confirmed by synthetic tests

- `MIN_SAMPLES = 10` applies to HR, EDA, TEMP, ACC summaries, and IBI rows.
- Standard deviations from pandas `Series.std()` match **sample** SD (`n−1`), not population SD (`n`).
- pNN50 uses **`> 50 ms`**, consistent with Task Force “more than 50 ms”.
- `hr_diff_rms` and `hrv_rmssd` differ in input, units, and typical magnitude.
- Temperature has no range feature.
- ACC summaries describe the block/EMA activity index, not raw magnitude.

---

## Stage 1 review addendum

Added after an independent review. Production code was **not** changed. Tests: `tests/test_feature_reference_stage1.py`.

### 1. LF/HF: mean of ordinates vs integrated band power (confirmed)

- Grid: `np.linspace(0.003, 0.4, 500)`, spacing 0.397/499. By exact rational arithmetic, LF (0.04 ≤ f ≤ 0.15) contains **138** points and HF (0.15 < f ≤ 0.40) contains **315**.
- Identity (true for any spectrum): `mean_LF / mean_HF = (sum_LF / sum_HF) × N_HF / N_LF`. Production returns the left side; the conventional band-power ratio corresponds to the sum (or trapezoid) ratio.
- Reference check: a synthetic IBI series with a 0.10 Hz and a 0.25 Hz sinusoid of equal amplitude carries equal power in each band. An independently implemented Lomb–Scargle periodogram (matching SciPy up to a constant) gives band-power ratios of 0.9998 (sum) and 0.9998 (trapezoid). Production returns 2.2822, i.e. 2.2826 × the true ratio.
- Classification: **implementation differs from the conventional definition**. Whether this should be corrected is a decision for the project owner. The test `test_EXPECTED_BY_STANDARD_production_ratio_equals_band_power_ratio` is marked `xfail(strict=True)` and will fail loudly once the formula is corrected, so the marker must then be removed.
- Mutation check: replacing `.mean()` with `.sum()` in a scratch copy made the xfail test report XPASS(strict) and the two characterisation tests fail, as intended.

### 2. IBI RMSSD and pNN50 with dropped or rejected beats (limitation, scientifically uncertain)

- Production differences **consecutive rows**; it never inspects `offset_s` for time-domain HRV and applies no artefact or ectopic rejection. NaN rows are dropped first, so rows on either side of a dropped beat become neighbours.
- Task Force definitions use **adjacent** NN intervals, so a difference spanning a missing beat is not strictly a successive-NN difference.
- Synthetic demonstration (IBI = 0.70 + 0.03k s, all true successive differences 30 ms; one row removed): production RMSSD = 33.28 ms (reference 30 ms, +10.9 %), pNN50 = 1/13 (reference 0).
- Missed-beat demonstration (`[0.8]*7 + [1.6] + [0.8]*7`): the series has zero variability but production gives RMSSD = 302.37 ms and pNN50 = 2/14, because one artefact produces two large differences.
- **Unresolved:** the dataset README defines the second IBI column as the interval from the previous beat, but does not say how a rejected beat is represented (a long interval, or an interval from the last accepted beat). How often either case occurs in the real recordings cannot be determined without the raw files, so the practical impact on the reported results is unknown.

### 3. SCR threshold behaviour

- The earlier "exactly 0.02" boundary test was a test-design flaw (floating-point value 0.020000000000000018). It was replaced by a test with ±0.001 margin. Strictness at exactly 0.02 is verified with a zero tonic, where `0.02 − 0.0` is exact.
- First-sample behaviour is documented above (no event for an initially high first sample).

### 4. Reference verification status

| Reference | What was verified | What was not verified |
|-----------|-------------------|-----------------------|
| Task Force (1996), *Circulation* 93(5):1043–1065, doi:10.1161/01.CIR.93.5.1043 | Bibliographic details (PubMed PMID 8598068). Table 1 wording for SDNN, RMSSD, NN50 and pNN50 as quoted by several secondary sources. | Primary text could not be opened; exact wording and recording-length guidance were not read in the original. |
| Shaffer & Ginsberg (2017), *Front Public Health* 5:258, doi:10.3389/fpubh.2017.00258 | Bibliographic details; LF/HF defined as ratio of band power; Table 4 ultra-short-term epochs. | Individual numbers in Table 4 were not cross-checked against the cited primary studies. |
| Boucsein et al. (2012), *Psychophysiology* 49(8):1017–1034, doi:10.1111/j.1469-8986.2012.01384.x | Bibliographic details and that it is an SPR committee report on publication recommendations. | That it prescribes any SCR amplitude threshold. Do not cite it for the 0.02 µS value. |
| Greco et al. (2016) cvxEDA, *IEEE TBME* 63(4):797–804, doi:10.1109/TBME.2015.2474131 | Bibliographic details and that it models SC as phasic + tonic + noise via convex optimisation. | Not used by the code; cited only as a reference for a validated decomposition. |
| Lomb (1976), *Astrophys Space Sci* 39(2):447–462, doi:10.1007/BF00648343 | Bibliographic details. | — |
| Scargle (1982), *Astrophys J* 263:835–853 (added) | Citation as listed by secondary sources. | Original not opened. |
| PhysioNet Wearable Device Dataset v1.0.1 (Hongn, Bosch, Prado, Bonomini, 2025), doi:10.13026/he0v-tf17 | Title, authors, version, date and DOI; 36 stress volunteers; IBI column definitions; opinion and subtraction tasks limited to 30 s. | — |
| Hongn et al. (2025), *Scientific Data* 12, doi:10.1038/s41597-025-04845-9 (added) | Listed on the dataset page as a companion publication. | Content not read. |

## References

1. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*. 1996;93(5):1043–1065. https://doi.org/10.1161/01.CIR.93.5.1043
2. Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. *Front Public Health*. 2017;5:258. https://doi.org/10.3389/fpubh.2017.00258
3. Boucsein W, et al. Publication recommendations for electrodermal measurements. *Psychophysiology*. 2012;49(8):1017–1034. https://doi.org/10.1111/j.1469-8986.2012.01384.x
4. Greco A, Valenza G, Lanata A, Scilingo EP, Citi L. cvxEDA: a convex optimization approach to electrodermal activity processing. *IEEE Trans Biomed Eng*. 2016;63(4):797–804. https://doi.org/10.1109/TBME.2015.2474131
5. Lomb NR. Least-squares frequency analysis of unequally spaced data. *Astrophys Space Sci*. 1976;39:447–462. https://doi.org/10.1007/BF00648343
6. Hongn A, Bosch F, Prado L, Bonomini P. Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (version 1.0.1). PhysioNet. 2025. https://doi.org/10.13026/he0v-tf17
7. Scargle JD. Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data. *Astrophys J*. 1982;263:835–853.
8. Hongn A, Bosch F, Prado LE, Ferrández JM, Bonomini MP. Wearable physiological signals under acute stress and exercise conditions. *Sci Data*. 2025;12. https://doi.org/10.1038/s41597-025-04845-9
