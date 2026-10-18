# Research Decisions — CrystalSym-Spec

**Last updated:** 2026-10-13

---

## Decision log

### Decision 1: RELAB is the only spectral library

- **Decision:** Use RELAB exclusively as the spectral data source.
- **Reason:** Project scope focuses on a single, well-characterized laboratory spectral library. Combining multiple libraries (e.g., RELAB + RRUFF + USGS) introduces inter-instrument calibration issues that are not yet addressed.
- **Evidence:** Phase 1 audit confirmed 1,017 `.tab` files across 20 mineral directories in `data/raw/RELAB/`.

### Decision 2: Raw data are immutable

- **Decision:** No raw `.tab` file in `data/raw/RELAB/` may be modified, deleted, or overwritten.
- **Reason:** Scientific reproducibility requires an immutable reference. All derived data must be traceable to the original files via MD5 hashes.
- **Evidence:** All 1,017 files verified readable with valid structure in Phase 1 audit.

### Decision 3: CF files are retained for provenance but excluded from mineral-spectrum analysis

- **Decision:** The 92 `cf` (container/filter blank) files remain in `data/raw/RELAB/` and are documented in the provenance table, but are flagged as excluded from future mineral-spectrum scientific analysis.
- **Reason:** CF files are background/blank measurements of the container/filter material (metadata: "CF <2 um", sample IDs ending in `-CF`). They are not mineral spectra. Including them would introduce non-physical spectra and create spurious cross-mineral duplicates.
- **Evidence:** Phase 1 audit (`relab_cf_investigation.md`): all 92 files have sample IDs ending in `-CF`, descriptions "CF <2 um", and all share researcher code C1M. `Diopside/cfpp108.tab` was investigated and confirmed NOT to be a CF blank (its sample ID is `PP-ALS-108`, a real diopside measurement). High confidence.

### Decision 4: Exact duplicate files are retained but cannot count as independent observations

- **Decision:** The 84 exact duplicate groups (comprising 168 files) remain in the raw data and are documented, but future analyses must not treat byte-identical spectra as independent evidence.
- **Reason:** Byte-identical files represent the same physical measurement stored in multiple mineral directories. Counting them as independent would inflate sample size and create train/test leakage.
- **Evidence:** Phase 1 audit (`phase1_step1_summary.md`): 84 groups of files with identical MD5 hashes, spanning 5 cross-mineral group types.

### Decision 5: Repeated measurements are retained

- **Decision:** Multiple measurements of the same specimen/sample are all retained as separate measurement records.
- **Reason:** Different measurement conditions (grain size, geometry, instrument) provide valuable scientific information. Prematurely discarding measurements would lose information. The specimen/sample grouping preserves the relationship.
- **Evidence:** ~300 sample IDs appear in 2-8 files each (Phase 1 audit `relab_sample_id_structure.md`).

### Decision 6: Sample/specimen grouping is preserved

- **Decision:** All measurements sharing the same RELAB sample ID receive the same `specimen_group_id` for future group-aware analysis.
- **Reason:** Measurements of the same specimen are not statistically independent. Group-aware splitting prevents data leakage in ML evaluation.
- **Evidence:** 604 unique RELAB sample IDs; 629 specimen/provenance groups; ~300 groups with multiple measurements (Phase 1 audit).

### Decision 7: Grain-size/measurement variants are retained initially

- **Decision:** Different grain-size fractions and measurement-condition variants are retained as separate records. Measurement conditions are recorded as metadata.
- **Reason:** Grain-size effects on spectral shape are scientifically meaningful. The choice of which variant to use (or whether to use all) depends on the research question and should be made explicitly, not by default.
- **Evidence:** Filename suffixes (a-g) indicate grain-size fractions; metadata records measurement geometry and instrument configuration.

### Decision 8: Mineral directory names are recorded as RELAB labels, not independently verified mineral identities

- **Decision:** The mineral directory name (e.g., "Augite", "Calcite") is recorded as the RELAB-observed label. No independent mineral identification is assumed.
- **Reason:** The directory name is the only indicator of mineral identity in the raw data. No XRD, chemical analysis, or other verification method is documented in the files. Cross-mineral duplicates (49 pyroxene pairs) demonstrate that mineral classification is not always unambiguous.
- **Evidence:** Phase 1 audit (`phase1_provenance_assessment.md`): directory name is the sole mineral identifier; 49 cross-mineral pyroxene duplicates reflect solid-solution ambiguity.

### Decision 9: Chemistry and crystal structure will be linked later

- **Decision:** No external chemistry or crystallographic data are obtained at this stage. These will be linked only after the canonical provenance structure is fully established.
- **Reason:** Chemistry and crystal structure are central to the project hypothesis but must be obtained from external sources (RELAB database, RRUFF, Mindat, literature). Linking them prematurely, before the provenance layer is validated, risks propagating errors.
- **Evidence:** Phase 1 audit confirmed no chemistry or crystal structure data in raw files (`relab_metadata_availability.md`).

### Decision 10: Wavelength resampling has not yet been defined

- **Decision:** No wavelength grid resampling, interpolation, or standardization has been performed.
- **Reason:** The dataset has 89 distinct wavelength grids with varying ranges and resolutions. The choice of target grid depends on the spectral range of scientific interest and the ML method. This decision requires explicit researcher input.
- **Evidence:** Phase 1 audit (`relab_raw_inventory.md`): 89 distinct (wl_min, wl_max, n_points) grids.

### Decision 11: Measurement uncertainty is retained as metadata

- **Decision:** The third spectral column (present in 461/1017 files) is retained as measurement uncertainty metadata. It is not used as an ML predictor at this stage.
- **Reason:** The third column exhibits properties consistent with measurement uncertainty (always positive, small relative to reflectance, larger at shorter wavelengths). Its exact statistical meaning (1-sigma, 2-sigma, standard error) is not documented. Using it as a predictor before understanding its meaning would be premature.
- **Evidence:** Phase 1 audit (`relab_column_interpretation.md`): numerical behavior consistent with uncertainty; no official documentation of statistical meaning.

### Decision 12: No ML split may separate measurements from the same specimen/sample group

- **Decision:** Future train/test splits must keep all measurements from the same specimen group together.
- **Reason:** Measurements of the same specimen share physical material and are correlated. Separating them across train/test sets creates data leakage and invalidates performance estimates.
- **Evidence:** Established in `docs/methodology.md`. Supported by Phase 1 provenance assessment.

### Decision 13: No derived scientific features are yet considered validated

- **Decision:** No spectral complexity measure, feature, or derived quantity is considered validated until its computation is documented, verified, and its scientific meaning is established.
- **Reason:** The project aims to relate crystal symmetry and chemistry to spectral complexity. "Complexity" must be precisely defined and its computation verified before it can be used as a target variable.
- **Evidence:** Project has not yet entered the feature-engineering or analysis phase.

### Decision 14: Mineral identity is initially a provenance variable, not the primary ML target

- **Decision:** Mineral directory names are treated as provenance/grouping labels during Phase 1. They are not assumed to be the primary ML target variable.
- **Reason:** Mineral identity in RELAB is assigned by the submitter and recorded only as the directory name. Cross-mineral duplicates (e.g., 49 pyroxene pairs across Augite/Diopside/Hypersthene/Enstatite) demonstrate that mineral classification is not always unambiguous. The project hypothesis concerns crystal symmetry and chemistry, not mineral classification per se. The appropriate target variable for ML must be defined after chemistry and crystallography data are linked.
- **Evidence:** Phase 1 provenance assessment; cross-mineral duplicate analysis.

### Decision 15: Leakage control must operate at the provenance/sample-group level

- **Decision:** All future train/test splits, cross-validation folds, and performance evaluations must enforce independence at the `specimen_group_id` level, not the individual file level.
- **Reason:** Multiple measurements of the same specimen (same physical material, different grain sizes, instruments, or dates) are correlated. Splitting at the file level would place correlated measurements in both training and test sets, inflating performance estimates and invalidating scientific conclusions. The `specimen_group_id` in `relab_measurements.csv` defines the correct unit of independence.
- **Evidence:** 629 specimen groups; ~300 groups contain 2-8 measurements each (Phase 1 audit).

---

## Phase 2 Step 2 — Spectral Domain and Target Grid — FROZEN

**Date frozen:** 2026-09-24

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** RELAB-only (925 candidates; 800 Category A + 125 Category B)

### Final decision

| Parameter | Value |
|---|---|
| Target spectral domain | 320–2550 nm |
| Target wavelength spacing | 5 nm |
| Expected target grid points | 447 |
| Spectra fully covering domain (D2) | 454 / 925 |
| D2 spectra natively supporting 5 nm | 390 / 925 |
| D2 spectra requiring resampling from native 10 nm | 64 / 925 |
| Minerals represented in D2 | 18 / 20 |
| Provenance groups covered in D2 | 382 / 602 |
| Provenance groups excluded from D2 | 220 / 602 |

### Population counts

- **454** of 925 candidates fully cover 320–2550 nm.
- **390** of those 454 natively support the 5 nm grid (median native spacing ≤ 5 nm).
- **64** of those 454 do not natively support 5 nm; all 64 have native 10 nm spacing. The 64 non-native spectra do not show evidence of systematic subgroup bias relative to the native-5-nm group (same dominant acquisition regime: VNIR spectrophotometer, 98% in both groups).
- **18** of 20 minerals are represented in D2.
- **Pyrope** (3 candidates) and **Zircon** (2 candidates) are excluded because their wavelength coverage does not extend to 320 nm.
- **220** of 602 provenance groups are excluded from D2. Exclusions are concentrated in Enstatite (81 groups), Olivine (39 groups), and Spinel (31 groups).

### Rationale for 5 nm spacing

5 nm was selected because it provides substantially greater native support (390 spectra, 42.2% of candidates) than 2 nm (20 spectra, 2.2%) while retaining finer sampling than 10 nm (which, although supported by all 454 D2 spectra, would discard fine spectral features relevant to complexity measurement at 224 output points).

### Nature of the domain boundaries

320 nm and 2550 nm are **pragmatic empirical coverage boundaries**, not fundamental physical boundaries. They align with the two most common native grids in the dataset (300–2600 nm / 461 points and 320–2550 nm / 447 points) and maximise population coverage while capturing the full visible and near-IR diagnostic range relevant to the project hypothesis.

### Scientific interpretation

> The selected domain and grid define the standardized spectral analysis population for subsequent preprocessing and modelling. They do not imply that spectral information outside 320–2550 nm is scientifically unimportant.

All later conclusions must be interpreted relative to the D2-supported population (454 spectra, 18 minerals, 382 provenance groups). Findings cannot be generalized to Pyrope, Zircon, or to provenance groups excluded by the coverage requirement without explicit re-validation.

### Limitations

The following limitations are formally recorded:

1. **Coverage-selected subset:** D2 represents a wavelength-coverage-selected subset (454/925 spectra) rather than the entire RELAB candidate population. Conclusions are conditional on this subset.

2. **Interpolation required for 64 spectra:** 64 of the 454 D2-covered spectra require interpolation/resampling from native 10 nm data to the 5 nm target grid. The median interpolation fraction is approximately 0.50, with a maximum interpolation distance of 5 nm. The effect of linear interpolation on complexity metrics has not yet been quantified.

3. **Two minerals absent:** Pyrope and Zircon are absent from D2 because of wavelength coverage (Pyrope: all 3 candidates start at 350 nm; Zircon: all 2 candidates start at 1800 nm).

4. **Non-uniform exclusion of provenance groups:** 220 excluded provenance groups are not uniformly distributed across minerals. Exclusions are concentrated in Enstatite (81 groups, 36.8% of exclusions), Olivine (39 groups), and Spinel (31 groups). This reflects the wavelength-coverage structure of these minerals' measurements (predominantly FTIR with narrower VNIR coverage) rather than a methodological bias.

5. **Pragmatic boundary choice:** The 320–2550 nm boundaries are pragmatic coverage-based choices rather than physically derived universal mineral-spectroscopy boundaries. They do not correspond to specific physical absorption features.

6. **Complexity metric sensitivity to spacing:** Spectral-complexity descriptors may be sensitive to wavelength spacing. Peak density (CV = 24.1%) and second-order variation (CV = 8.5%) show spacing-dependent behavior across 1–10 nm test grids. Complexity features must be evaluated for grid-related sensitivity during later preprocessing/feature engineering.

### Audit trail

This frozen decision is supported by the following validation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/domain_coverage_comparison.csv`
- `results/02_preprocessing_qc/grid_spacing_comparison.csv`
- `results/02_preprocessing_qc/phase2_step2_domain_grid_report.md`
- `results/02_preprocessing_qc/phase2_step2_decision_validation.md`
- `results/02_preprocessing_qc/phase2_step2_decision_validation.csv`
- `results/02_preprocessing_qc/phase2_step2_final_freeze.md`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- No processed spectral dataset was created during Step 2.
- No interpolation or resampling was performed.
- No spectral features were engineered.
- No ML or PySR analysis was performed.

---

## Phase 2 Step 3 — Duplicate-Wavelength Treatment — FROZEN

**Date frozen:** 2026-09-25

**Decision classification:** B — Differences demonstrably negligible

**Analysis population:** 925 candidates (800 Category A + 125 Category B)

### Final decision

**Rule:** Keep the first occurrence of each duplicated wavelength.

> **Deterministic duplicate-wavelength resolution by retaining the first occurrence, justified by the demonstrated negligible effect of the duplicate differences at the frozen 5-nm target resolution.**

This is a deterministic preprocessing representation, NOT a claim that the first measurement is physically superior. Detector/channel identity remains unknown (Classification D — no defensible identification).

### Evidence summary

- **29 files** contain duplicate wavelength observations (all Category B).
- **8,629 duplicate locations** identified.
- **Two distinct patterns:**
  - Type 1: 9 files with 1 duplicate each (9 locations); 6/9 outside the frozen 320-2550 nm domain.
  - Type 2: 20 files with 431 duplicates each (8,620 locations), all concentrated in approximately 833-1053 nm.
- All 20 Type 2 files share identical duplicate wavelength positions, indicating an acquisition/instrument-level pattern rather than a sample-specific phenomenon.
- Type 2 spectra use 0.1-nm native spacing in the duplicate region, compared with 0.2-nm spacing for corresponding non-duplicate `bir1` spectra.
- No RELAB metadata identifies detector/channel identity for any duplicate observation.
- Documented RELAB splice boundaries at 3700 nm and 20400 nm do not coincide with the duplicate region.
- Signed differences are approximately zero-centered: 50.3% positive, 48.8% negative.
- Mean absolute difference = 0.000541 reflectance units.
- Maximum absolute difference = 0.0024 reflectance units.
- Estimated maximum impact on the final 5-nm grid: < 0.002 reflectance units.
- Mean impact on the 5-nm grid: < 0.00002 reflectance units.

### Scientific interpretation

There is no evidence supporting a scientifically defensible preference for one duplicate observation over another. The differences are negligible relative to the frozen 5-nm target representation. The choice between "first" and "second" is arbitrary but inconsequential.

### Limitations

1. Detector/channel identity is unknown. The duplicates likely arise from overlapping detector channels in the near-IR spectrometer, but this cannot be confirmed from available metadata.
2. The "first occurrence" rule is deterministic, not physical. "Second occurrence" or "average" would have equivalent scientific validity.
3. The Type 2 duplicate pattern is instrument-level (identical positions across 20 files) but does not constitute confirmed detector-overlap identification.

### Audit trail

- `results/02_preprocessing_qc/phase2_step3_duplicate_wavelength_investigation.md`
- `results/02_preprocessing_qc/phase2_step3_duplicate_wavelength_investigation.csv`
- `results/02_preprocessing_qc/phase2_step3_duplicate_wavelength_freeze.md`

### Operational constraints at freeze

- No raw `.tab` files were modified.
- All 1,017 raw checksums verified unchanged (0 mismatches).
- No processed spectral dataset was created during this step.
- No interpolation was performed.
- No smoothing or normalization was performed.
- No duplicate handling has been implemented yet. The rule is frozen but not yet applied.

---

## Phase 2 Step 3 — Unsorted-Wavelength Treatment — FROZEN

**Date frozen:** 2026-09-26

**Decision classification:** A — Ordering-only correction

**Analysis population:** 925 candidates (800 Category A + 125 Category B); 4 spectra affected

### Final decision

**Rule:** For spectra containing an ordering-only wavelength anomaly, wavelength-value pairs may be sorted by ascending wavelength in memory, while preserving the original wavelength-reflectance and wavelength-uncertainty pairings.

> **Deterministic in-memory ordering correction for the four identified spectra, justified by the demonstrated absence of any spectral transformation, interpolation, averaging, or deletion.**

This is an ordering correction only. It is NOT interpolation, smoothing, averaging, spectral correction, detector correction, splice correction, or deletion of observations.

### Evidence summary

- **4 spectra** were identified with unsorted wavelengths.
- All 4 are `397`-prefix Category B files.
- All 4 were acquired during 25–27 March 1997 under identical instrument settings.
- All 4 contain exactly **one ordering violation**.
- The violation occurs at **row 1048 (0-based)** in every case.
- The wavelength sequence is **3305.4 nm followed by 3304.1 nm**.
- The ordering reversal is therefore **1.3 nm**.
- The anomaly is approximately **4.1 nm** from the documented 3300-nm splice.
- The identical pattern across all four files and proximity to the documented splice strongly suggest an acquisition/splice-boundary ordering artifact.
- This interpretation is an **inference**, not a confirmed detector/channel identification.
- No RELAB detector/channel metadata establishes the specific detector identity.
- All 4 violations lie **outside the frozen D2 domain** of 320–2550 nm.
- None of the 4 spectra covers the complete D2 domain.
- Therefore none of the 4 spectra belongs to the primary 454-spectrum D2 analysis population.
- All 4 files passed the basic numerical/data-integrity checks: no NaN/Inf values, no negative wavelengths, no negative reflectance, declared row counts matched parsed data rows, no duplicate wavelengths.
- In-memory sorting moves exactly **two values per file** by one row.
- All original wavelength-reflectance pairings are preserved.
- Sorting produces a strictly non-decreasing wavelength sequence.
- No interpolation, averaging, deletion, smoothing, or normalization is required.

### Scientific interpretation

The identical ordering anomaly in all four files, together with its proximity to the documented 3300-nm splice, is consistent with a splice-boundary acquisition/order artifact. The specific detector/channel mechanism cannot be established because detector/channel metadata are unavailable.

### Distinction: ordering correction vs. spectral transformation

Sorting wavelength-value pairs by ascending wavelength is a **representation correction**, not a spectral transformation. It changes the row order of existing data without modifying any wavelength, reflectance, or uncertainty value. No new values are created, no values are removed, and no pairings are altered. This is categorically distinct from interpolation, smoothing, normalization, detector correction, splice correction, or any operation that modifies spectral content.

### D2-domain relevance

All 4 affected spectra start at 900.4 nm, which is above the D2 lower boundary of 320 nm. None covers the full 320–2550 nm domain. The ordering violations occur at approximately 3304–3305 nm, well above the D2 upper boundary of 2550 nm. Therefore these 4 spectra are **outside the primary D2 analysis population** of 454 spectra, and the frozen treatment rule has no effect on the D2 population.

### Limitations

1. **Detector/channel identity unknown:** The RELAB metadata does not identify which detector or channel produced each observation. The ordering anomaly is consistent with a splice-boundary artifact but cannot be confirmed as a specific detector-concatenation error.
2. **Prefix 397 meaning unconfirmed:** The `397` filename prefix may denote a measurement campaign or instrument configuration, but no project documentation confirms this.
3. **Inference, not proof:** The splice-boundary interpretation is an inference based on the identical pattern across 4 files and the proximity to the documented 3300-nm splice. It must not be described as a definitively proven detector-concatenation error.

### Raw-data immutability

No raw `.tab` files were modified. The sorting rule applies only to in-memory operations during later preprocessing. Raw files remain the immutable reference for all derived data.

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/phase2_step3_unsorted_wavelength_investigation.md`
- `results/02_preprocessing_qc/phase2_step3_unsorted_wavelength_investigation.csv`
- `results/02_preprocessing_qc/phase2_step3_unsorted_wavelength_freeze.md`

### Operational constraints at freeze

- No raw `.tab` files were modified.
- No sorting was implemented.
- No processed spectra were created.
- No interpolation was performed.
- No smoothing was performed.
- No normalization was performed.
- No feature engineering was performed.
- No ML/PySR was performed.

---

## Phase 2 Step 3 — Splice-Boundary Treatment — FROZEN

**Date frozen:** 2026-09-28

**Decision classification:** A — No material discontinuity detected

**Analysis population:** 925 candidates (800 Category A + 125 Category B)

### Final decision

**Rule:** No splice-specific masking, correction, or exclusion will be applied to documented RELAB splice boundaries within the frozen 320–2550 nm analysis domain.

> **No splice-specific treatment is justified by the available evidence for documented RELAB splice boundaries within the frozen 320–2550 nm analysis domain.**

This decision does NOT mean that splice boundaries are physically guaranteed to be artifact-free. It means that the available evidence indicates no material discontinuity requiring special treatment in the selected analysis domain.

### Investigation scope

- All **925 RELAB analysis candidates** were examined.
- Splice metadata were extracted from RELAB raw file footers using documented patterns ("Connected at" and "Spliced" metadata).
- For each D2-relevant splice boundary: absolute discontinuity, relative discontinuity, local spectral variation (±25 nm window), discontinuity-to-variation ratio, local slope analysis, and 5-nm interpolation gap analysis were computed.
- Diagnostic assessment of implications for later 5-nm interpolation was performed.
- Potential sensitivity of derivative-, curvature-, peak-, variation-, and entropy-like spectral-complexity descriptors was considered.

### Evidence summary

- **249 spectra (26.9%)** of 925 candidates contain documented splice metadata.
- **367 total documented splice events** were identified.
- **96.7%** of documented splice events occur outside the frozen D2 domain.
- Only **12 documented splice boundaries** are relevant to D2 (320–2550 nm): 11 inside D2 + 1 near D2 above.
- These 12 events occur in 12 distinct spectra (no spectrum has multiple D2-relevant splices).
- All 12 D2-relevant boundaries were individually evaluated.
- All 12 were classified as **Category A: no material discontinuity detected**.
- **Maximum observed absolute discontinuity:** 0.00123 reflectance units (F0361, Diopside, 2500 nm).
- **Maximum discontinuity/local-variation ratio:** 0.67 (F0479, Enstatite, 2450 nm).
- Every observed discontinuity was smaller than ordinary local spectral variation for every investigated boundary.
- No D2-relevant splice demonstrated evidence requiring masking, correction, or exclusion.

### Scientific interpretation

> Within the frozen 320–2550 nm analysis domain, the documented RELAB splice boundaries investigated in this dataset do not show material discontinuities relative to ordinary local spectral variation. Therefore no splice-specific correction, masking, or exclusion is justified by the available evidence.

The following statements must NOT be made:

- That all RELAB splice boundaries are harmless.
- That detector transitions are physically perfect.
- That splice artifacts can never affect spectral complexity.
- That the absence of observed discontinuity proves the instrument has no splice effects.

The conclusion applies specifically to: **this RELAB dataset + the documented splice metadata + the frozen 320–2550 nm analysis domain.**

### Limitations

1. **Observed:** The 12 D2-relevant documented boundaries show negligible measured discontinuities.
2. **Inferred:** They therefore do not require special treatment under the current preprocessing protocol.
3. **Not established:** The investigation does not establish the physical detector/channel mechanism behind every RELAB splice.
4. **Detector/channel identity unknown:** RELAB metadata does not consistently document which detector or channel was used on each side of a splice boundary.
5. **Splice metadata completeness:** Only 249/925 spectra (26.9%) have documented splice metadata. Undocumented splices, if they exist, were not investigated.
6. **Generalizability:** This investigation applies only to the RELAB dataset as currently configured.

### Raw-data immutability

All 1,017 raw `.tab` files in `data/raw/RELAB/` remain untouched. No raw files were modified during the investigation or this freeze.

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/phase2_step3_splice_boundary_investigation.md`
- `results/02_preprocessing_qc/phase2_step3_splice_boundary_investigation.csv`
- `results/02_preprocessing_qc/phase2_step3_splice_boundary_freeze.md`
- `scripts/phase2_step3_splice_boundary_investigation.py`

### Operational constraints at freeze

- No raw files were modified.
- No splice correction was implemented.
- No splice regions were masked.
- No spectra were excluded because of splice boundaries.
- No interpolation was performed.
- No smoothing was performed.
- No normalization was performed.
- No feature engineering was performed.
- No ML/PySR was performed.

---

## Phase 2 Step 3 — Interpolation Method — FROZEN

**Date frozen:** 2026-09-28

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 454 D2-compatible spectra (390 native-5 nm + 64 native-10 nm)

### Final decision

**Rule:** Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation as the primary interpolation method for the 64 native-10-nm spectra when they are later resampled onto the frozen 5-nm D2 grid.

> **PCHIP interpolation is the primary method for resampling the 64 native-10-nm D2 spectra onto the frozen 320–2550 nm / 5-nm target grid, justified by slightly lower reconstruction error, better shape preservation, and near-zero bias in masked-reconstruction validation using 390 native-5-nm spectra.**

This is a preprocessing representation choice, NOT a claim that PCHIP reconstructs the unknown true physical spectrum better than every possible interpolation method.

### The 64 native-10-nm spectra

- **64** of the 454 D2-compatible spectra have native 10-nm spacing.
- All 64 have **uniform** 10-nm spacing with no anomalous internal gaps.
- All 64 can therefore be represented on the frozen 5-nm target grid using interpolation.
- **11** of 18 D2-covered minerals are represented among the 64 spectra.
- **56** unique provenance groups among the 64 spectra.
- 63 of 64 are VNIR spectrophotometer measurements; the 64 do not form a systematic acquisition-regime subgroup.

### Validation design

Validation used the **390 native-5-nm spectra** by masking every second 5-nm observation and reconstructing the masked observations using both linear and PCHIP interpolation. The actual 5-nm values at masked locations served as ground truth.

### Quantitative evidence

| Criterion | Linear | PCHIP |
|---|---:|---:|
| MAE | 0.000551 | **0.000525** |
| RMSE | 0.000944 | **0.000887** |
| Bias (mean signed error) | −0.000018 | **−0.000011** |
| Max absolute error | 0.007132 | **0.006604** |
| 95th percentile absolute error | 0.001575 | **0.001438** |

Per-spectrum MAE comparison: Linear better in 162 spectra, PCHIP better in 228 spectra.

### Shape-preservation evidence

- Neither method produced evidence of systematic spurious extrema.
- PCHIP better preserves spectral shape: captures curvature within intervals while preserving monotonicity.
- Linear interpolation systematically flattens spectral features (piecewise linear cannot capture curvature).
- PCHIP misses fewer extrema (12076 vs 12330 total).

### Mineral/provenance dependence

No systematic mineral-level or provenance-group dependence was detected. Variation in error across minerals reflects differences in spectral complexity rather than a method-specific bias.

### Frozen PCHIP decision

The interpolation will be:

- performed only in memory during preprocessing;
- applied only to the 64 spectra that genuinely require interpolation;
- restricted to the frozen D2 domain of 320–2550 nm;
- performed using the measured native wavelength–reflectance pairs;
- deterministic;
- reproducible;
- documented in the preprocessing provenance.

### Retention of linear interpolation as a sensitivity method

> Linear interpolation will be retained as a sensitivity-analysis alternative where appropriate.

The primary preprocessing method is PCHIP. Linear interpolation may later be used to determine whether scientific conclusions are sensitive to the interpolation choice. This sensitivity analysis has NOT been performed and is NOT required as part of this freeze.

### Complexity-preservation limitation

The validation experiment demonstrates that reducing 5-nm measurements to 10-nm sampling and reconstructing them back to 5 nm can reduce apparent spectral complexity.

> Interpolated spectra must not be treated as containing newly recovered spectral information between native measurements.

Interpolation creates a continuous representation on the target grid; it does not recover unmeasured physical information. This limitation must be preserved in the methodology.

### Limitations of empirical interpolation validation

1. **Proxy ground truth:** The validation uses native-5-nm spectra as proxy ground truth. The actual 64 native-10-nm spectra have unknown intermediate 5-nm values.
2. **Masking pattern:** The experiment masks every second 5-nm point, assuming alignment with the 10-nm grid. Different offsets may produce slightly different behavior.
3. **No noise modeling:** The reconstruction experiment uses actual measured values (including noise) as both input and reference.
4. **Physical resolution unknown:** Instrument spectral resolution is not recorded in RELAB metadata.
5. **Only two methods tested:** Only linear and PCHIP were compared per the investigation specification.
6. **Not universally superior:** PCHIP must not be described as universally superior. The evidence supports it for this dataset and validation design.

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/phase2_step3_interpolation_investigation.md`
- `results/02_preprocessing_qc/phase2_step3_interpolation_comparison.csv`
- `results/02_preprocessing_qc/phase2_step3_interpolation_summary.csv`
- `results/02_preprocessing_qc/phase2_step3_interpolation_freeze.md`
- `scripts/phase2_step3_interpolation_investigation.py`
- Diagnostic figures: `fig_interpolation_error_distributions.png`, `fig_interpolation_error_vs_wavelength.png`, `fig_interpolation_complexity_comparison.png`, `fig_interpolation_example_reconstructions.png`, `fig_interpolation_regional_comparison.png`

### Operational constraints at freeze

- No raw files were modified.
- All 1,017 raw checksums verified unchanged (0 mismatches).
- No final interpolation has been applied.
- No processed dataset has been generated.
- No final complexity features have been generated.
- No smoothing was performed.
- No normalization was performed.
- No feature engineering was performed.
- No ML/PySR was performed.

---

## Phase 2 Step 3 — Reflectance QC — FROZEN

**Date frozen:** 2026-09-29

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 925 candidates (800 Category A + 125 Category B before targeted review)

### Final decision

**Rule:** No spectrum is excluded from the analysis population on the basis of reflectance QC.

> **All 925 candidate spectra remain eligible for analysis subject to the previously frozen D2 coverage criterion and later preprocessing decisions. Reflectance QC identifies unusual values and patterns; it does not automatically classify them as measurement errors.**

### Final QC category counts

| Category | Description | Count |
|----------|-------------|------:|
| A | No reflectance validity problem detected | 892 |
| B | Unusual but potentially valid measurement | 32 |
| C | Ambiguous; requires caution/investigation | 1 |
| D | Clearly invalid numerical value | 0 |
| **Total** | | **925** |

### Frozen treatments

- **F0692 (Olivine):** Retained as Category B. 3 negative values (−0.014 to −0.038) near a ~1 μm absorption minimum; 3 values >1 (max ~1.232) as isolated single-point excursions. Anomalies are absorbed by the later PCHIP representation onto the frozen 5-nm grid. No clipping, no raw-data modification.
- **F0999 (Spinel glass):** Retained as Category C. 67 values >1 (max ~1.00738) forming a smooth systematic hump at 435–875 nm. Native 5-nm sampling matches the target grid, so the anomaly persists. No defensible raw-data correction is available. Flagged as an unresolved calibration/measurement issue for later sensitivity analysis. No clipping, no exclusion.
- **Group 1 (13 spectra):** Retained as Category B. Localized spikes in the ~800–1000 nm region, consistent with detector-transition behavior. Absorbed by PCHIP on the 5-nm grid.
- **Group 2 (17 Quartz spectra):** Retained as Category B. Constant-run/rounding artifacts at ~0.1-nm sampling disappear under the 5-nm representation.
- **F0019 (Albite):** Retained as Category B. Special merged-dataset case with constant runs and spikes; no evidence sufficient to exclude.
- **Category D:** 0 spectra. No spectrum is excluded for reflectance-QC reasons.

### Methodological distinction

The following four stages are distinguished:

1. Raw observations (measured wavelength–reflectance pairs in RELAB files).
2. QC flags (labels indicating unusual values or patterns).
3. Later D2 representation (PCHIP interpolation onto the frozen 5-nm grid).
4. Scientific exclusion (removing a spectrum from the analysis population — zero spectra).

The later PCHIP operation is a representation step. It must not be described as repairing or validating the underlying raw measurements.

### Evidence summary

- 925 analysis-candidate spectra investigated.
- 1 spectrum with negative values (F0692, 3 values, max |magnitude| 0.038).
- 2 spectra with values >1 (F0692: 3 values; F0999: 67 values).
- 273 spectra with single-point spikes (6631 total events, concentrated in 800–1000 nm detector-transition region).
- 1 spectrum with constant runs at native resolution (F0019); 17 Quartz spectra with short constant runs from rounding at 0.1-nm spacing.
- 0 spectra with NaN or Inf values.
- 0 spectra with zero values.

### Limitations

1. **F0999 physical cause unresolved.** The 67 values >1 persist in the final dataset. Resolution requires additional RELAB documentation about 1988 calibration procedures.
2. **Detector-transition mechanism not definitively established.** Group 1 spikes are consistent with detector-transition behavior but the specific mechanism is not confirmed by RELAB metadata.
3. **PCHIP is representation, not correction.** Absorption of anomalies by interpolation does not validate raw measurements.
4. **Spike detection thresholds are heuristic.** Different thresholds would yield different counts.
5. **No uncertainty semantics documented.** Most anomalous spectra lack uncertainty columns.

### Audit trail

- `results/02_preprocessing_qc/phase2_step3_reflectance_qc_investigation.md`
- `results/02_preprocessing_qc/phase2_step3_reflectance_qc.csv`
- `results/02_preprocessing_qc/phase2_step3_reflectance_qc_spectrum_summary.csv`
- `results/02_preprocessing_qc/phase2_step3_targeted_reflectance_review.md`
- `results/02_preprocessing_qc/phase2_step3_targeted_reflectance_review.csv`
- `results/02_preprocessing_qc/phase2_step3_reflectance_qc_freeze.md`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums verified unchanged (0 mismatches).
- No processed spectral dataset was created during this step.
- No interpolation was performed.
- No smoothing, normalization, or clipping was performed.
- No spectra were removed.
- No spectral features were engineered.
- No ML/PySR analysis was performed.

---

## Phase 2 Step 4 — Measurement-Uncertainty Treatment — FROZEN

**Date frozen:** 2026-09-30

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 1,017 raw RELAB files; 461 with third-column quantity; 454 D2-compatible spectra (377 with third-column quantity)

### Final decision

**Rule:** The RELAB third-column quantity will be retained as metadata and as a qualitative QC reference, but will not be used quantitatively in the primary spectral analysis.

> **The reported RELAB third-column measurement-uncertainty quantity is retained as metadata and QC reference only. Its exact statistical meaning remains unresolved. Quantitative use (inverse-variance weighting, propagation, confidence intervals) is prohibited in the primary analysis.**

This is a metadata-retention decision, NOT a claim that the spectra have equal physical measurement uncertainty. It means the available RELAB uncertainty-related quantity is insufficiently defined for defensible quantitative use.

### Terminology

The third column is referred to as:

> "RELAB third-column uncertainty-related measurement quantity"

or

> "reported RELAB third-column measurement-uncertainty quantity"

The following terms must NOT be used as definitive descriptions: standard deviation, standard error, one-sigma uncertainty, confidence interval, instrumental noise, or measurement error. The exact statistical meaning remains unresolved.

### Evidence summary

- **461 of 1,017** raw files (45.3%) contain a third column
- **377 of 454** D2-compatible spectra (83.0%) contain the third-column quantity
- **77 of 454** D2-compatible spectra (17.0%) do not contain the third-column quantity
- Third-column values are small positive numbers (typically 0.001–0.05), consistent with measurement uncertainty
- **6 spectra** contain extreme third-column values (up to ~29,000), orders of magnitude larger than typical values
- The third column is strongly correlated with specific instrument configurations (predominantly prefix `c1`, voltage 0.777)
- No official RELAB documentation establishes the exact statistical meaning

### Frozen treatments

- **Metadata retention:** All 461 reported third-column values are preserved in metadata
- **QC reference:** Third-column values may be inspected for QC purposes and unusual values flagged
- **No quantitative use:** Inverse-variance weighting, propagation, confidence intervals, and distributional assumptions are prohibited
- **Six extreme values:** Retained unchanged because their statistical meaning is not established to justify correction, clipping, or exclusion
- **Cross-campaign comparability:** The third-column quantity is not assumed to be directly comparable across all RELAB acquisition configurations
- **Native-10-nm spectra:** When the 64 native-10-nm spectra are represented on the 5-nm D2 grid, their third-column values will not be interpolated or propagated

### Allowed operations

- Preserve all reported third-column values in metadata
- Inspect third-column values for QC purposes
- Flag unusual values for documentation
- Retain for possible future methodological work if authoritative RELAB documentation becomes available
- Use as relative quality indicators in sensitivity analysis (not as absolute uncertainties)

### Prohibited operations

- Inverse-variance weighting
- Weighting spectra by the third column
- Propagating the third column mathematically
- Converting to sigma or standard error
- Calculating confidence intervals
- Assuming a probability distribution
- Normalizing into a percentage uncertainty
- Using as a measurement-error model
- Excluding spectra solely because the reported value is large
- Clipping extreme values

### Six extreme-value cases

| File ID | Mineral | Prefix | Max Value |
|---------|---------|--------|-----------|
| F0241 | Calcite | ca | 12.92 |
| F0242 | Calcite | ca | 8.42 |
| F0775 | Olivine | c1 | 29000.00 |
| F0779 | Orthoclase | c1 | 710.00 |
| F0951 | Spinel | c1 | 1100.00 |
| F0982 | Spinel | c2 | 1100.00 |

These values are retained unchanged. They are treated as a QC flag/documentation issue, not automatically as corrupted data.

### Scientific interpretation

> The absence of quantitative uncertainty weighting does NOT mean the spectra are assumed to have equal physical measurement uncertainty. It means the available RELAB uncertainty-related quantity is insufficiently defined for defensible quantitative use.

The decision to not use the third column quantitatively is a methodological caution, not a physical claim. It reflects the current state of knowledge about the RELAB dataset.

### Limitations

1. **No official RELAB documentation:** The interpretation rests on the RELAB field dictionary derived from direct file inspection.
2. **Statistical meaning undetermined:** Unknown whether values represent 1-sigma, 2-sigma, standard error, instrumental noise, or another quantity.
3. **Incomplete coverage:** Only 377 of 454 D2-compatible spectra (83.0%) contain the third-column quantity.
4. **Instrument-dependent:** Values vary systematically across instrument prefixes and voltages.
5. **No ground truth:** Without repeated measurements or reference materials, accuracy cannot be independently verified.
6. **Six extreme values unexplained:** Retained but not understood.
7. **Native-10-nm uncertainty not interpolated:** 53 native-10-nm spectra with third-column data will not have uncertainty values interpolated to the 5-nm grid.
8. **Conservative approach:** May discard useful information if the third column is later found to be well-characterized.

### Reopening conditions

This freeze may be reopened if:

1. Authoritative RELAB documentation becomes available establishing the statistical meaning
2. Repeated-measurement evidence demonstrates correspondence to a specific statistical quantity
3. Instrument-calibration evidence establishes the relationship for specific configurations
4. Methodological development produces a defensible framework for quantitative use despite unresolved semantics

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/phase2_step4_uncertainty_investigation.md`
- `results/02_preprocessing_qc/phase2_step4_uncertainty_spectrum_summary.csv`
- `results/02_preprocessing_qc/phase2_step4_uncertainty_point_summary.csv`
- `results/02_preprocessing_qc/phase2_step4_uncertainty_freeze.md`
- `results/02_preprocessing_qc/phase2_step4_summary.md`
- Diagnostic figures: `results/02_preprocessing_qc/figures/uncertainty_fig*.png`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums verified unchanged (0 mismatches).
- No processed spectral dataset was created during this step.
- No interpolation was performed.
- No smoothing, normalization, or clipping was performed.
- No spectra were removed.
- No uncertainty values were altered.
- No spectral features were engineered.
- No ML/PySR analysis was performed.

---

## Phase 2 Step 5 — Reflectance Normalization — FROZEN

**Date frozen:** 2026-10-01

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 454 D2-compatible spectra (390 native-5 nm + 64 native-10 nm)

### Final decision

**Rule:** The primary spectral analysis retains reflectance on its original measurement scale. No normalization is applied to the canonical primary spectra.

> **The primary representation is original reflectance (Method A). No normalization is applied.**

This decision is based on the principle that absolute reflectance amplitude may contain scientifically meaningful information related to absorption strength, scattering, and mineral optical behavior.

Absolute reflectance may contain mineralogically meaningful information, although its variation can also reflect acquisition geometry, grain size, instrument configuration, and other measurement factors.

Therefore, retaining the original scale preserves potentially meaningful information rather than assuming that amplitude variation is purely mineralogical.

### Method-specific decisions

- **Method A (No normalization):** Primary representation.
- **Method B (Vector normalization):** Potential sensitivity analysis only.
- **Method C (Maximum normalization):** Potential sensitivity analysis only.
- **Method D (Area normalization):** Potential sensitivity analysis only.
- **Method E (Continuum normalization):** Not part of the primary preprocessing pipeline.
- **Method F (Z-score normalization):** Not part of the primary preprocessing pipeline.

### Evidence summary

- **454 D2-compatible spectra** investigated (390 native-5 nm + 64 native-10 nm)
- **6 normalization methods** evaluated (A–F)
- **Scalar methods (B, C, D):** preserve peak positions and shape descriptors; alter amplitude-based descriptors proportionally
- **Continuum normalization (E):** wavelength-dependent; can introduce artificial local structure/extrema; alters both amplitude and shape descriptors
- **Z-score normalization (F):** substantially transforms amplitude relationships; largest transformation of all methods
- **Coefficient of variation of mean reflectance:** 0.38 (substantial amplitude variation across dataset)
- **Within-mineral vs between-mineral amplitude variation:** ratio of total to between-mineral variation = 1.68, suggesting acquisition effects contribute significantly

### Complexity implications

Normalization can materially alter spectral-complexity descriptors:

- Vector normalization: median relative change ~0.92 for first-order variation
- Maximum normalization: median relative change ~0.44
- Area normalization: median relative change ~0.996
- Continuum normalization: peak density median change +7.6 (artificial extrema)
- Z-score normalization: median relative change ~10.1

Therefore, normalization itself can become a source of variation in measured spectral complexity. This is a key methodological reason not to normalize the primary dataset.

### Scientific interpretation

> The decision is NOT that normalization is scientifically wrong. The decision is that normalization is not justified as the primary representation for this research question.

The project preserves original reflectance because doing so avoids discarding potentially meaningful amplitude information. Scalar normalization can later be used as a robustness/sensitivity analysis.

### Special cases

- **F0692 (Olivine):** Retained unchanged. Negative and >1 reflectance values not clipped.
- **F0999 (Spinel glass):** Retained unchanged. Category C flag remains frozen.

### Limitations

1. **No ground truth for normalization.** The choice is inherently scientific, not technical.
2. **Limited acquisition metadata.** Grain size and detailed illumination geometry are not systematically available.
3. **Continuum normalization is not universally applicable.** Appropriate for some minerals but not others.
4. **Complexity descriptors are not yet final.** Different descriptors may respond differently to normalization.
5. **Single-interpolation pipeline.** 64 native-10-nm spectra are interpolated via PCHIP before normalization.

### Reopening conditions

This freeze may be reopened if:

1. Authoritative evidence demonstrates that amplitude variation is exclusively acquisition-related
2. Methodological development produces a defensible framework for separating mineralogical from acquisition-related amplitude variation
3. Sensitivity analysis demonstrates that scientific conclusions are materially altered by normalization
4. External reference materials establish the correct normalization for mineral spectral complexity
5. Researcher review determines that the scientific question requires amplitude removal

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/phase2_step5_normalization_investigation.md`
- `results/02_preprocessing_qc/phase2_step5_normalization_comparison.csv`
- `results/02_preprocessing_qc/phase2_step5_normalization_summary.csv`
- `results/02_preprocessing_qc/phase2_step5_normalization_freeze.md`
- Diagnostic figures: `results/02_preprocessing_qc/figures/normalization_fig*.png`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums verified unchanged (0 mismatches).
- 454 D2-compatible spectra remain represented.
- 390 native-5-nm spectra remain unchanged.
- 64 native-10-nm spectra remain unchanged.
- No spectra excluded because of normalization.
- No normalized canonical dataset created.
- No interpolation was performed during this step.
- No smoothing, normalization, or clipping was performed.
- No feature engineering was performed.
- No ML/PySR analysis was performed.

---

## Phase 2 Step 6 — Smoothing / Denoising — FROZEN

**Date frozen:** 2026-10-03

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 454 D2-compatible spectra (390 native-5 nm + 64 native-10 nm)

### Final decision

**Rule:** No smoothing or denoising will be applied to the canonical primary spectral dataset.

> **Primary decision: no smoothing or denoising will be applied to the canonical spectral dataset. The original measured spectral structure will be retained because the available evidence does not establish that high-frequency variation is predominantly measurement noise, while smoothing demonstrably alters spectral-complexity descriptors. Controlled smoothing may be used later only as a separately documented sensitivity analysis.**

The primary spectra retain the measured spectral structure after the already-frozen wavelength-domain/grid treatment. The reason is scientific rather than cosmetic: spectral complexity is itself an object of investigation, and smoothing systematically changes extrema, peak density, curvature, derivatives, and related complexity descriptors.

### Evidence summary

- **454 D2-compatible spectra** investigated (390 native-5 nm + 64 native-10 nm)
- **15 smoothing configurations** evaluated (1 baseline + 14 candidate methods)
- **Savitzky-Golay configurations:** w=3/p=2, w=5/p=2, w=5/p=3, w=7/p=2, w=7/p=3, w=9/p=2, w=9/p=3
- **Moving average:** w=3, w=5
- **Gaussian:** sigma=5.0 nm, sigma=10.0 nm
- **LOWESS/LOESS:** frac=0.01, frac=0.02, frac=0.05
- **SG w=3/p=2** produced zero change to all tested complexity descriptors
- **All other configurations** systematically reduced peak density (~49%-91%), curvature (~53%-87%), extrema count, and second-order variation
- **Native-10-nm spectra** exhibited smaller residuals because PCHIP interpolation already suppresses fine-scale variation
- Reproducibility passed
- All **1,017** raw MD5 checksums remained unchanged
- No canonical smoothed dataset was created

### Scientific rationale

The observed high-frequency variation is mineral-dependent and cannot be demonstrated to be predominantly measurement noise from the available RELAB evidence. The dataset does not provide an independent ground-truth noise model or repeated-measurement reference sufficient to separate all fine-scale spectral structure into signal versus noise.

The absence of a defensible noise model means that smoothing cannot be justified as a primary preprocessing operation.

### Complexity implications

Smoothing changes local extrema, peak density, peak prominence, derivative structure, curvature, fine-scale variation, and entropy/roughness-like descriptors. These overlap directly with the project's concept of spectral complexity. The preprocessing operation could partially determine the quantity that the research is attempting to explain. This is the central reason for retaining the unsmoothed representation.

### SG w=3/p=2 status

SG w=3/p=2 is retained only as a documented near-neutral diagnostic/sensitivity transformation. It is not part of the canonical preprocessing pipeline because the investigation provides no scientific necessity for adding a transformation that produces negligible measurable benefit. No canonical SG-smoothed dataset is created.

### Methods excluded from primary pipeline

The following are NOT part of the primary preprocessing pipeline: Savitzky-Golay configurations producing material changes; moving average; Gaussian smoothing; LOWESS/LOESS. These methods are not ranked. No method is described as "bad." The decision is simply that none is sufficiently justified for the primary analysis.

### Limitations

1. **No independent RELAB ground-truth noise model.**
2. **No comprehensive repeated-measurement dataset establishing measurement noise.**
3. **Mineral-dependent fine-scale variation cannot be completely separated into signal and noise.**
4. **The native-10-nm spectra have already undergone PCHIP interpolation for diagnostic comparison**, affecting their apparent fine-scale variation.
5. **Complexity descriptors themselves can be sampling- and preprocessing-sensitive.**
6. **Absence of smoothing does not prove that every fine-scale feature is physically meaningful.** "No smoothing" means that the project avoids imposing an unvalidated smoothing model; it does not mean that every measured fluctuation is assumed to represent true mineral structure.

### Reopening conditions

This freeze may be reopened if:

1. Authoritative RELAB documentation becomes available establishing a ground-truth noise model.
2. Repeated-measurement evidence demonstrates that specific frequency bands are predominantly measurement noise.
3. Methodological development produces a defensible framework for separating signal from noise in the absence of ground truth.
4. Sensitivity analysis demonstrates that scientific conclusions are materially unaffected by controlled smoothing.
5. Researcher review determines that the scientific question requires smoothing as a primary preprocessing operation.

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/02_preprocessing_qc/phase2_step6_smoothing_investigation.md`
- `results/02_preprocessing_qc/phase2_step6_smoothing_comparison.csv`
- `results/02_preprocessing_qc/phase2_step6_smoothing_summary.csv`
- `results/02_preprocessing_qc/phase2_step6_smoothing_freeze.md`
- `results/02_preprocessing_qc/phase2_step6_summary.md`
- Diagnostic figures: `results/02_preprocessing_qc/figures/step6/fig1_raw_vs_smoothed.png` through `fig12_F0999_special_case.png`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums verified unchanged (0 mismatches).
- 454 D2-compatible spectra remain represented.
- 390 native-5-nm spectra remain unchanged.
- 64 native-10-nm spectra remain unchanged.
- 15 smoothing configurations remain documented.
- No smoothed canonical dataset was created.
- No final processed dataset was created.
- No final complexity features were created.
- No ML or PySR was run.
- No interpolation was performed during this step.
- No smoothing, normalization, or clipping was performed.
- No spectra were excluded.

---

## Phase 2 — Final Preprocessing Closure

**Date closed:** 2026-10-04

**Status:** COMPLETE / FROZEN

### Clarification — Reflectance-QC retention vs. canonical D2 inclusion

> **Reflectance-QC retention and canonical D2 inclusion are separate decisions. F0692 and F0999 were retained during reflectance QC, but are excluded from the final canonical D2 dataset because they do not fully cover 320–2550 nm. This exclusion is due to spectral coverage, not reflectance quality.**

| File ID | QC Category | wl_min (nm) | D2 coverage | Canonical D2 inclusion | Exclusion reason |
|---------|-------------|-------------|-------------|------------------------|------------------|
| F0692 | B | 1000.0 | Does not cover 320–1000 nm | Excluded | Insufficient D2 coverage |
| F0999 | C | 340.0 | Does not cover 320–340 nm | Excluded | Insufficient D2 coverage |

- QC category describes the quality characteristics observed during the reflectance investigation; it is not itself the criterion for canonical D2 inclusion.
- The D2 domain remains 320–2550 nm at 5 nm spacing (447 points).
- The canonical population remains 454 spectra (390 native-5-nm + 64 PCHIP native-10-nm).
- No spectrum described as "bad." F0692 and F0999 are simply outside the complete D2 coverage requirement.

### Final Phase 2 preprocessing pipeline

1. RELAB raw spectrum
2. Deterministic parsing
3. Duplicate wavelength keep-first
4. Wavelength ordering in memory
5. D2 coverage selection: 320–2550 nm
6. Direct use of native 5-nm values where applicable
7. PCHIP representation for native 10-nm spectra
8. No splice correction
9. No reflectance clipping/correction
10. No uncertainty propagation
11. No normalization
12. No smoothing
13. Canonical 5-nm D2 representation

No additional preprocessing operation may be added without reopening the relevant scientific decision.

---

## Phase 3 Step 2 — Spectral Complexity Descriptor Specification — FROZEN

**Date frozen:** 2026-10-05

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 454 D2-compatible spectra (390 native-5 nm + 64 PCHIP native-10 nm)

### Final decision

**Rule:** The spectral complexity descriptor set consists of exactly four primary descriptors and one sensitivity descriptor.

> **The primary descriptor set comprises four descriptors: prominent peak count (n_peaks_prom), mean prominent peak prominence (mean_peak_prominence), first-derivative RMS (first_deriv_rms), and second-derivative RMS (second_deriv_rms). One additional descriptor (total_variation) is retained as a sensitivity-only descriptor. Both entropy descriptors are rejected.**

This is a feature-definition freeze, NOT a claim that these descriptors capture "true physical complexity." The descriptors are methodological constructs that quantify specific aspects of spectral structure.

### Frozen primary descriptor set

| Descriptor | Code | Definition | Units | Median | Range |
|------------|------|------------|-------|--------|-------|
| D1: Prominent peak count | `n_peaks_prom` | Number of peaks with prominence ≥ 0.01 | count | 3.0 | 0-11 |
| D2: Mean prominent peak prominence | `mean_peak_prominence` | Mean prominence of detected peaks | reflectance | 0.097 | 0.0-0.361 |
| D3: First-derivative RMS | `first_deriv_rms` | RMS of first spectral derivative | reflectance/nm | 0.00082 | 0.000019-0.00292 |
| D4: Second-derivative RMS | `second_deriv_rms` | RMS of second spectral derivative | reflectance/nm² | 0.0000238 | 0.0000009-0.000353 |

### Frozen sensitivity descriptor

| Descriptor | Code | Definition | Units | Status |
|------------|------|------------|-------|--------|
| S1: Total variation | `total_variation` | Sum of absolute adjacent differences | reflectance | Sensitivity-only (r=0.914 with D3) |

### Frozen computational conventions

| Parameter | Value | Source |
|-----------|-------|--------|
| Prominence threshold | 0.01 reflectance units | Step 1 + Step 1A |
| Peak detection algorithm | `scipy.signal.find_peaks` | Step 1 |
| Minimum peak distance | 3 points (15 nm) | Step 1 |
| Derivative method | `numpy.gradient` (central differences) | Step 1 |
| Zero-peak convention | mean_peak_prominence = 0.0 | Step 1 |
| Smoothing | None | Phase 2 Step 6 |
| Normalization | None (original reflectance) | Phase 2 Step 5 |

### Rejected descriptors

**Entropy descriptors (rejected):**
- `spectral_entropy` — measures distributional property, not structural complexity; near-maximum values (100% within 5% of max)
- `first_deriv_entropy` — measures distributional property, not structural complexity; limited dynamic range

**Redundant descriptors (rejected):**
- 22 additional candidates rejected due to high redundancy (r > 0.90), mathematical identity, noise sensitivity, or linear transforms

### Evidence summary

- **30 candidate descriptors** investigated across 6 families in Phase 3 Step 1
- **6 prominence thresholds** evaluated in Phase 3 Step 1A (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
- **Threshold 0.01** selected as region of relative stability; all pairwise Spearman correlations > 0.95
- **Entropy assessment:** spectral_entropy median 8.78 bits (max 8.80 bits); 100% within 5% of max; effective range 0.294 bits
- **Redundancy analysis:** first_deriv_rms and total_variation correlated at r = 0.914 but have different interpretations (intensive vs cumulative)
- **Representation sensitivity:** second_deriv_rms shows >10% difference between native-5nm and PCHIP spectra; other primary descriptors <5%
- **Core descriptor robustness:** all 4 primary descriptors have strong interpretability, numerical stability, and complexity relationship
- **Historical reconciliation:** old "9 descriptors" from earlier project are superseded; different data source, preprocessing, and aggregation philosophy

### Scientific interpretation

> The four primary descriptors capture distinct aspects of spectral structure: feature density (D1), feature strength (D2), first-order spectral variation (D3), and second-order spectral variation (D4). They are methodological constructs, not claims about "true physical complexity."

The following statements must NOT be made:
- That `first_deriv_rms` measures "noise" (it measures average spectral variation rate)
- That `second_deriv_rms` measures "physical curvature" (it measures second-derivative magnitude)
- That the descriptors capture "true physical complexity" (they are methodological constructs)
- That the prominence threshold of 0.01 is "optimal" (it is a heuristic choice)

### Limitations

1. **No smoothing applied.** All descriptors operate on unsmoothed spectra. High-frequency measurement noise contributes to derivative-based descriptors. This is intentional (Phase 2 Step 6) but means descriptors conflate mineralogical structure with measurement noise.

2. **5 nm grid spacing.** All derivative descriptors are computed on the 5 nm grid. Peak density (CV = 24.1%) and second-order variation (CV = 8.5%) show spacing-dependent behavior.

3. **PCHIP representation for 64 spectra.** `second_deriv_rms` shows >10% median difference between native-5-nm and PCHIP spectra. Later analyses must account for this by including `representation_type` as a covariate.

4. **Prominence threshold is heuristic.** The choice of 0.01 is scientifically motivated but not uniquely determined. Different thresholds produce substantially different feature counts.

5. **No mineral-specific validation.** Descriptor behavior was not optimized or validated against mineral identity. This is intentional (no leakage) but means the frozen set has not been tested for mineralogical discrimination.

6. **Geometric curvature vs second derivative.** `second_deriv_rms` is the RMS of the second derivative, not geometric curvature. For spectral data where |dR/dwl| << 1, the two are approximately equivalent.

### Historical reconciliation

The old "9 descriptors" from the earlier project are marked as historical/superseded because:
- Different data source (mixed library vs RELAB-only)
- Different preprocessing (unknown vs frozen pipeline)
- Different treatment of entropy descriptors (included vs rejected)
- Different aggregation philosophy (single score vs individual descriptors)

The current 5-descriptor set (4 primary + 1 sensitivity) is based on frozen RELAB-only pipeline, rigorous robustness assessment, and clear scientific interpretation.

### Reopening conditions

This freeze may be reopened if:
1. Authoritative RELAB documentation becomes available establishing a ground-truth noise model
2. Repeated-measurement evidence demonstrates that specific frequency bands are predominantly measurement noise
3. Methodological development produces a defensible framework for separating signal from noise
4. Sensitivity analysis demonstrates that scientific conclusions are materially affected by the descriptor choice
5. Researcher review determines that the scientific question requires additional descriptors
6. External evidence demonstrates that the prominence threshold should be optimized for specific mineral groups

### Audit trail

This frozen decision is supported by the following investigation outputs (not modified during this freeze):

- `results/03_complexity_features/phase3_step1_descriptor_investigation.md`
- `results/03_complexity_features/phase3_step1_descriptor_values.csv`
- `results/03_complexity_features/phase3_step1_descriptor_summary.csv`
- `results/03_complexity_features/phase3_step1_descriptor_redundancy.csv`
- `results/03_complexity_features/phase3_step1a_descriptor_robustness.md`
- `results/03_complexity_features/phase3_step1a_threshold_sensitivity.csv`
- `results/03_complexity_features/phase3_step1a_representation_sensitivity.csv`
- `results/03_complexity_features/phase3_step1a_core_descriptor_assessment.csv`
- `results/03_complexity_features/phase3_step1a_entropy_assessment.csv`
- `results/03_complexity_features/phase3_step1a_historical_descriptor_reconciliation.md`
- `results/03_complexity_features/phase3_step2_descriptor_specification_freeze.md`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums remain unchanged.
- 454 D2-compatible spectra remain represented.
- No final canonical feature matrix has been created.
- No final complexity features have been calculated.
- No ML or PySR was run.
- No PCA or clustering was run.
- No mineral differences were tested.
- No chemistry or crystal symmetry was used.
- No scalar complexity score was constructed.
- No new descriptors were introduced.
- The Phase 2 canonical dataset remains unchanged.

---

## Phase 3 Step 3 — Canonical Feature Matrix Construction — FROZEN

**Date frozen:** 2026-10-05

**Decision classification:** B — Ready with explicit limitations

**Analysis population:** 454 D2-compatible spectra (390 native-5 nm + 64 PCHIP native-10 nm)

### Final decision

> **The canonical spectral-complexity feature matrix has been constructed for all 454 D2-compatible spectra using the four frozen primary descriptors (D1–D4). The computation is deterministic and reproducible. The Phase 2 canonical dataset remains unchanged.**

### Matrix structure

| Parameter | Value |
|---|---|
| Rows | 454 |
| Total columns | 9 |
| Primary feature columns | 4 (n_peaks_prom, mean_peak_prominence, first_deriv_rms, second_deriv_rms) |
| Metadata columns | 5 (canonical_spectrum_id, raw_filename, mineral, specimen_group_id, representation_type) |
| Zero-peak spectra | 14 (all have mean_peak_prominence = 0.0) |

### Descriptor statistics

| Descriptor | Min | Median | Max | Mean | Std |
|---|---|---|---|---|---|
| D1: n_peaks_prom | 0 | 3 | 11 | 2.978 | 1.729 |
| D2: mean_peak_prominence | 0.000000 | 0.090502 | 0.360834 | 0.108950 | 0.072064 |
| D3: first_deriv_rms | 1.86e-05 | 0.000824 | 0.002919 | 0.000797 | 0.000422 |
| D4: second_deriv_rms | 8.62e-07 | 2.38e-05 | 3.53e-04 | 3.80e-05 | 3.99e-05 |

### Validation results

| Category | Checks | Passed | Failed |
|---|---|---|---|
| Dataset integrity | 11 | 11 | 0 |
| Feature integrity | 11 | 11 | 0 |
| Reproducibility | 2 | 2 | 0 |
| **TOTAL** | **24** | **24** | **0** |

### Checksums

| File | SHA-256 |
|---|---|
| Input (Phase 2 canonical spectra) | ec6ec28671960a2e0b5fc362e98bb4f5b895b65901463b1992deb04798faf803 |
| Output (Phase 3 canonical features) | a761544a2c959db83ce5ee5acc7c537ad876283c733ee08ace3f081e2af6efda |

### Frozen computational conventions

- Prominence threshold: 0.01 reflectance units
- Peak detection: `scipy.signal.find_peaks` with `distance=3`, `prominence=0.01`
- Derivative method: `numpy.gradient` (central differences, second-order accurate at interior points)
- Wavelength grid: 447 points, 320–2550 nm, 5 nm spacing
- Zero-peak convention: `mean_peak_prominence = 0.0` when `n_peaks_prom = 0`
- Smoothing: None
- Normalization: None
- Sensitivity descriptor (S1, total_variation): NOT included in primary matrix

### Reproducibility

The computation was run twice from the same frozen Phase 2 input. Both runs produced identical row order, identical feature values, and identical output checksums.

### Limitations

1. **Descriptor correlation:** D3 and D4 are correlated (r = 0.81). They measure different derivative orders but share spectral-variation information. Downstream analyses must account for this.
2. **Representation sensitivity:** 64 PCHIP spectra may have different descriptor values than if they had been natively acquired at 5 nm. `representation_type` is preserved as metadata for covariate-aware analysis.
3. **No smoothing:** Descriptors are computed on unsmoothed spectra. High-frequency noise may affect derivative-based descriptors (D3, D4).
4. **Prominence threshold:** The 0.01 threshold is a heuristic choice, not an optimized parameter.
5. **No scalar complexity score:** The four descriptors are retained as separate columns. No aggregation into a single "complexity" value has been performed.

### Audit trail

- `scripts/phase3_step3_canonical_feature_matrix.py`
- `data/processed/phase3_canonical_spectral_features.csv`
- `results/03_complexity_features/phase3_step3_feature_matrix_validation.md`
- `results/03_complexity_features/phase3_step3_feature_matrix_validation.csv`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums remain unchanged.
- The Phase 2 canonical dataset remains unchanged (checksum verified).
- No ML or PySR was run.
- No PCA or clustering was run.
- No mineral differences were tested.
- No chemistry or crystal symmetry was used.
- No scalar complexity score was constructed.
- No new descriptors were introduced.
- No z-scoring or feature selection was applied.
- No uncertainty weighting was applied.

---

## Phase 3 Step 4 — Descriptor Distribution, Dependence, and Redundancy Audit — FROZEN

**Date frozen:** 2026-10-06

**Decision classification:** A — Diagnostic-only audit (no scientific decisions)

**Analysis population:** 454 D2-compatible spectra (390 native-5 nm + 64 PCHIP native-10 nm)

### Final decision

> **The four frozen primary descriptors have been audited for distributional properties, pairwise dependence, redundancy, scale variation, zero/discrete values, representation-type effects, provenance-group structure, and duplicate feature vectors. All audit components completed successfully. No descriptors exceed the redundancy thresholds (|r| ≥ 0.90). The descriptor set is ready for downstream scientific analysis.**

### Key findings

| Finding | Value |
|---|---|
| Strongest Pearson correlation | first_deriv_rms ↔ second_deriv_rms (r = 0.8079) |
| Strongest Spearman correlation | first_deriv_rms ↔ second_deriv_rms (ρ = 0.8190) |
| Strong pairs (|r| ≥ 0.90, Pearson) | 0 |
| Very strong pairs (|r| ≥ 0.95, Pearson) | 0 |
| Scale variation (max mean ratio) | 78,279× |
| Scale variation (max std ratio) | 43,292× |
| Zero-peak spectra | 14 (3.08%) |
| Representation types | 2 (native-5nm: 390, PCHIP: 64) |
| Provenance groups | 382 (319 singletons, 59 pairs, 4 groups with 3+) |
| Duplicate feature vectors | 98 rows in 49 groups (all group size = 2) |

### Descriptor statistics

| Descriptor | Min | Median | Max | Mean | Std | CV | Unique values |
|---|---|---|---|---|---|---|---|
| n_peaks_prom | 0 | 3 | 11 | 2.978 | 1.729 | 0.581 | 12 |
| mean_peak_prominence | 0.000000 | 0.090502 | 0.360834 | 0.108950 | 0.072064 | 0.661 | 392 |
| first_deriv_rms | 1.86e-05 | 0.000824 | 0.002919 | 0.000797 | 0.000422 | 0.530 | 405 |
| second_deriv_rms | 8.62e-07 | 2.38e-05 | 3.53e-04 | 3.80e-05 | 3.99e-05 | 1.050 | 405 |

### Pairwise correlation matrix (Pearson)

|  | n_peaks_prom | mean_peak_prominence | first_deriv_rms | second_deriv_rms |
|---|---|---|---|---|
| n_peaks_prom | 1.000 | -0.055 | 0.571 | 0.703 |
| mean_peak_prominence | -0.055 | 1.000 | 0.510 | 0.169 |
| first_deriv_rms | 0.571 | 0.510 | 1.000 | 0.808 |
| second_deriv_rms | 0.703 | 0.169 | 0.808 | 1.000 |

### Representation-type comparison

| Descriptor | native_5nm median (IQR) | PCHIP median (IQR) |
|---|---|---|
| n_peaks_prom | 3 (2–4) | 3 (2–4) |
| mean_peak_prominence | 0.0970 (0.061–0.162) | 0.0625 (0.043–0.089) |
| first_deriv_rms | 0.000850 (0.000491–0.001022) | 0.000543 (0.000435–0.000997) |
| second_deriv_rms | 2.52e-05 (1.82e-05–4.97e-05) | 1.65e-05 (1.29e-05–2.61e-05) |

### Validation results

| Check | Status |
|---|---|
| Input checksum verified | PASS |
| Row count = 454 | PASS |
| Column count = 9 | PASS |
| All descriptors finite | PASS |
| No missing values | PASS |
| Reproducibility (2 runs) | PASS |

### Audit trail

- `scripts/phase3_step4_descriptor_audit.py`
- `results/03_complexity_features/phase3_step4_descriptor_audit.md`
- `results/03_complexity_features/phase3_step4_descriptor_audit.csv`
- `results/03_complexity_features/figures/step4/fig_n_peaks_prom_distribution.png`
- `results/03_complexity_features/figures/step4/fig_mean_peak_prominence_distribution.png`
- `results/03_complexity_features/figures/step4/fig_first_deriv_rms_distribution.png`
- `results/03_complexity_features/figures/step4/fig_second_deriv_rms_distribution.png`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- The Phase 3 canonical feature matrix remains unchanged (checksum verified).
- No ML or PySR was run.
- No PCA or clustering was run.
- No mineral differences were tested.
- No chemistry or crystal symmetry was used.
- No descriptors were added, removed, or transformed.
- No z-scoring or standardization was applied.
- No scientific conclusions were drawn.

---

## Phase 4 Step 3 — Mineral Chemistry Representation and Analysis Eligibility — FROZEN

**Date frozen:** 2026-10-07

**Decision classification:** COMPLETE / FROZEN WITH EXPLICIT CHEMISTRY LIMITATIONS

**Analysis population:** 382 provenance groups (454 canonical spectra)

### Final decision

> **A traceable chemistry metadata layer has been constructed for all 382 provenance groups. All chemistry is type C (mineral-level ideal/reference). Zero groups have specimen-specific measured chemistry. Each group has been classified into one of four chemistry eligibility categories. No scientific hypothesis has been tested.**

### Chemistry evidence hierarchy

| Evidence level | Groups |
|---|---|
| Specimen-specific measured chemistry | 0 |
| Mineral-level ideal/reference chemistry | 382 |
| No chemistry available | 0 |

### Eligibility classification

| Category | Groups | Spectra | Definition |
|---|---|---|---|
| CHEM_REFERENCE_ELIGIBLE | 246 | 258 | Unambiguous ideal/reference chemistry |
| CHEM_AMBIGUOUS | 106 | 165 | Pyroxene solid-solution and/or intentional mixture |
| CHEM_TENTATIVE | 6 | 6 | Mineral-name vs description conflict or tentative provenance |
| CHEM_CONFLICT | 24 | 25 | Unexplained Augite-Olivine byte-identical duplicates |

### Key findings

| Finding | Value |
|---|---|
| Unique reference formulas parsed | 17 (of 18 minerals; Apatite resolved to fluorapatite end-member) |
| Constituent elements | 14 (Al, Be, C, Ca, F, Fe, H, K, Mg, Na, O, P, Si, Ti) |
| Polymorph-ambiguous groups | 88 (chemistry unambiguous — polymorphs share formula) |
| Solid-solution-affected groups | 85 |
| Mixture-affected groups | 57 |
| Formula-ambiguous groups | 14 (Apatite variable anion site) |
| High-conflict-flag groups | 30 (24 augite-olivine + 6 name-vs-description) |

### Formula parsing

All 18 mineral reference formulas were parsed into constituent elements and stoichiometric atom fractions. Special cases handled:

- **Parenthesized groups:** Dolomite `CaMg(CO3)2` → Ca:Mg:C:O = 1:1:2:6
- **OH groups:** Muscovite `KAl2AlSi3O10(OH)2` → K:Al:Si:O:H = 1:3:3:12:2
- **Variable anion sites:** Apatite `Ca5(PO4)3(F Cl OH)` → resolved to fluorapatite end-member `Ca5(PO4)3F`, flagged `formula_ambiguous = TRUE`
- **Solid-solution formulas:** Augite `CaNaMgFeAlSi2O6` → treated as idealized end-member composition

### Validation results

| Check | Status |
|---|---|
| Row count = 382 | PASS |
| All 18 formulas parsed successfully | PASS |
| Atom fractions sum to 1.0 (±1e-10) for all groups | PASS |
| Eligibility categories mutually exclusive | PASS |
| Eligibility covers all 382 groups | PASS |
| Conflict flags consistent with Step 1 | PASS |
| Chemistry type = C for all groups | PASS |
| Specimen-specific count = 0 | PASS |
| Reproducibility (2 runs, SHA-256 match) | PASS |
| Frozen input: Phase 2 D2 checksum | PASS |
| Frozen input: Phase 3 features checksum | PASS |
| Frozen input: Phase 4 provenance mapping checksum | PASS |

### Audit trail

- `scripts/phase4_step3_chemistry_representation.py`
- `data/processed/phase4_chemistry_metadata.csv`
- `results/04_chemistry_crystallography/phase4_step3_chemistry_eligibility.csv`
- `results/04_chemistry_crystallography/phase4_step3_chemistry_summary.csv`
- `results/04_chemistry_crystallography/phase4_step3_chemistry_validation.md`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All chemistry is type C (mineral-level ideal/reference). Zero specimen-specific measured chemistry.
- No scientific hypothesis was tested.
- No chemistry-spectrum correlations were computed.
- No ML, PCA, or clustering was run.
- No mineral difference tests were performed.
- No scalar chemistry scores were constructed.
- The Phase 2 canonical dataset remains unchanged (checksum verified).
- The Phase 3 canonical feature matrix remains unchanged (checksum verified).
- The Phase 4 provenance mapping remains unchanged (checksum verified).

---

## Phase 4 Step 4 — Integrated Analysis-Ready Dataset Construction — FROZEN

**Date frozen:** 2026-10-08

**Decision classification:** COMPLETE / FROZEN — INTEGRATED ANALYSIS-READY DATASET

**Analysis population:** 454 canonical spectra (382 provenance groups)

### Final decision

> **An integrated analysis-ready dataset has been constructed combining all Phase 2, Phase 3, and Phase 4 (Steps 1-3) outputs. The dataset preserves all spectral complexity descriptors, provenance metadata, symmetry metadata, chemistry metadata, and eligibility flags. No scientific hypothesis has been tested.**

### Integration approach

The integrated dataset was constructed by:

1. Starting from Phase 3 canonical feature matrix (454 rows)
2. Adding provenance_group_id via spectrum-to-group mapping
3. Merging symmetry metadata from Phase 4 Step 2
4. Merging chemistry metadata from Phase 4 Step 3
5. Preserving all descriptor values and eligibility flags unchanged

The hierarchy of inference (RELAB file → spectrum → provenance group → mineral identity → reference chemistry/symmetry) was preserved. No levels were flattened.

### Dataset dimensions

| Metric | Value |
|--------|-------|
| Total spectra | 454 |
| Total provenance groups | 382 |
| Minerals represented | 18 |
| Columns | 52 |

### Eligibility framework

| Population | Spectra | Criteria |
|------------|--------:|----------|
| PRIMARY_ANALYSIS_ELIGIBLE | 175 | Provenance not tentative + symmetry high confidence + chemistry reference eligible |
| SENSITIVITY_SYM_ELIGIBLE | 258 | Provenance not tentative + chemistry reference eligible (symmetry ambiguity allowed) |
| SENSITIVITY_CHEM_ELIGIBLE | 359 | Provenance not tentative + symmetry high confidence (chemistry ambiguity allowed) |

### Symmetry eligibility distribution

| Status | Spectra |
|--------|--------:|
| SYM_HIGH_CONFIDENCE | 359 |
| SYM_AMBIGUOUS | 89 |
| SYM_TENTATIVE | 6 |
| SYM_CONFLICT | 0 |

### Chemistry eligibility distribution

| Status | Spectra |
|--------|--------:|
| CHEM_REFERENCE_ELIGIBLE | 258 |
| CHEM_AMBIGUOUS | 165 |
| CHEM_TENTATIVE | 6 |
| CHEM_CONFLICT | 25 |

### Preserved limitations

All ambiguities from previous phases were explicitly preserved:

- **Symmetry:** polymorph ambiguity, uncertain space-group assignments, tentative provenance
- **Chemistry:** solid-solution effects, intentional mixtures, formula ambiguity, Augite-Olivine conflicts, mineral-description conflicts
- **Data representation:** native-5nm vs PCHIP-derived spectra

### Validation results

| Check | Status |
|-------|--------|
| Row count = 454 | PASS |
| All spectra linked to provenance groups | PASS |
| All symmetry metadata present | PASS |
| All chemistry metadata present | PASS |
| Phase 3 descriptors preserved unchanged | PASS |
| Eligibility flags consistent | PASS |
| Reproducibility (2 runs, SHA-256 match) | PASS |
| Frozen input: Phase 2 D2 checksum | PASS |
| Frozen input: Phase 3 features checksum | PASS |
| Frozen input: Phase 4 provenance mapping checksum | PASS |

### Audit trail

- `scripts/phase4_step4_integrated_dataset.py`
- `data/processed/phase4_integrated_analysis_dataset.csv`
- `results/04_chemistry_crystallography/phase4_step4_analysis_population_flags.csv`
- `results/04_chemistry_crystallography/phase4_step4_integrated_summary.csv`
- `results/04_chemistry_crystallography/phase4_step4_integrated_validation.md`

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All chemistry is type C (mineral-level ideal/reference). Zero specimen-specific measured chemistry.
- No scientific hypothesis was tested.
- No chemistry-spectrum correlations were computed.
- No symmetry-spectrum correlations were computed.
- No ML, PCA, or clustering was run.
- No mineral difference tests were performed.
- No combined complexity score was created.
- The Phase 2 canonical dataset remains unchanged (checksum verified).
- The Phase 3 canonical feature matrix remains unchanged (checksum verified).
- The Phase 4 provenance mapping remains unchanged (checksum verified).

---

---

## Phase 5 Step 1 — Exploratory Analysis (frozen 2026-10-08 14:52 UTC)

### Purpose

First exploratory scientific analysis examining relationships between spectral complexity
descriptors (D1–D4), crystal symmetry, and reference chemistry.

**NOT** a causal analysis, model optimization, or ML study.

### Populations used

| Population | Spectra | Provenance Groups |
|------------|---------|-------------------|
| PRIMARY | 175 | 166 |
| SENSITIVITY_SYM | 258 | 246 |
| SENSITIVITY_CHEM | 359 | 290 |
| FULL | 454 | 382 |

### Major structural findings

- Spectral descriptors vary across crystal systems with substantial overlap
- Crystal system is the most practical symmetry level; point groups and space groups are too sparse
- Element–descriptor associations are modest; chemistry is highly collinear (compositional constraint)
- Mineral identity simultaneously determines chemistry, symmetry, and spectral behavior
- Provenance dependence and duplicate feature vectors are present

### Major limitations

- **mineral_identity_confounding:** Chemistry and symmetry are both determined by mineral identity
- **provenance_dependence:** 5 provenance groups have multiple spectra
- **chemistry_representation:** Reference chemistry is ideal/reference, NOT measured specimen chemistry
- **symmetry_ambiguity:** 0 spectra have polymorph-ambiguous symmetry
- **grain_size:** Grain-size variation may influence spectral behavior; not modelled here
- **instrument_effects:** Measurement-related variation exists (Phase 2); no post-hoc correction applied
- **d2_selection:** Only spectra covering 320-2550 nm included; conclusions apply to D2 population only

### Modelling recommendations

- **Symmetry To Descriptors:** POSSIBLE_WITH_LIMITATIONS
- **Chemistry To Descriptors:** POSSIBLE_WITH_LIMITATIONS
- **Symmetry Plus Chemistry To Descriptors:** JUSTIFIED_FOR_NEXT_STEP
- **Mineral Identity To Descriptors:** JUSTIFIED_FOR_NEXT_STEP

### Status

**COMPLETE — EXPLORATORY ANALYSIS FINISHED**

No causal conclusions drawn. No future modelling decisions frozen prematurely.

---

## Pending decisions

The following decisions remain open and require researcher input:

1. **Pyroxene class structure:** How to handle Augite/Diopside/Hypersthene/Enstatite overlap (49 cross-mineral duplicates). Options: merge into super-class, choose one directory per file, flag as ambiguous.
2. **Augite-Olivine duplicates:** 3 pairs of byte-identical files between Augite and Olivine directories. Cause unknown (mixture? misclassification? database error?).
3. **External data sources:** Which chemistry/crystallography databases to query (RRUFF, Mindat, RELAB database, literature).
4. **Grain-size treatment:** Whether to treat grain-size variants as replicates, separate classes, or covariates.

---


---

## Decision 157 - Phase 5 Step 2: Formal Modelling

**Date:** 2026-10-09 03:02 UTC
**Status:** COMPLETE

### Models evaluated

- M0: Baseline (intercept + provenance dependence)
- M1: Symmetry (crystal system)
- M2: Chemistry (reduced element set)
- M3: Joint (symmetry + chemistry)
- M4: Mineral identity

### Primary population

- 175 spectra
- 166 provenance groups
- Cluster-robust SEs by provenance group

### Dependence treatment

Cluster-robust standard errors by provenance group to account for within-group
correlation. Mixed-effects models were considered but not used as primary due to
low within-group replication (166 groups, 175 observations).

### Chemistry representation

**Approach:** reduced_element_set
**Elements:** Al, Ca, Fe, Mg, Si
**Rationale:** Reduced element set excluding oxygen (anion baseline), rare elements (≤10 nonzero spectra or ≤2 minerals), and one element from each perfectly collinear pair. This provides a chemically meaningful representation with manageable collinearity for the 18 mineral reference formulas.

### Major confounding findings

Each mineral maps to exactly one crystal system. Crystal system and mineral identity
are completely confounded. Independent effects cannot be separated. Chemistry and
symmetry are both determined by mineral identity.

### Sensitivity findings

- Representation sensitivity: native-5nm vs PCHIP (differences present)
- Chemistry population sensitivity: PRIMARY vs SENSITIVITY_CHEM (stable)
- Symmetry population sensitivity: PRIMARY vs SENSITIVITY_SYM (stable)

### Joint symmetry + chemistry model support

The joint model provides explanatory improvement over baseline for at least some
descriptors. However, interpretation is limited by mineral-identity confounding.

### Interpretable ML justification

**Decision:** POSSIBLE_BUT_NOT_YET_NEEDED

Interpretable ML is possible but not yet needed. The current formal modelling
provides sufficient insight into symmetry-chemistry-descriptor associations.
PySR and symbolic regression are deferred.


---

## Phase 5 Step 3 — Model Robustness Audit (2026-10-09 07:54 UTC)

### Audit Outcome

The Phase 5 Step 2 formal modelling results were reproduced and audited across 8 dimensions (A–H).

**Critical issues found:** 2
**High-severity issues found:** 11

### Critical Findings

1. **Model type mislabelled**: n_peaks_prom models are labelled "Negative Binomial (count)" in the Step 2 report, but the implementation uses log(y+1) OLS with cluster-robust SEs. This is NOT a negative binomial model.

2. **Complete mineral–crystal system confounding**: Every mineral in the PRIMARY population maps to exactly one crystal system. Crystal system and mineral identity effects are completely confounded and cannot be distinguished.

3. **DF convention mismatch**: P-values use t(df=n-p) but cluster-robust inference should use t(df=n_groups-1). With n_groups≈166, this makes p-values anti-conservative for models with many parameters.

### Go/No-Go Decisions

| Decision | Verdict | Rationale |
|----------|---------|-----------|
| Interpretable ML | NO-GO | Complete confounding means ML would learn mineral identity, not symmetry |
| PySR / symbolic regression | NO-GO | Overfitting risk with 175 spectra and 14 minerals; results would be mineral-specific |
| Additional statistics | CONDITIONAL | Only to correct DF convention and report consistent CIs/p-values |
| No further modelling | RECOMMENDED | Current M0–M4 sufficient for descriptive purposes; confounding cannot be resolved |

### Recommendations

1. Correct the model type label in the Step 2 report (log-linear OLS, not negative binomial)
2. Fix DF convention for cluster-robust inference (use n_groups - 1)
3. Acknowledge complete confounding as a fundamental dataset limitation
4. Do not proceed with interpretable ML or PySR
5. Report compositional interpretation caveat for chemistry coefficients
6. No further modelling beyond the audit corrections

### Frozen Decision

**No further modelling beyond audit corrections.** The Phase 5 Step 2 results stand as descriptive summaries with the caveats documented above. The complete mineral–crystal system confounding is a dataset limitation that cannot be resolved by additional statistical methods.


---

## Phase 5 Step 3.1 — Audit Findings Review and Decision Reconciliation (2026-10-09)

### Purpose

Systematic review of the Phase 5 Step 3 audit findings, reconciling them with the Step 2 formal modelling results and the existing decision record. Determines which claims are retained, qualified, corrected, or withdrawn.

### Review Summary

Seven reviews (A–G) were conducted covering reproduction validity, degrees of freedom, response distributions, PCHIP comparison, confounding, scientific conclusions, and final decisions.

### Key Findings

1. **Numerical reproducibility confirmed**: All 20 models reproduced exactly (AIC and R² match within 1e-4). The Step 2 implementation is numerically correct.

2. **D1 model type corrected**: n_peaks_prom models use log(y+1) OLS with cluster-robust SEs, not negative binomial. The Step 2 report label "Negative Binomial (count)" is incorrect and must be replaced with "OLS with log-transformed response."

3. **DF convention issue is real but practically small**: P-values use t(df=n-p=174) instead of t(df=n_groups-1=165). With 166 clusters the practical impact on significance conclusions is negligible, but the convention is technically incorrect for cluster-robust inference. CIs use z=1.96 (infinite DF), creating an internal inconsistency with the t-based p-values.

4. **Complete confounding verified**: All 14 minerals in the primary population map to exactly one crystal system. This is a structural dataset limitation that no statistical method can overcome.

5. **PCHIP overfitting claim withdrawn**: The higher R² for PCHIP-resampled spectra is real, but the audit's attribution to overfitting is unsupported. The cause is unresolved — could be population composition, descriptor distribution shifts, representation effects, or overfitting to a small subset (64 spectra).

6. **Influential observations noted but not dispositive**: Large Cook's distances (up to 37.67) reflect small-sample sensitivity with 175 spectra and 166 groups, not data errors.

### Claims Disposition

| Status | Count | Claims |
|--------|-------|--------|
| Retained | 5 | Reproducibility, confounding, interpretable ML deferral, PySR deferral, mineral identity R² |
| Qualified | 5 | Crystal system associations, chemistry associations, joint model improvement, cluster-robust SE validity, influential observations |
| Corrected | 1 | D1 model type (log-linear OLS, not negative binomial) |
| Withdrawn | 1 | PCHIP overfitting explanation (cause unresolved) |

### Final Decisions

| Direction | Status | Rationale |
|-----------|--------|-----------|
| Conventional statistical modelling | PROCEED_WITH_LIMITATIONS | M0–M4 results are reproducible and descriptively valid; all claims must acknowledge confounding and compositional caveats |
| Interpretable machine learning | DEFER | Complete mineral–crystal system confounding means ML would learn mineral identity, not independent symmetry effects |
| Symbolic regression / PySR | NOT_JUSTIFIED_WITH_CURRENT_DATA | Overfitting risk with 175 spectra and 14 minerals; results would not generalise beyond current mineral set |
| Additional data acquisition | PROCEED | Improving independent mineral coverage (minerals spanning multiple crystal systems or crystal systems with multiple minerals) is the single most scientifically useful next action |

### Recommended Next Action

**Improve independent mineral coverage.** The complete mineral–crystal system confounding is the single largest limitation. Acquiring spectra from minerals that break the 1:1 mineral-to-crystal-system mapping would enable identification of independent symmetry effects. This is more valuable than fitting additional models to the current data.

### Frozen Decision (Updated)

**No further modelling beyond the audit corrections documented in Step 3 and Step 3.1.** The Phase 5 Step 2 results stand as descriptive summaries with the following mandatory caveats:

1. D1 (n_peaks_prom) models use log-linear OLS, not negative binomial; R² values are pseudo-R² on the log scale.
2. Crystal system and mineral identity are completely confounded; symmetry effects cannot be separated from mineral identity.
3. Chemistry coefficients reflect reference/ideal mineral composition, not specimen-specific measured chemistry, and are conditional associations affected by compositional constraints.
4. Cluster-robust inference uses an inconsistent DF convention (t for p-values, z for CIs); with 166 clusters the practical impact is small.
5. Higher R² for PCHIP-resampled spectra has an unresolved cause; overfitting is one of several plausible explanations.

### Outputs

- `results/05_exploratory_analysis/phase5_step3_1_decision_reconciliation.md` — full review with evidence table
- `results/05_exploratory_analysis/phase5_step3_1_claims_register.csv` — machine-readable claims register

---

## Phase 5 Step 4 — Independent Mineral Coverage and Data Acquisition Design

**Date:** 2026-10-10
**Status:** COMPLETE — COVERAGE ASSESSMENT AND ACQUISITION PLAN FINISHED

### Purpose

Research-design assessment to identify what additional data would most improve scientific identifiability of the central hypothesis: "Crystal symmetry constrains the possible spectral patterns, while chemical and structural variation modifies and multiplies those patterns, producing measurable spectral complexity."

This is NOT a new modelling phase. No frozen data were modified. No new spectra were acquired.

### Key Findings

1. **Complete mineral–crystal system confounding confirmed.** All 18 minerals map 1:1 to exactly one crystal system. No crystal-system effect can be distinguished from mineral-identity effects.

2. **Singleton crystal systems.** Tetragonal (Rutile only, 4 spectra) and triclinic (Microcline only, 4 spectra) have zero within-system replication by different mineral identities.

3. **No polymorph pairs in dataset.** No case exists where the same chemical formula appears in two different crystal structures. This is the strongest available test of the symmetry hypothesis and is entirely absent.

4. **Zero specimen-level chemistry.** All 18 minerals have reference/ideal chemistry only (type C). No specimen has measured composition.

5. **Effective independence is limited.** 339 of 454 spectra (74.7%) participate in duplicate relationships. 49 duplicate feature vector groups contain 98 spectra with identical feature vectors.

### Acquisition Priorities

| Priority | Focus | Candidates |
|----------|-------|------------|
| HIGH | Polymorph pairs (same formula, different structure) | Aragonite (CaCO3), Anatase/Brookite (TiO2), Beta-quartz (SiO2), Sanidine (KAlSi3O8), Clinoenstatite (Mg2Si2O6), Al2SiO5 polymorphs |
| HIGH | Singleton system diversity | Additional tetragonal mineral (e.g., Zircon), additional triclinic mineral (e.g., Anorthite) |
| HIGH | Specimen-level measured chemistry | Any mineral with EPMA/WDXRF on the same specimen as spectroscopy |
| MEDIUM | Within-system diversity | Additional hexagonal mineral, additional cubic mineral |
| LOW | Under-represented minerals | More spectra for Beryl (1), Grossular (1), Orthoclase (3), Andradite (3) |

### Decision: Pre-conditions for Further Modelling

No further statistical modelling should be undertaken until:

1. At least one polymorph pair is acquired and integrated.
2. At least one singleton crystal system gains a second mineral identity.
3. A systematic RELAB availability search confirms which HIGH-priority candidates are obtainable.
4. The feasibility of obtaining specimen-level measured chemistry is assessed.

### Frozen Decision

**No new modelling. No data integration. No pipeline modifications.** This step produced a research design assessment only. All frozen datasets and prior results remain unchanged.

### Outputs

- `results/05_exploratory_analysis/phase5_step4_coverage_baseline.md` — verified current coverage and identifiability limitations
- `results/05_exploratory_analysis/phase5_step4_coverage_baseline_data.json` — machine-readable baseline data
- `results/05_exploratory_analysis/phase5_step4_candidate_acquisition_plan.csv` — ranked candidate minerals and selection criteria
- `results/05_exploratory_analysis/phase5_step4_data_requirements.md` — metadata and measurement requirements for future observations
- `results/05_exploratory_analysis/phase5_step4_research_design_recommendation.md` — synthesis, acquisition priorities, unresolved questions, and pre-conditions for further modelling
- `scripts/phase5_step4_coverage_baseline.py` — reproducible baseline computation from frozen inputs

---

## Phase 5 Step 4.1 — RELAB Candidate Availability and Polymorph Verification Audit

**Date:** 2026-10-10  
**Status:** COMPLETE — LOCAL INVENTORY AUDIT FINISHED; REMOTE CATALOGUE VERIFICATION INCOMPLETE

### Purpose

Candidate-verification and availability audit to determine whether suitable RELAB spectral measurements and supporting specimen information are available for the high-priority candidates identified in Phase 5 Step 4.

This is an availability audit only. No new spectra were downloaded or integrated. No frozen data were modified.

### Key Findings

1. **Zero polymorph candidates verified in local inventory.** None of the target polymorph minerals (aragonite, anatase, brookite, beta-quartz, sanidine, clinoenstatite, andalusite, kyanite, sillimanite) are present as separate directories in the local RELAB raw data (`data/raw/RELAB/`).

2. **One system-diversity candidate partially available.** Zircon (ZrSiO4, tetragonal) is present in the local RELAB inventory with 2 spectral files (mazi01.tab, mazi02.tab) but has incomplete wavelength coverage (1800–2680 nm only; does not cover the full D2 domain of 320–2550 nm).

3. **One additional cubic mineral present but not integrated.** Pyrope (Mg3Al2Si3O12, cubic) is present in the local RELAB inventory but not included in the canonical dataset. Reason for exclusion unknown; may fail eligibility criteria.

4. **Remote catalogue verification incomplete.** Web searches for the official RELAB catalogue and PDS Geosciences Node did not yield direct access to mineral-specific inventory lists or search functionality. The candidate minerals may exist in the full RELAB database but cannot be confirmed without exhaustive catalogue access.

5. **No candidate meets VERIFIED_SUITABLE_CANDIDATE status.** All candidates remain either NOT_FOUND_IN_SEARCHED_RESOURCES, VERIFIED_BUT_INCOMPLETE, or CATALOGUE_RECORD_ONLY pending exhaustive RELAB catalogue verification.

### Local RELAB Inventory

The local RELAB raw data directory contains 20 mineral directories:

- **Included in canonical dataset (18 minerals):** Albite, Andradite, Apatite, Augite, Beryl, Calcite, Diopside, Dolomite, Enstatite, Grossular, Hypersthene, Microcline, Muscovite, Olivine, Orthoclase, Quartz, Rutile, Spinel
- **Not included in canonical dataset (2 minerals):** Pyrope (cubic garnet), Zircon (tetragonal)

Total spectral files: 3,089,830 data points across all minerals.

### Candidate Verification Results

| Candidate ID | Mineral or category | Status | Wavelength coverage | Scientific value |
|--------------|---------------------|--------|---------------------|------------------|
| C01 | Aragonite (CaCO3) | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | VERY HIGH |
| C02 | Anatase or Brookite (TiO2) | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | VERY HIGH |
| C03 | Beta-quartz (SiO2) | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | HIGH |
| C04 | Sanidine (KAlSi3O8) | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | HIGH |
| C05 | Clinoenstatite (Mg2Si2O6) | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | HIGH |
| C06 | Zircon (ZrSiO4) — additional tetragonal | VERIFIED_BUT_INCOMPLETE | INCOMPLETE (1800–2680 nm only) | HIGH |
| C07 | Additional triclinic mineral | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | MEDIUM |
| C08 | Additional hexagonal mineral | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | MEDIUM |
| C09 | Al2SiO5 polymorphs (Andalusite/Kyanite/Sillimanite) | NOT_FOUND_IN_SEARCHED_RESOURCES | UNKNOWN | EXCEPTIONAL (if available) |
| C10 | Pyrope (Mg3Al2Si3O12) — additional cubic | CATALOGUE_RECORD_ONLY | UNKNOWN | LOW-MEDIUM |
| C11 | Minerals with measured specimen chemistry | NOT_FOUND_IN_SEARCHED_RESOURCES | N/A | HIGH (if available) |
| C12 | Under-represented minerals (Beryl, Grossular, Orthoclase, Andradite) | CATALOGUE_RECORD_ONLY | UNKNOWN | LOW |
| C13 | Distinct specimens for high-duplicate minerals | CATALOGUE_RECORD_ONLY | UNKNOWN | LOW |

### Compatibility with Frozen Dataset

**Zircon (C06) assessment:**

- Wavelength coverage: INCOMPLETE (1800–2680 nm only; missing 320–1800 nm portion of D2 domain)
- Native resolution: HIGH (~0.6 nm spacing; finer than 5 nm D2 grid)
- Measurement conditions: COMPATIBLE (bidirectional reflectance, 30°/30° geometry)
- Specimen metadata: PARTIAL (sample IDs documented; grain size and preparation not documented in file headers)
- Chemistry: Zircon is ZrSiO4 (tetragonal, I41/amd), chemically distinct from Rutile (TiO2)

**All other candidates:** Cannot assess compatibility without access to spectral files or detailed catalogue records.

### Decision: Pre-conditions for Data Acquisition

The audit confirms that exhaustive RELAB catalogue verification is required before proceeding with data acquisition. The following actions are recommended:

1. **Direct RELAB database access:** Contact RELAB curators at Brown University for complete mineral inventory list and search for Tier 1 candidates (C09 Al2SiO5 polymorphs, C01 aragonite, C02 anatase/brookite).

2. **PDS Geosciences Node archive search:** Search PDS archive for RELAB spectral files using mineral name filters to verify candidate availability.

3. **Zircon partial integration evaluation:** Evaluate whether the 1800–2680 nm wavelength range of Zircon spectra is scientifically useful for the project's goals, or whether full D2 coverage (320–2550 nm) is required.

4. **Literature review:** Search publications citing RELAB data for mentions of candidate minerals to identify potential availability.

### Frozen Decision

**No new data integration. No pipeline modifications. No modelling.** This step produced an availability audit only. All frozen datasets and prior results remain unchanged.

The audit does not confirm or deny the availability of candidate minerals in the full RELAB database. Absence from the local raw data directory does not imply absence from RELAB. Exhaustive verification requires direct database access or curator collaboration.

### Outputs

- `results/05_exploratory_analysis/phase5_step4_1_relab_availability_audit.md` — full audit report with candidate verification results, compatibility assessment, and ranked opportunities
- `results/05_exploratory_analysis/phase5_step4_1_candidate_availability.csv` — machine-readable candidate availability records
- `results/05_exploratory_analysis/phase5_step4_1_search_log.csv` — detailed search log documenting all verification attempts

### Recommended Next Action

**Proceed to Phase 5 Step 4.2 (formal candidate prioritization and acquisition planning)** using the current audit as a baseline. Prioritize exhaustive RELAB catalogue verification for Tier 1 candidates before committing to data acquisition.

If Zircon (C06) is deemed scientifically useful despite incomplete wavelength coverage, evaluate eligibility for integration through the frozen pipeline. Otherwise, seek additional tetragonal minerals with full D2 coverage.

---

## Phase 5 Step 4.2 — Candidate Prioritisation and Acquisition Plan

**Date:** 2026-10-10
**Status:** COMPLETE (planning step only; no data acquired or integrated)

### Purpose

Formal prioritisation of 13 candidate acquisition opportunities identified in Phase 5 Step 4, reconciled against the RELAB availability audit (Step 4.1). Establishes ranked acquisition plan, curator enquiry draft, and acceptance criteria for future canonical integration.

### Key Decisions

#### 1. Priority Re-Ranking

Step 4 initial classifications were revised in light of Step 4.1 audit findings:

- **C09 (Al2SiO5 polymorphs):** Retained as HIGH priority. Strongest possible test of symmetry hypothesis — three polymorphs, one shared ideal formula, two crystal systems. Not found in local inventory; requires exhaustive RELAB catalogue verification. Space groups: andalusite Pnnm (corrected from Pbca), kyanite P-1, sillimanite Pbnm. Specimen-level chemistry not measured; ideal formula identity does not guarantee identical specimen composition.
- **C01 (Aragonite):** Retained as HIGH priority. Shared ideal formula (CaCO3) with Calcite but orthorhombic vs trigonal — direct polymorph test. Not found in local inventory. Specimen-level chemistry not measured; ideal formula identity does not guarantee identical specimen composition.
- **C02 (Anatase/Brookite):** Retained as HIGH priority. Shared ideal formula (TiO2) with Rutile; anatase is tetragonal (different space group), brookite is orthorhombic (different system). Not found in local inventory. Specimen-level chemistry not measured; ideal formula identity does not guarantee identical specimen composition.
- **C06 (Zircon):** Downgraded from HIGH to MEDIUM. VERIFIED_BUT_INCOMPLETE — wavelength coverage 1800.0-25920.7 nm (NOT 1800-2680 nm as previously reported), highly non-uniform sampling grid (0.6-129 nm spacing; mean 9.0 nm, median 2.2 nm), instrumental spectral resolution (FWHM) unknown. Does not cover full D2 domain (320-2550 nm). Cannot enter frozen pipeline. May be useful for separate partial-coverage analysis with explicit limitations.
- **C03 (Beta-quartz):** Downgraded from HIGH to MEDIUM. Feasibility concerns: stable only above 573°C; high-T measurements differ from RELAB ambient protocol; quenched beta may revert to alpha.
- **C04 (Sanidine):** Downgraded from HIGH to MEDIUM. Al-Si ordering degree uncertain; may be difficult to distinguish from orthoclase spectroscopically.
- **C05 (Clinoenstatite):** Downgraded from HIGH to MEDIUM. May invert to enstatite at ambient conditions; pure specimens rare.
- **C07 (Additional triclinic):** MEDIUM priority. Kyanite (from C09) would also address this gap.
- **C11 (Specimen chemistry):** Downgraded from HIGH to MEDIUM. Very low feasibility — specimen-level measured chemistry rarely available in spectral libraries; requires co-located EPMA/XRF analysis.
- **C08, C10, C12, C13:** LOW priority. Address within-system diversity or within-mineral replication but do not resolve mineral-system confounding.

#### 2. Pyrope Reconciliation

Step 4.1 audit noted Pyrope (C10) was "present but not in canonical dataset; reason for exclusion unknown." Investigation of file headers and prior decisions revealed three exclusion reasons:

1. **Wavelength coverage incomplete:** Spectra start at 350 nm, not 320 nm (Decision 16, line 129). Does not cover full D2 domain.
2. **Non-standard measurement geometry:** Detection angle 0.00° vs standard 30.00°. Incompatible with existing dataset geometry.
3. **Uncertain mineral identity:** File headers contain "GARNET ??" — identity not confirmed by XRD or other independent method.

**Decision:** Pyrope is NOT eligible for integration. It fails D2 compatibility, measurement protocol consistency, and identity confidence criteria. It remains classified as LOW priority (C10) for future consideration only if confirmed specimens with full D2 coverage and standard geometry become available.

#### 3. Zircon Assessment

Zircon (C06) is VERIFIED_BUT_INCOMPLETE:

- **Identity:** Confirmed (ZrSiO4, tetragonal, I41/amd). Chemically distinct from Rutile (TiO2).
- **Wavelength coverage:** 1800.0-25920.7 nm (NOT 1800-2680 nm as previously reported). Missing 320-1800 nm portion of D2 domain. Upper bound extends to 25921 nm, well beyond D2.
- **Sampling grid:** Highly non-uniform. Five spacing regimes from 0.6 nm (1800-2000 nm) to 129 nm (10000-25921 nm). Mean spacing 9.0 nm, median 2.2 nm. The previous report of "~0.6 nm spacing" incorrectly described only the minimum spacing in the shortest-wavelength region.
- **Instrumental spectral resolution:** Unknown. Sampling interval (spacing between measurements) is NOT the same as instrumental spectral resolution (FWHM/bandwidth). The latter is not documented in file headers.
- **Measurement conditions:** Standard RELAB protocol (30°/30° geometry).
- **Pipeline compatibility:** INCOMPATIBLE with frozen 13-step D2 workflow. Cannot enter canonical dataset.

**Decision:** Zircon cannot be integrated through the frozen pipeline. Option for separate partial-coverage analysis exists but requires explicit documentation of limitations (cannot compare directly with D2-domain minerals; introduces wavelength-dependent confounding). Not recommended unless no full-coverage tetragonal candidates are found.

#### 4. Curator Enquiry

Draft enquiry prepared for RELAB curators at Brown University. Enquiry requests:

- Availability of polymorph candidates (C09, C01, C02, C03, C04, C05)
- Metadata for Pyrope and Zircon (reason for exclusion from canonical dataset; identity confirmation; measurement protocol)
- Existence of additional tetragonal, triclinic, and hexagonal minerals with full D2 coverage
- Specimen-level chemistry metadata availability

**Status:** DRAFT prepared. NOT SENT. Requires review and contact details verification before dispatch.

#### 5. Acceptance Criteria

Seven acceptance criteria established for future canonical integration. Evidence types must be documented separately:

1. **Mineral identity and structural assignment:** Verified mineral ID by XRD or equivalent; crystal system and space group confirmed; polymorph identity confirmed if applicable. Each evidence type (mineral identity, polymorph identity, crystal system, space group) documented separately with confidence level.
2. **Provenance documentation:** Specimen ID, locality, collector, donation date; grain size and preparation documented.
3. **Spectral coverage and measurement metadata:** Wavelength range 320-2550 nm covers full D2 domain. Sampling interval (spacing between measurements) and instrumental spectral resolution (FWHM/bandwidth) documented separately — these are distinct properties. Wavelength grid uniformity assessed. If instrumental resolution unknown, explicitly flagged.
4. **Duplicate independence:** Genuinely independent specimens (different locality, different specimen); not repeat measurements of same specimen.
5. **Chemistry metadata:** Specimen-level measured chemistry (EPMA, XRF, ICP) highly desirable; mineral-level ideal chemistry acceptable if no solid solution variation. Ideal formula identity explicitly distinguished from measured specimen composition.
6. **Pipeline compatibility:** Compatible with frozen 13-step D2 workflow (native sampling interval ≤10 nm; wavelength range covers D2 domain; measurement geometry matches RELAB protocol). Note: "native resolution" refers to sampling interval, not instrumental spectral resolution.
7. **Integration decision:** Explicit documentation of which confounding gap the candidate addresses; whether it resolves mineral-system confounding or improves within-system diversity.

Partial-coverage candidates (e.g., Zircon) require separate acceptance criteria and explicit limitations documentation.

### Frozen Decisions

- **No data acquired or integrated.** This is a planning step only.
- **Frozen pipeline unchanged.** No modifications to 13-step D2 workflow.
- **Canonical dataset unchanged.** No new spectra added.
- **Pyrope and Zircon excluded from canonical integration** based on documented criteria.

### Outputs

- `results/05_exploratory_analysis/phase5_step4_2_acquisition_priority_matrix.csv` — ranked candidate matrix with priority, rationale, next actions, acceptance conditions.
- `results/05_exploratory_analysis/phase5_step4_2_acquisition_action_plan.md` — comprehensive narrative report with reconciliation, ranked opportunities, acquisition routes, curator enquiry reference, acceptance criteria.
- `results/05_exploratory_analysis/phase5_step4_2_curator_enquiry_draft.md` — draft enquiry to RELAB curators (DRAFT, NOT SENT).

### Recommended Next Action

**Await user review of acquisition plan and curator enquiry draft.** Upon approval, contact RELAB curators to verify candidate availability and request metadata. Do not proceed with data acquisition or integration until curator response received and candidates verified against acceptance criteria.

If curator contact is not feasible or does not yield results, proceed with literature review to identify alternative spectral databases or published datasets containing polymorph candidates with full D2 coverage.

Do not modify frozen pipeline or canonical dataset. Do not begin modelling.

### Correction Entry (2026-10-10)

**Date:** 2026-10-10  
**Type:** Factual correction and clarification  
**Scope:** Phase 5 Step 4.2 outputs only (priority matrix CSV, action plan, curator enquiry draft, this decision entry)

#### Corrections made

1. **Andalusite space group corrected:** Changed from Pbca to Pnnm in all Phase 5 Step 4.2 documents. Pbca is the space group for sillimanite, not andalusite. Andalusite is orthorhombic Pnnm. This correction affects C09 (Al2SiO5 polymorphs) in the priority matrix CSV, action plan Section 3.2, curator enquiry draft Tier 1 candidates, and this decision entry.

2. **Zircon wavelength range corrected:** Previous reports stated "1800-2680 nm" but inspection of raw RELAB files (mazi01.tab, mazi02.tab) shows the actual wavelength range is 1800.0-25920.7 nm. The "2680 nm" figure appears to have been confused with the point count (2681). The upper bound extends to 25921 nm, not 2680 nm. This correction affects C06 (Zircon) in the priority matrix CSV, action plan Section 2.1 and Section 3.3, curator enquiry draft Tier 3 candidates, and this decision entry.

3. **Zircon sampling grid characterization corrected:** Previous reports stated "~0.6 nm native spacing" but the actual sampling grid is highly non-uniform across 5 spacing regimes:
   - 1800-2000 nm: 0.6-0.8 nm spacing (mean 0.69 nm)
   - 2000-3000 nm: 0.7-1.8 nm spacing (mean 1.16 nm)
   - 3000-5000 nm: 1.7-4.9 nm spacing (mean 2.90 nm)
   - 5000-10000 nm: 4.8-19.3 nm spacing (mean 9.66 nm)
   - 10000-25921 nm: 19.3-129 nm spacing (mean 50.04 nm)
   
   Overall: mean 9.0 nm, median 2.2 nm, min 0.6 nm, max 129 nm. The "~0.6 nm" figure represents only the minimum spacing in the shortest-wavelength region, not the typical or mean spacing. This correction affects C06 in the priority matrix CSV, action plan Section 2.1 and Section 3.3, and this decision entry.

4. **Sampling interval vs instrumental spectral resolution distinguished:** Previous documents conflated sampling interval (spacing between wavelength measurements) with instrumental spectral resolution (FWHM/bandwidth). These are distinct properties. For zircon, the sampling interval varies from 0.6 to 129 nm, but the instrumental spectral resolution (FWHM) is unknown and not documented in RELAB file headers. All Phase 5 Step 4.2 documents now explicitly distinguish these two properties and mark instrumental resolution as unknown where not documented. This correction affects the priority matrix CSV (C06 evidence_available, missing_evidence, notes fields), action plan Section 2.1 and Section 6.3, curator enquiry draft Section 2.3 item 2, and this decision entry.

5. **Polymorph comparison language qualified:** Previous documents used unqualified phrases like "identical chemistry," "same formula," and "at fixed chemistry" when describing polymorph comparisons. These have been qualified to state "shared ideal formula" and to explicitly note that specimen-level chemistry has not been measured. Ideal formula identity does not guarantee identical specimen composition due to potential solid solution, impurities, or compositional variation. This correction affects C01, C02, C03, C04, C05, and C09 in the priority matrix CSV and action plan Section 3.2-3.3, and the curator enquiry draft Tier 1 candidates.

6. **Acceptance criteria strengthened:** The acceptance criteria in action plan Section 6 and this decision entry Section 5 have been updated to:
   - Explicitly separate evidence types (mineral identity, polymorph identity, crystal structure, chemistry, provenance, instrumental resolution)
   - Distinguish sampling interval from instrumental spectral resolution
   - Require documentation of wavelength grid uniformity
   - Clarify that "native resolution" in pipeline compatibility refers to sampling interval, not instrumental resolution
   - Note that ideal formula identity must be distinguished from measured specimen composition

#### Files modified

- `results/05_exploratory_analysis/phase5_step4_2_acquisition_priority_matrix.csv` — andalusite space group, zircon wavelength and grid, qualified polymorph language
- `results/05_exploratory_analysis/phase5_step4_2_acquisition_action_plan.md` — zircon metadata, andalusite space group, qualified polymorph language, sampling vs resolution distinction, strengthened acceptance criteria
- `results/05_exploratory_analysis/phase5_step4_2_curator_enquiry_draft.md` — andalusite space group, zircon wavelength, qualified polymorph language, sampling vs resolution in information requested
- `docs/research_decisions.md` — this correction entry plus inline corrections to Phase 5 Step 4.2 decision entry

#### Files NOT modified

- No raw RELAB files modified
- No frozen Phase 2-4 datasets modified
- No canonical feature matrix modified
- No statistical analyses modified
- No new spectra acquired or integrated

#### Validation

- All 1,017 raw RELAB checksums remain unchanged
- Frozen Phase 2 canonical dataset (454 spectra) unchanged
- Frozen Phase 3 feature matrix (454 rows) unchanged
- Frozen Phase 4 provenance mapping (382 groups) unchanged
- Zircon remains excluded from canonical D2 workflow
- Brookite remains preferred C02 target (orthorhombic, cross-system comparison)
- No pipeline modifications


---

## Phase 5 Step 4.3 — External Archive Search and Curator Enquiry Preparation

**Date:** 2026-10-10
**Status:** COMPLETE (search and preparation only; no data acquired or integrated)

### Purpose

Systematic external archive search for priority candidate minerals across RELAB and PDS resources. Preparation of a revised curator enquiry draft with verified contact details for final human review.

### Search Scope

1. **RELAB official website** (https://sites.brown.edu/relab/) — contact details, database structure
2. **PDS Geosciences Node Spectral Library** (https://pds-speclib.rsl.wustl.edu/) — archive scale, search interface, contact
3. **Mineral-specific web searches** — RELAB catalogue records for each priority candidate
4. **Relevant primary literature** — where it helps verify specimen identity or polymorph assignments

### Key Findings

#### 1. Contact Details Verified

- **Takahiro Hiroi** (RELAB operator): `takahiro_hiroi@brown.edu`
- **Ralph Milliken** (RELAB principal): `ralph_milliken@brown.edu`
- **PDS Spectral Library**: `speclib@wunder.wustl.edu`

Source: Official RELAB website and PDS Spectral Library website, accessed 2026-10-10.

#### 2. RELAB Database Scale Confirmed

- **12,503 specimens** with **23,598 spectra** of terrestrial and extraterrestrial materials.
- The local raw data directory (20 mineral directories, 1,017 files) represents a small curated subset of the full database.

#### 3. No Priority Candidate Found via Public Web Search

All priority candidates (C01–C09, C11) remain NOT_FOUND_IN_SEARCHED_RESOURCES. The RELAB website does not provide a searchable mineral inventory. The PDS Spectral Library search interface requires an interactive session. Public web searches returned predominantly Chinese-language resource summaries and general mineralogy literature, not direct RELAB catalogue records.

**Interpretation:** Absence from public web search results does not imply absence from the full RELAB database. Direct curator contact is the only viable route to verify availability.

#### 4. Zircon (C06) Status Unchanged

Zircon remains VERIFIED_BUT_INCOMPLETE. No evidence of additional zircon specimens with broader wavelength coverage was found in searched resources. The 2 known spectra (ZI-EAC-001, ZI-EAC-002) have wavelength range 1800.0–25920.7 nm with highly non-uniform sampling (0.6–129 nm spacing; mean 9.0 nm, median 2.2 nm).

#### 5. Curator Enquiry Revised

The Step 4.2 draft enquiry has been revised (Step 4.3) with:
- Verified contact details
- Refined candidate list (Tier 1: C09, C01, C02; Tier 2: C04, C05, C06, C07; Tier 3: C03, C11)
- Brookite prioritised over anatase for C02
- Explicit distinction between sampling interval and instrumental resolution
- Qualified polymorph comparison language ("shared ideal formula")
- Request for recommended search route if a more efficient method exists

### Frozen Decisions

- **No data acquired or integrated.** This is a search and preparation step only.
- **Frozen pipeline unchanged.** No modifications to 13-step D2 workflow.
- **Canonical dataset unchanged.** No new spectra added.
- **No enquiry sent.** The curator enquiry draft is marked DRAFT — NOT SENT and requires explicit approval before dispatch.
- **All candidate statuses unchanged.** No candidate promoted to VERIFIED_SUITABLE_CANDIDATE.

### Outputs

- `results/05_exploratory_analysis/phase5_step4_3_external_search_log.csv` — search log with entries S19–S32 documenting all external searches
- `results/05_exploratory_analysis/phase5_step4_3_candidate_availability_evidence.csv` — detailed evidence table for priority candidates
- `results/05_exploratory_analysis/phase5_step4_3_external_search_report.md` — short report (8 sections)
- `results/05_exploratory_analysis/phase5_step4_3_curator_enquiry_draft.md` — revised curator enquiry draft (DRAFT, NOT SENT)

### Recommended Next Action

**Research team reviews and approves the curator enquiry draft.** Once approved, the enquiry is sent to Takahiro Hiroi (`takahiro_hiroi@brown.edu`) with optional CC to Ralph Milliken and PDS Spectral Library. Allow 2–4 weeks for response. Evaluate response against acceptance criteria before proceeding with any data acquisition.

**Do NOT send the enquiry, acquire data, integrate spectra, modify frozen datasets, or begin modelling without a separate decision.**

### Decision: Anatase Space Group Typo Correction (P41/amd → I41/amd)

**Date:** 2026-10-12

**Summary:** Corrected an erroneous Anatase space-group symbol in descriptive documentation.

- **Original (incorrect):** `P41/amd`
- **Corrected:** `I41/amd` (space group No. 141, body-centred tetragonal)

**Scope:** Descriptive-only. The error appeared in the `chemistry_notes` field of the Rutile row in the mineral reference CSV and was copied into seven Phase 5 exploratory descriptive files. No structured crystallographic data, eligibility classifications, model inputs, or scientific conclusions were affected.

**Affected files (8 total):**
1. `data/processed/phase4_mineral_reference.csv` — Rutile row, `chemistry_notes` field
2. `results/05_exploratory_analysis/phase5_step4_coverage_baseline_data.json` — `mineral_reference_summary` Rutile entry, `chemistry_notes`
3. `results/05_exploratory_analysis/phase5_step4_candidate_acquisition_plan.csv` — C02 row, `target_space_group`
4. `results/05_exploratory_analysis/phase5_step4_1_candidate_availability.csv` — C02 row, `target_space_group`
5. `results/05_exploratory_analysis/phase5_step4_1_relab_availability_audit.md` — C02 scientific rationale
6. `results/05_exploratory_analysis/phase5_step4_2_curator_enquiry_draft.md` — Tier 1 candidate list
7. `results/05_exploratory_analysis/phase5_step4_2_acquisition_priority_matrix.csv` — C02 row, `target_space_group`
8. `results/05_exploratory_analysis/phase5_step4_2_acquisition_action_plan.md` — C02 scientific value description

**Audit finding:** Phase 4 Metadata Error Impact Audit confirmed that structured Rutile metadata (space group `P42/mnm`, No. 136, tetragonal) and all frozen Phase 2–4 analytical datasets are unaffected. No regeneration of frozen data is required.

**Validation performed:**
- Source CSV parses correctly; only the `chemistry_notes` field in the Rutile row was changed
- All 8 files no longer contain `P41/amd` as an Anatase assignment
- Corrected symbol `I41/amd` confirmed present in all 8 files
- Rutile structured metadata (`P42/mnm`, 136, tetragonal) unchanged
- No frozen Phase 2–4 analytical datasets were modified
- MD5 checksums recorded before and after editing

---

## Phase 5 Steps 4.5–4.7 — Polymorph Candidate Audit, Provenance Review, and Correspondence Tracking

**Date:** 2026-10-12
**Status:** COMPLETE (audit, review, and tracking only; no data acquired or integrated)

### Purpose

Three-step investigation of 180 raw spectral files for five polymorph candidate minerals (Anatase, Andalusite, Aragonite, Brookite, Kyanite) found in `data/raw/RELAB/` but not included in the Phase 1 freeze or any subsequent phase.

- **Step 4.5:** File inventory and D2 compatibility audit
- **Step 4.6:** Targeted provenance and metadata verification
- **Step 4.7:** Curator correspondence and evidence tracking

### Key Findings

1. **180 files inventoried.** All 180 reported files located. 180 distinct MD5 checksums. 143 distinct sample groups. 62 files are D2-compatible or potentially D2-compatible (4 suitable, 58 potentially suitable).

2. **One identity conflict confirmed.** Brookite/c1mr130.tab is byte-identical (MD5 `d614f4c1b5276a73994d990f19811435`) to the existing Phase 1-2 frozen record F0899 (Rutile/c1mr130.tab). The Brookite copy must be excluded pending curator resolution.

3. **Four exact D2 matches verified.** Andalusite c1sr73/c1sr73a and Kyanite c1sr72/c1sr72a all have complete embedded metadata. However, each pair shares a source sample (WAR-0482 and WAR-4482 respectively) and represents two grain-size fractions, not independent specimens.

4. **Specimen independence is the primary unresolved question.** At least 35 of the 62 candidates are likely grain-size fractions or replicate measurements of fewer physical specimens. Estimated independent specimen count ranges from 8 to 48 pending curator confirmation.

5. **10 curator questions prepared across 6 priority levels.** Q1 (identity conflict), Q2-Q5 (specimen independence), Q6-Q7 (missing metadata), Q8 (measurement geometry), Q9 (wavelength calibration), Q10 (polymorph pairing). All drafts; none sent.

6. **No correspondence has been sent.** All three draft enquiries (Steps 4.2, 4.3, 4.6) explicitly state "NOT SENT." No contact addresses documented in project for individual operators/PIs. RELAB contact details verified in Step 4.3 (takahiro_hiroi@brown.edu, ralph_milliken@brown.edu) but no enquiry dispatched.

### Frozen Decisions

- **No data acquired or integrated.** All 180 raw files remain unmodified.
- **No spectra preprocessed, interpolated, normalized, or resampled.**
- **Frozen Phase 1-4 datasets unchanged.**
- **No models rerun.**
- **Brookite/c1mr130.tab excluded** from any extension candidate pool pending curator resolution.
- **No enquiry sent** without separate approval decision.

### Outputs

**Step 4.5:**
- `results/05_exploratory_analysis/phase5_step4_5_polymorph_audit/phase5_step4_5_audit_report.md`
- `results/05_exploratory_analysis/phase5_step4_5_polymorph_audit/phase5_step4_5_file_inventory.csv`
- `results/05_exploratory_analysis/phase5_step4_5_polymorph_audit/phase5_step4_5_d2_compatibility.csv`

**Step 4.6:**
- `results/05_exploratory_analysis/phase5_step4_6_provenance_review/phase5_step4_6_candidate_metadata.csv`
- `results/05_exploratory_analysis/phase5_step4_6_provenance_review/phase5_step4_6_identity_conflicts.csv`
- `results/05_exploratory_analysis/phase5_step4_6_provenance_review/phase5_step4_6_curator_questions.md`
- `results/05_exploratory_analysis/phase5_step4_6_provenance_review/phase5_step4_6_provenance_review_report.md`

**Step 4.7:**
- `results/05_exploratory_analysis/phase5_step4_7_correspondence_evidence/phase5_step4_7_correspondence_tracker.csv`
- `results/05_exploratory_analysis/phase5_step4_7_correspondence_evidence/phase5_step4_7_evidence_tracker.csv`
- `results/05_exploratory_analysis/phase5_step4_7_correspondence_evidence/phase5_step4_7_correspondence_report.md`

### Recommended Next Action

**Research team reviews and approves the combined curator enquiry.** The Step 4.3 general availability draft and Step 4.6 polymorph-specific questions should be consolidated into a single comprehensive enquiry to the RELAB curator. Do not send, acquire data, integrate spectra, modify frozen datasets, or begin modelling without a separate decision.

After curator response:
1. Update evidence tracker with responses
2. Reassess specimen counts based on Q1-Q5 answers
3. Make evidence-based decision on whether a separately versioned spectral extension is justified
4. If justified, design extension protocol (separate from this step)

---

## Phase 5 Step 5.1 — Extension Dataset Architecture and Inclusion Framework Design

**Date:** 2026-10-13
**Status:** COMPLETE (design and documentation only; no data acquired or integrated)

### Purpose

Design the architecture and governance framework for a future CrystalSym-Spec spectral extension dataset. This step prepares the project for possible incorporation of additional mineral spectra (Anatase, Andalusite, Aragonite, Brookite, Kyanite, and future candidates) while preserving the integrity of the frozen canonical dataset.

### Core Design Principle

The existing 454-spectrum dataset remains **CANONICAL_v1**. It is frozen.

No extension spectrum may:
- replace any existing spectrum in CANONICAL_v1
- alter any previously computed feature values
- modify any previous modelling inputs or outputs
- invalidate any previous scientific conclusions

The extension dataset is a **separate, independently versioned collection** that coexists with CANONICAL_v1 but does not modify it.

### Key Findings

1. **Extension architecture designed.** Proposed directory structure with 8 managed files (raw_manifest, file_inventory, provenance_metadata, spectral_compatibility, specimen_relationships, inclusion_decisions, exclusion_log, extension_freeze_report). Each file has documented purpose, required fields, and update rules.

2. **Three-level inclusion criteria defined.** Level 1 (file integrity, identity, duplicate check) required for any inclusion. Level 2 (D2 compatibility, specimen grouping) required for quantitative analysis. Level 3 (polymorph identity, independence, chemistry) required for polymorph conclusions. Ideal formula identity is explicitly distinguished from measured specimen chemistry.

3. **Specimen independence framework established.** File count is separated from independent specimen count. Decision rules defined for grain-size fractions, replicate measurements, preparation variants, and unknown relationships. Current candidates: 180 files, estimated 12-158 independent specimens (range reflects unresolved curator questions).

4. **Compatibility framework defined.** Four categories: exact D2, resample-compatible, partial coverage, incompatible. No spectrum may be silently transformed between categories.

5. **Future polymorph analysis modules documented.** Module A (Al2SiO5: Andalusite + Kyanite), Module B (TiO2: Rutile + Brookite + Anatase), Module C (CaCO3: Calcite + Aragonite). None performed in this step.

6. **Blockers identified.** Brookite identity conflict (Q1), specimen independence for Andalusite/Kyanite/Aragonite (Q2-Q5), metadata gaps (Q6-Q7), wavelength calibration (Q9). All require curator response.

### Frozen Decisions

- **Canonical dataset remains CANONICAL_v1 and is frozen.** No extension spectrum modifies any canonical record.
- **Extension requires separate versioning.** First extension version would be `phase5_extension_v1`.
- **No spectra integrated during this step.** This is a design and documentation step only.
- **No modelling rerun.** No analysis performed.
- **File count is never reported as specimen count.** Both counts are always separate.
- **Ideal formula identity is not equivalent to measured specimen chemistry.** The distinction is maintained throughout.

### Outputs

- `results/05_exploratory_analysis/phase5_step5_1_extension_framework/phase5_step5_1_extension_architecture.md`
- `results/05_exploratory_analysis/phase5_step5_1_extension_framework/phase5_step5_1_inclusion_criteria.md`
- `results/05_exploratory_analysis/phase5_step5_1_extension_framework/phase5_step5_1_specimen_independence_framework.md`
- `results/05_exploratory_analysis/phase5_step5_1_extension_framework/phase5_step5_1_extension_status_report.md`

### Recommended Next Action

**Research team reviews and approves the consolidated curator enquiry** (`phase5_step4_8_consolidated_enquiry_draft.md`). After sending and receiving curator responses, reassess specimen counts and make the extension decision.

Do NOT send the enquiry, acquire data, integrate spectra, modify frozen datasets, or begin modelling without a separate decision.

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums remain unchanged.
- Frozen Phase 2 canonical dataset (454 spectra) unchanged.
- Frozen Phase 3 feature matrix (454 rows) unchanged.
- Frozen Phase 4 provenance mapping (382 groups) unchanged.
- No new spectra processed, interpolated, normalized, or resampled.
- No spectral features engineered.
- No ML/PySR analysis performed.
- No curator correspondence sent.

---

## Phase 5 Step 5.7 — Scientific Inclusion Adjudication

**Date:** 2026-10-15
**Status:** COMPLETE (adjudication decisions only; no spectra processed or datasets created)

### Purpose

Adjudicate the scientific inclusion status of 49 POP_1 candidates after Step 5.6 D2 preprocessing validation. Assign each candidate one of four dispositions: PRIMARY_ELIGIBLE, SENSITIVITY_ONLY, HOLD_PENDING_EVIDENCE, or EXCLUDE.

### Key Decision: Biogenic Carbonate Population Assignment

**Decision:** Seven biogenic carbonate candidates (EXT_066, EXT_078, EXT_079, EXT_080, EXT_107, EXT_108, EXT_117) are assigned **SENSITIVITY_ONLY** rather than PRIMARY_ELIGIBLE.

**Reason:** These candidates have confirmed aragonite identity (PDS XML labels identify them as aragonite from biogenic sources: mussel shell, Halimeda green alga). However, the Step 5.7 spec requires that we "do not assume that a coral-derived or otherwise biogenic sample is scientifically interchangeable with other aragonite samples" while also requiring that we "do not infer that biogenic origin alone makes a sample invalid as aragonite." The SENSITIVITY_ONLY disposition honors both constraints: biogenic carbonates are retained for analysis but in a separate population that allows assessment of whether their spectral properties differ systematically from mineral-separate aragonite.

**Evidence:** 
- PDS XML labels confirm aragonite identity for all 7 candidates (IC_018, IC_019, IC_020, IC_021 in identity conflict register)
- Material origin is biogenic (mussel shell, Halimeda alga, coral)
- No evidence of deliberate mixture or thermal treatment
- Step 5.3.1 flagged these as biogenic_carbonate with low severity
- Step 5.5 design placed biogenic candidates in POP_1 but noted the interchangeability question

**Alternatives considered:**
1. **PRIMARY_ELIGIBLE:** Rejected because it assumes biogenic and mineral-separate aragonite are interchangeable without evidence
2. **EXCLUDE:** Rejected because biogenic origin alone does not invalidate aragonite identity
3. **HOLD_PENDING_EVIDENCE:** Rejected because the evidence is sufficient for a disposition; the question is whether to treat them as interchangeable, not whether they are valid aragonite

### Adjudication Results

- **PRIMARY_ELIGIBLE:** 42 candidates (22 source groups)
- **SENSITIVITY_ONLY:** 7 candidates (7 source groups, all biogenic carbonate)
- **HOLD_PENDING_EVIDENCE:** 0 candidates
- **EXCLUDE:** 0 candidates

All 49 POP_1 candidates passed Step 5.6 technical validation (8/8 checks). The 7 biogenic candidates are technically eligible for D2 processing but are assigned to a separate sensitivity population for scientific reasons.

### Population Definitions

**POP_1_PRIMARY (42 files, 22 source groups):** Mineral-separate aragonite and non-aragonite minerals (Anatase, Andalusite, Kyanite) with confirmed identity and no biogenic origin. Includes grain-size fractions and replicate measurements. Independence bounds: lower=22, upper=42.

**POP_1_BIOGENIC (7 files, 7 source groups):** Biogenic carbonate aragonite (mussel shell, Halimeda alga). Confirmed aragonite identity but not assumed interchangeable with mineral-separate aragonite. Independence bounds: lower=7, upper=7.

**POP_1_TOTAL (49 files, 29 source groups):** Union of POP_1_PRIMARY and POP_1_BIOGENIC. Independence bounds: lower=29, upper=49.

### Specimen Independence Policy

Source groups are the unit of provenance. Files sharing a source group are related observations, not independent specimens. For primary analysis, either one representative spectrum per source group is selected, OR cluster-robust standard errors with source group as the clustering variable are used. Grain-size fractions from the same source are retained for within-source sensitivity analysis but each source is treated as a cluster.

### Unresolved Issues Carried Forward

1. **Grain-size fraction selection:** 3 Aragonite sources (JB-JLB-E59, JB-JLB-E60, JB-JLB-G40) each have 6 grain-size fractions. Whether to use all fractions or select one per source is a research question, not an adjudication question.
2. **Replicate measurement independence:** 17 multi-file replicate groups (83 files) have unresolved independence status. Whether these are spatial subsamples or instrumental replicates is unknown.
3. **Thermal treatment effects:** PC-SJG-024 has 10 natural + 3 thermally treated variants. Treated variants must not be pooled with natural.
4. **Source relationships between c1sg66/c2sg66/c3sg66:** Different PDS classifications (pure_natural vs biogenic_carbonate) and separate archive records prevent merging into a single source group.
5. **Crystal-system confounding:** Extension adds Aragonite spectral diversity but does not resolve the canonical dataset's complete mineral-identity/crystal-system confounding.

### Frozen Decisions

- **Biogenic carbonates are SENSITIVITY_ONLY, not PRIMARY_ELIGIBLE.** This is a scientific judgment, not a technical exclusion.
- **Source groups are the unit of provenance.** File count is never reported as specimen count.
- **Grain-size fractions remain in POP_1_PRIMARY.** Step 5.5 deliberately placed full-D2 fractions in POP_1; Step 5.4.2 established clustering policy.
- **No candidate is EXCLUDED.** All 49 POP_1 candidates have sufficient evidence for a disposition.
- **No spectra are processed.** This step assigns dispositions only.

### Outputs

- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_candidate_adjudication.csv`
- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_population_plan.csv`
- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_biogenic_carbonate_review.csv`
- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_specimen_independence_reconciliation.csv`
- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_scientific_scope_assessment.md`
- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_adjudication_report.md`
- `results/05_exploratory_analysis/phase5_step5_7_scientific_inclusion_adjudication/phase5_step5_7_decision_log.md`

### Recommended Next Action

**Separate extension processing may proceed** for POP_1_PRIMARY (42 candidates) and POP_1_BIOGENIC (7 candidates) as distinct populations. Preprocessing (D2 grid resampling) may begin for all 49 candidates. Scientific analysis must treat POP_1_BIOGENIC as a sensitivity population and assess whether biogenic aragonite spectra differ systematically from mineral-separate aragonite.

Do NOT pool POP_1_BIOGENIC with POP_1_PRIMARY without explicit evidence of interchangeability. Do NOT report file counts as specimen counts. Do NOT claim the extension resolves crystal-system confounding.

### Operational constraints at freeze

- No raw RELAB `.tab` files were modified.
- All 1,017 raw checksums remain unchanged.
- Frozen Phase 2 canonical dataset (454 spectra) unchanged.
- Frozen Phase 3 feature matrix (454 rows) unchanged.
- Frozen Phase 4 provenance mapping (382 groups) unchanged.
- Frozen Step 5.5 inclusion plan (49 POP_1 candidates) unchanged.
- No new spectra processed, interpolated, normalized, or resampled.
- No spectral features engineered.
- No ML/PySR analysis performed.
- No curator correspondence sent.

