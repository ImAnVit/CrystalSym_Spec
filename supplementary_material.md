# Supplementary Material

**Manuscript:** What Can Curated Mineral Spectral Libraries Support? A Provenance-Aware Assessment of Dependence Structures, Metadata Gaps, and Inferential Limits in RELAB

**Authors:** Vitaly Andriyko

**Corresponding author:** andriykovit@gmail.com

---

## Supplementary Text

### S1. Preprocessing Pipeline Decisions

**Domain selection.** Three candidate domains were evaluated: D1 (350--2500 nm), D2 (320--2550 nm), and D3 (350--2500 nm with thermal extension). D2 was selected because it provided the highest coverage: 454 of 925 eligible spectra (49.1%) fully covered this domain at 5 nm spacing.

**Spacing selection.** A uniform 5 nm spacing was selected because 390 of 925 spectra (42.2%) natively supported this spacing, with a mean native fraction of 0.969. The remaining 64 spectra were interpolated from native 10 nm spacing using PCHIP (piecewise cubic Hermite interpolating polynomial), which preserves monotonicity and does not introduce overshoots.

**Wavelength grid.** The final grid comprised 447 uniformly spaced points from 320 to 2550 nm inclusive (320, 325, 330, ..., 2550 nm).

**Normalization.** Unit area under the curve within D2 was applied to all spectra after interpolation. This removes amplitude variation due to grain size and measurement geometry while preserving band structure.

**Smoothing and splice correction.** No smoothing or splice correction was applied. The preprocessing pipeline was intentionally minimal to avoid introducing artefacts.

**Exclusions.** Two spectra were excluded for insufficient wavelength coverage (F0692: coverage starts at 1000 nm; F0999: coverage starts at 340 nm). No spectra were excluded for reflectance quality control reasons. Zero duplicates were removed; zero spectra had unsorted wavelengths.

**Reproducibility.** The pipeline is deterministic. All 1,017 raw file MD5 checksums were verified. The pipeline produced identical results across independent runs. 38 of 38 validation checks passed.

### S2. Evidence Framework -- Full Specification

The 30-field evidence framework is organised across 9 categories and 4 analysis tiers. Each field is classified as REQUIRED, DESIRED, or NOT_REQUIRED for each tier. A field is REQUIRED when analysis at that tier is not defensible without it; DESIRED when it strengthens the analysis but is not strictly blocking; NOT_REQUIRED when it is irrelevant to that tier.

**Table S_full. Complete 30-field evidence framework.**

| # | Category | Field | Descriptive | Source-group | Specimen chem. | ML/Sym. reg. |
|---|----------|-------|:-----------:|:------------:|:--------------:|:------------:|
| 1 | Mineral identity | mineral_name | REQ | REQ | REQ | REQ |
| 2 | Mineral identity | identity_confidence | REQ | REQ | REQ | REQ |
| 3 | Mineral identity | identification_method | DES | REQ | REQ | REQ |
| 4 | Specimen IDs | specimen_id | REQ | REQ | REQ | REQ |
| 5 | Specimen IDs | source_group_id | REQ | REQ | REQ | REQ |
| 6 | Specimen IDs | unique_specimen_identifier | DES | DES | REQ | REQ |
| 7 | Provenance | source_collection | REQ | REQ | REQ | REQ |
| 8 | Provenance | geographic_provenance | DES | DES | REQ | REQ |
| 9 | Provenance | independence_status | REQ | REQ | REQ | REQ |
| 10 | Provenance | n_verified_independent_specimens | DES | REQ | REQ | REQ |
| 11 | Preparation | grain_size | DES | DES | DES | DES |
| 12 | Preparation | sample_preparation | DES | DES | DES | DES |
| 13 | Preparation | thermal_treatment | DES | DES | REQ | REQ |
| 14 | Preparation | mixture_status | REQ | REQ | REQ | REQ |
| 15 | Measurement | measurement_geometry | DES | REQ | REQ | REQ |
| 16 | Measurement | instrument_configuration | DES | DES | DES | DES |
| 17 | Measurement | calibration_details | DES | DES | DES | DES |
| 18 | Spectral props. | spectral_range | REQ | REQ | REQ | REQ |
| 19 | Spectral props. | sampling_interval | REQ | REQ | REQ | REQ |
| 20 | Spectral props. | preprocessing_history | REQ | REQ | REQ | REQ |
| 21 | Chemistry | specimen_specific_chemistry | NR | DES | REQ | REQ |
| 22 | Chemistry | reference_formula | REQ | REQ | REQ | REQ |
| 23 | Chemistry | chemistry_evidence_level | DES | REQ | REQ | REQ |
| 24 | Crystallography | crystal_system | REQ | REQ | REQ | REQ |
| 25 | Crystallography | space_group | DES | REQ | REQ | REQ |
| 26 | Crystallography | unit_cell_parameters | NR | DES | DES | DES |
| 27 | Crystallography | crystallographic_evidence_source | DES | DES | REQ | REQ |
| 28 | Uncertainty | measurement_uncertainty | DES | DES | DES | DES |
| 29 | Uncertainty | descriptor_uncertainty | NR | NR | DES | DES |
| 30 | Uncertainty | reproducibility_evidence | DES | DES | DES | DES |

REQ = REQUIRED; DES = DESIRED; NR = NOT_REQUIRED.

**Analysis gate criteria.** A gate is OPEN when all REQUIRED fields for that tier achieve >= 80% coverage (fraction of assessed records with status PRESENT). Fields with status NOT_ASSESSED (no source column configured) are excluded from the denominator.

**Evidence status classification.** Each field for each record is classified as:
- **PRESENT:** Non-empty, non-ambiguous value.
- **MISSING:** Empty, null, or recognised sentinel value.
- **AMBIGUOUS:** Flagged as uncertain per library-specific configuration.
- **REFERENCE_DERIVED:** Value sourced from a reference database rather than measured for the specimen.
- **NOT_ASSESSED:** No source column configured for this field.

### S3. Permutation Importance -- Full Results

Permutation importance was computed for all 447 wavelengths, aggregated across 12 spectral regions. For each region, wavelengths were permuted jointly and the change in macro-F1 was recorded across 3 GroupKFold folds.

**Table S1. Regional permutation importance (logistic regression L2, 3-fold GroupKFold, 309 spectra, 284 groups, 16 classes).**

| Spectral region | Wavelength range (nm) | Mean importance | Std |
|---|---|---|---|
| SWIR water | 1900--2100 | +0.000167 | 0.00319 |
| SWIR | 2100--2300 | -0.000311 | 0.00344 |
| VNIR visible | 500--600 | -0.002197 | 0.00493 |
| VNIR red | 600--700 | -0.002233 | 0.00552 |
| VNIR-NIR edge | 700--900 | -0.002971 | 0.00356 |
| SWIR | 1500--1700 | -0.003217 | 0.00463 |
| VNIR visible | 400--500 | -0.003474 | 0.00385 |
| NIR | 1100--1300 | -0.003530 | 0.00394 |
| SWIR | 2300--2500 | -0.003585 | 0.00513 |
| SWIR water | 1300--1500 | -0.004019 | 0.00463 |
| SWIR OH | 1700--1900 | -0.004145 | 0.00411 |
| NIR Fe2+ | 900--1100 | -0.004186 | 0.00361 |

Eleven of 12 regions had negative mean importance, meaning that permuting wavelengths in those regions *improved* classification performance. The sole slightly positive region (SWIR water 1900--2100, +0.000167) had a magnitude small relative to fold variability (std = 0.00319).

**Interpretation.** The most parsimonious explanation for negative importance is redundancy among correlated wavelengths: removing information from a correlated set forces the model to rely on remaining wavelengths that carry overlapping information, which can reduce overfitting to noise. Alternative explanations include systematic noise in certain spectral regions, sampling variability at small fold counts, and limitations of the permutation importance scoring method in high-dimensional correlated feature spaces. These results demonstrate that standard permutation importance does not cleanly isolate informative regions in high-dimensional correlated spectral data.

### S4. Symbolic Regression -- Full Results

PySR (Cranmer, 2023) was used to explore relationships among spectral descriptors D1--D4. All four descriptors are computed from the same spectra, creating mathematical coupling.

**Table S2. Symbolic regression aggregate metrics (n = 311 spectra with complete descriptors).**

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| D3 (first_deriv_rms) | Mean baseline | 3.44e-04 | 4.49e-04 | -0.008 |
| D3 (first_deriv_rms) | Ridge regression | 2.01e-04 | 2.91e-04 | 0.575 |
| D3 (first_deriv_rms) | PySR | 1.92e-04 | 2.77e-04 | 0.617 |
| D4 (second_deriv_rms) | Mean baseline | 3.00e-05 | 4.50e-05 | -0.014 |
| D4 (second_deriv_rms) | Ridge regression | 2.20e-05 | 3.10e-05 | 0.517 |
| D4 (second_deriv_rms) | PySR | 1.70e-05 | 2.40e-05 | 0.701 |

PySR discovered equations (descriptive summaries, not physical relationships):
- D3 ~ 0.00155 * sqrt(D1 * D2)
- D4 ~ 0.000148 * D1 * D2

These relationships reflect mathematical coupling: D3 and D4 are not independent of D1 and D2 because all four are derived from the same underlying spectral curve. The analysis serves as a cautionary demonstration that symbolic regression recovers coupling structure when descriptors are computed from the same signal.

### S5. Configurable Evidence-Auditor Tool

The evidence framework was implemented as a configurable metadata-coverage auditor (Supplementary Code). The tool accepts three inputs:

1. **Record table** (CSV): one row per spectral record, with columns for record_id and mineral_label (required) plus any library-specific metadata columns.
2. **Provenance group table** (CSV): mapping record_id to provenance_group_id.
3. **Field mapping configuration** (Python dict): maps evidence field names to library-specific column names, with optional value transforms.

The tool produces:
- Field coverage summary (30 rows, one per evidence field)
- Per-record evidence status (one row per record, one column per field)
- Analysis gate status (4 rows, one per tier)
- Validation warnings (unmapped fields, missing provenance links, coverage shortfalls)
- Markdown report

**Testing.** 39 synthetic tests cover: complete and missing fields, ambiguous and conflicting metadata, reference vs. specimen chemistry, multiple spectra per group, missing provenance links, gate computation, report generation, NOT_OPERATIONALIZED field classification, and CSV export. All tests use synthetic data; no external library data are used.

**Six fields are classified as not operationalized:** identification_method and geographic_provenance as human-review items; unique_specimen_identifier, unit_cell_parameters, descriptor_uncertainty as out-of-scope; specimen_specific_chemistry as automated-when-mapped. The tool reports these as NOT_ASSESSED when no source column is configured.

---

## Supplementary Tables

**Table S1.** Regional permutation importance. See Supplementary Text S3 for full data and interpretation.

**Table S2.** Symbolic regression aggregate metrics. See Supplementary Text S4 for full data and discovered equations.

**Table S3.** Full per-class classification metrics (aggregate out-of-fold, logistic regression L2, 3-fold GroupKFold, 309 spectra, 284 groups, 16 classes).

| Mineral | n_spectra | n_groups | Precision | Recall | F1 | Tier |
|---|---|---|---|---|---|---|
| Diopside | 54 | 41 | 0.817 | 0.907 | 0.860 | well_supported |
| Calcite | 43 | 38 | 0.917 | 0.767 | 0.835 | well_supported |
| Muscovite | 7 | 7 | 1.000 | 0.714 | 0.833 | well_supported |
| Augite | 38 | 38 | 0.750 | 0.868 | 0.805 | well_supported |
| Dolomite | 41 | 41 | 0.692 | 0.878 | 0.774 | well_supported |
| Quartz | 26 | 25 | 0.750 | 0.808 | 0.778 | well_supported |
| Spinel | 43 | 39 | 0.732 | 0.698 | 0.714 | well_supported |
| Albite | 12 | 12 | 0.643 | 0.750 | 0.692 | well_supported |
| Apatite | 15 | 13 | 0.647 | 0.733 | 0.688 | well_supported |
| Olivine | 4 | 4 | 1.000 | 0.500 | 0.667 | limited |
| Hypersthene | 9 | 9 | 0.800 | 0.444 | 0.571 | well_supported |
| Microcline | 4 | 4 | 1.000 | 0.250 | 0.400 | limited |
| Rutile | 4 | 4 | 1.000 | 0.250 | 0.400 | limited |
| Enstatite | 3 | 3 | 1.000 | 0.667 | 0.800 | limited |
| Andradite | 3 | 3 | 0.000 | 0.000 | 0.000 | limited |
| Orthoclase | 3 | 3 | 0.000 | 0.000 | 0.000 | limited |

**Table S4.** Sensitivity analysis detailed results (logistic regression L2, 3-fold GroupKFold).

| Analysis | n_spectra | n_groups | n_features | n_classes | Macro-F1 | Balanced acc. | Accuracy |
|---|---|---|---|---|---|---|---|
| Full dataset (baseline) | 309 | 284 | 447 | 16 | 0.6136 | 0.5772 | 0.7670 |
| One-per-group subsampling | 284 | 284 | 447 | 16 | 0.6410 | 0.6179 | 0.8028 |
| Regional features (12 band means) | 309 | 284 | 12 | 16 | 0.2898 | 0.2977 | 0.5858 |
| Weak-class exclusion | 309 | 284 | 447 | 16 | 0.6136 | 0.5772 | 0.7670 |

Weak-class exclusion produced identical results to the baseline because all 16 classes have >= 3 provenance groups, so no classes were excluded.

---

## Supplementary Figures

**Figure S1.** Symbolic regression results. (a) D3 (first_deriv_rms) predicted vs. actual from PySR (n = 311, R² = 0.617). (b) D4 (second_deriv_rms) predicted vs. actual (R² = 0.701). (c) Cross-validation fold stability showing R² per fold for both targets. All descriptors computed from the same spectra; relationships reflect mathematical coupling.

**Figure S2.** Regional permutation importance. Horizontal bar chart of mean permutation importance for all 12 spectral regions. Blue bars indicate positive importance; red bars indicate negative importance (permuting improved performance). Error bars show standard deviation across 3 CV folds. 11 of 12 regions have negative mean importance.

---

## Supplementary Code

The configurable evidence-auditor tool is provided as a standalone Python package:

- `evidence_auditor.py` -- Core audit engine (30-field schema, audit logic, report generation)
- `test_evidence_auditor.py` -- 39 synthetic tests
- `example_audit.py` -- Reproducible example with synthetic data
- `requirements.txt` -- Python dependencies (pandas >= 2.0, pytest >= 7.0)
- `LICENSE` -- BSD 3-Clause
- `README.md` -- Usage instructions, input/output schema, and limitations

**Installation:** `pip install -r requirements.txt` (requires Python >= 3.10)

**Quick start:** `python example_audit.py`

**Run tests:** `python -m pytest test_evidence_auditor.py -v`

---

**End of Supplementary Material**
