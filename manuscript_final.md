# What Can Curated Mineral Spectral Libraries Support? A Provenance-Aware Assessment of Dependence Structures, Metadata Gaps, and Inferential Limits in RELAB

**Target journal:** *Computers & Geosciences* (Elsevier)

**Date:** 2026-10-18

**Manuscript type:** Research article

---

## Authors

Vitaly Andriyko
Independent Researcher, Nanjing, China

**Corresponding author:** Vitaly Andriyko, andriykovit@gmail.com, Nanjing, China.

---

## Abstract

Curated spectral libraries such as RELAB are widely used for mineral identification, yet individual spectra are typically treated as independent observations without accounting for provenance structures, the distinction between reference and specimen-specific chemistry, or the dependence relationships among spectra from the same physical source. We ask a practical question: what can researchers reliably infer from a curated mineral reflectance library when spectral records, provenance, mineral identity, chemistry, and crystallography are incompletely linked? We assembled a provenance-traced dataset of 454 reflectance spectra (320--2550 nm, 5 nm spacing) spanning 18 mineral labels and 382 provenance groups from the RELAB library. A 30-field evidence audit, with metadata recovery from raw file headers, quantified metadata completeness and defined analysis gates. Group-aware cross-validation -- an established practice adapted to the library's dependence structures -- was used to prevent group overlap between training and test folds. Under this design, logistic regression achieved a within-library aggregate out-of-fold macro-F1 of 0.6136 for 16-class mineral classification (309 spectra, 284 groups) versus a dummy baseline of 0.0248, with 10 of 16 classes reaching F1 >= 0.5 and performance positively associated with the number of independent groups per class. The original hypothesis -- that crystal symmetry independently constrains spectral complexity -- cannot be tested with these data because mineral identity and crystal system are completely confounded, no specimen-specific chemistry exists, and specimen independence is unverified. We present a transparent, reproducible assessment framework that documents what the data can and cannot support, and recommend that future spectral libraries adopt provenance tracking, specimen-specific chemistry, and machine-readable metadata to enable defensible inferential analysis.

**Keywords:** mineral spectroscopy, reflectance spectra, provenance-aware validation, group-aware cross-validation, data leakage, spectral library audit, RELAB

---

## 1. Introduction

### 1.1 The inferential problem in curated spectral libraries

Curated spectral libraries -- the RELAB collection (Pieters & Hiroi, 2004), the USGS Digital Spectral Library (Clark et al., 2007), and the ASTER Spectral Library (Baldridge et al., 2009) -- are essential infrastructure for mineral identification and comparative spectroscopy. Each spectrum is associated with a mineral label, and in most cases, reference chemistry and measurement conditions are recorded to varying degrees. However, the metadata in these libraries was designed for human curation and spectral matching, not for the requirements of modern machine-learning workflows, which demand careful attention to sample dependence, metadata completeness, and the distinction between population-level labels and specimen-level properties.

This creates an inferential problem. When individual spectra are treated as independent observations, analyses may conflate measurement replicates and grain-size series with independent specimens, reference chemistry with specimen-specific chemistry, and mineral-level labels with specimen-level properties. The resulting dependence structures and metadata gaps are invisible under standard analysis practices but can fundamentally limit what the data can support.

### 1.2 Dependence structures and data leakage in spectroscopic ML

Machine learning methods have been increasingly applied to mineral identification from spectral data. Hao et al. (2021) reviewed the landscape of ML-based mineral identification across chemistry, microscopy, and spectroscopy. Jahoda et al. (2021) demonstrated multi-method spectroscopy paired with ML for rapid mineral characterisation across multiple libraries. Carey et al. (2015) applied ML to Raman spectroscopy for mineral classification. These studies establish that mineral discrimination from spectral data is feasible under standard validation designs.

A recurring challenge is the treatment of spectra as independent when they are not. Many spectral libraries include grain-size series, measurement replicates, and spectra from the same mineral collected under varying conditions. When dependent spectra are randomly split between training and test sets, information leaks, inflating performance estimates. Guo et al. (2017) identified cross-validation mistakes in spectroscopic classification including splitting at the wrong level of dependence. Király and Tóth (2025) demonstrated that inappropriate handling of repeated measurements introduces data leakage in chemometric modelling. The principle that cross-validation must respect the highest level of dependence is well established in fields such as medical imaging (Varoquaux et al., 2018) and bioinformatics (Altmann et al., 2010).

Group-aware cross-validation -- ensuring that dependent observations are kept within the same fold -- is established practice in these fields. Its application to mineral spectral libraries has not been demonstrated.

### 1.3 Grain-size effects and spectral dependence

Grain size is a well-known control on mineral reflectance spectra (Mustard & Pieters, 1989; Clark, 1999). Spectral libraries that include grain-size series -- such as the USGS library and the 2D hyperspectral mineral library of Fasnacht et al. (2019) -- contain structured dependence: spectra from the same specimen at different grain sizes are not independent observations. This dependence must be accounted for in any validation design.

### 1.4 Research questions and contribution

This study addresses the practical question: **what can researchers reliably infer from a curated mineral reflectance library when spectral records, provenance, mineral identity, chemistry, and crystallography are incompletely linked?**

We illustrate the assessment through four analyses of the RELAB library:

1. **Evidence audit.** What metadata and evidence are available, and how complete are they?
2. **Group-aware classification.** Can mineral identities be discriminated under validation that prevents group overlap?
3. **Support and performance.** How does class-level evidence relate to classification performance?
4. **Inferential limits.** What evidence would be required to test crystal-structure effects on spectra?

Our contribution is a transparent, reproducible assessment of what one curated spectral library can and cannot support for inferential analysis. We do not claim that the assessment methods are themselves novel: group-aware cross-validation is established practice, evidence auditing follows standard data-quality principles, and permutation importance and symbolic regression are standard tools. The contribution is dataset-specific: we demonstrate what becomes visible when these principles are applied systematically to a curated mineral spectral library with full provenance tracking. We provide the assessment framework, the evidence audit, and the explicit separation of inferential levels as a resource that could inform future spectral-library studies.

---

## 2. Data and Methods

### 2.1 Data source

All data were obtained from the RELAB spectral library at Brown University (Pieters & Hiroi, 2004). RELAB is a NASA-supported multiuser spectroscopy facility. The local collection comprised 1,017 raw .tab files spanning 20 mineral directories.

*[RELAB acknowledgement text must conform to RELAB's required attribution and acknowledgement requirements. The specific acknowledgement wording should be verified against RELAB's current data-use policy before submission.]*

### 2.2 Population eligibility and frozen canonical dataset

From the 1,017 raw files, we applied eligibility criteria:

- **Category A** (800 spectra): No quality-control flags; passed all automated checks.
- **Category B** (125 spectra): Eligible with preprocessing (e.g., interpolation from native 10 nm spacing).
- **Category C** (92 files): Excluded -- calibration-file blanks with no spectral content.

The frozen canonical dataset comprised **454 spectra** that fully covered the D2 spectral domain (320--2550 nm) at 5 nm spacing (447 wavelength points). These 454 spectra spanned **18 mineral labels** and **382 provenance groups**. An additional extension dataset of 49 spectra (4 minerals) was processed separately and is not included in the primary analysis.

No spectra were added, removed, or modified after the dataset was frozen.

### 2.3 Provenance hierarchy and dependence structures

We constructed a six-level provenance hierarchy for each spectrum:

1. **File level:** unique .tab filename.
2. **Spectral content level:** MD5 hash of the spectral array.
3. **Sample ID level:** the sample identifier recorded in the RELAB metadata.
4. **Specimen group level:** inferred specimen identity based on sample ID, grain size, and measurement context.
5. **Duplicate relationship level:** documented duplicate or replicate relationships.
6. **Population category level:** Category A or B eligibility.

The analysis unit was the provenance group (not the individual spectrum). The 454 spectra mapped to 382 provenance groups, meaning that 63 groups contained more than one spectrum (accounting for 72 extra spectra beyond one per group). These multi-spectrum groups arose from grain-size series, measurement replicates, and spectra from the same measurement campaign.

**Important caveat:** Provenance groups are provisional units based on available metadata. They are not verified independent specimens. True specimen independence is unconfirmed.

### 2.4 Spectral preprocessing

Preprocessing followed the frozen pipeline established in Phase 2:

- **Domain selection:** D2 (320--2550 nm), highest coverage (454 of 925 eligible spectra, 49.1%).
- **Spacing selection:** 5 nm, as 390 of 925 spectra (42.2%) natively supported this spacing (mean native fraction 0.969).
- **Interpolation:** 64 spectra interpolated from native 10 nm to 5 nm spacing using PCHIP.
- **Wavelength grid:** 447 points (320, 325, ..., 2550 nm).
- **Normalisation:** Unit area under the curve within D2.

### 2.5 Spectral descriptors

Four spectral descriptors were computed for each spectrum:

- **D1 (n_peaks_prom):** Number of peaks with prominence exceeding a threshold.
- **D2 (mean_peak_prominence):** Mean prominence of detected peaks.
- **D3 (first_deriv_rms):** RMS of the first derivative.
- **D4 (second_deriv_rms):** RMS of the second derivative.

All descriptors were computed from the same spectra, creating mathematical coupling: D3 and D4 are not independent of D1 and D2 because all four are derived from the same underlying spectral curve.

### 2.6 Metadata recovery

Measurement metadata were recovered from raw .tab file headers:

| Metadata field | Coverage | Confidence |
|---|---|---|
| Grain size | 86.3% | High |
| Mixture status | 100% | High |
| Measurement geometry | 100% | High |
| Locality hints | 3.1% | Low |

### 2.7 Chemistry and crystallography

All chemistry in the dataset is IDEAL_REFERENCE chemistry -- mineral-level ideal or reference compositions, not specimen-specific measured chemistry. Zero of the 454 spectra have specimen-specific measured chemistry.

Crystal-system assignments were derived from mineral reference databases, not from specimen-specific XRD. RELAB does not routinely conduct XRD or crystallographic analyses (confirmed by curator correspondence with Dr. T. Hiroi). Nine mineral labels have polymorphs with different crystal systems, resulting in 91 spectra with ambiguous polymorph identity.

Mineral identity and crystal system are completely confounded: no mineral label occurs in more than one crystal system. Any apparent crystal-system effect on spectral descriptors is indistinguishable from a mineral-label effect.

### 2.8 Evidence audit framework

We developed a 30-field evidence audit organised across seven domains: identity, specimen, metadata, geometry, calibration, chemistry, and symmetry. Each field was scored for completeness and confidence for every spectrum.

The audit defined analysis gates:

| Gate | Criteria | Status |
|---|---|---|
| GATE_DESC_RESTRICTIVE | Descriptive analysis of available data | OPEN |
| GATE_MATCHED_STRATIFIED | Matched stratified comparison requiring specimen-specific chemistry | CLOSED |
| GATE_SPECIMEN_CHEMISTRY | Analysis requiring specimen-specific measured chemistry | CLOSED |
| GATE_INTERPRETABLE_ML | Inferential ML requiring verified independence | CLOSED |
| GATE_PYSR | Symbolic regression requiring independent descriptors | CLOSED |

Only GATE_DESC_RESTRICTIVE is open. The remaining gates require evidence that does not currently exist in RELAB. These gates are project-specific assessments of what the current data can support, not universal standards.

### 2.8a Configurable evidence-auditor tool

To demonstrate that the evidence framework is not RELAB-specific, we implemented it as a configurable metadata-coverage auditor (Supplementary Code). The tool accepts three inputs: (1) a spectral record table with one row per record, (2) a provenance group table defining group membership, and (3) a field-mapping configuration that maps library-specific column names to the 30 evidence fields. The tool produces a field coverage report, per-record evidence status, analysis gate status, and validation warnings.

The tool was tested exclusively with synthetic data (39 tests covering complete/missing fields, ambiguous metadata, reference vs specimen chemistry, provenance structure, gate computation, and CSV export). No external spectral library has been audited with this tool. The tool demonstrates that the framework can be applied to other libraries through configuration rather than redesign, but this transferability claim rests on the structural generality of the interface, not on empirical demonstration with external data.

Six of the 30 evidence fields are classified as not operationalized: identification_method and geographic_provenance as human-review items (flaggable but not auto-resolvable), and unique_specimen_identifier, unit_cell_parameters, descriptor_uncertainty, and specimen_specific_chemistry as out-of-scope or automated-when-mapped. The tool reports these fields as NOT_ASSESSED when no source column is configured, distinguishing absence of mapping from absence of evidence.

### 2.9 Group-aware classification design

**Classification task.** The classification task used **309 spectra** from **284 provenance groups** spanning **16 mineral classes**. Beryl and Grossular were excluded because each had fewer than 2 provenance groups, making group-aware cross-validation impossible. The distinction between the canonical dataset (454 spectra, 382 groups, 18 minerals) and the classification subset (309 spectra, 284 groups, 16 classes) is maintained throughout.

**Features.** 447 wavelength reflectance values (320--2550 nm, 5 nm spacing).

**Validation design.** Three-fold GroupKFold cross-validation with folds defined by provenance_group_id. This ensures **zero group overlap** across folds: no provenance group's spectra appear in both training and test sets. The 284 groups were split into 94/95/95 across folds 0/1/2.

**What group-aware validation prevents and what it does not prevent.** GroupKFold with folds defined by provenance_group_id prevents leakage from spectra within the same provenance group appearing in both training and test sets. It does not prevent all forms of dependence: spectra from different groups may still be dependent if groups from the same specimen were assigned different IDs; spectra from different specimens of the same mineral from the same locality may be geochemically similar; and all spectra were measured on the same instrument, creating potential instrument-level dependence. Group-aware validation is a necessary safeguard, not a complete solution.

**Preprocessing within CV.** All preprocessing, including StandardScaler, was fitted on training folds only.

**Models.** Dummy classifier (majority-class baseline), logistic regression with L2 regularisation, and random forest (100 trees, max_depth = 10).

**Metrics.** Primary: macro-averaged F1 score, computed from aggregate out-of-fold predictions. Secondary: accuracy, balanced accuracy, weighted F1, and fold-level metrics.

**Sensitivity analyses.** Three analyses: (i) one-per-group subsampling, (ii) regional features (12 band means), and (iii) weak-class exclusion.

### 2.10 Software

All analyses were performed in Python: pandas 3.0.4, numpy 2.5.0, scikit-learn 1.9.1 (Pedregosa et al., 2024). PySR (Cranmer, 2023; PySR, 2024) was used for exploratory symbolic regression. [PySR version was not recorded in the run manifest; the latest available version at the time of analysis is assumed.]

---

## 3. Results

### 3.1 Dataset characteristics

The frozen canonical dataset comprised 454 spectra spanning 18 mineral labels and 382 provenance groups (Table 1, Figure 1a). Minerals varied substantially in representation: Augite dominated with 126 spectra (82 groups), while Beryl and Grossular each contributed a single spectrum. Seven minerals had >= 20 provenance groups, 6 had 4--19 groups, and 5 had fewer than 4 groups.

**Table 1.** Per-mineral summary statistics for the frozen canonical dataset (454 spectra, 18 minerals, 382 provenance groups). Descriptor medians computed from the D2 domain (320--2550 nm, 5 nm spacing, unit-area normalised).

| Mineral | n_spectra | n_groups | Crystal system | D1 median | D2 median | D3 median | D4 median |
|---|---|---|---|---|---|---|---|
| Augite | 126 | 82 | Monoclinic | 2.0 | 0.1488 | 8.50e-04 | 1.90e-05 |
| Calcite | 49 | 43 | Trigonal | 4.0 | 0.0914 | 8.87e-04 | 6.08e-05 |
| Diopside | 56 | 43 | Monoclinic | 3.0 | 0.0709 | 7.10e-04 | 1.95e-05 |
| Dolomite | 47 | 47 | Trigonal | 4.0 | 0.0984 | 8.21e-04 | 3.55e-05 |
| Spinel | 43 | 39 | Isometric | 2.0 | 0.0785 | 6.00e-04 | 2.37e-05 |
| Olivine | 30 | 29 | Orthorhombic | 3.0 | 0.1609 | 1.11e-03 | 4.25e-05 |
| Quartz | 27 | 26 | Trigonal | 4.0 | 0.0601 | 8.03e-04 | 4.21e-05 |
| Apatite | 16 | 14 | Hexagonal | 4.0 | 0.0300 | 6.06e-04 | 3.62e-05 |
| Hypersthene | 16 | 15 | Orthorhombic | 2.0 | 0.1302 | 6.68e-04 | 2.22e-05 |
| Albite | 12 | 12 | Monoclinic* | 3.5 | 0.0308 | 3.95e-04 | 2.31e-05 |
| Enstatite | 9 | 9 | Orthorhombic | 3.0 | 0.1547 | 1.05e-03 | 2.30e-05 |
| Muscovite | 7 | 7 | Monoclinic | 6.0 | 0.1415 | 1.84e-03 | 1.83e-04 |
| Microcline | 4 | 4 | Triclinic+ | 4.0 | 0.0354 | 5.68e-04 | 2.34e-05 |
| Rutile | 4 | 4 | Tetragonal | 4.5 | 0.0572 | 1.20e-03 | 1.03e-04 |
| Andradite | 3 | 3 | Isometric | 2.0 | 0.0171 | 7.00e-04 | 4.68e-05 |
| Orthoclase | 3 | 3 | Monoclinic | 2.0 | 0.0176 | 3.19e-04 | 1.05e-05 |
| Beryl# | 1 | 1 | Hexagonal | 11.0 | 0.2255 | 2.92e-03 | 3.53e-04 |
| Grossular# | 1 | 1 | Isometric | 1.0 | 0.0451 | 1.37e-03 | 4.53e-05 |

\*Albite crystal system listed as monoclinic in the reference database (high-albite C2/m). Low albite is triclinic (P-1); the true crystal system for these specimens is uncertain.
+Microcline is triclinic (C-1, ordered low-temperature K-feldspar).
#Beryl and Grossular were excluded from the classification task.

### 3.2 Mineral classification under group-aware validation

Logistic regression achieved a within-library aggregate out-of-fold macro-F1 of 0.6136 under group-aware three-fold cross-validation (Table 2, Figure 2). The dummy classifier achieved 0.0248. Random forest achieved 0.3785.

**Table 2.** Overall classification metrics (aggregate out-of-fold, 3-fold GroupKFold, 309 spectra, 284 groups, 16 classes).

| Model | Macro-F1 | Balanced accuracy | Accuracy | Weighted F1 |
|---|---|---|---|---|
| Dummy (majority class) | 0.0248 | 0.0530 | 0.1392 | 0.0640 |
| Logistic regression (L2) | 0.6136 | 0.5772 | 0.7670 | 0.7556 |
| Random forest | 0.3785 | 0.3694 | 0.6214 | 0.5962 |

These performance figures are **within-library cross-validation results**. They describe how well mineral labels can be discriminated within this dataset under group-aware validation. They are not upper bounds on external performance, and no external validation was performed.

Ten of 16 classes achieved F1 >= 0.5 (Table 3). Six classes fell below this threshold: Andradite (F1 = 0.00, 3 spectra/3 groups), Orthoclase (F1 = 0.00, 3/3), Microcline (F1 = 0.40, 4/4), and Rutile (F1 = 0.40, 4/4). Enstatite (F1 = 0.80, 3/3) and Olivine (F1 = 0.67, 4/4) exceeded the threshold despite limited groups.

**Table 3.** Per-class classification metrics (aggregate out-of-fold, logistic regression L2, 16 classes, 309 spectra, 284 groups).

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

Logistic regression substantially outperformed random forest (macro-F1 0.6136 vs 0.3785), suggesting that linear decision boundaries suffice at this spectral resolution and sample size, or that the random forest hyperparameters were suboptimal.

### 3.3 Class support and performance

Class performance was positively associated with the number of provenance groups, but the relationship was not deterministic (Figure 3). For the classification subset, the 10 well-supported classes (as defined in Table 3) had a mean F1 of 0.754, while the 6 limited classes had a mean F1 of 0.478.

The most informative comparison is between Enstatite (3 groups, F1 = 0.80) and Andradite (also 3 groups, F1 = 0.00). Small class support is a risk factor for poor performance but not a deterministic predictor.

### 3.4 Sensitivity analyses

Three sensitivity analyses were conducted (Table 4).

**Table 4.** Sensitivity analysis results (logistic regression L2, 3-fold GroupKFold).

| Analysis | n_spectra | n_groups | n_features | n_classes | Macro-F1 |
|---|---|---|---|---|---|
| Full dataset (baseline) | 309 | 284 | 447 | 16 | 0.6136 |
| One-per-group subsampling | 284 | 284 | 447 | 16 | 0.6410 |
| Regional features (12 band means) | 309 | 284 | 12 | 16 | 0.2898 |
| Weak-class exclusion | 309 | 284 | 447 | 16 | 0.6136 |

One-per-group subsampling produced a slightly higher macro-F1 of 0.6410, suggesting that within-group spectra provide some redundant information but do not inflate performance. Regional features (12 band means) produced a substantially lower macro-F1 of 0.2898, indicating that wavelength-level features carry most discriminative information. Weak-class exclusion produced an identical macro-F1 of 0.6136 because all 16 classes have at least 3 provenance groups, so no classes were excluded.

### 3.5 Exploratory analyses: permutation importance and symbolic regression

Two additional analyses were conducted and are reported in brief here; full details are in the Supplementary Material.

**Permutation importance.** Permutation importance was computed for all 447 wavelengths across 12 spectral regions. Eleven of 12 regions had negative mean importance, meaning that permuting wavelengths in those regions improved classification performance. The sole slightly positive region was SWIR_water_1900_2100 (+0.000167), with a magnitude small relative to fold variability (std = 0.00319). These results are inconclusive: the most parsimonious explanation is redundancy among correlated wavelengths, but alternative explanations (systematic noise, sampling variability, scoring limitations) cannot be excluded. This analysis demonstrates that standard permutation importance does not cleanly isolate informative regions in high-dimensional correlated spectral data.

**Symbolic regression.** PySR was used to explore relationships among spectral descriptors D1--D4. The strongest relationships -- D1 + D2 predicting D3 (R² = 0.617) and D4 (R² = 0.701) -- reflect mathematical coupling, as all four descriptors are computed from the same spectrum (Supplementary Figure S1). The discovered equations (D3 ~ 0.00155 * sqrt(D1 * D2); D4 ~ 0.000148 * D1 * D2) are descriptive summaries of coupling structure, not physical relationships. This analysis serves as a cautionary demonstration: symbolic regression recovers coupling structure when descriptors are computed from the same signal.

---

## 4. Discussion

### 4.1 What the classification demonstrates

Mineral discrimination is feasible under group-aware cross-validation that prevents group overlap. Logistic regression achieved a within-library macro-F1 of 0.6136 versus a dummy baseline of 0.0248, with 10 of 16 classes reaching F1 >= 0.5. This demonstrates that mineral-label information is present in the reflectance spectra and can be extracted by a linear classifier even when leakage from dependent spectra is prevented.

This result must be interpreted within its scope. It is a within-library feasibility demonstration, not a claim about external generalisation. Four of 16 classes had F1 < 0.5, and the modest overall performance reflects the conservative validation design, limited sample sizes for some classes, and the inherent difficulty of discriminating minerals with similar compositions. The substantially better performance of logistic regression versus random forest suggests that linear decision boundaries suffice at this resolution, though suboptimal random forest hyperparameters cannot be excluded.

**Positioning relative to prior work.** Jahoda et al. (2021) achieved higher classification accuracy using convolutional neural networks and multi-method spectroscopy on larger, more diverse library sets with standard cross-validation. Our contribution is orthogonal: not "can we classify minerals?" but "what does it take to classify defensibly within a curated library, and what can the library actually support?"

### 4.2 What the evidence audit reveals

The 30-field evidence audit (Section 2.8, Figure 5) reveals that only descriptive analysis is supported by the current data. All gates requiring specimen-specific chemistry, verified independence, or reduced confounding are closed. This is not a failure of the analysis but a finding about the data: the metadata and evidence needed for stronger inferential claims do not exist in the current RELAB collection.

The audit framework is a transparent, reproducible assessment tool. It could be applied to other spectral libraries to assess analysis readiness before ML workflows are applied. We have implemented the framework as a configurable metadata-coverage auditor (Supplementary Code, Section 2.8a) that accepts library-specific field mappings and produces standardised coverage reports. The tool has been tested with synthetic data only; application to an external library is feasible given appropriate metadata but has not been performed.

### 4.3 Why the crystal symmetry hypothesis cannot be tested

The original hypothesis -- that crystal symmetry independently constrains spectral complexity -- requires three conditions that are not met:

1. **Specimen-specific chemistry:** 0% coverage. All chemistry is mineral-level reference composition.
2. **Verified specimen independence:** 0% verified. Provenance groups are provisional.
3. **Reduced mineral/symmetry confounding:** 0% reduction. Mineral identity and crystal system are completely confounded.

Curator correspondence with Dr. T. Hiroi confirmed: no hidden spectra exist beyond the local collection; unlisted polymorphs were likely not measured; RELAB does not routinely conduct XRD. The data needed to test the symmetry hypothesis do not exist in RELAB and cannot be obtained from the curator.

**This is a finding, not a failure.** The explicit criteria in Table 5 provide a roadmap for future data collection: any study claiming to test crystal-structure effects on spectra must demonstrate that all three conditions are met.

**Table 5.** Criteria for testing the crystal symmetry hypothesis.

| Condition | Requirement | Current status |
|---|---|---|
| Specimen-specific chemistry | Measured chemistry for each specimen | Not met (0%) |
| Verified specimen independence | Confirmed distinct specimens for each group | Not met (0%) |
| Reduced mineral/symmetry confounding | At least one polymorph pair with verified specimens | Not met (0%) |

### 4.4 Implications for spectral library practice

We recommend that future spectral library development adopt:

1. **Machine-readable metadata.** Specimen identifiers, chemistry (with evidence level), crystallography (with method), and measurement conditions in machine-readable format.
2. **Provenance tracking.** Every spectrum should have a traceable chain from physical specimen to digital record.
3. **Independence documentation.** Libraries should document dependence structures (grain-size series, replicates, measurement campaigns).
4. **Chemistry evidence levels.** Distinguish ideal/reference, characterised, and specimen-specific chemistry.
5. **Evidence auditing.** Before ML analyses, audit whether available metadata and evidence are sufficient for the intended analysis.

### 4.5 Limitations

1. **Single library.** All data are from RELAB. Results may not generalise.
2. **No specimen-specific chemistry.** Chemical effects cannot be assessed.
3. **Unverified specimen independence.** Provenance groups are provisional.
4. **Complete confounding.** Mineral identity and crystal system are perfectly aligned.
5. **No external validation.** All results are within-library cross-validation.
6. **Limited mineral representation.** Five minerals had fewer than 4 spectra in the canonical dataset.
7. **Descriptor coupling.** All four descriptors are computed from the same spectrum.

---

## 5. Conclusions

This study asked what researchers can reliably infer from a curated mineral reflectance library when spectral records, provenance, mineral identity, chemistry, and crystallography are incompletely linked. Using the RELAB spectral library (454 spectra, 18 mineral labels, 382 provenance groups), we conducted a 30-field evidence audit, recovered measurement metadata from raw file headers, and applied group-aware cross-validation to prevent group overlap.

The main findings are:

1. **Mineral classification is feasible within the library under conservative validation.** Logistic regression achieved within-library macro-F1 = 0.6136 versus dummy baseline 0.0248, with 10 of 16 classes reaching F1 >= 0.5. Performance was positively associated with the number of provenance groups per class, but small support did not deterministically predict poor performance.

2. **The evidence audit reveals that only descriptive analysis is supported.** All gates requiring specimen-specific chemistry, verified independence, or reduced confounding are closed.

3. **The crystal symmetry hypothesis cannot be tested with current data.** Complete mineral/crystal-system confounding, zero specimen-specific chemistry, and unverified specimen independence prevent any test.

4. **Standard methods applied to curated spectral data reveal dependence structures and metadata gaps that are invisible under typical practice.** Group-aware validation, evidence auditing, and provenance tracking make these visible.

The central contribution is a transparent, reproducible assessment of what one curated spectral library can and cannot support, with explicit separation of inferential levels -- prediction, attribution, and generalisation -- that are typically conflated. The assessment framework could inform future spectral-library studies, though its transferability has not been demonstrated.

---

## References

Altmann A, Toloşi L, Sander O, Lengauer T (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics* 26(10):1340--1347.

Baldridge AM, Hook SJ, Groves CI, Rivera G (2009). The ASTER spectral library version 2.0. *Remote Sensing of Environment* 113(4):711--715.

Carey C, Boucher T, Mahadevan S, Dyar MD, et al. (2015). Machine learning tools for mineral recognition and classification from Raman spectroscopy. *Journal of Raman Spectroscopy* 46(10):894--903.

Clark RN (1999). Spectroscopy of rocks and minerals, and principles of spectroscopy. *Manual of Remote Sensing* 3:293--348. Wiley.

Clark RN, Swayze GA, Wise R, Livo KE, Hoefen T, Kokaly R, Sutley SJ (2007). USGS digital spectral library splib06a. *U.S. Geological Survey Data Series* 231.

Cranmer MD (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. *arXiv preprint* arXiv:2305.01582.

Cranmer MD (2024). PySR: High-performance symbolic regression in Python and Julia. https://github.com/MilesCranmer/PySR.

Fasnacht L, Vogt M-L, Renard P, Brunner P (2019). A 2D hyperspectral library of mineral reflectance, from 900 to 2500 nm. *Scientific Data* 6:268.

Guo S, Bocklitz T, Neugebauer U, Popp J (2017). Common mistakes in cross-validating classification models. *Analytical Methods* 9(30):4410--4417.

Hao H-Z, Gu Q, Hu X-M (2021). Research advances and prospective in mineral intelligent identification based on machine learning. *Earth Science* 46(9):3091--3106.

Jahoda P, Drozdovskiy I, Payler SJ, Turchi L, Bessone L, Sauro F (2021). Machine learning for recognizing minerals from multispectral data. *The Analyst* 146(1):184--195.

Király P, Tóth G (2025). Being aware of data leakage and cross-validation scaling in chemometric model validation. *Journal of Chemometrics* 39(4):e70026.

Mustard JF, Pieters CM (1989). Photometric phase functions of common geologic minerals and applications to quantitative analysis of mineral mixture reflectance spectra. *Journal of Geophysical Research: Solid Earth* 94(B10):13619--13634.

Pedregosa F, Varoquaux G, Gramfort A, et al. (2024). scikit-learn: Machine learning in Python. https://scikit-learn.org.

Pieters CM, Hiroi T (2004). RELAB (Reflectance Experiment Laboratory): A NASA multiuser spectroscopy facility. *Lunar and Planetary Science Conference* 1720.

Varoquaux G, Raamana PR, Nelson D, Long P, Thirion B (2018). Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. *NeuroImage* 145:166--179.

---

## Acknowledgements

*[The following acknowledgement must be verified against RELAB's current data-use policy before submission.]*

The authors thank Dr. Takeshi Hiroi (RELAB, Brown University) for correspondence regarding the RELAB spectral library and for confirming the availability of measurement metadata. RELAB is supported by NASA.

---

## Data Availability Statement

The raw RELAB spectral data are publicly available through the RELAB facility at Brown University. The processed canonical dataset, analysis code, and frozen validation splits are available at [repository URL to be determined].

## Software Availability

The configurable evidence-auditor tool is provided as Supplementary Code. It is implemented in Python (pandas >= 2.0) with 39 synthetic tests, a reproducible example, and documentation (README). The tool is released under the BSD 3-Clause license. No restricted RELAB data are included; all tests use synthetic data.

---

## Figure Captions

**Figure 1. Dataset composition and provenance structure.** (a) Grouped bar chart showing n_spectra and n_provenance_groups for each of the 18 minerals in the frozen canonical dataset, sorted by n_provenance_groups ascending. Minerals with n_groups = 1 (Beryl, Grossular) are hatched. (b) Schematic of the provenance hierarchy using Augite as example: 126 spectra map to 82 provenance groups, which decompose into grain-size series and measurement replicates. Provenance groups are provisional units based on metadata; true specimen independence is unverified.

**Figure 2. Group-aware classification performance.** (a) 16 x 16 confusion matrix for logistic regression (L2) under aggregate out-of-fold 3-fold GroupKFold (309 spectra, 284 groups, 16 classes). Rows = true labels, columns = predicted labels. (b) Horizontal bar chart of per-class F1 scores sorted ascending, coloured by evaluation tier. Dashed line at F1 = 0.5.

**Figure 3. Class support vs performance.** Scatter plot of number of provenance groups (log scale) versus per-class F1 score for all 16 classification classes. Points coloured by evaluation tier. Enstatite (3 groups, F1 = 0.80) demonstrates that small support does not deterministically predict poor performance; Andradite (3 groups, F1 = 0.00) demonstrates the opposite.

**Figure 4. Symbolic regression results.** Three-panel figure: (a) D3 predicted vs actual from PySR (n = 311, R² = 0.617); (b) D4 predicted vs actual (R² = 0.701); (c) coefficient stability across folds. All descriptors from the same spectra; relationships reflect mathematical coupling.

**Figure 5. Evidence framework and analysis gates.** Left: Five analysis gates with criteria and current status. Only GATE_DESC_RESTRICTIVE is OPEN. Right: Three necessary conditions for testing the crystal symmetry hypothesis, all NOT MET, leading to the verdict that the hypothesis is untestable with current data. Gates are project-specific assessments, not universal standards.

---

## Declarations

**Funding:** This research received no external funding.

**Competing interests:** The author declares no competing interests.

**Author contributions:** Vitaly Andriyko: Conceptualization, methodology, software, validation, formal analysis, investigation, data curation, visualization, writing – original draft, and writing – review and editing.

**Ethics approval:** Not applicable.

**Consent to participate:** Not applicable.

**Consent for publication:** All authors have read and approved the final manuscript.

---

**End of Manuscript**
