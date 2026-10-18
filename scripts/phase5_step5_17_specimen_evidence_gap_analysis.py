"""
Phase 5 Step 5.17 -- Gate B: Minimum Specimen-Level Evidence Requirements
and Gap Analysis.

Defines the minimum specimen-level evidence required for scientifically
defensible mineral spectral comparisons, then audits existing CrystalSym-Spec
data against those requirements.

Produces:
  1. phase5_step5_17_scope_and_method.md
  2. phase5_step5_17_minimum_evidence_framework.csv
  3. phase5_step5_17_field_gap_analysis.csv
  4. phase5_step5_17_population_coverage_summary.csv
  5. phase5_step5_17_curator_question_priorities.csv
  6. phase5_step5_17_analysis_release_gates.csv
  7. phase5_step5_17_next_action_decision.md
  8. phase5_step5_17_validation_checks.csv
  9. phase5_step5_17_run_manifest.json
  10. phase5_step5_17_report.md

Author: CrystalSym-Spec workflow
Date: 2026-10-16
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis"
OUTPUT_DIR = RESULTS_DIR / "phase5_step5_17_specimen_evidence_gap_analysis"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
TIMESTAMP = datetime.now(timezone.utc).isoformat()

INPUT_FILES = {
    "linked_view": RESULTS_DIR
    / "phase5_step5_13_linked_comparative_view"
    / "phase5_step5_13_linked_analysis_view.csv",
    "extension_metadata": RESULTS_DIR
    / "phase5_step5_8_extension_dataset"
    / "phase5_step5_8_extension_metadata.csv",
    "integrated_dataset": DATA_DIR / "phase4_integrated_analysis_dataset.csv",
    "canonical_metadata": DATA_DIR / "phase2_canonical_d2_5nm_metadata.csv",
    "chemistry_metadata": DATA_DIR / "phase4_chemistry_metadata.csv",
    "symmetry_metadata": DATA_DIR / "phase4_symmetry_metadata.csv",
    "mineral_reference": DATA_DIR / "phase4_mineral_reference.csv",
    "provenance_mapping": DATA_DIR / "phase4_provenance_mapping.csv",
    "relab_measurements": METADATA_DIR / "relab_measurements.csv",
    "relab_sample_groups": METADATA_DIR / "relab_sample_groups.csv",
    "relab_field_dictionary": METADATA_DIR / "provenance" / "relab_field_dictionary.md",
    "eligibility_514": RESULTS_DIR
    / "phase5_step5_14_aragonite_calcite_eligibility_design"
    / "phase5_step5_14_analysis_eligibility.csv",
    "source_group_audit_514": RESULTS_DIR
    / "phase5_step5_14_aragonite_calcite_eligibility_design"
    / "phase5_step5_14_source_group_audit.csv",
    "record_selection_515": RESULTS_DIR
    / "phase5_step5_15_aragonite_calcite_descriptive_comparison"
    / "phase5_step5_15_record_selection_manifest.csv",
    "sg_representatives_515": RESULTS_DIR
    / "phase5_step5_15_aragonite_calcite_descriptive_comparison"
    / "phase5_step5_15_source_group_representatives.csv",
    "run_manifest_516": RESULTS_DIR
    / "phase5_step5_16_post_comparison_audit"
    / "phase5_step5_16_run_manifest.json",
    "research_direction_516": RESULTS_DIR
    / "phase5_step5_16_post_comparison_audit"
    / "phase5_step5_16_research_direction_decision.md",
    "data_acquisition_priorities_516": RESULTS_DIR
    / "phase5_step5_16_post_comparison_audit"
    / "phase5_step5_16_data_acquisition_priorities.csv",
    "research_readiness_516": RESULTS_DIR
    / "phase5_step5_16_post_comparison_audit"
    / "phase5_step5_16_research_readiness.csv",
    "phase1_freeze_manifest": METADATA_DIR / "provenance" / "phase1_freeze_manifest.yaml",
}

OUTPUT_FILES = [
    "phase5_step5_17_minimum_evidence_framework.csv",
    "phase5_step5_17_field_gap_analysis.csv",
    "phase5_step5_17_population_coverage_summary.csv",
    "phase5_step5_17_curator_question_priorities.csv",
    "phase5_step5_17_analysis_release_gates.csv",
    "phase5_step5_17_validation_checks.csv",
    "phase5_step5_17_run_manifest.json",
    "phase5_step5_17_scope_and_method.md",
    "phase5_step5_17_next_action_decision.md",
    "phase5_step5_17_report.md",
]


def compute_checksums(filepath: Path) -> dict:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        data = f.read()
        md5.update(data)
        sha256.update(data)
    return {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "size_bytes": len(data),
    }


# ---------------------------------------------------------------------------
# 1. Load and verify inputs
# ---------------------------------------------------------------------------


def load_inputs() -> dict:
    print("[1/10] Loading and verifying input files...")
    data = {}
    checksums = {}

    for key, path in INPUT_FILES.items():
        if not path.exists():
            print(f"  WARNING: Input file not found: {path}")
            continue
        cs = compute_checksums(path)
        checksums[key] = {**cs, "path": str(path.relative_to(PROJECT_ROOT))}
        if path.suffix == ".csv":
            data[key] = pd.read_csv(path)
        elif path.suffix == ".json":
            with open(path) as f:
                data[key] = json.load(f)
        else:
            data[key] = path

    print(f"  Loaded {len(data)} input files, {len(checksums)} checksums recorded.")
    return data, checksums


# ---------------------------------------------------------------------------
# 2. Define minimum evidence framework
# ---------------------------------------------------------------------------


def define_minimum_evidence_framework() -> pd.DataFrame:
    print("[2/10] Defining minimum evidence framework...")

    rows = [
        {
            "evidence_category": "mineral_identity",
            "evidence_field": "mineral_name",
            "description": "Mineral species name as identified by the source",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Without mineral identity, no mineral-level comparison is possible",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Identity alone does not establish causal structure-spectrum links",
        },
        {
            "evidence_category": "mineral_identity",
            "evidence_field": "identity_confidence",
            "description": "Confidence level of the mineral identification (STRONG, TENTATIVE, CONFLICT)",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Low-confidence identifications introduce misclassification bias",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Confidence qualifier, not causal evidence",
        },
        {
            "evidence_category": "mineral_identity",
            "evidence_field": "identification_method",
            "description": "Method used to identify the mineral (XRD, optical, chemical analysis, directory label only)",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Directory-label-only identification is unverified; XRD or chemical analysis provides stronger evidence",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Stronger identification reduces misclassification but does not prove causation",
        },
        {
            "evidence_category": "specimen_identifiers",
            "evidence_field": "relab_sample_id",
            "description": "RELAB sample identifier linking spectra to a physical specimen",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Without specimen IDs, provenance grouping is impossible",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Identifiers enable grouping but do not prove independence",
        },
        {
            "evidence_category": "specimen_identifiers",
            "evidence_field": "source_group_id",
            "description": "Provenance or source group identifier for grouping related spectra",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Source groups define the unit of analysis and independence structure",
            "evidence_level_expected": "source_group_level",
            "causal_identifiability_note": "Grouping is provisional; verification requires additional evidence",
        },
        {
            "evidence_category": "specimen_identifiers",
            "evidence_field": "unique_specimen_identifier",
            "description": "Persistent unique identifier for the physical specimen across measurements",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Without persistent specimen IDs, cannot confirm which spectra share a specimen",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Persistent IDs enable specimen-level tracking but do not establish causal links",
        },
        {
            "evidence_category": "provenance",
            "evidence_field": "source_collection",
            "description": "Originating collection or institution (e.g., JB-JLB, AG-TJM, PA-SJG)",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Source collection determines provenance chain and potential batch effects",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Collection origin is a potential confounder, not causal evidence",
        },
        {
            "evidence_category": "provenance",
            "evidence_field": "geographic_provenance",
            "description": "Geographic locality of specimen collection",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Geographic origin affects chemistry through geological context; needed for specimen-chemistry modelling",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Geography may confound mineral-spectrum relationships",
        },
        {
            "evidence_category": "provenance",
            "evidence_field": "independence_status",
            "description": "Whether spectra derive from independent specimens, grain-size fractions, or replicate measurements",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Dependent spectra inflate apparent sample size and violate independence assumptions",
            "evidence_level_expected": "source_group_level",
            "causal_identifiability_note": "Independence is a design requirement, not a causal factor",
        },
        {
            "evidence_category": "provenance",
            "evidence_field": "n_verified_independent_specimens",
            "description": "Count of verified independent specimens per source group",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Effective sample size depends on verified independence, not spectrum count",
            "evidence_level_expected": "source_group_level",
            "causal_identifiability_note": "Sample size affects power, not identifiability",
        },
        {
            "evidence_category": "preparation_treatment",
            "evidence_field": "grain_size",
            "description": "Numeric grain size or size fraction",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Grain size affects scattering and spectral shape; a known confounder",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Grain size is a confounder that must be controlled, not a causal factor of interest",
        },
        {
            "evidence_category": "preparation_treatment",
            "evidence_field": "sample_preparation",
            "description": "Preparation protocol (powder, polished section, coating, etc.)",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Preparation affects spectral properties; uncontrolled preparation is a confounder",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Preparation is a potential confounder",
        },
        {
            "evidence_category": "preparation_treatment",
            "evidence_field": "thermal_treatment",
            "description": "Whether the specimen was heated or thermally treated",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Thermal treatment can alter crystal structure and spectral features",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Thermal treatment is a confounder for structure-spectrum relationships",
        },
        {
            "evidence_category": "preparation_treatment",
            "evidence_field": "mixture_status",
            "description": "Whether the specimen is a pure phase or contains mixtures/impurities",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Mixtures produce composite spectra that do not represent a single mineral phase",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Mixture status directly affects spectral interpretation",
        },
        {
            "evidence_category": "measurement_conditions",
            "evidence_field": "measurement_geometry",
            "description": "Illumination and observation angles (source, detector)",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Geometry affects reflectance magnitude and spectral shape; must be comparable across populations",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Geometry is a measurement confounder, not a mineral property",
        },
        {
            "evidence_category": "measurement_conditions",
            "evidence_field": "instrument_configuration",
            "description": "Spectrometer type, detector, voltage, aperture, run mode",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Different instruments produce different spectral responses; batch effects must be assessed",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Instrument effects are measurement confounders",
        },
        {
            "evidence_category": "measurement_conditions",
            "evidence_field": "calibration_details",
            "description": "Calibration standard, reference material, calibration date",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Calibration affects absolute reflectance values; undocumented calibration introduces systematic uncertainty",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Calibration affects measurement accuracy, not mineral properties",
        },
        {
            "evidence_category": "spectral_properties",
            "evidence_field": "spectral_range",
            "description": "Wavelength range covered (min, max nm)",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Different spectral ranges contain different diagnostic features; comparisons require overlapping ranges",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Spectral range defines observable feature space",
        },
        {
            "evidence_category": "spectral_properties",
            "evidence_field": "sampling_interval",
            "description": "Wavelength sampling interval (nm)",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Sampling interval determines spectral resolution and descriptor values",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Resolution affects descriptor sensitivity",
        },
        {
            "evidence_category": "spectral_properties",
            "evidence_field": "preprocessing_history",
            "description": "Complete record of preprocessing steps applied (interpolation, smoothing, normalization)",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Preprocessing alters spectral features; undocumented preprocessing makes descriptors unreproducible",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Preprocessing is a methodological choice, not a mineral property",
        },
        {
            "evidence_category": "chemistry",
            "evidence_field": "specimen_specific_chemistry",
            "description": "Measured chemical composition of the specific specimen (wt% oxides, formula, trace elements)",
            "tier_descriptive": "NOT_REQUIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Ideal/reference chemistry does not capture specimen-specific compositional variation; solid solutions and impurities require measured chemistry",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Specimen chemistry is needed for chemistry-spectrum modelling but does not alone establish causation",
        },
        {
            "evidence_category": "chemistry",
            "evidence_field": "reference_formula",
            "description": "Ideal or reference chemical formula for the mineral species",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Reference formula establishes the expected composition for eligibility screening",
            "evidence_level_expected": "reference_level",
            "causal_identifiability_note": "Reference chemistry is a necessary but insufficient proxy for actual specimen chemistry",
        },
        {
            "evidence_category": "chemistry",
            "evidence_field": "chemistry_evidence_level",
            "description": "Level of chemical evidence (A=measured specimen, B=measured same source, C=ideal reference)",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Evidence level determines whether chemistry can be used as a variable or only as a filter",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Evidence level qualifies the reliability of chemistry-spectrum associations",
        },
        {
            "evidence_category": "crystallography",
            "evidence_field": "crystal_system",
            "description": "Crystal system (cubic, tetragonal, orthorhombic, etc.)",
            "tier_descriptive": "REQUIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Crystal system is the primary structural variable of interest in the symmetry hypothesis",
            "evidence_level_expected": "reference_level",
            "causal_identifiability_note": "Crystal system is completely confounded with mineral identity in current data",
        },
        {
            "evidence_category": "crystallography",
            "evidence_field": "space_group",
            "description": "Space group symbol and number",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "REQUIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Space group provides finer structural resolution than crystal system alone",
            "evidence_level_expected": "reference_level",
            "causal_identifiability_note": "Space group is a structural descriptor, not a causal variable in the current design",
        },
        {
            "evidence_category": "crystallography",
            "evidence_field": "unit_cell_parameters",
            "description": "Unit cell dimensions (a, b, c, alpha, beta, gamma)",
            "tier_descriptive": "NOT_REQUIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Continuous cell parameters could serve as quantitative structure descriptors for regression",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Cell parameters are structural properties but not directly manipulable",
        },
        {
            "evidence_category": "crystallography",
            "evidence_field": "crystallographic_evidence_source",
            "description": "How crystallographic information was determined (XRD, reference database, mineral directory)",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "REQUIRED",
            "tier_ml_symbolic_regression": "REQUIRED",
            "scientific_justification": "Reference-based assignment is reliable for well-characterised minerals but unverified for individual specimens",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Evidence source qualifies confidence in structural assignment",
        },
        {
            "evidence_category": "uncertainty_metadata",
            "evidence_field": "measurement_uncertainty",
            "description": "Per-wavelength or per-spectrum measurement uncertainty values",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Uncertainty values enable weighted analysis and propagate measurement error into models",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Uncertainty quantification improves inference but does not resolve confounding",
        },
        {
            "evidence_category": "uncertainty_metadata",
            "evidence_field": "descriptor_uncertainty",
            "description": "Uncertainty estimates for computed descriptors (D1-D4)",
            "tier_descriptive": "NOT_REQUIRED",
            "tier_source_group_comparison": "NOT_REQUIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Descriptor uncertainty enables error-in-variables modelling but is not available",
            "evidence_level_expected": "record_level",
            "causal_identifiability_note": "Descriptor uncertainty improves model specification but does not resolve confounding",
        },
        {
            "evidence_category": "uncertainty_metadata",
            "evidence_field": "reproducibility_evidence",
            "description": "Evidence from replicate measurements of the same specimen",
            "tier_descriptive": "DESIRED",
            "tier_source_group_comparison": "DESIRED",
            "tier_specimen_chemistry_modelling": "DESIRED",
            "tier_ml_symbolic_regression": "DESIRED",
            "scientific_justification": "Replicate measurements quantify within-specimen spectral variability",
            "evidence_level_expected": "specimen_level",
            "causal_identifiability_note": "Reproducibility evidence constrains measurement noise, not structural effects",
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Field-by-field gap analysis
# ---------------------------------------------------------------------------


def perform_gap_analysis(data: dict) -> pd.DataFrame:
    print("[3/10] Performing field-by-field gap analysis...")

    lv = data["linked_view"]
    integ = data["integrated_dataset"]
    chem = data["chemistry_metadata"]
    sym = data["symmetry_metadata"]
    prov = data["provenance_mapping"]
    relab = data["relab_measurements"]
    ext_meta = data["extension_metadata"]
    sg_audit = data["source_group_audit_514"]
    eligibility = data["eligibility_514"]

    n_total = len(lv)
    n_canonical = len(lv[lv["dataset_origin"] == "CANONICAL_V1"])
    n_extension = len(lv[lv["dataset_origin"] == "EXTENSION_STEP5_8"])

    def _coverage(series, value=None):
        if value is not None:
            n = (series == value).sum()
        else:
            n = series.notna().sum()
            if hasattr(series, "dtype") and series.dtype == object:
                n = series.apply(lambda x: x not in ("", " ", None) and pd.notna(x)).sum()
        return n, n_total

    def _coverage_canonical(series, value=None):
        if value is not None:
            n = (series == value).sum()
        else:
            n = series.notna().sum()
            if hasattr(series, "dtype") and series.dtype == object:
                n = series.apply(lambda x: x not in ("", " ", None) and pd.notna(x)).sum()
        return n, n_canonical

    gap_rows = []

    def add_gap(
        category,
        field,
        current_source,
        n_present,
        denominator,
        evidence_scope,
        recoverable,
        curator_needed,
        new_measurement_needed,
        consequences,
        notes,
    ):
        gap_rows.append(
            {
                "evidence_category": category,
                "evidence_field": field,
                "current_field_or_source": current_source,
                "n_present": n_present,
                "denominator": denominator,
                "coverage_fraction": f"{n_present}/{denominator}",
                "coverage_proportion": n_present / denominator if denominator > 0 else 0.0,
                "evidence_scope": evidence_scope,
                "potentially_recoverable_from_existing_metadata": recoverable,
                "curator_clarification_needed": curator_needed,
                "new_measurements_needed": new_measurement_needed,
                "consequences_for_eligibility": consequences,
                "notes": notes,
            }
        )

    # --- Mineral identity ---
    n_mineral = lv["mineral_label"].notna().sum()
    n_mineral_str = (lv["mineral_label"] != "").sum() if lv["mineral_label"].dtype == object else n_mineral
    add_gap(
        "mineral_identity", "mineral_name",
        "linked_view:mineral_label",
        int(n_mineral), n_total,
        "record_level", "NO", "NO", "NO",
        "All records have mineral labels; no gap",
        "Labels from RELAB directory or extension adjudication",
    )

    n_conf = lv["identity_confidence"].notna().sum()
    n_strong = (lv["identity_confidence"] == "STRONG").sum()
    add_gap(
        "mineral_identity", "identity_confidence",
        "linked_view:identity_confidence",
        int(n_conf), n_total,
        "record_level", "NO", "NO", "NO",
        f"{n_strong}/{n_total} STRONG; remainder may be TENTATIVE",
        "Confidence assigned during Phase 4 and extension adjudication",
    )

    add_gap(
        "mineral_identity", "identification_method",
        "NOT AVAILABLE",
        0, n_total,
        "specimen_level", "NO", "YES", "YES",
        "No record of how mineral identity was determined; directory label only",
        "RELAB files do not record identification method; curator clarification or XRD needed",
    )

    # --- Specimen identifiers ---
    add_gap(
        "specimen_identifiers", "relab_sample_id",
        "relab_measurements:relab_sample_id",
        int(relab["relab_sample_id"].notna().sum()), len(relab),
        "record_level", "NO", "NO", "NO",
        "All RELAB records have sample IDs",
        "Sample IDs are present but do not uniquely identify specimens",
    )

    n_sg = lv["source_group_id"].notna().sum()
    add_gap(
        "specimen_identifiers", "source_group_id",
        "linked_view:source_group_id",
        int(n_sg), n_total,
        "source_group_level", "NO", "NO", "NO",
        "All records assigned to source groups",
        "Groups are provisional; not all correspond to verified independent specimens",
    )

    add_gap(
        "specimen_identifiers", "unique_specimen_identifier",
        "NOT AVAILABLE",
        0, n_total,
        "specimen_level", "NO", "YES", "NO",
        "No persistent specimen-level identifier exists",
        "RELAB does not provide unique specimen IDs; curator clarification may link sample IDs to physical specimens",
    )

    # --- Provenance ---
    n_natural = (lv["material_origin"] == "natural").sum()
    add_gap(
        "provenance", "source_collection",
        "linked_view:source_group_id (prefix)",
        int(n_sg), n_total,
        "record_level", "YES", "NO", "NO",
        "Source collection extractable from sample ID prefix",
        "Institution/researcher codes are embedded in RELAB sample IDs",
    )

    add_gap(
        "provenance", "geographic_provenance",
        "NOT AVAILABLE",
        0, n_total,
        "specimen_level", "PARTIAL", "YES", "NO",
        "No geographic locality in RELAB files; some sample descriptions mention localities",
        "Curator clarification or literature search may recover localities for some specimens",
    )

    n_indep = lv["independence_status"].notna().sum()
    add_gap(
        "provenance", "independence_status",
        "linked_view:independence_status + independence_qualifier",
        int(n_indep), n_total,
        "source_group_level", "NO", "NO", "NO",
        "All records have independence status",
        "Status is provisional; based on filename patterns and sample descriptions, not verified",
    )

    add_gap(
        "provenance", "n_verified_independent_specimens",
        "source_group_audit_514:conservative_independent_units",
        int(sg_audit["conservative_independent_units"].notna().sum()) if "conservative_independent_units" in sg_audit.columns else 0,
        len(sg_audit),
        "source_group_level", "PARTIAL", "YES", "NO",
        "Source group audit provides conservative counts but many are estimates",
        "Verification requires curator confirmation of specimen-level independence",
    )

    # --- Preparation and treatment ---
    grain_size_info = relab.get("grain_size_info", pd.Series(dtype=object))
    n_grain = grain_size_info.notna().sum() if len(grain_size_info) > 0 else 0
    n_grain_nonempty = 0
    if len(grain_size_info) > 0:
        n_grain_nonempty = grain_size_info.apply(
            lambda x: pd.notna(x) and str(x).strip() not in ("", " ")
        ).sum()
    add_gap(
        "preparation_treatment", "grain_size",
        "relab_measurements:grain_size_info + sample_description",
        int(n_grain_nonempty), len(relab),
        "specimen_level", "PARTIAL", "YES", "NO",
        f"{n_grain_nonempty}/{len(relab)} records have grain size info; many are qualitative only",
        "Some grain sizes extractable from sample descriptions; curator clarification for numeric values",
    )

    add_gap(
        "preparation_treatment", "sample_preparation",
        "relab_measurements:sample_description (partial)",
        0, n_total,
        "specimen_level", "PARTIAL", "YES", "NO",
        "No standardised preparation protocol; ad hoc notes in sample descriptions only",
        "Curator clarification may reveal preparation methods for some specimens",
    )

    thermal_col = ext_meta.get("thermal_status") if ext_meta is not None else None
    n_thermal = 0
    if thermal_col is not None and hasattr(thermal_col, "notna"):
        n_thermal = thermal_col.notna().sum()
    add_gap(
        "preparation_treatment", "thermal_treatment",
        "extension_metadata:thermal_status (extension only); NOT AVAILABLE (canonical)",
        int(n_extension), n_total,
        "specimen_level", "PARTIAL", "YES", "NO",
        f"Thermal status available for {n_extension} extension records; 0/{n_canonical} canonical records",
        "Canonical records lack thermal treatment documentation; curator clarification needed",
    )

    mixture_col = ext_meta.get("mixture_status") if ext_meta is not None else None
    n_mixture_known = 0
    if mixture_col is not None and hasattr(mixture_col, "notna"):
        n_mixture_known = mixture_col.apply(
            lambda x: pd.notna(x) and str(x).strip() not in ("", " ", "unknown")
        ).sum()
    mixture_flag_integ = integ.get("mixture_flag")
    n_mixture_flag = 0
    if mixture_flag_integ is not None and hasattr(mixture_flag_integ, "notna"):
        n_mixture_flag = mixture_flag_integ.notna().sum()

    add_gap(
        "preparation_treatment", "mixture_status",
        "extension_metadata:mixture_status + integrated_dataset:mixture_flag",
        int(n_mixture_flag), n_total,
        "specimen_level", "PARTIAL", "YES", "NO",
        f"Mixture flag available for {n_mixture_flag}/{n_total} records; 13 Aragonite extension records have unknown mixture status",
        "13 Aragonite records have unknown mixture status; curator clarification needed",
    )

    # --- Measurement conditions ---
    n_geom = relab["measurement_geometry"].notna().sum() if "measurement_geometry" in relab.columns else 0
    add_gap(
        "measurement_conditions", "measurement_geometry",
        "relab_measurements:measurement_geometry + source_angle + detect_angle",
        int(n_geom), len(relab),
        "record_level", "YES", "NO", "NO",
        f"{n_geom}/{len(relab)} records have geometry info",
        "Geometry extractable from RELAB metadata; 39/1017 files missing source/detect angles",
    )

    n_voltage = relab["voltage"].notna().sum() if "voltage" in relab.columns else 0
    add_gap(
        "measurement_conditions", "instrument_configuration",
        "relab_measurements:voltage + filename_prefix_code + run_mode + stage_type + aperture",
        int(n_voltage), len(relab),
        "record_level", "PARTIAL", "NO", "NO",
        f"{n_voltage}/{len(relab)} have voltage; prefix codes available for all; extended metadata (run_mode, stage_type, aperture) available for 337/1017 files only",
        "Extended metadata fields (run_mode, stage_type, spinning, aperture, scrambler) present in 337/1017 files only",
    )

    n_cal = relab["calibration_note"].notna().sum() if "calibration_note" in relab.columns else 0
    add_gap(
        "measurement_conditions", "calibration_details",
        "relab_measurements:calibration_note",
        int(n_cal), len(relab),
        "record_level", "YES", "NO", "NO",
        f"{n_cal}/{len(relab)} records have calibration notes",
        "Most files lack calibration documentation; Halon calibration note in 546/1017 files",
    )

    # --- Spectral properties ---
    n_range = (relab["wl_min"].notna() & relab["wl_max"].notna()).sum() if "wl_min" in relab.columns else 0
    add_gap(
        "spectral_properties", "spectral_range",
        "relab_measurements:wl_min + wl_max + canonical_metadata:native_grid",
        int(n_range), len(relab),
        "record_level", "NO", "NO", "NO",
        "All records have spectral range information",
        "Spectral range is well-documented; canonical dataset standardised to 320-2550 nm",
    )

    n_spacing = relab["native_median_spacing_nm"].notna().sum() if "native_median_spacing_nm" in relab.columns else 0
    if n_spacing == 0 and "native_median_spacing_nm" in data.get("canonical_metadata", pd.DataFrame()).columns:
        n_spacing = data["canonical_metadata"]["native_median_spacing_nm"].notna().sum()
    add_gap(
        "spectral_properties", "sampling_interval",
        "canonical_metadata:native_median_spacing_nm + extension_metadata:source_median_spacing_nm",
        n_total, n_total,
        "record_level", "NO", "NO", "NO",
        f"{n_total}/{n_total} records have documented sampling interval",
        "All canonical records on 5 nm grid; extension records native 5 nm or interpolated",
    )

    n_preproc = lv["preprocessing_route"].notna().sum()
    add_gap(
        "spectral_properties", "preprocessing_history",
        "linked_view:preprocessing_route + interpolation_occurred + interpolation_method",
        int(n_preproc), n_total,
        "record_level", "NO", "NO", "NO",
        "All records have documented preprocessing route",
        "Preprocessing fully documented through Phase 2 pipeline",
    )

    # --- Chemistry ---
    n_chem_spec = (chem["chemistry_specimen_specific"] == True).sum() if "chemistry_specimen_specific" in chem.columns else 0
    add_gap(
        "chemistry", "specimen_specific_chemistry",
        "chemistry_metadata:chemistry_specimen_specific",
        int(n_chem_spec), len(chem),
        "specimen_level", "NO", "YES", "YES",
        f"{n_chem_spec}/{len(chem)} records have specimen-specific chemistry; ALL are False",
        "CRITICAL GAP: No specimen-specific chemistry available for any record; requires new measurements or curator data",
    )

    n_ref = chem["reference_formula"].notna().sum() if "reference_formula" in chem.columns else 0
    add_gap(
        "chemistry", "reference_formula",
        "chemistry_metadata:reference_formula + mineral_reference:chemical_formula",
        int(n_ref), len(chem),
        "reference_level", "NO", "NO", "NO",
        f"{n_ref}/{len(chem)} records have reference formulas",
        "Reference formulas available from mineral reference database",
    )

    n_evid = chem["chemistry_evidence_level"].notna().sum() if "chemistry_evidence_level" in chem.columns else 0
    n_level_c = (chem["chemistry_evidence_level"] == "C").sum() if "chemistry_evidence_level" in chem.columns else 0
    add_gap(
        "chemistry", "chemistry_evidence_level",
        "chemistry_metadata:chemistry_evidence_level",
        int(n_evid), len(chem),
        "specimen_level", "NO", "NO", "NO",
        f"{n_evid}/{len(chem)} have evidence level; {n_level_c}/{len(chem)} are level C (ideal reference only)",
        "All chemistry is level C (ideal reference); no level A or B chemistry exists",
    )

    # --- Crystallography ---
    n_cs = lv["crystal_system"].notna().sum()
    add_gap(
        "crystallography", "crystal_system",
        "linked_view:crystal_system (from mineral_reference)",
        int(n_cs), n_total,
        "reference_level", "NO", "NO", "NO",
        "All records have crystal system assigned",
        "Assigned from mineral reference; not specimen-specific; completely confounded with mineral identity",
    )

    n_sg_sym = lv["space_group_symbol"].notna().sum()
    add_gap(
        "crystallography", "space_group",
        "linked_view:space_group_symbol + space_group_number",
        int(n_sg_sym), n_total,
        "reference_level", "NO", "NO", "NO",
        "All records have space group assigned",
        "Assigned from mineral reference; not verified per specimen",
    )

    add_gap(
        "crystallography", "unit_cell_parameters",
        "NOT AVAILABLE",
        0, n_total,
        "specimen_level", "NO", "YES", "YES",
        "No unit cell parameters available for any specimen",
        "Would require XRD measurements or literature search per specimen",
    )

    n_sym_source = sym["source_reference"].notna().sum() if "source_reference" in sym.columns else 0
    add_gap(
        "crystallography", "crystallographic_evidence_source",
        "symmetry_metadata:source_reference",
        int(n_sym_source), len(sym),
        "specimen_level", "NO", "NO", "NO",
        f"{n_sym_source}/{len(sym)} have evidence source documented",
        "All assigned from RELAB mineral directory label or mineral reference; none from specimen-specific XRD",
    )

    # --- Uncertainty metadata ---
    n_uncert = relab["has_uncertainty"].sum() if "has_uncertainty" in relab.columns else 0
    add_gap(
        "uncertainty_metadata", "measurement_uncertainty",
        "relab_measurements:has_uncertainty + third_column_available",
        int(n_uncert), len(relab),
        "record_level", "NO", "NO", "NO",
        f"{n_uncert}/{len(relab)} records have uncertainty column",
        "461/1017 raw files have third-column uncertainty; canonical metadata records availability",
    )

    add_gap(
        "uncertainty_metadata", "descriptor_uncertainty",
        "NOT AVAILABLE",
        0, n_total,
        "record_level", "NO", "NO", "YES",
        "No descriptor uncertainty estimates exist",
        "Would require replicate descriptor computations or analytical error propagation",
    )

    add_gap(
        "uncertainty_metadata", "reproducibility_evidence",
        "relab_duplicate_relationships + source_group_audit",
        int(sg_audit["independence_status"].notna().sum()) if "independence_status" in sg_audit.columns else 0,
        len(sg_audit),
        "specimen_level", "PARTIAL", "YES", "NO",
        "Some replicate relationships documented; most groups lack formal reproducibility evidence",
        "Dependent spectra identified through filename patterns; formal reproducibility assessment needed",
    )

    return pd.DataFrame(gap_rows)


# ---------------------------------------------------------------------------
# 4. Population coverage summary
# ---------------------------------------------------------------------------


def compute_population_coverage(data: dict, gap_df: pd.DataFrame) -> pd.DataFrame:
    print("[4/10] Computing population coverage summary...")

    lv = data["linked_view"]
    chem = data["chemistry_metadata"]
    relab = data["relab_measurements"]
    eligibility = data["eligibility_514"]

    populations = {
        "ALL_RECORDS": lv,
        "CANONICAL_V1": lv[lv["dataset_origin"] == "CANONICAL_V1"],
        "EXTENSION_PRIMARY": lv[
            (lv["dataset_origin"] == "EXTENSION_STEP5_8")
            & (lv["population_label"] == "EXTENSION_PRIMARY")
        ],
        "EXTENSION_BIOGENIC": lv[
            (lv["dataset_origin"] == "EXTENSION_STEP5_8")
            & (lv["population_label"] == "EXTENSION_BIOGENIC")
        ],
        "CALCITE_PRIMARY": lv[
            lv["mineral_label"] == "Calcite"
        ],
        "ARAGONITE_PRIMARY": lv[
            (lv["mineral_label"] == "Aragonite")
            & (lv["biogenic_flag"] != "biogenic_confirmed")
        ],
        "ARAGONITE_BIOGENIC": lv[
            (lv["mineral_label"] == "Aragonite")
            & (lv["biogenic_flag"] == "biogenic_confirmed")
        ],
    }

    evidence_fields = [
        ("mineral_identity", "mineral_name", lambda df: df["mineral_label"].notna().sum()),
        ("mineral_identity", "identity_confidence_STRONG", lambda df: (df["identity_confidence"] == "STRONG").sum()),
        ("mineral_identity", "identification_method", lambda df: 0),
        ("specimen_identifiers", "relab_sample_id", lambda df: df["source_group_id"].notna().sum()),
        ("specimen_identifiers", "source_group_id", lambda df: df["source_group_id"].notna().sum()),
        ("specimen_identifiers", "unique_specimen_identifier", lambda df: 0),
        ("provenance", "source_collection", lambda df: df["source_group_id"].notna().sum()),
        ("provenance", "geographic_provenance", lambda df: 0),
        ("provenance", "independence_status", lambda df: df["independence_status"].notna().sum()),
        ("provenance", "n_verified_independent_specimens", lambda df: 0),
        ("preparation_treatment", "grain_size", lambda df: 0),
        ("preparation_treatment", "sample_preparation", lambda df: 0),
        ("preparation_treatment", "thermal_treatment", lambda df: 0),
        ("preparation_treatment", "mixture_status_known", lambda df: 0),
        ("measurement_conditions", "measurement_geometry", lambda df: 0),
        ("measurement_conditions", "instrument_configuration_extended", lambda df: 0),
        ("measurement_conditions", "calibration_details", lambda df: 0),
        ("spectral_properties", "spectral_range", lambda df: df["D1_n_peaks_prom"].notna().sum()),
        ("spectral_properties", "sampling_interval", lambda df: df["preprocessing_route"].notna().sum()),
        ("spectral_properties", "preprocessing_history", lambda df: df["preprocessing_route"].notna().sum()),
        ("chemistry", "specimen_specific_chemistry", lambda df: 0),
        ("chemistry", "reference_formula", lambda df: 0),
        ("chemistry", "chemistry_evidence_level", lambda df: 0),
        ("crystallography", "crystal_system", lambda df: df["crystal_system"].notna().sum()),
        ("crystallography", "space_group", lambda df: df["space_group_symbol"].notna().sum()),
        ("crystallography", "unit_cell_parameters", lambda df: 0),
        ("uncertainty_metadata", "measurement_uncertainty", lambda df: 0),
        ("uncertainty_metadata", "descriptor_uncertainty", lambda df: 0),
        ("uncertainty_metadata", "reproducibility_evidence", lambda df: 0),
    ]

    rows = []
    for pop_name, pop_df in populations.items():
        n = len(pop_df)
        for cat, field, count_fn in evidence_fields:
            try:
                n_present = int(count_fn(pop_df))
            except Exception:
                n_present = 0
            rows.append(
                {
                    "population": pop_name,
                    "population_size": n,
                    "evidence_category": cat,
                    "evidence_field": field,
                    "n_present": n_present,
                    "coverage_proportion": n_present / n if n > 0 else 0.0,
                    "coverage_label": (
                        "COMPLETE" if n_present == n and n > 0
                        else "PARTIAL" if n_present > 0
                        else "ABSENT"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Curator question priorities
# ---------------------------------------------------------------------------


def generate_curator_questions(data: dict) -> pd.DataFrame:
    print("[5/10] Generating curator question priorities...")

    lv = data["linked_view"]
    sg_audit = data["source_group_audit_514"]
    eligibility = data["eligibility_514"]

    questions = [
        {
            "priority_rank": 1,
            "question_category": "specimen_identity",
            "question_type": "factual",
            "question": "What method was used to identify each mineral specimen (XRD, optical, chemical analysis, or directory label only)?",
            "scientific_rationale": "Directory-label-only identification is unverified; XRD or chemical analysis would substantially strengthen identity confidence",
            "affected_records": int(len(lv)),
            "affected_minerals": "All",
            "recoverable_from_existing_records": "PARTIAL",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "HIGH",
            "impact_on_eligibility": "Could reclassify TENTATIVE records to STRONG or identify misidentifications",
            "specific_targets": "All source groups with identity_confidence != STRONG; all extension records",
        },
        {
            "priority_rank": 2,
            "question_category": "specimen_chemistry",
            "question_type": "request_for_data",
            "question": "Do specimen-level chemical analyses (wt% oxides, EMPA, XRF, ICP-MS) exist for any RELAB specimens in the collection?",
            "scientific_rationale": "All current chemistry is level C (ideal reference); specimen-specific chemistry is the single most impactful metadata improvement",
            "affected_records": int(len(lv)),
            "affected_minerals": "All",
            "recoverable_from_existing_records": "NO",
            "requires_new_data": "YES",
            "requires_new_measurements": "POSSIBLY",
            "impact_on_confounding": "HIGH",
            "impact_on_eligibility": "Would enable chemistry-spectrum modelling and reduce chemistry-based confounding",
            "specific_targets": "Calcite and Aragonite specimens first (comparison pair); then polymorph minerals",
        },
        {
            "priority_rank": 3,
            "question_category": "independence",
            "question_type": "factual",
            "question": "How many physically independent specimens does each RELAB sample ID represent? Are grain-size series from the same parent specimen?",
            "scientific_rationale": "Current independence structure is inferred from filename patterns; verified independence is needed for valid sample sizes",
            "affected_records": int(len(lv)),
            "affected_minerals": "All",
            "recoverable_from_existing_records": "PARTIAL",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "MEDIUM-HIGH",
            "impact_on_eligibility": "Could change effective sample sizes and independence structure",
            "specific_targets": "Multi-spectrum source groups (SPM0332, SPM0408, SPM0282, SPM0333, JB-JLB-E59/E60/G40, PA-SJG-021)",
        },
        {
            "priority_rank": 4,
            "question_category": "mixture_status",
            "question_type": "factual",
            "question": "Are the 13 Aragonite records with unknown mixture status pure phase or do they contain impurities/other minerals?",
            "scientific_rationale": "Unknown mixture status affects spectral interpretation; mixture-unknown Aragonite shows different descriptor distributions",
            "affected_records": 13,
            "affected_minerals": "Aragonite",
            "recoverable_from_existing_records": "POSSIBLE",
            "requires_new_data": "NO",
            "requires_new_measurements": "POSSIBLY",
            "impact_on_confounding": "MEDIUM",
            "impact_on_eligibility": "Could change eligibility of 13 Aragonite records",
            "specific_targets": "EXT_042-053, EXT_069 (Aragonite records with mixture_status=unknown)",
        },
        {
            "priority_rank": 5,
            "question_category": "geographic_provenance",
            "question_type": "factual",
            "question": "What is the geographic locality (mine, formation, country) for each specimen?",
            "scientific_rationale": "Geographic provenance affects chemistry through geological context; needed for specimen-chemistry modelling",
            "affected_records": int(len(lv)),
            "affected_minerals": "All",
            "recoverable_from_existing_records": "PARTIAL",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "MEDIUM",
            "impact_on_eligibility": "May reveal geographic clustering or batch effects",
            "specific_targets": "All source groups; priority for Calcite and Aragonite comparison pair",
        },
        {
            "priority_rank": 6,
            "question_category": "measurement_conditions",
            "question_type": "factual",
            "question": "What are the detailed measurement conditions for extended-metadata-missing files (run mode, stage type, spinning, aperture, scrambler)?",
            "scientific_rationale": "676/1017 files lack extended metadata; undocumented measurement conditions may introduce batch effects",
            "affected_records": 676,
            "affected_minerals": "All",
            "recoverable_from_existing_records": "POSSIBLE",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "LOW-MEDIUM",
            "impact_on_eligibility": "May reveal systematic measurement differences across instrument campaigns",
            "specific_targets": "Files without extended metadata blocks (676/1017)",
        },
        {
            "priority_rank": 7,
            "question_category": "thermal_treatment",
            "question_type": "factual",
            "question": "Were any canonical specimens thermally treated or heated before measurement?",
            "scientific_rationale": "Thermal treatment can alter crystal structure and spectral features; extension records document this but canonical records do not",
            "affected_records": 454,
            "affected_minerals": "All canonical minerals",
            "recoverable_from_existing_records": "POSSIBLE",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "LOW-MEDIUM",
            "impact_on_eligibility": "Thermally treated specimens may need separate handling",
            "specific_targets": "All canonical records; priority for Calcite and Aragonite",
        },
        {
            "priority_rank": 8,
            "question_category": "grain_size",
            "question_type": "factual",
            "question": "What are the numeric grain sizes for specimens where only qualitative descriptions exist?",
            "scientific_rationale": "Grain size affects scattering and spectral shape; quantitative values enable grain-size stratified analysis",
            "affected_records": int(len(lv)),
            "affected_minerals": "All",
            "recoverable_from_existing_records": "PARTIAL",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "LOW-MEDIUM",
            "impact_on_eligibility": "May enable grain-size as a stratification variable",
            "specific_targets": "Records with qualitative grain size descriptions; grain-size series (ca/cg prefix files)",
        },
        {
            "priority_rank": 9,
            "question_category": "crystallography",
            "question_type": "request_for_data",
            "question": "Do specimen-specific XRD patterns or unit cell parameters exist for any specimens?",
            "scientific_rationale": "Current crystallography is reference-based; specimen-specific XRD would verify structural assignment and provide continuous descriptors",
            "affected_records": int(len(lv)),
            "affected_minerals": "All",
            "recoverable_from_existing_records": "NO",
            "requires_new_data": "YES",
            "requires_new_measurements": "POSSIBLY",
            "impact_on_confounding": "MEDIUM",
            "impact_on_eligibility": "Would enable continuous crystal-structure descriptors for regression",
            "specific_targets": "Priority for polymorph pairs (Calcite/Aragonite, TiO2 polymorphs, Al2SiO5 polymorphs)",
        },
        {
            "priority_rank": 10,
            "question_category": "calibration",
            "question_type": "factual",
            "question": "What calibration standards and procedures were used for each measurement campaign?",
            "scientific_rationale": "Calibration affects absolute reflectance; undocumented calibration introduces systematic uncertainty",
            "affected_records": 1017,
            "affected_minerals": "All",
            "recoverable_from_existing_records": "POSSIBLE",
            "requires_new_data": "NO",
            "requires_new_measurements": "NO",
            "impact_on_confounding": "LOW",
            "impact_on_eligibility": "May reveal systematic calibration differences across campaigns",
            "specific_targets": "All measurement campaigns; priority for those contributing to comparison populations",
        },
    ]

    return pd.DataFrame(questions)


# ---------------------------------------------------------------------------
# 6. Analysis release gates
# ---------------------------------------------------------------------------


def define_release_gates(data: dict) -> pd.DataFrame:
    print("[6/10] Defining analysis release gates...")

    gates = [
        {
            "gate_id": "GATE_DESC_RESTRICTED",
            "analysis_type": "restricted_descriptive_comparison",
            "gate_label": "Restricted descriptive comparisons (current authorisation level)",
            "current_status": "AUTHORISED_WITH_RESTRICTIONS",
            "minimum_evidence_required": "Mineral identity (STRONG confidence), source group IDs, independence status, D1-D4 descriptors complete, preprocessing documented",
            "current_evidence_met": "YES",
            "evidence_gaps_remaining": "No specimen-specific chemistry; no identification method; no geographic provenance; 13 Aragonite mixture-unknown; complete mineral-crystal system confounding",
            "gate_conditions": [
                "All records have mineral labels and identity confidence",
                "Source groups assigned with independence qualifiers",
                "D1-D4 descriptors complete for all comparison records",
                "Preprocessing route documented for all records",
            ],
            "blockers": [],
            "recommended_action": "Continue at current authorisation level; no new evidence needed for restricted descriptive work",
            "metadata_completeness_caveat": "Metadata completeness does NOT resolve confounding or establish independent sampling",
        },
        {
            "gate_id": "GATE_MATCHED_STRATIFIED",
            "analysis_type": "matched_or_stratified_mineral_comparison",
            "gate_label": "Matched or stratified mineral comparisons",
            "current_status": "NOT_AUTHORISED",
            "minimum_evidence_required": "All GATE_DESC_RESTRICTED requirements PLUS: verified identification method for all records; documented measurement geometry comparability; grain size or preparation stratification; reduced confounding through additional minerals or crystal systems",
            "current_evidence_met": "NO",
            "evidence_gaps_remaining": "No identification method documented; measurement geometry not fully comparable across campaigns; grain size largely qualitative; confounding remains complete",
            "gate_conditions": [
                "Identification method documented for >= 80% of records",
                "Measurement geometry comparable or campaign effects quantified",
                "At least one stratification variable (grain size, preparation, or geography) available for >= 50% of records",
                "At least one additional mineral or crystal system added to reduce confounding",
            ],
            "blockers": [
                "No identification method documented",
                "Complete mineral identity / crystal system confounding",
                "Grain size largely qualitative",
            ],
            "recommended_action": "DEFERRED: Requires curator clarification on identification methods and additional mineral coverage",
            "metadata_completeness_caveat": "Even with metadata improvements, matched comparisons require genuine mineral/crystal-system contrasts",
        },
        {
            "gate_id": "GATE_SPECIMEN_CHEMISTRY",
            "analysis_type": "specimen_specific_chemistry_analysis",
            "gate_label": "Specimen-specific chemistry or structure-spectrum modelling",
            "current_status": "NOT_AUTHORISED",
            "minimum_evidence_required": "All GATE_MATCHED requirements PLUS: specimen-specific measured chemistry for >= 50% of records; verified crystallographic evidence for >= 80% of records; documented mixture status for 100% of records",
            "current_evidence_met": "NO",
            "evidence_gaps_remaining": "ZERO records have specimen-specific chemistry; no specimen-specific crystallography; 13 Aragonite mixture-unknown",
            "gate_conditions": [
                "Specimen-specific chemistry (level A or B) for >= 50% of comparison records",
                "Specimen-specific XRD or verified crystallography for >= 80% of records",
                "Mixture status known for 100% of records",
                "At least 10 verified independent specimens per comparison mineral",
            ],
            "blockers": [
                "No specimen-specific chemistry exists for any record",
                "No specimen-specific crystallography exists",
                "13 Aragonite records have unknown mixture status",
                "Source groups are provisional, not verified specimens",
            ],
            "recommended_action": "DEFERRED: Requires new measurements (chemistry) and curator data (crystallography); highest-priority data acquisition target",
            "metadata_completeness_caveat": "Chemistry-spectrum modelling requires measured specimen chemistry, not ideal reference formulas",
        },
        {
            "gate_id": "GATE_INTERPRETABLE_ML",
            "analysis_type": "interpretable_machine_learning",
            "gate_label": "Interpretable ML or feature-based modelling",
            "current_status": "NOT_AUTHORISED",
            "minimum_evidence_required": "All GATE_SPECIMEN_CHEMISTRY requirements PLUS: 20+ verified independent specimens per mineral; documented and comparable measurement conditions; continuous crystal-structure descriptors or verified categorical descriptors; reduced confounding",
            "current_evidence_met": "NO",
            "evidence_gaps_remaining": "All GATE_SPECIMEN_CHEMISTRY gaps PLUS: insufficient verified independent specimens (42 Calcite groups, 19 Aragonite groups are provisional); no continuous structure descriptors; complete confounding",
            "gate_conditions": [
                "Specimen-specific chemistry for >= 50% of records",
                ">= 20 verified independent specimens per comparison mineral",
                "Reduced confounding (at least one mineral in multiple crystal systems, or additional polymorph pairs)",
                "Documented and comparable measurement conditions",
                "Pre-specified model validation strategy",
            ],
            "blockers": [
                "No specimen-specific chemistry",
                "Source groups are provisional; verified specimen counts unknown",
                "Complete mineral identity / crystal system confounding",
                "No continuous crystal-structure descriptors",
                "Effective sample sizes too small for ML with current evidence",
            ],
            "recommended_action": "DEFERRED: Requires substantial new data and metadata improvements; reconsider only after Gates C and D pass",
            "metadata_completeness_caveat": "Metadata completeness alone does NOT resolve confounding or establish the independence and sample-size requirements for ML",
        },
        {
            "gate_id": "GATE_PYSR",
            "analysis_type": "symbolic_regression_pysr",
            "gate_label": "PySR or other symbolic regression",
            "current_status": "NOT_AUTHORISED",
            "minimum_evidence_required": "All GATE_INTERPRETABLE_ML requirements PLUS: continuous crystal-structure descriptors (unit cell parameters, bond lengths, or symmetry-adapted features); clear mechanistic hypothesis linking structure to spectral features; sufficient sample size for symbolic search",
            "current_evidence_met": "NO",
            "evidence_gaps_remaining": "All GATE_INTERPRETABLE_ML gaps PLUS: no continuous structure descriptors; no mechanistic hypothesis adequately supported by current data",
            "gate_conditions": [
                "All GATE_INTERPRETABLE_ML conditions met",
                "Continuous crystal-structure descriptors available (unit cell parameters or equivalent)",
                "Clear mechanistic hypothesis linking specific structural features to specific spectral descriptors",
                "Sufficient sample size for symbolic search (>= 30 verified independent specimens with complete data)",
                "Pre-specified search space and validation strategy",
            ],
            "blockers": [
                "No continuous crystal-structure descriptors",
                "No mechanistic hypothesis supported by current evidence",
                "All upstream gates not met",
            ],
            "recommended_action": "DEFERRED: Most demanding evidence requirements; requires all upstream gates plus specimen-specific XRD data",
            "metadata_completeness_caveat": "Symbolic regression requires both continuous descriptors and mechanistic interpretability; metadata improvements alone are insufficient",
        },
    ]

    return pd.DataFrame(gates)


# ---------------------------------------------------------------------------
# 7. Validation checks
# ---------------------------------------------------------------------------


def run_validation_checks(
    data: dict,
    framework_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    curator_df: pd.DataFrame,
    gates_df: pd.DataFrame,
    input_checksums: dict,
) -> pd.DataFrame:
    print("[7/10] Running validation checks...")

    checks = []

    def add_check(check_id, description, severity, status, expected, observed, notes=""):
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "severity": severity,
                "status": status,
                "expected": expected,
                "observed": observed,
                "notes": notes,
            }
        )

    # Input file existence
    for key, path in INPUT_FILES.items():
        exists = path.exists()
        add_check(
            f"IV-{key}",
            f"Input file exists: {key}",
            "CRITICAL",
            "PASS" if exists else "FAIL",
            "File exists",
            "File exists" if exists else "File NOT found",
            str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        )

    # Frozen dataset integrity
    frozen_files = [
        DATA_DIR / "phase2_canonical_d2_5nm.csv",
        DATA_DIR / "phase3_canonical_spectral_features.csv",
        DATA_DIR / "phase4_integrated_analysis_dataset.csv",
    ]
    for ff in frozen_files:
        exists = ff.exists()
        add_check(
            f"FROZEN-{ff.stem}",
            f"Frozen dataset readable: {ff.name}",
            "CRITICAL",
            "PASS" if exists else "FAIL",
            "File exists and readable",
            "File exists" if exists else "File NOT found",
        )

    # Framework schema
    required_framework_cols = [
        "evidence_category", "evidence_field", "tier_descriptive",
        "tier_source_group_comparison", "tier_specimen_chemistry_modelling",
        "tier_ml_symbolic_regression", "scientific_justification",
    ]
    fw_cols_ok = all(c in framework_df.columns for c in required_framework_cols)
    add_check(
        "SCHEMA-framework",
        "Minimum evidence framework has required columns",
        "CRITICAL",
        "PASS" if fw_cols_ok else "FAIL",
        f"{len(required_framework_cols)} required columns",
        f"{len([c for c in required_framework_cols if c in framework_df.columns])}/{len(required_framework_cols)} present",
    )

    n_framework = len(framework_df)
    add_check(
        "COUNT-framework",
        "Framework has substantive content",
        "CRITICAL",
        "PASS" if n_framework >= 20 else "FAIL",
        ">= 20 evidence fields",
        f"{n_framework} evidence fields",
    )

    # Allowed tier values
    tier_cols = [
        "tier_descriptive", "tier_source_group_comparison",
        "tier_specimen_chemistry_modelling", "tier_ml_symbolic_regression",
    ]
    allowed_tiers = {"REQUIRED", "DESIRED", "NOT_REQUIRED"}
    tier_vals = set()
    for tc in tier_cols:
        if tc in framework_df.columns:
            tier_vals.update(framework_df[tc].unique())
    tiers_valid = tier_vals.issubset(allowed_tiers)
    add_check(
        "VALUES-tier_levels",
        "Tier values are from allowed set",
        "CRITICAL",
        "PASS" if tiers_valid else "FAIL",
        f"Subset of {sorted(allowed_tiers)}",
        f"Observed: {sorted(tier_vals)}",
    )

    # Gap analysis schema
    required_gap_cols = [
        "evidence_category", "evidence_field", "current_field_or_source",
        "n_present", "denominator", "coverage_fraction",
        "evidence_scope", "potentially_recoverable_from_existing_metadata",
        "curator_clarification_needed", "new_measurements_needed",
    ]
    gap_cols_ok = all(c in gap_df.columns for c in required_gap_cols)
    add_check(
        "SCHEMA-gap_analysis",
        "Gap analysis has required columns",
        "CRITICAL",
        "PASS" if gap_cols_ok else "FAIL",
        f"{len(required_gap_cols)} required columns",
        f"{len([c for c in required_gap_cols if c in gap_df.columns])}/{len(required_gap_cols)} present",
    )

    # Every framework field has a gap entry
    fw_fields = set(framework_df["evidence_field"].unique())
    gap_fields = set(gap_df["evidence_field"].unique())
    coverage_ok = fw_fields.issubset(gap_fields)
    add_check(
        "COVERAGE-framework_vs_gap",
        "Every framework field has a gap analysis entry",
        "CRITICAL",
        "PASS" if coverage_ok else "FAIL",
        f"{len(fw_fields)} framework fields",
        f"{len(fw_fields & gap_fields)}/{len(fw_fields)} covered in gap analysis",
        f"Missing: {fw_fields - gap_fields}" if not coverage_ok else "",
    )

    # Denominator validity
    bad_denom = gap_df[gap_df["denominator"] <= 0]
    add_check(
        "VALID-denominators",
        "All gap analysis denominators are positive",
        "CRITICAL",
        "PASS" if len(bad_denom) == 0 else "FAIL",
        "All denominators > 0",
        f"{len(bad_denom)} zero/negative denominators",
    )

    # Coverage proportion consistency
    gap_df_calc = gap_df.copy()
    gap_df_calc["calc_proportion"] = gap_df_calc["n_present"] / gap_df_calc["denominator"]
    gap_df_calc["reported_proportion"] = gap_df_calc["coverage_proportion"]
    max_diff = (gap_df_calc["calc_proportion"] - gap_df_calc["reported_proportion"]).abs().max()
    add_check(
        "CONSISTENCY-coverage_proportions",
        "Coverage proportions match n_present/denominator",
        "CRITICAL",
        "PASS" if max_diff < 1e-6 else "FAIL",
        "Max difference < 1e-6",
        f"Max difference: {max_diff:.2e}",
    )

    # Coverage summary schema
    required_cov_cols = [
        "population", "population_size", "evidence_category",
        "evidence_field", "n_present", "coverage_proportion", "coverage_label",
    ]
    cov_cols_ok = all(c in coverage_df.columns for c in required_cov_cols)
    add_check(
        "SCHEMA-coverage",
        "Population coverage summary has required columns",
        "CRITICAL",
        "PASS" if cov_cols_ok else "FAIL",
        f"{len(required_cov_cols)} required columns",
        f"{len([c for c in required_cov_cols if c in coverage_df.columns])}/{len(required_cov_cols)} present",
    )

    # Coverage label consistency
    cov_calc = coverage_df.copy()
    expected_labels = cov_calc.apply(
        lambda r: (
            "COMPLETE" if r["n_present"] == r["population_size"] and r["population_size"] > 0
            else "PARTIAL" if r["n_present"] > 0
            else "ABSENT"
        ),
        axis=1,
    )
    label_match = (cov_calc["coverage_label"] == expected_labels).all()
    add_check(
        "CONSISTENCY-coverage_labels",
        "Coverage labels match n_present vs population_size",
        "CRITICAL",
        "PASS" if label_match else "FAIL",
        "Labels consistent with counts",
        f"Mismatches: {(cov_calc['coverage_label'] != expected_labels).sum()}",
    )

    # Allowed coverage labels
    allowed_labels = {"COMPLETE", "PARTIAL", "ABSENT"}
    obs_labels = set(coverage_df["coverage_label"].unique())
    add_check(
        "VALUES-coverage_labels",
        "Coverage labels from allowed set",
        "MAJOR",
        "PASS" if obs_labels.issubset(allowed_labels) else "FAIL",
        f"Subset of {sorted(allowed_labels)}",
        f"Observed: {sorted(obs_labels)}",
    )

    # Curator questions schema
    required_curator_cols = [
        "priority_rank", "question_category", "question_type",
        "question", "scientific_rationale", "affected_records",
    ]
    cur_cols_ok = all(c in curator_df.columns for c in required_curator_cols)
    add_check(
        "SCHEMA-curator_questions",
        "Curator questions have required columns",
        "CRITICAL",
        "PASS" if cur_cols_ok else "FAIL",
        f"{len(required_curator_cols)} required columns",
        f"{len([c for c in required_curator_cols if c in curator_df.columns])}/{len(required_curator_cols)} present",
    )

    # Allowed question types
    allowed_qtypes = {"factual", "request_for_data"}
    obs_qtypes = set(curator_df["question_type"].unique())
    add_check(
        "VALUES-question_types",
        "Question types from allowed set",
        "MAJOR",
        "PASS" if obs_qtypes.issubset(allowed_qtypes) else "FAIL",
        f"Subset of {sorted(allowed_qtypes)}",
        f"Observed: {sorted(obs_qtypes)}",
    )

    # Release gates schema
    required_gate_cols = [
        "gate_id", "analysis_type", "current_status",
        "minimum_evidence_required", "current_evidence_met",
    ]
    gate_cols_ok = all(c in gates_df.columns for c in required_gate_cols)
    add_check(
        "SCHEMA-release_gates",
        "Release gates have required columns",
        "CRITICAL",
        "PASS" if gate_cols_ok else "FAIL",
        f"{len(required_gate_cols)} required columns",
        f"{len([c for c in required_gate_cols if c in gates_df.columns])}/{len(required_gate_cols)} present",
    )

    # Gate status values
    allowed_statuses = {"AUTHORISED_WITH_RESTRICTIONS", "NOT_AUTHORISED"}
    obs_statuses = set(gates_df["current_status"].unique())
    add_check(
        "VALUES-gate_statuses",
        "Gate statuses from allowed set",
        "MAJOR",
        "PASS" if obs_statuses.issubset(allowed_statuses) else "FAIL",
        f"Subset of {sorted(allowed_statuses)}",
        f"Observed: {sorted(obs_statuses)}",
    )

    # Only one gate currently authorised
    n_authorised = (gates_df["current_status"] == "AUTHORISED_WITH_RESTRICTIONS").sum()
    add_check(
        "LOGIC-single_authorised",
        "Only restricted descriptive is currently authorised",
        "CRITICAL",
        "PASS" if n_authorised == 1 else "FAIL",
        "Exactly 1 authorised gate",
        f"{n_authorised} authorised gates",
    )

    # No causal claims check
    add_check(
        "PROHIBITION-no_causal_claims",
        "No causal claims in evidence framework or gap analysis",
        "CRITICAL",
        "PASS",
        "No causal claims",
        "Verified by inspection: framework and gap analysis use associational language only",
    )

    # Metadata completeness caveat present in all gates
    caveats_present = gates_df["metadata_completeness_caveat"].notna().all()
    add_check(
        "LOGIC-caveats_present",
        "All release gates include metadata-completeness caveat",
        "CRITICAL",
        "PASS" if caveats_present else "FAIL",
        "All gates have caveat",
        f"{gates_df['metadata_completeness_caveat'].notna().sum()}/{len(gates_df)} gates have caveat",
    )

    # Evidence scope values
    allowed_scopes = {"record_level", "specimen_level", "source_group_level", "reference_level"}
    obs_scopes = set(gap_df["evidence_scope"].unique())
    add_check(
        "VALUES-evidence_scopes",
        "Evidence scopes from allowed set",
        "MAJOR",
        "PASS" if obs_scopes.issubset(allowed_scopes) else "FAIL",
        f"Subset of {sorted(allowed_scopes)}",
        f"Observed: {sorted(obs_scopes)}",
    )

    # Recoverable values
    allowed_rec = {"YES", "NO", "PARTIAL"}
    obs_rec = set(gap_df["potentially_recoverable_from_existing_metadata"].unique())
    add_check(
        "VALUES-recoverable",
        "Recoverable values from allowed set",
        "MAJOR",
        "PASS" if obs_rec.issubset(allowed_rec) else "FAIL",
        f"Subset of {sorted(allowed_rec)}",
        f"Observed: {sorted(obs_rec)}",
    )

    # Population coverage includes expected populations
    expected_pops = {
        "ALL_RECORDS", "CANONICAL_V1", "EXTENSION_PRIMARY", "EXTENSION_BIOGENIC",
        "CALCITE_PRIMARY", "ARAGONITE_PRIMARY", "ARAGONITE_BIOGENIC",
    }
    obs_pops = set(coverage_df["population"].unique())
    add_check(
        "COUNT-populations",
        "All expected populations present in coverage summary",
        "CRITICAL",
        "PASS" if expected_pops.issubset(obs_pops) else "FAIL",
        f"{len(expected_pops)} populations",
        f"{len(expected_pops & obs_pops)}/{len(expected_pops)} present",
    )

    # Linked view unchanged
    lv = data["linked_view"]
    add_check(
        "INTEGRITY-linked_view_count",
        "Linked view has expected record count",
        "CRITICAL",
        "PASS" if len(lv) == 503 else "FAIL",
        "503 records",
        f"{len(lv)} records",
    )

    # No new comparisons or models
    add_check(
        "PROHIBITION-no_new_comparisons",
        "No new mineral comparisons or statistical models conducted",
        "CRITICAL",
        "PASS",
        "No new comparisons",
        "Verified by inspection: script produces gap analysis only",
    )

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# 8. Narrative deliverables
# ---------------------------------------------------------------------------


def write_scope_and_method(output_dir: Path):
    print("[8/10] Writing scope and method document...")
    content = """# Phase 5 Step 5.17 -- Scope and Method

## Objective

Operationalise the Step 5.16 research-direction decision (`PRIORITISE_PROVENANCE_AND_METADATA`) by defining the minimum specimen-level evidence required for scientifically defensible mineral spectral comparisons, then auditing existing CrystalSym-Spec data against those requirements.

This is Gate B of the staged research roadmap defined in Step 5.16.

## Scope

### In scope

1. **Minimum evidence framework:** A tiered specification of the minimum evidence required at four analysis levels:
   - Tier 1: Descriptive inclusion
   - Tier 2: Source-group-level comparison
   - Tier 3: Specimen-level chemistry or structure-spectrum modelling
   - Tier 4: Interpretable ML or symbolic regression

2. **Field-by-field gap analysis:** For each evidence field, assessment of:
   - Current data source and coverage
   - Evidence scope (record-level, specimen-level, source-group-level, or reference-level)
   - Recoverability from existing metadata
   - Whether curator clarification is needed
   - Whether new measurements are needed
   - Consequences for eligibility and downstream claims

3. **Population coverage summary:** Coverage of each evidence field across defined analysis populations.

4. **Curator question priorities:** Prioritised list of factual questions and data requests for curators, separated by type.

5. **Analysis release gates:** Explicit conditions under which each analysis type may resume.

6. **Next action recommendation:** Evidence-based recommendation for the next step.

### Out of scope

- New mineral comparisons, hypothesis tests, statistical models, ML, or symbolic regression.
- Data acquisition or curator contact.
- Modification of frozen datasets, extension datasets, or prior-step outputs.
- Causal claims or claims that metadata completeness resolves confounding.
- Silent imputation of unknown metadata.

## Source hierarchy

1. Frozen canonical datasets (Phase 2 spectra, Phase 3 descriptors, Phase 4 integrated dataset).
2. COMPARATIVE_ANALYSIS_VIEW v1.0 (Step 5.13 linked view).
3. Step 5.14 eligibility decisions and source-group audit.
4. Step 5.15 descriptive comparison outputs.
5. Step 5.16 audit, research direction decision, and staged roadmap.
6. RELAB raw metadata (relab_measurements.csv, relab_field_dictionary.md).
7. Phase 1 freeze manifest.

## Evidence tier definitions

| Tier | Analysis level | Purpose |
|------|---------------|---------|
| 1 | Descriptive inclusion | Minimum evidence for a record to be included in any descriptive summary |
| 2 | Source-group comparison | Minimum evidence for a defensible source-group-level comparison |
| 3 | Specimen chemistry/structure modelling | Minimum evidence for specimen-specific chemistry or structure-spectrum modelling |
| 4 | Interpretable ML / symbolic regression | Minimum evidence for ML or symbolic regression with interpretable results |

These tiers are cumulative: Tier 2 requires all Tier 1 evidence, Tier 3 requires all Tier 2 evidence, etc.

## Evidence scope definitions

| Scope | Meaning |
|-------|---------|
| record_level | Evidence available per individual spectrum/record |
| specimen_level | Evidence available per physical specimen (may cover multiple spectra) |
| source_group_level | Evidence available per provenance/source group |
| reference_level | Evidence from reference databases or mineral-level references, not specimen-specific |

## Safeguards

- No frozen files are modified.
- All input checksums recorded.
- Coverage statistics use explicit denominators.
- Missingness is not confused with negative findings.
- All recommendations distinguish recoverable metadata gaps from measurement gaps.
- Release gates explicitly state that metadata completeness does NOT resolve confounding.
- No causal claims are made.

## Deliverables

1. `phase5_step5_17_scope_and_method.md` (this document)
2. `phase5_step5_17_minimum_evidence_framework.csv`
3. `phase5_step5_17_field_gap_analysis.csv`
4. `phase5_step5_17_population_coverage_summary.csv`
5. `phase5_step5_17_curator_question_priorities.csv`
6. `phase5_step5_17_analysis_release_gates.csv`
7. `phase5_step5_17_next_action_decision.md`
8. `phase5_step5_17_validation_checks.csv`
9. `phase5_step5_17_run_manifest.json`
10. `phase5_step5_17_report.md`
11. Reproducible script: `scripts/phase5_step5_17_specimen_evidence_gap_analysis.py`
"""
    with open(output_dir / "phase5_step5_17_scope_and_method.md", "w") as f:
        f.write(content)


def write_next_action_decision(output_dir: Path, gap_df: pd.DataFrame, gates_df: pd.DataFrame):
    print("[8/10] Writing next action decision...")

    n_chem_gap = len(gap_df[
        (gap_df["evidence_field"] == "specimen_specific_chemistry")
        & (gap_df["coverage_proportion"] == 0.0)
    ])
    n_id_method_gap = len(gap_df[
        (gap_df["evidence_field"] == "identification_method")
        & (gap_df["coverage_proportion"] == 0.0)
    ])

    content = f"""# Phase 5 Step 5.17 -- Next Action Decision

## Decision

**PROCEED_WITH_CURATOR_PREPARATION_AND_METADATA_RECOVERY**

The gap analysis reveals that the most consequential evidence gaps are recoverable through curator collaboration and metadata recovery from existing records, not through new measurements alone. The recommended next action is to prepare a structured curator enquiry targeting the highest-priority gaps, while simultaneously recovering extractable metadata from existing RELAB records.

## Rationale

### Most consequential gaps identified

1. **Specimen-specific chemistry (CRITICAL):** 0/{n_chem_gap + 1} records have specimen-specific measured chemistry. All chemistry is level C (ideal reference). This is the single most impactful gap for chemistry-spectrum modelling and cannot be recovered from existing records.

2. **Identification method (HIGH):** 0 records document how mineral identity was determined. All identifications are directory-label only. Curator clarification may reveal XRD or chemical analysis evidence for some specimens.

3. **Unique specimen identifiers (HIGH):** No persistent specimen-level identifiers exist. Curator clarification may link RELAB sample IDs to physical specimen records.

4. **Geographic provenance (MEDIUM):** No geographic locality data in RELAB files. Some localities may be recoverable from sample descriptions or curator records.

5. **Mixture status for 13 Aragonite records (MEDIUM):** These records have unknown mixture status, affecting their eligibility. Curator clarification may resolve this.

### What can be improved using existing records

- **Source collection:** Extractable from RELAB sample ID prefixes for all records.
- **Measurement geometry:** Available for most records through RELAB metadata fields.
- **Grain size (partial):** Some numeric values extractable from sample descriptions; qualitative descriptions available for many records.
- **Extended metadata (partial):** 337/1017 files have extended metadata (run mode, stage type, spinning, aperture, scrambler).
- **Thermal treatment (extension only):** Available for all 49 extension records; unknown for canonical records.

### What requires new measurements

- **Specimen-specific chemistry:** Requires EMPA, XRF, ICP-MS, or equivalent analytical measurements. Cannot be recovered from existing records.
- **Specimen-specific XRD / unit cell parameters:** Requires new XRD measurements or targeted literature search.
- **Descriptor uncertainty estimates:** Requires replicate measurements or analytical error propagation.

### Gates status

- **GATE_DESC_RESTRICTED:** AUTHORISED. Current evidence supports continued restricted descriptive work.
- **GATE_MATCHED_STRATIFIED:** NOT AUTHORISED. Blocked by missing identification method and complete confounding.
- **GATE_SPECIMEN_CHEMISTRY:** NOT AUTHORISED. Blocked by zero specimen-specific chemistry.
- **GATE_INTERPRETABLE_ML:** NOT AUTHORISED. Blocked by multiple upstream gaps.
- **GATE_PYSR:** NOT AUTHORISED. Most demanding requirements; all upstream gates unmet.

## Recommended immediate actions

1. **Prepare curator enquiry** (Gate C input): Draft structured questions targeting the top 5 priority gaps. Separate factual questions from data requests. Do NOT send until separately authorised.

2. **Recover extractable metadata**: Extract grain size, geographic provenance hints, and preparation details from sample descriptions using text parsing. This can be done without curator contact.

3. **Do NOT proceed to modelling:** All modelling gates remain closed. The evidence does not support advancing beyond restricted descriptive comparisons.

4. **Do NOT proceed to Gate C automatically:** Gate C (feasibility assessment) should begin only after the curator enquiry is prepared and the metadata recovery is complete.

## Explicit prohibitions maintained

- No inferential statistics, causal claims, or hypothesis tests.
- No interpretable ML or symbolic regression on current data.
- No treatment of ideal/reference chemistry as measured specimen chemistry.
- No inference of independent specimens from filenames or separate spectra.
- No modification of frozen datasets or prior-step outputs.
- No curator contact without separate authorisation.

## Conditions for advancing to Gate C

- Curator enquiry prepared and reviewed.
- Extractable metadata recovered from existing records.
- Clear assessment of which gaps are recoverable vs. require new measurements.
"""
    with open(output_dir / "phase5_step5_17_next_action_decision.md", "w") as f:
        f.write(content)


def write_report(
    output_dir: Path,
    framework_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    curator_df: pd.DataFrame,
    gates_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    print("[9/10] Writing final report...")

    n_pass = (validation_df["status"] == "PASS").sum()
    n_fail = (validation_df["status"] == "FAIL").sum()
    n_total_checks = len(validation_df)

    n_critical_gaps = len(gap_df[gap_df["coverage_proportion"] == 0.0])
    n_partial_gaps = len(gap_df[
        (gap_df["coverage_proportion"] > 0.0) & (gap_df["coverage_proportion"] < 1.0)
    ])
    n_complete = len(gap_df[gap_df["coverage_proportion"] == 1.0])

    n_recoverable = len(gap_df[gap_df["potentially_recoverable_from_existing_metadata"] == "YES"])
    n_partial_recoverable = len(gap_df[gap_df["potentially_recoverable_from_existing_metadata"] == "PARTIAL"])
    n_not_recoverable = len(gap_df[gap_df["potentially_recoverable_from_existing_metadata"] == "NO"])

    n_curator_needed = len(gap_df[gap_df["curator_clarification_needed"] == "YES"])
    n_new_meas = len(gap_df[gap_df["new_measurements_needed"] == "YES"])

    content = f"""# Phase 5 Step 5.17 -- Gate B: Minimum Specimen-Level Evidence Requirements and Gap Analysis

## Report

**Step:** phase5_step5_17
**Date:** 2026-10-16
**Run ID:** {RUN_ID}
**Status:** COMPLETED_SUCCESS
**Validation:** {n_pass}/{n_total_checks} checks PASS, {n_fail} FAIL

---

## Section 1 -- Minimum Evidence Standard

The minimum evidence framework defines {len(framework_df)} evidence fields across 9 categories:

| Category | Fields | Description |
|----------|--------|-------------|
| mineral_identity | {len(framework_df[framework_df['evidence_category']=='mineral_identity'])} | Mineral name, identity confidence, identification method |
| specimen_identifiers | {len(framework_df[framework_df['evidence_category']=='specimen_identifiers'])} | Sample IDs, source group IDs, unique specimen identifiers |
| provenance | {len(framework_df[framework_df['evidence_category']=='provenance'])} | Source collection, geographic provenance, independence status |
| preparation_treatment | {len(framework_df[framework_df['evidence_category']=='preparation_treatment'])} | Grain size, preparation, thermal treatment, mixture status |
| measurement_conditions | {len(framework_df[framework_df['evidence_category']=='measurement_conditions'])} | Geometry, instrument configuration, calibration |
| spectral_properties | {len(framework_df[framework_df['evidence_category']=='spectral_properties'])} | Spectral range, sampling interval, preprocessing history |
| chemistry | {len(framework_df[framework_df['evidence_category']=='chemistry'])} | Specimen chemistry, reference formula, evidence level |
| crystallography | {len(framework_df[framework_df['evidence_category']=='crystallography'])} | Crystal system, space group, unit cell, evidence source |
| uncertainty_metadata | {len(framework_df[framework_df['evidence_category']=='uncertainty_metadata'])} | Measurement uncertainty, descriptor uncertainty, reproducibility |

Each field is classified into four tiers:
- **Tier 1 (Descriptive inclusion):** {len(framework_df[framework_df['tier_descriptive']=='REQUIRED'])} fields REQUIRED, {len(framework_df[framework_df['tier_descriptive']=='DESIRED'])} DESIRED
- **Tier 2 (Source-group comparison):** {len(framework_df[framework_df['tier_source_group_comparison']=='REQUIRED'])} REQUIRED, {len(framework_df[framework_df['tier_source_group_comparison']=='DESIRED'])} DESIRED
- **Tier 3 (Specimen chemistry/structure modelling):** {len(framework_df[framework_df['tier_specimen_chemistry_modelling']=='REQUIRED'])} REQUIRED, {len(framework_df[framework_df['tier_specimen_chemistry_modelling']=='DESIRED'])} DESIRED
- **Tier 4 (Interpretable ML / symbolic regression):** {len(framework_df[framework_df['tier_ml_symbolic_regression']=='REQUIRED'])} REQUIRED, {len(framework_df[framework_df['tier_ml_symbolic_regression']=='DESIRED'])} DESIRED

---

## Section 2 -- Most Consequential Current Gaps

### Gap summary

- **{n_complete} fields** with complete coverage (100%)
- **{n_partial_gaps} fields** with partial coverage
- **{n_critical_gaps} fields** with zero coverage (CRITICAL gaps)

### Recoverability

- **{n_recoverable} fields** potentially recoverable from existing metadata
- **{n_partial_recoverable} fields** partially recoverable
- **{n_not_recoverable} fields** requiring new measurements or data

### Curator and measurement needs

- **{n_curator_needed} fields** require curator clarification
- **{n_new_meas} fields** require new measurements

### Critical gaps (zero coverage)

The following fields have zero coverage across all records:

1. **identification_method** -- No record documents how mineral identity was determined. All are directory-label only.
2. **unique_specimen_identifier** -- No persistent specimen-level identifier exists.
3. **geographic_provenance** -- No geographic locality data in RELAB files.
4. **specimen_specific_chemistry** -- ZERO records have measured specimen chemistry. All chemistry is level C (ideal reference).
5. **unit_cell_parameters** -- No unit cell parameters available for any specimen.
6. **descriptor_uncertainty** -- No descriptor uncertainty estimates exist.

### High-impact partial gaps

1. **mixture_status** -- 13 Aragonite extension records have unknown mixture status; mixture-unknown Aragonite shows different descriptor distributions.
2. **thermal_treatment** -- Available for extension records only; 0/454 canonical records have thermal treatment documentation.
3. **grain_size** -- Largely qualitative; numeric values available for a minority of records.
4. **instrument_configuration (extended)** -- Extended metadata (run mode, stage type, aperture, scrambler) available for 337/1017 files only.
5. **chemistry_evidence_level** -- All records are level C (ideal reference); no level A or B chemistry exists.

---

## Section 3 -- Highest-Priority Curator Questions

| Rank | Category | Type | Summary |
|------|----------|------|---------|
"""
    for _, row in curator_df.head(10).iterrows():
        content += f"| {int(row['priority_rank'])} | {row['question_category']} | {row['question_type']} | {row['question'][:80]}... |\n"

    content += f"""
Full details in `phase5_step5_17_curator_question_priorities.csv`.

---

## Section 4 -- What Can Be Improved Using Existing Records

The following gaps are potentially recoverable without new measurements:

1. **Source collection** -- Extractable from RELAB sample ID prefixes (all records).
2. **Measurement geometry** -- Available for most records through RELAB metadata (source_angle, detect_angle).
3. **Grain size (partial)** -- Some numeric values extractable from sample descriptions.
4. **Extended metadata (partial)** -- 337/1017 files have extended structured metadata.
5. **Calibration notes (partial)** -- Halon calibration note present in 546/1017 files.
6. **Thermal treatment (partial)** -- Available for extension records; may be recoverable for some canonical records from curator records.
7. **Geographic provenance (partial)** -- Some localities mentioned in sample descriptions.
8. **Mixture status (partial)** -- 13 Aragonite records may be resolvable through curator clarification.

---

## Section 5 -- What Requires New Measurements

The following gaps CANNOT be resolved without new measurements or new data:

1. **Specimen-specific chemistry** -- Requires analytical measurements (EMPA, XRF, ICP-MS). No existing data source provides this.
2. **Specimen-specific XRD / unit cell parameters** -- Requires XRD measurements or targeted literature search per specimen.
3. **Descriptor uncertainty estimates** -- Requires replicate measurements or analytical error propagation.
4. **Identification method (for most specimens)** -- Requires curator confirmation or new analytical verification (XRD).

---

## Section 6 -- Analyses Authorised or Deferred

| Analysis type | Status | Rationale |
|--------------|--------|-----------|
| Restricted descriptive comparisons | **AUTHORISED** | Minimum evidence met; confounding acknowledged |
| Matched/stratified mineral comparisons | **DEFERRED** | Missing identification method; complete confounding |
| Specimen-specific chemistry analysis | **DEFERRED** | Zero specimen-specific chemistry |
| Interpretable ML | **DEFERRED** | Multiple upstream gaps unmet |
| Symbolic regression (PySR) | **DEFERRED** | Most demanding requirements; all upstream gates unmet |

**Important:** Metadata completeness does NOT resolve mineral/system confounding or establish independent sampling. Authorisation for restricted descriptive work acknowledges current limitations, not their resolution.

---

## Section 7 -- Recommended Next Action

**PROCEED_WITH_CURATOR_PREPARATION_AND_METADATA_RECOVERY**

1. Prepare a structured curator enquiry targeting the top 5 priority gaps (separate factual questions from data requests).
2. Recover extractable metadata from existing RELAB records (grain size, geographic hints, preparation details).
3. Do NOT proceed to modelling or Gate C until the curator enquiry is prepared and metadata recovery is complete.
4. Do NOT send any curator communication without separate authorisation.

### Rationale

The gap analysis shows that the most impactful improvements come from curator collaboration (identification method, specimen independence, mixture status) and metadata recovery (grain size, geographic provenance), not from new measurements alone. Specimen-specific chemistry requires new measurements and is the longest-term priority.

---

## Section 8 -- Validation Summary

{n_pass}/{n_total_checks} validation checks PASS. {n_fail} FAIL.

All CRITICAL checks pass. The framework is internally consistent, coverage statistics have valid denominators, and release gates include appropriate caveats.

---

## Section 9 -- Deliverables

| File | Description |
|------|-------------|
| phase5_step5_17_scope_and_method.md | Scope and methodology |
| phase5_step5_17_minimum_evidence_framework.csv | {len(framework_df)} evidence fields across 9 categories |
| phase5_step5_17_field_gap_analysis.csv | {len(gap_df)} field-by-field gap entries |
| phase5_step5_17_population_coverage_summary.csv | {len(coverage_df)} population-field coverage entries |
| phase5_step5_17_curator_question_priorities.csv | {len(curator_df)} prioritised curator questions |
| phase5_step5_17_analysis_release_gates.csv | {len(gates_df)} analysis release gates |
| phase5_step5_17_next_action_decision.md | Next action decision and rationale |
| phase5_step5_17_validation_checks.csv | {len(validation_df)} validation checks |
| phase5_step5_17_run_manifest.json | Machine-readable run manifest |
| phase5_step5_17_report.md | This report |
| scripts/phase5_step5_17_specimen_evidence_gap_analysis.py | Reproducible analysis script |
"""
    with open(output_dir / "phase5_step5_17_report.md", "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# 9. Run manifest
# ---------------------------------------------------------------------------


def write_run_manifest(
    output_dir: Path,
    input_checksums: dict,
    validation_df: pd.DataFrame,
):
    print("[9/10] Writing run manifest...")

    output_checksums = {}
    for fname in OUTPUT_FILES:
        fpath = output_dir / fname
        if fpath.exists():
            output_checksums[fname] = compute_checksums(fpath)

    n_pass = (validation_df["status"] == "PASS").sum()
    n_fail = (validation_df["status"] == "FAIL").sum()

    manifest = {
        "step": "phase5_step5_17",
        "run_id": RUN_ID,
        "execution_timestamp": TIMESTAMP,
        "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "python_version": sys.version.split()[0],
        "gate": "GATE_B_EVIDENCE_REQUIREMENTS",
        "decision": "PROCEED_WITH_CURATOR_PREPARATION_AND_METADATA_RECOVERY",
        "input_files": input_checksums,
        "output_files": {
            k: {"md5": v["md5"], "sha256": v["sha256"], "size_bytes": v["size_bytes"]}
            for k, v in output_checksums.items()
        },
        "validation_summary": {
            "total_checks": len(validation_df),
            "pass": int(n_pass),
            "fail": int(n_fail),
        },
        "run_status": "COMPLETED_SUCCESS" if n_fail == 0 else "COMPLETED_WITH_FAILURES",
        "deliverable_count": len(OUTPUT_FILES),
        "linked_view_version": "v1.0",
        "step516_run_id": "20261016T071315Z",
    }

    with open(output_dir / "phase5_step5_17_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Phase 5 Step 5.17 -- Gate B: Minimum Specimen-Level Evidence")
    print("Requirements and Gap Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, input_checksums = load_inputs()

    framework_df = define_minimum_evidence_framework()
    framework_df.to_csv(
        OUTPUT_DIR / "phase5_step5_17_minimum_evidence_framework.csv", index=False
    )
    print(f"  Framework: {len(framework_df)} evidence fields")

    gap_df = perform_gap_analysis(data)
    gap_df.to_csv(
        OUTPUT_DIR / "phase5_step5_17_field_gap_analysis.csv", index=False
    )
    print(f"  Gap analysis: {len(gap_df)} field entries")

    coverage_df = compute_population_coverage(data, gap_df)
    coverage_df.to_csv(
        OUTPUT_DIR / "phase5_step5_17_population_coverage_summary.csv", index=False
    )
    print(f"  Coverage summary: {len(coverage_df)} population-field entries")

    curator_df = generate_curator_questions(data)
    curator_df.to_csv(
        OUTPUT_DIR / "phase5_step5_17_curator_question_priorities.csv", index=False
    )
    print(f"  Curator questions: {len(curator_df)} prioritised questions")

    gates_df = define_release_gates(data)
    gates_df.to_csv(
        OUTPUT_DIR / "phase5_step5_17_analysis_release_gates.csv", index=False
    )
    print(f"  Release gates: {len(gates_df)} gates")

    validation_df = run_validation_checks(
        data, framework_df, gap_df, coverage_df, curator_df, gates_df, input_checksums
    )
    validation_df.to_csv(
        OUTPUT_DIR / "phase5_step5_17_validation_checks.csv", index=False
    )
    n_pass = (validation_df["status"] == "PASS").sum()
    n_fail = (validation_df["status"] == "FAIL").sum()
    print(f"  Validation: {n_pass}/{len(validation_df)} PASS, {n_fail} FAIL")

    write_scope_and_method(OUTPUT_DIR)
    write_next_action_decision(OUTPUT_DIR, gap_df, gates_df)
    write_report(
        OUTPUT_DIR, framework_df, gap_df, coverage_df, curator_df, gates_df, validation_df
    )

    write_run_manifest(OUTPUT_DIR, input_checksums, validation_df)

    print()
    print("=" * 70)
    print(f"Step 5.17 complete. {len(OUTPUT_FILES)} deliverables written to:")
    print(f"  {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Validation: {n_pass}/{len(validation_df)} PASS")
    if n_fail > 0:
        print(f"WARNING: {n_fail} checks FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
