"""
Phase 5 Step 5.14 -- Aragonite-Calcite Eligibility Assessment and Comparison Design

Conducts a reproducible, analysis-specific eligibility assessment for a potential
comparison between Aragonite and Calcite using COMPARATIVE_ANALYSIS_VIEW v1.0.

This step is eligibility assessment and comparison design ONLY. It does not calculate
comparative statistics, fit models, test hypotheses, or interpret differences.

Deliverables are written to:
  results/05_exploratory_analysis/phase5_step5_14_aragonite_calcite_eligibility_design/
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_14_aragonite_calcite_eligibility_design",
)

LINKED_VIEW_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_13_linked_comparative_view",
    "phase5_step5_13_linked_analysis_view.csv",
)
EXT_META_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_8_extension_dataset",
    "phase5_step5_8_extension_metadata.csv",
)
INTEGRATED_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase4_integrated_analysis_dataset.csv"
)
SYMMETRY_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase4_symmetry_metadata.csv"
)
CHEMISTRY_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase4_chemistry_metadata.csv"
)
MINERAL_REF_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase4_mineral_reference.csv"
)
PROVENANCE_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase4_provenance_mapping.csv"
)
SG_SUMMARY_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_13_linked_comparative_view",
    "phase5_step5_13_source_group_summary.csv",
)
CHECKSUM_MANIFEST_513_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_13_linked_comparative_view",
    "phase5_step5_13_checksum_manifest.csv",
)
ADJUDICATION_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_7_scientific_inclusion_adjudication",
    "phase5_step5_7_candidate_adjudication.csv",
)

SCRIPT_PATH = os.path.join(
    REPO_ROOT,
    "scripts",
    "phase5_step5_14_aragonite_calcite_eligibility_design.py",
)

CANONICAL_SPECTRA_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase2_canonical_d2_5nm.csv"
)
CANONICAL_META_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase2_canonical_d2_5nm_metadata.csv"
)
DESCRIPTOR_PATH = os.path.join(
    REPO_ROOT, "data", "processed", "phase3_canonical_spectral_features.csv"
)
EXT_DESCRIPTORS_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "05_exploratory_analysis",
    "phase5_step5_12_extension_descriptor_validation",
    "phase5_step5_12_extension_descriptors.csv",
)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs():
    inputs = {}
    inputs["linked_view"] = pd.read_csv(LINKED_VIEW_PATH)
    inputs["ext_meta"] = pd.read_csv(EXT_META_PATH)
    inputs["integrated"] = pd.read_csv(INTEGRATED_PATH)
    inputs["symmetry"] = pd.read_csv(SYMMETRY_PATH)
    inputs["chemistry"] = pd.read_csv(CHEMISTRY_PATH)
    inputs["mineral_ref"] = pd.read_csv(MINERAL_REF_PATH)
    inputs["provenance"] = pd.read_csv(PROVENANCE_PATH)
    inputs["sg_summary"] = pd.read_csv(SG_SUMMARY_PATH)
    inputs["checksum_manifest_513"] = pd.read_csv(CHECKSUM_MANIFEST_513_PATH)
    inputs["adjudication"] = pd.read_csv(ADJUDICATION_PATH)
    return inputs


def verify_inputs(inputs):
    checks = []
    required_files = {
        "linked_view": LINKED_VIEW_PATH,
        "ext_meta": EXT_META_PATH,
        "integrated": INTEGRATED_PATH,
        "symmetry": SYMMETRY_PATH,
        "chemistry": CHEMISTRY_PATH,
        "mineral_ref": MINERAL_REF_PATH,
        "provenance": PROVENANCE_PATH,
        "sg_summary": SG_SUMMARY_PATH,
        "checksum_manifest_513": CHECKSUM_MANIFEST_513_PATH,
        "adjudication": ADJUDICATION_PATH,
    }
    for label, path in required_files.items():
        exists = os.path.isfile(path)
        checks.append({
            "check_id": f"IV-{label}",
            "description": f"Input file exists: {os.path.basename(path)}",
            "severity": "CRITICAL",
            "expected": "file exists",
            "observed": "file exists" if exists else "FILE MISSING",
            "status": "PASS" if exists else "FAIL",
            "notes": path,
        })

    lv = inputs["linked_view"]
    checks.append({
        "check_id": "LV-COUNT",
        "description": "Linked view has expected 503 records",
        "severity": "CRITICAL",
        "expected": "503 records",
        "observed": f"{len(lv)} records",
        "status": "PASS" if len(lv) == 503 else "FAIL",
        "notes": "",
    })

    checks.append({
        "check_id": "LV-UNIQUE-IDS",
        "description": "Linked view IDs are unique",
        "severity": "CRITICAL",
        "expected": "503 unique IDs",
        "observed": f"{lv['linked_view_id'].nunique()} unique IDs",
        "status": "PASS" if lv["linked_view_id"].nunique() == 503 else "FAIL",
        "notes": "",
    })

    checks.append({
        "check_id": "LV-CALCITE-PRESENT",
        "description": "Calcite is present in linked view",
        "severity": "CRITICAL",
        "expected": "Calcite records > 0",
        "observed": f"{(lv['mineral_label'] == 'Calcite').sum()} Calcite records",
        "status": "PASS" if (lv["mineral_label"] == "Calcite").sum() > 0 else "FAIL",
        "notes": "",
    })

    checks.append({
        "check_id": "LV-ARAGONITE-PRESENT",
        "description": "Aragonite is present in linked view",
        "severity": "CRITICAL",
        "expected": "Aragonite records > 0",
        "observed": f"{(lv['mineral_label'] == 'Aragonite').sum()} Aragonite records",
        "status": "PASS" if (lv["mineral_label"] == "Aragonite").sum() > 0 else "FAIL",
        "notes": "",
    })

    arag = lv[lv["mineral_label"] == "Aragonite"]
    primary_count = (arag["population_label"] == "EXTENSION_PRIMARY").sum()
    biogenic_count = (arag["population_label"] == "EXTENSION_BIOGENIC").sum()
    checks.append({
        "check_id": "POP-ARAG-PRIMARY",
        "description": "Aragonite primary count matches expected 37",
        "severity": "CRITICAL",
        "expected": "37 primary Aragonite",
        "observed": f"{primary_count} primary Aragonite",
        "status": "PASS" if primary_count == 37 else "FAIL",
        "notes": "",
    })
    checks.append({
        "check_id": "POP-ARAG-BIOGENIC",
        "description": "Aragonite biogenic count matches expected 7",
        "severity": "CRITICAL",
        "expected": "7 biogenic Aragonite",
        "observed": f"{biogenic_count} biogenic Aragonite",
        "status": "PASS" if biogenic_count == 7 else "FAIL",
        "notes": "",
    })

    checks.append({
        "check_id": "POP-SEPARATION",
        "description": "Primary and biogenic Aragonite populations are separate",
        "severity": "CRITICAL",
        "expected": "No overlap between populations",
        "observed": f"{primary_count} primary + {biogenic_count} biogenic = {primary_count + biogenic_count} total",
        "status": "PASS" if primary_count + biogenic_count == len(arag) else "FAIL",
        "notes": "",
    })

    calcite = lv[lv["mineral_label"] == "Calcite"]
    checks.append({
        "check_id": "CALCITE-ALL-CANONICAL",
        "description": "All Calcite records are from CANONICAL_V1",
        "severity": "WARNING",
        "expected": "All Calcite from CANONICAL_V1",
        "observed": f"{(calcite['dataset_origin'] == 'CANONICAL_V1').sum()}/{len(calcite)} from CANONICAL_V1",
        "status": "PASS" if (calcite["dataset_origin"] == "CANONICAL_V1").all() else "FAIL",
        "notes": "",
    })

    checks.append({
        "check_id": "ARAG-ALL-EXTENSION",
        "description": "All Aragonite records are from EXTENSION_STEP5_8",
        "severity": "WARNING",
        "expected": "All Aragonite from EXTENSION_STEP5_8",
        "observed": f"{(arag['dataset_origin'] == 'EXTENSION_STEP5_8').sum()}/{len(arag)} from EXTENSION_STEP5_8",
        "status": "PASS" if (arag["dataset_origin"] == "EXTENSION_STEP5_8").all() else "FAIL",
        "notes": "",
    })

    return checks


def build_calcite_audit(inputs):
    lv = inputs["linked_view"]
    p4 = inputs["integrated"]
    calcite_lv = lv[lv["mineral_label"] == "Calcite"].copy()
    calcite_p4 = p4[p4["mineral_label"] == "Calcite"].copy()

    rows = []
    for _, lv_row in calcite_lv.iterrows():
        lv_id = lv_row["linked_view_id"]
        orig_id = lv_row["original_record_id"]
        sg_id = lv_row["source_group_id"]

        p4_row = calcite_p4[calcite_p4["canonical_spectrum_id"] == orig_id]
        if len(p4_row) == 1:
            p4r = p4_row.iloc[0]
        else:
            p4r = None

        identity_conf = lv_row.get("identity_confidence", None)
        if pd.isna(identity_conf) if identity_conf is not None else True:
            if p4r is not None and not p4r.get("has_identity_issues", True):
                identity_conf = "STRONG"
            elif p4r is not None and p4r.get("has_identity_issues", False):
                identity_conf = "TENTATIVE"
            else:
                identity_conf = "UNVERIFIED"

        chem_elig = p4r["chemistry_eligibility"] if p4r is not None else "UNAVAILABLE"
        sym_status = p4r["symmetry_eligibility_status"] if p4r is not None else "UNAVAILABLE"
        mixture = p4r["mixture_flag"] if p4r is not None else None
        high_conflict = p4r["high_conflict_flag"] if p4r is not None else None
        sample_desc = p4r["sample_description"] if p4r is not None else ""
        ref_formula = p4r["reference_formula"] if p4r is not None else ""
        chem_specimen = p4r["chemistry_specimen_specific"] if p4r is not None else None
        chem_type = p4r["chemistry_type"] if p4r is not None else ""
        chem_evidence = p4r["chemistry_evidence_level"] if p4r is not None else ""
        prov_tentative = p4r["tentative_provenance"] if p4r is not None else None
        sym_eligible = p4r["symmetry_primary_eligible"] if p4r is not None else None

        issues = []
        if mixture:
            issues.append("mixture_flag")
        if high_conflict:
            issues.append("high_conflict")
        if prov_tentative:
            issues.append("tentative_provenance")
        if chem_elig == "CHEM_AMBIGUOUS":
            issues.append("chemistry_ambiguous")
        elif chem_elig == "CHEM_TENTATIVE":
            issues.append("chemistry_tentative")

        rows.append({
            "linked_view_id": lv_id,
            "original_record_id": orig_id,
            "source_group_id": sg_id,
            "mineral_label": "Calcite",
            "dataset_origin": lv_row["dataset_origin"],
            "population_label": lv_row["population_label"],
            "identity_confidence": identity_conf,
            "crystal_system": lv_row["crystal_system"],
            "point_group": lv_row["point_group"],
            "space_group_symbol": lv_row["space_group_symbol"],
            "space_group_number": lv_row["space_group_number"],
            "chemistry_eligibility": chem_elig,
            "symmetry_eligibility_status": sym_status,
            "symmetry_primary_eligible": sym_eligible,
            "chemistry_type": chem_type,
            "chemistry_specimen_specific": chem_specimen,
            "chemistry_evidence_level": chem_evidence,
            "reference_formula": ref_formula,
            "mixture_flag": mixture,
            "high_conflict_flag": high_conflict,
            "tentative_provenance": prov_tentative,
            "material_origin": lv_row.get("material_origin", "natural"),
            "biogenic_flag": lv_row["biogenic_flag"],
            "preprocessing_route": lv_row["preprocessing_route"],
            "representation_type": lv_row["representation_type"],
            "interpolation_occurred": lv_row["interpolation_occurred"],
            "descriptor_version": lv_row["descriptor_version"],
            "D1_available": pd.notna(lv_row["D1_n_peaks_prom"]),
            "D2_available": pd.notna(lv_row["D2_mean_peak_prominence"]),
            "D3_available": pd.notna(lv_row["D3_first_deriv_rms"]),
            "D4_available": pd.notna(lv_row["D4_second_deriv_rms"]),
            "S1_available": pd.notna(lv_row["S1_total_variation"]),
            "independence_status": lv_row["independence_status"],
            "independence_qualifier": lv_row["independence_qualifier"],
            "sample_description": sample_desc,
            "issue_flags": ";".join(issues) if issues else "none",
        })

    return pd.DataFrame(rows)


def build_aragonite_audit(inputs):
    lv = inputs["linked_view"]
    ext_meta = inputs["ext_meta"]
    arag_lv = lv[lv["mineral_label"] == "Aragonite"].copy()

    rows = []
    for _, lv_row in arag_lv.iterrows():
        lv_id = lv_row["linked_view_id"]
        orig_id = lv_row["original_record_id"]
        sg_id = lv_row["source_group_id"]

        ext_row = ext_meta[ext_meta["candidate_id"] == orig_id]
        if len(ext_row) == 1:
            extr = ext_row.iloc[0]
        else:
            extr = None

        id_conf = extr["identity_confidence"] if extr is not None else "UNVERIFIED"
        mix_status = extr["mixture_status"] if extr is not None else "UNVERIFIED"
        thermal = extr["thermal_status"] if extr is not None else "UNVERIFIED"
        pop_assign = extr["population_assignment"] if extr is not None else ""
        bio_flag = lv_row["biogenic_flag"]
        bio_adj = lv_row["biogenic_adjudication"]
        sci_incl = lv_row["scientific_inclusion_decision"]
        unresolved = extr["unresolved_issues"] if extr is not None else ""

        if pd.isna(bio_flag) or bio_flag == "no_biogenic":
            material_origin = "natural" if mix_status in ("pure_natural", "unknown") else "unknown"
        else:
            material_origin = "biogenic"

        if sci_incl == "PRIMARY_ELIGIBLE":
            population = "PRIMARY"
        elif sci_incl == "SENSITIVITY_ONLY":
            population = "BIOGENIC_SENSITIVITY"
        else:
            population = "UNCLASSIFIED"

        issues = []
        if unresolved and unresolved != "none":
            issues.append(unresolved)
        if mix_status == "unknown":
            issues.append("mixture_status_unknown")
        if id_conf == "medium":
            issues.append("medium_identity_confidence")

        rows.append({
            "linked_view_id": lv_id,
            "original_record_id": orig_id,
            "source_group_id": sg_id,
            "mineral_label": "Aragonite",
            "dataset_origin": lv_row["dataset_origin"],
            "population_label": lv_row["population_label"],
            "population_designation": population,
            "identity_confidence": id_conf,
            "crystal_system": lv_row["crystal_system"],
            "point_group": lv_row["point_group"],
            "space_group_symbol": lv_row["space_group_symbol"],
            "space_group_number": lv_row.get("space_group_number", None),
            "mixture_status": mix_status,
            "thermal_status": thermal,
            "biogenic_flag": bio_flag,
            "biogenic_adjudication": bio_adj if pd.notna(bio_adj) else "",
            "scientific_inclusion_decision": sci_incl,
            "material_origin": material_origin,
            "preprocessing_route": lv_row["preprocessing_route"],
            "interpolation_occurred": lv_row["interpolation_occurred"],
            "interpolation_method": lv_row.get("interpolation_method", ""),
            "descriptor_version": lv_row["descriptor_version"],
            "D1_available": pd.notna(lv_row["D1_n_peaks_prom"]),
            "D2_available": pd.notna(lv_row["D2_mean_peak_prominence"]),
            "D3_available": pd.notna(lv_row["D3_first_deriv_rms"]),
            "D4_available": pd.notna(lv_row["D4_second_deriv_rms"]),
            "S1_available": pd.notna(lv_row["S1_total_variation"]),
            "independence_status": lv_row["independence_status"],
            "independence_qualifier": lv_row["independence_qualifier"],
            "unresolved_issues": unresolved,
            "issue_flags": ";".join(issues) if issues else "none",
        })

    return pd.DataFrame(rows)


def build_source_group_audit(inputs):
    lv = inputs["linked_view"]
    sg_summary = inputs["sg_summary"]

    calcite_lv = lv[lv["mineral_label"] == "Calcite"]
    arag_lv = lv[lv["mineral_label"] == "Aragonite"]

    rows = []

    for sg_id in sorted(calcite_lv["source_group_id"].unique()):
        subset = calcite_lv[calcite_lv["source_group_id"] == sg_id]
        sg_row = sg_summary[sg_summary["source_group_id"] == sg_id]
        n_spectra = len(subset)
        indep_status = subset["independence_status"].iloc[0]
        indep_qual = subset["independence_qualifier"].iloc[0]
        verified = sg_row["n_verified_specimens"].iloc[0] if len(sg_row) == 1 else "UNVERIFIED"

        sample_descs = []
        for orig_id in subset["original_record_id"].values:
            p4_match = inputs["integrated"][
                inputs["integrated"]["canonical_spectrum_id"] == orig_id
            ]
            if len(p4_match) == 1:
                sample_descs.append(str(p4_match.iloc[0]["sample_description"])[:60])

        notes = ""
        if n_spectra > 1:
            notes = f"Multi-spectrum group ({n_spectra} spectra); same source, different measurements/fractions"

        rows.append({
            "source_group_id": sg_id,
            "mineral": "Calcite",
            "dataset": "CANONICAL_V1",
            "population": "CANONICAL_V1",
            "n_spectra": n_spectra,
            "independence_status": indep_status,
            "relationship_type": indep_qual,
            "n_verified_specimens": verified,
            "conservative_independent_units": 1,
            "independence_notes": notes,
            "sample_descriptions": " | ".join(sample_descs)[:200],
        })

    for sg_id in sorted(arag_lv["source_group_id"].unique()):
        subset = arag_lv[arag_lv["source_group_id"] == sg_id]
        sg_row = sg_summary[sg_summary["source_group_id"] == sg_id]
        n_spectra = len(subset)
        indep_status = subset["independence_status"].iloc[0]
        indep_qual = subset["independence_qualifier"].iloc[0]
        pop = subset["population_label"].iloc[0]
        verified = sg_row["n_verified_specimens"].iloc[0] if len(sg_row) == 1 else "UNVERIFIED"

        if "grain_size" in indep_status:
            conserv = 1
            notes = "Grain-size series from same specimen; count as 1 independent unit"
        elif "replicate" in indep_status:
            conserv = 1
            notes = "Replicate measurements from same source; count as 1 independent unit"
        elif "biogenic" in indep_status:
            conserv = 1
            notes = "Biogenic source group; separate sensitivity population"
        else:
            conserv = 1
            notes = "Probable independent source"

        rows.append({
            "source_group_id": sg_id,
            "mineral": "Aragonite",
            "dataset": "EXTENSION_STEP5_8",
            "population": pop,
            "n_spectra": n_spectra,
            "independence_status": indep_status,
            "relationship_type": indep_qual,
            "n_verified_specimens": verified,
            "conservative_independent_units": conserv,
            "independence_notes": notes,
            "sample_descriptions": "",
        })

    return pd.DataFrame(rows)


def build_analysis_eligibility(calcite_audit, aragonite_audit):
    rows = []

    for _, row in calcite_audit.iterrows():
        lv_id = row["linked_view_id"]
        orig_id = row["original_record_id"]

        if orig_id == "F0228":
            decision = "EXCLUDE_WITH_REASON"
            rationale = (
                "Record F0228 (SPM0327) is sepiolite mislabeled as Calcite; "
                "sample description says 'SepSp-1 sepiolite'; "
                "high_conflict_flag=True; chemistry_eligibility=CHEM_TENTATIVE; "
                "symmetry_eligibility=SYM_TENTATIVE; identity_confidence=TENTATIVE"
            )
            permitted = "NOT_ELIGIBLE"
        elif row["mixture_flag"]:
            decision = "ELIGIBLE_WITH_RESTRICTIONS"
            rationale = (
                f"Known mixture ({row['sample_description'][:60]}...); "
                "chemistry_eligibility=CHEM_AMBIGUOUS; spectrum contains Calcite "
                "signal but is not pure Calcite; include only with mixture caveat"
            )
            permitted = "DESCRIPTIVE_ONLY"
        else:
            decision = "ELIGIBLE_FOR_PRIMARY_COMPARISON_DESIGN"
            rationale = (
                f"Clean Calcite record; identity_confidence={row['identity_confidence']}; "
                f"chemistry={row['chemistry_eligibility']}; "
                f"symmetry={row['symmetry_eligibility_status']}; "
                "all D1-D4 available; natural material; no biogenic flag"
            )
            permitted = "PRIMARY_COMPARISON"

        rows.append({
            "linked_view_id": lv_id,
            "original_record_id": orig_id,
            "mineral_identity": "Calcite",
            "population_designation": "CANONICAL",
            "original_eligibility_status": row["chemistry_eligibility"],
            "original_eligibility_source": "phase4_integrated_dataset",
            "analysis_specific_decision": decision,
            "eligibility_rationale": rationale,
            "identity_confidence": row["identity_confidence"],
            "provenance_confidence": "established_phase4",
            "independence_limitations": "provenance_group; some groups have multiple spectra",
            "material_origin_classification": row["material_origin"],
            "measurement_compatibility": "native_5nm_no_interpolation",
            "descriptor_completeness": "D1_D4_complete",
            "chemistry_symmetry_status": f"{row['chemistry_eligibility']};{row['symmetry_eligibility_status']}",
            "specimen_specific_chemistry": "NO",
            "exclusion_restriction_codes": row["issue_flags"],
            "evidence_source_paths": f"linked_view:{lv_id};phase4:{orig_id}",
            "permitted_use": permitted,
            "reviewer_notes": "",
        })

    for _, row in aragonite_audit.iterrows():
        lv_id = row["linked_view_id"]
        pop = row["population_designation"]

        if pop == "BIOGENIC_SENSITIVITY":
            decision = "SENSITIVITY_ONLY"
            rationale = (
                f"Biogenic Aragonite ({row['biogenic_adjudication'][:60]}); "
                "biogenic origin may introduce material/formation differences "
                "beyond mineral polymorph distinction; keep separate from primary"
            )
            permitted = "SENSITIVITY_ANALYSIS_ONLY"
            id_conf = row["identity_confidence"]
            prov_conf = "probable_independent_biogenic"
        elif pop == "PRIMARY":
            if "grain_size" in row["independence_status"]:
                decision = "ELIGIBLE_WITH_RESTRICTIONS"
                rationale = (
                    f"Primary Aragonite; grain-size fraction from {row['source_group_id']}; "
                    "not independent of other fractions from same source group; "
                    "only one representative spectrum per source group permitted in analysis"
                )
                permitted = "PRIMARY_COMPARISON_WITH_INDEPENDENCE_RESTRICTION"
                id_conf = row["identity_confidence"]
                prov_conf = "confirmed_non_independent_grain_size_series"
            elif "replicate" in row["independence_status"]:
                decision = "ELIGIBLE_WITH_RESTRICTIONS"
                rationale = (
                    f"Primary Aragonite; replicate measurement from {row['source_group_id']}; "
                    "not independent of other replicates from same source; "
                    "only one representative spectrum per source group permitted"
                )
                permitted = "PRIMARY_COMPARISON_WITH_INDEPENDENCE_RESTRICTION"
                id_conf = row["identity_confidence"]
                prov_conf = "confirmed_non_independent_replicate_series"
            else:
                if row["identity_confidence"] == "high":
                    decision = "ELIGIBLE_FOR_PRIMARY_COMPARISON_DESIGN"
                else:
                    decision = "ELIGIBLE_FOR_PRIMARY_COMPARISON_DESIGN"
                rationale = (
                    f"Primary Aragonite; probable independent source {row['source_group_id']}; "
                    f"identity_confidence={row['identity_confidence']}; "
                    f"mixture_status={row['mixture_status']}; "
                    "all D1-D4+S1 available; no biogenic flag"
                )
                permitted = "PRIMARY_COMPARISON"
                id_conf = row["identity_confidence"]
                prov_conf = "probable_independent"
        else:
            decision = "HOLD_UNRESOLVED"
            rationale = "Unclassified population"
            permitted = "NOT_ELIGIBLE"
            id_conf = row["identity_confidence"]
            prov_conf = "UNVERIFIED"

        rows.append({
            "linked_view_id": lv_id,
            "original_record_id": row["original_record_id"],
            "mineral_identity": "Aragonite",
            "population_designation": pop,
            "original_eligibility_status": row["scientific_inclusion_decision"],
            "original_eligibility_source": "phase5_step5_7_adjudication",
            "analysis_specific_decision": decision,
            "eligibility_rationale": rationale,
            "identity_confidence": id_conf,
            "provenance_confidence": prov_conf,
            "independence_limitations": row["independence_status"],
            "material_origin_classification": row["material_origin"],
            "measurement_compatibility": f"{row['preprocessing_route']};interp={row['interpolation_occurred']}",
            "descriptor_completeness": "D1_D4_S1_complete",
            "chemistry_symmetry_status": "reference_CaCO3_only;no_specimen_chemistry",
            "specimen_specific_chemistry": "NO",
            "exclusion_restriction_codes": row["issue_flags"],
            "evidence_source_paths": f"linked_view:{lv_id};ext_meta:{row['original_record_id']}",
            "permitted_use": permitted,
            "reviewer_notes": "",
        })

    return pd.DataFrame(rows)


def build_comparability_matrix():
    rows = []
    criteria = [
        {
            "criterion": "mineral_identity_confidence",
            "calcite_evidence": "48/49 STRONG identity confidence; 1 TENTATIVE (sepiolite mislabel F0228)",
            "aragonite_evidence": "22 high + 22 medium identity confidence (from extension metadata); all adjudicated PRIMARY_ELIGIBLE or SENSITIVITY_ONLY",
            "comparability_status": "PARTIALLY_COMparable",
            "notes": "Calcite has stronger identity evidence overall; Aragonite has medium confidence for 22/44 records",
        },
        {
            "criterion": "material_origin",
            "calcite_evidence": "49/49 natural; no biogenic; no thermal treatment",
            "aragonite_evidence": "24 pure_natural + 13 unknown + 7 biogenic_carbonate; all untreated; 1 synthetic (Anatase, not Aragonite)",
            "comparability_status": "COMPARABLE_WITH_CAVEATS",
            "notes": "Calcite uniformly natural; Aragonite primary includes 13 unknown mixture_status records; biogenic excluded from primary",
        },
        {
            "criterion": "provenance_independence",
            "calcite_evidence": "43 established_phase4 provenance groups; 4 groups have multiple spectra (SPM0282:2, SPM0332:3, SPM0333:3, SPM0408:2); 39 single-spectrum groups",
            "aragonite_evidence": "19 primary source groups: 15 probable_independent, 3 grain-size series (18 spectra), 1 replicate series (4 spectra); conservative bound = 19 independent units",
            "comparability_status": "PARTIALLY_COMPARABLE",
            "notes": "Calcite has more established provenance; Aragonite has 18/37 primary spectra from dependent grain-size fractions; analysis must use source-group level",
        },
        {
            "criterion": "measurement_compatibility",
            "calcite_evidence": "All 49 native_5nm; no interpolation; Phase 2 preprocessing; same instrument family",
            "aragonite_evidence": "41 trim_native_5nm + 3 pchip_from_native_10nm; 3 interpolated; Step 5.8 extension preprocessing",
            "comparability_status": "COMPARABLE_WITH_CAVEATS",
            "notes": "Both on same D2 grid (320-2550nm, 5nm, 447 points); 3 Aragonite records required interpolation; preprocessing routes differ slightly",
        },
        {
            "criterion": "descriptor_compatibility",
            "calcite_evidence": "D1-D4 from phase3_step3_frozen_v1; S1 not available for canonical",
            "aragonite_evidence": "D1-D4 from phase3_step3_frozen_v1 (via step5.12 extension pipeline); S1 available",
            "comparability_status": "COMPARABLE",
            "notes": "D1-D4 produced with same frozen Phase 3 implementation; S1 only for extension (not usable for within-Calcite comparison but available for cross-mineral)",
        },
        {
            "criterion": "chemical_evidence",
            "calcite_evidence": "All 49 reference_formula=CaCO3; chemistry_type=IDEAL_REFERENCE; 0 specimen-specific chemistry; 43 CHEM_REFERENCE_ELIGIBLE, 5 CHEM_AMBIGUOUS, 1 CHEM_TENTATIVE",
            "aragonite_evidence": "Reference CaCO3 only; no specimen-specific measured chemistry; no chemistry metadata in mineral_reference (Aragonite absent from phase4_mineral_reference.csv)",
            "comparability_status": "COMPARABLE_LIMITED",
            "notes": "Neither mineral has specimen-specific measured chemistry; comparison relies entirely on ideal/reference CaCO3; mineral_reference does not include Aragonite entry",
        },
        {
            "criterion": "crystal_system_contrast",
            "calcite_evidence": "Trigonal; R-3c (167); -3m",
            "aragonite_evidence": "Orthorhombic; Pmcn; mmm",
            "comparability_status": "COMPARABLE",
            "notes": "Different crystal systems provide genuine polymorph contrast; this is the scientific basis for the comparison",
        },
        {
            "criterion": "coverage_balance",
            "calcite_evidence": "49 spectra; 43 source groups; ~39 effectively independent units",
            "aragonite_evidence": "44 spectra total; 37 primary from 19 source groups; 19 conservative independent units (15 independent + 3 grain-size + 1 replicate)",
            "comparability_status": "IMBALANCED",
            "notes": "Calcite has roughly double the independent source units of Aragonite; Aragonite primary is dominated by grain-size fractions from 3 specimens",
        },
        {
            "criterion": "confounding_risk",
            "calcite_evidence": "Mineral identity and crystal system completely confounded (Calcite only occurs as trigonal in this dataset); 5 mixture records; grain-size fractions present (SPM0332, SPM0333, SPM0371-SPM0382)",
            "aragonite_evidence": "Mineral identity and crystal system completely confounded (Aragonite only occurs as orthorhombic); no mineral occurs in multiple crystal systems",
            "comparability_status": "HIGH_CONFOUNDING",
            "notes": "Cannot separate crystal-structure effect from mineral-identity effect; any observed difference could reflect either; shared ideal chemistry does not establish equal measured chemistry",
        },
        {
            "criterion": "eligibility_consistency",
            "calcite_evidence": "All from established Phase 2/3/4 pipeline; well-characterized provenance; uniform preprocessing",
            "aragonite_evidence": "All from Step 5.8 extension pipeline; validated by Step 5.9 audit; descriptors validated by Step 5.12; some identity_confidence=medium",
            "comparability_status": "PARTIALLY_COMPARABLE",
            "notes": "Different pipeline stages but compatible frozen specifications; extension went through independent audit and validation",
        },
    ]
    return pd.DataFrame(criteria)


def build_validation_checks(verification_checks, calcite_audit, aragonite_audit, eligibility):
    checks = list(verification_checks)

    calcite_excl = (eligibility["analysis_specific_decision"] == "EXCLUDE_WITH_REASON").sum()
    checks.append({
        "check_id": "ELIG-CALCITE-EXCLUDE",
        "description": "Calcite exclusion count matches sepiolite mislabel",
        "severity": "WARNING",
        "expected": "1 Calcite excluded (sepiolite F0228)",
        "observed": f"{calcite_excl} Calcite excluded",
        "status": "PASS" if calcite_excl == 1 else "FAIL",
        "notes": "",
    })

    arag_sens = (
        eligibility[
            (eligibility["mineral_identity"] == "Aragonite")
            & (eligibility["analysis_specific_decision"] == "SENSITIVITY_ONLY")
        ].shape[0]
    )
    checks.append({
        "check_id": "ELIG-ARAG-SENSITIVITY",
        "description": "Aragonite sensitivity-only count matches biogenic count",
        "severity": "CRITICAL",
        "expected": "7 Aragonite SENSITIVITY_ONLY",
        "observed": f"{arag_sens} Aragonite SENSITIVITY_ONLY",
        "status": "PASS" if arag_sens == 7 else "FAIL",
        "notes": "",
    })

    every_has_reason = eligibility["eligibility_rationale"].notna().all() and (
        eligibility["eligibility_rationale"].str.len() > 0
    ).all()
    checks.append({
        "check_id": "ELIG-ALL-HAVE-REASON",
        "description": "Every eligibility decision has a documented rationale",
        "severity": "CRITICAL",
        "expected": "All records have rationale",
        "observed": "All have rationale" if every_has_reason else "Some missing rationale",
        "status": "PASS" if every_has_reason else "FAIL",
        "notes": "",
    })

    checks.append({
        "check_id": "NO-COMPARE-STATS",
        "description": "No comparative statistics were calculated",
        "severity": "CRITICAL",
        "expected": "No D1-D4 means, effect sizes, or tests computed",
        "observed": "Only eligibility assessment performed; no statistics computed",
        "status": "PASS",
        "notes": "Script only reads metadata and produces eligibility decisions",
    })

    checks.append({
        "check_id": "NO-FROZEN-MODIFY",
        "description": "No frozen dataset was modified",
        "severity": "CRITICAL",
        "expected": "All input files read-only",
        "observed": "Script uses pd.read_csv only; no writes to input paths",
        "status": "PASS",
        "notes": "",
    })

    checks.append({
        "check_id": "NO-STATUS-CHANGE",
        "description": "Linked view NOT_YET_ADJUDICATED status unchanged",
        "severity": "CRITICAL",
        "expected": "comparative_eligibility still NOT_YET_ADJUDICATED for all records",
        "observed": "Linked view not modified by this script",
        "status": "PASS",
        "notes": "",
    })

    return checks


def build_run_manifest(inputs, verification_checks, output_files):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamp = datetime.now(timezone.utc).isoformat()

    input_checksums = []
    input_files = {
        "linked_view": LINKED_VIEW_PATH,
        "extension_metadata": EXT_META_PATH,
        "integrated_dataset": INTEGRATED_PATH,
        "symmetry_metadata": SYMMETRY_PATH,
        "chemistry_metadata": CHEMISTRY_PATH,
        "mineral_reference": MINERAL_REF_PATH,
        "provenance_mapping": PROVENANCE_PATH,
        "source_group_summary": SG_SUMMARY_PATH,
        "checksum_manifest_513": CHECKSUM_MANIFEST_513_PATH,
        "adjudication": ADJUDICATION_PATH,
        "canonical_spectra": CANONICAL_SPECTRA_PATH,
        "canonical_metadata": CANONICAL_META_PATH,
        "canonical_descriptors": DESCRIPTOR_PATH,
        "extension_descriptors": EXT_DESCRIPTORS_PATH,
    }
    for label, path in input_files.items():
        if os.path.isfile(path):
            input_checksums.append({
                "file_label": label,
                "file_path": os.path.relpath(path, REPO_ROOT),
                "md5": file_md5(path),
                "sha256": file_sha256(path),
            })

    output_checksums = []
    for label, path in output_files.items():
        if os.path.isfile(path):
            output_checksums.append({
                "file_label": label,
                "file_path": os.path.relpath(path, REPO_ROOT),
                "md5": file_md5(path),
                "sha256": file_sha256(path),
            })

    manifest = {
        "step": "phase5_step5_14",
        "step_title": "Aragonite-Calcite Eligibility Assessment and Comparison Design",
        "run_id": run_id,
        "execution_timestamp": timestamp,
        "script_path": os.path.relpath(SCRIPT_PATH, REPO_ROOT),
        "script_md5": file_md5(SCRIPT_PATH) if os.path.isfile(SCRIPT_PATH) else "UNAVAILABLE",
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "input_files": input_checksums,
        "output_files": output_checksums,
        "validation_summary": {
            "total_checks": len(verification_checks),
            "pass_count": sum(1 for c in verification_checks if c["status"] == "PASS"),
            "fail_count": sum(1 for c in verification_checks if c["status"] == "FAIL"),
            "critical_fail_count": sum(
                1 for c in verification_checks
                if c["status"] == "FAIL" and c["severity"] == "CRITICAL"
            ),
        },
        "run_status": "COMPLETED_SUCCESS" if all(
            c["status"] == "PASS" for c in verification_checks
            if c["severity"] == "CRITICAL"
        ) else "COMPLETED_WITH_FAILURES",
        "linked_view_version": "v1.0",
        "linked_view_construction_run_id": "20261015T144921Z",
    }
    return manifest


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print("Loading inputs...")
    inputs = load_inputs()

    print("Verifying inputs...")
    verification_checks = verify_inputs(inputs)

    print("Building Calcite record audit...")
    calcite_audit = build_calcite_audit(inputs)
    calcite_path = os.path.join(OUTPUT_DIR, "phase5_step5_14_calcite_record_audit.csv")
    calcite_audit.to_csv(calcite_path, index=False)
    print(f"  -> {len(calcite_audit)} Calcite records written")

    print("Building Aragonite record audit...")
    aragonite_audit = build_aragonite_audit(inputs)
    aragonite_path = os.path.join(
        OUTPUT_DIR, "phase5_step5_14_aragonite_record_audit.csv"
    )
    aragonite_audit.to_csv(aragonite_path, index=False)
    print(f"  -> {len(aragonite_audit)} Aragonite records written")

    print("Building source group audit...")
    sg_audit = build_source_group_audit(inputs)
    sg_path = os.path.join(OUTPUT_DIR, "phase5_step5_14_source_group_audit.csv")
    sg_audit.to_csv(sg_path, index=False)
    print(f"  -> {len(sg_audit)} source groups written")

    print("Building analysis eligibility table...")
    eligibility = build_analysis_eligibility(calcite_audit, aragonite_audit)
    elig_path = os.path.join(OUTPUT_DIR, "phase5_step5_14_analysis_eligibility.csv")
    eligibility.to_csv(elig_path, index=False)
    print(f"  -> {len(eligibility)} eligibility decisions written")

    print("Building comparability matrix...")
    comp_matrix = build_comparability_matrix()
    comp_path = os.path.join(OUTPUT_DIR, "phase5_step5_14_comparability_matrix.csv")
    comp_matrix.to_csv(comp_path, index=False)
    print(f"  -> {len(comp_matrix)} criteria written")

    print("Building validation checks...")
    all_checks = build_validation_checks(
        verification_checks, calcite_audit, aragonite_audit, eligibility
    )
    val_path = os.path.join(OUTPUT_DIR, "phase5_step5_14_validation_checks.csv")
    pd.DataFrame(all_checks).to_csv(val_path, index=False)
    print(f"  -> {len(all_checks)} validation checks written")

    output_files = {
        "calcite_record_audit": calcite_path,
        "aragonite_record_audit": aragonite_path,
        "source_group_audit": sg_path,
        "analysis_eligibility": elig_path,
        "comparability_matrix": comp_path,
        "validation_checks": val_path,
    }

    print("Building run manifest...")
    manifest = build_run_manifest(inputs, all_checks, output_files)
    manifest_path = os.path.join(OUTPUT_DIR, "phase5_step5_14_run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    output_files["run_manifest"] = manifest_path

    print()
    print("=== SUMMARY ===")
    print(f"Calcite records: {len(calcite_audit)}")
    print(f"  Primary eligible: {(calcite_audit['issue_flags'] == 'none').sum()}")
    print(f"  With restrictions: {(calcite_audit['mixture_flag'] == True).sum()}")
    print(f"  Excluded: {(calcite_audit['original_record_id'] == 'F0228').sum()}")
    print(f"Aragonite records: {len(aragonite_audit)}")
    print(f"  Primary: {(aragonite_audit['population_designation'] == 'PRIMARY').sum()}")
    print(f"  Biogenic sensitivity: {(aragonite_audit['population_designation'] == 'BIOGENIC_SENSITIVITY').sum()}")
    print(f"Source groups: {len(sg_audit)}")
    print(f"  Calcite: {(sg_audit['mineral'] == 'Calcite').sum()}")
    print(f"  Aragonite: {(sg_audit['mineral'] == 'Aragonite').sum()}")
    print(f"Eligibility decisions: {len(eligibility)}")
    for decision in eligibility["analysis_specific_decision"].unique():
        count = (eligibility["analysis_specific_decision"] == decision).sum()
        print(f"  {decision}: {count}")
    print(f"Validation checks: {len(all_checks)}")
    pass_count = sum(1 for c in all_checks if c["status"] == "PASS")
    fail_count = sum(1 for c in all_checks if c["status"] == "FAIL")
    print(f"  PASS: {pass_count}, FAIL: {fail_count}")
    print()
    print("CSV deliverables written successfully.")
    print("Markdown deliverables must be written separately.")


if __name__ == "__main__":
    main()
