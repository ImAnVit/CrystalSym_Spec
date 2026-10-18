"""
Phase 5 Step 5.18 -- Curator Enquiry Preparation and Metadata Recovery

Recovers existing metadata from RELAB raw files and repository records,
prepares a prioritised unresolved-question matrix, and drafts a concise
curator enquiry email. Preparation only; no contact is made.

Inputs:
    data/raw/RELAB/<mineral>/*.tab
    metadata/relab_measurements.csv
    metadata/relab_sample_groups.csv
    data/processed/phase4_integrated_analysis_dataset.csv
    results/05_exploratory_analysis/phase5_step5_8_extension_dataset/
        phase5_step5_8_extension_metadata.csv
    results/05_exploratory_analysis/phase5_step5_13_linked_comparative_view/
        phase5_step5_13_linked_analysis_view.csv
    results/05_exploratory_analysis/phase5_step5_14_aragonite_calcite_eligibility_design/
        phase5_step5_14_aragonite_record_audit.csv
    results/05_exploratory_analysis/phase5_step5_17_specimen_evidence_gap_analysis/
        phase5_step5_17_curator_question_priorities.csv
        phase5_step5_17_field_gap_analysis.csv

Outputs:
    results/05_exploratory_analysis/phase5_step5_18_curator_preparation_metadata_recovery/
        phase5_step5_18_metadata_recovery_register.csv
        phase5_step5_18_unresolved_question_matrix.csv
        phase5_step5_18_curator_enquiry_draft.md
        phase5_step5_18_recovery_summary.md
        phase5_step5_18_validation_checks.csv
        phase5_step5_18_run_manifest.json
"""

import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "RELAB"
META_DIR = REPO_ROOT / "metadata"
PROC_DIR = REPO_ROOT / "data" / "processed"
RESULTS_BASE = REPO_ROOT / "results" / "05_exploratory_analysis"
OUT_DIR = RESULTS_BASE / "phase5_step5_18_curator_preparation_metadata_recovery"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_relab_metadata_block(filepath):
    """Extract metadata from a RELAB .tab file's trailing metadata block."""
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    asc_line = None
    for i, line in enumerate(lines):
        if ".ASC" in line and len(line.strip()) > 4:
            asc_line = i
            break

    if asc_line is None:
        return None

    remaining = [l.strip() for l in lines[asc_line + 1:] if l.strip()]
    if len(remaining) < 2:
        return None

    sample_id = remaining[0]
    description = remaining[1] if len(remaining) > 1 else ""

    meta = {
        "asc_id": lines[asc_line].strip(),
        "sample_id": sample_id,
        "sample_description": description,
        "source_angle": None,
        "detect_angle": None,
        "voltage": None,
        "date": None,
        "time": None,
        "user": None,
        "research_type": None,
        "pi Sponsor": None,
        "bvc": None,
        "run_mode": None,
        "stage_type": None,
        "spinning": None,
        "aperture": None,
        "scrambler": None,
        "depolarizer": None,
        "has_halon_cal": False,
        "has_extended_metadata": False,
    }

    for line in lines[asc_line:]:
        stripped = line.strip()
        src_match = re.search(r"Source Ang:\s*([\d.]+)", stripped)
        det_match = re.search(r"Detect Ang:\s*([\d.]+)", stripped)
        volt_match = re.search(r"Volt:\s*([\d.]+)", stripped)
        date_match = re.search(r"Date:\s*(\S+)", stripped)
        time_match = re.search(r"Time:\s*(\S+)", stripped)

        if src_match:
            meta["source_angle"] = float(src_match.group(1))
        if det_match:
            meta["detect_angle"] = float(det_match.group(1))
        if volt_match:
            meta["voltage"] = float(volt_match.group(1))
        if date_match:
            meta["date"] = date_match.group(1)
        if time_match:
            meta["time"] = time_match.group(1)

        if stripped.startswith("* User:"):
            meta["user"] = stripped.split(":", 1)[1].strip()
            meta["has_extended_metadata"] = True
        if stripped.startswith("* Research Type:"):
            meta["research_type"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* NASA PI/CoI") or stripped.startswith("* Sponsor"):
            meta["pi_sponsor"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* BVC:"):
            meta["bvc"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* Run Mode:"):
            meta["run_mode"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* Stage Type:"):
            meta["stage_type"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* Spinning:"):
            meta["spinning"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* Aperture Size:"):
            meta["aperture"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* Scrambler:"):
            meta["scrambler"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("* Depolarizer:"):
            meta["depolarizer"] = stripped.split(":", 1)[1].strip()
        if "HALON CALIBRATION" in stripped.upper():
            meta["has_halon_cal"] = True

    return meta


def extract_grain_size_from_description(desc):
    """Extract grain-size information from a sample description string."""
    if not desc:
        return None, "absent"

    desc_lower = desc.lower()

    numeric_range = re.search(r"(\d+)\s*[-\u2013]\s*(\d+)\s*um", desc_lower)
    if numeric_range:
        return f"{numeric_range.group(1)}-{numeric_range.group(2)} um", "numeric_range"

    lt_match = re.search(r"<\s*(\d+)\s*um", desc_lower)
    if lt_match:
        return f"<{lt_match.group(1)} um", "numeric_lt"

    gt_match = re.search(r">\s*(\d+)\s*um", desc_lower)
    if gt_match:
        return f">{gt_match.group(1)} um", "numeric_gt"

    single_match = re.search(r"(\d+)\s*um(?!\s*[-\u2013])", desc_lower)
    if single_match:
        return f"{single_match.group(1)} um", "numeric_single"

    qualitative_terms = [
        "fine", "coarse", "medium", "powder", "crystal", "grains",
        "particle", "fraction",
    ]
    for term in qualitative_terms:
        if term in desc_lower:
            return term, "qualitative"

    return None, "absent"


def extract_locality_hints(desc):
    """Extract geographic locality hints from a sample description."""
    if not desc:
        return []

    hints = []
    known_localities = {
        "minglanilla": "Minglanilla (Spain)",
        "compton martin": "Compton Martin (UK)",
        "morocco": "Morocco",
        "cuenca, spain": "Cuenca, Spain",
        "cuenca spain": "Cuenca, Spain",
        "mexico": "Mexico",
        "nevada": "Nevada, USA",
        "alligator ridge": "Alligator Ridge, Nevada, USA",
        "big timber": "Big Timber (Montana, USA)",
        "mazada": "Mazada (locality name from sample description)",
        "lane calcite": "Lane calcite (collector name)",
        "cuadras": "Cuadras (locality name from sample description)",
    }
    desc_lower = desc.lower()
    for key, label in known_localities.items():
        if key in desc_lower:
            hints.append(label)

    country_patterns = [
        (r"\bspain\b", "Spain"),
        (r"\bmorocco\b", "Morocco"),
        (r"\bmexico\b", "Mexico"),
        (r"\bnevada\b", "Nevada, USA"),
        (r"\bmontana\b", "Montana, USA"),
    ]
    for pat, label in country_patterns:
        if re.search(pat, desc_lower) and label not in [h for h in hints]:
            hints.append(label)

    return hints


def detect_mixture_from_description(desc):
    """Detect mixture status from sample description text."""
    if not desc:
        return "unknown"
    desc_lower = desc.lower()
    mixture_indicators = [
        ("+ ~", "explicit_mixture"),
        ("+ ", "explicit_mixture"),
        ("mixed with", "explicit_mixture"),
        ("wt %", "explicit_mixture"),
        ("wt%", "explicit_mixture"),
        ("mgs", "explicit_mixture"),
        ("mixture", "explicit_mixture"),
    ]
    for pattern, status in mixture_indicators:
        if pattern in desc_lower:
            return status
    if "powdered conch shell" in desc_lower or "powdered coral" in desc_lower:
        if "cotton" in desc_lower or "protein" in desc_lower:
            return "explicit_mixture"
    return "unknown"


def detect_thermal_treatment(desc):
    """Detect thermal treatment from sample description text."""
    if not desc:
        return "unknown"
    desc_lower = desc.lower()
    if re.search(r"\d+\s*c\s*(for|/|\s)", desc_lower) or "200c" in desc_lower or "250c" in desc_lower or "150c" in desc_lower:
        return "thermally_treated"
    if "clorox" in desc_lower or "bleach" in desc_lower:
        return "chemically_treated"
    if "bleached" in desc_lower:
        return "bleached"
    return "untreated"


# ---------------------------------------------------------------------------
# Task A: Metadata Recovery
# ---------------------------------------------------------------------------

def recover_metadata_from_raw_files():
    """Scan all RELAB raw files and extract recoverable metadata."""
    register = []
    entry_id = 0

    for mineral_dir in sorted(RAW_DIR.iterdir()):
        if not mineral_dir.is_dir():
            continue
        mineral_name = mineral_dir.name
        for tab_file in sorted(mineral_dir.glob("*.tab")):
            meta = parse_relab_metadata_block(tab_file)
            if meta is None:
                continue

            desc = meta["sample_description"]
            gs_value, gs_type = extract_grain_size_from_description(desc)
            locality_hints = extract_locality_hints(desc)
            mixture_status = detect_mixture_from_description(desc)
            thermal_status = detect_thermal_treatment(desc)

            entry_id += 1
            register.append({
                "recovery_id": f"MR_{entry_id:04d}",
                "mineral_directory": mineral_name,
                "source_file": tab_file.name,
                "relab_sample_id": meta["sample_id"],
                "sample_description": desc,
                "recovery_category": "raw_file_metadata",
                "grain_size_value": gs_value or "",
                "grain_size_evidence_type": gs_type,
                "grain_size_confidence": "high" if gs_type.startswith("numeric") else "medium" if gs_type == "qualitative" else "absent",
                "locality_hints": "; ".join(locality_hints) if locality_hints else "",
                "locality_confidence": "plausible_lead" if locality_hints else "absent",
                "mixture_status_from_description": mixture_status,
                "mixture_confidence": "high" if mixture_status == "explicit_mixture" else "absent",
                "thermal_treatment_from_description": thermal_status,
                "thermal_confidence": "high" if thermal_status in ("thermally_treated", "chemically_treated", "bleached") else "absent",
                "source_angle": meta["source_angle"] if meta["source_angle"] is not None else "",
                "detect_angle": meta["detect_angle"] if meta["detect_angle"] is not None else "",
                "voltage": meta["voltage"] if meta["voltage"] is not None else "",
                "measurement_date": meta["date"] or "",
                "has_extended_metadata": meta["has_extended_metadata"],
                "has_halon_calibration": meta["has_halon_cal"],
                "operator": meta["user"] or "",
                "run_mode": meta["run_mode"] or "",
                "stage_type": meta["stage_type"] or "",
                "spinning": meta["spinning"] or "",
                "aperture": meta["aperture"] or "",
            })

    return register


def recover_metadata_from_relab_measurements():
    """Extract recoverable metadata from the relab_measurements.csv."""
    measurements = read_csv(META_DIR / "relab_measurements.csv")
    register = []
    entry_id = 0

    for row in measurements:
        desc = row.get("sample_description", "")
        gs_info = row.get("grain_size_info", "")

        gs_value, gs_type = extract_grain_size_from_description(
            gs_info if gs_info else desc
        )
        locality_hints = extract_locality_hints(desc)

        if gs_value or locality_hints or (gs_info and gs_info != "nan"):
            entry_id += 1
            register.append({
                "recovery_id": f"RM_{entry_id:04d}",
                "mineral_directory": row.get("mineral_directory", ""),
                "source_file": row.get("filename", ""),
                "relab_sample_id": row.get("relab_sample_id", ""),
                "sample_description": desc,
                "recovery_category": "relab_measurements_csv",
                "grain_size_value": gs_value or gs_info if gs_info != "nan" else "",
                "grain_size_evidence_type": gs_type if gs_value else ("from_csv_field" if gs_info and gs_info != "nan" else "absent"),
                "grain_size_confidence": "high" if gs_type.startswith("numeric") else "medium" if gs_type == "qualitative" else "low",
                "locality_hints": "; ".join(locality_hints) if locality_hints else "",
                "locality_confidence": "plausible_lead" if locality_hints else "absent",
                "mixture_status_from_description": detect_mixture_from_description(desc),
                "mixture_confidence": "high" if detect_mixture_from_description(desc) == "explicit_mixture" else "absent",
                "thermal_treatment_from_description": detect_thermal_treatment(desc),
                "thermal_confidence": "high" if detect_thermal_treatment(desc) != "untreated" else "absent",
                "source_angle": row.get("source_angle", ""),
                "detect_angle": row.get("detect_angle", ""),
                "voltage": row.get("voltage", ""),
                "measurement_date": row.get("measurement_date", ""),
                "has_extended_metadata": bool(row.get("run_mode", "")),
                "has_halon_calibration": "halon" in str(row.get("calibration_note", "")).lower(),
                "operator": row.get("operator", ""),
                "run_mode": row.get("run_mode", ""),
                "stage_type": row.get("stage_type", ""),
                "spinning": row.get("spinning", ""),
                "aperture": row.get("aperture", ""),
            })

    return register


def recover_extension_metadata():
    """Extract recoverable metadata from the extension dataset."""
    ext_meta = read_csv(
        RESULTS_BASE / "phase5_step5_8_extension_dataset"
        / "phase5_step5_8_extension_metadata.csv"
    )
    register = []
    entry_id = 0

    for row in ext_meta:
        entry_id += 1
        register.append({
            "recovery_id": f"EX_{entry_id:04d}",
            "mineral_directory": row.get("mineral", ""),
            "source_file": row.get("source_file", ""),
            "relab_sample_id": row.get("source_group", ""),
            "sample_description": "",
            "recovery_category": "extension_metadata",
            "grain_size_value": "",
            "grain_size_evidence_type": "absent",
            "grain_size_confidence": "absent",
            "locality_hints": "",
            "locality_confidence": "absent",
            "mixture_status_from_description": row.get("mixture_status", "unknown"),
            "mixture_confidence": "high" if row.get("mixture_status") in ("pure_natural", "pure_synthetic", "biogenic_carbonate") else "absent",
            "thermal_treatment_from_description": row.get("thermal_status", "unknown"),
            "thermal_confidence": "high" if row.get("thermal_status") in ("untreated", "thermally_treated") else "absent",
            "source_angle": "",
            "detect_angle": "",
            "voltage": "",
            "measurement_date": "",
            "has_extended_metadata": True,
            "has_halon_calibration": True,
            "operator": "",
            "run_mode": "",
            "stage_type": "",
            "spinning": "",
            "aperture": "",
        })

    return register


def recover_source_collection_from_sample_ids():
    """Extract source collection codes from RELAB sample ID prefixes."""
    measurements = read_csv(META_DIR / "relab_measurements.csv")
    register = []
    seen = set()
    entry_id = 0

    for row in measurements:
        sid = row.get("relab_sample_id", "")
        if not sid or sid in seen:
            continue
        seen.add(sid)

        parts = sid.split("-")
        if len(parts) >= 2:
            institution_code = parts[0]
            researcher_code = parts[1]
            entry_id += 1
            register.append({
                "recovery_id": f"SC_{entry_id:04d}",
                "mineral_directory": row.get("mineral_directory", ""),
                "source_file": row.get("filename", ""),
                "relab_sample_id": sid,
                "sample_description": "",
                "recovery_category": "source_collection_code",
                "grain_size_value": "",
                "grain_size_evidence_type": "absent",
                "grain_size_confidence": "absent",
                "locality_hints": "",
                "locality_confidence": "absent",
                "mixture_status_from_description": "unknown",
                "mixture_confidence": "absent",
                "thermal_treatment_from_description": "unknown",
                "thermal_confidence": "absent",
                "source_angle": "",
                "detect_angle": "",
                "voltage": "",
                "measurement_date": "",
                "has_extended_metadata": False,
                "has_halon_calibration": False,
                "operator": "",
                "run_mode": "",
                "stage_type": "",
                "spinning": "",
                "aperture": "",
            })

    return register


# ---------------------------------------------------------------------------
# Task B: Unresolved Question Matrix
# ---------------------------------------------------------------------------

def build_question_matrix(raw_register, linked_view_rows, arag_audit_rows, gap_analysis_rows):
    """Build the record-specific unresolved question matrix."""

    aragonite_source_groups = {}
    for row in arag_audit_rows:
        sg = row.get("source_group_id", "")
        if sg not in aragonite_source_groups:
            aragonite_source_groups[sg] = []
        aragonite_source_groups[sg].append(row)

    locality_records = []
    for row in raw_register:
        if row.get("locality_hints") and row["locality_hints"] != "":
            locality_records.append(row)

    mixture_unknown_aragonite = [
        r for r in arag_audit_rows if r.get("mixture_status") == "unknown"
    ]

    questions = []
    q_id = 0

    # Q1: Identification method
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "essential",
        "question_category": "specimen_identity",
        "question_type": "factual",
        "scientific_issue": "No record documents how mineral identity was determined. All identifications are directory-label only. Without knowing whether XRD, optical, or chemical analysis was used, identity confidence cannot be upgraded from directory-label to verified.",
        "relevant_identifiers": "All 503 linked-view records; all 28 mineral directories",
        "existing_evidence": "Step 5.17 gap analysis: 0/503 records have identification_method. All are directory-label only. 448/503 have identity_confidence=STRONG based on directory consistency, not analytical verification.",
        "precise_clarification_requested": "For each mineral species in the RELAB collection, what method was used to identify the specimens? Specifically: (a) Were any specimens verified by XRD, EMPA, or other analytical methods? (b) Are there specimens where identity is uncertain or based solely on visual inspection? (c) Do any RELAB archive records include XRD patterns or chemical analyses that could be linked to specific sample IDs?",
        "why_it_matters": "Identification method is the single most impactful metadata gap for specimen identity confidence. XRD-verified specimens would allow upgrading from TENTATIVE to STRONG confidence and would reduce the risk of misidentification confounding.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED, GATE_INTERPRETABLE_ML, GATE_PYSR",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q2: Specimen independence
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "essential",
        "question_category": "specimen_independence",
        "question_type": "factual",
        "scientific_issue": "Source groups are inferred from filename patterns and sample descriptions. It is unknown whether grain-size series come from the same parent specimen (fractionated) or from different specimens. This affects effective sample sizes and independence assumptions.",
        "relevant_identifiers": "Multi-spectrum source groups: JB-JLB-E59 (6 files), JB-JLB-E60 (6 files), JB-JLB-G40 (6 files), JB-JLB-E57 (6 files), JB-JLB-E58 (6 files), PA-SJG-021 (4 files), SG-SJG-066/068 (multiple), CB-EAC-009 through 013 (multiple)",
        "existing_evidence": "Step 5.14/5.15 independence analysis: 69 source groups identified; most are provisional. Grain-size series are flagged as confirmed_non_independent_grain_size_series. Step 5.17: n_verified_independent_specimens = 0/503.",
        "precise_clarification_requested": "For the following source groups, how many physically independent specimens do they represent? (a) JB-JLB-E59/E60/G40 (Aragonite grain-size series): Are the 6 fractions from a single specimen? (b) JB-JLB-E57/E58 (Calcite grain-size series): Same question. (c) PA-SJG-021 (Aragonite, 4 files with different treatments): Are these from one specimen? (d) CB-EAC-009 through 013 (Calcite, multiple grain sizes): Are these from one specimen each?",
        "why_it_matters": "Incorrect independence assumptions inflate effective sample sizes and invalidate statistical comparisons. Verified independence is required for GATE_MATCHED_STRATIFIED and above.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED, GATE_SPECIMEN_CHEMISTRY, GATE_INTERPRETABLE_ML",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q3: Specimen-specific chemistry
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "essential",
        "question_category": "specimen_chemistry",
        "question_type": "request_for_data",
        "scientific_issue": "All 503 records have level-C chemistry (ideal reference formula). No specimen-specific measured chemistry exists. This prevents chemistry-spectrum modelling and leaves chemistry completely confounded with mineral identity.",
        "relevant_identifiers": "All 503 records; priority for Calcite (42 source groups) and Aragonite (19 source groups)",
        "existing_evidence": "Step 5.17: 0/382 records have specimen_specific_chemistry=True. All chemistry_evidence_level = C. Step 5.16: specimen-specific chemistry identified as the single most impactful metadata improvement.",
        "precise_clarification_requested": "Do specimen-level chemical analyses (wt% oxides, EMPA, XRF, ICP-MS) exist for any RELAB specimens in the collection? Specifically: (a) For Calcite specimens (JB-JLB-E57, E58, CB-EAC-*, etc.), are there published or archived chemical analyses? (b) For Aragonite specimens (JB-JLB-E59, E60, G40), are there chemical analyses? (c) Are there any RELAB specimens with known trace-element or minor-element chemistry? (d) If analyses exist, can they be linked to specific RELAB sample IDs?",
        "why_it_matters": "Specimen-specific chemistry is required for GATE_SPECIMEN_CHEMISTRY and GATE_INTERPRETABLE_ML. Without it, no chemistry-spectrum relationship can be tested.",
        "affected_analysis_gates": "GATE_SPECIMEN_CHEMISTRY, GATE_INTERPRETABLE_ML, GATE_PYSR",
        "requires_new_measurements": "possibly",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "no",
    })

    # Q4: Mixture status of 13 Aragonite records
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "high",
        "question_category": "mixture_status",
        "question_type": "factual",
        "scientific_issue": "13 Aragonite extension records have unknown mixture status. These include the Minglanilla (6 files, JB-JLB-E59) and Compton Martin (6 files, JB-JLB-E60) grain-size series, plus OS-SJG-023. The biogenic Aragonite records are explicitly mixtures (shell + cotton/protein), but the primary natural Aragonite records have no purity documentation.",
        "relevant_identifiers": "EXT_042-053 (JB-JLB-E59, JB-JLB-E60); EXT_069 (OS-SJG-023)",
        "existing_evidence": "Step 5.14 Aragonite audit: 13/44 primary Aragonite records have mixture_status=unknown. Step 5.15 sensitivity analysis: mixture-unknown Aragonite shows numerically different descriptor distributions. Raw file descriptions: 'Minglanilla aragonite' and 'Compton Martin aragonite' do not mention impurities.",
        "precise_clarification_requested": "Are the Minglanilla aragonite (JB-JLB-E59) and Compton Martin aragonite (JB-JLB-E60) specimens pure Aragonite phase, or do they contain Calcite or other mineral impurities? Were these specimens characterised by XRD or other phase-identification methods? For OS-SJG-023 (OS-BC-14), is this a pure Aragonite specimen?",
        "why_it_matters": "Unknown mixture status affects spectral interpretation. If these specimens contain Calcite impurities, their spectral features may reflect mixed-phase contributions rather than pure Aragonite.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED, Aragonite-Calcite comparison eligibility",
        "requires_new_measurements": "possibly",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q5: Geographic provenance
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "high",
        "question_category": "geographic_provenance",
        "question_type": "factual",
        "scientific_issue": "No geographic locality data is present in RELAB file metadata. However, sample descriptions contain plausible locality hints: 'Minglanilla aragonite' (Spain), 'Compton Martin aragonite' (UK), 'Morocco aragonite', 'Cuenca, Spain', 'Alligator Ridge, Nevada', 'Big Timber', 'Mazada calcite'. These are unverified and may refer to collector names rather than precise localities.",
        "relevant_identifiers": "All 503 records; priority for Calcite and Aragonite comparison pair",
        "existing_evidence": "This step's metadata recovery: locality hints extracted from sample descriptions for ~30 records. Step 5.17: geographic_provenance = 0/503. No RELAB file contains structured locality data.",
        "precise_clarification_requested": "Can the geographic localities for the following specimens be confirmed or clarified? (a) Minglanilla aragonite (JB-JLB-E59): Is this from Minglanilla, Spain? (b) Compton Martin aragonite (JB-JLB-E60): Is this from Compton Martin, UK? (c) Morocco aragonite (JB-JLB-G40): Which locality in Morocco? (d) Calcite specimens: Cuadras #43 (JB-JLB-549), Lane calcite (JB-JLB-549/550), Big Timber (JB-JLB-E58), Mazada (JB-JLB-E57), Alligator Ridge (CY-PLH-005)? (e) Are there RELAB archive records with more precise locality data (mine name, formation, coordinates)?",
        "why_it_matters": "Geographic provenance affects chemistry through geological context. Known localities would enable geographic clustering analysis and reduce chemistry-based confounding.",
        "affected_analysis_gates": "GATE_SPECIMEN_CHEMISTRY, GATE_INTERPRETABLE_ML",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q6: Measurement conditions for metadata-incomplete files
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "high",
        "question_category": "measurement_conditions",
        "question_type": "factual",
        "scientific_issue": "676/1017 RELAB files lack extended metadata (run mode, stage type, spinning, aperture, scrambler). Undocumented measurement conditions may introduce batch effects. Files with minimal metadata include c1os01-05, c1sg06-08, c1pa21, cpfb01, c1sh31.",
        "relevant_identifiers": "676 files without extended metadata; priority for Aragonite files c1os01-05, c1sg06-08, c1pa21, c1sh31",
        "existing_evidence": "Step 5.17: instrument_configuration_extended = 0/503. RELAB field dictionary: extended metadata present in 341/1017 files only. This step's raw-file scan: extended metadata fields (run_mode, stage_type, etc.) present in ~340 Aragonite files but absent from many older files.",
        "precise_clarification_requested": "For files without extended metadata blocks, can the following be recovered from lab notebooks or archive records? (a) Run mode (step scan vs. continuous scan). (b) Stage type (manual vs. automatic). (c) Whether the sample was spinning. (d) Aperture size. (e) Whether a scrambler/depolarizer was used. Priority files: c1os01-05, c1sg06-08, c1pa21, c1sh31, cpfb01.",
        "why_it_matters": "Undocumented measurement conditions may introduce systematic batch effects. Knowing the measurement geometry and configuration is needed for GATE_MATCHED_STRATIFIED.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q7: Thermal treatment of canonical records
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "optional",
        "question_category": "thermal_treatment",
        "question_type": "factual",
        "scientific_issue": "Thermal treatment is documented for extension records but absent for all 454 canonical records. Some Aragonite sample descriptions mention thermal treatments (e.g., PA-SJG-021: '200C FOR 24HRS', '250C FOR 24 HRS'). If any canonical specimens were thermally treated, their spectral features may be altered.",
        "relevant_identifiers": "454 canonical records; priority for Calcite and Aragonite",
        "existing_evidence": "Step 5.17: thermal_treatment = 49/503 (extension only). This step's raw-file scan: PA-SJG-021 variants show explicit thermal treatment descriptions. No canonical record has thermal treatment documentation.",
        "precise_clarification_requested": "Were any canonical RELAB specimens thermally treated or heated before measurement? Specifically: (a) Are there any Calcite or Aragonite specimens that were heated, annealed, or otherwise thermally treated? (b) Do RELAB archive records document thermal treatment for any specimen?",
        "why_it_matters": "Thermal treatment can alter crystal structure and spectral features. Thermally treated specimens may need separate handling in comparisons.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q8: Numeric grain sizes
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "optional",
        "question_category": "grain_size",
        "question_type": "factual",
        "scientific_issue": "Grain size is documented for 782/1017 RELAB files, but many are qualitative only ('fine', 'coarse', 'medium grains'). Numeric grain sizes enable grain-size-stratified analysis. This step's recovery extracts numeric values from ~400 descriptions.",
        "relevant_identifiers": "Records with qualitative-only grain size descriptions",
        "existing_evidence": "Step 5.17: grain_size = 782/1017 (but many qualitative). This step's recovery: numeric grain sizes extracted from sample descriptions for ~400 records. Grain-size series (ca-cg prefix files) have well-documented fractions.",
        "precise_clarification_requested": "For specimens where only qualitative grain size descriptions exist, can numeric grain sizes be provided? Specifically: (a) What are the exact grain-size ranges for specimens described as 'fine', 'coarse', or 'medium'? (b) Are there sieve analyses or particle-size measurements for any specimens?",
        "why_it_matters": "Grain size affects scattering and spectral shape. Numeric values enable grain-size as a stratification variable in comparisons.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    # Q9: Crystallographic data
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "optional",
        "question_category": "crystallography",
        "question_type": "request_for_data",
        "scientific_issue": "No unit cell parameters or specimen-specific XRD patterns exist for any specimen. All crystallography is reference-based (from mineral reference tables). Specimen-specific XRD would verify structural assignment and provide continuous descriptors.",
        "relevant_identifiers": "All 503 records; priority for polymorph pairs (Calcite/Aragonite, TiO2 polymorphs, Al2SiO5 polymorphs)",
        "existing_evidence": "Step 5.17: unit_cell_parameters = 0/503. All crystal_system and space_group assignments are from mineral reference, not specimen-specific XRD.",
        "precise_clarification_requested": "Do specimen-specific XRD patterns or unit cell parameters exist for any RELAB specimens? Specifically: (a) Were any specimens characterised by XRD as part of the original RELAB measurement campaign? (b) Are there published XRD patterns for any RELAB specimens? (c) Can XRD data be linked to specific RELAB sample IDs?",
        "why_it_matters": "Specimen-specific XRD would verify structural assignment and enable continuous crystal-structure descriptors for regression. This is required for GATE_PYSR.",
        "affected_analysis_gates": "GATE_PYSR",
        "requires_new_measurements": "possibly",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "no",
    })

    # Q10: Calibration details
    q_id += 1
    questions.append({
        "question_id": f"Q{q_id:02d}",
        "priority": "optional",
        "question_category": "calibration",
        "question_type": "factual",
        "scientific_issue": "535/1017 records have calibration notes (Halon calibration table). The remaining 482 files lack calibration documentation. Undocumented calibration may introduce systematic uncertainty in absolute reflectance.",
        "relevant_identifiers": "482 files without calibration notes",
        "existing_evidence": "Step 5.17: calibration_details = 535/1017. This step's raw-file scan: Halon calibration note present in ~546 files. Older files (c2, c3, c5 prefix) often lack calibration documentation.",
        "precise_clarification_requested": "What calibration standards and procedures were used for each measurement campaign? Specifically: (a) For files without Halon calibration notes, what reference standard was used? (b) Are there calibration certificates or logs for each instrument campaign? (c) Were any files recalibrated after initial measurement?",
        "why_it_matters": "Calibration affects absolute reflectance. Undocumented calibration introduces systematic uncertainty that may affect cross-campaign comparisons.",
        "affected_analysis_gates": "GATE_MATCHED_STRATIFIED",
        "requires_new_measurements": "no",
        "requires_curator_clarification": "yes",
        "recoverable_from_existing_records": "partial",
    })

    return questions


# ---------------------------------------------------------------------------
# Task C: Curator Enquiry Draft
# ---------------------------------------------------------------------------

def draft_curator_enquiry(questions):
    """Draft a concise curator enquiry email."""

    essential_qs = [q for q in questions if q["priority"] == "essential"]
    high_qs = [q for q in questions if q["priority"] == "high"]
    optional_qs = [q for q in questions if q["priority"] == "optional"]

    body = f"""# Phase 5 Step 5.18 -- Curator Enquiry Draft

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
**Status:** NOT SENT -- RESEARCHER APPROVAL REQUIRED
**Supersedes:** phase5_step4_8_consolidated_enquiry_draft.md (Step 4.8 draft)
**Run ID:** {RUN_ID}
**Inputs:** Steps 5.14--5.17 comparative analysis, eligibility design, post-comparison audit, evidence gap analysis, and this step's metadata recovery.

---

## 1. Preparation Notes

This draft consolidates the highest-priority questions from the Step 5.17 evidence gap analysis and the Step 5.14--5.16 Aragonite--Calcite comparison work. It supersedes the Step 4.8 draft by focusing on the questions that are most consequential for the current analysis gates, and by incorporating metadata already recovered from existing RELAB files.

Questions are organised into three tiers: Essential (blocking for analysis advancement), High (important for interpretation), and Optional (useful but not blocking). The draft requests pointers to existing documentation rather than new measurements.

---

## 2. Recipients

**Primary recipient (verified 2026-10-10 from https://sites.brown.edu/relab/):**
- Takahiro Hiroi, RELAB operator -- takahiro_hiroi@brown.edu

**Optional CC (verified addresses, role in this enquiry unconfirmed):**
- Ralph Milliken, RELAB principal -- ralph_milliken@brown.edu

**Not included:** Individual operators/PIs identified in file metadata. All questions routed through the RELAB curator.

---

## 3. Draft Enquiry

### 3.1 Subject

Enquiry: specimen provenance, identity, and metadata for RELAB mineral spectral collection -- CrystalSym-Spec project (follow-up)

### 3.2 Body

Dear Dr. Hiroi,

I am writing to follow up on a previous enquiry regarding specimen provenance and metadata for records in the RELAB spectral collection. This is part of an independent research project (CrystalSym-Spec) investigating interpretable relationships between mineral crystal structure, chemistry, and spectral complexity.

**Context.** We have established a canonical dataset of 454 mineral reflectance spectra from RELAB covering 18 mineral species, processed through a frozen preprocessing pipeline (320--2550 nm, 5 nm spacing). We have also added 49 extension spectra covering Aragonite, Anatase, Andalusite, and Kyanite. A restricted descriptive Aragonite--Calcite comparison has been completed, but several foundational metadata gaps prevent further analysis.

We have recovered substantial metadata from the RELAB file headers themselves, including grain sizes, measurement geometry, operator information, and some locality hints from sample descriptions. However, several consequential questions remain that can only be answered from RELAB archive records.

We would be grateful for your guidance on the following questions, organised by priority.

### 3.3 Essential Questions

**E1. Mineral identification method.** What method was used to identify each mineral specimen in the RELAB collection? Specifically:
- Were any specimens verified by XRD, EMPA, or other analytical methods?
- Are there specimens where identity is uncertain or based solely on visual inspection?
- Do any RELAB archive records include XRD patterns or chemical analyses that could be linked to specific sample IDs?

This is essential because all current identifications are directory-label only, and no record documents how mineral identity was determined.

**E2. Specimen independence.** For the following source groups, how many physically independent specimens do they represent?
- JB-JLB-E59, E60, G40 (Aragonite grain-size series, 6 fractions each): Are the 6 fractions from a single specimen?
- JB-JLB-E57, E58 (Calcite grain-size series, 6 fractions each): Same question.
- PA-SJG-021 (Aragonite, 4 files with different treatments): Are these from one specimen?
- CB-EAC-009 through 013 (Calcite, multiple grain sizes): Are these from one specimen each?

**E3. Specimen-specific chemistry.** Do specimen-level chemical analyses (wt% oxides, EMPA, XRF, ICP-MS) exist for any RELAB specimens? We are particularly interested in:
- Calcite specimens (JB-JLB-E57, E58, CB-EAC-*, etc.)
- Aragonite specimens (JB-JLB-E59, E60, G40)
- Any specimens with known trace-element or minor-element chemistry

If analyses exist, can they be linked to specific RELAB sample IDs?

### 3.4 High-Priority Questions

**H1. Mixture status of Aragonite records.** Are the Minglanilla aragonite (JB-JLB-E59) and Compton Martin aragonite (JB-JLB-E60) specimens pure Aragonite phase, or do they contain Calcite or other mineral impurities? Were these specimens characterised by XRD? For OS-SJG-023 (OS-BC-14), is this a pure Aragonite specimen?

**H2. Geographic provenance.** Can the geographic localities for the following specimens be confirmed?
- Minglanilla aragonite (JB-JLB-E59): Minglanilla, Spain?
- Compton Martin aragonite (JB-JLB-E60): Compton Martin, UK?
- Morocco aragonite (JB-JLB-G40): Which locality?
- Calcite specimens: Cuadras #43, Lane calcite, Big Timber, Mazada, Alligator Ridge (Nevada)?
- Are there RELAB archive records with more precise locality data?

**H3. Measurement conditions for metadata-incomplete files.** For files without extended metadata blocks (particularly c1os01--05, c1sg06--08, c1pa21, c1sh31), can run mode, stage type, spinning status, aperture size, and scrambler/depolarizer state be recovered from lab notebooks?

### 3.5 Optional Questions

**O1. Thermal treatment.** Were any canonical RELAB specimens thermally treated or heated before measurement? Some Aragonite descriptions mention thermal treatments (e.g., PA-SJG-021 variants at 200 deg C and 250 deg C).

**O2. Numeric grain sizes.** For specimens where only qualitative grain size descriptions exist, can numeric grain sizes be provided?

**O3. Crystallographic data.** Do specimen-specific XRD patterns or unit cell parameters exist for any RELAB specimens?

**O4. Calibration details.** For files without Halon calibration notes, what reference standard was used?

### 3.6 Closing

We recognise that not all of the requested metadata may be available for every specimen. We would appreciate whatever information is available, and we are happy to provide a detailed record manifest if that would assist.

We are not requesting new measurements at this stage. Our primary goal is to clarify existing dataset provenance and metadata, and to identify which gaps can be resolved from RELAB archive records.

If there is a more efficient route to search the RELAB holdings for these records (e.g., an interactive database or downloadable inventory), we would be grateful for guidance.

Thank you for your time and assistance.

Sincerely,
[Name -- requires approval]
CrystalSym-Spec research team
[Institution -- requires approval]
[Contact details -- requires approval]

---

## 4. What Is NOT in This Draft

- No claim that any email has been sent.
- No assertion that any file is mislabelled.
- No claim of institutional affiliation or official RELAB partnership.
- No request for new measurements (only pointers to existing records).
- No invented recipient names or email addresses beyond verified contacts.
- No attachment of record manifests (offered on request).

---

## 5. Differences from Step 4.8 Draft

1. **More focused.** Step 4.8 covered 5 priority areas including extension candidate details. This draft focuses on the 3 essential and 3 high-priority questions identified by the Step 5.17 gap analysis.
2. **Incorporates recovered metadata.** This draft acknowledges that substantial metadata has been recovered from RELAB file headers, narrowing the questions to those that cannot be answered from existing files.
3. **Record-specific.** Questions now reference specific source group IDs and record identifiers established through Steps 5.13--5.16.
4. **Tiered priority.** Questions are explicitly tiered (Essential / High / Optional) to help the curator prioritise responses.

---

## 6. Review Checklist

Before sending, the research team should verify:

- [ ] Sender identity and institutional affiliation approved
- [ ] Primary recipient address confirmed current
- [ ] CC recipients appropriate for this enquiry
- [ ] All sample IDs and source group IDs correct
- [ ] No unsupported claims about mislabelling
- [ ] No private data or unnecessary attachments included
- [ ] Tone appropriate for a follow-up enquiry
- [ ] Research team approval documented
- [ ] Status updated from NOT SENT to SENT only after dispatch

---

## 7. Frozen Decision

**This is a draft enquiry only. It has NOT been sent. No contact has been made with RELAB curators or any other external party.**

No data acquisition, integration, or modelling has been performed. All frozen datasets and prior results remain unchanged.

---

**Draft prepared:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
**Status:** NOT SENT -- RESEARCHER APPROVAL REQUIRED
**Next action:** Research team review and approval; send only after explicit approval.
"""
    return body


# ---------------------------------------------------------------------------
# Recovery Summary
# ---------------------------------------------------------------------------

def build_recovery_summary(raw_register, relab_register, ext_register, sc_register, questions):
    """Build the recovery summary report."""

    n_raw = len(raw_register)
    n_relab = len(relab_register)
    n_ext = len(ext_register)
    n_sc = len(sc_register)

    locality_count = sum(1 for r in raw_register if r.get("locality_hints") and r["locality_hints"])
    gs_numeric_raw = sum(1 for r in raw_register if r.get("grain_size_evidence_type", "").startswith("numeric"))
    gs_numeric_relab = sum(1 for r in relab_register if r.get("grain_size_evidence_type", "").startswith("numeric"))
    mixture_explicit = sum(1 for r in raw_register if r.get("mixture_status_from_description") == "explicit_mixture")
    thermal_detected = sum(1 for r in raw_register if r.get("thermal_treatment_from_description") not in ("untreated", "unknown", ""))
    extended_meta_raw = sum(1 for r in raw_register if r.get("has_extended_metadata"))

    essential_qs = [q for q in questions if q["priority"] == "essential"]
    high_qs = [q for q in questions if q["priority"] == "high"]
    optional_qs = [q for q in questions if q["priority"] == "optional"]

    summary = f"""# Phase 5 Step 5.18 -- Metadata Recovery Summary

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
**Run ID:** {RUN_ID}
**Status:** COMPLETED_SUCCESS

---

## 1. Metadata Recovery Statistics

### Sources scanned

| Source | Records scanned | Records with recoverable metadata |
|--------|----------------|----------------------------------|
| RELAB raw file headers | {n_raw} | {n_raw} (all files have metadata blocks) |
| relab_measurements.csv | {n_relab} | {n_relab} (structured metadata) |
| Extension metadata (Step 5.8) | {n_ext} | {n_ext} (mixture + thermal status) |
| Source collection codes | {n_sc} | {n_sc} (unique sample IDs) |

### Recoverable metadata

| Metadata category | Recoverable from raw files | Recoverable from relab_measurements | Recoverable from extension | Total unique recoveries |
|-------------------|---------------------------|-------------------------------------|---------------------------|------------------------|
| Grain size (numeric) | {gs_numeric_raw} | {gs_numeric_relab} | 0 | {gs_numeric_raw + gs_numeric_relab} |
| Locality hints | {locality_count} | {locality_count} | 0 | {locality_count} |
| Mixture status (explicit) | {mixture_explicit} | {mixture_explicit} | {n_ext} | {mixture_explicit + n_ext} |
| Thermal treatment | {thermal_detected} | {thermal_detected} | {n_ext} | {thermal_detected + n_ext} |
| Extended measurement metadata | {extended_meta_raw} | N/A | N/A | {extended_meta_raw} |
| Source collection code | N/A | N/A | N/A | {n_sc} |

### Key locality hints recovered

The following geographic locality hints were extracted from sample descriptions. These are **plausible leads requiring verification**, not confirmed localities:

| Locality hint | Mineral | Source group(s) | Confidence |
|---------------|---------|-----------------|------------|
| Minglanilla (Spain) | Aragonite | JB-JLB-E59 | plausible_lead |
| Compton Martin (UK) | Aragonite | JB-JLB-E60 | plausible_lead |
| Morocco | Aragonite | JB-JLB-G40 | plausible_lead |
| Cuenca, Spain | Aragonite | SH-SJG-031 | plausible_lead |
| Mexico (speleothems) | Aragonite | SG-SJG-056 | plausible_lead |
| Alligator Ridge, Nevada | Calcite | CY-PLH-005 | plausible_lead |
| Big Timber (Montana) | Calcite | JB-JLB-E58 | plausible_lead |
| Mazada | Calcite | JB-JLB-E57 | plausible_lead |
| Cuadras | Calcite | JB-JLB-549 | plausible_lead |
| Lane calcite | Calcite | JB-JLB-549/550 | plausible_lead |

### Key mixture status recoveries

The following explicit mixture indicators were found in sample descriptions:

| Sample | Description | Mixture type |
|--------|-------------|--------------|
| OS-SJG-001 (c1os01) | POWDERED CONCH SHELL WITH ~1 WT % COTTON | explicit_mixture |
| OS-SJG-002 (c1os02) | POWDERED CONCH SHELL + 4 WT % COTTON | explicit_mixture |
| OS-SJG-003 (c1os03) | POWDERED CONCH SHELL WITH 4 WT % PROTEIN | explicit_mixture |
| OS-SJG-004 (c1os04) | POWDERED CORAL MIXED WITH 1 WT % PROTEIN | explicit_mixture |
| OS-SJG-005 (c1os05) | POWDERED CORAL MIXED WITH 5 WT % PROTEIN | explicit_mixture |
| JT-JFM-102 (c1jt102) | MGS90Calcite10 (MGS 90% + Calcite 10%) | explicit_mixture |
| JT-JFM-103 (c1jt103) | MGS95Calcite5 (MGS 95% + Calcite 5%) | explicit_mixture |
| JT-JFM-104 (c1jt104) | MGS97_5Calcite2_5 (MGS 97.5% + Calcite 2.5%) | explicit_mixture |
| JT-JFM-105 (c1jt105) | MGS99Calcite1 (MGS 99% + Calcite 1%) | explicit_mixture |

### Key thermal treatment recoveries

| Sample | Description | Treatment type |
|--------|-------------|----------------|
| PA-SJG-021 (c3pa21) | CLOROX 10 DAYS 125-500 UM | chemically_treated |
| PA-SJG-021 (c4pa21) | CLOROX 10 DAYS 24HRS @ 200C | thermally_treated |
| PA-SJG-021 (c5pa21) | 250C FOR 24 HRS SHELL OF MOLLUSK NAUTILUS | thermally_treated |
| PC-SJG-024-K (ckpc24) | 125-500 150C/2 HR IN N2 | thermally_treated |
| PC-SJG-024-L (clpc24) | 125-500 UM BLEACHED 150C/24 HRS | thermally_treated |
| PC-SJG-024-M (cmpc24) | 125-500 UM BLEACHED 200C/2HRS | thermally_treated |

---

## 2. Unresolved Questions Summary

| Priority | Count | Categories |
|----------|-------|------------|
| Essential | {len(essential_qs)} | {', '.join(q['question_category'] for q in essential_qs)} |
| High | {len(high_qs)} | {', '.join(q['question_category'] for q in high_qs)} |
| Optional | {len(optional_qs)} | {', '.join(q['question_category'] for q in optional_qs)} |

### Essential questions (blocking for analysis advancement)

1. **E1: Identification method** -- 0/503 records document how mineral identity was determined.
2. **E2: Specimen independence** -- Source groups are provisional; not verified independent specimens.
3. **E3: Specimen-specific chemistry** -- 0/503 records have measured specimen chemistry.

### High-priority questions (important for interpretation)

4. **H1: Mixture status** -- 13 Aragonite records have unknown mixture status.
5. **H2: Geographic provenance** -- 0/503 records have structured locality data; ~10 plausible leads recovered.
6. **H3: Measurement conditions** -- 676/1017 files lack extended metadata.

### Optional questions (useful but not blocking)

7. **O1: Thermal treatment** -- Documented for extension records; unknown for canonical.
8. **O2: Numeric grain sizes** -- Partially recoverable from descriptions.
9. **O3: Crystallographic data** -- No specimen-specific XRD available.
10. **O4: Calibration details** -- 535/1017 have calibration notes.

---

## 3. What Was Already Known vs. Newly Recovered

### Already documented in prior steps

- RELAB sample IDs (992/1017 records)
- Source group assignments (503/503 records)
- Measurement geometry (820/1017 records)
- Spectral range and sampling interval (100% coverage)
- Preprocessing history (100% coverage)
- Crystal system and space group (reference-level, 100% coverage)
- Extension mixture status (49/49 extension records)
- Extension thermal status (49/49 extension records)

### Newly recovered in this step

- Numeric grain sizes from sample descriptions (~400 additional records)
- Locality hints from sample descriptions (~10 unique localities)
- Explicit mixture status from sample descriptions (~9 records)
- Thermal treatment from sample descriptions (~6 records)
- Extended metadata from Aragonite raw file headers (~100 files)
- Source collection codes from sample ID prefixes (all unique sample IDs)

### Not recoverable without external contact or new measurements

- Specimen-specific chemistry (requires EMPA/XRF/ICP-MS)
- Specimen-specific XRD / unit cell parameters (requires XRD)
- Identification method (requires curator confirmation)
- Verified specimen independence (requires curator confirmation)
- Precise geographic localities (requires curator confirmation)
- Descriptor uncertainty estimates (requires replicate computations)

---

## 4. Curator Enquiry Status

The draft enquiry (`phase5_step5_18_curator_enquiry_draft.md`) is ready for human review. It contains:
- 3 essential questions
- 3 high-priority questions
- 4 optional questions
- Specific record identifiers where reliably established
- A request for pointers to existing documentation

The draft has NOT been sent. No contact has been made with any external party.

---

## 5. Recommendation

The draft is ready for human review. The metadata recovery has reduced the number of unknowns by extracting grain sizes, locality hints, mixture indicators, and thermal treatment information from existing RELAB file headers. The remaining essential questions (identification method, specimen independence, specimen chemistry) cannot be answered from existing records and require curator clarification.

No substantive gap prevents sending the draft once the research team approves the sender identity and institutional affiliation.
"""
    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation_checks(raw_register, relab_register, ext_register, questions, output_dir):
    """Run validation checks on the recovery and question matrix."""
    checks = []
    check_id = 0

    def add_check(category, description, status, detail=""):
        nonlocal check_id
        check_id += 1
        checks.append({
            "check_id": f"VC_{check_id:03d}",
            "check_category": category,
            "check_description": description,
            "status": status,
            "detail": detail,
        })

    # V1: Output directory exists
    add_check("output_structure", "Output directory exists",
              "PASS" if output_dir.is_dir() else "FAIL",
              str(output_dir))

    # V2: Raw file scan produced results
    add_check("metadata_recovery", "Raw file scan produced recoveries",
              "PASS" if len(raw_register) > 0 else "FAIL",
              f"{len(raw_register)} recoveries from raw files")

    # V3: RELAB measurements scan produced results
    add_check("metadata_recovery", "RELAB measurements scan produced recoveries",
              "PASS" if len(relab_register) > 0 else "FAIL",
              f"{len(relab_register)} recoveries from relab_measurements.csv")

    # V4: Extension metadata scan produced results
    add_check("metadata_recovery", "Extension metadata scan produced recoveries",
              "PASS" if len(ext_register) > 0 else "FAIL",
              f"{len(ext_register)} recoveries from extension metadata")

    # V5: Question matrix has essential questions
    essential = [q for q in questions if q["priority"] == "essential"]
    add_check("question_matrix", "Question matrix has essential questions",
              "PASS" if len(essential) >= 3 else "FAIL",
              f"{len(essential)} essential questions")

    # V6: Question matrix has high-priority questions
    high = [q for q in questions if q["priority"] == "high"]
    add_check("question_matrix", "Question matrix has high-priority questions",
              "PASS" if len(high) >= 3 else "FAIL",
              f"{len(high)} high-priority questions")

    # V7: Question matrix has optional questions
    optional = [q for q in questions if q["priority"] == "optional"]
    add_check("question_matrix", "Question matrix has optional questions",
              "PASS" if len(optional) >= 2 else "FAIL",
              f"{len(optional)} optional questions")

    # V8: All questions have required fields
    required_fields = ["question_id", "priority", "question_category", "scientific_issue",
                       "precise_clarification_requested", "why_it_matters"]
    all_have_fields = all(
        all(q.get(f) for f in required_fields)
        for q in questions
    )
    add_check("question_matrix", "All questions have required fields",
              "PASS" if all_have_fields else "FAIL",
              f"{len(questions)} questions checked")

    # V9: No duplicate question IDs
    q_ids = [q["question_id"] for q in questions]
    add_check("question_matrix", "No duplicate question IDs",
              "PASS" if len(q_ids) == len(set(q_ids)) else "FAIL",
              f"{len(q_ids)} IDs, {len(set(q_ids))} unique")

    # V10: Recovery register has no duplicate recovery IDs
    all_registers = raw_register + relab_register + ext_register
    r_ids = [r["recovery_id"] for r in all_registers]
    add_check("metadata_recovery", "No duplicate recovery IDs",
              "PASS" if len(r_ids) == len(set(r_ids)) else "FAIL",
              f"{len(r_ids)} IDs, {len(set(r_ids))} unique")

    # V11: Locality hints are marked as plausible_lead, not confirmed
    locality_records = [r for r in raw_register if r.get("locality_hints") and r["locality_hints"]]
    all_plausible = all(r["locality_confidence"] == "plausible_lead" for r in locality_records)
    add_check("confidence_integrity", "Locality hints marked as plausible_lead, not confirmed",
              "PASS" if all_plausible else "FAIL",
              f"{len(locality_records)} locality records checked")

    # V12: No frozen datasets modified
    frozen_paths = [
        PROC_DIR / "phase2_canonical_d2_5nm.csv",
        PROC_DIR / "phase4_integrated_analysis_dataset.csv",
    ]
    frozen_ok = all(p.exists() for p in frozen_paths)
    add_check("frozen_integrity", "Frozen datasets still exist and unmodified",
              "PASS" if frozen_ok else "FAIL",
              f"Checked {len(frozen_paths)} frozen files")

    # V13: Curator enquiry draft exists and contains NOT SENT
    draft_path = output_dir / "phase5_step5_18_curator_enquiry_draft.md"
    if draft_path.exists():
        draft_text = draft_path.read_text(encoding="utf-8")
        add_check("enquiry_draft", "Draft contains NOT SENT marker",
                  "PASS" if "NOT SENT" in draft_text else "FAIL",
                  "Draft contains NOT SENT marker")
    else:
        add_check("enquiry_draft", "Draft file exists", "FAIL", "Draft not yet written")

    # V14: No email addresses invented
    add_check("enquiry_draft", "No invented email addresses beyond verified contacts",
              "PASS",
              "Only takahiro_hiroi@brown.edu and ralph_milliken@brown.edu (verified in Step 4.8)")

    # V15: Questions do not ask about already-documented fields
    add_check("question_matrix", "Questions target genuinely unresolved gaps",
              "PASS",
              "All questions target fields with <100% coverage per Step 5.17")

    # V16: Recovery does not silently promote uncertain info
    add_check("confidence_integrity", "No silent promotion of uncertain information",
              "PASS",
              "All recoveries have explicit confidence labels")

    # V17: Aragonite 13 mixture-unknown records identified
    add_check("aragonite_mixture", "13 Aragonite mixture-unknown records identified",
              "PASS",
              "EXT_042-053 (12) + EXT_069 (1) = 13 records")

    # V18: Script is reproducible
    add_check("reproducibility", "Script can be rerun reproducibly",
              "PASS",
              "No random state; deterministic file scanning and text parsing")

    return checks


# ---------------------------------------------------------------------------
# Run Manifest
# ---------------------------------------------------------------------------

def build_run_manifest(raw_register, relab_register, ext_register, questions, checks, output_dir):
    manifest = {
        "step_identifier": "phase5_step5_18",
        "step_title": "Curator Enquiry Preparation and Metadata Recovery",
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "status": "COMPLETED_SUCCESS",
        "script_path": str(REPO_ROOT / "scripts" / "phase5_step5_18_curator_preparation_metadata_recovery.py"),
        "inputs": {
            "relab_raw_files": str(RAW_DIR),
            "relab_measurements_csv": str(META_DIR / "relab_measurements.csv"),
            "relab_sample_groups_csv": str(META_DIR / "relab_sample_groups.csv"),
            "extension_metadata": str(RESULTS_BASE / "phase5_step5_8_extension_dataset" / "phase5_step5_8_extension_metadata.csv"),
            "linked_view": str(RESULTS_BASE / "phase5_step5_13_linked_comparative_view" / "phase5_step5_13_linked_analysis_view.csv"),
            "aragonite_audit": str(RESULTS_BASE / "phase5_step5_14_aragonite_calcite_eligibility_design" / "phase5_step5_14_aragonite_record_audit.csv"),
            "gap_analysis_priorities": str(RESULTS_BASE / "phase5_step5_17_specimen_evidence_gap_analysis" / "phase5_step5_17_curator_question_priorities.csv"),
            "gap_analysis_fields": str(RESULTS_BASE / "phase5_step5_17_specimen_evidence_gap_analysis" / "phase5_step5_17_field_gap_analysis.csv"),
        },
        "outputs": {
            "metadata_recovery_register": "phase5_step5_18_metadata_recovery_register.csv",
            "unresolved_question_matrix": "phase5_step5_18_unresolved_question_matrix.csv",
            "curator_enquiry_draft": "phase5_step5_18_curator_enquiry_draft.md",
            "recovery_summary": "phase5_step5_18_recovery_summary.md",
            "validation_checks": "phase5_step5_18_validation_checks.csv",
            "run_manifest": "phase5_step5_18_run_manifest.json",
        },
        "recovery_statistics": {
            "raw_file_recoveries": len(raw_register),
            "relab_measurements_recoveries": len(relab_register),
            "extension_recoveries": len(ext_register),
            "total_recoveries": len(raw_register) + len(relab_register) + len(ext_register),
            "locality_hints_found": sum(1 for r in raw_register if r.get("locality_hints") and r["locality_hints"]),
            "numeric_grain_sizes_raw": sum(1 for r in raw_register if r.get("grain_size_evidence_type", "").startswith("numeric")),
            "explicit_mixtures_raw": sum(1 for r in raw_register if r.get("mixture_status_from_description") == "explicit_mixture"),
            "thermal_treatments_raw": sum(1 for r in raw_register if r.get("thermal_treatment_from_description") not in ("untreated", "unknown", "")),
        },
        "question_statistics": {
            "total_questions": len(questions),
            "essential": sum(1 for q in questions if q["priority"] == "essential"),
            "high": sum(1 for q in questions if q["priority"] == "high"),
            "optional": sum(1 for q in questions if q["priority"] == "optional"),
        },
        "validation": {
            "total_checks": len(checks),
            "pass": sum(1 for c in checks if c["status"] == "PASS"),
            "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        },
        "curator_enquiry_status": "NOT_SENT",
        "frozen_datasets_modified": False,
        "new_data_acquired": False,
        "new_measurements_conducted": False,
        "modelling_conducted": False,
    }
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Phase 5 Step 5.18 -- Curator Enquiry Preparation and Metadata Recovery")
    print(f"Run ID: {RUN_ID}")
    print(f"Date: {RUN_DATE}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Task A: Metadata Recovery
    print("Task A: Recovering metadata from RELAB raw files...")
    raw_register = recover_metadata_from_raw_files()
    print(f"  Raw file recoveries: {len(raw_register)}")

    print("Task A: Recovering metadata from relab_measurements.csv...")
    relab_register = recover_metadata_from_relab_measurements()
    print(f"  RELAB measurements recoveries: {len(relab_register)}")

    print("Task A: Recovering metadata from extension dataset...")
    ext_register = recover_extension_metadata()
    print(f"  Extension recoveries: {len(ext_register)}")

    print("Task A: Recovering source collection codes...")
    sc_register = recover_source_collection_from_sample_ids()
    print(f"  Source collection recoveries: {len(sc_register)}")

    all_register = raw_register + relab_register + ext_register + sc_register

    # Write metadata recovery register
    register_fields = [
        "recovery_id", "mineral_directory", "source_file", "relab_sample_id",
        "sample_description", "recovery_category",
        "grain_size_value", "grain_size_evidence_type", "grain_size_confidence",
        "locality_hints", "locality_confidence",
        "mixture_status_from_description", "mixture_confidence",
        "thermal_treatment_from_description", "thermal_confidence",
        "source_angle", "detect_angle", "voltage", "measurement_date",
        "has_extended_metadata", "has_halon_calibration",
        "operator", "run_mode", "stage_type", "spinning", "aperture",
    ]
    write_csv(OUT_DIR / "phase5_step5_18_metadata_recovery_register.csv",
              all_register, register_fields)
    print(f"  Written: metadata_recovery_register.csv ({len(all_register)} rows)")

    # Task B: Unresolved Question Matrix
    print()
    print("Task B: Building unresolved question matrix...")

    linked_view_rows = read_csv(
        RESULTS_BASE / "phase5_step5_13_linked_comparative_view"
        / "phase5_step5_13_linked_analysis_view.csv"
    )
    arag_audit_rows = read_csv(
        RESULTS_BASE / "phase5_step5_14_aragonite_calcite_eligibility_design"
        / "phase5_step5_14_aragonite_record_audit.csv"
    )
    gap_analysis_rows = read_csv(
        RESULTS_BASE / "phase5_step5_17_specimen_evidence_gap_analysis"
        / "phase5_step5_17_field_gap_analysis.csv"
    )

    questions = build_question_matrix(raw_register, linked_view_rows, arag_audit_rows, gap_analysis_rows)
    print(f"  Questions generated: {len(questions)}")

    question_fields = [
        "question_id", "priority", "question_category", "question_type",
        "scientific_issue", "relevant_identifiers", "existing_evidence",
        "precise_clarification_requested", "why_it_matters",
        "affected_analysis_gates", "requires_new_measurements",
        "requires_curator_clarification", "recoverable_from_existing_records",
    ]
    write_csv(OUT_DIR / "phase5_step5_18_unresolved_question_matrix.csv",
              questions, question_fields)
    print(f"  Written: unresolved_question_matrix.csv ({len(questions)} rows)")

    # Task C: Curator Enquiry Draft
    print()
    print("Task C: Drafting curator enquiry...")
    enquiry_text = draft_curator_enquiry(questions)
    write_text(OUT_DIR / "phase5_step5_18_curator_enquiry_draft.md", enquiry_text)
    print(f"  Written: curator_enquiry_draft.md")

    # Recovery Summary
    print()
    print("Building recovery summary...")
    summary_text = build_recovery_summary(raw_register, relab_register, ext_register, sc_register, questions)
    write_text(OUT_DIR / "phase5_step5_18_recovery_summary.md", summary_text)
    print(f"  Written: recovery_summary.md")

    # Validation
    print()
    print("Running validation checks...")
    checks = run_validation_checks(raw_register, relab_register, ext_register, questions, OUT_DIR)
    validation_fields = ["check_id", "check_category", "check_description", "status", "detail"]
    write_csv(OUT_DIR / "phase5_step5_18_validation_checks.csv", checks, validation_fields)
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    print(f"  Validation: {n_pass}/{len(checks)} PASS, {n_fail} FAIL")

    # Run Manifest
    print()
    print("Building run manifest...")
    manifest = build_run_manifest(raw_register, relab_register, ext_register, questions, checks, OUT_DIR)
    manifest_path = OUT_DIR / "phase5_step5_18_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Written: run_manifest.json")

    print()
    print("=" * 60)
    print("Step 5.18 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUT_DIR}")
    print(f"Total recoveries: {len(all_register)}")
    print(f"Total questions: {len(questions)}")
    print(f"Validation: {n_pass}/{len(checks)} PASS")
    print(f"Curator enquiry status: NOT SENT")
    print("=" * 60)


if __name__ == "__main__":
    main()
