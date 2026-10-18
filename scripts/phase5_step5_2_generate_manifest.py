"""
Phase 5 Step 5.2 — Extension Candidate Manifest Generator
Reads source CSVs from Steps 4.5, 4.6, 4.7 and produces:
1. phase5_step5_2_extension_candidate_manifest.csv (180 rows)
2. phase5_step5_2_metadata_gap_register.csv
3. phase5_step5_2_candidate_summary.csv
"""

import csv
import os
from collections import defaultdict

BASE = "C:/Users/andri/Projects/crystalsym-spec/results/05_exploratory_analysis"
OUT = f"{BASE}/phase5_step5_2_candidate_manifest"

# --- Read source files ---

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

inventory = read_csv(f"{BASE}/phase5_step4_5_polymorph_audit/phase5_step4_5_file_inventory.csv")
d2_compat = read_csv(f"{BASE}/phase5_step4_5_polymorph_audit/phase5_step4_5_d2_compatibility.csv")
duplicates = read_csv(f"{BASE}/phase5_step4_5_polymorph_audit/phase5_step4_5_duplicate_review.csv")
candidate_meta = read_csv(f"{BASE}/phase5_step4_6_provenance_review/phase5_step4_6_candidate_metadata.csv")
identity_conflicts_raw = open(f"{BASE}/phase5_step4_6_provenance_review/phase5_step4_6_identity_conflicts.csv", "r", encoding="utf-8").read()
# Fix line break in "classif\nication" header
identity_conflicts_raw = identity_conflicts_raw.replace("classif\nication", "classification")
identity_conflicts = list(csv.DictReader(identity_conflicts_raw.splitlines()))
evidence_tracker = read_csv(f"{BASE}/phase5_step4_7_correspondence_evidence/phase5_step4_7_evidence_tracker.csv")

print(f"Read {len(inventory)} inventory rows")
print(f"Read {len(d2_compat)} D2 compatibility rows")
print(f"Read {len(duplicates)} duplicate review rows")
print(f"Read {len(candidate_meta)} candidate metadata rows")
print(f"Read {len(identity_conflicts)} identity conflict rows")
print(f"Read {len(evidence_tracker)} evidence tracker rows")

# --- Build lookup indices ---

# D2 compatibility by filename
d2_by_file = {}
for row in d2_compat:
    d2_by_file[row["filename"]] = row

# Candidate metadata by filename (CAND_001 to CAND_062)
cand_by_file = {}
for row in candidate_meta:
    cand_by_file[row["filename"]] = row

# Duplicate groups by filename
dup_by_file = defaultdict(list)
for row in duplicates:
    dup_by_file[row["filename"]].append(row)

# Identity conflicts by filename
conflict_by_file = {}
for row in identity_conflicts:
    fname = row.get("filename", "")
    if fname:
        conflict_by_file[fname] = row
    # Also index by rel_path components
    for key in ("file_a_rel_path", "file_b_rel_path"):
        val = row.get(key)
        if val:
            fn = val.split("/")[-1]
            conflict_by_file[fn] = row

# --- Determine manifest fields ---

# For each of the 180 files, we need:
# - manifest_id (EXT_001 to EXT_180)
# - All inventory fields
# - D2 compatibility classification
# - Candidate ID if applicable (CAND_001 to CAND_062)
# - Identity conflict status
# - Duplicate group info
# - Metadata completeness assessment
# - Evidence gaps
# - Inclusion eligibility (preliminary)

def assess_metadata_completeness(row):
    """Assess metadata completeness from trailing_meta_lines and candidate metadata."""
    fname = row["filename"]
    cand = cand_by_file.get(fname)

    trailing = int(row.get("trailing_meta_lines", 0) or 0)

    if cand:
        # We have Step 4.6 metadata
        has_sample_desc = bool(cand.get("sample_description", "").strip() and
                               cand["sample_description"].lower() not in ("not available", "unknown", ""))
        has_operator = bool(cand.get("operator", "").strip() and
                           cand["operator"].lower() not in ("unknown", ""))
        has_grain = bool(cand.get("grain_size_info", "").strip() and
                        cand["grain_size_info"].lower() not in ("unknown", ""))
        has_geom = bool(cand.get("source_angle", "").strip() and
                       cand["source_angle"].lower() not in ("unknown", ""))

        if has_sample_desc and has_operator and has_grain:
            return "full"
        elif has_sample_desc or has_operator or has_grain:
            return "partial"
        elif trailing > 0:
            return "minimal"
        else:
            return "none"
    else:
        # Not in Step 4.6 candidate set
        if trailing >= 15:
            return "partial"
        elif trailing >= 6:
            return "minimal"
        elif trailing > 0:
            return "minimal"
        else:
            return "none"


def assess_evidence_gaps(row):
    """Identify specific evidence gaps for each file."""
    fname = row["filename"]
    mineral = row["mineral"]
    cand = cand_by_file.get(fname)
    gaps = []

    trailing = int(row.get("trailing_meta_lines", 0) or 0)

    # Check identity conflict
    if fname in conflict_by_file:
        gaps.append("identity_conflict_unresolved")

    # Check metadata gaps
    if trailing == 0:
        gaps.append("no_trailing_metadata")

    if cand:
        # Check specific metadata fields
        if not cand.get("sample_description", "").strip() or \
           cand["sample_description"].lower() in ("not available", "unknown"):
            gaps.append("no_sample_description")
        if not cand.get("operator", "").strip() or \
           cand["operator"].lower() in ("unknown", ""):
            gaps.append("no_operator_recorded")
        if not cand.get("grain_size_info", "").strip() or \
           cand["grain_size_info"].lower() in ("unknown", ""):
            gaps.append("no_grain_size")
        if not cand.get("source_angle", "").strip() or \
           cand["source_angle"].lower() in ("unknown", ""):
            gaps.append("no_measurement_geometry")
        if not cand.get("pi_sponsor", "").strip() or \
           cand["pi_sponsor"].lower() in ("unknown", ""):
            gaps.append("no_pi_sponsor")

        # Check specimen independence
        rep_rel = cand.get("replicate_relationship", "").strip()
        if rep_rel and "unresolved" in rep_rel.lower():
            gaps.append("specimen_independence_unresolved")

        # Check calibration notes
        cal_note = cand.get("calibration_note", "").strip()
        if cal_note and cal_note.lower() != "nbs halon":
            gaps.append("calibration_note_requires_verification")
    else:
        # Not in candidate set — check if metadata is available at all
        if trailing < 10:
            gaps.append("limited_metadata")

    # Grid-related gaps
    d2 = d2_by_file.get(fname)
    if d2:
        compat = d2.get("d2_compatibility", "")
        if compat == "potentially_suitable":
            if row.get("grid_type") == "native_10nm":
                gaps.append("requires_pchip_resampling")
            elif "trimming" in d2.get("d2_notes", ""):
                gaps.append("requires_grid_trimming")

    return "; ".join(gaps) if gaps else "none"


def determine_eligibility(row):
    """Preliminary eligibility assessment based on available evidence."""
    fname = row["filename"]
    mineral = row["mineral"]
    d2 = d2_by_file.get(fname, {})
    compat = d2.get("d2_compatibility", "")

    # Identity conflict = excluded pending resolution
    if fname in conflict_by_file:
        return "excluded_pending_identity_resolution"

    # D2 compatibility
    if compat == "suitable":
        return "eligible_exact_d2"
    elif compat == "potentially_suitable":
        return "eligible_with_preprocessing"
    elif compat == "incomplete":
        return "partial_coverage_only"
    elif compat == "unsuitable":
        return "excluded_unsuitable_coverage"
    else:
        return "unclassified"


def get_duplicate_info(fname):
    """Get duplicate group info for a file."""
    dups = dup_by_file.get(fname, [])
    if not dups:
        return "", ""

    types = set()
    groups = set()
    for d in dups:
        types.add(d["duplicate_type"])
        groups.add(d["duplicate_group_id"])

    return "; ".join(sorted(types)), "; ".join(sorted(groups))


def get_candidate_id(fname):
    """Get CAND_ID if file is in the Step 4.6 candidate set."""
    cand = cand_by_file.get(fname)
    if cand:
        return cand.get("candidate_id", "")
    return ""


# --- Generate manifest ---

manifest_rows = []
for i, row in enumerate(inventory, 1):
    fname = row["filename"]
    mineral = row["mineral"]
    d2 = d2_by_file.get(fname, {})

    dup_type, dup_group = get_duplicate_info(fname)
    meta_completeness = assess_metadata_completeness(row)
    evidence_gaps = assess_evidence_gaps(row)
    eligibility = determine_eligibility(row)
    cand_id = get_candidate_id(fname)

    # Cross-mineral conflict check
    cross_conflict = ""
    if fname in conflict_by_file:
        cross_conflict = "identity_conflict_brookite_rutile_c1mr130"

    manifest_rows.append({
        "manifest_id": f"EXT_{i:03d}",
        "candidate_id": cand_id,
        "mineral": mineral,
        "filename": fname,
        "rel_path": row["rel_path"],
        "md5": row["md5"],
        "sha256": row["sha256"],
        "file_size_bytes": row["file_size"],
        "actual_rows": row["actual_rows"],
        "n_cols": row["n_cols"],
        "has_uncertainty": row["has_uncertainty"],
        "wl_min": row["wl_min"],
        "wl_max": row["wl_max"],
        "grid_type": row["grid_type"],
        "median_spacing": row["median_spacing"],
        "is_uniform_grid": row["is_uniform_grid"],
        "trailing_meta_lines": row["trailing_meta_lines"],
        "specimen_id": row["specimen_id"],
        "sample_group": row["sample_group"],
        "d2_compatibility": d2.get("d2_compatibility", ""),
        "d2_n_points_in_range": d2.get("d2_n_points_in_range", ""),
        "d2_covers_full_interval": d2.get("d2_covers_full_interval", ""),
        "d2_notes": d2.get("d2_notes", ""),
        "metadata_completeness": meta_completeness,
        "duplicate_type": dup_type,
        "duplicate_group_id": dup_group,
        "cross_mineral_conflict": cross_conflict,
        "evidence_gaps": evidence_gaps,
        "preliminary_eligibility": eligibility,
        "correspondence_status": "not_sent",
        "curator_response_received": "no",
    })

# Write manifest
manifest_fields = list(manifest_rows[0].keys())
with open(f"{OUT}/phase5_step5_2_extension_candidate_manifest.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_fields)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nWrote {len(manifest_rows)} manifest rows")

# --- Generate metadata gap register ---

gap_rows = []
gap_id = 0
for mrow in manifest_rows:
    gaps = mrow["evidence_gaps"]
    if gaps and gaps != "none":
        for gap in gaps.split("; "):
            gap_id += 1
            # Determine severity
            if gap in ("identity_conflict_unresolved",):
                severity = "blocking"
            elif gap in ("specimen_independence_unresolved", "no_trailing_metadata"):
                severity = "high"
            elif gap in ("no_sample_description", "no_operator_recorded", "no_measurement_geometry"):
                severity = "medium"
            elif gap in ("requires_pchip_resampling", "requires_grid_trimming"):
                severity = "low"
            else:
                severity = "medium"

            # Determine resolution path
            if "identity_conflict" in gap:
                resolution = "curator_confirmation_of_mineral_label"
            elif "specimen_independence" in gap:
                resolution = "curator_confirmation_of_specimen_relationships"
            elif "no_trailing_metadata" in gap or "no_sample_description" in gap:
                resolution = "curator_recovery_from_archival_sources"
            elif "no_measurement_geometry" in gap:
                resolution = "curator_provision_of_geometry_records"
            elif "requires_pchip" in gap or "requires_grid" in gap:
                resolution = "preprocessing_pipeline_application"
            elif "calibration" in gap:
                resolution = "spectral_inspection_or_curator_confirmation"
            else:
                resolution = "further_investigation"

            gap_rows.append({
                "gap_id": f"GAP_{gap_id:03d}",
                "manifest_id": mrow["manifest_id"],
                "filename": mrow["filename"],
                "mineral": mrow["mineral"],
                "gap_type": gap,
                "severity": severity,
                "affected_questions": "",
                "current_evidence": "",
                "resolution_path": resolution,
                "reversible": "yes" if severity != "blocking" else "case_by_case",
                "blocking_for_inclusion": "yes" if severity in ("blocking", "high") else "no",
            })

# Map affected questions
for grow in gap_rows:
    fn = grow["filename"]
    gap = grow["gap_type"]
    if "identity_conflict" in gap:
        grow["affected_questions"] = "Q1"
    elif "specimen_independence" in gap:
        if fn.startswith("c1sr73") or fn.startswith("c1sr72"):
            grow["affected_questions"] = "Q2; Q3"
        elif fn.startswith("c1jb") or fn.startswith("c1jbg"):
            grow["affected_questions"] = "Q4"
        elif fn.startswith("cacb") or fn.startswith("c1os") or fn.startswith("c") and "pa21" in fn:
            grow["affected_questions"] = "Q5"
        else:
            grow["affected_questions"] = "Q4; Q5"
    elif "calibration" in gap:
        grow["affected_questions"] = "Q9"
    elif "no_trailing_metadata" in gap or "no_sample_description" in gap:
        if fn.startswith("c1mr"):
            grow["affected_questions"] = "Q7"
        else:
            grow["affected_questions"] = "Q6"
    elif "no_measurement_geometry" in gap:
        grow["affected_questions"] = "Q8"

gap_fields = list(gap_rows[0].keys()) if gap_rows else []
with open(f"{OUT}/phase5_step5_2_metadata_gap_register.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=gap_fields)
    writer.writeheader()
    writer.writerows(gap_rows)

print(f"Wrote {len(gap_rows)} gap register rows")

# --- Generate candidate summary ---

# Count by mineral
mineral_counts = defaultdict(lambda: {
    "total_files": 0, "suitable": 0, "potentially_suitable": 0,
    "incomplete": 0, "unsuitable": 0, "identity_conflict": 0,
    "metadata_full": 0, "metadata_partial": 0, "metadata_minimal": 0, "metadata_none": 0,
    "eligible_exact_d2": 0, "eligible_with_preprocessing": 0,
    "partial_coverage_only": 0, "excluded_pending": 0, "excluded_unsuitable": 0,
})

for mrow in manifest_rows:
    mineral = mrow["mineral"]
    mc = mineral_counts[mineral]
    mc["total_files"] += 1

    compat = mrow["d2_compatibility"]
    if compat == "suitable":
        mc["suitable"] += 1
    elif compat == "potentially_suitable":
        mc["potentially_suitable"] += 1
    elif compat == "incomplete":
        mc["incomplete"] += 1
    elif compat == "unsuitable":
        mc["unsuitable"] += 1

    if mrow["cross_mineral_conflict"]:
        mc["identity_conflict"] += 1

    meta = mrow["metadata_completeness"]
    if meta == "full":
        mc["metadata_full"] += 1
    elif meta == "partial":
        mc["metadata_partial"] += 1
    elif meta == "minimal":
        mc["metadata_minimal"] += 1
    elif meta == "none":
        mc["metadata_none"] += 1

    elig = mrow["preliminary_eligibility"]
    if elig == "eligible_exact_d2":
        mc["eligible_exact_d2"] += 1
    elif elig == "eligible_with_preprocessing":
        mc["eligible_with_preprocessing"] += 1
    elif elig == "partial_coverage_only":
        mc["partial_coverage_only"] += 1
    elif elig == "excluded_pending_identity_resolution":
        mc["excluded_pending"] += 1
    elif elig == "excluded_unsuitable_coverage":
        mc["excluded_unsuitable"] += 1

summary_rows = []
for mineral in ["Anatase", "Andalusite", "Aragonite", "Brookite", "Kyanite"]:
    mc = mineral_counts[mineral]
    summary_rows.append({
        "mineral": mineral,
        "total_files": mc["total_files"],
        "d2_suitable": mc["suitable"],
        "d2_potentially_suitable": mc["potentially_suitable"],
        "d2_incomplete": mc["incomplete"],
        "d2_unsuitable": mc["unsuitable"],
        "identity_conflicts": mc["identity_conflict"],
        "metadata_full": mc["metadata_full"],
        "metadata_partial": mc["metadata_partial"],
        "metadata_minimal": mc["metadata_minimal"],
        "metadata_none": mc["metadata_none"],
        "eligible_exact_d2": mc["eligible_exact_d2"],
        "eligible_with_preprocessing": mc["eligible_with_preprocessing"],
        "partial_coverage_only": mc["partial_coverage_only"],
        "excluded_pending_identity_resolution": mc["excluded_pending"],
        "excluded_unsuitable_coverage": mc["excluded_unsuitable"],
        "estimated_specimens_lower": "",
        "estimated_specimens_upper": "",
        "specimen_evidence_status": "unconfirmed",
        "key_dependency": "",
    })

# Add specimen estimates from Step 5.1
spec_estimates = {
    "Anatase": ("2", "3", "Metadata"),
    "Andalusite": ("2", "4", "Q2"),
    "Aragonite": ("7", "145", "Q4; Q5"),
    "Brookite": ("0", "4", "Q1; Q7"),
    "Kyanite": ("1", "2", "Q3"),
}
for sr in summary_rows:
    lo, hi, dep = spec_estimates[sr["mineral"]]
    sr["estimated_specimens_lower"] = lo
    sr["estimated_specimens_upper"] = hi
    sr["key_dependency"] = dep

# Add total row
totals = {
    "mineral": "TOTAL",
    "total_files": sum(sr["total_files"] for sr in summary_rows),
    "d2_suitable": sum(sr["d2_suitable"] for sr in summary_rows),
    "d2_potentially_suitable": sum(sr["d2_potentially_suitable"] for sr in summary_rows),
    "d2_incomplete": sum(sr["d2_incomplete"] for sr in summary_rows),
    "d2_unsuitable": sum(sr["d2_unsuitable"] for sr in summary_rows),
    "identity_conflicts": sum(sr["identity_conflicts"] for sr in summary_rows),
    "metadata_full": sum(sr["metadata_full"] for sr in summary_rows),
    "metadata_partial": sum(sr["metadata_partial"] for sr in summary_rows),
    "metadata_minimal": sum(sr["metadata_minimal"] for sr in summary_rows),
    "metadata_none": sum(sr["metadata_none"] for sr in summary_rows),
    "eligible_exact_d2": sum(sr["eligible_exact_d2"] for sr in summary_rows),
    "eligible_with_preprocessing": sum(sr["eligible_with_preprocessing"] for sr in summary_rows),
    "partial_coverage_only": sum(sr["partial_coverage_only"] for sr in summary_rows),
    "excluded_pending_identity_resolution": sum(sr["excluded_pending_identity_resolution"] for sr in summary_rows),
    "excluded_unsuitable_coverage": sum(sr["excluded_unsuitable_coverage"] for sr in summary_rows),
    "estimated_specimens_lower": "12",
    "estimated_specimens_upper": "158",
    "specimen_evidence_status": "unconfirmed",
    "key_dependency": "Q1-Q5",
}
summary_rows.append(totals)

summary_fields = list(summary_rows[0].keys())
with open(f"{OUT}/phase5_step5_2_candidate_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Wrote {len(summary_rows)} summary rows (including total)")

# --- Print validation stats ---
print("\n--- Validation ---")
print(f"Total manifest rows: {len(manifest_rows)}")
assert len(manifest_rows) == 180, f"Expected 180 rows, got {len(manifest_rows)}"

# Check mineral counts
for mineral in ["Anatase", "Andalusite", "Aragonite", "Brookite", "Kyanite"]:
    count = sum(1 for m in manifest_rows if m["mineral"] == mineral)
    print(f"  {mineral}: {count} files")

# Check D2 compatibility counts
compat_counts = defaultdict(int)
for m in manifest_rows:
    compat_counts[m["d2_compatibility"]] += 1
print(f"D2 compatibility: {dict(compat_counts)}")

# Check distinct MD5s
md5s = set(m["md5"] for m in manifest_rows)
print(f"Distinct MD5s: {len(md5s)}")
assert len(md5s) == 180, f"Expected 180 distinct MD5s, got {len(md5s)}"

print("\nAll validations passed.")
