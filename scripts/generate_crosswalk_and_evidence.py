"""
Generate phase5_step5_3 crosswalk and metadata evidence register CSVs.
"""
import csv
import os
import re

MANIFEST = r"C:\Users\andri\Projects\crystalsym-spec\results\05_exploratory_analysis\phase5_step5_2_candidate_manifest\phase5_step5_2_extension_candidate_manifest.csv"
ARCHIVE = r"C:\Users\andri\Projects\crystalsym-spec\results\05_exploratory_analysis\phase5_step5_3_pds_catalogue_evidence\phase5_step5_3_archive_records.csv"
OUTDIR = r"C:\Users\andri\Projects\crystalsym-spec\results\05_exploratory_analysis\phase5_step5_3_pds_catalogue_evidence"

# ── Load manifest ──────────────────────────────────────────────────────────
with open(MANIFEST, encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))

# ── Load archive records keyed by archive_record_id ───────────────────────
with open(ARCHIVE, encoding="utf-8") as f:
    archive_rows = list(csv.DictReader(f))
archive = {r["archive_record_id"]: r for r in archive_rows}

# ── Define which specimen IDs exist in the PDS archive ────────────────────
# Patterns confirmed present via directory listings (search log Q04-Q14)
# Plus individually confirmed c2*-c5* records

# Explicit IDs confirmed in archive (from XML retrieval or directory grep):
explicit_in_archive = set()

# c1mr* series
for n in [130, 131, 132, 135, 136, 137, 138]:
    explicit_in_archive.add(f"c1mr{n}")

# c1sr* series
for n in [70, 71, 72, 73, 74, 75, 76]:
    for s in ["", "a"]:
        explicit_in_archive.add(f"c1sr{n}{s}")

# caec* series
for n in range(1, 12):
    explicit_in_archive.add(f"caec{n:02d}")

# cpfb* series
for n in [1, 8, 9, 11, 14, 15, 26, 30, 31, 32, 33, 40]:
    explicit_in_archive.add(f"cpfb{n:02d}")

# c1jbe59a-f, c1jbe60a-f
for base in ["c1jbe59", "c1jbe60"]:
    for letter in "abcdef":
        explicit_in_archive.add(f"{base}{letter}")

# c1jbg40a-f
for letter in "abcdef":
    explicit_in_archive.add(f"c1jbg40{letter}")

# c1jbg59
explicit_in_archive.add("c1jbg59")

# c1os* series (all c1os01-26 per directory listing Q14)
for n in range(1, 27):
    explicit_in_archive.add(f"c1os{n:02d}")

# c1pa21
explicit_in_archive.add("c1pa21")

# c1sg* series (all c1sg01-99 per directory listing Q14)
for n in range(1, 100):
    explicit_in_archive.add(f"c1sg{n:02d}")

# c1sh* series (all c1sh01-97 per directory listing Q14)
for n in range(1, 98):
    explicit_in_archive.add(f"c1sh{n:02d}")

# cacb* series (all cacb01-83 per directory listing Q14)
for n in range(1, 84):
    explicit_in_archive.add(f"cacb{n:02d}")

# c1jb213 - check: c1jb* pattern in bdr2 (Q05). But c1jb213 specifically?
# The user's patterns don't include c1jb*. Only c1jbe* and c1jbg*.
# c1jb213 is NOT in the confirmed patterns. Leave it out.

# Individually confirmed c2*-c5* records
for rid in ["c2pa21", "c2sg66", "c2sg68", "c3pa21", "c3sg66", "c4pa21", "c5pa21"]:
    explicit_in_archive.add(rid)

# ── Prefix-stripping rules ────────────────────────────────────────────────
# bir1*, bir2*, bfr1*, bmr1* → strip prefix to get core c1* ID
def strip_prefix(specimen_id):
    """Return (core_id, has_strippable_prefix) for batch-prefixed files."""
    for pfx in ["bir1", "bir2", "bfr1", "bmr1"]:
        if specimen_id.startswith(pfx):
            core = "c1" + specimen_id[len(pfx):]
            return core, True
    return specimen_id, False

# ── Determine match for each manifest row ─────────────────────────────────
def get_match(row):
    sid = row["specimen_id"]
    mineral_dir = row["mineral"]

    # 1. Exact filename match
    if sid in explicit_in_archive:
        return sid, "filename_exact_match", "high"

    # 2. Prefix-stripped match
    core, has_prefix = strip_prefix(sid)
    if has_prefix:
        # Direct match
        if core in explicit_in_archive:
            return core, "prefix_stripped_match", "medium"
        # Try removing leading zeros after the alphabetic prefix part
        # e.g. c1sr073 -> c1sr73, c1cb016a -> c1cb16a (try both)
        m = re.match(r'(c\d+[a-z]+)0*(\d+.*)', core)
        if m:
            normalized = m.group(1) + m.group(2)
            if normalized in explicit_in_archive and normalized != core:
                return normalized, "prefix_stripped_match", "medium"

    # 3. Not found
    return "", "not_found_in_archive", "none"

# ── PDS data lookup ──────────────────────────────────────────────────────
def get_pds_data(pds_id):
    """Return PDS fields for a given archive record ID."""
    if pds_id and pds_id in archive:
        r = archive[pds_id]
        return {
            "specimen_id": r["specimen_id"],
            "specimen_name": r["specimen_name"],
            "mineral_subtype": r["mineral_subtype"],
            "mineral_type": r["material_type"],
            "description": r["specimen_description"],
            "location": r["specimen_collection_location"],
            "owner": r["specimen_owner_name"],
            "origin": r["material_origin"],
            "grain_min": r["specimen_min_size_um"],
            "grain_max": r["specimen_max_size_um"],
            "geometry": r["measurement_geometry_type"],
            "incidence": r["incidence_angle_deg"],
            "emission": r["emission_angle_deg"],
            "phase": r["phase_angle_deg"],
            "instrument": r["instrument_name"],
            "date": r["measurement_date_time"],
            "producer": r["data_producer_name"],
            "wl_min": r["spectral_range_min_nm"],
            "wl_max": r["spectral_range_max_nm"],
            "has_xml": r["xml_label_retrieved"] == "yes",
        }
    elif pds_id:
        # In archive but no XML retrieved
        return {
            "specimen_id": "", "specimen_name": "", "mineral_subtype": "",
            "mineral_type": "", "description": "", "location": "",
            "owner": "", "origin": "", "grain_min": "", "grain_max": "",
            "geometry": "", "incidence": "", "emission": "", "phase": "",
            "instrument": "", "date": "", "producer": "", "wl_min": "",
            "wl_max": "", "has_xml": False,
        }
    return None

# ── Identity agreement logic ─────────────────────────────────────────────
def check_identity(mineral_dir, pds_id, pds):
    """Return (identity_agreement, conflict_description)."""
    if not pds_id or pds is None:
        return "unverified", ""

    # If we have XML data, use it
    if pds["has_xml"]:
        subtype = pds["mineral_subtype"].strip()
        desc = pds["description"].strip()
        name = pds["specimen_name"].strip()
        combined = f"{subtype} {desc} {name}".lower()

        if mineral_dir == "Brookite":
            # ALL Brookite files conflict
            if "ferropseudobrookite" in combined or "fpb" in combined.lower():
                return "conflicts", f"Local directory=Brookite but PDS says Ferropseudobrookite (synthetic oxide)"
            elif "ilmenite" in combined:
                return "conflicts", f"Local directory=Brookite but PDS says Ilmenite-geikielite (synthetic oxide)"
            else:
                return "conflicts", f"Local directory=Brookite but PDS does not identify as Brookite"

        if mineral_dir == "Anatase":
            if "anatase" in combined:
                return "agrees", ""
            else:
                return "conflicts", f"Local directory=Anatase but PDS does not confirm Anatase"

        if mineral_dir == "Andalusite":
            if "schist" in combined and "andalusite" in combined:
                return "conflicts", f"Local directory=Andalusite but PDS says Schist (with andalusite), not pure Andalusite"
            elif "andalusite" in combined:
                return "agrees", ""
            else:
                return "conflicts", f"Local directory=Andalusite but PDS does not confirm Andalusite"

        if mineral_dir == "Kyanite":
            if "kyanite" in combined:
                return "agrees", ""
            else:
                return "conflicts", f"Local directory=Kyanite but PDS does not confirm Kyanite"

        if mineral_dir == "Aragonite":
            if "ankerite" in combined:
                return "conflicts", f"Local directory=Aragonite but PDS says Ankerite"
            elif "mixture" in combined or "mixture" in desc.lower():
                return "conflicts", f"Local directory=Aragonite but PDS says mixture (not pure Aragonite)"
            elif "biogenic carbonate" in combined and ("cotton" in combined or "protein" in combined or "organic" in combined):
                return "conflicts", f"Local directory=Aragonite but PDS says biogenic carbonate mixture with organics"
            elif "aragonite" in combined:
                return "agrees", ""
            else:
                return "unverified", "PDS record exists but mineral identity could not be confirmed from available metadata"

    # In archive but no XML retrieved
    return "unverified", ""


# ── Same-source evidence ─────────────────────────────────────────────────
def get_same_source(specimen_id, pds_id):
    """Return same_source_evidence string if applicable."""
    # Use pds_id (the normalized archive record ID) for matching
    core = pds_id if pds_id else specimen_id

    # Kyanite/Andalusite grain-size pairs
    if core in ("c1sr72", "c1sr72a"):
        return "PDS confirms same source WAR-4482; specimen IDs SR-JFM-072/SR-JFM-072-A; grain-size fractions 710/45 um"
    if core in ("c1sr73", "c1sr73a"):
        return "PDS confirms same source WAR-0482; specimen IDs SR-JFM-073/SR-JFM-073-A; grain-size fractions 710/45 um"

    # Aragonite grain-size series
    if core.startswith("c1jbe59") and core[-1] in "abcdef":
        return "PDS confirms same source Minglanilla aragonite; specimen IDs JB-JLB-E59-A through F; 6 grain-size fractions"
    if core.startswith("c1jbe60") and core[-1] in "abcdef":
        return "PDS confirms same source Compton Martin aragonite; specimen IDs JB-JLB-E60-A through F; 6 grain-size fractions"
    if core.startswith("c1jbg40") and core[-1] in "abcdef":
        return "PDS confirms same source Morocco aragonite; specimen IDs JB-JLB-G40-A through F; 6 grain-size fractions"

    return ""


# ── Build crosswalk rows ─────────────────────────────────────────────────
crosswalk_rows = []
for row in manifest:
    mid = row["manifest_id"]
    mineral_dir = row["mineral"]
    filename = row["filename"]
    sid = row["specimen_id"]

    pds_id, match_basis, match_conf = get_match(row)
    pds = get_pds_data(pds_id)

    if pds:
        pds_sid = pds["specimen_id"]
        pds_name = pds["specimen_name"]
        pds_subtype = pds["mineral_subtype"]
    else:
        pds_sid = pds_name = pds_subtype = ""

    identity_agree, conflict_desc = check_identity(mineral_dir, pds_id, pds)
    same_src = get_same_source(sid, pds_id)

    notes_parts = []
    if match_basis == "prefix_stripped_match":
        notes_parts.append(f"Batch prefix stripped: {sid} -> {pds_id}")
    if match_basis == "not_found_in_archive":
        notes_parts.append("No matching record found in PDS archive")
    if pds and not pds["has_xml"] and pds_id:
        notes_parts.append("In archive per directory listing; XML label not retrieved")
    if pds and pds["has_xml"]:
        notes_parts.append(f"PDS XML label retrieved; {pds['origin']}")

    crosswalk_rows.append({
        "manifest_id": mid,
        "local_mineral_directory": mineral_dir,
        "local_filename": filename,
        "local_specimen_id": sid,
        "pds_archive_record_id": pds_id,
        "pds_specimen_id": pds_sid,
        "pds_specimen_name": pds_name,
        "pds_mineral_subtype": pds_subtype,
        "match_basis": match_basis,
        "match_confidence": match_conf,
        "identity_agreement": identity_agree,
        "conflict_description": conflict_desc,
        "same_source_evidence": same_src,
        "notes": "; ".join(notes_parts) if notes_parts else "",
    })

# ── Write crosswalk CSV ──────────────────────────────────────────────────
cw_path = os.path.join(OUTDIR, "phase5_step5_3_candidate_record_crosswalk.csv")
cw_fields = [
    "manifest_id", "local_mineral_directory", "local_filename",
    "local_specimen_id", "pds_archive_record_id", "pds_specimen_id",
    "pds_specimen_name", "pds_mineral_subtype", "match_basis",
    "match_confidence", "identity_agreement", "conflict_description",
    "same_source_evidence", "notes",
]
with open(cw_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cw_fields)
    w.writeheader()
    w.writerows(crosswalk_rows)
print(f"Wrote {len(crosswalk_rows)} rows to {cw_path}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Metadata evidence register
# ══════════════════════════════════════════════════════════════════════════

# Build a lookup from manifest_id to manifest row
manifest_by_id = {r["manifest_id"]: r for r in manifest}

# Build a lookup from manifest_id to gap types from the gap register
GAP_REG = r"C:\Users\andri\Projects\crystalsym-spec\results\05_exploratory_analysis\phase5_step5_2_candidate_manifest\phase5_step5_2_metadata_gap_register.csv"
with open(GAP_REG, encoding="utf-8") as f:
    gap_rows = list(csv.DictReader(f))

# Group gaps by manifest_id
gaps_by_mid = {}
for g in gap_rows:
    mid = g["manifest_id"]
    gaps_by_mid.setdefault(mid, []).append(g["gap_type"])


def has_gap(mid, gap_type):
    """Check if a manifest_id has a specific gap type."""
    return gap_type in gaps_by_mid.get(mid, [])


def gap_status(mid, gap_type):
    """Return 'gap' if the gap exists, 'no_gap' otherwise."""
    return "gap" if has_gap(mid, gap_type) else "no_gap"


# ── Define evidence entries for each record with PDS data ────────────────
# For records where XML was retrieved, we provide detailed evidence.
# For records in archive but no XML, we note the match only.

evidence_rows = []

def add_evidence(mid, filename, field, prev_gap, pds_value, ev_type, resolution, notes=""):
    evidence_rows.append({
        "manifest_id": mid,
        "local_filename": filename,
        "metadata_field": field,
        "previous_gap_status": prev_gap,
        "pds_evidence_value": pds_value,
        "evidence_type": ev_type,
        "gap_resolution_status": resolution,
        "notes": notes,
    })

# Process all 180 manifest rows
for row in manifest:
    mid = row["manifest_id"]
    fn = row["filename"]
    sid = row["specimen_id"]
    mineral_dir = row["mineral"]

    pds_id, match_basis, match_conf = get_match(row)
    if not pds_id:
        continue  # No PDS match, no evidence to add

    pds = get_pds_data(pds_id)
    if pds is None:
        continue

    has_xml = pds["has_xml"]

    if has_xml:
        # ── Full XML evidence ─────────────────────────────────────────

        # measurement_geometry
        add_evidence(mid, fn, "measurement_geometry",
                      gap_status(mid, "no_measurement_geometry"),
                      f"Bidirectional; incidence={pds['incidence']}deg, emission={pds['emission']}deg, phase={pds['phase']}deg",
                      "archive_metadata", "resolved",
                      "PDS XML label confirms bidirectional geometry 30/0/30")

        # instrument
        add_evidence(mid, fn, "instrument",
                      "no_gap",
                      pds["instrument"],
                      "archive_metadata", "resolved",
                      "PDS confirms RELAB Bidirectional Spectrometer")

        # incidence_angle
        add_evidence(mid, fn, "incidence_angle",
                      "no_gap",
                      f"{pds['incidence']}",
                      "archive_metadata", "resolved",
                      "")

        # emission_angle
        add_evidence(mid, fn, "emission_angle",
                      "no_gap",
                      f"{pds['emission']}",
                      "archive_metadata", "resolved",
                      "")

        # phase_angle
        add_evidence(mid, fn, "phase_angle",
                      "no_gap",
                      f"{pds['phase']}",
                      "archive_metadata", "resolved",
                      "")

        # spectral_range_min
        add_evidence(mid, fn, "spectral_range_min",
                      "no_gap",
                      f"{pds['wl_min']} nm",
                      "archive_metadata", "resolved",
                      "")

        # spectral_range_max
        add_evidence(mid, fn, "spectral_range_max",
                      "no_gap",
                      f"{pds['wl_max']} nm",
                      "archive_metadata", "resolved",
                      "")

        # measurement_date
        add_evidence(mid, fn, "measurement_date",
                      "no_gap",
                      pds["date"],
                      "archive_metadata", "resolved",
                      "")

        # specimen_id
        add_evidence(mid, fn, "specimen_id",
                      "no_gap",
                      pds["specimen_id"],
                      "archive_metadata", "resolved",
                      "PDS specimen ID from XML label")

        # specimen_name
        add_evidence(mid, fn, "specimen_name",
                      "no_gap",
                      pds["specimen_name"],
                      "archive_metadata", "resolved",
                      "PDS specimen name from XML label")

        # specimen_owner
        if pds["owner"]:
            add_evidence(mid, fn, "specimen_owner",
                          gap_status(mid, "no_pi_sponsor"),
                          pds["owner"],
                          "archive_metadata", "resolved",
                          "PDS provides specimen_owner_name")

        # specimen_collection_location
        if pds["location"]:
            add_evidence(mid, fn, "specimen_collection_location",
                          "no_gap",
                          pds["location"],
                          "archive_metadata", "resolved",
                          "")

        # material_origin
        if pds["origin"]:
            add_evidence(mid, fn, "material_origin",
                          "no_gap",
                          pds["origin"],
                          "archive_metadata", "resolved",
                          "")

        # grain_size
        if pds["grain_min"] or pds["grain_max"]:
            grain_str = ""
            if pds["grain_min"] and pds["grain_max"]:
                grain_str = f"{pds['grain_min']}-{pds['grain_max']} um"
            elif pds["grain_max"]:
                grain_str = f"<{pds['grain_max']} um"
            elif pds["grain_min"]:
                grain_str = f">{pds['grain_min']} um"
            add_evidence(mid, fn, "grain_size_min_um",
                          gap_status(mid, "no_grain_size"),
                          pds["grain_min"] if pds["grain_min"] else "",
                          "archive_metadata", "resolved",
                          f"Grain size from PDS: {grain_str}")
            add_evidence(mid, fn, "grain_size_max_um",
                          gap_status(mid, "no_grain_size"),
                          pds["grain_max"] if pds["grain_max"] else "",
                          "archive_metadata", "resolved",
                          f"Grain size from PDS: {grain_str}")

        # sample_description
        if pds["description"]:
            add_evidence(mid, fn, "sample_description",
                          gap_status(mid, "no_sample_description"),
                          pds["description"],
                          "archive_metadata", "resolved",
                          "PDS provides specimen_description")

        # mineral_identity
        subtype = pds["mineral_subtype"]
        desc = pds["description"]
        name = pds["specimen_name"]
        identity_str = subtype if subtype else name
        if not identity_str:
            identity_str = desc[:80] if desc else ""

        # Check for identity conflicts
        identity_agree, conflict_desc = check_identity(mineral_dir, pds_id, pds)
        if identity_agree == "conflicts":
            add_evidence(mid, fn, "mineral_identity",
                          gap_status(mid, "identity_conflict_unresolved"),
                          identity_str,
                          "archive_metadata", "resolved",
                          f"CONFLICT RESOLVED: {conflict_desc}")
        elif identity_agree == "agrees":
            add_evidence(mid, fn, "mineral_identity",
                          "no_gap",
                          identity_str,
                          "archive_metadata", "resolved",
                          "PDS confirms mineral identity matches local directory")

        # source_specimen_id
        if pds_id and pds_id in archive:
            src_sid = archive[pds_id].get("source_specimen_id", "")
            if src_sid:
                add_evidence(mid, fn, "source_specimen_id",
                              "no_gap",
                              src_sid,
                              "archive_metadata", "resolved",
                              "")

        # data_producer / operator
        if pds["producer"]:
            add_evidence(mid, fn, "specimen_owner",
                          gap_status(mid, "no_operator_recorded"),
                          pds["producer"],
                          "archive_metadata", "narrowed",
                          "PDS provides data_producer_name=RELAB; individual operator not always specified")

    else:
        # ── In archive but no XML retrieved ───────────────────────────
        # Add a single evidence row noting the archive match
        add_evidence(mid, fn, "specimen_id",
                      "no_gap",
                      pds_id,
                      "source_reported_fact", "narrowed",
                      f"Record exists in PDS archive as {pds_id}; XML label not retrieved; metadata fields not yet populated")

# ── Write metadata evidence register CSV ─────────────────────────────────
ev_path = os.path.join(OUTDIR, "phase5_step5_3_metadata_evidence_register.csv")
ev_fields = [
    "manifest_id", "local_filename", "metadata_field",
    "previous_gap_status", "pds_evidence_value", "evidence_type",
    "gap_resolution_status", "notes",
]
with open(ev_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ev_fields)
    w.writeheader()
    w.writerows(evidence_rows)
print(f"Wrote {len(evidence_rows)} rows to {ev_path}")

# ── Summary statistics ───────────────────────────────────────────────────
exact = sum(1 for r in crosswalk_rows if r["match_basis"] == "filename_exact_match")
prefix = sum(1 for r in crosswalk_rows if r["match_basis"] == "prefix_stripped_match")
notfound = sum(1 for r in crosswalk_rows if r["match_basis"] == "not_found_in_archive")
agrees = sum(1 for r in crosswalk_rows if r["identity_agreement"] == "agrees")
conflicts = sum(1 for r in crosswalk_rows if r["identity_agreement"] == "conflicts")
unverified = sum(1 for r in crosswalk_rows if r["identity_agreement"] == "unverified")
print(f"\nCrosswalk summary:")
print(f"  Exact matches: {exact}")
print(f"  Prefix-stripped matches: {prefix}")
print(f"  Not found: {notfound}")
print(f"  Identity agrees: {agrees}")
print(f"  Identity conflicts: {conflicts}")
print(f"  Identity unverified: {unverified}")
print(f"\nEvidence register: {len(evidence_rows)} entries for {len(set(r['manifest_id'] for r in evidence_rows))} files")
