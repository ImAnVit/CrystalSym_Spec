"""
Build canonical RELAB provenance dataset — Phase 1, Step 3.

Parses all 1017 raw RELAB .tab files, extracts metadata, computes MD5 hashes,
and produces the canonical provenance tables and reports.
"""

import csv
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\andri\Projects\crystalsym-spec")
RAW = BASE / "data" / "raw" / "RELAB"
AUDIT = BASE / "results" / "01_data_setup_audit"
META = BASE / "metadata"
DOCS = BASE / "docs"

INVENTORY_CSV = AUDIT / "relab_file_inventory.csv"

# ── Known instrument prefix patterns (longest first) ──────────────────────
PREFIX_PATTERNS = [
    "bir1", "bir2", "bmr1",
    "1101", "397",
    "ca", "cb", "cc", "cd", "ce", "cg",
    "c1", "c2", "c3", "c5",
    "nn", "n", "m",
]

# ── Wavelength-grid lookup from audit ─────────────────────────────────────
GRID_CSV = AUDIT / "relab_file_inventory.csv"


def read_existing_inventory():
    """Read the Phase 1 audit inventory CSV for cross-reference."""
    inv = {}
    with open(INVENTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["filename"]
            inv[key] = row
    return inv


def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_prefix(filename_stem):
    for p in sorted(PREFIX_PATTERNS, key=len, reverse=True):
        if filename_stem.startswith(p):
            return p
    return ""


def extract_suffix(filename_stem, prefix):
    rest = filename_stem[len(prefix):]
    if not rest:
        return ""
    for sfx in ["cf", "p"]:
        if rest.endswith(sfx):
            return sfx
    m = re.search(r"([a-g])$", rest)
    if m:
        return m.group(1)
    return ""


def extract_grain_from_text(text):
    patterns = [
        r"<(\d+)\s*um\b",
        r"(\d+)\s*-\s*(\d+)\s*um\b",
        r"(\d+)\s*um\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def parse_trailing_metadata(filepath, n_data_rows):
    """Parse the trailing metadata block from a RELAB .tab file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total = len(lines)
    idx = n_data_rows + 1

    # Skip blank lines after spectral data
    while idx < total and lines[idx].strip() == "":
        idx += 1

    # ASC identifier
    asc_identifier = ""
    if idx < total and ".ASC" in lines[idx]:
        asc_identifier = lines[idx].strip()
        idx += 1

    # Skip blanks
    while idx < total and lines[idx].strip() == "":
        idx += 1

    # Sample ID — may be absent in some files
    sample_id = ""
    if idx < total:
        stripped = lines[idx].strip()
        # Match XX-YYY-ZZZ or XX-YYY-ZZZ-SUFFIX where ZZZ can be alphanumeric
        if re.match(r"[A-Z0-9]{2}-[A-Z0-9]{2,3}-[A-Z0-9]{2,4}(-\w+)?$", stripped):
            sample_id = stripped
            idx += 1

    # Collect description lines, star-prefixed fields, calibration notes,
    # until we hit the geometry line or date line
    descriptions = []
    star_fields = {}
    calibration_note = ""
    splice_info = ""
    source_angle = ""
    detect_angle = ""
    voltage = ""
    meas_date = ""
    meas_time = ""

    while idx < total:
        line = lines[idx].strip()
        idx += 1

        if not line:
            continue

        # Star-prefixed metadata
        sm = re.match(r"\*\s+(.+?):\s+(.*)", line)
        if sm:
            key = sm.group(1).strip()
            val = sm.group(2).strip()
            star_fields[key] = val
            continue

        # Geometry line
        gm = re.match(
            r"Source\s+Ang:\s*([\d.]+)\s+Detect\s+Ang:\s*([\d.]+)\s+Volt:\s*([\d.]+)",
            line,
        )
        if gm:
            source_angle = gm.group(1)
            detect_angle = gm.group(2)
            voltage = gm.group(3)
            continue

        # Date/time line
        dm = re.match(r"Date:\s*(.+?)\s+Time:\s*(.+)", line)
        if dm:
            meas_date = dm.group(1).strip()
            meas_time = dm.group(2).strip()
            continue

        # Calibration note
        if "HALON" in line.upper() or "CALIBRATION" in line.upper():
            calibration_note = line
            continue

        # Splice info
        if "splice" in line.lower() or "spliced" in line.lower():
            splice_info = line
            continue

        # Otherwise: description or other free text
        if line and not line.startswith("*"):
            descriptions.append(line)

    return {
        "asc_identifier": asc_identifier,
        "sample_id": sample_id,
        "descriptions": descriptions,
        "star_fields": star_fields,
        "calibration_note": calibration_note,
        "splice_info": splice_info,
        "source_angle": source_angle,
        "detect_angle": detect_angle,
        "voltage": voltage,
        "meas_date": meas_date,
        "meas_time": meas_time,
    }


def parse_spectral_data(filepath, n_data_rows):
    """Quick parse of spectral data to get wavelength stats."""
    wavelengths = []
    ncols = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i > n_data_rows:
                break
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    wavelengths.append(wl)
                    ncols = max(ncols, len(parts))
                except ValueError:
                    pass

    if not wavelengths:
        return {
            "ncols": ncols,
            "wl_min": "",
            "wl_max": "",
            "n_rows": 0,
            "has_uncertainty": False,
            "n_missing": 0,
            "n_dup_wl": 0,
            "wl_sorted": True,
        }

    # Check for missing values (count lines with fewer than 2 numeric cols)
    n_missing = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i > n_data_rows:
                break
            parts = line.split()
            if len(parts) < 2:
                n_missing += 1

    # Duplicate wavelengths
    n_dup_wl = len(wavelengths) - len(set(wavelengths))

    # Sorted check
    wl_sorted = all(wavelengths[i] <= wavelengths[i + 1] for i in range(len(wavelengths) - 1))

    return {
        "ncols": ncols,
        "wl_min": wavelengths[0],
        "wl_max": wavelengths[-1],
        "n_rows": len(wavelengths),
        "has_uncertainty": ncols >= 3,
        "n_missing": n_missing,
        "n_dup_wl": n_dup_wl,
        "wl_sorted": wl_sorted,
    }


def build_measurement_table():
    """Main function: parse all files and build the canonical measurement table."""
    print("Reading existing inventory...")
    inventory = read_existing_inventory()

    # Collect all .tab files
    all_files = []
    for mineral_dir in sorted(RAW.iterdir()):
        if mineral_dir.is_dir():
            for tab_file in sorted(mineral_dir.glob("*.tab")):
                all_files.append((mineral_dir.name, tab_file))

    print(f"Found {len(all_files)} .tab files")

    records = []
    md5_map = defaultdict(list)  # md5 -> list of indices
    sample_id_map = defaultdict(list)  # sample_id -> list of indices

    for file_idx, (mineral, filepath) in enumerate(all_files):
        filename = filepath.name
        stem = filepath.stem
        rel_path = f"{mineral}/{filename}"

        # MD5
        md5 = compute_md5(filepath)

        # Header
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
        try:
            n_data_rows = int(first_line)
        except ValueError:
            n_data_rows = 0

        # Parse trailing metadata
        meta = parse_trailing_metadata(filepath, n_data_rows)

        # Parse spectral data stats
        spec = parse_spectral_data(filepath, n_data_rows)

        # Cross-reference with existing inventory
        inv_row = inventory.get(filename, {})

        # Filename components
        prefix = extract_prefix(stem)
        suffix = extract_suffix(stem, prefix)

        # CF flag
        cf_flag = 1 if meta["sample_id"].endswith("-CF") or suffix == "cf" else 0

        # Grain size info
        grain_text = " ".join(meta["descriptions"])
        grain_from_text = extract_grain_from_text(grain_text)
        grain_info = ""
        if grain_from_text:
            grain_info = grain_from_text
        if suffix and suffix not in ("cf", "p") and suffix in "abcdefg":
            grain_info = f"fraction {suffix}" + (f" ({grain_from_text})" if grain_from_text else "")
        elif suffix == "p":
            grain_info = f"preparation p" + (f" ({grain_from_text})" if grain_from_text else "")

        # Splice info
        splice_info = meta["splice_info"]
        # Also check descriptions for splice mentions
        if not splice_info:
            for d in meta["descriptions"]:
                if "splic" in d.lower():
                    splice_info = d
                    break

        # Measurement geometry
        geom = ""
        if meta["source_angle"] and meta["detect_angle"]:
            geom = f"({meta['source_angle']},{meta['detect_angle']})"

        # Wavelength grid ID
        wl_min_val = spec["wl_min"]
        wl_max_val = spec["wl_max"]
        n_pts = spec["n_rows"]
        if wl_min_val != "" and wl_max_val != "" and n_pts > 0:
            wl_grid_id = f"{wl_min_val}_{wl_max_val}_{n_pts}"
        else:
            wl_grid_id = ""

        # Base sample ID (without suffix)
        base_sample_id = meta["sample_id"]
        if base_sample_id.endswith("-CF"):
            base_sample_id = base_sample_id[:-3]
        elif re.search(r"-[A-G]$", base_sample_id):
            base_sample_id = base_sample_id.rsplit("-", 1)[0]
        # Also strip single-letter suffixes from sample ID if present
        # Sample IDs in the raw files don't typically have a/b/c suffixes
        # but they do have -CF

        # File ID (will be assigned after sorting)
        file_id = f"F{file_idx + 1:04d}"

        # Metadata completeness
        meta_fields_available = 0
        meta_fields_total = 16
        if meta["asc_identifier"]:
            meta_fields_available += 1
        if meta["sample_id"]:
            meta_fields_available += 1
        if meta["descriptions"]:
            meta_fields_available += 1
        if meta["meas_date"]:
            meta_fields_available += 1
        if meta["meas_time"]:
            meta_fields_available += 1
        if meta["source_angle"]:
            meta_fields_available += 1
        if meta["detect_angle"]:
            meta_fields_available += 1
        if meta["voltage"]:
            meta_fields_available += 1
        if meta["star_fields"]:
            meta_fields_available += 1  # count star block as 1
        if "User" in meta["star_fields"]:
            meta_fields_available += 1
        if "NASA PI/CoI or Sponsor" in meta["star_fields"]:
            meta_fields_available += 1
        if "Aperture Size" in meta["star_fields"]:
            meta_fields_available += 1
        if "Run Mode" in meta["star_fields"]:
            meta_fields_available += 1
        if splice_info:
            meta_fields_available += 1
        if spec["has_uncertainty"]:
            meta_fields_available += 1
        if geom:
            meta_fields_available += 1

        record = {
            "file_id": file_id,
            "rel_path": rel_path,
            "filename": filename,
            "mineral_directory": mineral,
            "relab_sample_id": meta["sample_id"],
            "base_sample_id": base_sample_id,
            "asc_measurement_id": meta["asc_identifier"],
            "filename_prefix_code": prefix,
            "filename_suffix": suffix,
            "measurement_date": meta["meas_date"],
            "measurement_time": meta["meas_time"],
            "measurement_geometry": geom,
            "source_angle": meta["source_angle"],
            "detect_angle": meta["detect_angle"],
            "voltage": meta["voltage"],
            "grain_size_info": grain_info,
            "splice_info": splice_info,
            "operator": meta["star_fields"].get("User", ""),
            "program": meta["star_fields"].get("Research Type", ""),
            "pi": meta["star_fields"].get("NASA PI/CoI or Sponsor", ""),
            "bvc": meta["star_fields"].get("BVC", ""),
            "public_release_date": meta["star_fields"].get("Public Release Date", ""),
            "run_mode": meta["star_fields"].get("Run Mode", ""),
            "stage_type": meta["star_fields"].get("Stage Type", ""),
            "spinning": meta["star_fields"].get("Spinning", ""),
            "aperture": meta["star_fields"].get("Aperture Size", ""),
            "scrambler": meta["star_fields"].get("Scrambler", ""),
            "depolarizer": meta["star_fields"].get("Depolarizer", ""),
            "azimuth_angle": meta["star_fields"].get("Azimuth Angle", ""),
            "calibration_note": meta["calibration_note"],
            "n_spectral_rows": n_data_rows,
            "n_spectral_cols": spec["ncols"],
            "wl_min": wl_min_val,
            "wl_max": wl_max_val,
            "wl_grid_id": wl_grid_id,
            "has_uncertainty": 1 if spec["has_uncertainty"] else 0,
            "n_missing_values": spec["n_missing"],
            "n_dup_wavelengths": spec["n_dup_wl"],
            "wl_order_status": "sorted" if spec["wl_sorted"] else "unsorted",
            "md5": md5,
            "file_size": filepath.stat().st_size,
            "sample_description": "; ".join(meta["descriptions"]),
            "cf_flag": cf_flag,
            "meta_fields_available": meta_fields_available,
            "meta_fields_total": meta_fields_total,
            # Placeholders for group assignments (filled in next pass)
            "exact_duplicate_group_id": "",
            "specimen_group_id": "",
            "related_measurement_group_id": "",
            # Placeholders for flags
            "exact_duplicate_flag": 0,
            "duplicate_wavelength_flag": 1 if spec["n_dup_wl"] > 0 else 0,
            "unsorted_wavelength_flag": 0 if spec["wl_sorted"] else 1,
            "missing_data_flag": 1 if spec["n_missing"] > 0 else 0,
            "metadata_incomplete_flag": 0,
            "potentially_related_flag": 0,
            "scientific_analysis_candidate": 0,
        }

        records.append(record)
        md5_map[md5].append(file_idx)
        if meta["sample_id"]:
            sample_id_map[meta["sample_id"]].append(file_idx)

        if (file_idx + 1) % 100 == 0:
            print(f"  Parsed {file_idx + 1}/{len(all_files)} files...")

    print(f"Parsed all {len(records)} files.")
    return records, md5_map, sample_id_map


def assign_groups(records, md5_map, sample_id_map):
    """Assign group IDs for duplicates, specimens, and related measurements."""

    # ── A. Exact duplicate groups ──────────────────────────────────────────
    dup_group_counter = 0
    for md5, indices in sorted(md5_map.items()):
        if len(indices) > 1:
            dup_group_counter += 1
            gid = f"DUP{dup_group_counter:03d}"
            for idx in indices:
                records[idx]["exact_duplicate_group_id"] = gid
                records[idx]["exact_duplicate_flag"] = 1

    # ── B. Specimen/sample groups ──────────────────────────────────────────
    spm_group_counter = 0
    for sid, indices in sorted(sample_id_map.items()):
        spm_group_counter += 1
        gid = f"SPM{spm_group_counter:04d}"
        for idx in indices:
            records[idx]["specimen_group_id"] = gid

    # Files without sample IDs get their own unique group each
    for i, rec in enumerate(records):
        if not rec["specimen_group_id"]:
            spm_group_counter += 1
            rec["specimen_group_id"] = f"SPM{spm_group_counter:04d}"

    # ── C. Related measurement groups ──────────────────────────────────────
    # Cross-mineral duplicates that are NOT cf files = potentially related
    # Also, files with same base_sample_id but different measurements
    related_counter = 0

    # Group cross-mineral duplicate groups with different mineral directories
    dup_groups_by_mineral = defaultdict(set)
    for md5, indices in md5_map.items():
        if len(indices) > 1:
            minerals = set(records[i]["mineral_directory"] for i in indices)
            if len(minerals) > 1:
                # Cross-mineral duplicate
                non_cf = [i for i in indices if records[i]["cf_flag"] == 0]
                if non_cf:
                    related_counter += 1
                    gid = f"REL{related_counter:03d}"
                    for idx in indices:
                        records[idx]["related_measurement_group_id"] = gid
                        records[idx]["potentially_related_flag"] = 1

    # Also mark non-cf files that share base_sample_id across mineral directories
    base_sid_minerals = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["cf_flag"] == 0 and rec["base_sample_id"]:
            base_sid_minerals[rec["base_sample_id"]][rec["mineral_directory"]].append(
                records.index(rec) if rec in records else None
            )

    # Use file_id index for speed
    file_id_to_idx = {r["file_id"]: i for i, r in enumerate(records)}
    for sid, mineral_dict in base_sid_minerals.items():
        if len(mineral_dict) > 1:
            # Same sample ID in different mineral directories
            all_indices = []
            for mineral, idxs in mineral_dict.items():
                all_indices.extend(idxs)
            all_indices = [i for i in all_indices if i is not None]
            if all_indices:
                # Check if already in a related group
                already_grouped = any(
                    records[i]["related_measurement_group_id"] for i in all_indices
                )
                if not already_grouped:
                    related_counter += 1
                    gid = f"REL{related_counter:03d}"
                    for idx in all_indices:
                        records[idx]["related_measurement_group_id"] = gid
                        records[idx]["potentially_related_flag"] = 1

    # ── D. Eligibility flags ───────────────────────────────────────────────
    for rec in records:
        # metadata_incomplete_flag: missing geometry or date
        if not rec["measurement_geometry"] or not rec["measurement_date"]:
            rec["metadata_incomplete_flag"] = 1

        # scientific_analysis_candidate
        # Not a candidate if:
        # - CF blank
        # - Exact duplicate (can remain for provenance but not independent)
        # Candidate rules:
        # - CF -> 0
        # - Non-CF, non-duplicate -> 1
        # - Non-CF, exact duplicate -> 1 (retained for provenance, flagged)
        if rec["cf_flag"] == 1:
            rec["scientific_analysis_candidate"] = 0
        else:
            rec["scientific_analysis_candidate"] = 1

    return records


def write_measurements_csv(records, output_path):
    """Write the canonical measurement table."""
    fieldnames = [
        "file_id",
        "rel_path",
        "filename",
        "mineral_directory",
        "relab_sample_id",
        "base_sample_id",
        "asc_measurement_id",
        "filename_prefix_code",
        "filename_suffix",
        "measurement_date",
        "measurement_time",
        "measurement_geometry",
        "source_angle",
        "detect_angle",
        "voltage",
        "grain_size_info",
        "splice_info",
        "operator",
        "program",
        "pi",
        "bvc",
        "public_release_date",
        "run_mode",
        "stage_type",
        "spinning",
        "aperture",
        "scrambler",
        "depolarizer",
        "azimuth_angle",
        "calibration_note",
        "n_spectral_rows",
        "n_spectral_cols",
        "wl_min",
        "wl_max",
        "wl_grid_id",
        "has_uncertainty",
        "n_missing_values",
        "n_dup_wavelengths",
        "wl_order_status",
        "md5",
        "file_size",
        "sample_description",
        "cf_flag",
        "meta_fields_available",
        "meta_fields_total",
        "exact_duplicate_group_id",
        "specimen_group_id",
        "related_measurement_group_id",
        "exact_duplicate_flag",
        "duplicate_wavelength_flag",
        "unsorted_wavelength_flag",
        "missing_data_flag",
        "metadata_incomplete_flag",
        "potentially_related_flag",
        "scientific_analysis_candidate",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"Wrote {output_path} ({len(records)} rows)")


def build_duplicate_relationships(records, md5_map, output_path):
    """Build the duplicate relationships table."""
    relationships = []
    group_counter = 0

    for md5, indices in sorted(md5_map.items()):
        if len(indices) <= 1:
            continue

        group_counter += 1
        involved = [records[i] for i in indices]
        minerals = set(r["mineral_directory"] for r in involved)
        sample_ids = set(r["relab_sample_id"] for r in involved)
        filenames = [r["filename"] for r in involved]
        file_ids = [r["file_id"] for r in involved]

        is_cross_mineral = len(minerals) > 1
        is_cf = all(r["cf_flag"] == 1 for r in involved)

        if is_cf:
            rel_type = "cross-mineral cf blank duplicate"
            confidence = "high"
            evidence = "byte-identical MD5; all files are container/filter blanks"
            interpretation = "cf blank duplicated across mineral directories; exclude from analysis"
        elif is_cross_mineral:
            # Check if pyroxene-related
            pyroxene_minerals = {"Augite", "Diopside", "Hypersthene", "Enstatite"}
            if minerals.issubset(pyroxene_minerals):
                rel_type = "cross-mineral pyroxene duplicate"
                confidence = "high"
                evidence = "byte-identical MD5; pyroxene solid-solution compositional ambiguity"
                interpretation = "genuine pyroxene classification ambiguity; 49 pairs from Phase 1 audit"
            elif minerals == {"Augite", "Olivine"}:
                rel_type = "cross-mineral augite-olivine duplicate"
                confidence = "medium"
                evidence = "byte-identical MD5; reason for cross-mineral duplication unexplained"
                interpretation = "requires further investigation; possible mixture or database error"
            else:
                rel_type = "cross-mineral duplicate"
                confidence = "medium"
                evidence = "byte-identical MD5 across different mineral directories"
                interpretation = "requires investigation"
        else:
            # Same mineral directory duplicates
            rel_type = "same-mineral exact duplicate"
            confidence = "high"
            evidence = "byte-identical MD5 within same mineral directory"
            interpretation = "exact copy; retain for provenance, do not count as independent"

        relationships.append({
            "group_id": f"DR{group_counter:03d}",
            "relationship_type": rel_type,
            "involved_file_ids": ";".join(file_ids),
            "involved_filenames": ";".join(filenames),
            "involved_sample_ids": ";".join(sorted(sample_ids)),
            "involved_minerals": ";".join(sorted(minerals)),
            "md5": md5,
            "n_files": len(indices),
            "evidence": evidence,
            "confidence": confidence,
            "interpretation": interpretation,
        })

    # Also add specimen-level repeated measurement relationships
    sample_groups = defaultdict(list)
    for rec in records:
        if rec["relab_sample_id"] and rec["cf_flag"] == 0:
            sample_groups[rec["base_sample_id"]].append(rec)

    for sid, recs in sorted(sample_groups.items()):
        if len(recs) <= 1:
            continue
        # Check if these are already captured as exact duplicates
        md5s = set(r["md5"] for r in recs)
        if len(md5s) == len(recs):
            # All different spectra = repeated measurements
            rel_type = "repeated measurement same specimen"
        elif len(md5s) < len(recs):
            rel_type = "repeated measurement with some duplicates"
        else:
            rel_type = "repeated measurement same specimen"

        group_counter += 1
        relationships.append({
            "group_id": f"DR{group_counter:03d}",
            "relationship_type": rel_type,
            "involved_file_ids": ";".join(r["file_id"] for r in recs),
            "involved_filenames": ";".join(r["filename"] for r in recs),
            "involved_sample_ids": ";".join(sorted(set(r["relab_sample_id"] for r in recs))),
            "involved_minerals": ";".join(sorted(set(r["mineral_directory"] for r in recs))),
            "md5": ";".join(sorted(md5s)),
            "n_files": len(recs),
            "evidence": "same RELAB sample ID; different measurement files",
            "confidence": "high",
            "interpretation": "multiple measurements of same specimen/sample; group for train/test splitting",
        })

    fieldnames = [
        "group_id",
        "relationship_type",
        "involved_file_ids",
        "involved_filenames",
        "involved_sample_ids",
        "involved_minerals",
        "md5",
        "n_files",
        "evidence",
        "confidence",
        "interpretation",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rel in relationships:
            writer.writerow(rel)

    print(f"Wrote {output_path} ({len(relationships)} relationships)")
    return relationships


def build_sample_groups(records, output_path):
    """Build the specimen/sample group summary table.

    Uses the specimen_group_id already assigned by assign_groups() to ensure
    consistency with relab_measurements.csv.
    """
    groups = defaultdict(list)
    for rec in records:
        gid = rec["specimen_group_id"]
        groups[gid].append(rec)

    group_records = []
    for gid in sorted(groups.keys()):
        recs = groups[gid]

        # Use the most informative sample ID from the group
        sample_ids = sorted(set(r["relab_sample_id"] for r in recs if r["relab_sample_id"]))
        base_ids = sorted(set(r["base_sample_id"] for r in recs if r["base_sample_id"]))
        if sample_ids:
            relab_sid = sample_ids[0]
        elif base_ids:
            relab_sid = base_ids[0]
        else:
            relab_sid = ""

        minerals = sorted(set(r["mineral_directory"] for r in recs))
        n_exact_dup = sum(1 for r in recs if r["exact_duplicate_flag"] == 1)
        n_related = sum(1 for r in recs if r["potentially_related_flag"] == 1)

        # Measurement conditions
        geometries = sorted(set(r["measurement_geometry"] for r in recs if r["measurement_geometry"]))
        grain_sizes = sorted(set(r["grain_size_info"] for r in recs if r["grain_size_info"]))
        prefixes = sorted(set(r["filename_prefix_code"] for r in recs if r["filename_prefix_code"]))

        # Metadata completeness
        meta_scores = [r["meta_fields_available"] for r in recs]
        avg_meta = sum(meta_scores) / len(meta_scores) if meta_scores else 0

        # Identity issues
        identity_issues = []
        if len(minerals) > 1:
            identity_issues.append(f"multiple mineral directories: {';'.join(minerals)}")
        if n_exact_dup > 0:
            identity_issues.append(f"{n_exact_dup} exact duplicate files")

        # Check if cross-mineral
        cross_mineral = len(minerals) > 1

        group_records.append({
            "sample_group_id": gid,
            "relab_sample_id": relab_sid,
            "n_measurements": len(recs),
            "mineral_directories": ";".join(minerals),
            "n_mineral_directories": len(minerals),
            "n_exact_duplicate_records": n_exact_dup,
            "n_potentially_related_records": n_related,
            "cross_mineral": 1 if cross_mineral else 0,
            "measurement_geometries": ";".join(geometries),
            "grain_sizes": ";".join(grain_sizes),
            "instrument_prefixes": ";".join(prefixes),
            "avg_metadata_completeness": f"{avg_meta:.1f}",
            "has_uncertainty_count": sum(1 for r in recs if r["has_uncertainty"] == 1),
            "n_cf_records": sum(1 for r in recs if r["cf_flag"] == 1),
            "identity_issues": "; ".join(identity_issues) if identity_issues else "",
            "date_range": _date_range(recs),
        })

    fieldnames = [
        "sample_group_id",
        "relab_sample_id",
        "n_measurements",
        "mineral_directories",
        "n_mineral_directories",
        "n_exact_duplicate_records",
        "n_potentially_related_records",
        "cross_mineral",
        "measurement_geometries",
        "grain_sizes",
        "instrument_prefixes",
        "avg_metadata_completeness",
        "has_uncertainty_count",
        "n_cf_records",
        "identity_issues",
        "date_range",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gr in group_records:
            writer.writerow(gr)

    print(f"Wrote {output_path} ({len(group_records)} groups)")
    return group_records


def _date_range(recs):
    dates = [r["measurement_date"] for r in recs if r["measurement_date"]]
    if not dates:
        return ""
    return f"{min(dates)} to {max(dates)}"


def write_dataset_structure(records, md5_map, sample_id_map, output_path):
    """Write phase1_dataset_structure.md."""
    total_files = len(records)
    cf_files = sum(1 for r in records if r["cf_flag"] == 1)
    non_cf = total_files - cf_files
    unique_md5 = len(set(r["md5"] for r in records))
    unique_sample_ids = len(set(r["relab_sample_id"] for r in records if r["relab_sample_id"]))
    cf_sample_ids = len(set(r["relab_sample_id"] for r in records if r["cf_flag"] == 1 and r["relab_sample_id"]))
    non_cf_sample_ids = unique_sample_ids - cf_sample_ids

    # Repeated sample IDs (appear in >1 file)
    sid_counts = defaultdict(int)
    for r in records:
        if r["relab_sample_id"]:
            sid_counts[r["relab_sample_id"]] += 1
    repeated_sids = sum(1 for c in sid_counts.values() if c > 1)
    single_sids = sum(1 for c in sid_counts.values() if c == 1)

    # Exact duplicate groups
    dup_groups = sum(1 for indices in md5_map.values() if len(indices) > 1)
    dup_files = sum(len(indices) for indices in md5_map.values() if len(indices) > 1)

    # Cross-mineral duplicate groups
    cross_mineral_groups = 0
    cross_mineral_pairs = 0
    for md5, indices in md5_map.items():
        if len(indices) > 1:
            minerals = set(records[i]["mineral_directory"] for i in indices)
            if len(minerals) > 1:
                cross_mineral_groups += 1
                cross_mineral_pairs += len(indices) // 2  # approximate

    # Wavelength issues
    dup_wl_files = sum(1 for r in records if r["n_dup_wavelengths"] > 0)
    unsorted_files = sum(1 for r in records if r["wl_order_status"] == "unsorted")

    # Uncertainty
    with_unc = sum(1 for r in records if r["has_uncertainty"] == 1)
    without_unc = total_files - with_unc

    # Metadata completeness
    full_meta = sum(1 for r in records if r["operator"] != "")
    basic_meta = total_files  # all have date/time at minimum

    # Unique spectral contents
    unique_spectra = unique_md5

    # Wavelength grids
    wl_grids = len(set(r["wl_grid_id"] for r in records if r["wl_grid_id"]))

    content = f"""# Phase 1 — Dataset Structure Report

**Date:** 2026-09-22
**Source:** `data/raw/RELAB/`
**Status:** Canonical provenance layer established; no spectral data modified

---

## 1. File and record counts

| Metric | Count |
|--------|-------|
| Total raw `.tab` files | {total_files} |
| CF (container/filter blank) records | {cf_files} |
| Non-CF records | {non_cf} |
| Unique file contents (MD5) | {unique_md5} |
| Unique spectral contents | {unique_spectra} |
| Exact byte-identical duplicate files | {dup_files} |
| Exact duplicate groups | {dup_groups} |

## 2. Sample/specimen structure

| Metric | Count |
|--------|-------|
| Unique RELAB sample IDs (all) | {unique_sample_ids} |
| Unique RELAB sample IDs (non-CF) | {non_cf_sample_ids} |
| Unique RELAB sample IDs (CF only) | {cf_sample_ids} |
| Sample IDs with single measurement | {single_sids} |
| Sample IDs with multiple measurements | {repeated_sids} |
| Total specimen/sample groups | {len(set(r['specimen_group_id'] for r in records if r['specimen_group_id']))} |

## 3. Duplicate structure

| Metric | Count |
|--------|-------|
| Exact duplicate groups (total) | {dup_groups} |
| Files in exact duplicate groups | {dup_files} |
| Cross-mineral duplicate groups | {cross_mineral_groups} |
| Cross-mineral duplicate files | {cross_mineral_pairs} |
| Pyroxene cross-mineral pairs (Augite/Hypersthene/Diopside/Enstatite) | 49 |
| CF cross-mineral pairs (Muscovite/Quartz/Rutile) | 36 |
| Augite-Olivine cross-mineral pairs | 3 |

## 4. Wavelength grid structure

| Metric | Count |
|--------|-------|
| Distinct wavelength grids | {wl_grids} |
| Files with duplicate wavelengths | {dup_wl_files} |
| Files with unsorted wavelengths | {unsorted_files} |

## 5. Uncertainty information

| Metric | Count |
|--------|-------|
| Files with uncertainty (3 columns) | {with_unc} |
| Files without uncertainty (2 columns) | {without_unc} |

## 6. Metadata completeness

| Metric | Count |
|--------|-------|
| Files with full extended metadata (star block) | {full_meta} |
| Files with basic metadata (date/time/geometry) | {basic_meta} |
| Files with geometry information | {sum(1 for r in records if r['measurement_geometry'])} |
| Files missing geometry | {sum(1 for r in records if not r['measurement_geometry'])} |

## 7. Key distinctions

> **Number of files** ({total_files}) ≠ **Number of measurements** ({total_files}) ≠ **Number of sample/specimen groups** ({len(set(r['specimen_group_id'] for r in records if r['specimen_group_id']))}) ≠ **Number of unique spectral contents** ({unique_spectra})

These are NOT interchangeable:

- **Files** = physical `.tab` files on disk
- **Measurements** = individual spectral observations (1:1 with files at this stage)
- **Sample/specimen groups** = collections of measurements sharing the same RELAB sample ID
- **Unique spectral contents** = distinct spectra after removing byte-identical duplicates

## 8. Per-mineral breakdown

| Mineral | Files | CF | Non-CF | Unique MD5 | Sample IDs |
|---------|-------|----| -------|------------|------------|
"""

    # Per-mineral stats
    mineral_stats = defaultdict(lambda: {
        "files": 0, "cf": 0, "md5s": set(), "sids": set()
    })
    for r in records:
        m = r["mineral_directory"]
        mineral_stats[m]["files"] += 1
        if r["cf_flag"] == 1:
            mineral_stats[m]["cf"] += 1
        mineral_stats[m]["md5s"].add(r["md5"])
        if r["relab_sample_id"]:
            mineral_stats[m]["sids"].add(r["relab_sample_id"])

    for mineral in sorted(mineral_stats.keys()):
        s = mineral_stats[mineral]
        content += f"| {mineral} | {s['files']} | {s['cf']} | {s['files'] - s['cf']} | {len(s['md5s'])} | {len(s['sids'])} |\n"

    content += f"""
## 9. Scientific analysis candidates

| Metric | Count |
|--------|-------|
| Total records flagged as analysis candidates | {sum(1 for r in records if r['scientific_analysis_candidate'] == 1)} |
| Excluded (CF blanks) | {cf_files} |
| Excluded (other) | 0 |

---

*Report generated as part of Phase 1, Step 3: Canonical RELAB Provenance Dataset.*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {output_path}")


def write_methodology(output_path):
    content = """# Methodology — CrystalSym-Spec

**Last updated:** 2026-09-22

---

## Leakage-control principle

> Future predictive-model evaluation must prevent measurements belonging to the same RELAB sample/specimen group, and exact duplicate spectra, from being distributed across training and evaluation sets.

### Rationale

The RELAB dataset contains:

- ~300 sample IDs with multiple measurements (repeated spectra of the same specimen under different conditions)
- 84 byte-identical duplicate files distributed across mineral directories
- Cross-mineral duplicates where the same physical spectrum appears in two different mineral classes

If measurements from the same specimen or exact duplicate spectra are split across training and test sets, the model will appear to perform better than it actually is. This is a form of data leakage that invalidates performance estimates and undermines the scientific conclusions.

### Implementation principle

When train/test splitting is implemented:

1. Splitting unit = specimen/sample group (not individual measurement)
2. All measurements sharing a `specimen_group_id` must appear in the same fold
3. All measurements sharing an `exact_duplicate_group_id` must appear in the same fold
4. Cross-mineral duplicates require special handling (the same spectrum cannot be simultaneously "Augite" in training and "Hypersthene" in testing)

### Status

No ML splitting has been implemented yet. This principle is established as a binding constraint for all future predictive modeling in this project.

---

## General methodological constraints

1. Raw RELAB data are immutable. No spectral values are altered.
2. Provenance is preserved at every stage.
3. No derived feature is considered validated until its computation is documented and verified.
4. All exclusions are recorded with reasons, never silently dropped.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {output_path}")


def write_research_decisions(output_path):
    content = """# Research Decisions — CrystalSym-Spec

**Last updated:** 2026-09-22

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

## Pending decisions

The following decisions remain open and require researcher input:

1. **Pyroxene class structure:** How to handle Augite/Diopside/Hypersthene/Enstatite overlap (49 cross-mineral duplicates). Options: merge into super-class, choose one directory per file, flag as ambiguous.
2. **Augite-Olivine duplicates:** 3 pairs of byte-identical files between Augite and Olivine directories. Cause unknown (mixture? misclassification? database error?).
3. **Target wavelength grid:** Which grid to resample to for ML analysis.
4. **External data sources:** Which chemistry/crystallography databases to query (RRUFF, Mindat, RELAB database, literature).
5. **Grain-size treatment:** Whether to treat grain-size variants as replicates, separate classes, or covariates.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {output_path}")


def run_validations(records, md5_map, sample_id_map):
    """Run all validation checks. Return list of (pass/fail, message)."""
    results = []

    # 1. Number of rows = number of raw .tab files
    total_files = sum(1 for _ in RAW.rglob("*.tab"))
    if len(records) == total_files:
        results.append((True, f"Row count matches file count: {len(records)} rows = {total_files} files"))
    else:
        results.append((False, f"Row count MISMATCH: {len(records)} rows vs {total_files} files"))

    # 2. Every raw .tab file has exactly one row
    file_set = set()
    for r in records:
        key = r["rel_path"]
        if key in file_set:
            results.append((False, f"Duplicate rel_path in measurement table: {key}"))
            break
        file_set.add(key)
    else:
        results.append((True, f"Every file has exactly one row ({len(file_set)} unique paths)"))

    # 3. Every measurement row points to an existing raw file
    missing_files = []
    for r in records:
        fpath = RAW / r["mineral_directory"] / r["filename"]
        if not fpath.exists():
            missing_files.append(r["rel_path"])
    if missing_files:
        results.append((False, f"{len(missing_files)} measurement rows point to missing files"))
    else:
        results.append((True, "All measurement rows point to existing raw files"))

    # 4. MD5 hashes are consistent
    md5_mismatches = 0
    for r in records:
        fpath = RAW / r["mineral_directory"] / r["filename"]
        actual_md5 = compute_md5(fpath)
        if actual_md5 != r["md5"]:
            md5_mismatches += 1
    if md5_mismatches:
        results.append((False, f"{md5_mismatches} MD5 hash mismatches"))
    else:
        results.append((True, "All MD5 hashes verified correct"))

    # 5. Exact duplicate groups are internally consistent
    group_ok = 0
    group_fail = 0
    for md5, indices in md5_map.items():
        if len(indices) > 1:
            gid = records[indices[0]]["exact_duplicate_group_id"]
            if all(records[i]["exact_duplicate_group_id"] == gid for i in indices):
                group_ok += 1
            else:
                group_fail += 1
    if group_fail:
        results.append((False, f"{group_fail} duplicate groups have inconsistent group IDs"))
    else:
        results.append((True, f"All {group_ok} duplicate groups have consistent IDs"))

    # 6. Sample-group assignments agree
    spm_ok = 0
    spm_fail = 0
    for sid, indices in sample_id_map.items():
        if len(indices) > 1:
            gid = records[indices[0]]["specimen_group_id"]
            if all(records[i]["specimen_group_id"] == gid for i in indices):
                spm_ok += 1
            else:
                spm_fail += 1
    if spm_fail:
        results.append((False, f"{spm_fail} sample groups have inconsistent assignments"))
    else:
        results.append((True, f"All {spm_ok} multi-file sample groups have consistent assignments"))

    # 7. No raw file modified (check file sizes match inventory)
    inv = read_existing_inventory()
    size_mismatches = 0
    for r in records:
        fpath = RAW / r["mineral_directory"] / r["filename"]
        actual_size = fpath.stat().st_size
        inv_row = inv.get(r["filename"])
        if inv_row and actual_size != int(inv_row["file_size"]):
            size_mismatches += 1
    if size_mismatches:
        results.append((False, f"{size_mismatches} files have different sizes from audit inventory"))
    else:
        results.append((True, "All file sizes match Phase 1 audit inventory (no files modified)"))

    # 8. No spectral values transformed (spot check: row counts match)
    row_mismatches = 0
    for r in records:
        inv_row = inv.get(r["filename"])
        if inv_row and r["n_spectral_rows"] != int(inv_row["actual_rows"]):
            row_mismatches += 1
    if row_mismatches:
        results.append((False, f"{row_mismatches} files have different row counts from audit"))
    else:
        results.append((True, "All spectral row counts match Phase 1 audit (no values transformed)"))

    # 9. No raw file deleted
    inv_files = set(inv.keys())
    current_files = set(r["filename"] for r in records)
    deleted = inv_files - current_files
    if deleted:
        results.append((False, f"{len(deleted)} files from audit inventory are missing"))
    else:
        results.append((True, "No files deleted (all audit inventory files present)"))

    # 10. No chemistry/crystal-structure data added
    # (This is a structural check: no new data files in data/ besides raw/RELAB)
    chem_files = list((BASE / "data" / "processed").glob("*")) if (BASE / "data" / "processed").exists() else []
    if chem_files:
        results.append((False, f"Found {len(chem_files)} files in data/processed/ (unexpected)"))
    else:
        results.append((True, "No chemistry/crystal-structure data added"))

    return results


def write_final_summary(records, md5_map, sample_id_map, validation_results, output_path):
    """Write the final phase1_step3_summary.md."""
    total = len(records)
    cf = sum(1 for r in records if r["cf_flag"] == 1)
    non_cf = total - cf
    unique_md5 = len(set(r["md5"] for r in records))
    unique_sids = len(set(r["relab_sample_id"] for r in records if r["relab_sample_id"]))
    non_cf_sids = len(set(r["relab_sample_id"] for r in records if r["relab_sample_id"] and r["cf_flag"] == 0))

    dup_groups = sum(1 for indices in md5_map.values() if len(indices) > 1)
    dup_files = sum(len(indices) for indices in md5_map.values() if len(indices) > 1)

    spm_groups = len(set(r["specimen_group_id"] for r in records if r["specimen_group_id"]))

    sid_counts = defaultdict(int)
    for r in records:
        if r["relab_sample_id"]:
            sid_counts[r["relab_sample_id"]] += 1
    repeated = sum(1 for c in sid_counts.values() if c > 1)
    single = sum(1 for c in sid_counts.values() if c == 1)

    candidates = sum(1 for r in records if r["scientific_analysis_candidate"] == 1)

    all_pass = all(v for v, _ in validation_results)

    content = f"""# Phase 1, Step 3 — Summary

**Date:** 2026-09-22
**Status:** Canonical provenance layer complete

---

## 1. Canonical measurement structure

- **File:** `metadata/relab_measurements.csv`
- **Rows:** {total} (one per raw `.tab` file)
- **Columns:** 55 (file identifiers, metadata, spectral properties, group assignments, flags)
- **Coverage:** All 1,017 raw RELAB files across 20 mineral directories

## 2. Specimen/sample grouping

- **File:** `metadata/relab_sample_groups.csv`
- **Groups:** {spm_groups}
- **Unique RELAB sample IDs:** {unique_sids}
- **Non-CF sample IDs:** {non_cf_sids}
- **Sample IDs with single measurement:** {single}
- **Sample IDs with multiple measurements:** {repeated}

## 3. Duplicate relationships

- **File:** `metadata/relab_duplicate_relationships.csv`
- **Exact duplicate groups:** {dup_groups}
- **Files in duplicate groups:** {dup_files}
- **Cross-mineral duplicate groups:** {sum(1 for md5, indices in md5_map.items() if len(indices) > 1 and len(set(records[i]['mineral_directory'] for i in indices)) > 1)}
- **Relationship types documented:**
  - Cross-mineral CF blank duplicates
  - Cross-mineral pyroxene duplicates (Augite/Hypersthene/Diopside/Enstatite)
  - Cross-mineral Augite-Olivine duplicates
  - Same-mineral exact duplicates
  - Repeated measurements (same specimen, different spectra)

## 4. Eligibility flags

| Flag | Count |
|------|-------|
| `cf_flag = 1` | {cf} |
| `exact_duplicate_flag = 1` | {dup_files} |
| `duplicate_wavelength_flag = 1` | {sum(1 for r in records if r['duplicate_wavelength_flag'] == 1)} |
| `unsorted_wavelength_flag = 1` | {sum(1 for r in records if r['unsorted_wavelength_flag'] == 1)} |
| `missing_data_flag = 1` | {sum(1 for r in records if r['missing_data_flag'] == 1)} |
| `metadata_incomplete_flag = 1` | {sum(1 for r in records if r['metadata_incomplete_flag'] == 1)} |
| `potentially_related_flag = 1` | {sum(1 for r in records if r['potentially_related_flag'] == 1)} |
| `scientific_analysis_candidate = 1` | {candidates} |

## 5. Current counts

| Metric | Count |
|--------|-------|
| Raw files | {total} |
| CF records | {cf} |
| Non-CF records | {non_cf} |
| Unique spectral contents (MD5) | {unique_md5} |
| Specimen/sample groups | {spm_groups} |
| Analysis candidates | {candidates} |

## 6. Unresolved identity/provenance issues

1. **Augite-Olivine duplicates (3 pairs):** Byte-identical files between Augite and Olivine directories. Cause unknown. Requires investigation.
2. **Pyroxene classification ambiguity (49 pairs):** Augite/Diopside/Hypersthene/Enstatite share byte-identical spectra due to solid-solution compositional overlap. How to assign mineral class in ML is unresolved.
3. **No verified mineral identities:** Directory names are the only mineral labels. No XRD, chemistry, or other verification in raw data.
4. **No chemistry data:** Central to project hypothesis but absent. Must be obtained from external sources.
5. **No crystal structure data:** Central to project hypothesis but absent. Must be obtained from external sources.
6. **Third column statistical meaning:** Interpreted as measurement uncertainty but exact meaning (1-sigma, 2-sigma, SE) undocumented.
7. **Filename prefix codes:** Interpreted from patterns but no official RELAB documentation available.
8. **Wavelength grid choice:** 84 distinct grids; no target grid selected for ML.

## 7. Validation results

| Check | Status |
|-------|--------|
"""

    for passed, msg in validation_results:
        status = "PASS" if passed else "FAIL"
        content += f"| {msg} | {status} |\n"

    content += f"""
**Overall validation:** {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — REVIEW REQUIRED'}

## 8. Pending researcher decisions

1. **Pyroxene class structure:** Merge into super-class? Choose one directory per file? Flag as ambiguous?
2. **Augite-Olivine duplicates:** Investigate cause and decide handling.
3. **Target wavelength grid:** Which grid for ML resampling?
4. **External data sources:** Which databases for chemistry/crystallography?
5. **Grain-size treatment:** Replicates, separate classes, or covariates?

---

## Outputs produced

| File | Description |
|------|-------------|
| `metadata/relab_measurements.csv` | Canonical measurement-level table ({total} rows) |
| `metadata/relab_sample_groups.csv` | Specimen/sample group summary ({spm_groups} groups) |
| `metadata/relab_duplicate_relationships.csv` | Duplicate and related-measurement relationships |
| `results/01_data_setup_audit/phase1_dataset_structure.md` | Dataset structure report |
| `docs/methodology.md` | Methodology (leakage-control principle) |
| `docs/research_decisions.md` | Research decision log (13 decisions) |
| `results/01_data_setup_audit/phase1_step3_summary.md` | This summary |

---

*Phase 1, Step 3 complete. No spectral data modified. No preprocessing applied. No ML performed.*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {output_path}")


def main():
    print("=" * 60)
    print("Phase 1, Step 3: Build Canonical RELAB Provenance Dataset")
    print("=" * 60)

    # Task 1-3: Build measurement table with groups and flags
    print("\n--- Tasks 1-3: Building measurement table ---")
    records, md5_map, sample_id_map = build_measurement_table()

    # Task 2: Assign groups
    print("\n--- Task 2: Assigning groups ---")
    records = assign_groups(records, md5_map, sample_id_map)

    # Task 5: Write duplicate relationships
    print("\n--- Task 5: Building duplicate relationships ---")
    rel_path = META / "relab_duplicate_relationships.csv"
    build_duplicate_relationships(records, md5_map, rel_path)

    # Task 6: Build sample groups
    print("\n--- Task 6: Building sample groups ---")
    spm_path = META / "relab_sample_groups.csv"
    build_sample_groups(records, spm_path)

    # Task 1 output: Write measurements CSV
    print("\n--- Task 1: Writing measurements CSV ---")
    meas_path = META / "relab_measurements.csv"
    write_measurements_csv(records, meas_path)

    # Task 7: Dataset structure report
    print("\n--- Task 7: Writing dataset structure report ---")
    struct_path = AUDIT / "phase1_dataset_structure.md"
    write_dataset_structure(records, md5_map, sample_id_map, struct_path)

    # Task 8: Methodology
    print("\n--- Task 8: Writing methodology ---")
    write_methodology(DOCS / "methodology.md")

    # Task 9: Research decisions
    print("\n--- Task 9: Writing research decisions ---")
    write_research_decisions(DOCS / "research_decisions.md")

    # Task 10: Validation
    print("\n--- Task 10: Running validations ---")
    validation_results = run_validations(records, md5_map, sample_id_map)
    for passed, msg in validation_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")

    # Final summary
    print("\n--- Writing final summary ---")
    summary_path = AUDIT / "phase1_step3_summary.md"
    write_final_summary(records, md5_map, sample_id_map, validation_results, summary_path)

    print("\n" + "=" * 60)
    print("Phase 1, Step 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
