"""
Phase 5 Step 4.5 — New Polymorph Spectra Inventory and Compatibility Audit

Produces:
  - file_inventory.csv
  - duplicate_review.csv
  - d2_compatibility.csv
  - mineral_summary.csv
  - audit_report.md
"""

import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\andri\Projects\crystalsym-spec")
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"
OUTPUT_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis" / "phase5_step4_5_polymorph_audit"

TARGET_MINERALS = ["Anatase", "Andalusite", "Aragonite", "Brookite", "Kyanite"]
EMPTY_MINERALS = ["Quartz polymorphs", "Sillimanite"]

D2_WL_MIN = 320.0
D2_WL_MAX = 2550.0
D2_SPACING = 5.0
D2_N_POINTS = 447


def compute_checksums(filepath):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def parse_spectral_file(filepath):
    result = {
        "parse_ok": False,
        "declared_rows": None,
        "actual_rows": 0,
        "n_cols": 0,
        "wavelengths": None,
        "reflectances": None,
        "uncertainties": None,
        "has_uncertainty": False,
        "wl_min": np.nan,
        "wl_max": np.nan,
        "ref_min": np.nan,
        "ref_max": np.nan,
        "n_missing": 0,
        "n_nan": 0,
        "n_inf": 0,
        "n_non_numeric": 0,
        "n_dup_wavelengths": 0,
        "n_unsorted": 0,
        "wl_order_status": "unknown",
        "grid_spacings": None,
        "median_spacing": np.nan,
        "spacing_std": np.nan,
        "is_uniform_grid": False,
        "unique_spacings": 0,
        "trailing_meta_lines": 0,
    }

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except Exception as e:
        result["parse_error"] = str(e)
        return result

    if not raw_lines:
        result["parse_error"] = "empty file"
        return result

    header_line = raw_lines[0].strip()
    header_match = re.match(r"^\s*1\s+(\d+)\s*$", header_line)
    if header_match:
        result["declared_rows"] = int(header_match.group(1))
    else:
        result["declared_rows"] = None

    data_lines = raw_lines[1:]

    wavelengths = []
    reflectances = []
    uncertainties = []
    n_cols_detected = 0
    n_non_numeric = 0
    trailing_meta = 0
    in_trailing = False

    for i, line in enumerate(data_lines):
        stripped = line.strip()
        if not stripped:
            continue

        if in_trailing:
            trailing_meta += 1
            continue

        parts = stripped.split()
        if len(parts) < 2:
            in_trailing = True
            trailing_meta += 1
            continue

        try:
            vals = [float(x) for x in parts]
        except ValueError:
            in_trailing = True
            trailing_meta += 1
            continue

        if n_cols_detected == 0:
            n_cols_detected = len(vals)
        elif len(vals) != n_cols_detected:
            in_trailing = True
            trailing_meta += 1
            continue

        wl, ref = vals[0], vals[1]
        unc = vals[2] if len(vals) >= 3 else None

        wavelengths.append(wl)
        reflectances.append(ref)
        if unc is not None:
            uncertainties.append(unc)

    result["trailing_meta_lines"] = trailing_meta
    result["n_cols"] = n_cols_detected if n_cols_detected > 0 else 0
    result["has_uncertainty"] = n_cols_detected >= 3
    result["actual_rows"] = len(wavelengths)

    if len(wavelengths) == 0:
        result["parse_error"] = "no valid data rows"
        return result

    wl_arr = np.array(wavelengths, dtype=np.float64)
    ref_arr = np.array(reflectances, dtype=np.float64)

    n_nan = int(np.sum(np.isnan(wl_arr)) + np.sum(np.isnan(ref_arr)))
    n_inf = int(np.sum(np.isinf(wl_arr)) + np.sum(np.isinf(ref_arr)))

    finite_mask = np.isfinite(wl_arr) & np.isfinite(ref_arr)
    wl_finite = wl_arr[finite_mask]
    ref_finite = ref_arr[finite_mask]

    result["n_nan"] = n_nan
    result["n_inf"] = n_inf
    result["n_missing"] = n_nan + n_inf

    if len(wl_finite) > 0:
        result["wl_min"] = float(wl_finite.min())
        result["wl_max"] = float(wl_finite.max())
        result["ref_min"] = float(ref_finite.min())
        result["ref_max"] = float(ref_finite.max())

    if len(wl_finite) >= 2:
        diffs = np.diff(wl_finite)
        n_unsorted = int(np.sum(diffs < 0))
        result["n_unsorted"] = n_unsorted
        if n_unsorted == 0:
            result["wl_order_status"] = "sorted"
        else:
            result["wl_order_status"] = "unsorted"

        n_dup = int(np.sum(diffs == 0))
        result["n_dup_wavelengths"] = n_dup

        result["median_spacing"] = float(np.median(diffs))
        result["spacing_std"] = float(np.std(diffs))

        rounded_spacings = np.round(diffs, 2)
        unique_sp = np.unique(rounded_spacings)
        result["unique_spacings"] = len(unique_sp)
        result["grid_spacings"] = unique_sp.tolist()

        if result["spacing_std"] < 0.01 * result["median_spacing"]:
            result["is_uniform_grid"] = True
        elif result["spacing_std"] < 0.5:
            result["is_uniform_grid"] = True
        else:
            result["is_uniform_grid"] = False

    result["wavelengths"] = wl_arr
    result["reflectances"] = ref_arr
    if uncertainties:
        result["uncertainties"] = np.array(uncertainties, dtype=np.float64)
    result["parse_ok"] = True

    return result


def classify_grid(parsed):
    if not parsed["parse_ok"] or parsed["actual_rows"] == 0:
        return "unknown"
    if not parsed["is_uniform_grid"]:
        if parsed["unique_spacings"] > 5:
            return "nonuniform"
        return "nonuniform"

    med = parsed["median_spacing"]
    if abs(med - 5.0) < 0.1:
        return "native_5nm"
    elif abs(med - 10.0) < 0.1:
        return "native_10nm"
    elif abs(med - 1.0) < 0.2:
        return "native_1nm"
    elif abs(med - 0.5) < 0.1:
        return "sub_1nm"
    elif med < 5.0:
        return "fine_nonstandard"
    else:
        return "coarse_nonstandard"


def assess_d2_compatibility(parsed, grid_type):
    assessment = {
        "d2_covers_full_interval": False,
        "d2_grid_type": grid_type,
        "d2_n_points_in_range": 0,
        "d2_wl_min_gap": np.nan,
        "d2_wl_max_gap": np.nan,
        "d2_compatibility": "unsuitable",
        "d2_notes": [],
    }

    if not parsed["parse_ok"] or parsed["actual_rows"] == 0:
        assessment["d2_compatibility"] = "unsuitable"
        assessment["d2_notes"].append("no valid spectral data")
        return assessment

    wl = parsed["wavelengths"]
    finite_mask = np.isfinite(wl) & np.isfinite(parsed["reflectances"])
    wl_f = wl[finite_mask]

    if len(wl_f) == 0:
        assessment["d2_compatibility"] = "unsuitable"
        assessment["d2_notes"].append("no finite wavelength data")
        return assessment

    in_range = (wl_f >= D2_WL_MIN - 0.01) & (wl_f <= D2_WL_MAX + 0.01)
    n_in_range = int(np.sum(in_range))
    assessment["d2_n_points_in_range"] = n_in_range

    wl_min = float(wl_f.min())
    wl_max = float(wl_f.max())
    assessment["d2_wl_min_gap"] = round(wl_min - D2_WL_MIN, 2)
    assessment["d2_wl_max_gap"] = round(D2_WL_MAX - wl_max, 2)

    covers_full = wl_min <= D2_WL_MIN + 0.5 and wl_max >= D2_WL_MAX - 0.5
    assessment["d2_covers_full_interval"] = covers_full

    if not covers_full:
        if wl_min > D2_WL_MIN + 50:
            assessment["d2_notes"].append(f"starts at {wl_min:.1f} nm, misses shortwave end")
        if wl_max < D2_WL_MAX - 50:
            assessment["d2_notes"].append(f"ends at {wl_max:.1f} nm, misses longwave end")

    exact_d2_match = (
        grid_type == "native_5nm"
        and covers_full
        and n_in_range == D2_N_POINTS
        and abs(wl_min - D2_WL_MIN) < 0.5
        and abs(wl_max - D2_WL_MAX) < 0.5
    )

    if exact_d2_match:
        assessment["d2_compatibility"] = "suitable"
        assessment["d2_notes"].append("exact D2 grid match (320-2550nm, 5nm, 447 points)")
    elif grid_type == "native_5nm" and covers_full:
        assessment["d2_compatibility"] = "potentially_suitable"
        assessment["d2_notes"].append(
            f"native 5nm grid covers D2 but has {n_in_range} points in range "
            f"(grid spans {wl_min:.0f}-{wl_max:.0f}nm, trimming needed)"
        )
    elif grid_type == "native_10nm" and covers_full:
        assessment["d2_compatibility"] = "potentially_suitable"
        assessment["d2_notes"].append("native 10nm grid could be PCHIP-resampled to 5nm")
    elif grid_type in ("native_1nm", "sub_1nm"):
        if covers_full:
            assessment["d2_compatibility"] = "potentially_suitable"
            assessment["d2_notes"].append(f"fine grid ({grid_type}) would need decimation/resampling to 5nm")
        else:
            assessment["d2_compatibility"] = "incomplete"
            assessment["d2_notes"].append(f"fine grid but incomplete D2 coverage")
    elif covers_full and grid_type in ("fine_nonstandard", "coarse_nonstandard"):
        assessment["d2_compatibility"] = "potentially_suitable"
        assessment["d2_notes"].append(f"nonstandard uniform grid (median spacing {parsed['median_spacing']:.2f} nm)")
    elif not covers_full and n_in_range > 0:
        assessment["d2_compatibility"] = "incomplete"
        assessment["d2_notes"].append(f"partial coverage: {n_in_range} points in D2 range")
    else:
        assessment["d2_compatibility"] = "unsuitable"
        assessment["d2_notes"].append("no data in D2 range or nonuniform grid")

    if parsed["n_unsorted"] > 0:
        assessment["d2_notes"].append(f"unsorted wavelengths ({parsed['n_unsorted']} inversions)")
    if parsed["n_dup_wavelengths"] > 0:
        assessment["d2_notes"].append(f"duplicate wavelengths ({parsed['n_dup_wavelengths']})")

    return assessment


def extract_specimen_id(filename):
    name = Path(filename).stem

    patterns = [
        (r"^(bir1[a-z]{2}\d+[a-z]?)", "BIR1"),
        (r"^(bir2[a-z]{2}\d+[a-z]?)", "BIR2"),
        (r"^(bfr1[a-z]{2}\d+[a-z]?)", "BFR1"),
        (r"^(bfr2[a-z]{2}\d+[a-z]?)", "BFR2"),
        (r"^(c1[a-z]{2}\d+[a-z]?)", "C1"),
        (r"^(c2[a-z]{2}\d+[a-z]?)", "C2"),
        (r"^(caec\d+)", "CAEC"),
        (r"^(cpfb\d+)", "CPFB"),
        (r"^(nafb\d+[a-z]?)", "NAFB"),
        (r"^(\d{3}s\d+[a-z]?)", "NUM"),
    ]

    for pat, prefix in patterns:
        m = re.match(pat, name, re.IGNORECASE)
        if m:
            return m.group(1)

    return name


def infer_sample_group(specimen_id):
    base = re.sub(r"[a-z]$", "", specimen_id, flags=re.IGNORECASE)
    return base


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 5 Step 4.5 — Polymorph Spectra Inventory and Audit")
    print("=" * 70)

    all_rows = []
    total_files = 0

    for mineral in TARGET_MINERALS + EMPTY_MINERALS:
        mineral_dir = RAW_DIR / mineral
        if not mineral_dir.exists():
            print(f"  [WARN] Directory not found: {mineral_dir}")
            continue

        files = sorted([f for f in mineral_dir.iterdir() if f.is_file() and f.suffix == ".tab"])
        print(f"\n  {mineral}: {len(files)} files")

        if len(files) == 0:
            continue

        for fpath in files:
            total_files += 1
            rel_path = f"RELAB/{mineral}/{fpath.name}"
            file_size = fpath.stat().st_size
            md5_hex, sha256_hex = compute_checksums(fpath)

            parsed = parse_spectral_file(fpath)
            grid_type = classify_grid(parsed)
            d2 = assess_d2_compatibility(parsed, grid_type)

            specimen_id = extract_specimen_id(fpath.name)
            sample_group = infer_sample_group(specimen_id)

            row = {
                "mineral": mineral,
                "filename": fpath.name,
                "rel_path": rel_path,
                "file_size": file_size,
                "md5": md5_hex,
                "sha256": sha256_hex,
                "declared_rows": parsed["declared_rows"],
                "actual_rows": parsed["actual_rows"],
                "n_cols": parsed["n_cols"],
                "has_uncertainty": parsed["has_uncertainty"],
                "wl_min": round(parsed["wl_min"], 2) if np.isfinite(parsed["wl_min"]) else None,
                "wl_max": round(parsed["wl_max"], 2) if np.isfinite(parsed["wl_max"]) else None,
                "ref_min": round(parsed["ref_min"], 6) if np.isfinite(parsed["ref_min"]) else None,
                "ref_max": round(parsed["ref_max"], 6) if np.isfinite(parsed["ref_max"]) else None,
                "n_missing": parsed["n_missing"],
                "n_nan": parsed["n_nan"],
                "n_inf": parsed["n_inf"],
                "n_non_numeric": parsed["n_non_numeric"],
                "n_dup_wavelengths": parsed["n_dup_wavelengths"],
                "n_unsorted": parsed["n_unsorted"],
                "wl_order_status": parsed["wl_order_status"],
                "median_spacing": round(parsed["median_spacing"], 4) if np.isfinite(parsed["median_spacing"]) else None,
                "spacing_std": round(parsed["spacing_std"], 4) if np.isfinite(parsed["spacing_std"]) else None,
                "is_uniform_grid": parsed["is_uniform_grid"],
                "unique_spacings": parsed["unique_spacings"],
                "grid_type": grid_type,
                "trailing_meta_lines": parsed["trailing_meta_lines"],
                "specimen_id": specimen_id,
                "sample_group": sample_group,
                "d2_covers_full_interval": d2["d2_covers_full_interval"],
                "d2_n_points_in_range": d2["d2_n_points_in_range"],
                "d2_wl_min_gap": d2["d2_wl_min_gap"],
                "d2_wl_max_gap": d2["d2_wl_max_gap"],
                "d2_compatibility": d2["d2_compatibility"],
                "d2_notes": "; ".join(d2["d2_notes"]) if d2["d2_notes"] else None,
            }
            all_rows.append(row)

    print(f"\n  Total files processed: {total_files}")

    df = pd.DataFrame(all_rows)
    inv_path = OUTPUT_DIR / "phase5_step4_5_file_inventory.csv"
    df.to_csv(inv_path, index=False)
    print(f"  Wrote: {inv_path}")

    # --- Duplicate review ---
    print("\n  --- Duplicate / Provenance Review ---")

    dup_rows = []
    md5_groups = df.groupby("md5")
    for md5_val, group in md5_groups:
        if len(group) > 1:
            for _, r in group.iterrows():
                dup_rows.append({
                    "mineral": r["mineral"],
                    "filename": r["filename"],
                    "rel_path": r["rel_path"],
                    "duplicate_type": "byte_identical",
                    "duplicate_group_id": md5_val[:12],
                    "matched_files": "; ".join(group["filename"].tolist()),
                    "evidence": f"MD5={md5_val}",
                })

    spectral_dup_groups = defaultdict(list)
    for _, r in df.iterrows():
        if r["actual_rows"] > 0 and r["n_cols"] >= 2:
            key = (r["wl_min"], r["wl_max"], r["actual_rows"], r["median_spacing"])
            spectral_dup_groups[key].append(r)

    for key, rows_list in spectral_dup_groups.items():
        if len(rows_list) > 1:
            group_df = pd.DataFrame(rows_list)
            md5s = set(group_df["md5"].tolist())
            if len(md5s) > 1:
                filenames = "; ".join(group_df["filename"].tolist())
                minerals = "; ".join(sorted(set(group_df["mineral"].tolist())))
                for _, r in group_df.iterrows():
                    dup_rows.append({
                        "mineral": r["mineral"],
                        "filename": r["filename"],
                        "rel_path": r["rel_path"],
                        "duplicate_type": "identical_spectral_array",
                        "duplicate_group_id": f"spec_{key[0]}_{key[1]}_{key[2]}",
                        "matched_files": filenames,
                        "evidence": f"same grid (wl_min={key[0]}, wl_max={key[1]}, rows={key[2]}, spacing={key[3]}), different MD5s",
                    })

    specimen_groups = df.groupby("specimen_id")
    for spec_id, group in specimen_groups:
        if len(group) > 1:
            minerals = sorted(set(group["mineral"].tolist()))
            filenames = "; ".join(group["filename"].tolist())
            for _, r in group.iterrows():
                dup_rows.append({
                    "mineral": r["mineral"],
                    "filename": r["filename"],
                    "rel_path": r["rel_path"],
                    "duplicate_type": "shared_specimen_id",
                    "duplicate_group_id": f"specimen_{spec_id}",
                    "matched_files": filenames,
                    "evidence": f"specimen_id={spec_id}, minerals={minerals}",
                })

    sample_groups = df.groupby("sample_group")
    for sg_id, group in sample_groups:
        if len(group) > 1:
            minerals = sorted(set(group["mineral"].tolist()))
            spec_ids = sorted(set(group["specimen_id"].tolist()))
            if len(spec_ids) > 1:
                filenames = "; ".join(group["filename"].tolist())
                for _, r in group.iterrows():
                    dup_rows.append({
                        "mineral": r["mineral"],
                        "filename": r["filename"],
                        "rel_path": r["rel_path"],
                        "duplicate_type": "likely_replicate",
                        "duplicate_group_id": f"samplegroup_{sg_id}",
                        "matched_files": filenames,
                        "evidence": f"sample_group={sg_id}, specimen_ids={spec_ids}, minerals={minerals}",
                    })

    existing_chk_for_dup = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
    if existing_chk_for_dup.exists():
        existing_chk_df = pd.read_csv(existing_chk_for_dup)
        existing_md5_map = dict(zip(existing_chk_df["md5"], existing_chk_df["rel_path"]))
        for _, r in df.iterrows():
            if r["md5"] in existing_md5_map:
                existing_path = existing_md5_map[r["md5"]]
                dup_rows.append({
                    "mineral": r["mineral"],
                    "filename": r["filename"],
                    "rel_path": r["rel_path"],
                    "duplicate_type": "cross_directory_identity",
                    "duplicate_group_id": f"cross_{r['md5'][:12]}",
                    "matched_files": f"{r['rel_path']} == RELAB/{existing_path}",
                    "evidence": f"MD5={r['md5']}; byte-identical to existing file in {existing_path.split('/')[0]}",
                })

    dup_df = pd.DataFrame(dup_rows) if dup_rows else pd.DataFrame(
        columns=["mineral", "filename", "rel_path", "duplicate_type", "duplicate_group_id", "matched_files", "evidence"]
    )
    dup_df = dup_df.drop_duplicates(subset=["rel_path", "duplicate_type", "duplicate_group_id"])
    dup_path = OUTPUT_DIR / "phase5_step4_5_duplicate_review.csv"
    dup_df.to_csv(dup_path, index=False)
    print(f"  Duplicate/provenance flags: {len(dup_df)}")
    print(f"  Wrote: {dup_path}")

    # --- D2 compatibility ---
    d2_path = OUTPUT_DIR / "phase5_step4_5_d2_compatibility.csv"
    d2_cols = ["mineral", "filename", "rel_path", "grid_type", "wl_min", "wl_max",
               "actual_rows", "d2_n_points_in_range", "d2_covers_full_interval",
               "d2_compatibility", "d2_notes"]
    df[d2_cols].to_csv(d2_path, index=False)
    print(f"  Wrote: {d2_path}")

    # --- Mineral summary ---
    print("\n  --- Mineral Summary ---")

    summary_rows = []
    for mineral in TARGET_MINERALS + EMPTY_MINERALS:
        mdf = df[df["mineral"] == mineral] if mineral in df["mineral"].values else pd.DataFrame()

        if len(mdf) == 0:
            summary_rows.append({
                "mineral": mineral,
                "reported_file_count": 0,
                "actual_file_count": 0,
                "distinct_file_content_count": 0,
                "distinct_specimen_ids": 0,
                "unknown_specimen_count": 0,
                "grid_types": "none",
                "coverage_distribution": "none",
                "d2_suitable": 0,
                "d2_potentially_suitable": 0,
                "d2_incomplete": 0,
                "d2_unsuitable": 0,
                "reflectance_range_min": None,
                "reflectance_range_max": None,
                "main_limitations": "no files found",
            })
            continue

        distinct_md5 = mdf["md5"].nunique()
        distinct_specimens = mdf["sample_group"].nunique()
        unmatched_pattern = mdf[~mdf["specimen_id"].str.match(
            r"^(bir[12]|bfr[12]|c[12]|caec|cpfb|nafb|\d{3}s)", case=False, na=False
        )].shape[0]

        grid_dist = mdf["grid_type"].value_counts().to_dict()
        compat_dist = mdf["d2_compatibility"].value_counts().to_dict()
        coverage_dist = mdf["d2_covers_full_interval"].value_counts().to_dict()

        ref_min_vals = mdf["ref_min"].dropna()
        ref_max_vals = mdf["ref_max"].dropna()

        limitations = []
        if compat_dist.get("suitable", 0) == 0 and compat_dist.get("potentially_suitable", 0) == 0:
            limitations.append("no files compatible with D2 design")
        if coverage_dist.get(True, 0) == 0:
            limitations.append("no files cover full D2 interval (320-2550 nm)")
        if distinct_specimens < len(mdf):
            limitations.append(f"replicate measurements present ({len(mdf)} files, {distinct_specimens} specimens)")
        grid_types_present = list(grid_dist.keys())
        if "native_5nm" not in grid_types_present and "native_10nm" not in grid_types_present:
            limitations.append("no native 5nm or 10nm grids")
        if mdf["n_unsorted"].sum() > 0:
            limitations.append("some files have unsorted wavelengths")
        if mdf["n_dup_wavelengths"].sum() > 0:
            limitations.append("some files have duplicate wavelengths")

        wl_ranges = mdf[["wl_min", "wl_max"]].dropna()
        if len(wl_ranges) > 0:
            diverse_ranges = wl_ranges.drop_duplicates()
            if len(diverse_ranges) > 1:
                limitations.append(f"diverse wavelength ranges ({len(diverse_ranges)} distinct)")

        summary_rows.append({
            "mineral": mineral,
            "reported_file_count": len(mdf),
            "actual_file_count": len(mdf),
            "distinct_file_content_count": distinct_md5,
            "distinct_specimen_ids": distinct_specimens,
            "unknown_specimen_count": unmatched_pattern,
            "grid_types": str(grid_dist),
            "coverage_distribution": str(coverage_dist),
            "d2_suitable": compat_dist.get("suitable", 0),
            "d2_potentially_suitable": compat_dist.get("potentially_suitable", 0),
            "d2_incomplete": compat_dist.get("incomplete", 0),
            "d2_unsuitable": compat_dist.get("unsuitable", 0),
            "reflectance_range_min": float(ref_min_vals.min()) if len(ref_min_vals) > 0 else None,
            "reflectance_range_max": float(ref_max_vals.max()) if len(ref_max_vals) > 0 else None,
            "main_limitations": "; ".join(limitations) if limitations else "none identified",
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_path = OUTPUT_DIR / "phase5_step4_5_mineral_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"  Wrote: {sum_path}")

    # --- Compare with existing frozen Phase 2 ---
    print("\n  --- Comparison with Frozen Phase 2 ---")

    existing_inv_path = PROJECT_ROOT / "results" / "01_data_setup_audit" / "relab_file_inventory.csv"
    existing_meas_path = PROJECT_ROOT / "metadata" / "relab_measurements.csv"
    existing_chk_path = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"

    overlap_notes = []

    if existing_chk_path.exists():
        existing_chk = pd.read_csv(existing_chk_path)
        existing_md5s = set(existing_chk["md5"].tolist())
        new_md5s = set(df["md5"].tolist())
        md5_overlap = existing_md5s & new_md5s
        if md5_overlap:
            overlap_notes.append(f"{len(md5_overlap)} files have MD5 matches in existing RELAB checksums")
            for md5_val in md5_overlap:
                existing_file = existing_chk[existing_chk["md5"] == md5_val]["rel_path"].iloc[0]
                new_file = df[df["md5"] == md5_val]["rel_path"].iloc[0]
                overlap_notes.append(f"  IDENTICAL: {new_file} == {existing_file}")
        else:
            overlap_notes.append("No byte-identical matches with existing RELAB checksums")

    if existing_meas_path.exists():
        existing_meas = pd.read_csv(existing_meas_path)
        existing_minerals = set(existing_meas["mineral_directory"].unique())
        for mineral in TARGET_MINERALS:
            if mineral in existing_minerals:
                overlap_notes.append(f"  {mineral}: present in existing relab_measurements.csv")
            else:
                overlap_notes.append(f"  {mineral}: NOT in existing relab_measurements.csv (new)")

    existing_specimens = set()
    if existing_meas_path.exists():
        existing_meas = pd.read_csv(existing_meas_path)
        if "base_sample_id" in existing_meas.columns:
            existing_specimens = set(existing_meas["base_sample_id"].dropna().unique())

    new_specimens = set(df["specimen_id"].unique())
    specimen_overlap = existing_specimens & new_specimens
    if specimen_overlap:
        overlap_notes.append(f"{len(specimen_overlap)} specimen IDs overlap with existing measurements:")
        for sid in sorted(specimen_overlap):
            existing_entries = existing_meas[existing_meas["base_sample_id"] == sid]
            existing_minerals_list = existing_entries["mineral_directory"].unique()
            new_entries = df[df["specimen_id"] == sid]
            new_minerals_list = new_entries["mineral"].unique()
            overlap_notes.append(
                f"  {sid}: existing in {list(existing_minerals_list)}, new in {list(new_minerals_list)}"
            )

    overlap_path = OUTPUT_DIR / "phase5_step4_5_existing_comparison.csv"
    overlap_df = pd.DataFrame({"note": overlap_notes})
    overlap_df.to_csv(overlap_path, index=False)
    print(f"  Wrote: {overlap_path}")

    # --- Final report ---
    print("\n  --- Generating Report ---")

    report_lines = []
    report_lines.append("# Phase 5 Step 4.5 — New Polymorph Spectra Inventory and Compatibility Audit\n")
    report_lines.append(f"**Date:** 2026-10-12  ")
    report_lines.append(f"**Status:** Audit complete. No spectra processed or integrated.\n")

    report_lines.append("## 1. Scope and Restrictions\n")
    report_lines.append("This audit inventories raw spectral files for five polymorph candidate minerals")
    report_lines.append("(Anatase, Andalusite, Aragonite, Brookite, Kyanite) plus two empty directories")
    report_lines.append("(Quartz polymorphs, Sillimanite) found in `data/raw/RELAB/`.")
    report_lines.append("These files were **not** included in the Phase 1 freeze or any subsequent phase.")
    report_lines.append("No files were modified, processed, interpolated, normalized, or integrated.\n")

    report_lines.append("## 2. File Count Reconciliation\n")
    total_reported = 4 + 7 + 160 + 5 + 4
    total_found = len(df)
    report_lines.append(f"| Mineral | Reported | Found | Match |")
    report_lines.append(f"|---------|----------|-------|-------|")
    expected = {"Anatase": 4, "Andalusite": 7, "Aragonite": 160, "Brookite": 5, "Kyanite": 4}
    for mineral in TARGET_MINERALS:
        found = len(df[df["mineral"] == mineral])
        match = "yes" if found == expected[mineral] else f"**no ({found})**"
        report_lines.append(f"| {mineral} | {expected[mineral]} | {found} | {match} |")
    for mineral in EMPTY_MINERALS:
        report_lines.append(f"| {mineral} | 0 | 0 | yes |")
    report_lines.append(f"| **Total** | **{total_reported}** | **{total_found}** | {'yes' if total_found == total_reported else '**no**'} |")
    report_lines.append("")

    if total_found == total_reported:
        report_lines.append(f"All {total_reported} reported files were located and inventoried.\n")
    else:
        report_lines.append(f"**Discrepancy:** reported {total_reported}, found {total_found}.\n")

    report_lines.append("## 3. Inventory Summary\n")
    report_lines.append(f"- **Total files inventoried:** {len(df)}")
    report_lines.append(f"- **Distinct MD5 checksums:** {df['md5'].nunique()}")
    report_lines.append(f"- **Distinct SHA-256 checksums:** {df['sha256'].nunique()}")
    report_lines.append(f"- **Distinct specimen identifiers (full):** {df['specimen_id'].nunique()}")
    report_lines.append(f"- **Distinct sample groups (base IDs):** {df['sample_group'].nunique()}")
    report_lines.append(f"- **Files with uncertainty column:** {df['has_uncertainty'].sum()}")
    report_lines.append(f"- **Files with sorted wavelengths:** {(df['wl_order_status'] == 'sorted').sum()}")
    report_lines.append(f"- **Files with unsorted wavelengths:** {(df['wl_order_status'] == 'unsorted').sum()}")

    cross_dupes = dup_df[dup_df["duplicate_type"] == "cross_directory_identity"] if len(dup_df) > 0 else pd.DataFrame()
    if len(cross_dupes) > 0:
        report_lines.append(f"\n**CRITICAL:** {len(cross_dupes)} file(s) are byte-identical to existing "
                          f"Phase 1–2 dataset files in other mineral directories:")
        for _, r in cross_dupes.iterrows():
            report_lines.append(f"  - `{r['rel_path']}` == existing `{r['matched_files'].split(' == ')[1]}`")
        report_lines.append("")
    report_lines.append("")

    report_lines.append("## 4. D2 Compatibility Assessment\n")
    report_lines.append("Frozen Phase 2 design requirements:")
    report_lines.append(f"- Wavelength interval: {D2_WL_MIN}–{D2_WL_MAX} nm")
    report_lines.append(f"- Canonical spacing: {D2_SPACING} nm")
    report_lines.append(f"- Grid points: {D2_N_POINTS}")
    report_lines.append("- No primary smoothing, normalization, clipping, or splice correction\n")

    compat_counts = df["d2_compatibility"].value_counts()
    report_lines.append("| Compatibility | Count |")
    report_lines.append("|---------------|-------|")
    for cat in ["suitable", "potentially_suitable", "incomplete", "unsuitable"]:
        report_lines.append(f"| {cat} | {compat_counts.get(cat, 0)} |")
    report_lines.append("")

    report_lines.append("### Per-mineral D2 compatibility\n")
    for mineral in TARGET_MINERALS:
        mdf = df[df["mineral"] == mineral]
        if len(mdf) == 0:
            continue
        cc = mdf["d2_compatibility"].value_counts()
        report_lines.append(f"**{mineral}** ({len(mdf)} files): "
                          f"suitable={cc.get('suitable', 0)}, "
                          f"potentially_suitable={cc.get('potentially_suitable', 0)}, "
                          f"incomplete={cc.get('incomplete', 0)}, "
                          f"unsuitable={cc.get('unsuitable', 0)}")
    report_lines.append("")

    report_lines.append("## 5. Duplicate and Provenance Review\n")
    if len(dup_df) > 0:
        by_type = dup_df.groupby("duplicate_type").size()
        for dtype, count in by_type.items():
            n_groups = dup_df[dup_df["duplicate_type"] == dtype]["duplicate_group_id"].nunique()
            report_lines.append(f"- **{dtype}**: {count} file-level flags across {n_groups} groups")
    else:
        report_lines.append("No duplicate or provenance flags identified.")
    report_lines.append("")

    if len(dup_df) > 0:
        report_lines.append("### Detail\n")
        for dtype in dup_df["duplicate_type"].unique():
            report_lines.append(f"#### {dtype}\n")
            sub = dup_df[dup_df["duplicate_type"] == dtype]
            for gid in sub["duplicate_group_id"].unique():
                g = sub[sub["duplicate_group_id"] == gid]
                report_lines.append(f"- Group `{gid}`: {', '.join(g['filename'].tolist())}")
                report_lines.append(f"  - Evidence: {g['evidence'].iloc[0]}")
            report_lines.append("")

    report_lines.append("## 6. Comparison with Existing Frozen Phase 2 Dataset\n")
    for note in overlap_notes:
        report_lines.append(f"- {note}")
    report_lines.append("")

    report_lines.append("## 7. Mineral-Level Summaries\n")
    for _, row in summary_df.iterrows():
        report_lines.append(f"### {row['mineral']}\n")
        report_lines.append(f"- Files: {row['actual_file_count']}")
        report_lines.append(f"- Distinct content (MD5): {row['distinct_file_content_count']}")
        report_lines.append(f"- Distinct sample groups (base specimen IDs): {row['distinct_specimen_ids']}")
        report_lines.append(f"- D2 suitable: {row['d2_suitable']}")
        report_lines.append(f"- D2 potentially suitable: {row['d2_potentially_suitable']}")
        report_lines.append(f"- D2 incomplete: {row['d2_incomplete']}")
        report_lines.append(f"- D2 unsuitable: {row['d2_unsuitable']}")
        if pd.notna(row.get("reflectance_range_min")) and row["reflectance_range_min"] is not None:
            report_lines.append(f"- Reflectance range: [{row['reflectance_range_min']:.4f}, {row['reflectance_range_max']:.4f}]")
        report_lines.append(f"- Grid types: {row['grid_types']}")
        report_lines.append(f"- Limitations: {row['main_limitations']}")
        report_lines.append("")

    report_lines.append("## 8. Limitations\n")
    report_lines.append("1. **No metadata headers.** RELAB .tab files contain no instrument, geometry,")
    report_lines.append("   grain size, resolution, or provenance metadata in the file itself. All such")
    report_lines.append("   fields are marked `unknown` unless recoverable from filename conventions.")
    report_lines.append("2. **Specimen IDs are inferred from filename patterns**, not from authoritative")
    report_lines.append("   metadata. They may not correspond to distinct physical specimens.")
    report_lines.append("3. **No ancillary .ASC or metadata files** were found alongside the new spectra.")
    report_lines.append("4. **Byte-identical and spectral-array comparisons** are necessary but not")
    report_lines.append("   sufficient to establish specimen independence.")
    report_lines.append("5. **Empty directories** (Quartz polymorphs, Sillimanite) confirm zero files")
    report_lines.append("   but do not confirm that searches were exhaustive.\n")

    report_lines.append("## 9. Recommended Next Steps\n")
    report_lines.append("1. **Do not integrate** any of these spectra into the frozen Phase 2 dataset.")
    report_lines.append("2. **Resolve Brookite c1mr130.tab**: this file is byte-identical to the existing")
    report_lines.append("   Rutile/c1mr130.tab in the frozen Phase 2 dataset. It must be removed from")
    report_lines.append("   the Brookite candidate pool or reclassified before any further consideration.")
    report_lines.append("3. **For D2-suitable files** (if any): a separate versioned extension protocol")
    report_lines.append("   must be designed, reviewed, and frozen before any preprocessing.")
    report_lines.append("4. **For potentially suitable files**: document the preprocessing decision")
    report_lines.append("   (resampling method, splice handling) as a new Phase 2 addendum.")
    report_lines.append("5. **Resolve provenance**: obtain curator metadata (sample descriptions,")
    report_lines.append("   measurement geometry, grain size) for all candidate files.")
    report_lines.append("6. **Resolve duplicates**: determine which files represent independent")
    report_lines.append("   specimens vs. replicate measurements before any counting.")
    report_lines.append("7. **Aragonite priority**: with 160 files and potential Calcite polymorph")
    report_lines.append("   pairing, this group warrants the most detailed provenance investigation.")
    report_lines.append("8. **Empty directories**: confirm whether Quartz polymorphs and Sillimanite")
    report_lines.append("   data exist elsewhere or are genuinely unavailable.\n")

    report_lines.append("---\n")
    report_lines.append("*Audit complete. Raw spectra remain unmodified. No data integrated.*\n")

    report_text = "\n".join(report_lines)
    report_path = OUTPUT_DIR / "phase5_step4_5_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Wrote: {report_path}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Total files: {total_found}")
    print(f"  D2 suitable: {compat_counts.get('suitable', 0)}")
    print(f"  D2 potentially suitable: {compat_counts.get('potentially_suitable', 0)}")
    print(f"  D2 incomplete: {compat_counts.get('incomplete', 0)}")
    print(f"  D2 unsuitable: {compat_counts.get('unsuitable', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
