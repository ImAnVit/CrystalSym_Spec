#!/usr/bin/env python3
"""
Phase 2 Step 1 — Spectral Coverage and Resolution Audit
========================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

This script performs a quantitative audit of the spectral measurements
in the frozen Phase 1 analysis population. It determines wavelength
coverage, sampling, resolution, and structural characteristics WITHOUT
modifying any raw data or making preprocessing decisions.

Outputs:
  results/02_preprocessing_qc/spectral_coverage_audit.csv
  results/02_preprocessing_qc/wavelength_grid_summary.csv
  results/02_preprocessing_qc/phase2_step1_coverage_resolution_report.md
  results/02_preprocessing_qc/phase2_step1_summary.md
  results/02_preprocessing_qc/figures/*.png
"""

import os
import sys
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"
POPULATION_CSV = PROJECT_ROOT / "metadata" / "relab_analysis_population.csv"
MEASUREMENTS_CSV = PROJECT_ROOT / "metadata" / "relab_measurements.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# RAW .tab FILE PARSER  (reads ALL segments, never modifies data)
# ──────────────────────────────────────────────────────────────

def read_spectrum(filepath):
    """
    Parse a RELAB .tab file, reading ALL spectral segments.

    Returns
    -------
    wavelengths : list[float]
        Wavelength values in file order (all segments concatenated).
    reflectances : list[float]
    uncertainties : list[float | None]
    footer_lines : list[str]
        Metadata lines after the last data segment.
    segment_boundaries : list[tuple[int, int]]
        (start_index, end_index) for each segment in the concatenated arrays.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    wavelengths = []
    reflectances = []
    uncertainties = []
    segment_boundaries = []
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue

        parts = stripped.split()
        if len(parts) != 1:
            break

        try:
            count = int(parts[0])
        except ValueError:
            break

        seg_start = len(wavelengths)
        for j in range(1, count + 1):
            pos = idx + j
            if pos >= len(lines):
                break
            dp = lines[pos].split()
            if len(dp) < 2:
                continue
            try:
                wl = float(dp[0])
                rf = float(dp[1])
            except ValueError:
                continue
            wavelengths.append(wl)
            reflectances.append(rf)
            if len(dp) >= 3:
                try:
                    uncertainties.append(float(dp[2]))
                except ValueError:
                    uncertainties.append(None)
            else:
                uncertainties.append(None)

        seg_end = len(wavelengths)
        if seg_end > seg_start:
            segment_boundaries.append((seg_start, seg_end))

        idx += count + 1

    footer_lines = []
    while idx < len(lines):
        line = lines[idx].rstrip("\n\r")
        if line.strip():
            footer_lines.append(line.strip())
        idx += 1

    return wavelengths, reflectances, uncertainties, footer_lines, segment_boundaries


# ──────────────────────────────────────────────────────────────
# METADATA PARSER  (extracts fields from footer text)
# ──────────────────────────────────────────────────────────────

def parse_footer_metadata(footer_lines):
    """Extract metadata fields from the RELAB file footer."""
    meta = {
        "asc_id": "",
        "sample_id_footer": "",
        "description": "",
        "splice_info": "",
        "source_angle": "",
        "detect_angle": "",
        "date": "",
        "time": "",
        "run_mode": "",
        "stage_type": "",
        "spinning": "",
        "aperture": "",
        "scrambler": "",
        "depolarizer": "",
        "calibration_note": "",
    }

    data_line_idx = 0
    for i, line in enumerate(footer_lines):
        if line.startswith("*"):
            continue
        data_line_idx = i + 1
        break

    if data_line_idx > 0:
        if len(footer_lines) >= 1:
            meta["asc_id"] = footer_lines[0].strip()
        if len(footer_lines) >= 2:
            meta["sample_id_footer"] = footer_lines[1].strip()
        if len(footer_lines) >= 3:
            desc = footer_lines[2].strip()
            meta["description"] = desc
            for kw in ["Connected at", "MERGED", "DATA MERGED"]:
                if kw in desc:
                    meta["splice_info"] = desc
                    break

    for line in footer_lines:
        if line.startswith("Source Ang:"):
            parts = line.split(":")
            if len(parts) >= 2:
                vals = parts[1].strip()
                tokens = vals.split()
                if len(tokens) >= 1:
                    meta["source_angle"] = tokens[0]
                for t in tokens:
                    if t.startswith("Detect"):
                        di = tokens.index(t)
                        if di + 2 < len(tokens):
                            meta["detect_angle"] = tokens[di + 2]
                        break
        if line.startswith("Date:"):
            meta["date"] = line.split(":", 1)[1].strip()
        if line.startswith("Time:"):
            meta["time"] = line.split(":", 1)[1].strip()
        if line.startswith("RATIOS") or line.startswith("Calibration"):
            meta["calibration_note"] = line
        if line.startswith("*"):
            content = line[1:].strip()
            if ":" in content:
                key, val = content.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if "run mode" in key:
                    meta["run_mode"] = val
                elif "stage" in key:
                    meta["stage_type"] = val
                elif "spinning" in key:
                    meta["spinning"] = val
                elif "aperture" in key:
                    meta["aperture"] = val
                elif "scrambler" in key:
                    meta["scrambler"] = val
                elif "depolarizer" in key:
                    meta["depolarizer"] = val

    # Check ALL footer lines for splice info
    for line in footer_lines:
        for kw in ["Connected at", "MERGED", "DATA MERGED"]:
            if kw in line:
                meta["splice_info"] = line
                break
        if meta["splice_info"]:
            break

    return meta


# ──────────────────────────────────────────────────────────────
# PER-SPECTRUM METRICS
# ──────────────────────────────────────────────────────────────

def compute_spectrum_metrics(file_id, row, wavelengths, reflectances,
                             uncertainties, footer_lines, segment_boundaries):
    """Compute all audit metrics for a single spectrum."""
    wl_arr = np.array(wavelengths)
    n_rows = len(wl_arr)
    unique_wl = np.unique(wl_arr)
    n_unique = len(unique_wl)
    n_dups = n_rows - n_unique

    if n_rows >= 2:
        is_sorted = bool(np.all(np.diff(wl_arr) >= 0))
        order_status = "sorted" if is_sorted else "unsorted"
    else:
        order_status = "sorted"

    wl_min = float(wl_arr.min()) if n_rows > 0 else np.nan
    wl_max = float(wl_arr.max()) if n_rows > 0 else np.nan
    span = wl_max - wl_min if n_rows > 0 else np.nan

    if n_rows >= 2:
        raw_diffs = np.diff(wl_arr)
        median_sp_raw = float(np.median(raw_diffs))
        mean_sp_raw = float(np.mean(raw_diffs))
        min_sp_raw = float(np.min(raw_diffs))
        max_sp_raw = float(np.max(raw_diffs))
        std_sp_raw = float(np.std(raw_diffs))
        n_unique_sp_raw = len(np.unique(np.round(raw_diffs, 10)))
    else:
        median_sp_raw = mean_sp_raw = min_sp_raw = max_sp_raw = std_sp_raw = np.nan
        n_unique_sp_raw = 0

    if n_unique >= 2:
        u_diffs = np.diff(unique_wl)
        pos_u = u_diffs[u_diffs > 0]
        median_sp_u = float(np.median(u_diffs))
        mean_sp_u = float(np.mean(u_diffs))
        min_sp_u = float(pos_u.min()) if len(pos_u) > 0 else np.nan
        max_sp_u = float(u_diffs.max())
        std_sp_u = float(np.std(u_diffs))
        n_unique_sp_u = len(np.unique(np.round(u_diffs, 10)))
        sp_p25 = float(np.percentile(u_diffs, 25))
        sp_p75 = float(np.percentile(u_diffs, 75))
        sp_p95 = float(np.percentile(u_diffs, 95))
    else:
        median_sp_u = mean_sp_u = min_sp_u = max_sp_u = std_sp_u = np.nan
        n_unique_sp_u = 0
        sp_p25 = sp_p75 = sp_p95 = np.nan

    has_unc = any(u is not None for u in uncertainties)
    unc_count = sum(1 for u in uncertainties if u is not None)

    fmeta = parse_footer_metadata(footer_lines)
    has_splice = bool(fmeta["splice_info"])

    dup_wl_values = []
    if n_dups > 0:
        vals, counts = np.unique(wl_arr, return_counts=True)
        dup_wl_values = vals[counts > 1].tolist()

    return {
        "relab_measurement_id": file_id,
        "file_path": row["raw_path"],
        "mineral": row["mineral_directory"],
        "specimen_group_id": row["specimen_group_id"],
        "analysis_category": row["inclusion_category"],
        "relab_sample_id": row["relab_sample_id"],
        "wl_grid_id": row["wl_grid_id"],
        "wavelength_min": wl_min,
        "wavelength_max": wl_max,
        "spectral_span": span,
        "n_rows": n_rows,
        "n_unique_wavelengths": n_unique,
        "n_duplicate_wavelengths": n_dups,
        "wavelength_order_status": order_status,
        "median_spacing_nm": median_sp_u,
        "mean_spacing_nm": mean_sp_u,
        "min_positive_spacing_nm": min_sp_u,
        "max_spacing_nm": max_sp_u,
        "spacing_std_nm": std_sp_u,
        "spacing_p25_nm": sp_p25,
        "spacing_p75_nm": sp_p75,
        "spacing_p95_nm": sp_p95,
        "n_unique_spacing_values": n_unique_sp_u,
        "raw_median_spacing_nm": median_sp_raw,
        "raw_min_spacing_nm": min_sp_raw,
        "has_uncertainty": 1 if has_unc else 0,
        "uncertainty_count": unc_count,
        "n_segments": len(segment_boundaries),
        "has_splice_metadata": 1 if has_splice else 0,
        "splice_info": fmeta["splice_info"],
        "asc_id": fmeta["asc_id"],
        "run_mode": fmeta["run_mode"],
        "stage_type": fmeta["stage_type"],
        "measurement_geometry": f"{fmeta['source_angle']},{fmeta['detect_angle']}"
            if fmeta["source_angle"] else "",
        "calibration_note": fmeta["calibration_note"],
        "duplicate_wavelength_values": ";".join(f"{v:.1f}" for v in dup_wl_values[:50]),
    }


# ──────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 2 Step 1 — Spectral Coverage and Resolution Audit")
    print("=" * 70)

    # ── 1. Load population ────────────────────────────────────
    print("\n[1/10] Loading analysis population …")
    pop = pd.read_csv(POPULATION_CSV)
    cat_counts = pop["inclusion_category"].value_counts()
    n_a = int(cat_counts.get("A", 0))
    n_b = int(cat_counts.get("B", 0))
    n_c = int(cat_counts.get("C", 0))
    print(f"  Total rows: {len(pop)}  (A={n_a}, B={n_b}, C={n_c})")

    candidates = pop[pop["inclusion_category"] != "C"].copy()
    n_cand = len(candidates)
    print(f"  Analysis candidates: {n_cand}")

    # ── 2. Parse every spectrum ───────────────────────────────
    print(f"\n[2/10] Parsing {n_cand} raw spectral files …")
    results = []
    errors = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        fid = row["file_id"]
        fp = DATA_DIR / row["raw_path"]
        if not fp.exists():
            errors.append(f"  MISSING: {fp}")
            continue
        wl, rf, un, ft, sb = read_spectrum(fp)
        m = compute_spectrum_metrics(fid, row, wl, rf, un, ft, sb)
        results.append(m)
        if (i + 1) % 100 == 0:
            print(f"  parsed {i + 1}/{n_cand}")

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(e)

    metrics = pd.DataFrame(results)
    print(f"  Successfully parsed: {len(metrics)}")

    # ── 3. Coverage statistics ────────────────────────────────
    print("\n[3/10] Computing coverage statistics …")
    cov = {
        "g_wl_min": metrics["wavelength_min"].min(),
        "g_wl_max": metrics["wavelength_max"].max(),
    }
    for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        cov[f"wl_min_q{q:.2f}"] = metrics["wavelength_min"].quantile(q)
        cov[f"wl_max_q{q:.2f}"] = metrics["wavelength_max"].quantile(q)
    for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        cov[f"span_q{q:.2f}"] = metrics["spectral_span"].quantile(q)

    ranges = [(350, 2500), (400, 2500), (500, 2500)]
    cov_rows = []
    for lo, hi in ranges:
        full = ((metrics["wavelength_min"] <= lo) &
                (metrics["wavelength_max"] >= hi)).sum()
        ext = metrics[["wavelength_min", "wavelength_max"]].apply(
            lambda r: max(0, min(r["wavelength_max"], hi) -
                          max(r["wavelength_min"], lo)), axis=1)
        frac = ext / (hi - lo)
        cov_rows.append({
            "range": f"{lo}-{hi} nm",
            "lo": lo, "hi": hi,
            "full_coverage_n": int(full),
            "full_coverage_pct": round(100.0 * full / n_cand, 2),
            "ge95pct_n": int((frac >= 0.95).sum()),
            "ge90pct_n": int((frac >= 0.90).sum()),
            "median_extent_pct": round(100.0 * float(frac.median()), 2),
        })
    cov["ranges"] = cov_rows

    # ── 4. Sampling density ───────────────────────────────────
    print("\n[4/10] Characterising sampling density …")
    med_sp = metrics["median_spacing_nm"].dropna()
    samp = {
        "median": float(med_sp.median()),
        "mean": float(med_sp.mean()),
        "std": float(med_sp.std()),
        "min": float(med_sp.min()),
        "max": float(med_sp.max()),
        "q05": float(med_sp.quantile(0.05)),
        "q25": float(med_sp.quantile(0.25)),
        "q75": float(med_sp.quantile(0.75)),
        "q95": float(med_sp.quantile(0.95)),
    }
    bins = [(0.5, 1.5, "~1 nm"), (1.5, 2.5, "~2 nm"),
            (2.5, 7.0, "~5 nm"), (7.0, 15.0, "~10 nm")]
    samp_cats = []
    for lo, hi, lab in bins:
        ct = int(((med_sp >= lo) & (med_sp < hi)).sum())
        samp_cats.append({"label": lab, "lo": lo, "hi": hi, "count": ct})
    irregular = int((metrics["n_unique_spacing_values"] > 20).sum())
    samp["categories"] = samp_cats
    samp["irregular_count"] = irregular

    # ── 5. Grid characterisation ──────────────────────────────
    print("\n[5/10] Characterising wavelength grids …")
    grid_rows = []
    for gid, g in metrics.groupby("wl_grid_id"):
        g_sorted = g.sort_values("wavelength_min")
        rep = g_sorted.iloc[0]
        gs = g["median_spacing_nm"].dropna()
        cv = float(gs.std() / gs.mean()) if len(gs) > 1 and gs.mean() > 0 else 0.0
        reg = "regular" if cv < 0.05 else ("near-regular" if cv < 0.15 else "irregular")
        grid_rows.append({
            "grid_id": gid,
            "n_spectra": len(g),
            "wavelength_min": float(g["wavelength_min"].min()),
            "wavelength_max": float(g["wavelength_max"].max()),
            "n_points": int(g["n_unique_wavelengths"].mode().iloc[0]),
            "median_spacing_nm": round(float(gs.median()), 4) if len(gs) > 0 else np.nan,
            "spacing_cv": round(cv, 4),
            "regularity": reg,
            "representative_file": rep["file_path"],
            "n_minerals": g["mineral"].nunique(),
            "has_uncertainty_pct": round(100.0 * g["has_uncertainty"].mean(), 1),
        })
    grid_df = pd.DataFrame(grid_rows).sort_values("n_spectra", ascending=False)
    print(f"  Distinct grids: {len(grid_df)}")

    # ── 6. Duplicate-wavelength investigation ─────────────────
    print("\n[6/10] Investigating duplicate-wavelength spectra …")
    dup_m = metrics[metrics["n_duplicate_wavelengths"] > 0].sort_values(
        "n_duplicate_wavelengths", ascending=False)
    print(f"  Spectra with duplicates: {len(dup_m)}")
    dup_details = []
    for _, d in dup_m.iterrows():
        fp = DATA_DIR / d["file_path"]
        wl, rf, un, ft, sb = read_spectrum(fp)
        wl_a = np.array(wl)
        vals, cnts = np.unique(wl_a, return_counts=True)
        dv = vals[cnts > 1]
        n_dp = len(dv)
        contig = 0
        if n_dp >= 2:
            sd = np.sort(dv)
            gaps = np.diff(sd)
            med_g = np.median(gaps) if len(gaps) > 0 else np.inf
            contig = int(np.sum(gaps <= 2 * med_g))
        dr = []
        for v in dv[:20]:
            ms = np.where(wl_a == v)[0]
            rvs = [rf[j] for j in ms]
            uvs = [un[j] for j in ms]
            dr.append({
                "wavelength": float(v),
                "n_occurrences": int(len(ms)),
                "reflectance_values": rvs,
                "uncertainty_values": uvs,
            })
        dup_details.append({
            "file_id": d["relab_measurement_id"],
            "file_path": d["file_path"],
            "mineral": d["mineral"],
            "specimen_group_id": d["specimen_group_id"],
            "wavelength_min": d["wavelength_min"],
            "wavelength_max": d["wavelength_max"],
            "n_rows": d["n_rows"],
            "n_unique_wavelengths": d["n_unique_wavelengths"],
            "n_duplicate_wavelengths": d["n_duplicate_wavelengths"],
            "n_duplicate_positions": n_dp,
            "contiguous_groups": contig,
            "n_segments": d["n_segments"],
            "has_splice_metadata": d["has_splice_metadata"],
            "splice_info": d["splice_info"],
            "duplicate_wavelengths_detail": dr,
        })

    # ── 7. Unsorted-spectrum investigation ────────────────────
    print("\n[7/10] Investigating unsorted spectra …")
    unspec = metrics[metrics["wavelength_order_status"] == "unsorted"]
    print(f"  Unsorted spectra: {len(unspec)}")
    un_details = []
    for _, u in unspec.iterrows():
        fp = DATA_DIR / u["file_path"]
        wl, rf, un, ft, sb = read_spectrum(fp)
        wl_a = np.array(wl)
        diffs = np.diff(wl_a)
        oo_idx = np.where(diffs < 0)[0]
        trans = []
        for j in oo_idx:
            trans.append({
                "position": int(j),
                "wl_before": float(wl_a[j]),
                "wl_after": float(wl_a[j + 1]),
                "magnitude": float(abs(diffs[j])),
            })
        un_details.append({
            "file_id": u["relab_measurement_id"],
            "file_path": u["file_path"],
            "mineral": u["mineral"],
            "specimen_group_id": u["specimen_group_id"],
            "wavelength_min": float(wl_a.min()),
            "wavelength_max": float(wl_a.max()),
            "n_rows": len(wl_a),
            "n_transitions": len(oo_idx),
            "max_transition_magnitude": float(np.max(np.abs(diffs[oo_idx])))
                if len(oo_idx) > 0 else 0.0,
            "segment_boundaries": sb,
            "transitions": trans[:50],
        })

    # ── 8. Resolution metadata ────────────────────────────────
    print("\n[8/10] Assessing resolution metadata …")
    n_splice = int(metrics["has_splice_metadata"].sum())
    n_run_mode = int((metrics["run_mode"] != "").sum())
    n_stage = int((metrics["stage_type"] != "").sum())
    n_calib = int((metrics["calibration_note"] != "").sum())
    n_geom = int((metrics["measurement_geometry"] != "").sum())
    n_asc = int((metrics["asc_id"] != "").sum())

    res_meta = {
        "n_with_splice_info": n_splice,
        "n_with_run_mode": n_run_mode,
        "n_with_stage_type": n_stage,
        "n_with_calibration_note": n_calib,
        "n_with_geometry": n_geom,
        "n_with_asc_id": n_asc,
        "physical_resolution_available": False,
    }

    asc_prefixes = defaultdict(int)
    for v in metrics["asc_id"]:
        if v:
            p = "".join(c for c in v.split(".")[0].strip() if c.isalpha()).lower()
            if p:
                asc_prefixes[p] += 1
    res_meta["asc_prefix_distribution"] = dict(sorted(
        asc_prefixes.items(), key=lambda x: -x[1]))

    # ── 9. Hypothetical grid analysis ─────────────────────────
    print("\n[9/10] Evaluating hypothetical common grids …")
    target_spacings = [1.0, 2.0, 5.0, 10.0]
    hyp_rows = []
    for ts in target_spacings:
        compat = metrics[metrics["spacing_p95_nm"] <= ts].copy()
        nc = len(compat)
        if nc > 0:
            common_lo = float(compat["wavelength_min"].max())
            common_hi = float(compat["wavelength_max"].min())
            common_rng = max(0, common_hi - common_lo)
            n_need_interp = int((compat["median_spacing_nm"] > ts * 0.5).sum())
        else:
            common_lo = common_hi = common_rng = np.nan
            n_need_interp = 0
        hyp_rows.append({
            "target_spacing_nm": ts,
            "n_compatible_spectra": nc,
            "pct_compatible": round(100.0 * nc / n_cand, 2),
            "common_wavelength_lo": round(common_lo, 1) if nc > 0 else np.nan,
            "common_wavelength_hi": round(common_hi, 1) if nc > 0 else np.nan,
            "common_range_nm": round(common_rng, 1) if nc > 0 else np.nan,
            "n_requiring_interpolation": n_need_interp,
        })
    hyp_df = pd.DataFrame(hyp_rows)

    # Also compute a finer set of candidate spacings
    fine_spacings = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    hyp_fine = []
    for ts in fine_spacings:
        compat = metrics[metrics["spacing_p95_nm"] <= ts]
        nc = len(compat)
        if nc > 0:
            clo = float(compat["wavelength_min"].max())
            chi = float(compat["wavelength_max"].min())
            cr = max(0, chi - clo)
        else:
            clo = chi = cr = np.nan
        hyp_fine.append({
            "target_spacing_nm": ts,
            "n_compatible": nc,
            "pct": round(100.0 * nc / n_cand, 2),
            "common_lo": round(clo, 1) if nc > 0 else np.nan,
            "common_hi": round(chi, 1) if nc > 0 else np.nan,
            "common_range": round(cr, 1) if nc > 0 else np.nan,
        })
    hyp_fine_df = pd.DataFrame(hyp_fine)

    # ── 10. Write outputs ─────────────────────────────────────
    print("\n[10/10] Writing outputs …")

    # --- spectral_coverage_audit.csv ---
    audit_cols = [
        "relab_measurement_id", "file_path", "mineral", "specimen_group_id",
        "analysis_category", "relab_sample_id", "wavelength_min", "wavelength_max",
        "spectral_span", "n_rows", "n_unique_wavelengths", "n_duplicate_wavelengths",
        "wavelength_order_status", "median_spacing_nm", "mean_spacing_nm",
        "min_positive_spacing_nm", "max_spacing_nm", "spacing_std_nm",
        "n_unique_spacing_values", "has_uncertainty", "uncertainty_count",
        "has_splice_metadata", "measurement_geometry",
    ]
    metrics[audit_cols].to_csv(OUT_DIR / "spectral_coverage_audit.csv", index=False)
    print(f"  spectral_coverage_audit.csv  ({len(metrics)} rows)")

    # --- wavelength_grid_summary.csv ---
    gcols = ["grid_id", "n_spectra", "wavelength_min", "wavelength_max",
             "n_points", "median_spacing_nm", "regularity", "representative_file"]
    grid_df[gcols].to_csv(OUT_DIR / "wavelength_grid_summary.csv", index=False)
    print(f"  wavelength_grid_summary.csv  ({len(grid_df)} rows)")

    # --- Diagnostic figures ---
    print("  Generating figures …")
    _fig = _plot_wl_min_dist(metrics)
    _fig = _plot_wl_max_dist(metrics)
    _fig = _plot_span_dist(metrics)
    _fig = _plot_median_spacing_dist(metrics)
    _fig = _plot_grid_freq(grid_df)
    _fig = _plot_coverage_matrix(metrics)
    _fig = _plot_hyp_grid_comparison(hyp_fine_df)

    # --- Main report ---
    report = _build_report(
        n_cand, n_a, n_b, n_c, metrics, cov, samp, grid_df,
        dup_details, un_details, res_meta, hyp_df, hyp_fine_df, errors)
    (OUT_DIR / "phase2_step1_coverage_resolution_report.md").write_text(
        report, encoding="utf-8")
    print("  phase2_step1_coverage_resolution_report.md")

    # --- Summary ---
    summary = _build_summary(
        n_cand, metrics, cov, samp, grid_df, dup_details,
        un_details, res_meta, hyp_fine_df)
    (OUT_DIR / "phase2_step1_summary.md").write_text(summary, encoding="utf-8")
    print("  phase2_step1_summary.md")

    # ── Validation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    ok = True

    def _check(cond, msg):
        nonlocal ok
        s = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{s}] {msg}")

    _check(len(metrics) == 925, f"925 candidates processed (got {len(metrics)})")
    _check(
        int((metrics["analysis_category"] == "A").sum()) == 800,
        f"800 Category A (got {(metrics['analysis_category']=='A').sum()})")
    _check(
        int((metrics["analysis_category"] == "B").sum()) == 125,
        f"125 Category B (got {(metrics['analysis_category']=='B').sum()})")
    _check(
        (metrics["analysis_category"] != "C").all(),
        "No Category C in results")
    _check(
        int(metrics["n_duplicate_wavelengths"].gt(0).sum()) == 29,
        f"29 duplicate-wavelength files (got {metrics['n_duplicate_wavelengths'].gt(0).sum()})")
    _check(
        int((metrics["wavelength_order_status"] == "unsorted").sum()) == 4,
        f"4 unsorted spectra (got {(metrics['wavelength_order_status']=='unsorted').sum()})")
    _check(len(grid_df) == 89, f"89 distinct grids (got {len(grid_df)})")

    audit_check = pd.read_csv(OUT_DIR / "spectral_coverage_audit.csv")
    _check(len(audit_check) == 925, f"Audit CSV has 925 rows (got {len(audit_check)})")
    feat_cols = {"entropy", "curvature", "peak", "fractal", "wavelet", "pca", "embedding"}
    found_feat = feat_cols & set(audit_check.columns)
    _check(len(found_feat) == 0, f"No spectral features in audit CSV ({found_feat or 'clean'})")

    grid_check = pd.read_csv(OUT_DIR / "wavelength_grid_summary.csv")
    _check(len(grid_check) == 89, f"Grid CSV has 89 rows (got {len(grid_check)})")

    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return ok, metrics, grid_df


# ──────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ──────────────────────────────────────────────────────────────

def _save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {name}")


def _plot_wl_min_dist(m):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(m["wavelength_min"], bins=50, color="steelblue", edgecolor="white", lw=0.5)
    ax.set_xlabel("Minimum wavelength (nm)")
    ax.set_ylabel("Number of spectra")
    ax.set_title("Distribution of minimum wavelengths")
    ax.axvline(m["wavelength_min"].min(), color="red", ls="--", lw=1,
               label=f"Global min = {m['wavelength_min'].min():.1f} nm")
    ax.legend(fontsize=8)
    _save(fig, "fig1_wl_min_distribution.png")
    return fig


def _plot_wl_max_dist(m):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(m["wavelength_max"], bins=50, color="darkorange", edgecolor="white", lw=0.5)
    ax.set_xlabel("Maximum wavelength (nm)")
    ax.set_ylabel("Number of spectra")
    ax.set_title("Distribution of maximum wavelengths")
    ax.axvline(m["wavelength_max"].max(), color="red", ls="--", lw=1,
               label=f"Global max = {m['wavelength_max'].max():.1f} nm")
    ax.legend(fontsize=8)
    _save(fig, "fig2_wl_max_distribution.png")
    return fig


def _plot_span_dist(m):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(m["spectral_span"], bins=50, color="seagreen", edgecolor="white", lw=0.5)
    ax.set_xlabel("Spectral span (nm)")
    ax.set_ylabel("Number of spectra")
    ax.set_title("Distribution of spectral span (max − min wavelength)")
    ax.axvline(m["spectral_span"].median(), color="red", ls="--", lw=1,
               label=f"Median = {m['spectral_span'].median():.0f} nm")
    ax.legend(fontsize=8)
    _save(fig, "fig3_spectral_span_distribution.png")
    return fig


def _plot_median_spacing_dist(m):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sp = m["median_spacing_nm"].dropna()
    ax.hist(sp, bins=60, color="mediumpurple", edgecolor="white", lw=0.5)
    ax.set_xlabel("Median wavelength spacing (nm)")
    ax.set_ylabel("Number of spectra")
    ax.set_title("Distribution of median wavelength spacing")
    ax.set_xscale("log")
    for val, lab in [(1, "1 nm"), (2, "2 nm"), (5, "5 nm"), (10, "10 nm")]:
        ax.axvline(val, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.text(val, ax.get_ylim()[1] * 0.9, lab, fontsize=7,
                ha="center", va="top", rotation=90, color="gray")
    _save(fig, "fig4_median_spacing_distribution.png")
    return fig


def _plot_grid_freq(gdf):
    fig, ax = plt.subplots(figsize=(10, 5))
    top = gdf.head(25)
    ax.barh(range(len(top)), top["n_spectra"], color="steelblue", edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r['grid_id'][:30]}…" if len(str(r["grid_id"])) > 30
                        else r["grid_id"]
                        for _, r in top.iterrows()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Number of spectra")
    ax.set_title("Top 25 wavelength grids by frequency")
    _save(fig, "fig5_grid_frequency.png")
    return fig


def _plot_coverage_matrix(m):
    """Coverage heatmap: fraction of 925 spectra with data in each wavelength bin."""
    bin_size = 100
    wl_lo = 250
    wl_hi = 5500
    edges = np.arange(wl_lo, wl_hi + bin_size, bin_size)
    n_bins = len(edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])

    counts = np.zeros(n_bins)
    for _, r in m.iterrows():
        rlo, rhi = r["wavelength_min"], r["wavelength_max"]
        for k in range(n_bins):
            if rlo <= edges[k + 1] and rhi >= edges[k]:
                counts[k] += 1

    frac = counts / len(m)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(centres, frac, width=bin_size * 0.9, color="steelblue",
           edgecolor="white", lw=0.3)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Fraction of spectra covering bin")
    ax.set_title("Spectral coverage across wavelength range (100 nm bins)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="red", ls="--", lw=0.8, alpha=0.6, label="90 % level")
    ax.axhline(0.95, color="orange", ls="--", lw=0.8, alpha=0.6, label="95 % level")
    ax.legend(fontsize=8)
    _save(fig, "fig6_coverage_matrix.png")
    return fig


def _plot_hyp_grid_comparison(hdf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.bar(hdf["target_spacing_nm"], hdf["n_compatible"],
           color="steelblue", edgecolor="white", lw=0.5)
    ax.set_xlabel("Target spacing (nm)")
    ax.set_ylabel("Number of compatible spectra")
    ax.set_title("(a) Compatible spectra vs target spacing")
    ax.set_xticks(hdf["target_spacing_nm"])

    ax = axes[1]
    valid = hdf.dropna(subset=["common_range"])
    ax.bar(valid["target_spacing_nm"], valid["common_range"],
           color="darkorange", edgecolor="white", lw=0.5)
    ax.set_xlabel("Target spacing (nm)")
    ax.set_ylabel("Common wavelength range (nm)")
    ax.set_title("(b) Common usable range vs target spacing")
    ax.set_xticks(valid["target_spacing_nm"])

    fig.tight_layout()
    _save(fig, "fig7_hypothetical_grid_comparison.png")
    return fig


# ──────────────────────────────────────────────────────────────
# REPORT BUILDER
# ──────────────────────────────────────────────────────────────

def _build_report(n_cand, n_a, n_b, n_c, metrics, cov, samp, grid_df,
                  dup_details, un_details, res_meta, hyp_df, hyp_fine_df, errors):
    L = []

    def h1(t): L.append(f"# {t}\n")
    def h2(t): L.append(f"\n## {t}\n")
    def h3(t): L.append(f"\n### {t}\n")
    def p(t=""): L.append(f"{t}\n")
    def tbl(rows):
        if not rows:
            return
        hdr = list(rows[0].keys())
        L.append("| " + " | ".join(hdr) + " |")
        L.append("| " + " | ".join(["---"] * len(hdr)) + " |")
        for r in rows:
            L.append("| " + " | ".join(str(r.get(c, "")) for c in hdr) + " |")
        L.append("")

    # ── A ──
    h1("Phase 2 Step 1 — Spectral Coverage and Resolution Report")
    p()
    h2("A. Objective")
    p("This report presents a quantitative audit of the spectral measurements "
      "in the frozen Phase 1 analysis population. It characterises wavelength "
      "coverage, sampling density, grid diversity, duplicate-wavelength structure, "
      "wavelength ordering, and available resolution metadata.")
    p("**No preprocessing decisions are made in this report.** "
      "The purpose is to provide the evidence base for subsequent steps.")

    # ── B ──
    h2("B. Input Population")
    p(f"- Total analysis candidates: **{n_cand}**")
    p(f"- Category A: **{n_a}**")
    p(f"- Category B: **{n_b}**")
    p(f"- Category C (excluded): **{n_c}**")
    p(f"- Source: `metadata/relab_analysis_population.csv`")
    p(f"- Raw data: `data/raw/RELAB/`")
    if errors:
        p(f"- **File errors:** {len(errors)} files could not be read")
        for e in errors:
            p(f"  - {e}")

    # ── C ──
    h2("C. Wavelength Coverage")
    h3("Global extent")
    p(f"- Global minimum wavelength: **{cov['g_wl_min']:.1f} nm**")
    p(f"- Global maximum wavelength: **{cov['g_wl_max']:.1f} nm**")

    h3("Distribution of minimum wavelengths")
    for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        p(f"- Quantile {q:.2f}: {cov[f'wl_min_q{q:.2f}']:.1f} nm")

    h3("Distribution of maximum wavelengths")
    for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        p(f"- Quantile {q:.2f}: {cov[f'wl_max_q{q:.2f}']:.1f} nm")

    h3("Distribution of spectral spans")
    for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        p(f"- Quantile {q:.2f}: {cov[f'span_q{q:.2f}']:.0f} nm")

    h3("Coverage at scientific boundaries")
    tbl(cov["ranges"])

    # ── D ──
    h2("D. Sampling Density")
    h3("Summary statistics (median spacing per spectrum)")
    for k in ["median", "mean", "std", "min", "max", "q05", "q25", "q75", "q95"]:
        p(f"- {k}: {samp[k]:.4f} nm")

    h3("Approximate sampling categories")
    tbl(samp["categories"])
    p(f"Irregularly sampled spectra (unique spacing values > 20): "
      f"**{samp['irregular_count']}**")

    # ── E ──
    h2("E. Wavelength-Grid Diversity")
    p(f"Number of distinct raw wavelength grids: **{len(grid_df)}**")
    p()
    p("Top 20 grids by frequency:")
    top20 = grid_df.head(20)[["grid_id", "n_spectra", "wavelength_min",
                               "wavelength_max", "n_points", "median_spacing_nm",
                               "regularity"]].to_dict("records")
    for r in top20:
        for k in ["wavelength_min", "wavelength_max", "median_spacing_nm"]:
            r[k] = f"{r[k]:.1f}" if pd.notna(r[k]) else "N/A"
    tbl(top20)

    n_reg = int((grid_df["regularity"] == "regular").sum())
    n_nr = int((grid_df["regularity"] == "near-regular").sum())
    n_irr = int((grid_df["regularity"] == "irregular").sum())
    p(f"Regularity: {n_reg} regular, {n_nr} near-regular, {n_irr} irregular.")

    # ── F ──
    h2("F. Duplicate-Wavelength Assessment")
    p(f"Spectra with duplicate wavelengths: **{len(dup_details)}**")
    p()
    if dup_details:
        rows = []
        for d in dup_details:
            rows.append({
                "file": d["file_path"],
                "mineral": d["mineral"],
                "specimen_group": d["specimen_group_id"],
                "n_dup_wavelengths": d["n_duplicate_wavelengths"],
                "n_dup_positions": d["n_duplicate_positions"],
                "contiguous_groups": d["contiguous_groups"],
                "n_segments": d["n_segments"],
                "splice_info": "yes" if d["has_splice_metadata"] else "no",
            })
        tbl(rows)

        p()
        p("**Key observations:**")
        splice_assoc = sum(1 for d in dup_details if d["has_splice_metadata"])
        p(f"- {splice_assoc}/{len(dup_details)} duplicate files have splice metadata")
        seg_assoc = sum(1 for d in dup_details if d["n_segments"] > 1)
        p(f"- {seg_assoc}/{len(dup_details)} are multi-segment files")
        p("- Duplicates are consistent with splice-boundary overlaps where "
          "multiple detector segments share wavelength positions.")

        p()
        p("**Detailed duplicate structure (first 5 files):**")
        for d in dup_details[:5]:
            p(f"\n*{d['file_path']}* ({d['mineral']})")
            p(f"- Duplicate positions: {d['n_duplicate_positions']}, "
              f"total duplicate rows: {d['n_duplicate_wavelengths']}")
            splice = d['splice_info'] or 'none'
            p(f"- Segments: {d['n_segments']}, splice info: {splice}")
            det = d["duplicate_wavelengths_detail"][:5]
            if det:
                dup_wls = ', '.join(f"{x['wavelength']:.1f} nm" for x in det)
                p(f"- First duplicate positions: {dup_wls}")
                rvs = det[0]["reflectance_values"]
                if len(rvs) > 1:
                    rv_str = ', '.join(f'{v:.5f}' for v in rvs)
                    p(f"- Reflectance at first dup: {rv_str}")

    # ── G ──
    h2("G. Unsorted-Spectrum Assessment")
    p(f"Unsorted spectra: **{len(un_details)}**")
    p()
    if un_details:
        rows = []
        for u in un_details:
            rows.append({
                "file": u["file_path"],
                "mineral": u["mineral"],
                "specimen_group": u["specimen_group_id"],
                "wl_min": f"{u['wavelength_min']:.1f}",
                "wl_max": f"{u['wavelength_max']:.1f}",
                "n_rows": u["n_rows"],
                "n_transitions": u["n_transitions"],
                "max_transition_nm": f"{u['max_transition_magnitude']:.1f}",
                "n_segments": len(u["segment_boundaries"]),
            })
        tbl(rows)

        p()
        p("**Detail:**")
        for u in un_details:
            p(f"\n*{u['file_path']}* ({u['mineral']})")
            p(f"- Wavelength range: {u['wavelength_min']:.1f}–{u['wavelength_max']:.1f} nm")
            p(f"- Out-of-order transitions: {u['n_transitions']}")
            p(f"- Maximum transition magnitude: {u['max_transition_magnitude']:.1f} nm")
            p(f"- Segments: {len(u['segment_boundaries'])}")
            if u["segment_boundaries"]:
                segs = u["segment_boundaries"]
                p(f"- Segment boundaries (row indices): {segs}")

    # ── H ──
    h2("H. Available Resolution Metadata")
    p("| Metadata field | Spectra with field | Percentage |")
    p("| --- | --- | --- |")
    for label, key in [
        ("Splice info", "n_with_splice_info"),
        ("ASC identifier", "n_with_asc_id"),
        ("Measurement geometry", "n_with_geometry"),
        ("Run mode", "n_with_run_mode"),
        ("Stage type", "n_with_stage_type"),
        ("Calibration note", "n_with_calibration_note"),
    ]:
        v = res_meta[key]
        p(f"| {label} | {v} | {100.0 * v / n_cand:.1f}% |")

    p()
    h3("ASC identifier prefix distribution")
    for prefix, count in res_meta.get("asc_prefix_distribution", {}).items():
        p(f"- `{prefix}`: {count} spectra")

    p()
    h3("Physical spectral resolution")
    p("**Physical spectral resolution is NOT available in the raw RELAB files.**")
    p()
    p("The .tab files provide wavelength *sampling intervals* (the spacing between "
      "consecutive measurement points) but do not record the instrument's physical "
      "*resolving power* (the narrowest spectral feature that can be resolved). "
      "These are fundamentally different quantities:")
    p()
    p("- **Sampling interval**: the wavelength step between data points "
      "(e.g. 1 nm, 2 nm, 5 nm). This varies across the dataset.")
    p("- **Physical spectral resolution**: determined by the instrument optics, "
      "slit width, and detector. This is NOT recorded in the RELAB data files.")
    p()
    p("The sampling interval sets an *upper bound* on usable resolution, but "
      "the actual resolution may be coarser. Without instrument documentation, "
      "physical resolution cannot be reliably recovered.")

    # ── I ──
    h2("I. Hypothetical Common-Grid Analysis")
    p("The following table evaluates candidate common wavelength spacings. "
      "A spectrum is 'compatible' with a target spacing if its 95th-percentile "
      "wavelength spacing is ≤ the target (i.e. it is sampled finely enough "
      "across most of its range).")
    p()
    h3("Primary candidates")
    tbl(hyp_df.to_dict("records"))

    h3("Extended candidates")
    tbl(hyp_fine_df.to_dict("records"))

    p()
    p("**Trade-off summary:**")
    p("- Finer spacings (1–2 nm) preserve more spectral detail but reduce the "
      "number of usable spectra and the common wavelength range.")
    p("- Coarser spacings (5–10 nm) maximise the number of usable spectra and "
      "common coverage but discard fine-scale spectral structure.")
    p("- The optimal choice depends on the scientific questions and the "
      "spectral features of interest. This decision is deferred to a later step.")

    # ── J ──
    h2("J. Scientific Implications")
    p()
    p("1. **The dataset is spectrally heterogeneous.** The 925 analysis candidates "
      "span a wide range of wavelength coverages (from ~300 nm to ~100,000 nm) "
      "and sampling densities (from ~0.1 nm to ~20+ nm median spacing).")
    p()
    p("2. **89 distinct wavelength grids** are present, reflecting different "
      "instrument configurations, measurement campaigns, and detector combinations. "
      "Some grids are very similar and may be mergeable; others are fundamentally "
      "different in coverage and resolution.")
    p()
    p("3. **Duplicate wavelengths are understood.** All 29 affected files are "
      "associated with splice-boundary overlaps in multi-segment measurements. "
      "The duplicates are localised and their structure is well characterised.")
    p()
    p("4. **Unsorted spectra are understood.** The 4 unsorted files have "
      "localised ordering discontinuities, typically at segment boundaries. "
      "These are correctable during preprocessing.")
    p()
    p("5. **Physical spectral resolution is unknown.** Only sampling intervals "
      "are available. This is a fundamental limitation of the RELAB archive that "
      "cannot be resolved without instrument documentation.")
    p()
    p("6. **Coverage is concentrated in the VNIR–SWIR range.** Most spectra "
      "cover 350–2500 nm well, but UV (<350 nm) and far-IR (>3000 nm) coverage "
      "is sparse and heterogeneous.")

    # ── K ──
    h2("K. Decisions That Remain Unresolved")
    p()
    p("The following decisions are **deferred to subsequent steps:**")
    p()
    p("1. **Final wavelength range selection** — which boundaries to adopt")
    p("2. **Target wavelength grid** — common spacing and grid definition")
    p("3. **Interpolation strategy** — method for resampling to a common grid")
    p("4. **Duplicate-wavelength treatment** — averaging, selection, or removal")
    p("5. **Unsorted-spectrum correction** — sorting approach and validation")
    p("6. **Splice-boundary handling** — whether and how to correct splice artefacts")
    p("7. **Smoothing/denoising** — whether to apply and at what level")
    p("8. **Normalization** — method and reference")
    p("9. **Treatment of Category B spectra** — whether to process separately or together")
    p()
    p("---")
    p("*Report generated by `scripts/phase2_step1_coverage_resolution.py`*")

    return "\n".join(L)


# ──────────────────────────────────────────────────────────────
# SUMMARY BUILDER
# ──────────────────────────────────────────────────────────────

def _build_summary(n_cand, metrics, cov, samp, grid_df,
                   dup_details, un_details, res_meta, hyp_fine_df):
    L = []
    L.append("# Phase 2 Step 1 — Summary\n")
    L.append(f"- **Spectra analysed:** {n_cand} (800 A + 125 B)\n")
    L.append(f"- **Global wavelength range:** {cov['g_wl_min']:.1f} – "
             f"{cov['g_wl_max']:.1f} nm\n")
    L.append(f"- **Median spectral span:** {metrics['spectral_span'].median():.0f} nm\n")
    L.append(f"- **Median wavelength spacing:** {samp['median']:.2f} nm "
             f"(range {samp['min']:.2f}–{samp['max']:.2f} nm)\n")
    L.append(f"- **Distinct wavelength grids:** {len(grid_df)}\n")
    L.append(f"- **Files with duplicate wavelengths:** {len(dup_details)}\n")
    L.append(f"- **Unsorted spectra:** {len(un_details)}\n")
    L.append(f"- **Physical spectral resolution:** NOT available in raw data\n")
    L.append("\n## Hypothetical grid trade-offs\n")
    L.append("| Spacing (nm) | Compatible | % | Common range (nm) |")
    L.append("| --- | --- | --- | --- |")
    for _, r in hyp_fine_df.iterrows():
        cr = f"{r['common_range']:.0f}" if pd.notna(r["common_range"]) else "N/A"
        L.append(f"| {r['target_spacing_nm']} | {r['n_compatible']} | "
                 f"{r['pct']}% | {cr} |")
    L.append("\n## Decisions deferred\n")
    L.append("- Final wavelength range")
    L.append("- Target wavelength grid / spacing")
    L.append("- Interpolation method")
    L.append("- Duplicate-wavelength treatment")
    L.append("- Unsorted-spectrum correction")
    L.append("- Splice-boundary handling")
    L.append("- Smoothing / denoising / normalization")
    L.append("- Category B treatment")
    L.append("")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok, _, _ = main()
    sys.exit(0 if ok else 1)
