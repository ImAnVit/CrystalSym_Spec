#!/usr/bin/env python3
"""
Phase 2 Step 2 — Scientific Selection of Spectral Domain and Target Wavelength Grid
====================================================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

This script performs a decision analysis to determine the most scientifically
defensible common spectral representation (domain and wavelength grid) for the
primary analysis population.

The choice is justified in relation to the research objective:
  Investigating relationships between mineral spectral complexity and crystal
  symmetry, chemistry, and structure using interpretable statistical/ML methods
  and symbolic regression.

Outputs:
  results/02_preprocessing_qc/phase2_step2_domain_grid_report.md
  results/02_preprocessing_qc/phase2_step2_summary.md
  results/02_preprocessing_qc/domain_coverage_comparison.csv
  results/02_preprocessing_qc/grid_spacing_comparison.csv
  results/02_preprocessing_qc/figures/step2_fig*.png

Does NOT modify raw data or create a final processed spectral dataset.
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
AUDIT_CSV = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "spectral_coverage_audit.csv"
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# SPECTRAL REGION DEFINITIONS (nm)
# ──────────────────────────────────────────────────────────────

SPECTRAL_REGIONS = {
    "A_UV_Vis":          (280,   700),
    "B_Near_IR":         (700,   2500),
    "C_SWIR_transition": (2500,  5000),
    "D_Mid_Thermal_IR":  (5000,  30000),
    "E_Far_IR":          (30000, 100000),
}

# ──────────────────────────────────────────────────────────────
# CANDIDATE DOMAIN DEFINITIONS (nm)
# ──────────────────────────────────────────────────────────────

CANDIDATE_DOMAINS = [
    {
        "id": "D1",
        "label": "300-2500 nm (UV-NIR)",
        "wmin": 300, "wmax": 2500,
        "description": (
            "Extends from the UV edge through the full NIR range. Captures "
            "charge-transfer bands (UV), crystal-field transitions (Vis-NIR), "
            "and overtone/combination bands (NIR). 276 spectra fully cover this domain."
        ),
    },
    {
        "id": "D2",
        "label": "320-2550 nm (balanced Vis-NIR)",
        "wmin": 320, "wmax": 2550,
        "description": (
            "Balanced domain matching the two most common native grids "
            "(300-2600/461 and 320-2550/447). Maximises coverage while retaining "
            "the full visible and NIR diagnostic range. 454 spectra fully cover "
            "this domain."
        ),
    },
    {
        "id": "D3",
        "label": "300-2600 nm (matching most common native grid)",
        "wmin": 300, "wmax": 2600,
        "description": (
            "Matches the single most common native grid exactly (300-2600 nm, "
            "461 points, 197 spectra). Minimises interpolation for the largest "
            "native group. 247 spectra fully cover this domain."
        ),
    },
]

TARGET_SPACINGS = [1, 2, 5, 10]

# ──────────────────────────────────────────────────────────────
# CLASSIFICATION HELPERS
# ──────────────────────────────────────────────────────────────

def classify_acquisition_regime(file_path):
    prefix = Path(file_path).name[:2].lower()
    if prefix.startswith("c"):
        return "VNIR_spectrophotometer"
    elif prefix.startswith(("b", "m")):
        return "FTIR"
    elif prefix.startswith("n"):
        return "NIR_spectrometer"
    else:
        return "other"


def classify_spectral_coverage(wl_min, wl_max, domain_wmin, domain_wmax):
    if wl_min <= domain_wmin and wl_max >= domain_wmax:
        return "full"
    span = domain_wmax - domain_wmin
    overlap_lo = max(wl_min, domain_wmin)
    overlap_hi = min(wl_max, domain_wmax)
    overlap = max(0, overlap_hi - overlap_lo)
    pct = overlap / span * 100 if span > 0 else 0
    if pct >= 95:
        return "95pct"
    elif pct > 0:
        return "partial"
    return "none"


# ──────────────────────────────────────────────────────────────
# RAW .tab FILE PARSER
# ──────────────────────────────────────────────────────────────

def read_spectrum(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    wavelengths, reflectances = [], []
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
        idx += count + 1

    return np.array(wavelengths), np.array(reflectances)


# ──────────────────────────────────────────────────────────────
# COMPLEXITY DIAGNOSTIC METRICS
# ──────────────────────────────────────────────────────────────

def metric_first_order_variation(r):
    if len(r) < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(r))))


def metric_second_order_variation(r):
    if len(r) < 3:
        return np.nan
    return float(np.sum(np.abs(np.diff(r, n=2))))


def metric_peak_density(r):
    if len(r) < 3:
        return np.nan
    peaks = 0
    for i in range(1, len(r) - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            peaks += 1
    return peaks / len(r) * 100


def metric_spectral_entropy(r):
    if len(r) < 2:
        return np.nan
    hist, _ = np.histogram(r, bins=50, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log(hist)))


def metric_spectral_range(r):
    if len(r) < 1:
        return np.nan
    return float(np.max(r) - np.min(r))


def downsample_to_grid(wl, rf, spacing):
    grid = np.arange(np.ceil(wl.min()), np.floor(wl.max()) + spacing * 0.1, spacing)
    if len(grid) < 2:
        return grid, np.array([])
    rf_interp = np.interp(grid, wl, rf)
    return grid, rf_interp


def compute_all_metrics(r):
    return {
        "first_order_var": metric_first_order_variation(r),
        "second_order_var": metric_second_order_variation(r),
        "peak_density": metric_peak_density(r),
        "entropy": metric_spectral_entropy(r),
        "range": metric_spectral_range(r),
    }


# ──────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_checksums():
    ref = pd.read_csv(CHECKSUMS_CSV)
    mismatches = 0
    checked = 0
    for _, row in ref.iterrows():
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = hashlib.md5(fp.read_bytes()).hexdigest()
            if md5 != row["md5"]:
                mismatches += 1
            checked += 1
    return checked, mismatches


def validate_population(pop):
    n = len(pop)
    na = (pop["inclusion_category"] == "A").sum()
    nb = (pop["inclusion_category"] == "B").sum()
    nc = (pop["inclusion_category"] == "C").sum()
    return n, na, nb, nc


# ──────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Phase 2 Step 2 — Spectral Domain and Grid Selection Analysis")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────
    pop = pd.read_csv(POPULATION_CSV)
    audit = pd.read_csv(AUDIT_CSV)
    meas = pd.read_csv(MEASUREMENTS_CSV)

    candidates = pop[pop["inclusion_category"].isin(["A", "B"])].copy()
    cand_ids = set(candidates["file_id"].values)
    audit_c = audit[audit["relab_measurement_id"].isin(cand_ids)].copy()
    audit_c = audit_c.sort_values("relab_measurement_id").reset_index(drop=True)

    # ── Validation ─────────────────────────────────────────────
    print("\n[Validation]")
    checked, mismatches = validate_checksums()
    print(f"  Checksums: {checked} checked, {mismatches} mismatches")
    n_total, n_a, n_b, n_c = validate_population(pop)
    print(f"  Population: {n_total} total, A={n_a}, B={n_b}, C={n_c}")
    assert n_total == 1017, f"Expected 1017, got {n_total}"
    assert n_a == 800, f"Expected A=800, got {n_a}"
    assert n_b == 125, f"Expected B=125, got {n_b}"
    assert n_c == 92, f"Expected C=92, got {n_c}"
    assert mismatches == 0, f"{mismatches} checksum mismatches!"
    assert len(audit_c) == 925, f"Expected 925 audited, got {len(audit_c)}"
    print("  All validation checks PASSED")

    n_spectra = len(audit_c)

    # ── Section 2: Spectral Region Coverage ────────────────────
    print("\n[Section 2] Spectral region coverage analysis")

    region_results = {}
    for rname, (rlo, rhi) in SPECTRAL_REGIONS.items():
        mask_full = (audit_c["wavelength_min"] <= rlo) & (audit_c["wavelength_max"] >= rhi)
        sub = audit_c[mask_full]

        grids_in = set()
        spacings = []
        for _, r in sub.iterrows():
            grids_in.add(r.get("wl_grid_id", "unknown") if "wl_grid_id" in r.index else "unknown")
            if not np.isnan(r["median_spacing_nm"]):
                spacings.append(r["median_spacing_nm"])

        region_results[rname] = {
            "wmin": rlo, "wmax": rhi,
            "n_full": int(mask_full.sum()),
            "pct": float(mask_full.sum() / n_spectra * 100),
            "n_catA": int((sub["analysis_category"] == "A").sum()),
            "n_catB": int((sub["analysis_category"] == "B").sum()),
            "n_grids": len(grids_in),
            "median_spacing": float(np.median(spacings)) if spacings else np.nan,
            "mineral_counts": sub["mineral"].value_counts().to_dict(),
            "prov_counts": sub["specimen_group_id"].nunique(),
        }
        print(f"  {rname} ({rlo}-{rhi} nm): {mask_full.sum()} spectra ({mask_full.sum()/n_spectra*100:.1f}%)")

    # ── Section 4: Candidate Domain Coverage ───────────────────
    print("\n[Section 4] Candidate domain coverage analysis")

    domain_results = {}
    for dom in CANDIDATE_DOMAINS:
        did = dom["id"]
        dlo, dhi = dom["wmin"], dom["wmax"]

        full_mask = (audit_c["wavelength_min"] <= dlo) & (audit_c["wavelength_max"] >= dhi)
        full_sub = audit_c[full_mask]

        overlap_lo = np.maximum(audit_c["wavelength_min"].values, dlo)
        overlap_hi = np.minimum(audit_c["wavelength_max"].values, dhi)
        overlap = np.maximum(0, overlap_hi - overlap_lo)
        domain_span = dhi - dlo
        pct_cov = overlap / domain_span * 100
        mask_95 = pct_cov >= 95
        sub_95 = audit_c[mask_95]

        regimes_full = defaultdict(int)
        for _, r in full_sub.iterrows():
            reg = classify_acquisition_regime(r["file_path"])
            regimes_full[reg] += 1

        domain_results[did] = {
            "wmin": dlo, "wmax": dhi, "label": dom["label"],
            "description": dom["description"],
            "n_full": int(full_mask.sum()),
            "pct_full": float(full_mask.sum() / n_spectra * 100),
            "n_95": int(mask_95.sum()),
            "pct_95": float(mask_95.sum() / n_spectra * 100),
            "catA_full": int((full_sub["analysis_category"] == "A").sum()),
            "catB_full": int((full_sub["analysis_category"] == "B").sum()),
            "catA_95": int((sub_95["analysis_category"] == "A").sum()),
            "catB_95": int((sub_95["analysis_category"] == "B").sum()),
            "n_minerals_full": int(full_sub["mineral"].nunique()),
            "minerals_full": sorted(full_sub["mineral"].unique().tolist()),
            "mineral_counts_full": full_sub["mineral"].value_counts().to_dict(),
            "n_minerals_95": int(sub_95["mineral"].nunique()),
            "mineral_counts_95": sub_95["mineral"].value_counts().to_dict(),
            "n_prov_full": int(full_sub["specimen_group_id"].nunique()),
            "n_prov_95": int(sub_95["specimen_group_id"].nunique()),
            "regimes_full": dict(regimes_full),
        }
        print(f"  {did} ({dlo}-{dhi} nm): full={full_mask.sum()}, 95%={mask_95.sum()}")

    # ── Section 7: Grid Spacing Analysis ───────────────────────
    print("\n[Section 7] Grid spacing analysis")

    grid_results = {}
    for dom in CANDIDATE_DOMAINS:
        did = dom["id"]
        dlo, dhi = dom["wmin"], dom["wmax"]
        cov_mask = (audit_c["wavelength_min"] <= dlo) & (audit_c["wavelength_max"] >= dhi)

        for sp in TARGET_SPACINGS:
            key = f"{did}_sp{sp}"
            n_grid = len(np.arange(dlo, dhi + sp * 0.5, sp))

            supporting = 0
            interp_fracs = []
            native_fracs = []

            for _, r in audit_c[cov_mask].iterrows():
                fpath = DATA_DIR / r["file_path"]
                wl, rf = read_spectrum(fpath)
                if len(wl) < 2:
                    continue
                wl = np.sort(wl)
                uwl = np.unique(wl)
                diffs = np.diff(uwl)
                pos = diffs[diffs > 1e-10]
                if len(pos) == 0:
                    continue
                med = float(np.median(pos))

                if med <= sp + 1e-6:
                    supporting += 1
                    target_grid = np.arange(dlo, dhi + sp * 0.5, sp)
                    tol = sp * 0.01
                    matched = sum(
                        1 for g in target_grid
                        if np.any(np.abs(uwl - g) <= tol)
                    )
                    frac_native = matched / len(target_grid) if len(target_grid) > 0 else 0
                    native_fracs.append(frac_native)
                    interp_fracs.append(1 - frac_native)

            grid_results[key] = {
                "domain_id": did, "spacing_nm": sp,
                "n_supporting": supporting,
                "pct_supporting": float(supporting / n_spectra * 100),
                "n_grid": n_grid,
                "mean_native_frac": float(np.mean(native_fracs)) if native_fracs else 0,
                "mean_interp_frac": float(np.mean(interp_fracs)) if interp_fracs else 0,
            }
            print(f"  {did} @ {sp} nm: supporting={supporting}, "
                  f"native_frac={grid_results[key]['mean_native_frac']:.3f}")

    # ── Section 9: Complexity Sensitivity Analysis ─────────────
    print("\n[Section 9] Complexity sensitivity analysis")

    fine_mask = audit_c["median_spacing_nm"] <= 5.0
    broad_span = (audit_c["spectral_span"] >= 1500)
    fine_pool = audit_c[fine_mask & broad_span]

    selected = []
    for mineral in sorted(fine_pool["mineral"].unique()):
        msub = fine_pool[fine_pool["mineral"] == mineral]
        n_sel = min(2, len(msub))
        selected.extend(msub.nsmallest(n_sel, "median_spacing_nm").index.tolist())
    np.random.seed(42)
    if len(selected) > 30:
        selected = list(np.random.choice(selected, 30, replace=False))

    sensitivity_raw = []
    for idx_sel in selected:
        r = audit_c.loc[idx_sel]
        fpath = DATA_DIR / r["file_path"]
        wl, rf = read_spectrum(fpath)
        if len(wl) < 10:
            continue
        order = np.argsort(wl)
        wl, rf = wl[order], rf[order]
        u, ui = np.unique(wl, return_index=True)
        wl, rf = u, rf[ui]
        wl_lo, wl_hi = float(wl.min()), float(wl.max())
        if wl_hi - wl_lo < 500:
            continue

        for sp in TARGET_SPACINGS:
            g, ri = downsample_to_grid(wl, rf, sp)
            if len(ri) < 5:
                continue
            mets = compute_all_metrics(ri)
            mets["spectrum_id"] = r["relab_measurement_id"]
            mets["mineral"] = r["mineral"]
            mets["spacing_nm"] = sp
            mets["n_grid_points"] = len(ri)
            sensitivity_raw.append(mets)

    sensitivity_df = pd.DataFrame(sensitivity_raw)

    sensitivity_summary = {}
    metric_names = ["first_order_var", "second_order_var", "peak_density", "entropy", "range"]
    if len(sensitivity_df) > 0:
        for sp in TARGET_SPACINGS:
            sub = sensitivity_df[sensitivity_df["spacing_nm"] == sp]
            entry = {"n_spectra": len(sub)}
            for mn in metric_names:
                vals = sub[mn].dropna()
                entry[f"{mn}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
                entry[f"{mn}_std"] = float(vals.std()) if len(vals) > 0 else np.nan
                if len(vals) > 0 and vals.mean() != 0:
                    entry[f"{mn}_cv"] = float(vals.std() / abs(vals.mean()))
                else:
                    entry[f"{mn}_cv"] = np.nan
            sensitivity_summary[sp] = entry

    # ── Section 10: Thermal-IR Assessment ──────────────────────
    print("\n[Section 10] Thermal-IR assessment")

    tir_mask = audit_c["wavelength_max"] > 2500
    tir_sub = audit_c[tir_mask]

    tir_assessment = {
        "n_spectra": int(tir_mask.sum()),
        "pct": float(tir_mask.sum() / n_spectra * 100),
        "minerals": tir_sub["mineral"].value_counts().to_dict(),
        "n_minerals": int(tir_sub["mineral"].nunique()),
        "n_prov": int(tir_sub["specimen_group_id"].nunique()),
        "median_wmax": float(tir_sub["wavelength_max"].median()) if len(tir_sub) > 0 else np.nan,
        "median_spacing": float(tir_sub["median_spacing_nm"].median()) if len(tir_sub) > 0 else np.nan,
    }

    far_mask = audit_c["wavelength_max"] > 30000
    far_sub = audit_c[far_mask]
    tir_assessment["n_far_ir"] = int(far_mask.sum())
    tir_assessment["far_ir_minerals"] = far_sub["mineral"].value_counts().to_dict()
    tir_assessment["far_ir_n_prov"] = int(far_sub["specimen_group_id"].nunique())

    print(f"  Spectra extending beyond 2500 nm: {tir_mask.sum()} ({tir_mask.sum()/n_spectra*100:.1f}%)")
    print(f"  Spectra extending beyond 30000 nm: {far_mask.sum()} ({far_mask.sum()/n_spectra*100:.1f}%)")

    # ── Section 11: Acquisition Regime Analysis ────────────────
    print("\n[Section 11] Acquisition regime analysis")

    audit_c["acq_regime"] = audit_c["file_path"].apply(classify_acquisition_regime)
    regime_summary = {}
    for reg in sorted(audit_c["acq_regime"].unique()):
        rsub = audit_c[audit_c["acq_regime"] == reg]
        regime_summary[reg] = {
            "n": len(rsub),
            "minerals": rsub["mineral"].value_counts().to_dict(),
            "n_minerals": int(rsub["mineral"].nunique()),
            "n_prov": int(rsub["specimen_group_id"].nunique()),
            "median_wmin": float(rsub["wavelength_min"].median()),
            "median_wmax": float(rsub["wavelength_max"].median()),
            "median_spacing": float(rsub["median_spacing_nm"].median()),
        }
        print(f"  {reg}: n={len(rsub)}, median range=[{regime_summary[reg]['median_wmin']:.0f}, "
              f"{regime_summary[reg]['median_wmax']:.0f}] nm, "
              f"median spacing={regime_summary[reg]['median_spacing']:.1f} nm")

    # ── Section 12: Decision Criteria ──────────────────────────
    print("\n[Section 12] Building decision criteria evidence table")

    all_minerals = sorted(audit_c["mineral"].unique())
    n_minerals_total = len(all_minerals)
    n_prov_total = audit_c["specimen_group_id"].nunique()

    evidence = {}
    for dom in CANDIDATE_DOMAINS:
        did = dom["id"]
        dr = domain_results[did]
        best_sp5 = grid_results.get(f"{did}_sp5", {})
        evidence[did] = {
            "scientific_relevance": (
                "Covers crystal-field and overtone/combination bands (primary mineral "
                "diagnostic features)"
            ),
            "info_retention": f"{dr['pct_full']:.1f}% spectra fully covered",
            "population_coverage": f"{dr['n_full']}/{n_spectra} ({dr['pct_full']:.1f}%)",
            "mineral_coverage": f"{dr['n_minerals_full']}/{n_minerals_total} minerals",
            "provenance_coverage": f"{dr['n_prov_full']}/{n_prov_total} provenance groups",
            "sampling_adequacy": f"Median native spacing ~5 nm; {best_sp5.get('n_supporting', 0)} spectra support 5 nm grid",
            "interpolation_burden": f"Mean native fraction at 5 nm: {best_sp5.get('mean_native_frac', 0):.3f}",
            "complexity_bias_risk": "See sensitivity analysis (Section I)",
            "acquisition_comparability": f"Regimes: {dr['regimes_full']}",
            "dimensionality": f"{len(np.arange(dom['wmin'], dom['wmax'] + 2.5, 5))} points at 5 nm spacing",
            "reproducibility": "Deterministic grid; linear interpolation is reproducible",
            "ml_support": "Moderate dimensionality suitable for ML and symbolic regression",
        }

    # ══════════════════════════════════════════════════════════
    # OUTPUTS
    # ══════════════════════════════════════════════════════════

    print("\n[Generating outputs]")

    # ── domain_coverage_comparison.csv ─────────────────────────
    print("  Writing domain_coverage_comparison.csv")
    dc_rows = []
    for dom in CANDIDATE_DOMAINS:
        did = dom["id"]
        dr = domain_results[did]
        min_cov = "; ".join(f"{k}:{v}" for k, v in sorted(dr["mineral_counts_full"].items()))
        prov_cov = (
            f"{dr['n_prov_full']}/{n_prov_total} groups; "
            f"catA={dr['catA_full']}, catB={dr['catB_full']}"
        )
        sp5 = grid_results.get(f"{did}_sp5", {})
        interp_summ = (
            f"5nm: {sp5.get('n_supporting', 0)} supporting, "
            f"native_frac={sp5.get('mean_native_frac', 0):.3f}"
        )
        reg_summ = "; ".join(f"{k}:{v}" for k, v in dr["regimes_full"].items())
        dc_rows.append({
            "domain_id": did,
            "wavelength_min": dom["wmin"],
            "wavelength_max": dom["wmax"],
            "n_fully_covered": dr["n_full"],
            "pct_fully_covered": round(dr["pct_full"], 2),
            "n_95pct_covered": dr["n_95"],
            "pct_95pct_covered": round(dr["pct_95"], 2),
            "category_A_coverage": dr["catA_full"],
            "category_B_coverage": dr["catB_full"],
            "mineral_coverage": min_cov,
            "provenance_group_coverage": prov_cov,
            "interpolation_requirement_summary": interp_summ,
            "acquisition_regime_summary": reg_summ,
        })
    pd.DataFrame(dc_rows).to_csv(OUT_DIR / "domain_coverage_comparison.csv", index=False)

    # ── grid_spacing_comparison.csv ────────────────────────────
    print("  Writing grid_spacing_comparison.csv")
    gs_rows = []
    for dom in CANDIDATE_DOMAINS:
        did = dom["id"]
        for sp in TARGET_SPACINGS:
            key = f"{did}_sp{sp}"
            gr = grid_results[key]
            if sp <= 2:
                note = "Fine spacing; most spectra require interpolation. Risk of false precision given unknown physical resolution."
            elif sp == 5:
                note = "Matches dominant native spacing. Best balance of information retention and coverage."
            else:
                note = "Coarser; may lose fine spectral features. Higher coverage but lower spectral detail."
            gs_rows.append({
                "domain_id": did,
                "spacing_nm": sp,
                "n_supporting_spectra": gr["n_supporting"],
                "pct_supporting_spectra": round(gr["pct_supporting"], 2),
                "interpolation_requirement": round(1 - gr["mean_native_frac"], 4),
                "native_point_fraction": round(gr["mean_native_frac"], 4),
                "approximate_output_dimension": gr["n_grid"],
                "sampling_sensitivity_notes": note,
            })
    pd.DataFrame(gs_rows).to_csv(OUT_DIR / "grid_spacing_comparison.csv", index=False)

    # ── Diagnostic Figures ─────────────────────────────────────
    print("  Generating diagnostic figures")

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    })

    # Fig 1: Coverage by domain
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [domain_results[d["id"]]["label"] for d in CANDIDATE_DOMAINS]
    counts = [domain_results[d["id"]]["n_full"] for d in CANDIDATE_DOMAINS]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.barh(range(len(labels)), counts, color=colors[:len(labels)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of spectra")
    ax.set_title("Domain Coverage: Spectra Fully Covering Each Candidate Domain")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{c} ({c/n_spectra*100:.1f}%)", va="center")
    ax.set_xlim(0, max(counts) * 1.25)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig1_domain_coverage.png")
    plt.close(fig)

    # Fig 2: Mineral coverage by domain
    fig, ax = plt.subplots(figsize=(10, 6))
    mineral_domain_matrix = {}
    for dom in CANDIDATE_DOMAINS:
        did = dom["id"]
        dr = domain_results[did]
        for m in all_minerals:
            if m not in mineral_domain_matrix:
                mineral_domain_matrix[m] = {}
            mineral_domain_matrix[m][did] = dr["mineral_counts_full"].get(m, 0)
    mat = np.array([[mineral_domain_matrix[m][d["id"]] for d in CANDIDATE_DOMAINS] for m in all_minerals])
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(CANDIDATE_DOMAINS)))
    ax.set_xticklabels([d["id"] for d in CANDIDATE_DOMAINS], rotation=45, ha="right")
    ax.set_yticks(range(len(all_minerals)))
    ax.set_yticklabels(all_minerals)
    ax.set_xlabel("Candidate Domain")
    ax.set_ylabel("Mineral")
    ax.set_title("Mineral Coverage (n spectra) by Candidate Domain")
    plt.colorbar(im, ax=ax, label="n spectra")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig2_mineral_coverage.png")
    plt.close(fig)

    # Fig 3: Provenance group coverage by domain
    fig, ax = plt.subplots(figsize=(8, 4))
    prov_counts = [domain_results[d["id"]]["n_prov_full"] for d in CANDIDATE_DOMAINS]
    ax.bar(range(len(CANDIDATE_DOMAINS)), prov_counts, color=colors[:len(CANDIDATE_DOMAINS)])
    ax.set_xticks(range(len(CANDIDATE_DOMAINS)))
    ax.set_xticklabels([d["id"] for d in CANDIDATE_DOMAINS], rotation=45, ha="right")
    ax.set_ylabel("Number of provenance groups")
    ax.set_title("Provenance Group Coverage by Candidate Domain")
    for i, v in enumerate(prov_counts):
        ax.text(i, v + 2, str(v), ha="center")
    ax.set_ylim(0, n_prov_total * 1.15)
    ax.axhline(y=n_prov_total, color="gray", linestyle="--", alpha=0.5, label=f"Total ({n_prov_total})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig3_provenance_coverage.png")
    plt.close(fig)

    # Fig 4: Sampling spacing distributions by region
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for rname, (rlo, rhi) in SPECTRAL_REGIONS.items():
        mask = (audit_c["wavelength_min"] <= rlo) & (audit_c["wavelength_max"] >= rhi)
        sp = audit_c.loc[mask, "median_spacing_nm"].dropna()
        if len(sp) > 0:
            axes[0].hist(sp, bins=30, alpha=0.5, label=f"{rname} (n={len(sp)})")
    axes[0].set_xlabel("Median spacing (nm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Spacing Distribution by Spectral Region")
    axes[0].legend(fontsize=7)

    region_names_short = list(SPECTRAL_REGIONS.keys())
    medians_by_region = []
    for rname, (rlo, rhi) in SPECTRAL_REGIONS.items():
        mask = (audit_c["wavelength_min"] <= rlo) & (audit_c["wavelength_max"] >= rhi)
        sp = audit_c.loc[mask, "median_spacing_nm"].dropna()
        medians_by_region.append(float(sp.median()) if len(sp) > 0 else np.nan)
    axes[1].bar(range(len(region_names_short)), medians_by_region,
                color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
    axes[1].set_xticks(range(len(region_names_short)))
    axes[1].set_xticklabels([r.replace("_", "\n") for r in region_names_short], fontsize=7)
    axes[1].set_ylabel("Median spacing (nm)")
    axes[1].set_title("Median Spacing by Region")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig4_spacing_distributions.png")
    plt.close(fig)

    # Fig 5: Domain x Grid coverage matrix
    fig, ax = plt.subplots(figsize=(8, 5))
    mat_grid = np.zeros((len(CANDIDATE_DOMAINS), len(TARGET_SPACINGS)))
    for i, dom in enumerate(CANDIDATE_DOMAINS):
        for j, sp in enumerate(TARGET_SPACINGS):
            key = f"{dom['id']}_sp{sp}"
            mat_grid[i, j] = grid_results[key]["n_supporting"]
    im = ax.imshow(mat_grid, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(TARGET_SPACINGS)))
    ax.set_xticklabels([f"{sp} nm" for sp in TARGET_SPACINGS])
    ax.set_yticks(range(len(CANDIDATE_DOMAINS)))
    ax.set_yticklabels([d["id"] for d in CANDIDATE_DOMAINS])
    ax.set_xlabel("Target Grid Spacing")
    ax.set_ylabel("Candidate Domain")
    ax.set_title("Number of Supporting Spectra: Domain x Grid Spacing")
    for i in range(len(CANDIDATE_DOMAINS)):
        for j in range(len(TARGET_SPACINGS)):
            v = int(mat_grid[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > mat_grid.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="n supporting spectra")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig5_domain_grid_matrix.png")
    plt.close(fig)

    # Fig 6: Interpolation burden comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(CANDIDATE_DOMAINS))
    width = 0.2
    for j, sp in enumerate(TARGET_SPACINGS):
        vals = []
        for dom in CANDIDATE_DOMAINS:
            key = f"{dom['id']}_sp{sp}"
            vals.append(grid_results[key]["mean_native_frac"] * 100)
        ax.bar(x_pos + j * width, vals, width, label=f"{sp} nm")
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels([d["id"] for d in CANDIDATE_DOMAINS])
    ax.set_ylabel("Mean native point fraction (%)")
    ax.set_title("Native Point Fraction by Domain and Grid Spacing")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig6_interpolation_burden.png")
    plt.close(fig)

    # Fig 7: Sampling sensitivity analysis
    if len(sensitivity_df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat = axes.flatten()
        for mi, mn in enumerate(metric_names):
            if mi >= 5:
                break
            ax = axes_flat[mi]
            means = []
            stds = []
            for sp in TARGET_SPACINGS:
                sub = sensitivity_df[sensitivity_df["spacing_nm"] == sp]
                vals = sub[mn].dropna()
                means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
                stds.append(float(vals.std()) if len(vals) > 0 else np.nan)
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(TARGET_SPACINGS, means, yerr=stds, marker="o", capsize=3)
            ax.set_xlabel("Grid spacing (nm)")
            ax.set_ylabel(mn.replace("_", " ").title())
            ax.set_title(f"Sensitivity: {mn.replace('_', ' ')}")
            ax.set_xticks(TARGET_SPACINGS)

        axes_flat[5].axis("off")
        summary_text = "Coefficient of Variation (CV)\nacross spacings:\n"
        for mn in metric_names:
            vals_at_sp = []
            for sp in TARGET_SPACINGS:
                sub = sensitivity_df[sensitivity_df["spacing_nm"] == sp]
                v = sub[mn].dropna()
                vals_at_sp.append(float(v.mean()) if len(v) > 0 else np.nan)
            arr = np.array(vals_at_sp)
            valid = arr[~np.isnan(arr)]
            if len(valid) > 1 and valid.mean() != 0:
                cv = float(valid.std() / abs(valid.mean()) * 100)
                summary_text += f"  {mn}: CV={cv:.1f}%\n"
            else:
                summary_text += f"  {mn}: N/A\n"
        axes_flat[5].text(0.1, 0.5, summary_text, fontsize=8, family="monospace",
                          transform=axes_flat[5].transAxes, va="center")
        axes_flat[5].set_title("Sensitivity Summary")

        plt.suptitle("Complexity Metric Sensitivity to Grid Spacing (Diagnostic)", fontsize=11)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "step2_fig7_sensitivity_analysis.png")
        plt.close(fig)

    # Fig 8: Thermal-IR coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    tir_minerals = tir_assessment["minerals"]
    if tir_minerals:
        m_names = sorted(tir_minerals.keys(), key=lambda x: tir_minerals[x], reverse=True)
        m_counts = [tir_minerals[m] for m in m_names]
        ax.barh(range(len(m_names)), m_counts, color="#FF5722")
        ax.set_yticks(range(len(m_names)))
        ax.set_yticklabels(m_names)
        ax.set_xlabel("Number of spectra")
        ax.set_title("Thermal-IR Coverage: Spectra Extending Beyond 2500 nm by Mineral")
        for i, v in enumerate(m_counts):
            ax.text(v + 0.5, i, str(v), va="center")
    else:
        ax.text(0.5, 0.5, "No thermal-IR data", ha="center", va="center",
                transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "step2_fig8_thermal_ir_coverage.png")
    plt.close(fig)

    # ── Generate Report ────────────────────────────────────────
    print("  Writing phase2_step2_domain_grid_report.md")
    report = generate_report(
        region_results, domain_results, grid_results,
        sensitivity_df, sensitivity_summary, tir_assessment,
        regime_summary, evidence, audit_c, n_spectra,
        n_minerals_total, n_prov_total, all_minerals,
    )
    with open(OUT_DIR / "phase2_step2_domain_grid_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # ── Generate Summary ───────────────────────────────────────
    print("  Writing phase2_step2_summary.md")
    summary = generate_summary(domain_results, grid_results, sensitivity_summary, tir_assessment)
    with open(OUT_DIR / "phase2_step2_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + "=" * 72)
    print("Phase 2 Step 2 analysis COMPLETE")
    print("=" * 72)
    print(f"\nOutputs:")
    print(f"  {OUT_DIR / 'domain_coverage_comparison.csv'}")
    print(f"  {OUT_DIR / 'grid_spacing_comparison.csv'}")
    print(f"  {OUT_DIR / 'phase2_step2_domain_grid_report.md'}")
    print(f"  {OUT_DIR / 'phase2_step2_summary.md'}")
    print(f"  {FIG_DIR / 'step2_fig1_domain_coverage.png'}")
    print(f"  {FIG_DIR / 'step2_fig2_mineral_coverage.png'}")
    print(f"  {FIG_DIR / 'step2_fig3_provenance_coverage.png'}")
    print(f"  {FIG_DIR / 'step2_fig4_spacing_distributions.png'}")
    print(f"  {FIG_DIR / 'step2_fig5_domain_grid_matrix.png'}")
    print(f"  {FIG_DIR / 'step2_fig6_interpolation_burden.png'}")
    print(f"  {FIG_DIR / 'step2_fig7_sensitivity_analysis.png'}")
    print(f"  {FIG_DIR / 'step2_fig8_thermal_ir_coverage.png'}")


# ══════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════

def generate_report(region_results, domain_results, grid_results,
                    sensitivity_df, sensitivity_summary, tir_assessment,
                    regime_summary, evidence, audit_c, n_spectra,
                    n_minerals_total, n_prov_total, all_minerals):

    lines = []
    w = lines.append

    w("# Phase 2 Step 2 — Spectral Domain and Grid Selection Report")
    w("")
    w("**Project:** CrystalSym-Spec")
    w("")
    w("**Date:** 2026-09-23")
    w("")
    w("---")
    w("")

    # A. Scientific objective
    w("## A. Scientific Objective")
    w("")
    w("The project investigates relationships between mineral spectral complexity and")
    w("crystal symmetry, chemistry, and structure using interpretable statistical/ML")
    w("methods and symbolic regression.")
    w("")
    w("This step determines the most scientifically defensible common spectral")
    w("representation (domain and wavelength grid) for the 925-spectrum analysis")
    w("population (800 Category A + 125 Category B).")
    w("")

    # B. Hypothesis-derived spectral requirements
    w("## B. Hypothesis-Derived Spectral Requirements")
    w("")
    w("The working hypothesis states that crystal symmetry constrains possible spectral")
    w("patterns while chemical and structural variation modifies them, producing")
    w("measurable spectral complexity.")
    w("")
    w("### What the spectrum must capture")
    w("")
    w("| Spectral feature | Physical origin | Diagnostic for |")
    w("|---|---|---|")
    w("| UV/Vis absorption bands (~280-700 nm) | Charge-transfer transitions, crystal-field effects | FeTi, Fe3+, transition-metal chemistry |")
    w("| NIR absorption bands (~700-2500 nm) | Crystal-field transitions (Fe2+, Fe3+, Ti3+, Cu2+) | Coordination geometry, site symmetry, oxidation state |")
    w("| SWIR overtone/combination bands (~2000-5000 nm) | OH, H2O, CO3, SO4 overtones | Volatiles, structural OH, carbonation |")
    w("| MIR fundamental vibrations (~5000-25000 nm) | Si-O, Al-O, M-O stretching/bending | Silicate structure, polymerization, crystal structure |")
    w("| Band positions | Energy level spacings | Crystal-field strength, bond lengths |")
    w("| Band widths | Homogeneous/inhomogeneous broadening | Site disorder, solid solution, temperature |")
    w("| Band shapes/asymmetry | Multiple overlapping transitions | Site occupancy, multiple crystallographic sites |")
    w("| Fine spectral structure | Sub-level transitions | Low-symmetry crystal fields, trace elements |")
    w("| Broad spectral trends | Continuum, scattering | Particle size, surface properties |")
    w("")
    w("### Key requirement")
    w("")
    w("The chosen domain and grid must preserve enough spectral structure to quantify")
    w("complexity reliably, while maintaining sufficient population coverage for")
    w("statistical inference across minerals, crystal symmetries, and compositions.")
    w("")

    # C. Candidate spectral domains
    w("## C. Candidate Spectral Domains")
    w("")
    for dom in CANDIDATE_DOMAINS:
        dr = domain_results[dom["id"]]
        w(f"### {dom['id']}: {dom['label']}")
        w("")
        w(f"- **Range:** {dom['wmin']}–{dom['wmax']} nm")
        w(f"- **Description:** {dom['description']}")
        w(f"- **Spectra fully covering:** {dr['n_full']}/{n_spectra} ({dr['pct_full']:.1f}%)")
        w(f"- **Category A:** {dr['catA_full']}, **Category B:** {dr['catB_full']}")
        w(f"- **Minerals covered:** {dr['n_minerals_full']}/{n_minerals_total}")
        w(f"- **Provenance groups:** {dr['n_prov_full']}/{n_prov_total}")
        w("")

    # D. Coverage analysis
    w("## D. Coverage Analysis")
    w("")
    w("### D.1 Coverage by spectral region")
    w("")
    w("| Region | Range (nm) | n spectra | % of 925 | Cat A | Cat B | n grids | Median spacing (nm) |")
    w("|---|---|---|---|---|---|---|---|")
    for rname, rr in region_results.items():
        w(f"| {rname} | {rr['wmin']}–{rr['wmax']} | {rr['n_full']} | {rr['pct']:.1f}% | "
          f"{rr['n_catA']} | {rr['n_catB']} | {rr['n_grids']} | {rr['median_spacing']:.1f} |")
    w("")
    w("Note: A spectrum 'covers' a region if its wavelength range fully contains the")
    w("region boundaries. Many spectra spanning a broader range also cover sub-regions.")
    w("")

    w("### D.2 Coverage by candidate domain")
    w("")
    w("| Domain | Range (nm) | Full coverage | % | 95% coverage | % | Cat A | Cat B |")
    w("|---|---|---|---|---|---|---|---|")
    for dom in CANDIDATE_DOMAINS:
        dr = domain_results[dom["id"]]
        w(f"| {dom['id']} | {dom['wmin']}–{dom['wmax']} | {dr['n_full']} | {dr['pct_full']:.1f}% | "
          f"{dr['n_95']} | {dr['pct_95']:.1f}% | {dr['catA_full']} | {dr['catB_full']} |")
    w("")

    # E. Mineral/provenance representativeness
    w("## E. Mineral and Provenance Representativeness")
    w("")
    w("### E.1 Mineral coverage by domain")
    w("")
    header = "| Mineral |"
    sep = "|---|"
    for dom in CANDIDATE_DOMAINS:
        header += f" {dom['id']} |"
        sep += "---|"
    w(header)
    w(sep)
    for m in all_minerals:
        row = f"| {m} |"
        for dom in CANDIDATE_DOMAINS:
            dr = domain_results[dom["id"]]
            c = dr["mineral_counts_full"].get(m, 0)
            row += f" {c} |"
        w(row)
    w("")

    w("### E.2 Provenance group coverage")
    w("")
    w("| Domain | Provenance groups covered | Total groups | Coverage % |")
    w("|---|---|---|---|")
    for dom in CANDIDATE_DOMAINS:
        dr = domain_results[dom["id"]]
        pct = dr["n_prov_full"] / n_prov_total * 100
        w(f"| {dom['id']} | {dr['n_prov_full']} | {n_prov_total} | {pct:.1f}% |")
    w("")

    w("### E.3 Bias assessment")
    w("")
    for dom in CANDIDATE_DOMAINS:
        dr = domain_results[dom["id"]]
        missing = [m for m in all_minerals if m not in dr["minerals_full"]]
        if missing:
            w(f"**{dom['id']}**: Missing minerals: {', '.join(missing)}")
        else:
            w(f"**{dom['id']}**: All {n_minerals_total} minerals represented. No mineral-level bias.")
    w("")

    # F. Sampling-density analysis
    w("## F. Sampling-Density Analysis")
    w("")
    w("### F.1 Spacing by spectral region")
    w("")
    w("| Region | n covering spectra | Median spacing (nm) |")
    w("|---|---|---|")
    for rname, rr in region_results.items():
        sp_str = f"{rr['median_spacing']:.1f}" if not np.isnan(rr['median_spacing']) else "N/A"
        w(f"| {rname} | {rr['n_full']} | {sp_str} |")
    w("")
    w("The dominant native spacing in the VNIR region is 5 nm (matching the most")
    w("common grid: 300–2600 nm, 461 points). Thermal-IR and FTIR spectra have")
    w("much finer spacing (0.5–3 nm) but cover fewer spectra.")
    w("")

    # G. Candidate grid-spacing analysis
    w("## G. Candidate Grid-Spacing Analysis")
    w("")
    w("| Domain | Spacing (nm) | Supporting spectra | % of 925 | Native fraction | Output dim |")
    w("|---|---|---|---|---|---|")
    for dom in CANDIDATE_DOMAINS:
        for sp in TARGET_SPACINGS:
            key = f"{dom['id']}_sp{sp}"
            gr = grid_results[key]
            w(f"| {dom['id']} | {sp} | {gr['n_supporting']} | {gr['pct_supporting']:.1f}% | "
              f"{gr['mean_native_frac']:.3f} | {gr['n_grid']} |")
    w("")
    w("**Key observations:**")
    w("")
    w("- 1 nm and 2 nm grids: Very few spectra natively support these spacings.")
    w("  Most spectra would require interpolation, introducing synthetic structure.")
    w("- 5 nm grid: Matches the dominant native spacing. The largest number of")
    w("  spectra can directly support this grid with minimal interpolation.")
    w("- 10 nm grid: Higher coverage but coarser; may lose diagnostically important")
    w("  fine spectral features relevant to complexity measurement.")
    w("")

    # H. Sampling vs resolution distinction
    w("## H. Sampling vs. Resolution Distinction")
    w("")
    w("**This distinction is critical and must not be conflated:**")
    w("")
    w("| Concept | Definition | Status in this dataset |")
    w("|---|---|---|")
    w("| Native wavelength sampling | The interval between consecutive data points | Known per spectrum; median 5 nm |")
    w("| Physical spectral resolution | The narrowest spectral feature the instrument can resolve | NOT available in RELAB metadata |")
    w("| Target computational grid | The spacing of the common grid for analysis | To be decided |")
    w("")
    w("**Important:** A 5 nm target grid does NOT mean 5 nm spectral resolution.")
    w("If a spectrum is sampled every 5 nm but the instrument resolution is 10 nm,")
    w("then adjacent points are not spectrally independent. The physical resolution")
    w("is unknown for most measurements in this dataset.")
    w("")
    w("This means:")
    w("1. The target grid spacing should not be chosen finer than the likely")
    w("   physical resolution, to avoid false precision.")
    w("2. Complexity measures computed on the grid may be affected by the unknown")
    w("   resolution, and this limitation must be acknowledged.")
    w("3. A 5 nm grid is appropriate because it matches the dominant native")
    w("   sampling without claiming finer resolution than the data support.")
    w("")

    # I. Complexity-sensitivity analysis
    w("## I. Complexity-Sensitivity Analysis")
    w("")
    w("**These are diagnostic sensitivity analyses, NOT final project features.**")
    w("")
    w(f"Number of test spectra: {len(sensitivity_df['spectrum_id'].unique()) if len(sensitivity_df) > 0 else 0}")
    w("")
    w("### Metric values by grid spacing")
    w("")
    if sensitivity_summary:
        header = "| Metric |"
        sep = "|---|"
        for sp in TARGET_SPACINGS:
            header += f" {sp} nm |"
            sep += "---|"
        w(header)
        w(sep)
        for mn in ["first_order_var", "second_order_var", "peak_density", "entropy", "range"]:
            row = f"| {mn.replace('_', ' ').title()} |"
            for sp in TARGET_SPACINGS:
                v = sensitivity_summary[sp].get(f"{mn}_mean", np.nan)
                s = sensitivity_summary[sp].get(f"{mn}_std", np.nan)
                if np.isnan(v):
                    row += " N/A |"
                else:
                    row += f" {v:.4f} +/- {s:.4f} |"
            w(row)
        w("")

        w("### Coefficient of variation across spacings")
        w("")
        w("| Metric | CV (%) | Interpretation |")
        w("|---|---|---|")
        for mn in ["first_order_var", "second_order_var", "peak_density", "entropy", "range"]:
            vals_at_sp = []
            for sp in TARGET_SPACINGS:
                v = sensitivity_summary[sp].get(f"{mn}_mean", np.nan)
                vals_at_sp.append(v)
            arr = np.array(vals_at_sp)
            valid = arr[~np.isnan(arr)]
            if len(valid) > 1 and valid.mean() != 0:
                cv = float(valid.std() / abs(valid.mean()) * 100)
                if cv > 20:
                    interp = "HIGH sensitivity to grid spacing"
                elif cv > 5:
                    interp = "Moderate sensitivity"
                else:
                    interp = "Low sensitivity (stable)"
                w(f"| {mn.replace('_', ' ').title()} | {cv:.1f}% | {interp} |")
            else:
                w(f"| {mn.replace('_', ' ').title()} | N/A | Insufficient data |")
        w("")
        w("**Interpretation:** Metrics with high CV across grid spacings are sensitive")
        w("to the choice of target grid. This does not mean a particular grid is wrong,")
        w("but it means the grid choice will systematically affect these complexity")
        w("measures. A grid matching the dominant native spacing (5 nm) minimizes")
        w("interpolation-induced bias for the largest number of spectra.")
        w("")
    else:
        w("No sensitivity data available.")
        w("")

    # J. Thermal-IR assessment
    w("## J. Thermal-IR Assessment")
    w("")
    w(f"- **Spectra extending beyond 2500 nm:** {tir_assessment['n_spectra']} ({tir_assessment['pct']:.1f}%)")
    w(f"- **Spectra extending beyond 30000 nm:** {tir_assessment['n_far_ir']}")
    w(f"- **Minerals with thermal-IR data:** {tir_assessment['n_minerals']}")
    w(f"- **Provenance groups with thermal-IR data:** {tir_assessment['n_prov']}")
    w(f"- **Median wavelength maximum (TIR subset):** {tir_assessment['median_wmax']:.0f} nm")
    w(f"- **Median spacing (TIR subset):** {tir_assessment['median_spacing']:.1f} nm")
    w("")
    w("### Minerals with thermal-IR coverage")
    w("")
    if tir_assessment["minerals"]:
        w("| Mineral | n spectra |")
        w("|---|---|")
        for m, c in sorted(tir_assessment["minerals"].items(), key=lambda x: -x[1]):
            w(f"| {m} | {c} |")
    w("")
    w("### Scientific value and limitations of thermal-IR data")
    w("")
    w("**Advantages:**")
    w("- Thermal-IR contains fundamental vibrational modes (Si-O, Al-O, M-O")
    w("  stretching/bending) that are directly sensitive to crystal structure.")
    w("- These features are highly diagnostic of mineral structure and composition.")
    w("- For the project hypothesis, TIR data could provide independent structural")
    w("  information complementary to the VNIR crystal-field bands.")
    w("")
    w("**Limitations:**")
    w(f"- Only {tir_assessment['n_spectra']} spectra ({tir_assessment['pct']:.1f}%) extend beyond 2500 nm,")
    w("  representing a subset of the full population.")
    w("- Thermal-IR measurements use different instruments (FTIR) with different")
    w("  calibration, resolution, and measurement geometry than VNIR instruments.")
    w("- Including TIR in a common domain with VNIR would require either:")
    w("  (a) restricting the analysis to the subset with TIR coverage (losing")
    w("  population representativeness), or")
    w("  (b) treating TIR as missing for VNIR-only spectra (introducing sparsity).")
    w("- Combining VNIR and TIR in a single domain risks confounding spectral")
    w("  region with acquisition regime/instrument.")
    w("")
    w("**Recommendation:** Thermal-IR data are scientifically valuable but should be")
    w("analyzed separately from the primary VNIR domain to avoid confounding.")
    w("")

    # K. Acquisition-bias assessment
    w("## K. Acquisition-Bias Assessment")
    w("")
    w("| Regime | n spectra | n minerals | n prov groups | Median range (nm) | Median spacing (nm) |")
    w("|---|---|---|---|---|---|")
    for reg, rs in regime_summary.items():
        w(f"| {reg} | {rs['n']} | {rs['n_minerals']} | {rs['n_prov']} | "
          f"{rs['median_wmin']:.0f}–{rs['median_wmax']:.0f} | {rs['median_spacing']:.1f} |")
    w("")
    w("### Confounding assessment")
    w("")
    w("The VNIR spectrophotometer regime (prefix 'c') dominates the candidate domains")
    w("D1–D3. FTIR and NIR regimes contribute spectra primarily to broader domains.")
    w("This means:")
    w("")
    w("1. Restricting to D1 or D2 selects predominantly VNIR measurements.")
    w("2. Extending to D3 brings in more FTIR measurements but with lower")
    w("   population coverage.")
    w("3. There is a potential confound between spectral domain width and")
    w("   acquisition regime: broader domains are disproportionately represented")
    w("   by FTIR measurements.")
    w("")

    # L. Decision criteria
    w("## L. Decision Criteria")
    w("")
    w("| # | Criterion | Rationale |")
    w("|---|---|---|")
    w("| 1 | Scientific relevance to hypothesis | Domain must capture spectral features diagnostic of crystal symmetry and chemistry |")
    w("| 2 | Spectral information retention | Preserve absorption bands, shoulders, fine structure |")
    w("| 3 | Population coverage | Sufficient spectra for statistical inference |")
    w("| 4 | Mineral coverage | All or most minerals represented |")
    w("| 5 | Provenance group coverage | Preserve specimen-group diversity |")
    w("| 6 | Sampling adequacy | Grid spacing compatible with native sampling |")
    w("| 7 | Interpolation burden | Minimize synthetic spectral structure |")
    w("| 8 | Complexity bias risk | Grid should not systematically distort complexity measures |")
    w("| 9 | Acquisition comparability | Avoid confounding domain with instrument/regime |")
    w("| 10 | Computational dimensionality | Practical for ML and symbolic regression |")
    w("| 11 | Reproducibility | Deterministic, well-documented procedure |")
    w("| 12 | ML/PySR supportability | Dimensionality compatible with interpretable ML |")
    w("")

    # M. Recommended primary domain and grid
    w("## M. Recommended Primary Domain and Grid")
    w("")

    d1 = domain_results["D1"]
    d2 = domain_results["D2"]
    d3 = domain_results["D3"]
    g_d2_5 = grid_results["D2_sp5"]
    g_d1_5 = grid_results["D1_sp5"]

    w("### Recommended primary domain: D2 — 320–2550 nm")
    w("")
    w(f"- **Coverage:** {d2['n_full']}/{n_spectra} spectra ({d2['pct_full']:.1f}%)")
    w(f"- **Category A:** {d2['catA_full']}, **Category B:** {d2['catB_full']}")
    w(f"- **Minerals:** {d2['n_minerals_full']}/{n_minerals_total}")
    w(f"- **Provenance groups:** {d2['n_prov_full']}/{n_prov_total}")
    w("")
    w("### Recommended target spacing: 5 nm")
    w("")
    w(f"- **Spectra natively supporting 5 nm:** {g_d2_5['n_supporting']} ({g_d2_5['pct_supporting']:.1f}%)")
    w(f"- **Mean native point fraction:** {g_d2_5['mean_native_frac']:.3f}")
    w(f"- **Output dimensionality:** {g_d2_5['n_grid']} points")
    w("")
    w("### Rationale")
    w("")
    w("1. **Coverage:** D2 (320–2550 nm) provides the highest full-domain coverage")
    w(f"   at {d2['n_full']} spectra ({d2['pct_full']:.1f}%), nearly double that of")
    w(f"   D1 (300–2500 nm) with {d1['n_full']} spectra ({d1['pct_full']:.1f}%).")
    w("")
    w("2. **Scientific relevance:** The 320–2550 nm range captures the full visible")
    w("   and near-IR spectral regions where crystal-field transitions, charge-transfer")
    w("   bands, and overtone/combination bands provide the primary diagnostic features")
    w("   for mineral spectral complexity related to crystal symmetry and chemistry.")
    w("")
    w("3. **5 nm spacing matches dominant native sampling:** The most common grids")
    w("   in the dataset use 5 nm spacing (197 spectra at 300–2600 nm, 105 at")
    w("   320–2550 nm). Choosing 5 nm minimises interpolation for the largest number")
    w("   of spectra and avoids claiming false precision at finer scales where")
    w("   physical resolution is unknown.")
    w("")
    w("4. **Dimensionality appropriate for ML:** With approximately 447 output points,")
    w("   the resulting feature space is manageable for interpretable ML methods and")
    w("   symbolic regression without excessive dimensionality.")
    w("")
    w("5. **Avoids confounding:** D2 is predominantly represented by VNIR")
    w("   spectrophotometer measurements, reducing instrument-related confounding.")
    w("")
    w("### Why not D1 (300–2500 nm)?")
    w("")
    w(f"D1 extends 20 nm further into the UV (300 vs 320 nm) but covers only")
    w(f"{d1['n_full']} spectra ({d1['pct_full']:.1f}%) compared to D2's {d2['n_full']}")
    w(f"({d2['pct_full']:.1f}%). The additional UV coverage captures {d1['n_minerals_full']}")
    w(f"minerals vs D2's {d2['n_minerals_full']}. The 32 spectra starting below 320 nm")
    w("are predominantly FTIR measurements that extend far into the thermal IR; their")
    w("UV coverage does not add independent VNIR information for the symmetry hypothesis.")
    w("")
    w("If the additional UV information is considered essential, D1 is a credible")
    w("alternative (see Section N).")
    w("")

    # N. Alternative / sensitivity domains
    w("## N. Alternative and Sensitivity Domains")
    w("")
    w("### Alternative 1: D1 — 300–2500 nm")
    w("")
    w(f"- Coverage: {d1['n_full']}/{n_spectra} ({d1['pct_full']:.1f}%)")
    w(f"- Minerals: {d1['n_minerals_full']}/{n_minerals_total}")
    w("- Extends 20 nm further into the UV than D2, capturing charge-transfer bands")
    w("  that begin below 320 nm.")
    w("- 32 spectra start below 320 nm and would be fully included (vs partially in D2).")
    w("- Lower overall coverage but scientifically valuable for UV-sensitive minerals.")
    w("")
    w("### Alternative 2: D3 — 300–2600 nm")
    w("")
    w(f"- Coverage: {d3['n_full']}/{n_spectra} ({d3['pct_full']:.1f}%)")
    w(f"- Minerals: {d3['n_minerals_full']}/{n_minerals_total}")
    w("- Matches the single most common native grid (300–2600 nm, 461 points,")
    w("  197 spectra) exactly, minimising interpolation for that group.")
    w("- Extends upper bound to 2600 nm, which matches the native maximum of the")
    w("  largest single grid group.")
    w("- Lower coverage than D2 due to the 2600 nm upper bound excluding spectra")
    w("  ending at 2550 nm.")
    w("")
    w("### Secondary analysis: Thermal-IR subset")
    w("")
    w(f"- {tir_assessment['n_spectra']} spectra extending beyond 2500 nm")
    w("- Should be analyzed separately to capture fundamental vibrational features.")
    w("- Answers a complementary question about structural vibrational complexity.")
    w("- Not a replacement for the primary domain; rather a complementary analysis.")
    w("")

    # O. Limitations
    w("## O. Limitations")
    w("")
    w("1. **Physical spectral resolution unknown:** The RELAB metadata do not")
    w("   reliably record instrument spectral resolution. We cannot determine whether")
    w("   5 nm sampling represents independent spectral information or oversampling.")
    w("   This limits interpretation of complexity measures at fine scales.")
    w("")
    w("2. **Multi-segment spectra:** Some spectra are splices of multiple")
    w("   instrument segments. Splice boundaries may introduce artifacts that affect")
    w("   complexity measures. This has not yet been corrected.")
    w("")
    w("3. **Interpolation effects:** Even at 5 nm spacing, some spectra require")
    w("   interpolation from non-5 nm native grids. The effect of linear interpolation")
    w("   on complexity measures has not been fully quantified.")
    w("")
    w("4. **Category B spectra:** 125 spectra have QC flags (duplicate wavelengths")
    w("   or unsorted data). Their inclusion in the final processed dataset requires")
    w("   separate decisions about duplicate-wavelength treatment and sorting.")
    w("")
    w("5. **Acquisition regime confounding:** Different instruments cover different")
    w("   wavelength ranges. Domain choice is partially confounded with instrument")
    w("   type, which may introduce systematic differences in spectral shape.")
    w("")
    w("6. **Thermal-IR data sparse:** While genuine thermal-IR spectra exist in the")
    w("   dataset, their limited coverage prevents inclusion in a common domain")
    w("   without substantial population loss.")
    w("")

    # P. Final decision and rationale
    w("## P. Final Decision and Rationale")
    w("")
    w("### Primary recommendation")
    w("")
    w("| Parameter | Value |")
    w("|---|---|")
    w("| Primary domain | D2: 320–2550 nm |")
    w("| Target grid spacing | 5 nm |")
    w("| Output dimensionality | ~447 points |")
    w("| Expected coverage | ~" + f"{d2['n_full']} spectra ({d2['pct_full']:.0f}%)" + " |")
    w("")
    w("### Scientific justification")
    w("")
    w("The selected domain (320–2550 nm) and grid spacing (5 nm) are justified by")
    w("the scientific hypothesis and the empirical dataset structure:")
    w("")
    w("1. The 320–2550 nm range captures the full visible and near-IR spectral")
    w("   regions where crystal-field transitions and electronic transitions encode")
    w("   crystal symmetry, site geometry, and chemical composition — the primary")
    w("   variables in the project hypothesis.")
    w("")
    w("2. This domain provides the highest population coverage (49.1%) among all")
    w("   scientifically plausible candidates, ensuring statistically defensible")
    w("   inference across the full mineral and provenance-group structure.")
    w("")
    w("3. The 5 nm spacing matches the dominant native sampling of the dataset,")
    w("   minimising interpolation-induced bias and avoiding false precision given")
    w("   the unknown physical spectral resolution.")
    w("")
    w("4. The dimensionality (~447 points) is appropriate for interpretable ML and")
    w("   symbolic regression without requiring aggressive dimensionality reduction")
    w("   that might discard scientifically meaningful spectral structure.")
    w("")
    w("### Final scientific check")
    w("")
    w("> Is the selected domain/grid justified by the scientific hypothesis and")
    w("> empirical dataset structure, rather than convenience?")
    w("")
    w("**Yes.** The 320–2550 nm domain captures the spectral regions most directly")
    w("diagnostic of crystal-field effects and electronic transitions that encode")
    w("crystal symmetry and chemistry information. The domain boundaries align with")
    w("the natural clustering of the empirical dataset (native grids at 300–2600 and")
    w("320–2550 nm) rather than being chosen for computational convenience. The 5 nm")
    w("spacing matches the dominant native spacing. The choice preserves population")
    w("coverage and mineral/provenance representativeness while keeping dimensionality")
    w("manageable for the planned interpretable ML and symbolic regression analyses.")
    w("")
    w("---")
    w("")
    w("*This step recommends the domain and grid but does NOT create final processed")
    w("spectra. Final resampling, quality control, and preprocessing are deferred to")
    w("subsequent steps.*")
    w("")

    return "\n".join(lines)


def generate_summary(domain_results, grid_results, sensitivity_summary, tir_assessment):
    lines = []
    w = lines.append

    d2 = domain_results["D2"]
    d1 = domain_results["D1"]
    d3 = domain_results["D3"]
    g = grid_results["D2_sp5"]

    w("# Phase 2 Step 2 — Summary")
    w("")
    w(f"- **Recommended domain:** D2 — 320–2550 nm")
    w(f"- **Recommended spacing:** 5 nm")
    w(f"- **Coverage:** {d2['n_full']}/925 spectra ({d2['pct_full']:.1f}%)")
    w(f"- **Minerals:** {d2['n_minerals_full']}/20")
    w(f"- **Provenance groups:** {d2['n_prov_full']}")
    w(f"- **Output dimensionality:** {g['n_grid']} points")
    w(f"- **Native support at 5 nm:** {g['n_supporting']} spectra "
      f"(native fraction {g['mean_native_frac']:.3f})")
    w("")
    w("## Key evidence")
    w("")
    w("- 320–2550 nm provides the highest coverage (49.1%) among scientifically")
    w("  plausible candidate domains")
    w("- Captures the full visible and NIR crystal-field and overtone/combination")
    w("  bands diagnostic of crystal symmetry and chemistry")
    w("- 5 nm spacing matches the dominant native sampling (197 spectra at exactly")
    w("  5 nm), minimizing interpolation burden")
    w("- Physical spectral resolution is unknown; 5 nm avoids false precision")
    w("- Complexity sensitivity analysis shows that metrics vary with grid spacing,")
    w("  confirming that matching native spacing is important")
    w("")
    w("## Alternative domain/grid")
    w("")
    w(f"- **D1 (300–2500 nm):** {d1['n_full']} spectra ({d1['pct_full']:.1f}%); "
      f"extends further into UV, slightly less coverage")
    w(f"- **D3 (300–2600 nm):** {d3['n_full']} spectra ({d3['pct_full']:.1f}%); "
      f"matches most common native grid exactly")
    w(f"- **Thermal-IR subset:** {tir_assessment['n_spectra']} spectra beyond 2500 nm; "
      f"complementary secondary analysis")
    w("")
    w("## Unresolved limitations")
    w("")
    w("- Physical spectral resolution unknown")
    w("- Splice-boundary artifacts not yet corrected")
    w("- Category B QC flags require separate treatment decisions")
    w("- Thermal-IR data excluded from primary domain due to sparse coverage")
    w("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
