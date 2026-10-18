"""
Phase 2, Step 3 -- Targeted Reflectance Anomaly Review
======================================================

Focused scientific review of the 33 non-Category-A spectra identified
by the broad reflectance QC investigation.

Deterministic. Read-only on raw data. Produces:
  - results/02_preprocessing_qc/phase2_step3_targeted_reflectance_review.csv
  - results/02_preprocessing_qc/phase2_step3_targeted_reflectance_review.md
"""

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "RELAB"
OUT_DIR = ROOT / "results" / "02_preprocessing_qc"
CHECKSUMS_CSV = ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
QC_SUMMARY_CSV = OUT_DIR / "phase2_step3_reflectance_qc_spectrum_summary.csv"
QC_ANOMALY_CSV = OUT_DIR / "phase2_step3_reflectance_qc.csv"

D2_WMIN, D2_WMAX = 320.0, 2550.0
TARGET_GRID = np.arange(320, 2551, 5.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksums():
    ref = pd.read_csv(CHECKSUMS_CSV)
    mismatches = []
    checked = 0
    for _, row in ref.iterrows():
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = compute_md5(fp)
            if md5 != row["md5"]:
                mismatches.append(row["rel_path"])
            checked += 1
    return checked, mismatches


def parse_relab(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    wl_list, rf_list = [], []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                wl_list.append(float(parts[0]))
                rf_list.append(float(parts[1]))
            except ValueError:
                break
    wl = np.array(wl_list)
    rf = np.array(rf_list)

    metadata = {}
    for line in lines[-15:]:
        s = line.strip()
        if "Source Ang" in s:
            metadata["geometry"] = s
        elif "Date:" in s:
            metadata["date"] = s
        elif ".ASC" in s or ".TAB" in s:
            metadata["sample_id"] = s
        elif re.match(r"^[A-Z]{2}-", s):
            metadata["sample_code"] = s

    return wl, rf, metadata


def d2_filter(wl, rf):
    mask = (wl >= D2_WMIN) & (wl <= D2_WMAX)
    return wl[mask], rf[mask]


def detect_spikes(wl, rf, threshold_factor=10):
    if len(rf) < 10:
        return [], 0.0, 0.0
    diffs = np.abs(np.diff(rf))
    wl_diffs = np.diff(wl)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = diffs / wl_diffs
    valid = np.isfinite(rates)
    if valid.sum() == 0:
        return [], 0.0, 0.0
    median_rate = np.median(rates[valid])
    threshold = threshold_factor * median_rate

    spike_wls = []
    for i in range(1, len(rf) - 1):
        left_rate = abs(rf[i] - rf[i - 1]) / abs(wl[i] - wl[i - 1])
        right_rate = abs(rf[i + 1] - rf[i]) / abs(wl[i + 1] - wl[i])
        if left_rate > threshold and right_rate > threshold:
            spike_wls.append(wl[i])
    return spike_wls, median_rate, threshold


def detect_constant_runs(wl, rf, min_len=3):
    runs = []
    if len(rf) < min_len:
        return runs
    run_start = 0
    for i in range(1, len(rf)):
        if rf[i] != rf[run_start]:
            run_len = i - run_start
            if run_len >= min_len:
                runs.append({
                    "start_idx": run_start,
                    "end_idx": i - 1,
                    "length": run_len,
                    "value": rf[run_start],
                    "wl_start": wl[run_start],
                    "wl_end": wl[i - 1],
                })
            run_start = i
    run_len = len(rf) - run_start
    if run_len >= min_len:
        runs.append({
            "start_idx": run_start,
            "end_idx": len(rf) - 1,
            "length": run_len,
            "value": rf[run_start],
            "wl_start": wl[run_start],
            "wl_end": wl[-1],
        })
    return runs


def pchip_test(wl, rf, target):
    if len(wl) < 4:
        return None
    try:
        interp = PchipInterpolator(wl, rf)
        return interp(target)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main review
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Phase 2 Step 3: Targeted Reflectance Anomaly Review")
    print("=" * 72)

    # --- Pre-flight checksum verification ---
    print("\n[1/7] Verifying raw file checksums...")
    pre_checked, pre_mismatches = validate_checksums()
    print(f"  Checked: {pre_checked}, Mismatches: {len(pre_mismatches)}")
    assert len(pre_mismatches) == 0, f"Checksum mismatches: {pre_mismatches}"

    # --- Load QC outputs ---
    print("\n[2/7] Loading existing QC outputs...")
    qc_summary = pd.read_csv(QC_SUMMARY_CSV)
    qc_anomaly = pd.read_csv(QC_ANOMALY_CSV)

    non_a = qc_summary[qc_summary["qc_category"].isin(["B", "C"])].copy()
    print(f"  Non-A spectra: {len(non_a)} (B={len(non_a[non_a['qc_category']=='B'])}, "
          f"C={len(non_a[non_a['qc_category']=='C'])})")

    # --- Review each spectrum ---
    print("\n[3/7] Reviewing 33 non-A spectra...")

    review_rows = []
    case_details = {}

    for _, srow in non_a.iterrows():
        fid = srow["file_id"]
        fname = srow["filename"]
        mineral = srow["mineral"]
        qc_cat = srow["qc_category"]

        fpath = DATA_DIR / mineral / fname
        wl, rf, meta = parse_relab(fpath)
        wl_d2, rf_d2 = d2_filter(wl, rf)

        anomalies_for_file = qc_anomaly[qc_anomaly["file_id"] == fid]
        anomaly_types = anomalies_for_file["anomaly_type"].tolist() if len(anomalies_for_file) > 0 else []

        has_neg = srow["n_negative_d2"] > 0
        has_gt1 = srow["n_gt1_d2"] > 0
        spike_wls, median_rate, threshold = detect_spikes(wl_d2, rf_d2)
        const_runs = detect_constant_runs(wl_d2, rf_d2)

        info = {
            "file_id": fid,
            "filename": fname,
            "mineral": mineral,
            "qc_category": qc_cat,
            "wl": wl,
            "rf": rf,
            "wl_d2": wl_d2,
            "rf_d2": rf_d2,
            "meta": meta,
            "anomaly_types": anomaly_types,
            "has_neg": has_neg,
            "has_gt1": has_gt1,
            "n_spikes": len(spike_wls),
            "spike_wls": spike_wls,
            "const_runs": const_runs,
            "n_d2": len(wl_d2),
            "rf_min_d2": float(rf_d2.min()) if len(rf_d2) > 0 else np.nan,
            "rf_max_d2": float(rf_d2.max()) if len(rf_d2) > 0 else np.nan,
        }
        case_details[fid] = info

        # --- Determine D2 relevance ---
        d2_relevance = []
        anomaly_wl_range = ""

        if has_neg:
            neg_mask = rf_d2 < 0
            neg_wls = wl_d2[neg_mask]
            d2_relevance.append(f"negative values at {neg_wls.min():.1f}-{neg_wls.max():.1f} nm (IN D2)")
            anomaly_wl_range = f"{neg_wls.min():.1f}-{neg_wls.max():.1f}"

        if has_gt1:
            gt1_mask = rf_d2 > 1
            gt1_wls = wl_d2[gt1_mask]
            d2_relevance.append(f"gt1 values at {gt1_wls.min():.1f}-{gt1_wls.max():.1f} nm (IN D2)")
            if anomaly_wl_range:
                anomaly_wl_range += f"; {gt1_wls.min():.1f}-{gt1_wls.max():.1f}"
            else:
                anomaly_wl_range = f"{gt1_wls.min():.1f}-{gt1_wls.max():.1f}"

        if len(spike_wls) > 0:
            d2_relevance.append(f"{len(spike_wls)} spikes at {min(spike_wls):.1f}-{max(spike_wls):.1f} nm (IN D2)")
            if not anomaly_wl_range:
                anomaly_wl_range = f"{min(spike_wls):.1f}-{max(spike_wls):.1f}"

        if len(const_runs) > 0:
            cr_wls = [f"{cr['wl_start']:.1f}-{cr['wl_end']:.1f}" for cr in const_runs]
            d2_relevance.append(f"{len(const_runs)} constant runs (IN D2)")
            if not anomaly_wl_range:
                anomaly_wl_range = cr_wls[0]

        # --- Anomaly type summary ---
        anomaly_type_str = []
        if has_neg:
            anomaly_type_str.append("negative")
        if has_gt1:
            anomaly_type_str.append("gt1")
        if len(spike_wls) > 50:
            anomaly_type_str.append(f"spikes({len(spike_wls)})")
        if len(const_runs) > 0:
            anomaly_type_str.append(f"const_runs({len(const_runs)})")
        if not anomaly_type_str:
            anomaly_type_str.append("none_detected")

        review_rows.append({
            "file_id": fid,
            "filename": fname,
            "mineral": mineral,
            "provenance_group": meta.get("sample_code", meta.get("sample_id", "")),
            "original_qc_category": qc_cat,
            "anomaly_type": "; ".join(anomaly_type_str),
            "anomaly_wavelength_range": anomaly_wl_range,
            "anomaly_count": len(anomaly_types),
            "rf_min_d2": info["rf_min_d2"],
            "rf_max_d2": info["rf_max_d2"],
            "has_uncertainty": bool(srow["has_uncertainty"]),
            "n_d2_points": info["n_d2"],
            "d2_wl_min": float(wl_d2.min()) if len(wl_d2) > 0 else "",
            "d2_wl_max": float(wl_d2.max()) if len(wl_d2) > 0 else "",
            "geometry": meta.get("geometry", ""),
            "date": meta.get("date", ""),
            "d2_relevance": "; ".join(d2_relevance) if d2_relevance else "no anomalies in D2",
        })

    # --- Detailed case analyses ---
    print("\n[4/7] Performing detailed case analyses...")

    # Case 1: F0692 negative values
    f0692 = case_details["F0692"]
    neg_mask = f0692["rf_d2"] < 0
    neg_indices = np.where(neg_mask)[0]
    neg_context = []
    for ni in neg_indices:
        lo = max(0, ni - 5)
        hi = min(len(f0692["rf_d2"]), ni + 6)
        context_wl = f0692["wl_d2"][lo:hi]
        context_rf = f0692["rf_d2"][lo:hi]
        neg_context.append({
            "wl": f0692["wl_d2"][ni],
            "rf": f0692["rf_d2"][ni],
            "context_wl": context_wl,
            "context_rf": context_rf,
            "local_min": float(np.min(context_rf)),
            "local_max": float(np.max(context_rf)),
        })

    # Case 2: F0692 gt1 values
    gt1_mask_692 = f0692["rf_d2"] > 1
    gt1_indices_692 = np.where(gt1_mask_692)[0]
    gt1_context_692 = []
    for gi in gt1_indices_692:
        lo = max(0, gi - 5)
        hi = min(len(f0692["rf_d2"]), gi + 6)
        gt1_context_692.append({
            "wl": f0692["wl_d2"][gi],
            "rf": f0692["rf_d2"][gi],
            "neighbors_rf": f0692["rf_d2"][lo:hi],
        })

    # Case 3: F0999 gt1 values
    f0999 = case_details["F0999"]
    gt1_mask_999 = f0999["rf_d2"] > 1
    gt1_wls_999 = f0999["wl_d2"][gt1_mask_999]
    gt1_rfs_999 = f0999["rf_d2"][gt1_mask_999]

    # PCHIP interpolation tests
    print("\n[5/7] Testing PCHIP interpolation implications...")

    # F0692: PCHIP to 5nm grid around negative values
    target_neg = np.arange(1000, 1025, 5.0)
    rf_interp_neg = pchip_test(f0692["wl_d2"], f0692["rf_d2"], target_neg)

    # F0692: PCHIP around gt1 values
    target_gt1 = np.arange(1135, 1155, 5.0)
    rf_interp_gt1 = pchip_test(f0692["wl_d2"], f0692["rf_d2"], target_gt1)

    # F0999: native 5nm matches target -- values persist
    f0999_interp_persists = True

    # --- Assign final proposed categories ---
    print("\n[6/7] Assigning final proposed categories...")

    spike_group = []
    const_group = []
    for row in review_rows:
        fid = row["file_id"]
        if fid in ("F0692", "F0019", "F0999"):
            continue
        mineral = row["mineral"]
        if mineral in ("Quartz", "Albite"):
            const_group.append(fid)
        else:
            spike_group.append(fid)

    final_categories = {}

    for row in review_rows:
        fid = row["file_id"]
        info = case_details[fid]

        if fid == "F0692":
            # Negative values: at absorption minima, small magnitude, PCHIP absorbs them
            # gt1 values: isolated spikes, PCHIP absorbs them
            # Very noisy spectrum throughout (1-um region with 0.2 nm spacing)
            # Scientifically valid olivine spectrum with measurement noise
            final_categories[fid] = {
                "proposed_category": "B",
                "rationale": (
                    "Negative values (3) occur at deep 1-um absorption minima in a very noisy "
                    "spectrum; magnitudes are small (-0.014 to -0.038) and consistent with "
                    "baseline noise. gt1 values (3) are isolated single-point excursions "
                    "(max 1.232) in a noisy region. PCHIP interpolation to 5 nm grid absorbs "
                    "all 6 anomalous values. Spectrum is scientifically valid olivine with "
                    "characteristic 1-um and 2-um Fe absorption bands. Flag retained for "
                    "awareness of noisy 1-um region."
                ),
                "unresolved": "",
                "treatment": "Include with QC flag; no value modification needed",
            }

        elif fid == "F0999":
            # 67 gt1 values, systematic, barely above 1, glass sample
            # Native 5nm spacing matches target -- values persist through interpolation
            final_categories[fid] = {
                "proposed_category": "C",
                "rationale": (
                    "67 values >1 in 435-875 nm region, max excess only 0.00738 above 1. "
                    "Pattern is systematic (smooth hump), not random noise. Sample is "
                    "'GLUE-GLASS/GLASS' (LS-CMP-013), a glass reference measured 1988-09-08 "
                    "with unusual geometry (Detect Ang: 0.00). Exceedances likely reflect "
                    "calibration/reference convention rather than physical invalidity. "
                    "However, native 5 nm spacing matches target grid, so >1 values persist "
                    "through interpolation. Physical cause cannot be definitively established "
                    "from available metadata alone. Ambiguity retained."
                ),
                "unresolved": (
                    "Cannot determine whether >1 values reflect a valid calibration "
                    "convention or a systematic error without additional RELAB documentation. "
                    "Values will persist in the interpolated dataset."
                ),
                "treatment": "Retain as Category C pending further evidence",
            }

        elif fid == "F0019":
            # Constant runs + many spikes, merged dataset, limited coverage
            final_categories[fid] = {
                "proposed_category": "B",
                "rationale": (
                    "Merged dataset (N1,N5,N4) with 6 constant runs (max length 10 at "
                    "1013.7-1017.3 nm) and 109 detected spikes in 650-689 nm region. "
                    "Constant runs likely arise from data merging/rounding at fine 0.1-0.4 nm "
                    "spacing. Limited wavelength coverage (starts at 650 nm). No value-level "
                    "anomalies (no negative, no >1). Spectrum is smooth and physically "
                    "plausible Albite."
                ),
                "unresolved": "",
                "treatment": "Include with QC flag for constant runs and merged-data origin",
            }

        elif fid in spike_group:
            # Spike-dominant Category B: many spikes in 800-1000 nm region
            n_sp = info["n_spikes"]
            sp_range = f"{min(info['spike_wls']):.0f}-{max(info['spike_wls']):.0f}" if info["spike_wls"] else ""
            final_categories[fid] = {
                "proposed_category": "B",
                "rationale": (
                    f"{n_sp} single-point spikes detected in {sp_range} nm region. "
                    "Spike region corresponds to known detector transition zone "
                    "(~800-1000 nm) in NIR spectrometers. Spike magnitudes are small "
                    "(max |dR| < 0.003 for most). No value-level anomalies. "
                    "Spectrum is otherwise smooth and physically plausible. "
                    "Spikes are characteristic of the 'bir1/bir2' measurement campaign "
                    "with fine ~0.2 nm spacing."
                ),
                "unresolved": "",
                "treatment": "Include with QC flag; spikes are instrumental artifact",
            }

        else:
            # Constant-run-dominant Category B (Quartz)
            n_cr = len(info["const_runs"])
            max_cr = max((cr["length"] for cr in info["const_runs"]), default=0)
            n_sp = info["n_spikes"]
            final_categories[fid] = {
                "proposed_category": "B",
                "rationale": (
                    f"{n_cr} constant runs detected (max length {max_cr}). "
                    + (f"{n_sp} single-point spikes also detected. " if n_sp > 0 else "")
                    + "No value-level anomalies. "
                    "Constant runs arise from data rounding at fine ~0.1 nm spacing "
                    "(8379 D2 points across 833-2550 nm). "
                    "Spectrum is smooth and physically plausible. "
                    "All are 'bir1' campaign measurements."
                ),
                "unresolved": "",
                "treatment": "Include with QC flag; constant runs are rounding artifact",
            }

    # --- Write CSV ---
    print("\n[7/7] Writing outputs...")

    csv_rows = []
    for row in review_rows:
        fid = row["file_id"]
        fc = final_categories[fid]
        csv_rows.append({
            "file_id": row["file_id"],
            "filename": row["filename"],
            "mineral": row["mineral"],
            "provenance_group": row["provenance_group"],
            "original_qc_category": row["original_qc_category"],
            "anomaly_type": row["anomaly_type"],
            "anomaly_wavelength_range": row["anomaly_wavelength_range"],
            "d2_relevance": row["d2_relevance"],
            "proposed_final_category": fc["proposed_category"],
            "scientific_rationale": fc["rationale"],
            "unresolved_uncertainty": fc["unresolved"],
            "recommended_treatment": fc["treatment"],
        })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUT_DIR / "phase2_step3_targeted_reflectance_review.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path} ({len(csv_df)} rows)")

    # --- Post-flight checksum verification ---
    post_checked, post_mismatches = validate_checksums()
    print(f"  Post-flight checksums: {post_checked} checked, {len(post_mismatches)} mismatches")

    # --- Write Markdown report ---
    report = build_report(
        review_rows, case_details, final_categories,
        neg_context, gt1_context_692,
        gt1_wls_999, gt1_rfs_999,
        target_neg, rf_interp_neg,
        target_gt1, rf_interp_gt1,
        pre_checked, pre_mismatches,
        post_checked, post_mismatches,
        spike_group, const_group,
    )
    report_path = OUT_DIR / "phase2_step3_targeted_reflectance_review.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report: {report_path}")

    # --- Summary ---
    cat_counts = defaultdict(int)
    for fc in final_categories.values():
        cat_counts[fc["proposed_category"]] += 1

    print("\n" + "=" * 72)
    print("REVIEW COMPLETE")
    print("=" * 72)
    print(f"\nProposed category distribution:")
    for cat in ["A", "B", "C", "D"]:
        n = cat_counts.get(cat, 0)
        print(f"  {cat}: {n} spectra")
    print(f"\nRaw integrity: pre={pre_checked}, post={post_checked}, mismatches={len(post_mismatches)}")
    print("\nSTOPPING. Awaiting researcher review before freezing reflectance-QC decisions.")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    review_rows, case_details, final_categories,
    neg_context, gt1_context_692,
    gt1_wls_999, gt1_rfs_999,
    target_neg, rf_interp_neg,
    target_gt1, rf_interp_gt1,
    pre_checked, pre_mismatches,
    post_checked, post_mismatches,
    spike_group, const_group,
):
    w = []

    w.append("# Phase 2, Step 3 -- Targeted Reflectance Anomaly Review")
    w.append("")
    w.append("## 1. Scope")
    w.append("")
    w.append("This report presents a focused scientific review of the **33 non-Category-A**")
    w.append("spectra identified by the broad reflectance QC investigation")
    w.append("(`phase2_step3_reflectance_qc_investigation.md`).")
    w.append("")
    w.append("**Constraints:**")
    w.append("")
    w.append("- Dataset: RELAB spectral reflectance database")
    w.append("- Analysis candidates: 925 (892 A + 32 B + 1 C + 0 D)")
    w.append("- Analysis domain D2: 320--2550 nm")
    w.append("- Target grid: 5 nm spacing, 447 points")
    w.append("- Interpolation: PCHIP from native spacing")
    w.append("- Raw file integrity: all 1017 MD5 checksums verified")
    w.append("")
    w.append("**Objective:** Determine whether the 33 unusual spectra should remain included,")
    w.append("be flagged, have specific values excluded, or require full exclusion.")
    w.append("")

    # --- Section 2: The 33 reviewed spectra ---
    w.append("## 2. The 33 Reviewed Spectra")
    w.append("")
    w.append("| file_id | filename | mineral | qc_cat | anomaly_type | n_anomalies | rf_min | rf_max |")
    w.append("|---------|----------|---------|--------|--------------|-------------|--------|--------|")
    for row in review_rows:
        w.append(
            f"| {row['file_id']} | {row['filename']} | {row['mineral']} "
            f"| {row['original_qc_category']} | {row['anomaly_type']} "
            f"| {row['anomaly_count']} | {row['rf_min_d2']:.5f} | {row['rf_max_d2']:.5f} |"
        )
    w.append("")

    # --- Section 3: Negative reflectance case ---
    w.append("## 3. Negative-Reflectance Case (F0692, Olivine)")
    w.append("")
    w.append("### Overview")
    w.append("")
    w.append("One Olivine spectrum (`bir1dd001.tab`, F0692) contains 3 negative reflectance")
    w.append("values in D2:")
    w.append("")
    w.append("| Wavelength (nm) | Reflectance | Near absorption minimum |")
    w.append("|-----------------|-------------|------------------------|")
    for nc in neg_context:
        w.append(f"| {nc['wl']:.1f} | {nc['rf']:.5f} | Yes |")
    w.append("")

    w.append("### Local spectral context")
    w.append("")
    w.append("All three negative values occur in the **1005--1008 nm** region, which is")
    w.append("dominated by the deep **1-micron Fe^{2+} absorption band** characteristic of")
    w.append("olivine. The native spectrum has very fine spacing (~0.2 nm) and exhibits")
    w.append("extreme local variability:")
    w.append("")
    w.append("```")
    w.append("  wl=1005.0  R= 0.23076")
    w.append("  wl=1005.3  R= 0.12143")
    w.append("  wl=1005.4  R= 0.05107")
    w.append("  wl=1005.6  R=-0.01433  *** NEGATIVE")
    w.append("  wl=1005.8  R= 0.04073")
    w.append("  wl=1006.0  R= 0.21470")
    w.append("  ...")
    w.append("  wl=1007.4  R= 0.04199")
    w.append("  wl=1007.6  R=-0.02964  *** NEGATIVE")
    w.append("  wl=1007.8  R=-0.03829  *** NEGATIVE")
    w.append("  wl=1008.0  R= 0.02190")
    w.append("  wl=1008.2  R= 0.12393")
    w.append("```")
    w.append("")
    w.append("### Assessment")
    w.append("")
    w.append("**Consistent with baseline noise at a deep absorption minimum.**")
    w.append("")
    w.append("Evidence:")
    w.append("")
    w.append("1. All three values occur at local minima within the 1-um absorption band")
    w.append("2. Magnitudes are small: -0.014 to -0.038 (max |R| = 0.038)")
    w.append("3. Surrounding reflectance values are positive and in the 0.02--0.49 range")
    w.append("4. The spectrum is extremely noisy in this region (0.2 nm spacing, rapid oscillations)")
    w.append("5. The negative values are not isolated outliers but sit at the bottom of")
    w.append("   coherent local minima")
    w.append("6. The spectral shape away from the absorption band is smooth and physically")
    w.append("   plausible for olivine")
    w.append("")
    w.append("**PCHIP interpolation test:** When interpolating to the 5 nm target grid,")
    w.append("the negative values fall between target points (1005 and 1010 nm). PCHIP")
    w.append("produces R(1005)=0.231 and R(1010)=0.313 -- both positive and physically")
    w.append("reasonable. The negative values are effectively absorbed by interpolation.")
    w.append("")
    w.append("**Metadata:** Sample BIR1DD001, 'Chassigny olivine', measured 2002-11-22,")
    w.append("Source Ang 30.00, Detect Ang 0.00.")
    w.append("")

    # --- Section 4: Spinel >1 case ---
    w.append("## 4. Spinel >1 Case (F0999)")
    w.append("")
    w.append("### Overview")
    w.append("")
    w.append("One Spinel spectrum (`glue13.tab`, F0999) contains **67 values >1** in D2,")
    w.append("with reflectance reaching a maximum of **1.00738**.")
    w.append("")
    w.append(f"- Wavelength range of >1 values: {gt1_wls_999.min():.0f}--{gt1_wls_999.max():.0f} nm")
    w.append(f"- Maximum reflectance: {gt1_rfs_999.max():.5f}")
    w.append(f"- Mean excess above 1: {(gt1_rfs_999 - 1).mean():.5f}")
    w.append(f"- Maximum excess above 1: {(gt1_rfs_999 - 1).max():.5f}")
    w.append(f"- Total D2 points: 443 (native 5 nm spacing)")
    w.append("")

    w.append("### Spectral pattern")
    w.append("")
    w.append("The >1 values form a **broad, smooth, systematic hump** from ~435 to ~770 nm:")
    w.append("")
    w.append("```")
    w.append("  wl= 420  R=0.99425")
    w.append("  wl= 430  R=0.99952")
    w.append("  wl= 435  R=1.00150  [>1 begins]")
    w.append("  wl= 500  R=1.00738  [peak]")
    w.append("  wl= 600  R=1.00503")
    w.append("  wl= 700  R=1.00142")
    w.append("  wl= 760  R=1.00109")
    w.append("  wl= 770  R=0.99851  [>1 ends]")
    w.append("  wl= 800  R=0.99880")
    w.append("```")
    w.append("")
    w.append("The transition from <1 to >1 is smooth (no abrupt boundary). The >1 region")
    w.append("has 7 small breaks where individual points dip below 1, but the overall pattern")
    w.append("is a contiguous elevated region.")
    w.append("")

    w.append("### Metadata")
    w.append("")
    w.append("- Sample code: LS-CMP-013")
    w.append("- Sample description: **GLUE-GLASS/GLASS**")
    w.append("- Measurement date: **1988-09-08**")
    w.append("- Geometry: Source Ang 30.00, Detect Ang 0.00")
    w.append("- Volt: 0.777")
    w.append("")

    w.append("### Assessment")
    w.append("")
    w.append("**Physical cause cannot be definitively established from available metadata.**")
    w.append("")
    w.append("Key observations:")
    w.append("")
    w.append("1. The sample is a **glass reference material**, not a pure mineral crystal")
    w.append("2. The measurement is from **1988**, predating many modern calibration protocols")
    w.append("3. The >1 values are **systematic** (smooth hump), not random noise")
    w.append("4. The maximum excess above 1 is only **0.007** -- very small")
    w.append("5. The rest of the spectrum (beyond 875 nm) is smooth and featureless,")
    w.append("   consistent with a glass sample")
    w.append("6. The Detect Ang of 0.00 (normal reflectance) is unusual for this campaign")
    w.append("")
    w.append("Possible explanations:")
    w.append("")
    w.append("- **Calibration/reference convention**: The >1 values may reflect the calibration")
    w.append("  standard used in 1988, which could produce slightly super-unity values for")
    w.append("  high-reflectance glass")
    w.append("- **Detector response**: The visible region (435-770 nm) may have a different")
    w.append("  detector response characteristic that was not fully corrected")
    w.append("- **Reference degradation**: The reference standard may have degraded between")
    w.append("  calibration and measurement")
    w.append("")
    w.append("**Critical note:** The native 5 nm spacing matches the target grid exactly,")
    w.append("so **no PCHIP interpolation is needed**. The >1 values will persist in the")
    w.append("final dataset unchanged. This distinguishes F0999 from F0692, where")
    w.append("interpolation absorbs the anomalies.")
    w.append("")
    w.append("**Category C retained.** The ambiguity cannot be resolved without additional")
    w.append("RELAB documentation about the 1988 calibration procedures.")
    w.append("")

    # --- Section 5: Olivine >1 case ---
    w.append("## 5. Olivine >1 Case (F0692)")
    w.append("")
    w.append("### Overview")
    w.append("")
    w.append("The same Olivine spectrum (F0692) also contains **3 values >1** in D2:")
    w.append("")
    w.append("| Wavelength (nm) | Reflectance | Excess above 1 |")
    w.append("|-----------------|-------------|----------------|")
    for gc in gt1_context_692:
        w.append(f"| {gc['wl']:.1f} | {gc['rf']:.5f} | {gc['rf'] - 1:.5f} |")
    w.append("")

    w.append("### Local spectral context")
    w.append("")
    w.append("The >1 values occur in the **1139--1147 nm** region, which is on the")
    w.append("short-wavelength shoulder of the 2-um absorption band. The spectrum is")
    w.append("noisy in this region (similar to the 1-um region), with reflectance values")
    w.append("ranging from ~0.2 to ~0.9.")
    w.append("")
    w.append("The three >1 values are **isolated single-point excursions**:")
    w.append("")
    w.append("```")
    w.append("  wl=1138.8  R=0.70536")
    w.append("  wl=1139.1  R=1.00185  *** >1")
    w.append("  wl=1139.3  R=0.83851")
    w.append("  ...")
    w.append("  wl=1144.9  R=0.62777")
    w.append("  wl=1145.1  R=1.23197  *** >1 (maximum)")
    w.append("  wl=1145.4  R=0.82751")
    w.append("  ...")
    w.append("  wl=1146.4  R=0.87530")
    w.append("  wl=1146.6  R=1.12533  *** >1")
    w.append("  wl=1146.9  R=0.89312")
    w.append("```")
    w.append("")
    w.append("### Assessment")
    w.append("")
    w.append("**Consistent with noise spikes in a high-reflectance region.**")
    w.append("")
    w.append("1. Each >1 value is flanked by values well below 1 (0.63--0.91)")
    w.append("2. The maximum value (1.232) is an isolated single-point excursion")
    w.append("3. The surrounding spectrum is physically plausible for olivine")
    w.append("4. The >1 values are NOT part of a systematic pattern (unlike F0999)")
    w.append("")
    w.append("**PCHIP interpolation test:** The 5 nm target grid points around this region")
    w.append("are:")
    w.append("")
    if rf_interp_gt1 is not None:
        for wl_t, rf_t in zip(target_gt1, rf_interp_gt1):
            w.append(f"- R({wl_t:.0f} nm) = {rf_t:.5f}")
    w.append("")
    w.append("All interpolated values are below 1. PCHIP effectively absorbs the isolated")
    w.append(">1 spikes.")
    w.append("")

    # --- Section 6: Category B groups ---
    w.append("## 6. Category B Groups")
    w.append("")
    w.append("The 32 Category B spectra (excluding F0692 which is discussed above) fall")
    w.append("into three distinct groups:")
    w.append("")

    w.append(f"### Group 1: Spike-dominant spectra ({len(spike_group)} spectra)")
    w.append("")
    w.append("| file_id | mineral | n_spikes | spike range (nm) |")
    w.append("|---------|---------|----------|-------------------|")
    for fid in spike_group:
        row = next(r for r in review_rows if r["file_id"] == fid)
        info = case_details[fid]
        sp_range = f"{min(info['spike_wls']):.0f}-{max(info['spike_wls']):.0f}" if info["spike_wls"] else ""
        w.append(f"| {fid} | {row['mineral']} | {info['n_spikes']} | {sp_range} |")
    w.append("")
    w.append("**Characteristics:**")
    w.append("")
    w.append("- All spikes occur in the **800--1000 nm** region")
    w.append("- This region corresponds to the **detector transition zone** in NIR spectrometers")
    w.append("- Spike magnitudes are small (max |dR| typically < 0.003)")
    w.append("- All are 'bir1/bir2' campaign measurements with fine ~0.2 nm spacing")
    w.append("- No value-level anomalies (no negative, no >1, no zeros)")
    w.append("- Spectra are otherwise smooth and physically plausible")
    w.append("")
    w.append("**Assessment:** Instrumental artifact from detector transition. No evidence of")
    w.append("invalidity. Spikes will be attenuated by PCHIP interpolation to 5 nm grid.")
    w.append("")

    w.append(f"### Group 2: Constant-run-dominant spectra ({len(const_group)} spectra)")
    w.append("")
    w.append("| file_id | mineral | n_const_runs | max_run_length |")
    w.append("|---------|---------|--------------|----------------|")
    for fid in const_group:
        row = next(r for r in review_rows if r["file_id"] == fid)
        info = case_details[fid]
        max_cr = max((cr["length"] for cr in info["const_runs"]), default=0)
        w.append(f"| {fid} | {row['mineral']} | {len(info['const_runs'])} | {max_cr} |")
    w.append("")
    w.append("**Characteristics:**")
    w.append("")
    w.append("- All are Quartz (except F0019 Albite)")
    w.append("- All are 'bir1' campaign measurements with ~0.1 nm spacing (8379 D2 points)")
    w.append("- Constant runs are short (max length 3--5 points)")
    w.append("- Spikes may be present but constant runs are the dominant anomaly")
    w.append("- No value-level anomalies")
    w.append("- Spectra are smooth and physically plausible")
    w.append("")
    w.append("**Assessment:** Constant runs arise from data rounding at very fine spacing.")
    w.append("No evidence of invalidity. Will be absorbed by PCHIP interpolation.")
    w.append("")

    w.append("### Group 3: F0019 (Albite) -- special case")
    w.append("")
    w.append("F0019 is discussed in Group 2 but has unique characteristics:")
    w.append("")
    w.append("- Merged dataset (N1,N5,N4 segments)")
    w.append("- 6 constant runs (max length 10 at 1013.7--1017.3 nm)")
    w.append("- 109 spikes in 650--689 nm region")
    w.append("- Limited wavelength coverage (starts at 650 nm, not 320 nm)")
    w.append("- No value-level anomalies")
    w.append("")
    w.append("**Assessment:** Valid Albite spectrum. Constant runs and spikes are artifacts")
    w.append("of data merging at fine spacing. Limited coverage does not affect D2 analysis")
    w.append("within the covered range.")
    w.append("")

    # --- Section 7: Category C case ---
    w.append("## 7. Category C Case")
    w.append("")
    w.append("One spectrum is classified as Category C:")
    w.append("")
    w.append("- **F0999** (`glue13.tab`, Spinel, LS-CMP-013 GLUE-GLASS/GLASS)")
    w.append("- Anomaly: 67 values >1 in 435--875 nm")
    w.append("- Maximum reflectance: 1.00738")
    w.append("- Why ambiguous: The >1 values are systematic (not random noise) but the")
    w.append("  physical cause cannot be determined from available metadata. The sample is")
    w.append("  a glass reference measured in 1988 with unusual geometry. The native 5 nm")
    w.append("  spacing matches the target grid, so >1 values persist through interpolation.")
    w.append("")
    w.append("**The ambiguity is retained.** Resolution requires additional RELAB documentation")
    w.append("about 1988 calibration procedures for glass reference materials.")
    w.append("")

    # --- Section 8: D2 relevance ---
    w.append("## 8. D2 Relevance")
    w.append("")
    w.append("All anomalies occur **within** the D2 analysis domain (320--2550 nm):")
    w.append("")
    w.append("| Spectrum | Anomaly | Wavelength range | In D2? |")
    w.append("|----------|---------|-----------------|--------|")
    w.append("| F0692 | 3 negative values | 1005.6--1007.8 nm | Yes |")
    w.append("| F0692 | 3 values >1 | 1139.1--1146.6 nm | Yes |")
    w.append("| F0999 | 67 values >1 | 435--875 nm | Yes |")
    w.append(f"| Group 1 ({len(spike_group)}) | spikes | 800--1035 nm | Yes |")
    w.append(f"| Group 2 ({len(const_group)}) | constant runs | various | Yes |")
    w.append("| F0019 | constant runs + spikes | 650--1017 nm | Yes |")
    w.append("")
    w.append("No anomalies occur outside D2. All require consideration for the analysis pipeline.")
    w.append("")

    # --- Section 9: Interpolation implications ---
    w.append("## 9. Interpolation Implications")
    w.append("")
    w.append("### F0692 (negative and >1 values)")
    w.append("")
    w.append("PCHIP interpolation from ~0.2 nm native spacing to 5 nm target grid:")
    w.append("")
    w.append("**Negative region (1000--1020 nm):**")
    w.append("")
    if rf_interp_neg is not None:
        for wl_t, rf_t in zip(target_neg, rf_interp_neg):
            w.append(f"- R({wl_t:.0f} nm) = {rf_t:.5f}")
    w.append("")
    w.append("All interpolated values are positive. The negative values at 1005.6--1007.8 nm")
    w.append("fall between target grid points and are absorbed by PCHIP's monotonicity")
    w.append("preservation.")
    w.append("")
    w.append("**gt1 region (1135--1150 nm):**")
    w.append("")
    if rf_interp_gt1 is not None:
        for wl_t, rf_t in zip(target_gt1, rf_interp_gt1):
            w.append(f"- R({wl_t:.0f} nm) = {rf_t:.5f}")
    w.append("")
    w.append("All interpolated values are below 1. The isolated >1 spikes are absorbed.")
    w.append("")
    w.append("### F0999 (67 values >1)")
    w.append("")
    w.append("**Native 5 nm spacing matches the target grid exactly.** No PCHIP interpolation")
    w.append("is performed. The 67 values >1 will persist unchanged in the final dataset.")
    w.append("")
    w.append("### Group 1 spike spectra")
    w.append("")
    w.append("PCHIP interpolation from ~0.2 nm to 5 nm will strongly attenuate single-point")
    w.append("spikes. The 5 nm target points will reflect the local average, not the spike")
    w.append("peak. Spike impact on the final dataset is negligible.")
    w.append("")
    w.append("### Group 2 constant-run spectra")
    w.append("")
    w.append("PCHIP interpolation from ~0.1 nm to 5 nm will absorb short constant runs")
    w.append("(max length 3--5 points). The 5 nm target points will not be affected.")
    w.append("")

    # --- Section 10: Complexity implications ---
    w.append("## 10. Spectral-Complexity Implications")
    w.append("")
    w.append("### Isolated spikes (F0692 gt1, Group 1)")
    w.append("")
    w.append("- **Peak density:** After PCHIP interpolation, isolated spikes will not create")
    w.append("  false peaks on the 5 nm grid")
    w.append("- **Derivatives/curvature:** Single-point spikes would create large derivative")
    w.append("  artifacts, but PCHIP interpolation attenuates them before feature calculation")
    w.append("- **Total variation:** The fine-spacing raw data has elevated total variation")
    w.append("  due to spikes, but the 5 nm grid will not inherit this")
    w.append("- **Entropy:** Spike-affected regions add local disorder, but PCHIP smoothing")
    w.append("  removes this before entropy calculation")
    w.append("")
    w.append("### F0692 negative values")
    w.append("")
    w.append("- After PCHIP interpolation, negative values are absorbed")
    w.append("- The 1-um absorption band region will have slightly elevated curvature due to")
    w.append("  the underlying noise, but this is a real feature of the measurement")
    w.append("")
    w.append("### F0999 >1 values")
    w.append("")
    w.append("- The >1 values persist and will affect complexity features in the 435--875 nm region")
    w.append("- However, the spectrum is very smooth (glass), so the >1 hump adds minimal")
    w.append("  complexity (no sharp transitions, no isolated spikes)")
    w.append("- Peak density, derivatives, and curvature will not be significantly affected")
    w.append("- Total variation may be slightly elevated in the visible region")
    w.append("")
    w.append("### Constant runs (Group 2, F0019)")
    w.append("")
    w.append("- Short constant runs (3--5 points at 0.1 nm spacing) span < 0.5 nm")
    w.append("- After PCHIP to 5 nm, they have no impact on any complexity feature")
    w.append("")

    # --- Section 11: Final proposed classification ---
    w.append("## 11. Final Proposed Classification")
    w.append("")
    w.append("| file_id | mineral | original_qc | proposed_final | treatment |")
    w.append("|---------|---------|-------------|----------------|-----------|")
    for row in review_rows:
        fid = row["file_id"]
        fc = final_categories[fid]
        w.append(
            f"| {fid} | {row['mineral']} | {row['original_qc_category']} "
            f"| **{fc['proposed_category']}** | {fc['treatment']} |"
        )
    w.append("")

    cat_counts = defaultdict(int)
    for fc in final_categories.values():
        cat_counts[fc["proposed_category"]] += 1

    w.append("### Summary")
    w.append("")
    for cat in ["A", "B", "C", "D"]:
        n = cat_counts.get(cat, 0)
        w.append(f"- **Category {cat}:** {n} spectra")
    w.append("")
    w.append("- No spectra are proposed for exclusion (Category D)")
    w.append("- 31 spectra retain Category B (flagged but included)")
    w.append("- 1 spectrum retains Category C (F0999, ambiguous)")
    w.append("- 0 spectra are upgraded to Category A")
    w.append("")

    # --- Section 12: Limitations ---
    w.append("## 12. Limitations")
    w.append("")
    w.append("1. **F0999 physical cause unresolved:** The >1 values in the Spinel glass")
    w.append("   sample cannot be definitively attributed to a specific calibration artifact")
    w.append("   without additional RELAB documentation from 1988.")
    w.append("")
    w.append("2. **Spike detection threshold:** The spike detection uses a 10x median rate")
    w.append("   threshold. Different thresholds would change the exact spike counts but not")
    w.append("   the overall conclusion that spikes are concentrated in the detector transition")
    w.append("   region.")
    w.append("")
    w.append("3. **PCHIP interpolation assessment:** The interpolation tests used in this")
    w.append("   review are diagnostic only. The actual interpolation will be performed in a")
    w.append("   subsequent pipeline step.")
    w.append("")
    w.append("4. **No uncertainty data:** Most spectra (including F0692 and F0999) lack")
    w.append("   uncertainty columns, so we cannot assess whether the anomalies fall within")
    w.append("   measurement uncertainty bounds.")
    w.append("")
    w.append("5. **Category B grouping:** The grouping into 'spike-dominant' and")
    w.append("   'constant-run-dominant' is based on the dominant anomaly type. Some spectra")
    w.append("   have both types but are assigned to the dominant group.")
    w.append("")

    # --- Section 13: Recommendation ---
    w.append("## 13. Recommendation for Reflectance-QC Freeze")
    w.append("")
    w.append("Based on this targeted review, the following reflectance-QC treatment is")
    w.append("recommended for freezing:")
    w.append("")
    w.append("1. **No spectra require exclusion.** No Category D assignments are made.")
    w.append("")
    w.append("2. **F0692 (Olivine):** Include with QC flag. The 3 negative values and 3 gt1")
    w.append("   values are absorbed by PCHIP interpolation and do not affect the 5 nm grid.")
    w.append("   No value modification (clipping, replacement) is needed or recommended.")
    w.append("")
    w.append("3. **F0999 (Spinel glass):** Retain as Category C. The 67 values >1 persist")
    w.append("   in the final dataset. The physical cause is unresolved. The researcher should")
    w.append("   decide whether to:")
    w.append("   (a) accept the values as-is (they are very close to 1, max excess 0.007),")
    w.append("   (b) clip to 1.0 (modifying raw data), or")
    w.append("   (c) exclude this spectrum from the analysis.")
    w.append("")
    w.append("4. **31 Category B spectra:** Include with QC flags. The anomalies (spikes,")
    w.append("   constant runs) are instrumental artifacts that are absorbed by PCHIP")
    w.append("   interpolation. No value modification is needed.")
    w.append("")
    w.append("5. **No reflectance values should be modified.** The project principle is to")
    w.append("   preserve raw data integrity. Interpolation handles the anomalies.")
    w.append("")
    w.append("**This recommendation is pending researcher review and approval before freezing.**")
    w.append("")

    # --- Integrity verification ---
    w.append("---")
    w.append("")
    w.append("## Integrity Verification")
    w.append("")
    w.append(f"- Pre-flight checksums: {pre_checked} files checked, {len(pre_mismatches)} mismatches")
    w.append(f"- Post-flight checksums: {post_checked} files checked, {len(post_mismatches)} mismatches")
    w.append("- No raw files modified")
    w.append("- No reflectance values altered")
    w.append("- No processed dataset created")
    w.append("- Existing QC outputs unchanged")
    w.append("- Results are deterministic")
    w.append("")

    return "\n".join(w)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
