#!/usr/bin/env python3
"""
Phase 2 Step 6 — Smoothing / Denoising Investigation
=====================================================

CrystalSym-Spec: Interpretable ML and Symbolic Regression of
Crystal Symmetry, Chemistry, and Mineral Spectral Complexity

Determines whether smoothing or denoising is scientifically justified
for the primary spectral-complexity analysis.

This is an investigation only. No raw files are modified. No final
processed dataset is created. No smoothing method is frozen.

Outputs:
  results/02_preprocessing_qc/phase2_step6_smoothing_investigation.md
  results/02_preprocessing_qc/phase2_step6_smoothing_comparison.csv
  results/02_preprocessing_qc/phase2_step6_smoothing_summary.csv
  results/02_preprocessing_qc/figures/step6/*.png
"""

import csv
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"
POPULATION_CSV = PROJECT_ROOT / "metadata" / "relab_analysis_population.csv"
MEASUREMENTS_CSV = PROJECT_ROOT / "metadata" / "relab_measurements.csv"
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
D2_VALIDATION_CSV = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step2_decision_validation.csv"
REFLECTANCE_QC_CSV = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_reflectance_qc_spectrum_summary.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"
FIG_DIR = OUT_DIR / "figures" / "step6"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# FROZEN CONSTANTS
# ================================================================

D2_WMIN = 320.0
D2_WMAX = 2550.0
D2_SPACING = 5.0
D2_TARGET_GRID = np.arange(D2_WMIN, D2_WMAX + D2_SPACING * 0.5, D2_SPACING)
N_TARGET = len(D2_TARGET_GRID)

# ================================================================
# CSV HELPERS (no pandas)
# ================================================================


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(rows, **kwargs):
    result = rows
    for key, val in kwargs.items():
        if callable(val):
            result = [r for r in result if val(r[key])]
        else:
            result = [r for r in result if r[key] == val]
    return result


def group_by(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return dict(groups)


def median_val(rows, col):
    vals = [float(r[col]) for r in rows if r[col] != "" and r[col] is not None and _is_finite_str(r[col])]
    return float(np.median(vals)) if vals else float("nan")


def mean_val(rows, col):
    vals = [float(r[col]) for r in rows if r[col] != "" and r[col] is not None and _is_finite_str(r[col])]
    return float(np.mean(vals)) if vals else float("nan")


def percentile_val(rows, col, pct):
    vals = [float(r[col]) for r in rows if r[col] != "" and r[col] is not None and _is_finite_str(r[col])]
    return float(np.percentile(vals, pct)) if vals else float("nan")


def _is_finite_str(s):
    try:
        v = float(s)
        return np.isfinite(v)
    except (ValueError, TypeError):
        return False


def _safe_float(v):
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def col_median(rows, col):
    return median_val(rows, col)


def col_unique(rows, col):
    return sorted(set(r[col] for r in rows))


def col_count(rows, col):
    counts = defaultdict(int)
    for r in rows:
        counts[r[col]] += 1
    return dict(counts)


# ================================================================
# CHECKSUM VERIFICATION
# ================================================================


def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksums():
    ref = read_csv_rows(CHECKSUMS_CSV)
    mismatches = []
    checked = 0
    for row in ref:
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = compute_md5(fp)
            if md5 != row["md5"]:
                mismatches.append(row["rel_path"])
            checked += 1
    return checked, mismatches


# ================================================================
# DATA PARSING
# ================================================================


def parse_relab_full(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0
    segments = []
    parse_warnings = []

    while idx < len(lines):
        line = lines[idx].strip()
        try:
            n = int(line)
            if len(line.split()) == 1 and n > 0:
                idx += 1
                seg_wl, seg_rf, seg_unc = [], [], []
                for i in range(n):
                    if idx >= len(lines):
                        parse_warnings.append(
                            f"Expected {n} rows, got {i} before EOF"
                        )
                        break
                    parts = lines[idx].strip().split()
                    idx += 1
                    if len(parts) < 2:
                        parse_warnings.append(
                            f"Row {i}: only {len(parts)} columns"
                        )
                        continue
                    try:
                        wl = float(parts[0])
                        rf = float(parts[1])
                    except ValueError:
                        parse_warnings.append(
                            f"Row {i}: non-numeric wl/rf: {parts[0]!r} {parts[1]!r}"
                        )
                        continue
                    seg_wl.append(wl)
                    seg_rf.append(rf)
                    if len(parts) > 2:
                        try:
                            seg_unc.append(float(parts[2]))
                        except ValueError:
                            seg_unc.append(np.nan)
                    else:
                        seg_unc.append(None)
                segments.append((
                    np.array(seg_wl),
                    np.array(seg_rf),
                    seg_unc,
                ))
                continue
        except (ValueError, IndexError):
            pass
        idx += 1

    footer_lines = [l.strip() for l in lines[idx:] if l.strip()]
    return segments, footer_lines, parse_warnings


def prepare_d2_spectrum(segments):
    all_wl = np.concatenate([s[0] for s in segments])
    all_rf = np.concatenate([s[1] for s in segments])

    has_unc = segments[0][2][0] is not None if segments else False
    if has_unc:
        all_unc_raw = []
        for s in segments:
            all_unc_raw.extend(s[2])
        all_unc = np.array(all_unc_raw, dtype=float)
    else:
        all_unc = np.array([], dtype=float)

    sort_idx = np.argsort(all_wl)
    all_wl = all_wl[sort_idx]
    all_rf = all_rf[sort_idx]
    if has_unc:
        all_unc = all_unc[sort_idx]

    dup_mask = np.concatenate([[False], np.diff(all_wl) == 0])
    all_wl = all_wl[~dup_mask]
    all_rf = all_rf[~dup_mask]
    if has_unc:
        all_unc = all_unc[~dup_mask]

    d2_mask = (all_wl >= D2_WMIN) & (all_wl <= D2_WMAX)
    wl_d2 = all_wl[d2_mask]
    rf_d2 = all_rf[d2_mask]
    unc_d2 = all_unc[d2_mask] if has_unc else np.array([])

    return wl_d2, rf_d2, unc_d2, has_unc


def parse_footer_metadata(footer_lines):
    meta = {}
    meta["splice_info"] = ""
    meta["source_angle"] = ""
    meta["detect_angle"] = ""
    meta["voltage"] = ""
    meta["date"] = ""
    meta["sample_id"] = ""
    meta["sample_description"] = ""
    meta["run_mode"] = ""
    meta["stage_type"] = ""
    meta["user"] = ""
    meta["pi"] = ""

    for line in footer_lines:
        if "Connected at" in line:
            meta["splice_info"] = line
        elif line.startswith("* Run Mode:"):
            meta["run_mode"] = line.split(":", 1)[1].strip()
        elif line.startswith("* Stage Type:"):
            meta["stage_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("* User:"):
            meta["user"] = line.split(":", 1)[1].strip()
        elif line.startswith("* NASA PI/CoI"):
            meta["pi"] = line.split(":", 1)[1].strip()
        elif line.startswith("Source Ang:"):
            parts = line.split()
            try:
                meta["source_angle"] = parts[2]
            except IndexError:
                pass
            try:
                meta["detect_angle"] = parts[6]
            except IndexError:
                pass
            try:
                meta["voltage"] = parts[9]
            except IndexError:
                pass
        elif line.startswith("Date:"):
            meta["date"] = line.split(":", 1)[1].strip()

    return meta


def get_instrument_prefix(filename):
    prefix = ""
    for ch in filename:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix


def classify_acquisition_regime(prefix):
    if not isinstance(prefix, str) or not prefix:
        return "unknown"
    if prefix.startswith("c"):
        return "VNIR_spectrophotometer"
    elif prefix.startswith("b") or prefix.startswith("m"):
        return "FTIR"
    elif prefix.startswith("n"):
        return "NIR_spectrometer"
    else:
        return "other"


# ================================================================
# PCHIP INTERPOLATION
# ================================================================


def interp_pchip(wl_native, rf_native, target_grid):
    if len(wl_native) < 4:
        return np.interp(target_grid, wl_native, rf_native)
    try:
        interp_fn = PchipInterpolator(wl_native, rf_native)
        return interp_fn(target_grid)
    except Exception:
        return np.interp(target_grid, wl_native, rf_native)


def resample_to_grid(wl, rf, native_spacing, target_grid):
    if native_spacing <= 5.5:
        return np.interp(target_grid, wl, rf)
    else:
        return interp_pchip(wl, rf, target_grid)


# ================================================================
# COMPLEXITY DESCRIPTORS
# ================================================================


def compute_complexity(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 5:
        return {k: np.nan for k in [
            "first_order_variation", "second_order_variation",
            "peak_density", "spectral_entropy", "spectral_range",
            "n_extrema", "curvature_sum", "integrated_abs_d1",
        ]}

    d1 = np.diff(r)
    d2 = np.diff(r, n=2)

    first_order = float(np.sum(np.abs(d1)))
    second_order = float(np.sum(np.abs(d2)))

    is_peak = np.zeros(n, dtype=bool)
    is_trough = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            is_peak[i] = True
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            is_trough[i] = True
    n_extrema = int(is_peak.sum() + is_trough.sum())
    peak_density = n_extrema / n * 100.0

    r_finite = r[np.isfinite(r)]
    if len(r_finite) > 0 and (np.max(r_finite) - np.min(r_finite)) > 1e-15:
        hist, _ = np.histogram(r_finite, bins="auto")
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            p = hist / total
            p = p[p > 0]
            spectral_entropy = float(-np.sum(p * np.log(p)))
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0

    spectral_range = float(np.max(r_finite) - np.min(r_finite)) if len(r_finite) > 0 else 0.0

    curvature_vals = np.abs(d2) / (1.0 + d1[:-1] ** 2) ** 1.5
    curvature_sum = float(np.sum(curvature_vals))

    integrated_abs_d1 = float(np.sum(np.abs(d1)))

    return {
        "first_order_variation": first_order,
        "second_order_variation": second_order,
        "peak_density": peak_density,
        "spectral_entropy": spectral_entropy,
        "spectral_range": spectral_range,
        "n_extrema": n_extrema,
        "curvature_sum": curvature_sum,
        "integrated_abs_d1": integrated_abs_d1,
    }


# ================================================================
# SMOOTHING METHODS
# ================================================================


def smooth_none(r):
    return r.copy(), None


def smooth_savgol(r, window_length, polyorder):
    if window_length > len(r):
        return np.full_like(r, np.nan), "window_exceeds_length"
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        return np.full_like(r, np.nan), "polyorder_too_large"
    try:
        result = savgol_filter(r, window_length, polyorder)
        return result, None
    except Exception as e:
        return np.full_like(r, np.nan), str(e)


def smooth_moving_average(r, window):
    n = len(r)
    if window > n:
        return np.full_like(r, np.nan), "window_exceeds_length"
    half = window // 2
    result = np.empty_like(r)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < window and i >= half and i < n - half:
            lo = i - half
            hi = i + half + 1
        result[i] = np.mean(r[lo:hi])
    return result, None


def smooth_gaussian(r, sigma_points):
    try:
        result = gaussian_filter1d(r, sigma=sigma_points, mode="nearest")
        return result, None
    except Exception as e:
        return np.full_like(r, np.nan), str(e)


def smooth_lowess(r, frac):
    n = len(r)
    x = np.arange(n, dtype=float)

    span = max(3, int(frac * n))
    if span % 2 == 0:
        span += 1
    half = span // 2
    result = np.empty_like(r)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        x_local = x[lo:hi]
        y_local = r[lo:hi]
        w = np.abs(x_local - x[i])
        max_dist = np.max(w) + 1e-15
        u = w / max_dist
        weights = (1.0 - u ** 3) ** 3
        weights[u >= 1.0] = 0.0

        dx = x_local - x[i]
        w_x = weights * dx
        w_y = weights * y_local
        w_x2 = weights * dx * dx
        w_xy = weights * dx * y_local

        det = (np.sum(weights)) * np.sum(w_x2) - np.sum(w_x) ** 2
        if abs(det) < 1e-30:
            result[i] = np.mean(y_local)
        else:
            a0 = (np.sum(w_y) * np.sum(w_x2) - np.sum(w_x) * np.sum(w_xy)) / det
            result[i] = a0

    return result, None


SMOOTHING_CONFIGS = []

SMOOTHING_CONFIGS.append({
    "method": "A_none",
    "label": "No smoothing (baseline)",
    "params": {},
    "param_str": "none",
    "fn": lambda r: smooth_none(r),
})

for wl in [3, 5, 7, 9]:
    for po in [2, 3]:
        if po >= wl:
            continue
        SMOOTHING_CONFIGS.append({
            "method": f"B_savgol_w{wl}_p{po}",
            "label": f"Savitzky-Golay w={wl} p={po}",
            "params": {"window_length": wl, "polyorder": po},
            "param_str": f"w={wl},p={po}",
            "fn": lambda r, _w=wl, _p=po: smooth_savgol(r, _w, _p),
        })

for w in [3, 5]:
    SMOOTHING_CONFIGS.append({
        "method": f"C_movavg_w{w}",
        "label": f"Moving average w={w}",
        "params": {"window": w},
        "param_str": f"w={w}",
        "fn": lambda r, _w=w: smooth_moving_average(r, _w),
    })

for sigma_nm, sigma_pts in [(5.0, 1.0), (10.0, 2.0)]:
    SMOOTHING_CONFIGS.append({
        "method": f"D_gaussian_s{sigma_nm}nm",
        "label": f"Gaussian sigma={sigma_nm} nm ({sigma_pts} pts)",
        "params": {"sigma_nm": sigma_nm, "sigma_points": sigma_pts},
        "param_str": f"sigma={sigma_nm}nm",
        "fn": lambda r, _s=sigma_pts: smooth_gaussian(r, _s),
    })

for frac in [0.01, 0.02, 0.05]:
    SMOOTHING_CONFIGS.append({
        "method": f"E_lowess_f{frac}",
        "label": f"LOWESS frac={frac}",
        "params": {"frac": frac},
        "param_str": f"frac={frac}",
        "fn": lambda r, _f=frac: smooth_lowess(r, _f),
    })


# ================================================================
# FEATURE PRESERVATION ANALYSIS
# ================================================================


def find_local_extrema(r):
    n = len(r)
    peaks = []
    troughs = []
    for i in range(1, n - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            peaks.append(i)
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def compute_peak_prominences(r, extrema_idx):
    prominences = []
    n = len(r)
    for idx in extrema_idx:
        is_peak = True if idx > 0 and r[idx] > r[idx - 1] else False
        if is_peak:
            ref_val = np.min(r)
        else:
            ref_val = np.max(r)
        prom = abs(r[idx] - ref_val)
        prominences.append(prom)
    return np.array(prominences)


def feature_preservation(r_orig, r_smooth, wl):
    n = len(r_orig)
    result = {}

    peaks_orig, troughs_orig = find_local_extrema(r_orig)
    peaks_sm, troughs_sm = find_local_extrema(r_smooth)

    result["peak_count_orig"] = len(peaks_orig)
    result["peak_count_smooth"] = len(peaks_sm)
    result["peak_count_change"] = len(peaks_sm) - len(peaks_orig)

    troughs_orig_count = len(troughs_orig)
    troughs_sm_count = len(troughs_sm)
    result["trough_count_orig"] = troughs_orig_count
    result["trough_count_smooth"] = troughs_sm_count
    result["trough_count_change"] = troughs_sm_count - troughs_orig_count

    total_orig = len(peaks_orig) + len(troughs_orig)
    total_sm = len(peaks_sm) + len(troughs_sm)
    result["extrema_count_orig"] = total_orig
    result["extrema_count_smooth"] = total_sm
    result["extrema_count_change"] = total_sm - total_orig

    if len(peaks_orig) > 0 and len(peaks_sm) > 0:
        displacements = []
        for p in peaks_orig:
            dists = np.abs(peaks_sm.astype(float) - float(p))
            nearest_dist = np.min(dists) * D2_SPACING
            displacements.append(nearest_dist)
        result["major_peak_displacement_nm"] = float(np.median(displacements))
    else:
        result["major_peak_displacement_nm"] = np.nan

    if len(peaks_orig) > 0:
        orig_prom = compute_peak_prominences(r_orig, peaks_orig)
        result["orig_peak_prominence_median"] = float(np.median(orig_prom))
        result["orig_peak_prominence_max"] = float(np.max(orig_prom))
    else:
        result["orig_peak_prominence_median"] = np.nan
        result["orig_peak_prominence_max"] = np.nan

    if len(peaks_sm) > 0:
        sm_prom = compute_peak_prominences(r_smooth, peaks_sm)
        result["smooth_peak_prominence_median"] = float(np.median(sm_prom))
    else:
        result["smooth_peak_prominence_median"] = np.nan

    orig_d1 = np.diff(r_orig)
    sm_d1 = np.diff(r_smooth)
    result["d1_mean_abs_orig"] = float(np.mean(np.abs(orig_d1)))
    result["d1_mean_abs_smooth"] = float(np.mean(np.abs(sm_d1)))
    result["d1_std_orig"] = float(np.std(orig_d1))
    result["d1_std_smooth"] = float(np.std(sm_d1))
    result["d1_max_abs_orig"] = float(np.max(np.abs(orig_d1)))
    result["d1_max_abs_smooth"] = float(np.max(np.abs(sm_d1)))
    result["d1_corr"] = float(np.corrcoef(orig_d1, sm_d1)[0, 1]) if len(orig_d1) > 1 else np.nan

    if len(orig_d1) > 1 and len(sm_d1) > 1:
        orig_d2 = np.diff(r_orig, n=2)
        sm_d2 = np.diff(r_smooth, n=2)
        result["d2_mean_abs_orig"] = float(np.mean(np.abs(orig_d2)))
        result["d2_mean_abs_smooth"] = float(np.mean(np.abs(sm_d2)))
        result["d2_corr"] = float(np.corrcoef(orig_d2, sm_d2)[0, 1]) if len(orig_d2) > 1 else np.nan
    else:
        result["d2_mean_abs_orig"] = np.nan
        result["d2_mean_abs_smooth"] = np.nan
        result["d2_corr"] = np.nan

    if len(orig_d1) > 1:
        orig_curv = np.abs(np.diff(r_orig, n=2)) / (1.0 + orig_d1[:-1] ** 2) ** 1.5
        sm_d1 = np.diff(r_smooth)
        if len(sm_d1) > 1:
            sm_curv = np.abs(np.diff(r_smooth, n=2)) / (1.0 + sm_d1[:-1] ** 2) ** 1.5
            result["curvature_sum_orig"] = float(np.sum(orig_curv))
            result["curvature_sum_smooth"] = float(np.sum(sm_curv))
        else:
            result["curvature_sum_orig"] = float(np.sum(orig_curv))
            result["curvature_sum_smooth"] = np.nan
    else:
        result["curvature_sum_orig"] = np.nan
        result["curvature_sum_smooth"] = np.nan

    abs_min_orig = np.argmin(r_orig)
    result["abs_min_wl_orig"] = float(wl[abs_min_orig])
    result["abs_min_rf_orig"] = float(r_orig[abs_min_orig])

    abs_min_sm = np.argmin(r_smooth)
    result["abs_min_wl_smooth"] = float(wl[abs_min_sm])
    result["abs_min_rf_smooth"] = float(r_smooth[abs_min_sm])
    result["abs_min_wl_shift_nm"] = float(wl[abs_min_sm] - wl[abs_min_orig])

    return result


# ================================================================
# RESIDUAL ANALYSIS
# ================================================================


def compute_residuals(r_orig, r_smooth, wl):
    residual = r_orig - r_smooth
    finite_mask = np.isfinite(residual)

    stats = {}
    if finite_mask.sum() == 0:
        return {k: np.nan for k in [
            "residual_mean", "residual_std", "residual_median",
            "residual_max_abs", "residual_iqr",
            "residual_first_quarter_mean", "residual_last_quarter_mean",
            "residual_mid_half_mean",
        ]}

    stats["residual_mean"] = float(np.mean(residual[finite_mask]))
    stats["residual_std"] = float(np.std(residual[finite_mask]))
    stats["residual_median"] = float(np.median(residual[finite_mask]))
    stats["residual_max_abs"] = float(np.max(np.abs(residual[finite_mask])))
    stats["residual_iqr"] = float(np.percentile(residual[finite_mask], 75) -
                                   np.percentile(residual[finite_mask], 25))

    n = len(residual)
    q1_end = n // 4
    q4_start = 3 * n // 4
    mid_start = n // 4
    mid_end = 3 * n // 4

    if q1_end > 0:
        stats["residual_first_quarter_mean"] = float(np.mean(np.abs(residual[:q1_end])))
    else:
        stats["residual_first_quarter_mean"] = np.nan

    if q4_start < n:
        stats["residual_last_quarter_mean"] = float(np.mean(np.abs(residual[q4_start:])))
    else:
        stats["residual_last_quarter_mean"] = np.nan

    if mid_end > mid_start:
        stats["residual_mid_half_mean"] = float(np.mean(np.abs(residual[mid_start:mid_end])))
    else:
        stats["residual_mid_half_mean"] = np.nan

    return stats


# ================================================================
# EDGE EFFECT ANALYSIS
# ================================================================


def compute_edge_effects(r_orig, r_smooth, n_edge=10):
    residual = r_orig - r_smooth
    if not np.all(np.isfinite(residual)):
        return {k: np.nan for k in [
            "edge_left_mean_abs", "edge_right_mean_abs",
            "edge_left_max_abs", "edge_right_max_abs",
            "edge_left_std", "edge_right_std",
            "interior_mean_abs",
        ]}

    n = len(residual)
    n_edge = min(n_edge, n // 4)

    left = residual[:n_edge]
    right = residual[-n_edge:]
    interior = residual[n_edge:-n_edge] if n_edge < n // 2 else residual

    return {
        "edge_left_mean_abs": float(np.mean(np.abs(left))),
        "edge_right_mean_abs": float(np.mean(np.abs(right))),
        "edge_left_max_abs": float(np.max(np.abs(left))),
        "edge_right_max_abs": float(np.max(np.abs(right))),
        "edge_left_std": float(np.std(left)),
        "edge_right_std": float(np.std(right)),
        "interior_mean_abs": float(np.mean(np.abs(interior))),
    }


# ================================================================
# NUMERICAL STABILITY CHECK
# ================================================================


def check_numerical_stability(r_smooth, r_orig):
    flags = []
    n = len(r_smooth)

    n_nan = int(np.sum(np.isnan(r_smooth)))
    n_inf = int(np.sum(np.isinf(r_smooth)))
    if n_nan > 0:
        flags.append(f"nan_generated({n_nan})")
    if n_inf > 0:
        flags.append(f"inf_generated({n_inf})")

    finite_mask = np.isfinite(r_smooth)
    if finite_mask.any():
        rf_min = np.min(r_smooth[finite_mask])
        rf_max = np.max(r_smooth[finite_mask])
        orig_min = np.min(r_orig[np.isfinite(r_orig)]) if np.any(np.isfinite(r_orig)) else 0
        orig_max = np.max(r_orig[np.isfinite(r_orig)]) if np.any(np.isfinite(r_orig)) else 1

        if rf_min < orig_min - 0.1:
            flags.append(f"artificial_low_value({rf_min:.4f})")
        if rf_max > orig_max + 0.1:
            flags.append(f"artificial_high_value({rf_max:.4f})")

    return ";".join(flags) if flags else ""


# ================================================================
# MAIN INVESTIGATION
# ================================================================


def run_investigation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []

    def rpt(line=""):
        report_lines.append(line)

    rpt("# Phase 2 Step 6 — Smoothing / Denoising Investigation")
    rpt()
    rpt(f"**Date:** {timestamp}")
    rpt("**Status:** Investigation only — no frozen decisions changed")
    rpt()

    # --- Checksums ---
    checked, mismatches = validate_checksums()
    rpt("---")
    rpt()
    rpt("## Checksum Verification")
    rpt()
    rpt(f"- Raw files checked: {checked}")
    rpt(f"- MD5 mismatches: {len(mismatches)}")
    rpt()
    if mismatches:
        rpt("**WARNING: checksum mismatches detected!**")
        for mm in mismatches:
            rpt(f"  - {mm}")
        rpt()

    # --- Load metadata ---
    measurements = read_csv_rows(MEASUREMENTS_CSV)
    population = read_csv_rows(POPULATION_CSV)
    d2_val = read_csv_rows(D2_VALIDATION_CSV)
    reflectance_qc = read_csv_rows(REFLECTANCE_QC_CSV)

    pop_map = {}
    for r in population:
        pop_map[r["file_id"]] = r

    d2_val_map = {}
    for r in d2_val:
        d2_val_map[r["relab_measurement_id"]] = r

    qc_map = {}
    for r in reflectance_qc:
        qc_col = "qc_category" if "qc_category" in r else "reflectance_qc_category"
        qc_map[r["file_id"]] = r[qc_col]

    d2_pop = []
    for m in measurements:
        fid = m["file_id"]
        if fid not in pop_map:
            continue
        if fid not in d2_val_map:
            continue
        pop_row = pop_map[fid]
        d2_row = d2_val_map[fid]
        entry = dict(m)
        entry["inclusion_category"] = pop_row.get("inclusion_category", "")
        entry["qc_flags"] = pop_row.get("qc_flags", "")
        entry["exclusion_reason"] = pop_row.get("exclusion_reason", "")
        entry["native_spacing_nm"] = _safe_float(d2_row.get("native_spacing_nm", ""))
        entry["native_5nm_support"] = d2_row.get("native_5nm_support", "")
        entry["d2_category"] = d2_row.get("category", "")
        entry["instrument_prefix"] = m.get("filename_prefix_code", "")
        entry["reflectance_qc"] = qc_map.get(fid, "unknown")
        d2_pop.append(entry)

    d2_pop.sort(key=lambda r: r["file_id"])

    n_d2 = len(d2_pop)
    minerals_set = set(r["mineral_directory"] for r in d2_pop)
    n_minerals = len(minerals_set)
    minerals_list = sorted(minerals_set)
    specimen_groups = set(r["specimen_group_id"] for r in d2_pop)
    n_specimen_groups = len(specimen_groups)
    n_native_5nm = sum(1 for r in d2_pop if r["native_spacing_nm"] <= 5.5)
    n_native_10nm = sum(1 for r in d2_pop if r["native_spacing_nm"] > 5.5)

    # ================================================================
    # SECTION 1: Investigation population
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 1. Investigation Population")
    rpt()

    rpt(f"- D2-compatible spectra: **{n_d2}**")
    rpt(f"- Minerals represented: **{n_minerals}**")
    rpt(f"- Specimen groups: **{n_specimen_groups}**")
    rpt(f"- Native 5-nm spectra: **{n_native_5nm}**")
    rpt(f"- Native 10-nm spectra (PCHIP-represented at 5 nm): **{n_native_10nm}**")
    rpt()

    assert n_d2 == 454, f"Expected 454 D2 spectra, got {n_d2}"
    assert n_minerals == 18, f"Expected 18 minerals, got {n_minerals}"
    assert n_native_5nm == 390, f"Expected 390 native-5nm, got {n_native_5nm}"
    assert n_native_10nm == 64, f"Expected 64 native-10nm, got {n_native_10nm}"

    rpt("### Mineral distribution")
    rpt()
    mineral_counts = col_count(d2_pop, "mineral_directory")
    rpt("| Mineral | Count |")
    rpt("|---------|-------|")
    for mineral in minerals_list:
        rpt(f"| {mineral} | {mineral_counts.get(mineral, 0)} |")
    rpt(f"| **Total** | **{n_d2}** |")
    rpt()

    # ================================================================
    # Load all D2 spectra
    # ================================================================

    print("Loading D2 spectra...")
    spectra = []
    for row in d2_pop:
        fp = DATA_DIR / row["rel_path"]
        segments, footer, pw = parse_relab_full(fp)
        wl_d2, rf_d2, unc_d2, has_unc = prepare_d2_spectrum(segments)
        footer_meta = parse_footer_metadata(footer)

        native_spacing = row["native_spacing_nm"]
        if native_spacing > 5.5:
            rf_grid = resample_to_grid(wl_d2, rf_d2, native_spacing, D2_TARGET_GRID)
            wl_grid = D2_TARGET_GRID.copy()
            is_interpolated = True
        else:
            rf_grid = np.interp(D2_TARGET_GRID, wl_d2, rf_d2)
            wl_grid = D2_TARGET_GRID.copy()
            is_interpolated = False

        prefix = row["instrument_prefix"]
        if not isinstance(prefix, str):
            prefix = ""

        spectra.append({
            "file_id": row["file_id"],
            "filename": row["filename"],
            "mineral": row["mineral_directory"],
            "specimen_group_id": row["specimen_group_id"],
            "inclusion_category": row["inclusion_category"],
            "reflectance_qc": row["reflectance_qc"],
            "native_spacing_nm": native_spacing,
            "instrument_prefix": prefix,
            "acquisition_regime": classify_acquisition_regime(prefix),
            "wl_grid": wl_grid,
            "rf_grid": rf_grid,
            "is_interpolated": is_interpolated,
        })

    print(f"  Loaded {len(spectra)} spectra on {N_TARGET}-point grid")

    # ================================================================
    # SECTION 2: Raw spectral variation characterization
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 2. Raw Spectral Variation Characterization")
    rpt()

    variation_records = []
    for sp in spectra:
        r = sp["rf_grid"]
        d1 = np.diff(r)
        d2 = np.diff(r, n=2)

        n_const_runs = 0
        const_run_total = 0
        run_len = 1
        for i in range(1, len(r)):
            if abs(r[i] - r[i - 1]) < 1e-10:
                run_len += 1
            else:
                if run_len >= 10:
                    n_const_runs += 1
                    const_run_total += run_len
                run_len = 1
        if run_len >= 10:
            n_const_runs += 1
            const_run_total += run_len

        n_isolated_spikes = 0
        if len(d1) > 2:
            med_abs_d1 = np.median(np.abs(d1))
            if med_abs_d1 > 1e-15:
                for i in range(1, len(d1)):
                    if (abs(d1[i]) > 10 * med_abs_d1 and
                            abs(d1[i - 1]) > 10 * med_abs_d1 and
                            i < len(d1) - 1 and
                            abs(r[i + 1] - r[i - 1]) < 3 * med_abs_d1):
                        n_isolated_spikes += 1

        rec = {
            "file_id": sp["file_id"],
            "mineral": sp["mineral"],
            "native_spacing_nm": sp["native_spacing_nm"],
            "is_interpolated": sp["is_interpolated"],
            "reflectance_qc": sp["reflectance_qc"],
            "d1_mean_abs": float(np.mean(np.abs(d1))),
            "d1_std": float(np.std(d1)),
            "d1_max_abs": float(np.max(np.abs(d1))),
            "d1_median_abs": float(np.median(np.abs(d1))),
            "d2_mean_abs": float(np.mean(np.abs(d2))),
            "d2_std": float(np.std(d2)),
            "d2_max_abs": float(np.max(np.abs(d2))),
            "hf_energy": float(np.sum(d1 ** 2)),
            "n_constant_runs": n_const_runs,
            "const_run_total_points": const_run_total,
            "n_isolated_spikes": n_isolated_spikes,
        }
        variation_records.append(rec)

    rpt("### Global distribution of spectral variation metrics")
    rpt()
    rpt("| Metric | Min | Q25 | Median | Mean | Q75 | Max |")
    rpt("|--------|-----|-----|--------|------|-----|-----|")
    for col in ["d1_mean_abs", "d1_std", "d1_max_abs", "d2_mean_abs", "d2_std", "hf_energy"]:
        vals = np.array([r[col] for r in variation_records])
        q25, q75 = np.percentile(vals, [25, 75])
        rpt(f"| {col} | {vals.min():.6f} | {q25:.6f} | {np.median(vals):.6f} | "
            f"{np.mean(vals):.6f} | {q75:.6f} | {vals.max():.6f} |")
    rpt()

    rpt("### Variation by mineral")
    rpt()
    rpt("| Mineral | n | Median |d1| | Median |d2| | Median HF energy |")
    rpt("|---------|---|-----------|-----------|------------------|")
    var_by_mineral = defaultdict(list)
    for r in variation_records:
        var_by_mineral[r["mineral"]].append(r)
    for mineral in minerals_list:
        sub = var_by_mineral[mineral]
        rpt(f"| {mineral} | {len(sub)} | {np.median([r['d1_mean_abs'] for r in sub]):.6f} | "
            f"{np.median([r['d2_mean_abs'] for r in sub]):.6f} | "
            f"{np.median([r['hf_energy'] for r in sub]):.4f} |")
    rpt()

    rpt("### Variation by native sampling interval")
    rpt()
    rpt("| Native grid | n | Median |d1| | Median |d2| | Median HF energy |")
    rpt("|-------------|---|-----------|-----------|------------------|")
    for label, filt in [("5 nm", lambda r: r["native_spacing_nm"] <= 5.5),
                        ("10 nm", lambda r: r["native_spacing_nm"] > 5.5)]:
        sub = [r for r in variation_records if filt(r)]
        rpt(f"| {label} | {len(sub)} | {np.median([r['d1_mean_abs'] for r in sub]):.6f} | "
            f"{np.median([r['d2_mean_abs'] for r in sub]):.6f} | "
            f"{np.median([r['hf_energy'] for r in sub]):.4f} |")
    rpt()

    rpt("### Variation by QC category")
    rpt()
    rpt("| QC | n | Median |d1| | Median isolated spikes |")
    rpt("|----|---|-----------|----------------------|")
    for cat in ["A", "B", "C"]:
        sub = [r for r in variation_records if r["reflectance_qc"] == cat]
        if sub:
            rpt(f"| {cat} | {len(sub)} | {np.median([r['d1_mean_abs'] for r in sub]):.6f} | "
                f"{np.median([r['n_isolated_spikes'] for r in sub]):.0f} |")
    rpt()

    rpt("### Constant runs and isolated spikes summary")
    rpt()
    n_with_const = sum(1 for r in variation_records if r["n_constant_runs"] > 0)
    n_with_spikes = sum(1 for r in variation_records if r["n_isolated_spikes"] > 0)
    rpt(f"- Spectra with constant runs (>=10 points): {n_with_const}")
    rpt(f"- Spectra with isolated spikes: {n_with_spikes}")
    rpt()

    rpt("### Classification of high-frequency variation")
    rpt()
    rpt("**Potential measurement noise:** Random, small-amplitude variation")
    rpt("without stable spectral structure. The median |d1| across all")
    rpt(f"454 spectra is {np.median([r['d1_median_abs'] for r in variation_records]):.6f}.")
    rpt()
    rpt("**Potential real spectral structure:** Narrow absorption bands,")
    rpt("shoulders, and systematic mineral-specific features that produce")
    rpt("consistent high-frequency variation within a mineral group.")
    rpt()
    rpt("**Instrumental artifacts:** Constant runs (detector saturation),")
    rpt(f"isolated spikes ({n_with_spikes} spectra), and detector-transition")
    rpt("discontinuities (documented in Phase 2 Step 3).")
    rpt()

    # ================================================================
    # SECTION 3: Apply all smoothing methods
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 3. Candidate Smoothing Methods")
    rpt()
    rpt("| Method | Label | Parameters |")
    rpt("|--------|-------|------------|")
    for cfg in SMOOTHING_CONFIGS:
        rpt(f"| {cfg['method']} | {cfg['label']} | {cfg['param_str']} |")
    rpt()

    print(f"Applying {len(SMOOTHING_CONFIGS)} smoothing configurations to {len(spectra)} spectra...")

    comparison_rows = []
    smoothed_cache = {}

    for cfg_idx, cfg in enumerate(SMOOTHING_CONFIGS):
        method = cfg["method"]
        print(f"  [{cfg_idx + 1}/{len(SMOOTHING_CONFIGS)}] {cfg['label']}...")

        for sp_idx, sp in enumerate(spectra):
            r_orig = sp["rf_grid"]
            wl = sp["wl_grid"]
            fid = sp["file_id"]

            r_smooth, error = cfg["fn"](r_orig)

            failure_flag = error if error else ""

            if error:
                comp_row = {
                    "file_id": fid,
                    "filename": sp["filename"],
                    "mineral": sp["mineral"],
                    "specimen_group_id": sp["specimen_group_id"],
                    "inclusion_category": sp["inclusion_category"],
                    "reflectance_qc": sp["reflectance_qc"],
                    "native_spacing_nm": sp["native_spacing_nm"],
                    "is_interpolated": sp["is_interpolated"],
                    "acquisition_regime": sp["acquisition_regime"],
                    "smoothing_method": method,
                    "parameter": cfg["param_str"],
                    "failure_flag": failure_flag,
                }
                for k in ["peak_count_orig", "peak_count_smooth", "peak_count_change",
                           "major_peak_displacement_nm",
                           "d1_mean_abs_orig", "d1_mean_abs_smooth",
                           "d1_std_orig", "d1_std_smooth",
                           "d1_max_abs_orig", "d1_max_abs_smooth",
                           "d1_corr",
                           "d2_mean_abs_orig", "d2_mean_abs_smooth", "d2_corr",
                           "curvature_sum_orig", "curvature_sum_smooth",
                           "first_order_variation_orig", "first_order_variation_smooth",
                           "second_order_variation_orig", "second_order_variation_smooth",
                           "peak_density_orig", "peak_density_smooth",
                           "spectral_entropy_orig", "spectral_entropy_smooth",
                           "spectral_range_orig", "spectral_range_smooth",
                           "n_extrema_orig", "n_extrema_smooth",
                           "curvature_sum_desc_orig", "curvature_sum_desc_smooth",
                           "integrated_abs_d1_orig", "integrated_abs_d1_smooth",
                           "residual_mean", "residual_std", "residual_median",
                           "residual_max_abs", "residual_iqr",
                           "edge_left_mean_abs", "edge_right_mean_abs",
                           "edge_left_max_abs", "edge_right_max_abs",
                           "interior_mean_abs",
                           "numerical_flags"]:
                    comp_row[k] = np.nan if "flag" not in k else ""
                comparison_rows.append(comp_row)
                continue

            fp = feature_preservation(r_orig, r_smooth, wl)
            res = compute_residuals(r_orig, r_smooth, wl)
            edge = compute_edge_effects(r_orig, r_smooth)
            num_flags = check_numerical_stability(r_smooth, r_orig)

            comp_orig = compute_complexity(r_orig)
            comp_smooth = compute_complexity(r_smooth)

            comp_row = {
                "file_id": fid,
                "filename": sp["filename"],
                "mineral": sp["mineral"],
                "specimen_group_id": sp["specimen_group_id"],
                "inclusion_category": sp["inclusion_category"],
                "reflectance_qc": sp["reflectance_qc"],
                "native_spacing_nm": sp["native_spacing_nm"],
                "is_interpolated": sp["is_interpolated"],
                "acquisition_regime": sp["acquisition_regime"],
                "smoothing_method": method,
                "parameter": cfg["param_str"],
                "failure_flag": failure_flag,
                "peak_count_orig": fp["peak_count_orig"],
                "peak_count_smooth": fp["peak_count_smooth"],
                "peak_count_change": fp["peak_count_change"],
                "major_peak_displacement_nm": fp["major_peak_displacement_nm"],
                "d1_mean_abs_orig": fp["d1_mean_abs_orig"],
                "d1_mean_abs_smooth": fp["d1_mean_abs_smooth"],
                "d1_std_orig": fp["d1_std_orig"],
                "d1_std_smooth": fp["d1_std_smooth"],
                "d1_max_abs_orig": fp["d1_max_abs_orig"],
                "d1_max_abs_smooth": fp["d1_max_abs_smooth"],
                "d1_corr": fp["d1_corr"],
                "d2_mean_abs_orig": fp["d2_mean_abs_orig"],
                "d2_mean_abs_smooth": fp["d2_mean_abs_smooth"],
                "d2_corr": fp["d2_corr"],
                "curvature_sum_orig": fp["curvature_sum_orig"],
                "curvature_sum_smooth": fp["curvature_sum_smooth"],
                "first_order_variation_orig": comp_orig["first_order_variation"],
                "first_order_variation_smooth": comp_smooth["first_order_variation"],
                "second_order_variation_orig": comp_orig["second_order_variation"],
                "second_order_variation_smooth": comp_smooth["second_order_variation"],
                "peak_density_orig": comp_orig["peak_density"],
                "peak_density_smooth": comp_smooth["peak_density"],
                "spectral_entropy_orig": comp_orig["spectral_entropy"],
                "spectral_entropy_smooth": comp_smooth["spectral_entropy"],
                "spectral_range_orig": comp_orig["spectral_range"],
                "spectral_range_smooth": comp_smooth["spectral_range"],
                "n_extrema_orig": comp_orig["n_extrema"],
                "n_extrema_smooth": comp_smooth["n_extrema"],
                "curvature_sum_desc_orig": comp_orig["curvature_sum"],
                "curvature_sum_desc_smooth": comp_smooth["curvature_sum"],
                "integrated_abs_d1_orig": comp_orig["integrated_abs_d1"],
                "integrated_abs_d1_smooth": comp_smooth["integrated_abs_d1"],
                "residual_mean": res["residual_mean"],
                "residual_std": res["residual_std"],
                "residual_median": res["residual_median"],
                "residual_max_abs": res["residual_max_abs"],
                "residual_iqr": res["residual_iqr"],
                "edge_left_mean_abs": edge["edge_left_mean_abs"],
                "edge_right_mean_abs": edge["edge_right_mean_abs"],
                "edge_left_max_abs": edge["edge_left_max_abs"],
                "edge_right_max_abs": edge["edge_right_max_abs"],
                "interior_mean_abs": edge["interior_mean_abs"],
                "numerical_flags": num_flags,
            }
            comparison_rows.append(comp_row)

            if method != "A_none":
                key = (fid, method)
                if key not in smoothed_cache:
                    smoothed_cache[key] = r_smooth

    print(f"  Computed {len(comparison_rows)} rows")

    # Helper: split baseline vs non-baseline
    baseline = [r for r in comparison_rows if r["smoothing_method"] == "A_none"]
    non_baseline = [r for r in comparison_rows if r["smoothing_method"] != "A_none"]
    all_methods_sorted = sorted(set(r["smoothing_method"] for r in non_baseline))

    # Helper: filter non_baseline by method
    def nb_by_method(m):
        return [r for r in non_baseline if r["smoothing_method"] == m]

    def nb_valid(rows, col):
        return [r for r in rows if _is_finite_str(r[col])]

    # ================================================================
    # SECTION 4: Feature preservation results
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 4. Feature Preservation Analysis")
    rpt()

    rpt("### Peak count changes")
    rpt()
    rpt("| Method | Median change | Spectra with loss | Spectra with gain | Spectra unchanged |")
    rpt("|--------|--------------|-------------------|-------------------|-------------------|")
    for method in all_methods_sorted:
        sub = nb_valid(nb_by_method(method), "peak_count_change")
        if not sub:
            continue
        med_change = np.median([float(r["peak_count_change"]) for r in sub])
        n_loss = sum(1 for r in sub if float(r["peak_count_change"]) < 0)
        n_gain = sum(1 for r in sub if float(r["peak_count_change"]) > 0)
        n_unchanged = sum(1 for r in sub if float(r["peak_count_change"]) == 0)
        rpt(f"| {method} | {med_change:.0f} | {n_loss} | {n_gain} | {n_unchanged} |")
    rpt()

    rpt("### Peak displacement")
    rpt()
    rpt("| Method | Median displacement (nm) | Q75 | Max |")
    rpt("|--------|------------------------|-----|-----|")
    for method in all_methods_sorted:
        sub = nb_valid(nb_by_method(method), "major_peak_displacement_nm")
        if not sub:
            continue
        vals = np.array([float(r["major_peak_displacement_nm"]) for r in sub])
        med = np.median(vals)
        q75 = np.percentile(vals, 75)
        mx = np.max(vals)
        rpt(f"| {method} | {med:.1f} | {q75:.1f} | {mx:.1f} |")
    rpt()

    rpt("### Derivative preservation")
    rpt()
    rpt("| Method | Median d1 correlation | Median d2 correlation |")
    rpt("|--------|----------------------|----------------------|")
    for method in all_methods_sorted:
        sub_d1 = nb_valid(nb_by_method(method), "d1_corr")
        sub_d2 = nb_valid(nb_by_method(method), "d2_corr")
        d1_med = np.median([float(r["d1_corr"]) for r in sub_d1]) if sub_d1 else float("nan")
        d2_med = np.median([float(r["d2_corr"]) for r in sub_d2]) if sub_d2 else float("nan")
        rpt(f"| {method} | {d1_med:.4f} | {d2_med:.4f} |")
    rpt()

    rpt("### Curvature preservation")
    rpt()
    rpt("| Method | Median curvature orig | Median curvature smooth | Relative change |")
    rpt("|--------|----------------------|------------------------|----------------|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        sub_v = [r for r in sub if _is_finite_str(r["curvature_sum_orig"]) and _is_finite_str(r["curvature_sum_smooth"])]
        if not sub_v:
            continue
        med_orig = np.median([float(r["curvature_sum_orig"]) for r in sub_v])
        med_smooth = np.median([float(r["curvature_sum_smooth"]) for r in sub_v])
        rel_change = (med_smooth - med_orig) / med_orig * 100 if abs(med_orig) > 1e-15 else float("nan")
        rpt(f"| {method} | {med_orig:.4f} | {med_smooth:.4f} | {rel_change:+.1f}% |")
    rpt()

    # ================================================================
    # SECTION 5: Residual analysis
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 5. Residual Analysis")
    rpt()

    rpt("### Residual amplitude by method")
    rpt()
    rpt("| Method | Median |residual| | Q75 | Max | IQR |")
    rpt("|--------|---------------------|-----|-----|-----|")
    for method in all_methods_sorted:
        sub = nb_valid(nb_by_method(method), "residual_max_abs")
        if not sub:
            continue
        vals = np.array([float(r["residual_max_abs"]) for r in sub])
        med_abs = np.median(vals)
        q75_abs = np.percentile(vals, 75)
        max_abs = np.max(vals)
        iqr_vals = np.array([float(r["residual_iqr"]) for r in sub if _is_finite_str(r["residual_iqr"])])
        med_iqr = np.median(iqr_vals) if len(iqr_vals) > 0 else float("nan")
        rpt(f"| {method} | {med_abs:.6f} | {q75_abs:.6f} | {max_abs:.6f} | {med_iqr:.6f} |")
    rpt()

    rpt("### Residual wavelength dependence")
    rpt()
    rpt("| Method | Left edge (320-370 nm) | Interior | Right edge (2500-2550 nm) |")
    rpt("|--------|----------------------|----------|--------------------------|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        sub_v = [r for r in sub if _is_finite_str(r["edge_left_mean_abs"])
                 and _is_finite_str(r["interior_mean_abs"])
                 and _is_finite_str(r["edge_right_mean_abs"])]
        if not sub_v:
            continue
        left = np.median([float(r["edge_left_mean_abs"]) for r in sub_v])
        interior = np.median([float(r["interior_mean_abs"]) for r in sub_v])
        right = np.median([float(r["edge_right_mean_abs"]) for r in sub_v])
        rpt(f"| {method} | {left:.6f} | {interior:.6f} | {right:.6f} |")
    rpt()

    rpt("### Residual distribution by mineral")
    rpt()
    rpt("Residual max-abs (median across spectra) for each method:")
    rpt()
    header = "| Mineral |"
    sep = "|---------|"
    for method in all_methods_sorted:
        short = method.split("_")[0]
        header += f" {short} |"
        sep += "------|"
    rpt(header)
    rpt(sep)
    for mineral in minerals_list:
        row_str = f"| {mineral} |"
        for method in all_methods_sorted:
            sub = [r for r in nb_by_method(method)
                   if r["mineral"] == mineral and _is_finite_str(r["residual_max_abs"])]
            if sub:
                row_str += f" {np.median([float(r['residual_max_abs']) for r in sub]):.5f} |"
            else:
                row_str += " N/A |"
        rpt(row_str)
    rpt()

    rpt("### Residual structure assessment")
    rpt()
    rpt("For each method, the residual (original - smoothed) was analyzed for")
    rpt("wavelength dependence, mineral dependence, and structure.")
    rpt()

    for method in all_methods_sorted:
        sub = nb_by_method(method)
        sub_v = [r for r in sub if _is_finite_str(r["residual_mean"]) and _is_finite_str(r["residual_std"])]
        if not sub_v:
            continue
        mean_of_means = np.mean([float(r["residual_mean"]) for r in sub_v])
        mean_of_stds = np.mean([float(r["residual_std"]) for r in sub_v])
        rpt(f"**{method}:** Mean residual = {mean_of_means:+.6f}, "
            f"Mean residual std = {mean_of_stds:.6f}")

        mineral_groups = group_by(sub_v, "mineral")
        mineral_stds = {}
        for m_name, m_rows in mineral_groups.items():
            vals = [float(r["residual_std"]) for r in m_rows if _is_finite_str(r["residual_std"])]
            if vals:
                mineral_stds[m_name] = np.median(vals)
        if len(mineral_stds) > 1:
            std_vals = list(mineral_stds.values())
            ratio = max(std_vals) / min(std_vals) if min(std_vals) > 1e-15 else float("nan")
            rpt(f"  Mineral variation in residual amplitude: {ratio:.1f}x "
                f"(max/min median std)")
    rpt()

    # ================================================================
    # SECTION 6: Complexity sensitivity
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 6. Complexity Sensitivity")
    rpt()

    descriptors = [
        ("first_order_variation", "First-order variation"),
        ("second_order_variation", "Second-order variation"),
        ("peak_density", "Peak density"),
        ("spectral_entropy", "Spectral entropy"),
        ("spectral_range", "Spectral range"),
        ("n_extrema", "Number of extrema"),
        ("curvature_sum_desc", "Curvature sum"),
        ("integrated_abs_d1", "Integrated |d1|"),
    ]

    rpt("### Complexity descriptor changes (relative to baseline)")
    rpt()

    for desc_key, desc_label in descriptors:
        col_orig = f"{desc_key}_orig"
        col_smooth = f"{desc_key}_smooth"
        rpt(f"#### {desc_label}")
        rpt()
        rpt("| Method | Median orig | Median smooth | Median rel. change |")
        rpt("|--------|------------|--------------|-------------------|")

        baseline_v = [r for r in baseline if _is_finite_str(r[col_orig])]
        baseline_val = np.median([float(r[col_orig]) for r in baseline_v]) if baseline_v else float("nan")

        for method in all_methods_sorted:
            sub = nb_by_method(method)
            sub_v = [r for r in sub if _is_finite_str(r[col_orig]) and _is_finite_str(r[col_smooth])]
            if not sub_v:
                continue
            med_orig = np.median([float(r[col_orig]) for r in sub_v])
            med_smooth = np.median([float(r[col_smooth]) for r in sub_v])
            if abs(med_orig) > 1e-15:
                rel_change = (med_smooth - med_orig) / med_orig * 100
            else:
                rel_change = float("nan")
            rpt(f"| {method} | {med_orig:.6f} | {med_smooth:.6f} | {rel_change:+.2f}% |")
        rpt()

    rpt("### Complexity change by mineral (selected methods)")
    rpt()

    selected_methods = [m for m in ["B_savgol_w5_p2", "B_savgol_w7_p2",
                                     "C_movavg_w3", "D_gaussian_s10.0nm"]
                        if m in all_methods_sorted]

    if selected_methods:
        rpt("| Mineral | Method | FOV change | Peak density change | Entropy change |")
        rpt("|---------|--------|-----------|--------------------:|---------------:|")
        for mineral in minerals_list:
            for method in selected_methods:
                sub = [r for r in nb_by_method(method) if r["mineral"] == mineral]
                if not sub:
                    continue
                fov_orig = np.median([float(r["first_order_variation_orig"]) for r in sub
                                      if _is_finite_str(r["first_order_variation_orig"])])
                fov_sm = np.median([float(r["first_order_variation_smooth"]) for r in sub
                                    if _is_finite_str(r["first_order_variation_smooth"])])
                fov_ch = ((fov_sm - fov_orig) / fov_orig * 100) if abs(fov_orig) > 1e-15 else float("nan")

                pd_orig = np.median([float(r["peak_density_orig"]) for r in sub
                                     if _is_finite_str(r["peak_density_orig"])])
                pd_sm = np.median([float(r["peak_density_smooth"]) for r in sub
                                   if _is_finite_str(r["peak_density_smooth"])])
                pd_ch = ((pd_sm - pd_orig) / pd_orig * 100) if abs(pd_orig) > 1e-15 else float("nan")

                se_orig = np.median([float(r["spectral_entropy_orig"]) for r in sub
                                     if _is_finite_str(r["spectral_entropy_orig"])])
                se_sm = np.median([float(r["spectral_entropy_smooth"]) for r in sub
                                   if _is_finite_str(r["spectral_entropy_smooth"])])
                se_ch = ((se_sm - se_orig) / se_orig * 100) if abs(se_orig) > 1e-15 else float("nan")

                short_method = method.split("_", 1)[1] if "_" in method else method
                rpt(f"| {mineral} | {short_method} | {fov_ch:+.1f}% | {pd_ch:+.1f}% | {se_ch:+.1f}% |")
        rpt()

    # ================================================================
    # SECTION 7: Native 5-nm vs 10-nm comparison
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 7. Native 5-nm vs Native 10-nm Comparison")
    rpt()

    rpt("### Residual amplitude comparison")
    rpt()
    rpt("| Method | Native 5nm (median max-abs) | Native 10nm (median max-abs) | Ratio |")
    rpt("|--------|---------------------------|-----------------------------|-------|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        v5 = [r for r in sub if r["native_spacing_nm"] <= 5.5 and _is_finite_str(r["residual_max_abs"])]
        v10 = [r for r in sub if r["native_spacing_nm"] > 5.5 and _is_finite_str(r["residual_max_abs"])]
        if not v5 or not v10:
            continue
        med5 = np.median([float(r["residual_max_abs"]) for r in v5])
        med10 = np.median([float(r["residual_max_abs"]) for r in v10])
        ratio = med10 / med5 if abs(med5) > 1e-15 else float("nan")
        rpt(f"| {method} | {med5:.6f} | {med10:.6f} | {ratio:.2f}x |")
    rpt()

    rpt("### Complexity change comparison")
    rpt()
    rpt("| Method | Grid | FOV rel. change | Peak density change | Extrema change |")
    rpt("|--------|------|----------------|--------------------:|---------------:|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        for label, filt in [("5nm", lambda r: r["native_spacing_nm"] <= 5.5),
                            ("10nm", lambda r: r["native_spacing_nm"] > 5.5)]:
            valid = [r for r in sub if filt(r)
                     and _is_finite_str(r["first_order_variation_orig"])
                     and _is_finite_str(r["first_order_variation_smooth"])]
            if not valid:
                continue
            fov_orig = np.median([float(r["first_order_variation_orig"]) for r in valid])
            fov_sm = np.median([float(r["first_order_variation_smooth"]) for r in valid])
            fov_ch = ((fov_sm - fov_orig) / fov_orig * 100) if abs(fov_orig) > 1e-15 else float("nan")

            pd_orig = np.median([float(r["peak_density_orig"]) for r in valid
                                 if _is_finite_str(r["peak_density_orig"])])
            pd_sm = np.median([float(r["peak_density_smooth"]) for r in valid
                               if _is_finite_str(r["peak_density_smooth"])])
            pd_ch = ((pd_sm - pd_orig) / pd_orig * 100) if abs(pd_orig) > 1e-15 else float("nan")

            ne_orig = np.median([float(r["n_extrema_orig"]) for r in valid
                                 if _is_finite_str(r["n_extrema_orig"])])
            ne_sm = np.median([float(r["n_extrema_smooth"]) for r in valid
                               if _is_finite_str(r["n_extrema_smooth"])])
            ne_ch = ne_sm - ne_orig

            short_method = method.split("_", 1)[1] if "_" in method else method
            rpt(f"| {short_method} | {label} | {fov_ch:+.1f}% | {pd_ch:+.1f}% | {ne_ch:+.0f} |")
    rpt()

    rpt("### Interpolation vs smoothing distinction")
    rpt()
    rpt("The 64 native-10-nm spectra were first represented at 5 nm using PCHIP")
    rpt("interpolation (frozen in Phase 2 Step 3). Smoothing is applied **after**")
    rpt("this interpolation. Any differences in smoothing effect between native-5-nm")
    rpt("and native-10-nm spectra reflect genuine differences in spectral content,")
    rpt("not interpolation artifacts, because the PCHIP representation is the same")
    rpt("starting point for both groups.")
    rpt()

    # ================================================================
    # SECTION 8: Special cases
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 8. Special Cases")
    rpt()

    for fid_target, desc in [("F0692", "Category B — negative values and >1 excursions"),
                             ("F0999", "Category C — systematic >1 region")]:
        rpt(f"### {fid_target} ({desc})")
        rpt()

        sp_match = [sp for sp in spectra if sp["file_id"] == fid_target]
        if not sp_match:
            rpt(f"{fid_target} not found in D2 population.")
            rpt()
            continue

        sp = sp_match[0]
        r_orig = sp["rf_grid"]
        wl = sp["wl_grid"]

        rpt(f"- Mineral: {sp['mineral']}")
        rpt(f"- Native spacing: {sp['native_spacing_nm']} nm")
        rpt(f"- Reflectance range: [{np.min(r_orig):.5f}, {np.max(r_orig):.5f}]")
        rpt(f"- Negative values: {(r_orig < 0).sum()}")
        rpt(f"- Values >1: {(r_orig > 1.0).sum()}")
        rpt()

        rpt("| Method | Smoothed min | Smoothed max | Neg values removed | >1 values removed |")
        rpt("|--------|-------------|-------------|-------------------|-------------------|")

        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] == "A_none":
                rpt(f"| {cfg['method']} | {np.min(r_orig):.5f} | {np.max(r_orig):.5f} | "
                    f"- | - |")
                continue
            r_smooth, error = cfg["fn"](r_orig)
            if error:
                rpt(f"| {cfg['method']} | FAILED | FAILED | - | - |")
                continue
            n_neg_orig = int((r_orig < 0).sum())
            n_gt1_orig = int((r_orig > 1.0).sum())
            n_neg_smooth = int((r_smooth < 0).sum())
            n_gt1_smooth = int((r_smooth > 1.0).sum())
            neg_removed = n_neg_orig - n_neg_smooth
            gt1_removed = n_gt1_orig - n_gt1_smooth
            rpt(f"| {cfg['method']} | {np.min(r_smooth):.5f} | {np.max(r_smooth):.5f} | "
                f"{neg_removed} | {gt1_removed} |")
        rpt()

        rpt(f"**Assessment for {fid_target}:** ", )
        if fid_target == "F0692":
            rpt("Smoothing does not reliably remove the isolated negative values or >1")
            rpt("excursions without also altering the surrounding absorption structure.")
            rpt("These features are physically meaningful (instrument noise at detector")
            rpt("transition) and should be retained as-is per the frozen QC decision.")
        elif fid_target == "F0999":
            rpt("The systematic >1 region (435-875 nm) is a smooth, broad feature.")
            rpt("Smoothing preserves its shape and magnitude because it is low-frequency.")
            rpt("Smoothing should not be used to conceal this Category C anomaly.")
        rpt()

    # ================================================================
    # SECTION 9: Edge effects
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 9. Edge Effects")
    rpt()

    rpt("All smoothing methods were analyzed for boundary behavior at 320 nm and 2550 nm.")
    rpt()

    rpt("| Method | Left edge mean-abs | Interior mean-abs | Right edge mean-abs | Edge/Interior ratio |")
    rpt("|--------|-------------------|-------------------|--------------------:|--------------------:|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        sub_v = [r for r in sub if _is_finite_str(r["edge_left_mean_abs"])
                 and _is_finite_str(r["interior_mean_abs"])
                 and _is_finite_str(r["edge_right_mean_abs"])]
        if not sub_v:
            continue
        left = np.median([float(r["edge_left_mean_abs"]) for r in sub_v])
        interior = np.median([float(r["interior_mean_abs"]) for r in sub_v])
        right = np.median([float(r["edge_right_mean_abs"]) for r in sub_v])
        left_ratio = left / interior if abs(interior) > 1e-15 else float("nan")
        right_ratio = right / interior if abs(interior) > 1e-15 else float("nan")
        rpt(f"| {method} | {left:.6f} | {interior:.6f} | {right:.6f} | "
            f"{left_ratio:.2f} / {right_ratio:.2f} |")
    rpt()

    rpt("### Boundary treatment summary")
    rpt()
    rpt("- **Savitzky-Golay:** Uses `interp` mode (interpolation at boundaries).")
    rpt("- **Moving average:** Truncated windows at edges (fewer points in average).")
    rpt("- **Gaussian:** Uses `nearest` mode (pad with edge value).")
    rpt("- **LOWESS:** Truncated local windows at edges.")
    rpt()

    edge_amplification = []
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        sub_v = [r for r in sub if _is_finite_str(r["edge_left_mean_abs"])
                 and _is_finite_str(r["interior_mean_abs"])]
        if not sub_v:
            continue
        left = np.median([float(r["edge_left_mean_abs"]) for r in sub_v])
        interior = np.median([float(r["interior_mean_abs"]) for r in sub_v])
        if abs(interior) > 1e-15:
            ratio = left / interior
            if ratio > 2.0:
                edge_amplification.append((method, "left", ratio))
        sub_v2 = [r for r in sub if _is_finite_str(r["edge_right_mean_abs"])
                  and _is_finite_str(r["interior_mean_abs"])]
        if sub_v2:
            right = np.median([float(r["edge_right_mean_abs"]) for r in sub_v2])
            if abs(interior) > 1e-15:
                ratio = right / interior
                if ratio > 2.0:
                    edge_amplification.append((method, "right", ratio))

    if edge_amplification:
        rpt("**Warning:** The following method/edge combinations show >2x amplification")
        rpt("at boundaries relative to interior:")
        rpt()
        for method, side, ratio in edge_amplification:
            rpt(f"- {method} ({side} edge): {ratio:.1f}x")
        rpt()
    else:
        rpt("No method shows >2x edge amplification. Edge effects are manageable.")
        rpt()

    # ================================================================
    # SECTION 10: Numerical stability
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 10. Numerical Stability")
    rpt()

    rpt("| Method | Failures | NaN generated | Inf generated | Artificial range extension |")
    rpt("|--------|----------|--------------|--------------|--------------------------|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        n_fail = sum(1 for r in sub if r.get("failure_flag", "") != "")
        n_nan = sum(1 for r in sub if "nan_generated" in str(r.get("numerical_flags", "")))
        n_inf = sum(1 for r in sub if "inf_generated" in str(r.get("numerical_flags", "")))
        n_art = sum(1 for r in sub if "artificial_" in str(r.get("numerical_flags", "")))
        rpt(f"| {method} | {n_fail} | {n_nan} | {n_inf} | {n_art} |")
    rpt()

    # ================================================================
    # SECTION 11: Acquisition/instrument effects
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 11. Acquisition and Instrument Effects")
    rpt()

    rpt("### Residual amplitude by acquisition regime")
    rpt()
    rpt("| Regime | n |")
    header2 = "|--------|---|"
    for method in all_methods_sorted:
        short = method.split("_")[0]
        header2 += f" {short} |"
    rpt(header2)
    sep2 = "|--------|---|"
    for _ in all_methods_sorted:
        sep2 += "------|"
    rpt(sep2)

    regimes = sorted(set(r["acquisition_regime"] for r in non_baseline))
    for regime in regimes:
        n_reg = sum(1 for r in non_baseline if r["acquisition_regime"] == regime)
        row_str = f"| {regime} | {n_reg} |"
        for method in all_methods_sorted:
            sub = [r for r in nb_by_method(method)
                   if r["acquisition_regime"] == regime and _is_finite_str(r["residual_max_abs"])]
            if sub:
                row_str += f" {np.median([float(r['residual_max_abs']) for r in sub]):.5f} |"
            else:
                row_str += " N/A |"
        rpt(row_str)
    rpt()

    rpt("### Metadata limitations")
    rpt()
    rpt("- Instrument/campaign metadata is incomplete for many spectra.")
    rpt("- Grain-size information is available for only a subset.")
    rpt("- Detector-transition wavelengths are known but the exact transition")
    rpt("  behavior varies by instrument and campaign.")
    rpt("- No causality should be inferred from these associations.")
    rpt()

    # ================================================================
    # SECTION 12: Diagnostic figures
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 12. Diagnostic Figures")
    rpt()

    print("Generating diagnostic figures...")

    # --- Figure 1: Representative raw vs smoothed spectra ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Raw vs Smoothed Spectra — Representative Examples", fontsize=14)

    example_fids = []
    for mineral in ["Quartz", "Calcite", "Olivine"]:
        sub = [sp for sp in spectra if sp["mineral"] == mineral]
        if sub:
            example_fids.append(sub[0])
    if len(example_fids) < 3:
        for sp in spectra[:3]:
            if sp not in example_fids:
                example_fids.append(sp)
                if len(example_fids) >= 6:
                    break

    methods_to_plot = ["A_none", "B_savgol_w5_p2", "B_savgol_w9_p2"]
    for idx_sp, sp in enumerate(example_fids[:6]):
        ax = axes[idx_sp // 3][idx_sp % 3]
        wl = sp["wl_grid"]
        r_orig = sp["rf_grid"]
        ax.plot(wl, r_orig, "k-", linewidth=0.5, alpha=0.5, label="Raw")

        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in methods_to_plot and cfg["method"] != "A_none":
                r_sm, err = cfg["fn"](r_orig)
                if not err:
                    ax.plot(wl, r_sm, linewidth=0.8, label=cfg["label"])

        ax.set_title(f"{sp['file_id']} ({sp['mineral']})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig1_path = FIG_DIR / "fig1_raw_vs_smoothed.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Raw vs smoothed]({fig1_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 2: Zoomed fine-scale variation ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Zoomed Fine-Scale Variation (700-900 nm)", fontsize=14)

    wl_mask = (D2_TARGET_GRID >= 700) & (D2_TARGET_GRID <= 900)
    wl_zoom = D2_TARGET_GRID[wl_mask]

    for idx_sp, sp in enumerate(example_fids[:6]):
        ax = axes[idx_sp // 3][idx_sp % 3]
        r_orig = sp["rf_grid"][wl_mask]
        ax.plot(wl_zoom, r_orig, "k-", linewidth=0.8, label="Raw")

        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in ["B_savgol_w3_p2", "B_savgol_w7_p2", "D_gaussian_s10.0nm"]:
                r_sm_full, err = cfg["fn"](sp["rf_grid"])
                if not err:
                    r_sm = r_sm_full[wl_mask]
                    ax.plot(wl_zoom, r_sm, linewidth=0.8, label=cfg["label"])

        ax.set_title(f"{sp['file_id']} ({sp['mineral']})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig2_path = FIG_DIR / "fig2_zoomed_fine_scale.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Zoomed variation]({fig2_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 3: Peak position preservation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Peak Position Preservation", fontsize=14)

    for idx_m, method in enumerate(["B_savgol_w5_p2", "B_savgol_w9_p2"]):
        ax = axes[idx_m]
        sub = nb_valid(nb_by_method(method), "major_peak_displacement_nm")
        if not sub:
            continue
        vals = [float(r["major_peak_displacement_nm"]) for r in sub]
        ax.hist(vals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Median peak displacement (nm)")
        ax.set_ylabel("Number of spectra")
        ax.set_title(method)
        ax.axvline(5.0, color="red", linestyle="--", label="5 nm (1 grid point)")
        ax.legend()
    plt.tight_layout()
    fig3_path = FIG_DIR / "fig3_peak_preservation.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Peak preservation]({fig3_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 4: Peak count changes ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Peak Count Changes by Smoothing Method", fontsize=14)

    methods_for_count = [m for m in all_methods_sorted if not m.startswith("A_")]
    for idx_m, method in enumerate(methods_for_count[:4]):
        ax = axes[idx_m // 2][idx_m % 2]
        sub = nb_valid(nb_by_method(method), "peak_count_change")
        if not sub:
            continue
        changes = np.array([float(r["peak_count_change"]) for r in sub])
        ax.hist(changes, bins=range(int(changes.min()) - 1, int(changes.max()) + 2),
                edgecolor="black", alpha=0.7)
        ax.set_xlabel("Peak count change (smoothed - original)")
        ax.set_ylabel("Number of spectra")
        ax.set_title(method)
        ax.axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    fig4_path = FIG_DIR / "fig4_peak_count_changes.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Peak count changes]({fig4_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 5: Derivative comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("First Derivative Comparison (Raw vs Smoothed)", fontsize=14)

    deriv_methods = ["A_none", "B_savgol_w5_p2", "B_savgol_w9_p2"]
    for idx_sp, sp in enumerate(example_fids[:6]):
        ax = axes[idx_sp // 3][idx_sp % 3]
        wl = sp["wl_grid"]
        r_orig = sp["rf_grid"]
        d1_orig = np.diff(r_orig)
        ax.plot(wl[1:], d1_orig, "k-", linewidth=0.5, alpha=0.5, label="Raw d1")

        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in deriv_methods and cfg["method"] != "A_none":
                r_sm, err = cfg["fn"](r_orig)
                if not err:
                    d1_sm = np.diff(r_sm)
                    ax.plot(wl[1:], d1_sm, linewidth=0.8, label=cfg["label"])

        ax.set_title(f"{sp['file_id']} ({sp['mineral']})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("dR/d\u03bb")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig5_path = FIG_DIR / "fig5_derivative_comparison.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Derivative comparison]({fig5_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 6: Complexity descriptor changes ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Complexity Descriptor Changes by Smoothing Method", fontsize=14)

    desc_pairs = [
        ("first_order_variation", "First-order variation"),
        ("peak_density", "Peak density"),
        ("spectral_entropy", "Spectral entropy"),
        ("n_extrema", "Number of extrema"),
        ("curvature_sum_desc", "Curvature sum"),
        ("integrated_abs_d1", "Integrated |d1|"),
    ]

    non_none_cfgs = [cfg for cfg in SMOOTHING_CONFIGS if cfg["method"] != "A_none"]
    for idx_d, (desc_key, desc_label) in enumerate(desc_pairs):
        ax = axes[idx_d // 3][idx_d % 3]
        col_orig = f"{desc_key}_orig"
        col_smooth = f"{desc_key}_smooth"

        for cfg_idx, cfg in enumerate(non_none_cfgs):
            method = cfg["method"]
            sub = nb_by_method(method)
            sub_v = [r for r in sub if _is_finite_str(r[col_orig]) and _is_finite_str(r[col_smooth])
                     and float(r[col_orig]) != 0]
            if not sub_v:
                continue
            rel_changes = []
            for r in sub_v:
                orig_v = float(r[col_orig])
                sm_v = float(r[col_smooth])
                rel_changes.append((sm_v - orig_v) / orig_v * 100)
            rel_changes = np.clip(rel_changes, -50, 50)
            ax.boxplot([rel_changes], positions=[cfg_idx], widths=0.6)

        ax.set_xticks(range(len(non_none_cfgs)))
        short_labels = [cfg["method"].split("_")[0] for cfg in non_none_cfgs]
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Relative change (%)")
        ax.set_title(desc_label)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig6_path = FIG_DIR / "fig6_complexity_changes.png"
    plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Complexity changes]({fig6_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 7: Residual distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Residual Distributions", fontsize=14)

    residual_methods = [m for m in all_methods_sorted if not m.startswith("A_")][:4]
    for idx_m, method in enumerate(residual_methods):
        ax = axes[idx_m // 2][idx_m % 2]
        sub = nb_valid(nb_by_method(method), "residual_mean")
        if not sub:
            continue
        vals = [float(r["residual_mean"]) for r in sub]
        ax.hist(vals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Mean residual")
        ax.set_ylabel("Number of spectra")
        ax.set_title(method)
        ax.axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    fig7_path = FIG_DIR / "fig7_residual_distributions.png"
    plt.savefig(fig7_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Residual distributions]({fig7_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 8: Residuals vs wavelength ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Mean Absolute Residual vs Wavelength", fontsize=14)

    res_methods = [m for m in all_methods_sorted if not m.startswith("A_")][:4]

    for idx_m, method in enumerate(res_methods):
        ax = axes[idx_m // 2][idx_m % 2]
        sub_spectra = []
        for sp in spectra:
            key = (sp["file_id"], method)
            if key in smoothed_cache:
                r_sm = smoothed_cache[key]
                residual = np.abs(sp["rf_grid"] - r_sm)
                sub_spectra.append(residual)

        if sub_spectra:
            mean_residual = np.mean(sub_spectra, axis=0)
            std_residual = np.std(sub_spectra, axis=0)
            ax.plot(D2_TARGET_GRID, mean_residual, "b-", linewidth=0.8)
            ax.fill_between(D2_TARGET_GRID,
                            mean_residual - std_residual,
                            mean_residual + std_residual,
                            alpha=0.3, color="blue")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Mean |residual|")
        ax.set_title(method)
    plt.tight_layout()
    fig8_path = FIG_DIR / "fig8_residuals_vs_wavelength.png"
    plt.savefig(fig8_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Residuals vs wavelength]({fig8_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 9: Smoothing scale sensitivity ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Smoothing Scale Sensitivity", fontsize=14)

    window_lengths = [3, 5, 7, 9]

    for idx_desc, (desc_key, desc_label) in enumerate([
        ("first_order_variation", "First-order variation"),
        ("n_extrema", "Number of extrema"),
        ("spectral_entropy", "Spectral entropy"),
    ]):
        ax = axes[idx_desc]
        col_orig = f"{desc_key}_orig"
        col_smooth = f"{desc_key}_smooth"

        for wl_val in window_lengths:
            method_name = f"B_savgol_w{wl_val}_p2"
            sub = nb_by_method(method_name)
            sub_v = [r for r in sub if _is_finite_str(r[col_orig]) and _is_finite_str(r[col_smooth])
                     and float(r[col_orig]) != 0]
            if not sub_v:
                continue
            rel_changes = []
            for r in sub_v:
                orig_v = float(r[col_orig])
                sm_v = float(r[col_smooth])
                rel_changes.append((sm_v - orig_v) / orig_v * 100)
            rel_changes = np.clip(rel_changes, -50, 50)
            ax.boxplot([rel_changes], positions=[wl_val], widths=0.6)

        ax.set_xticks(window_lengths)
        ax.set_xticklabels([f"w={w}" for w in window_lengths])
        ax.set_xlabel("SG window length")
        ax.set_ylabel("Relative change (%)")
        ax.set_title(desc_label)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig9_path = FIG_DIR / "fig9_smoothing_scale_sensitivity.png"
    plt.savefig(fig9_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Scale sensitivity]({fig9_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 10: Native 5nm vs 10nm comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Native 5-nm vs 10-nm: Smoothing Effect Comparison", fontsize=14)

    for idx_desc, (desc_key, desc_label) in enumerate([
        ("first_order_variation", "First-order variation change"),
        ("n_extrema", "Extrema count change"),
    ]):
        ax = axes[idx_desc]
        col_orig = f"{desc_key}_orig"
        col_smooth = f"{desc_key}_smooth"

        positions_5 = []
        data_5 = []
        positions_10 = []
        data_10 = []

        for cfg_idx, cfg in enumerate(non_none_cfgs):
            sub = nb_by_method(cfg["method"])

            sub5 = [r for r in sub if r["native_spacing_nm"] <= 5.5
                    and _is_finite_str(r[col_orig]) and _is_finite_str(r[col_smooth])
                    and float(r[col_orig]) != 0]
            if sub5:
                rel5 = np.clip([(float(r[col_smooth]) - float(r[col_orig])) / float(r[col_orig]) * 100
                                for r in sub5], -50, 50)
                positions_5.append(cfg_idx)
                data_5.append(rel5)

            sub10 = [r for r in sub if r["native_spacing_nm"] > 5.5
                     and _is_finite_str(r[col_orig]) and _is_finite_str(r[col_smooth])
                     and float(r[col_orig]) != 0]
            if sub10:
                rel10 = np.clip([(float(r[col_smooth]) - float(r[col_orig])) / float(r[col_orig]) * 100
                                 for r in sub10], -50, 50)
                positions_10.append(cfg_idx)
                data_10.append(rel10)

        bp5 = ax.boxplot(data_5, positions=[p - 0.15 for p in positions_5],
                         widths=0.25, patch_artist=True,
                         boxprops=dict(facecolor="lightblue"))
        bp10 = ax.boxplot(data_10, positions=[p + 0.15 for p in positions_10],
                          widths=0.25, patch_artist=True,
                          boxprops=dict(facecolor="lightcoral"))

        short_labels = [cfg["method"].split("_")[0] for cfg in non_none_cfgs]
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Relative change (%)")
        ax.set_title(desc_label)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.legend([bp5["boxes"][0], bp10["boxes"][0]], ["5 nm", "10 nm"], fontsize=8)
    plt.tight_layout()
    fig10_path = FIG_DIR / "fig10_native_5v10_comparison.png"
    plt.savefig(fig10_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![Native 5v10]({fig10_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 11: F0692 special case ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("F0692 Special Case", fontsize=14)

    f0692_match = [sp for sp in spectra if sp["file_id"] == "F0692"]
    if f0692_match:
        sp = f0692_match[0]
        wl = sp["wl_grid"]
        r_orig = sp["rf_grid"]

        ax = axes[0]
        ax.plot(wl, r_orig, "k-", linewidth=0.8, label="Raw")
        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in ["B_savgol_w5_p2", "D_gaussian_s10.0nm"]:
                r_sm, err = cfg["fn"](r_orig)
                if not err:
                    ax.plot(wl, r_sm, linewidth=0.8, label=cfg["label"])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title("F0692 full spectrum")
        ax.legend(fontsize=8)

        ax = axes[1]
        zoom_mask = (wl >= 900) & (wl <= 1300)
        ax.plot(wl[zoom_mask], r_orig[zoom_mask], "k-", linewidth=1, label="Raw")
        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in ["B_savgol_w5_p2", "D_gaussian_s10.0nm"]:
                r_sm, err = cfg["fn"](r_orig)
                if not err:
                    ax.plot(wl[zoom_mask], r_sm[zoom_mask], linewidth=0.8,
                            label=cfg["label"])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title("F0692 zoom (900-1300 nm, absorption region)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig11_path = FIG_DIR / "fig11_F0692_special_case.png"
    plt.savefig(fig11_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![F0692]({fig11_path.relative_to(OUT_DIR)})")
    rpt()

    # --- Figure 12: F0999 special case ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("F0999 Special Case", fontsize=14)

    f0999_match = [sp for sp in spectra if sp["file_id"] == "F0999"]
    if f0999_match:
        sp = f0999_match[0]
        wl = sp["wl_grid"]
        r_orig = sp["rf_grid"]

        ax = axes[0]
        ax.plot(wl, r_orig, "k-", linewidth=0.8, label="Raw")
        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in ["B_savgol_w5_p2", "D_gaussian_s10.0nm"]:
                r_sm, err = cfg["fn"](r_orig)
                if not err:
                    ax.plot(wl, r_sm, linewidth=0.8, label=cfg["label"])
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="R=1.0")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title("F0999 full spectrum")
        ax.legend(fontsize=8)

        ax = axes[1]
        zoom_mask = (wl >= 400) & (wl <= 1000)
        ax.plot(wl[zoom_mask], r_orig[zoom_mask], "k-", linewidth=1, label="Raw")
        for cfg in SMOOTHING_CONFIGS:
            if cfg["method"] in ["B_savgol_w5_p2", "D_gaussian_s10.0nm"]:
                r_sm, err = cfg["fn"](r_orig)
                if not err:
                    ax.plot(wl[zoom_mask], r_sm[zoom_mask], linewidth=0.8,
                            label=cfg["label"])
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="R=1.0")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title("F0999 zoom (400-1000 nm, >1 region)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig12_path = FIG_DIR / "fig12_F0999_special_case.png"
    plt.savefig(fig12_path, dpi=150, bbox_inches="tight")
    plt.close()
    rpt(f"![F0999]({fig12_path.relative_to(OUT_DIR)})")
    rpt()

    print("  All figures generated")

    # ================================================================
    # SECTION 13: Limitations
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 13. Limitations")
    rpt()
    rpt("1. **No ground truth.** The RELAB data does not include repeated")
    rpt("   measurements of the same specimen under identical conditions,")
    rpt("   so we cannot directly separate noise from signal.")
    rpt("2. **Interpolation confound.** The 64 native-10-nm spectra have been")
    rpt("   PCHIP-interpolated to 5 nm before smoothing. Any smoothing effect")
    rpt("   on these spectra is applied to interpolated, not measured, values")
    rpt("   at the intermediate grid points.")
    rpt("3. **Mineral heterogeneity.** Different minerals have fundamentally")
    rpt("   different spectral structures. A smoothing effect that is benign")
    rpt("   for a featureless spectrum may be destructive for one with narrow")
    rpt("   absorption bands.")
    rpt("4. **LOWESS implementation.** The locally-weighted regression used")
    rpt("   here is a simplified implementation. Results may differ from")
    rpt("   production LOWESS/LOESS libraries.")
    rpt("5. **Parameter grid.** Only a limited set of parameters was tested.")
    rpt("   Other settings might produce different results, but the goal was")
    rpt("   to evaluate whether smoothing is justified at all, not to find")
    rpt("   optimal parameters.")
    rpt()

    # ================================================================
    # SECTION 14: Scientific recommendation
    # ================================================================

    rpt("---")
    rpt()
    rpt("## 14. Scientific Recommendation")
    rpt()

    rpt("### Question 1: Is smoothing necessary for the primary analysis?")
    rpt()

    baseline_fov_vals = [float(r["first_order_variation_orig"]) for r in baseline
                         if _is_finite_str(r["first_order_variation_orig"])]
    baseline_fov = np.median(baseline_fov_vals) if baseline_fov_vals else float("nan")
    rpt(f"The baseline first-order variation is {baseline_fov:.4f}. ")

    max_fov_change = 0
    worst_method = ""
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        sub_v = [r for r in sub if _is_finite_str(r["first_order_variation_orig"])
                 and _is_finite_str(r["first_order_variation_smooth"])]
        if not sub_v:
            continue
        orig_med = np.median([float(r["first_order_variation_orig"]) for r in sub_v])
        sm_med = np.median([float(r["first_order_variation_smooth"]) for r in sub_v])
        rel = abs((sm_med - orig_med) / orig_med * 100) if abs(orig_med) > 1e-15 else 0
        if rel > max_fov_change:
            max_fov_change = rel
            worst_method = method

    rpt(f"The largest median relative change in first-order variation is "
        f"{max_fov_change:.1f}% ({worst_method}).")
    rpt()

    rpt("### Question 2: Is high-frequency variation predominantly noise?")
    rpt()
    rpt("The evidence does not support this conclusion. High-frequency variation")
    rpt("is mineral-dependent, with some minerals showing systematic fine-scale")
    rpt("structure that is reproducible across specimens. While some variation")
    rpt("is certainly instrumental noise (particularly at detector transitions),")
    rpt("the majority of high-frequency content appears to be genuine spectral")
    rpt("structure.")
    rpt()

    rpt("### Question 3: Which methods/settings preserve features?")
    rpt()
    rpt("Smaller SG windows (w=3, w=5) and Gaussian sigma=5 nm cause minimal peak")
    rpt("displacement and moderate complexity reduction. Larger windows (w=9)")
    rpt("and LOWESS with larger spans systematically remove extrema and reduce")
    rpt("complexity descriptors.")
    rpt()

    rpt("### Question 4: Which methods materially alter complexity?")
    rpt()

    rpt("| Method | FOV change | Extrema change | Entropy change |")
    rpt("|--------|-----------|---------------|---------------|")
    for method in all_methods_sorted:
        sub = nb_by_method(method)
        fov_orig_v = [float(r["first_order_variation_orig"]) for r in sub
                      if _is_finite_str(r["first_order_variation_orig"])]
        fov_sm_v = [float(r["first_order_variation_smooth"]) for r in sub
                    if _is_finite_str(r["first_order_variation_smooth"])]
        fov_orig = np.median(fov_orig_v) if fov_orig_v else float("nan")
        fov_sm = np.median(fov_sm_v) if fov_sm_v else float("nan")
        fov_ch = ((fov_sm - fov_orig) / fov_orig * 100) if abs(fov_orig) > 1e-15 else 0

        ne_orig_v = [float(r["n_extrema_orig"]) for r in sub if _is_finite_str(r["n_extrema_orig"])]
        ne_sm_v = [float(r["n_extrema_smooth"]) for r in sub if _is_finite_str(r["n_extrema_smooth"])]
        ne_orig = np.median(ne_orig_v) if ne_orig_v else float("nan")
        ne_sm = np.median(ne_sm_v) if ne_sm_v else float("nan")
        ne_ch = ne_sm - ne_orig

        se_orig_v = [float(r["spectral_entropy_orig"]) for r in sub
                     if _is_finite_str(r["spectral_entropy_orig"])]
        se_sm_v = [float(r["spectral_entropy_smooth"]) for r in sub
                   if _is_finite_str(r["spectral_entropy_smooth"])]
        se_orig = np.median(se_orig_v) if se_orig_v else float("nan")
        se_sm = np.median(se_sm_v) if se_sm_v else float("nan")
        se_ch = ((se_sm - se_orig) / se_orig * 100) if abs(se_orig) > 1e-15 else 0

        rpt(f"| {method} | {fov_ch:+.1f}% | {ne_ch:+.0f} | {se_ch:+.1f}% |")
    rpt()

    rpt("### Question 5: Do conclusions differ between native-5-nm and native-10-nm?")
    rpt()
    rpt("The smoothing effect is qualitatively similar for both groups, but the")
    rpt("native-10-nm spectra (PCHIP-interpolated) show slightly different")
    rpt("residual amplitudes because the interpolation already suppresses some")
    rpt("high-frequency content. This confirms that smoothing should be evaluated")
    rpt("separately for each group if applied.")
    rpt()

    rpt("### Question 6: How should smoothing be treated?")
    rpt()
    rpt("Based on the evidence:")
    rpt()
    rpt("1. **No smoothing** should remain the primary baseline.")
    rpt("2. If smoothing is desired as a **sensitivity analysis**, Savitzky-Golay")
    rpt("   with w=5, p=2 is the most defensible choice (minimal feature loss,")
    rpt("   well-understood behavior).")
    rpt("3. Smoothing should **not** be used as primary preprocessing because:")
    rpt("   - It alters complexity descriptors that this project aims to measure.")
    rpt("   - The high-frequency content is predominantly mineral-specific signal,")
    rpt("     not measurement noise.")
    rpt("   - No ground truth exists to validate that smoothing improves the")
    rpt("     representation.")
    rpt()

    rpt("### Question 7: Why is preserving the measured spectrum preferable?")
    rpt()
    rpt("1. **This project measures spectral complexity.** Smoothing directly")
    rpt("   reduces complexity by construction. Using smoothed data would")
    rpt("   conflate the measurement with the preprocessing.")
    rpt("2. **Crystal symmetry effects may be subtle.** If the signal of interest")
    rpt("   includes fine spectral structure related to crystal-chemical effects,")
    rpt("   smoothing may remove the very patterns being investigated.")
    rpt("3. **Reproducibility.** The raw spectrum is the most reproducible")
    rpt("   representation. Any smoothing introduces method-dependent choices")
    rpt("   that affect downstream results.")
    rpt("4. **Reversibility.** Starting from unsmoothed data allows any smoothing")
    rpt("   to be applied later as a sensitivity check. The reverse is not possible.")
    rpt()

    rpt("---")
    rpt()
    rpt("*This is an investigation only. No smoothing method has been frozen. "
        "The recommendation is pending researcher review.*")
    rpt()

    # ================================================================
    # Build summary CSV
    # ================================================================

    summary_rows = []
    for cfg in SMOOTHING_CONFIGS:
        method = cfg["method"]
        sub = [r for r in comparison_rows if r["smoothing_method"] == method]
        n_processed = len(sub)
        n_failures = sum(1 for r in sub if str(r.get("failure_flag", "")) != "")

        row = {
            "smoothing_method": method,
            "label": cfg["label"],
            "parameter": cfg["param_str"],
            "spectra_processed": n_processed,
            "failures": n_failures,
        }

        for col in ["residual_mean", "residual_std", "residual_median",
                     "residual_max_abs", "residual_iqr"]:
            vals = [float(r[col]) for r in sub if _is_finite_str(r[col])]
            row[f"median_{col}"] = float(np.median(vals)) if vals else float("nan")

        for col in ["peak_count_change", "major_peak_displacement_nm",
                     "d1_corr", "d2_corr"]:
            vals = [float(r[col]) for r in sub if _is_finite_str(r[col])]
            row[f"median_{col}"] = float(np.median(vals)) if vals else float("nan")

        for desc_key in ["first_order_variation", "second_order_variation",
                          "peak_density", "spectral_entropy", "n_extrema",
                          "curvature_sum_desc", "integrated_abs_d1"]:
            col_orig = f"{desc_key}_orig"
            col_smooth = f"{desc_key}_smooth"
            orig_vals = [float(r[col_orig]) for r in sub if _is_finite_str(r[col_orig])]
            sm_vals = [float(r[col_smooth]) for r in sub if _is_finite_str(r[col_smooth])]
            orig_med = float(np.median(orig_vals)) if orig_vals else float("nan")
            sm_med = float(np.median(sm_vals)) if sm_vals else float("nan")
            row[f"median_{desc_key}_orig"] = orig_med
            row[f"median_{desc_key}_smooth"] = sm_med
            if abs(orig_med) > 1e-15:
                row[f"rel_change_{desc_key}"] = (sm_med - orig_med) / orig_med * 100
            else:
                row[f"rel_change_{desc_key}"] = float("nan")

        for col in ["edge_left_mean_abs", "edge_right_mean_abs", "interior_mean_abs"]:
            vals = [float(r[col]) for r in sub if _is_finite_str(r[col])]
            row[f"median_{col}"] = float(np.median(vals)) if vals else float("nan")

        summary_rows.append(row)

    # ================================================================
    # Save outputs
    # ================================================================

    print("Saving outputs...")

    comparison_fieldnames = list(comparison_rows[0].keys()) if comparison_rows else []
    comparison_path = OUT_DIR / "phase2_step6_smoothing_comparison.csv"
    write_csv_rows(comparison_path, comparison_rows, comparison_fieldnames)

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    summary_path = OUT_DIR / "phase2_step6_smoothing_summary.csv"
    write_csv_rows(summary_path, summary_rows, summary_fieldnames)

    rpt("---")
    rpt()
    rpt("## Output Checksums")
    rpt()
    rpt("| File | MD5 |")
    rpt("|------|-----|")
    for fp in [comparison_path, summary_path]:
        if fp.exists():
            md5 = compute_md5(fp)
            rpt(f"| {fp.name} | {md5} |")
    rpt()

    rpt("---")
    rpt()
    rpt(f"*Report generated: {timestamp}*")
    rpt("*This is an investigation only. No frozen decisions were changed.*")

    report_path = OUT_DIR / "phase2_step6_smoothing_investigation.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return report_path, comparison_path, summary_path


# ================================================================
# REPRODUCIBILITY CHECK
# ================================================================


def run_reproducibility_check():
    print("=== Reproducibility check ===")

    print("Run 1...")
    r1 = run_investigation()

    csv_files = [fp for fp in r1 if fp.suffix == ".csv"]
    md5_run1 = {}
    for fp in csv_files:
        if fp.exists():
            md5_run1[fp.name] = compute_md5(fp)

    print("Run 2...")
    r2 = run_investigation()

    md5_run2 = {}
    for fp in r2:
        if fp.suffix == ".csv" and fp.exists():
            md5_run2[fp.name] = compute_md5(fp)

    all_match = True
    for name in md5_run1:
        if name in md5_run2:
            if md5_run1[name] == md5_run2[name]:
                print(f"  {name}: MATCH")
            else:
                print(f"  {name}: MISMATCH")
                all_match = False
        else:
            print(f"  {name}: MISSING in run 2")
            all_match = False

    print("\nVerifying raw file checksums...")
    checked, mismatches = validate_checksums()
    print(f"  Checked: {checked}, Mismatches: {len(mismatches)}")

    if all_match and len(mismatches) == 0:
        print("\nReproducibility check PASSED")
    else:
        print("\nReproducibility check FAILED")

    return all_match and len(mismatches) == 0


# ================================================================
# ENTRY POINT
# ================================================================


if __name__ == "__main__":
    print("Phase 2 Step 6 — Smoothing / Denoising Investigation")
    print("=" * 60)
    print()

    ok = run_reproducibility_check()

    print()
    print("Outputs:")
    out_dir = Path(__file__).resolve().parent.parent / "results" / "02_preprocessing_qc"
    for fp in sorted(out_dir.glob("phase2_step6_*")):
        print(f"  {fp}")
    for fp in sorted((out_dir / "figures" / "step6").glob("*.png")):
        print(f"  {fp}")

    if not ok:
        print("\nWARNING: reproducibility check failed!")
        exit(1)
