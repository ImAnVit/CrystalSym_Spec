"""
Phase 2 Step 3 — Splice Boundary Investigation

Investigates all documented splice boundaries in the 925 RELAB analysis candidates.
Produces:
  - phase2_step3_splice_boundary_investigation.csv (one row per D2-relevant splice)
  - phase2_step3_splice_boundary_investigation.md (full report)

Does NOT modify raw files. Does NOT implement preprocessing rules.
"""

import hashlib
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RELAB"
POPULATION_CSV = PROJECT_ROOT / "metadata" / "relab_analysis_population.csv"
CHECKSUMS_CSV = PROJECT_ROOT / "metadata" / "provenance" / "relab_raw_checksums.csv"
OUT_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc"

D2_MIN = 320.0
D2_MAX = 2550.0
TARGET_SPACING = 5.0

# Analysis windows
WIDE_WINDOW_NM = 25.0  # ±25 nm for context
NARROW_WINDOW_NM = 5.0  # ±5 nm for local analysis


def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_relab_full(filepath):
    """Parse RELAB file into segments and footer."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0
    segments = []
    data_ended = False

    while idx < len(lines) and not data_ended:
        line = lines[idx].strip()

        # Skip blank lines at the start
        if not line:
            idx += 1
            continue

        # Try to parse as segment count
        try:
            n = int(line)
            if len(line.split()) == 1 and n > 0:
                idx += 1
                seg_wl, seg_rf, seg_unc = [], [], []
                for _ in range(n):
                    if idx >= len(lines):
                        break
                    parts = lines[idx].strip().split()
                    if len(parts) >= 2:
                        try:
                            seg_wl.append(float(parts[0]))
                            seg_rf.append(float(parts[1]))
                            seg_unc.append(float(parts[2]) if len(parts) > 2 else None)
                        except ValueError:
                            # Not a data line, must be footer
                            data_ended = True
                            break
                    else:
                        # Not a data line
                        data_ended = True
                        break
                    idx += 1

                if not data_ended and len(seg_wl) == n:
                    segments.append((np.array(seg_wl), np.array(seg_rf), np.array(seg_unc)))
                continue
        except ValueError:
            # Not a number, must be footer
            data_ended = True
            continue

        idx += 1

    # Everything remaining is footer
    footer_lines = [l.strip() for l in lines[idx:] if l.strip()]
    return segments, footer_lines


def extract_splice_wavelengths(footer_lines):
    """
    Extract splice wavelengths from footer metadata.
    Returns list of (wavelength_nm, raw_text) tuples.
    """
    splice_wavelengths = []

    for line in footer_lines:
        line_lower = line.lower()

        # Pattern 1: "Connected at X & Y um" or "Connected at X, Y nm"
        if "connected at" in line_lower:
            # Check if unit is um/μm or nm
            unit_match = re.search(r'(um|nm|μm)', line_lower)
            if unit_match:
                unit = unit_match.group(1)
                # Extract all numbers before the unit
                # Remove the "connected at" prefix
                after_prefix = re.sub(r'connected\s+at\s+', '', line_lower)
                # Remove the unit suffix
                before_unit = re.sub(r'\s*(um|nm|μm).*', '', after_prefix)
                # Split by comma, ampersand, or whitespace
                parts = re.split(r'[,&\s]+', before_unit.strip())

                for part in parts:
                    part = part.strip()
                    if part:
                        try:
                            val = float(part)
                            if unit in ['um', 'μm']:
                                val_nm = val * 1000.0
                            else:
                                val_nm = val
                            splice_wavelengths.append((val_nm, line))
                        except ValueError:
                            pass
            else:
                # No unit specified, check for plain numbers (assume nm if > 100)
                plain_matches = re.findall(r'connected at\s+([\d.,\s&]+)', line_lower)
                for match_str in plain_matches:
                    parts = re.split(r'[,&\s]+', match_str.strip())
                    for part in parts:
                        part = part.strip()
                        if part and part not in ['at', 'and']:
                            try:
                                val = float(part)
                                if val > 100:  # Assume nm
                                    splice_wavelengths.append((val, line))
                            except ValueError:
                                pass

        # Pattern 2: "spliced X nm" or "spliced @ X"
        elif "spliced" in line_lower:
            # Check for explicit unit
            matches = re.findall(r'([\d.]+)\s*(nm)', line_lower)
            if matches:
                for val_str, unit in matches:
                    try:
                        val = float(val_str)
                        splice_wavelengths.append((val, line))
                    except ValueError:
                        pass
            else:
                # No unit, look for numbers after @ or "spliced"
                matches = re.findall(r'[@\s]\s*([\d.]+)', line_lower)
                for val_str in matches:
                    try:
                        val = float(val_str)
                        if val > 100:  # Assume nm
                            splice_wavelengths.append((val, line))
                    except ValueError:
                        pass

    return splice_wavelengths


def find_nearest_indices(wavelengths, target_wl, window_nm):
    """Find indices within window of target wavelength."""
    mask = np.abs(wavelengths - target_wl) <= window_nm
    return np.where(mask)[0]


def compute_local_metrics(wl, rf, splice_wl):
    """
    Compute local spectral metrics around a splice boundary.
    Returns dict of metrics.
    """
    # Find points immediately before and after splice
    before_mask = wl < splice_wl
    after_mask = wl > splice_wl

    if not np.any(before_mask) or not np.any(after_mask):
        return None

    # Get points near the boundary
    before_idx = np.where(before_mask)[0]
    after_idx = np.where(after_mask)[0]

    # Last point before splice
    before_idx_last = before_idx[-1]
    wl_before = wl[before_idx_last]
    rf_before = rf[before_idx_last]

    # First point after splice
    after_idx_first = after_idx[0]
    wl_after = wl[after_idx_first]
    rf_after = rf[after_idx_first]

    # Absolute discontinuity
    abs_disc = abs(rf_after - rf_before)

    # Relative discontinuity (avoid division by zero)
    mean_rf = (rf_before + rf_after) / 2.0
    rel_disc = abs_disc / mean_rf if mean_rf > 0 else np.nan

    # Local slopes (using ±5 nm window)
    wide_before_idx = find_nearest_indices(wl, splice_wl, NARROW_WINDOW_NM)
    wide_before_idx = wide_before_idx[wl[wide_before_idx] < splice_wl]
    wide_after_idx = find_nearest_indices(wl, splice_wl, NARROW_WINDOW_NM)
    wide_after_idx = wide_after_idx[wl[wide_after_idx] > splice_wl]

    slope_before = np.nan
    slope_after = np.nan

    if len(wide_before_idx) >= 2:
        wl_seg = wl[wide_before_idx]
        rf_seg = rf[wide_before_idx]
        if np.std(wl_seg) > 0:
            slope_before = np.polyfit(wl_seg, rf_seg, 1)[0]

    if len(wide_after_idx) >= 2:
        wl_seg = wl[wide_after_idx]
        rf_seg = rf[wide_after_idx]
        if np.std(wl_seg) > 0:
            slope_after = np.polyfit(wl_seg, rf_seg, 1)[0]

    slope_change = np.nan
    if not np.isnan(slope_before) and not np.isnan(slope_after):
        slope_change = abs(slope_after - slope_before)

    # Local variation in ±25 nm window (excluding splice point itself)
    wide_window_idx = find_nearest_indices(wl, splice_wl, WIDE_WINDOW_NM)
    if len(wide_window_idx) >= 3:
        rf_window = rf[wide_window_idx]
        local_variation = np.std(rf_window)
        local_range = np.ptp(rf_window)
    else:
        local_variation = np.nan
        local_range = np.nan

    # Compare discontinuity to local variation
    disc_vs_variation = np.nan
    if not np.isnan(local_variation) and local_variation > 0:
        disc_vs_variation = abs_disc / local_variation

    return {
        'wl_before': wl_before,
        'rf_before': rf_before,
        'wl_after': wl_after,
        'rf_after': rf_after,
        'wl_gap': wl_after - wl_before,
        'abs_disc': abs_disc,
        'rel_disc': rel_disc,
        'slope_before': slope_before,
        'slope_after': slope_after,
        'slope_change': slope_change,
        'local_variation': local_variation,
        'local_range': local_range,
        'disc_vs_variation': disc_vs_variation,
    }


def classify_splice(metrics):
    """
    Classify splice based on evidence.
    Returns (category, explanation).
    """
    if metrics is None:
        return ('INSUFFICIENT', 'Could not compute local metrics')

    abs_disc = metrics['abs_disc']
    rel_disc = metrics['rel_disc']
    disc_vs_var = metrics['disc_vs_variation']
    slope_change = metrics['slope_change']

    # Category A: No material discontinuity
    if abs_disc < 0.005 and (np.isnan(disc_vs_var) or disc_vs_var < 1.0):
        return ('A', f'Negligible discontinuity (abs_disc={abs_disc:.4f})')

    # Category D: Potentially problematic
    if (not np.isnan(disc_vs_var) and disc_vs_var > 3.0) or \
       (not np.isnan(rel_disc) and rel_disc > 0.10):
        return ('D', f'Large discontinuity (abs_disc={abs_disc:.4f}, ratio={disc_vs_var:.2f})')

    # Category C: Ambiguous
    if (not np.isnan(disc_vs_var) and disc_vs_var > 1.5) or \
       (not np.isnan(rel_disc) and rel_disc > 0.05):
        return ('C', f'Moderate discontinuity (abs_disc={abs_disc:.4f}, ratio={disc_vs_var:.2f})')

    # Category B: Measurable but small
    return ('B', f'Small discontinuity (abs_disc={abs_disc:.4f}, ratio={disc_vs_var:.2f})')


def diagnostic_interpolation_check(wl, rf, splice_wl):
    """
    Diagnostic: check what would happen with 5-nm interpolation near splice.
    Returns dict of diagnostic metrics.
    """
    # Find nearest native points around splice
    before_mask = wl < splice_wl
    after_mask = wl > splice_wl

    if not np.any(before_mask) or not np.any(after_mask):
        return None

    wl_before = wl[before_mask][-1]
    rf_before = rf[before_mask][-1]
    wl_after = wl[after_mask][0]
    rf_after = rf[after_mask][0]

    # Check if 5-nm grid points would fall in the gap
    gap = wl_after - wl_before
    if gap < 0.1:  # No significant gap
        return {'gap_nm': gap, 'would_interpolate': False}

    # Count how many 5-nm points would fall in gap
    grid_start = np.ceil(wl_before / 5.0) * 5.0
    grid_end = np.floor(wl_after / 5.0) * 5.0
    n_grid_in_gap = max(0, int((grid_end - grid_start) / 5.0) + 1)

    # Linear interpolation would span the discontinuity
    # Estimate interpolation error as fraction of discontinuity
    abs_disc = abs(rf_after - rf_before)

    return {
        'gap_nm': gap,
        'would_interpolate': n_grid_in_gap > 0,
        'n_grid_points_in_gap': n_grid_in_gap,
        'abs_disc': abs_disc,
        'max_interp_error_est': abs_disc if n_grid_in_gap > 0 else 0.0,
    }


def main():
    print("=" * 80)
    print("Phase 2 Step 3 — Splice Boundary Investigation")
    print("=" * 80)

    # Load population
    pop = pd.read_csv(POPULATION_CSV)
    candidates = pop[pop["inclusion_category"].isin(["A", "B"])].copy()
    print(f"\nAnalysis candidates: {len(candidates)}")

    # Pre-flight checksum verification
    print("\nVerifying raw checksums (pre-flight)...")
    ref = pd.read_csv(CHECKSUMS_CSV)
    pre_mismatches = 0
    for _, row in ref.iterrows():
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = compute_md5(fp)
            if md5 != row["md5"]:
                pre_mismatches += 1
                print(f"  MISMATCH: {row['rel_path']}")
    print(f"  Checked: {len(ref)}, Mismatches: {pre_mismatches}")
    assert pre_mismatches == 0, "Raw file checksum mismatch detected!"
    print("  [OK] All checksums verified")

    # ========================================================================
    # TASK 1: Identify all documented splice metadata
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: Identifying all documented splice metadata")
    print("=" * 80)

    all_splice_records = []

    for idx, row in candidates.iterrows():
        file_id = row['file_id']
        raw_path = DATA_DIR / row['raw_path']
        mineral = row['mineral_directory']
        sample_id = row['relab_sample_id']
        prov_group = row['specimen_group_id']

        if not raw_path.exists():
            continue

        segments, footer = parse_relab_full(raw_path)

        # Concatenate all segments
        wl_all = np.concatenate([s[0] for s in segments])
        rf_all = np.concatenate([s[1] for s in segments])

        # Extract splice wavelengths
        splice_info = extract_splice_wavelengths(footer)

        if splice_info:
            for splice_wl, raw_text in splice_info:
                # Determine D2 relevance
                if splice_wl < D2_MIN:
                    d2_relevance = 'below_D2'
                elif splice_wl > D2_MAX:
                    d2_relevance = 'above_D2'
                else:
                    d2_relevance = 'inside_D2'

                # Check if within ±50 nm of D2 boundaries (potentially relevant)
                if D2_MIN - 50 <= splice_wl <= D2_MIN:
                    d2_relevance = 'near_D2_below'
                elif D2_MAX <= splice_wl <= D2_MAX + 50:
                    d2_relevance = 'near_D2_above'

                all_splice_records.append({
                    'file_id': file_id,
                    'mineral': mineral,
                    'sample_id': sample_id,
                    'provenance_group': prov_group,
                    'splice_wl_nm': splice_wl,
                    'd2_relevance': d2_relevance,
                    'raw_splice_text': raw_text,
                    'wl_min': wl_all.min(),
                    'wl_max': wl_all.max(),
                    'n_points': len(wl_all),
                })

    print(f"\nSpectra with splice metadata: {len(set(r['file_id'] for r in all_splice_records))}")
    print(f"Total splice events: {len(all_splice_records)}")

    # Count by D2 relevance
    d2_counts = {}
    for r in all_splice_records:
        rel = r['d2_relevance']
        d2_counts[rel] = d2_counts.get(rel, 0) + 1

    print("\nSplice wavelengths by D2 relevance:")
    for rel in ['below_D2', 'near_D2_below', 'inside_D2', 'near_D2_above', 'above_D2']:
        if rel in d2_counts:
            print(f"  {rel}: {d2_counts[rel]}")

    # ========================================================================
    # TASK 2-6: Analyze D2-relevant splice boundaries
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 2-6: Analyzing D2-relevant splice boundaries")
    print("=" * 80)

    # Filter to D2-relevant splices (inside or near D2)
    d2_relevant = [r for r in all_splice_records
                   if r['d2_relevance'] in ['inside_D2', 'near_D2_below', 'near_D2_above']]

    print(f"\nD2-relevant splice events: {len(d2_relevant)}")

    detailed_results = []

    for rec in d2_relevant:
        file_id = rec['file_id']
        splice_wl = rec['splice_wl_nm']

        # Re-parse file to get spectral data
        raw_path = DATA_DIR / candidates[candidates['file_id'] == file_id]['raw_path'].values[0]
        segments, footer = parse_relab_full(raw_path)
        wl_all = np.concatenate([s[0] for s in segments])
        rf_all = np.concatenate([s[1] for s in segments])

        # Sort by wavelength (in case of unsorted)
        sort_idx = np.argsort(wl_all)
        wl_all = wl_all[sort_idx]
        rf_all = rf_all[sort_idx]

        # Compute local metrics
        metrics = compute_local_metrics(wl_all, rf_all, splice_wl)

        # Diagnostic interpolation check
        interp_diag = diagnostic_interpolation_check(wl_all, rf_all, splice_wl)

        # Classify splice
        category, explanation = classify_splice(metrics)

        # Build result record
        result = {
            'file_id': file_id,
            'mineral': rec['mineral'],
            'provenance_group': rec['provenance_group'],
            'sample_id': rec['sample_id'],
            'splice_wl_nm': splice_wl,
            'd2_relevance': rec['d2_relevance'],
            'raw_splice_text': rec['raw_splice_text'],
        }

        # Add metrics
        if metrics:
            result.update({
                'wl_before_nm': metrics['wl_before'],
                'rf_before': metrics['rf_before'],
                'wl_after_nm': metrics['wl_after'],
                'rf_after': metrics['rf_after'],
                'wl_gap_nm': metrics['wl_gap'],
                'abs_disc': metrics['abs_disc'],
                'rel_disc': metrics['rel_disc'],
                'slope_before': metrics['slope_before'],
                'slope_after': metrics['slope_after'],
                'slope_change': metrics['slope_change'],
                'local_variation': metrics['local_variation'],
                'local_range': metrics['local_range'],
                'disc_vs_variation': metrics['disc_vs_variation'],
            })
        else:
            result.update({
                'wl_before_nm': np.nan,
                'rf_before': np.nan,
                'wl_after_nm': np.nan,
                'rf_after': np.nan,
                'wl_gap_nm': np.nan,
                'abs_disc': np.nan,
                'rel_disc': np.nan,
                'slope_before': np.nan,
                'slope_after': np.nan,
                'slope_change': np.nan,
                'local_variation': np.nan,
                'local_range': np.nan,
                'disc_vs_variation': np.nan,
            })

        # Add interpolation diagnostics
        if interp_diag:
            result.update({
                'interp_gap_nm': interp_diag['gap_nm'],
                'would_interpolate': interp_diag['would_interpolate'],
                'n_grid_points_in_gap': interp_diag.get('n_grid_points_in_gap', 0),
                'max_interp_error_est': interp_diag['max_interp_error_est'],
            })
        else:
            result.update({
                'interp_gap_nm': np.nan,
                'would_interpolate': False,
                'n_grid_points_in_gap': 0,
                'max_interp_error_est': np.nan,
            })

        # Add classification
        result['evidence_category'] = category
        result['classification_notes'] = explanation

        # Complexity sensitivity notes
        if category == 'A':
            result['complexity_sensitivity'] = 'Low — negligible discontinuity'
        elif category == 'B':
            result['complexity_sensitivity'] = 'Low-Moderate — small discontinuity, monitor derivatives'
        elif category == 'C':
            result['complexity_sensitivity'] = 'Moderate-High — may affect 2nd derivative and peak density'
        elif category == 'D':
            result['complexity_sensitivity'] = 'High — likely affects multiple complexity metrics'
        else:
            result['complexity_sensitivity'] = 'Unknown — insufficient data'

        detailed_results.append(result)

        print(f"  {file_id} @ {splice_wl:.1f} nm: {category} - {explanation}")

    # ========================================================================
    # Save CSV
    # ========================================================================
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "phase2_step3_splice_boundary_investigation.csv"

    if detailed_results:
        df_out = pd.DataFrame(detailed_results)
        df_out.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"[OK] Saved {csv_path}")
    else:
        print("[WARN] No D2-relevant splice events found")
        # Save empty CSV with headers
        df_empty = pd.DataFrame(columns=[
            'file_id', 'mineral', 'provenance_group', 'sample_id', 'splice_wl_nm',
            'd2_relevance', 'raw_splice_text', 'wl_before_nm', 'rf_before',
            'wl_after_nm', 'rf_after', 'wl_gap_nm', 'abs_disc', 'rel_disc',
            'slope_before', 'slope_after', 'slope_change', 'local_variation',
            'local_range', 'disc_vs_variation', 'interp_gap_nm', 'would_interpolate',
            'n_grid_points_in_gap', 'max_interp_error_est', 'evidence_category',
            'classification_notes', 'complexity_sensitivity'
        ])
        df_empty.to_csv(csv_path, index=False)
        print(f"[OK] Saved empty {csv_path}")

    # ========================================================================
    # Post-investigation checksum verification
    # ========================================================================
    print("\nVerifying raw checksums (post-investigation)...")
    post_mismatches = 0
    for _, row in ref.iterrows():
        fp = DATA_DIR / row["rel_path"]
        if fp.exists():
            md5 = compute_md5(fp)
            if md5 != row["md5"]:
                post_mismatches += 1
    print(f"  Checked: {len(ref)}, Mismatches: {post_mismatches}")
    assert post_mismatches == 0, "Raw file checksum mismatch detected!"
    print("  [OK] All checksums still verified")

    print("\n" + "=" * 80)
    print("Investigation complete")
    print("=" * 80)

    return detailed_results


if __name__ == "__main__":
    results = main()
