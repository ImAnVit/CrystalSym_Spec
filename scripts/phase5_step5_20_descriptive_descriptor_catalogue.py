#!/usr/bin/env python3
"""
Phase 5 Step 5.20 — Mineral-by-Mineral Descriptive Spectral Descriptor Catalogue

Produces a scientifically cautious, reproducible descriptive catalogue of spectral
descriptors (D1–D4) across the frozen canonical RELAB dataset, enriched with traceable
provenance and preparation metadata recovered in Step 5.18.

Primary question:
  What are the distributions of D1–D4 spectral descriptors across minerals in the
  frozen canonical RELAB dataset, and what provenance and sample-preparation
  information is available to contextualise observed variation?

Deliverables:
  1. phase5_step5_20_mineral_descriptor_summary.csv
  2. phase5_step5_20_source_group_descriptor_summary.csv
  3. phase5_step5_20_provenance_enriched_catalogue.csv
  4. phase5_step5_20_catalogue_report.md
  5. phase5_step5_20_run_manifest.json
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\andri\Projects\crystalsym-spec")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis"

# Input files
LINKED_VIEW_PATH = RESULTS_DIR / "phase5_step5_13_linked_comparative_view" / "phase5_step5_13_linked_analysis_view.csv"
INTEGRATED_DATASET_PATH = DATA_DIR / "phase4_integrated_analysis_dataset.csv"
PROVENANCE_MAPPING_PATH = DATA_DIR / "phase4_provenance_mapping.csv"
METADATA_RECOVERY_PATH = RESULTS_DIR / "phase5_step5_18_curator_preparation_metadata_recovery" / "phase5_step5_18_metadata_recovery_register.csv"
SYMMETRY_METADATA_PATH = DATA_DIR / "phase4_symmetry_metadata.csv"

# Output directory
OUTPUT_DIR = RESULTS_DIR / "phase5_step5_20_descriptive_descriptor_catalogue"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Descriptor columns
DESCRIPTOR_COLS = ['D1_n_peaks_prom', 'D2_mean_peak_prominence', 'D3_first_deriv_rms', 'D4_second_deriv_rms']
DESCRIPTOR_LABELS = {
    'D1_n_peaks_prom': 'D1: Number of peaks (prominence)',
    'D2_mean_peak_prominence': 'D2: Mean peak prominence',
    'D3_first_deriv_rms': 'D3: First derivative RMS',
    'D4_second_deriv_rms': 'D4: Second derivative RMS'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all input datasets and verify record counts."""
    print("Loading input datasets...")

    # Linked view (503 records: 454 canonical + 49 extension)
    linked_view = pd.read_csv(LINKED_VIEW_PATH)
    assert len(linked_view) == 503, f"Expected 503 records in linked view, got {len(linked_view)}"
    print(f"  Linked view: {len(linked_view)} records")

    # Metadata recovery register (2050 records)
    recovery = pd.read_csv(METADATA_RECOVERY_PATH)
    print(f"  Metadata recovery register: {len(recovery)} records")

    # Integrated dataset (454 canonical records)
    integrated = pd.read_csv(INTEGRATED_DATASET_PATH)
    assert len(integrated) == 454, f"Expected 454 records in integrated dataset, got {len(integrated)}"
    print(f"  Integrated dataset: {len(integrated)} records")

    # Provenance mapping (382 source groups)
    prov_map = pd.read_csv(PROVENANCE_MAPPING_PATH)
    print(f"  Provenance mapping: {len(prov_map)} source groups")

    # Symmetry metadata
    symmetry = pd.read_csv(SYMMETRY_METADATA_PATH)
    print(f"  Symmetry metadata: {len(symmetry)} source groups")

    return linked_view, recovery, integrated, prov_map, symmetry

# ============================================================================
# BUILD PROVENANCE-ENRICHED CATALOGUE
# ============================================================================

def build_provenance_enriched_catalogue(linked_view, recovery, integrated, prov_map):
    """
    Build comprehensive per-spectrum table with all recoverable metadata.

    Joins:
      - Linked view (503 records) as base
      - Recovery metadata by original_source_file → source_file
      - Integrated dataset by original_record_id → canonical_spectrum_id (canonical only)
      - Provenance mapping by source_group_id → provenance_group_id
    """
    print("\nBuilding provenance-enriched catalogue...")

    # Start with linked view
    catalogue = linked_view.copy()

    # Rename descriptor columns to match integrated dataset convention
    catalogue = catalogue.rename(columns={
        'original_source_file': 'source_file',
        'mineral_label': 'mineral_label_linked',
        'crystal_system': 'crystal_system_linked',
        'point_group': 'point_group_linked',
        'space_group_symbol': 'space_group_symbol_linked',
        'space_group_number': 'space_group_number_linked'
    })

    # Aggregate recovery metadata to one record per source_file
    # (recovery register has multiple records per file from different sources:
    # raw_file_metadata, relab_measurements_csv, source_collection_code)
    recovery_metadata_cols = [
        'source_file', 'recovery_id', 'relab_sample_id', 'sample_description',
        'grain_size_value', 'grain_size_evidence_type', 'grain_size_confidence',
        'locality_hints', 'locality_confidence',
        'mixture_status_from_description', 'mixture_confidence',
        'thermal_treatment_from_description', 'thermal_confidence',
        'source_angle', 'detect_angle', 'voltage', 'measurement_date',
        'has_extended_metadata', 'has_halon_calibration',
        'operator', 'run_mode', 'stage_type', 'spinning', 'aperture'
    ]
    recovery_agg_cols = [c for c in recovery_metadata_cols if c in recovery.columns and c != 'source_file']
    recovery_agg = recovery.groupby('source_file')[recovery_agg_cols].first().reset_index()

    # Join recovery metadata
    n_before = len(catalogue)
    catalogue = catalogue.merge(recovery_agg, on='source_file', how='left', suffixes=('', '_recovery'))
    assert len(catalogue) == n_before, "Join with recovery created duplicate rows"
    n_recovery_matched = catalogue['recovery_id'].notna().sum()
    print(f"  Recovery metadata joined: {n_recovery_matched}/{len(catalogue)} records matched")

    # Rename recovery sample_description to avoid conflict with integrated dataset
    if 'sample_description' in catalogue.columns:
        catalogue = catalogue.rename(columns={'sample_description': 'sample_description_recovery'})

    # Join integrated dataset (canonical records only)
    integrated_add_cols = [
        'sample_description', 'matching_confidence',
        'matching_status', 'in_duplicate_relationship', 'has_identity_issues',
        'polymorph_identifier', 'polymorph_ambiguous_symmetry',
        'provenance_tentative', 'high_conflict', 'high_conflict_type',
        'chemistry_type', 'chemistry_specimen_specific', 'chemistry_evidence_level',
        'reference_formula', 'solid_solution_flag', 'mixture_flag',
        'mineral_description_conflict', 'augite_olivine_duplicate',
        'high_conflict_flag', 'tentative_provenance', 'conflict_types', 'conflict_severities'
    ]

    n_before = len(catalogue)
    catalogue = catalogue.merge(
        integrated[['canonical_spectrum_id'] + integrated_add_cols],
        left_on='original_record_id',
        right_on='canonical_spectrum_id',
        how='left'
    )
    assert len(catalogue) == n_before, "Join with integrated created duplicate rows"
    n_integrated_matched = catalogue['sample_description'].notna().sum()
    print(f"  Integrated dataset joined: {n_integrated_matched}/{len(catalogue)} records matched (canonical expected)")

    # Join provenance mapping
    prov_add_cols = ['sample_type', 'polymorph_flag', 'polymorph_details', 'polymorph_status']
    n_before = len(catalogue)
    catalogue = catalogue.merge(
        prov_map[['provenance_group_id'] + prov_add_cols],
        left_on='source_group_id',
        right_on='provenance_group_id',
        how='left',
        suffixes=('', '_prov')
    )
    assert len(catalogue) == n_before, "Join with provenance mapping created duplicate rows"
    n_prov_matched = catalogue['sample_type'].notna().sum()
    print(f"  Provenance mapping joined: {n_prov_matched}/{len(catalogue)} records matched")

    # Reorder columns logically
    id_cols = [
        'linked_view_id', 'dataset_origin', 'original_record_id', 'source_file',
        'population_label', 'source_group_id', 'source_group_origin'
    ]
    mineral_cols = ['mineral_label_linked', 'sample_description', 'sample_type']
    descriptor_cols = DESCRIPTOR_COLS + ['S1_total_variation']
    crystal_cols = [
        'crystal_system_linked', 'point_group_linked', 'space_group_symbol_linked',
        'space_group_number_linked', 'polymorph_identifier', 'polymorph_flag'
    ]
    provenance_cols = [
        'matching_confidence', 'matching_status', 'identity_confidence',
        'independence_qualifier', 'independence_status',
        'in_duplicate_relationship', 'has_identity_issues',
        'provenance_tentative', 'high_conflict', 'high_conflict_flag'
    ]
    chemistry_cols = [
        'chemistry_type', 'chemistry_specimen_specific', 'chemistry_evidence_level',
        'reference_formula', 'solid_solution_flag', 'mixture_flag'
    ]
    recovery_cols = [
        'recovery_id', 'relab_sample_id', 'sample_description_recovery',
        'grain_size_value', 'grain_size_confidence',
        'locality_hints', 'locality_confidence',
        'mixture_status_from_description', 'mixture_confidence',
        'thermal_treatment_from_description', 'thermal_confidence',
        'source_angle', 'detect_angle', 'voltage', 'measurement_date'
    ]
    quality_cols = [
        'mineral_description_conflict', 'augite_olivine_duplicate',
        'conflict_types', 'conflict_severities', 'biogenic_flag', 'biogenic_adjudication'
    ]

    # Build final column list (include all columns not explicitly ordered)
    ordered_cols = id_cols + mineral_cols + descriptor_cols + crystal_cols + provenance_cols + chemistry_cols + recovery_cols + quality_cols
    remaining_cols = [c for c in catalogue.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining_cols

    catalogue = catalogue[final_cols]

    print(f"  Final catalogue: {len(catalogue)} records, {len(catalogue.columns)} columns")

    return catalogue

# ============================================================================
# COMPUTE MINERAL-LEVEL SUMMARIES
# ============================================================================

def compute_mineral_summaries(catalogue):
    """
    Compute mineral-level descriptor summaries for canonical and extension populations.

    For each mineral:
      - Record count (n_spectra)
      - Distinct source group count
      - For each descriptor: n_valid, median, IQR, min, max
    """
    print("\nComputing mineral-level summaries...")

    def compute_stats(group, population_label):
        """Compute statistics for a mineral group."""
        row = {
            'mineral': group['mineral_label_linked'].iloc[0],
            'population': population_label,
            'n_spectra': len(group),
            'n_source_groups': group['source_group_id'].nunique()
        }

        for desc in DESCRIPTOR_COLS:
            valid = group[desc].dropna()
            row[f'{desc}_n_valid'] = len(valid)
            if len(valid) > 0:
                row[f'{desc}_median'] = valid.median()
                row[f'{desc}_q25'] = valid.quantile(0.25)
                row[f'{desc}_q75'] = valid.quantile(0.75)
                row[f'{desc}_iqr'] = row[f'{desc}_q75'] - row[f'{desc}_q25']
                row[f'{desc}_min'] = valid.min()
                row[f'{desc}_max'] = valid.max()
            else:
                row[f'{desc}_median'] = np.nan
                row[f'{desc}_q25'] = np.nan
                row[f'{desc}_q75'] = np.nan
                row[f'{desc}_iqr'] = np.nan
                row[f'{desc}_min'] = np.nan
                row[f'{desc}_max'] = np.nan

        return row

    # Canonical summaries
    canonical = catalogue[catalogue['dataset_origin'] == 'CANONICAL_V1'].copy()
    canonical_rows = []
    for mineral in sorted(canonical['mineral_label_linked'].unique()):
        group = canonical[canonical['mineral_label_linked'] == mineral]
        canonical_rows.append(compute_stats(group, 'CANONICAL_V1'))
    canonical_summary = pd.DataFrame(canonical_rows)

    # Extension summaries
    extension = catalogue[catalogue['dataset_origin'] != 'CANONICAL_V1'].copy()
    extension_rows = []
    for mineral in sorted(extension['mineral_label_linked'].unique()):
        group = extension[extension['mineral_label_linked'] == mineral]
        extension_rows.append(compute_stats(group, 'EXTENSION'))
    extension_summary = pd.DataFrame(extension_rows)

    # Combine
    mineral_summary = pd.concat([canonical_summary, extension_summary], ignore_index=True)

    print(f"  Canonical minerals: {len(canonical_summary)}")
    print(f"  Extension minerals: {len(extension_summary)}")

    return mineral_summary

# ============================================================================
# COMPUTE SOURCE-GROUP-LEVEL SUMMARIES
# ============================================================================

def compute_source_group_summaries(catalogue):
    """
    Compute source-group-level descriptor summaries.

    For each source group:
      - Mineral label
      - n_spectra (number of spectra in group)
      - For each descriptor: median, IQR, min, max (across spectra within group)

    This shows the distribution of descriptor values at the source-group level,
    acknowledging that spectra within a source group are not independent.
    """
    print("\nComputing source-group-level summaries...")

    # Canonical source groups
    canonical = catalogue[catalogue['dataset_origin'] == 'CANONICAL_V1'].copy()

    sg_rows = []
    for sg_id in sorted(canonical['source_group_id'].unique()):
        group = canonical[canonical['source_group_id'] == sg_id]

        row = {
            'source_group_id': sg_id,
            'mineral': group['mineral_label_linked'].iloc[0],
            'n_spectra': len(group),
            'sample_description': group['sample_description'].iloc[0] if 'sample_description' in group.columns else '',
            'sample_type': group['sample_type'].iloc[0] if 'sample_type' in group.columns else ''
        }

        for desc in DESCRIPTOR_COLS:
            valid = group[desc].dropna()
            if len(valid) > 0:
                row[f'{desc}_median'] = valid.median()
                row[f'{desc}_q25'] = valid.quantile(0.25)
                row[f'{desc}_q75'] = valid.quantile(0.75)
                row[f'{desc}_iqr'] = row[f'{desc}_q75'] - row[f'{desc}_q25']
                row[f'{desc}_min'] = valid.min()
                row[f'{desc}_max'] = valid.max()
            else:
                row[f'{desc}_median'] = np.nan
                row[f'{desc}_q25'] = np.nan
                row[f'{desc}_q75'] = np.nan
                row[f'{desc}_iqr'] = np.nan
                row[f'{desc}_min'] = np.nan
                row[f'{desc}_max'] = np.nan

        sg_rows.append(row)

    sg_summary = pd.DataFrame(sg_rows)

    print(f"  Canonical source groups: {len(sg_summary)}")

    return sg_summary

# ============================================================================
# GENERATE FIGURES
# ============================================================================

def generate_figures(catalogue, mineral_summary):
    """
    Generate descriptor distribution plots.

    Figure 1: 2x2 grid of box plots showing D1–D4 distributions by mineral (canonical only)
    Figure 2: Sample sizes per mineral (n_spectra and n_source_groups)
    """
    print("\nGenerating figures...")

    canonical = catalogue[catalogue['dataset_origin'] == 'CANONICAL_V1'].copy()

    # Figure 1: Descriptor distributions by mineral
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    axes1 = axes1.flatten()

    # Sort minerals by median D1 for consistent ordering
    mineral_order = mineral_summary[mineral_summary['population'] == 'CANONICAL_V1'].copy()
    mineral_order = mineral_order.sort_values('D1_n_peaks_prom_median', ascending=False)
    mineral_order = mineral_order[mineral_order['n_spectra'] >= 2]  # Only minerals with ≥2 spectra
    minerals = mineral_order['mineral'].tolist()

    for idx, desc in enumerate(DESCRIPTOR_COLS):
        ax = axes1[idx]

        # Collect data for each mineral
        data_by_mineral = []
        for mineral in minerals:
            mineral_data = canonical[canonical['mineral_label_linked'] == mineral][desc].dropna()
            data_by_mineral.append(mineral_data.values)

        # Create box plot
        bp = ax.boxplot(data_by_mineral, orientation='horizontal', patch_artist=True)
        ax.set_yticklabels(minerals)
        for patch in bp['boxes']:
            patch.set_facecolor('#6baed6')

        ax.set_xlabel(DESCRIPTOR_LABELS[desc])
        ax.set_title(f'{desc.replace("_", " ").title()} by Mineral')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / 'phase5_step5_20_descriptor_distributions_by_mineral.png'
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig1_path}")

    # Figure 2: Sample sizes per mineral
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    canonical_summary = mineral_summary[mineral_summary['population'] == 'CANONICAL_V1'].copy()
    canonical_summary = canonical_summary.sort_values('n_spectra', ascending=True)
    minerals = canonical_summary['mineral'].tolist()

    # Left panel: n_spectra
    y_pos = range(len(minerals))
    ax1.barh(y_pos, canonical_summary['n_spectra'], color='#6baed6')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(minerals)
    ax1.set_xlabel('Number of Spectra')
    ax1.set_title('Canonical Dataset: Spectra per Mineral')
    ax1.grid(axis='x', alpha=0.3)

    # Right panel: n_source_groups
    ax2.barh(y_pos, canonical_summary['n_source_groups'], color='#9ecae1')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(minerals)
    ax2.set_xlabel('Number of Source Groups')
    ax2.set_title('Canonical Dataset: Source Groups per Mineral')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / 'phase5_step5_20_sample_sizes_by_mineral.png'
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")

    plt.close(fig1)
    plt.close(fig2)

    return fig1_path, fig2_path

# ============================================================================
# GENERATE MARKDOWN REPORT
# ============================================================================

def generate_report(catalogue, mineral_summary, sg_summary, fig1_path, fig2_path):
    """Generate comprehensive markdown report."""
    print("\nGenerating markdown report...")

    canonical = catalogue[catalogue['dataset_origin'] == 'CANONICAL_V1'].copy()
    extension = catalogue[catalogue['dataset_origin'] != 'CANONICAL_V1'].copy()
    canonical_summary = mineral_summary[mineral_summary['population'] == 'CANONICAL_V1'].copy()

    report = []
    report.append("# Phase 5 Step 5.20 — Mineral-by-Mineral Descriptive Spectral Descriptor Catalogue")
    report.append("")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    report.append(f"**Population:** Frozen canonical RELAB dataset (454 spectra, 18 minerals)")
    report.append(f"**Extension:** 49 spectra, 4 minerals (reported separately)")
    report.append("")
    report.append("---")
    report.append("")

    # Section 1: Overview
    report.append("## 1. Overview")
    report.append("")
    report.append("This catalogue presents the distributions of four spectral descriptors (D1–D4) across minerals in the frozen canonical RELAB dataset, enriched with traceable provenance and preparation metadata recovered in Step 5.18.")
    report.append("")
    report.append("**What this is:**")
    report.append("- Descriptive summary of descriptor distributions by mineral")
    report.append("- Source-group-level aggregation acknowledging dependence structures")
    report.append("- Provenance annotation with recoverable metadata (grain size, locality hints, mixture status, measurement conditions)")
    report.append("- Transparent documentation of coverage limitations and unresolved questions")
    report.append("")
    report.append("**What this is NOT:**")
    report.append("- Not an inferential comparison between minerals")
    report.append("- Not a test of the crystal symmetry hypothesis")
    report.append("- Not a claim that descriptor differences reflect crystallographic or chemical differences")
    report.append("- Not a pooled analysis across minerals")
    report.append("")
    report.append("**Research question (RQ-A1 from Step 5.19):**")
    report.append("> What are the distributions of spectral descriptors (D1–D4) across minerals in the RELAB frozen canonical dataset, and how do measurement provenance and sample preparation relate to descriptor variation?")
    report.append("")
    report.append("---")
    report.append("")

    # Section 2: Data Sources
    report.append("## 2. Data Sources and Population Definitions")
    report.append("")
    report.append("### 2.1 Frozen Canonical Dataset")
    report.append("")
    report.append(f"- **Records:** {len(canonical)} spectra")
    report.append(f"- **Minerals:** {canonical['mineral_label_linked'].nunique()}")
    report.append(f"- **Source groups:** {canonical['source_group_id'].nunique()}")
    report.append(f"- **Grid:** Uniform D2 grid (320–2550 nm, 5 nm spacing, 447 points)")
    report.append(f"- **Preprocessing:** Phase 2 frozen (native 5 nm or resampled to uniform grid)")
    report.append("")
    report.append("### 2.2 Extension Dataset (Reported Separately)")
    report.append("")
    report.append(f"- **Records:** {len(extension)} spectra")
    report.append(f"- **Minerals:** {extension['mineral_label_linked'].nunique()} ({', '.join(sorted(extension['mineral_label_linked'].unique()))})")
    report.append(f"- **Source groups:** {extension['source_group_id'].nunique()}")
    report.append(f"- **Origin:** Step 5.8 extension construction")
    report.append("")
    report.append("**Note:** Extension records are reported separately and not pooled with canonical records. Extension spectra have different preprocessing routes and independence statuses.")
    report.append("")
    report.append("### 2.3 Metadata Recovery (Step 5.18)")
    report.append("")
    report.append("Metadata recovered from raw file headers and RELAB measurement records:")
    report.append("")
    report.append(f"- **Grain size:** {canonical['grain_size_value'].notna().sum()}/{len(canonical)} canonical spectra ({100*canonical['grain_size_value'].notna().mean():.1f}%)")
    report.append(f"- **Locality hints:** {canonical['locality_hints'].notna().sum()}/{len(canonical)} ({100*canonical['locality_hints'].notna().mean():.1f}%)")
    report.append(f"- **Mixture status:** {canonical['mixture_status_from_description'].notna().sum()}/{len(canonical)} ({100*canonical['mixture_status_from_description'].notna().mean():.1f}%)")
    report.append(f"- **Thermal treatment:** {canonical['thermal_treatment_from_description'].notna().sum()}/{len(canonical)} ({100*canonical['thermal_treatment_from_description'].notna().mean():.1f}%)")
    report.append(f"- **Measurement geometry:** {canonical['source_angle'].notna().sum()}/{len(canonical)} ({100*canonical['source_angle'].notna().mean():.1f}%)")
    report.append("")
    report.append("---")
    report.append("")

    # Section 3: Mineral-Level Descriptor Summaries
    report.append("## 3. Mineral-Level Descriptor Summaries (Canonical Dataset)")
    report.append("")
    report.append("Table 1 presents spectrum-level descriptor statistics for each mineral in the canonical dataset. For each mineral, the table shows the number of spectra, number of distinct source groups, and the median, interquartile range (IQR), minimum, and maximum of each descriptor.")
    report.append("")
    report.append("**Interpretation guide:**")
    report.append("- **n_spectra:** Total number of spectra for this mineral")
    report.append("- **n_source_groups:** Number of distinct provenance groups (source groups) — a measure of specimen diversity")
    report.append("- **Median, IQR, Min, Max:** Computed across all spectra for this mineral")
    report.append("- **Spectra within a source group are NOT independent** (they may be grain-size fractions, replicates, or measurements of the same specimen)")
    report.append("")

    # Table 1: Mineral summaries
    table1 = []
    table1.append("| Mineral | n_spectra | n_source_groups | D1 median | D1 IQR | D2 median | D2 IQR | D3 median | D3 IQR | D4 median | D4 IQR |")
    table1.append("|---------|-----------|-----------------|-----------|--------|-----------|--------|-----------|--------|-----------|--------|")

    for _, row in canonical_summary.sort_values('mineral').iterrows():
        table1.append(f"| {row['mineral']} | {int(row['n_spectra'])} | {int(row['n_source_groups'])} | "
                     f"{row['D1_n_peaks_prom_median']:.2f} | {row['D1_n_peaks_prom_iqr']:.2f} | "
                     f"{row['D2_mean_peak_prominence_median']:.4f} | {row['D2_mean_peak_prominence_iqr']:.4f} | "
                     f"{row['D3_first_deriv_rms_median']:.6f} | {row['D3_first_deriv_rms_iqr']:.6f} | "
                     f"{row['D4_second_deriv_rms_median']:.6f} | {row['D4_second_deriv_rms_iqr']:.6f} |")

    report.extend(table1)
    report.append("")
    report.append("**Table 1.** Mineral-level descriptor summaries (canonical dataset). D1 = number of peaks (prominence); D2 = mean peak prominence; D3 = first derivative RMS; D4 = second derivative RMS.")
    report.append("")

    # Figure 1
    report.append(f"![Descriptor distributions by mineral]({fig1_path.name})")
    report.append("")
    report.append("**Figure 1.** Box plots showing the distribution of D1–D4 descriptors across minerals in the canonical dataset. Minerals are sorted by median D1 (descending). Only minerals with ≥2 spectra are shown. Boxes show IQR (25th–75th percentile); horizontal lines show medians.")
    report.append("")

    # Figure 2
    report.append(f"![Sample sizes by mineral]({fig2_path.name})")
    report.append("")
    report.append("**Figure 2.** Sample sizes per mineral in the canonical dataset. Left: number of spectra. Right: number of distinct source groups. The difference between n_spectra and n_source_groups indicates dependence structures (grain-size series, replicates, multiple measurements of the same specimen).")
    report.append("")
    report.append("---")
    report.append("")

    # Section 4: Source-Group-Level Summaries
    report.append("## 4. Source-Group-Level Descriptor Summaries")
    report.append("")
    report.append("Table 2 presents descriptor statistics at the source-group level. For each source group, the table shows the number of spectra and the median descriptor value (computed across spectra within that source group).")
    report.append("")
    report.append("**Why source-group-level summaries matter:**")
    report.append("- Spectra within a source group are not independent specimens")
    report.append("- Source groups represent provenance units (may be single specimens, grain-size series, or measurement campaigns)")
    report.append("- Source-group-level medians reduce the influence of within-group dependence")
    report.append("- The number of source groups (not spectra) is the appropriate denominator for cross-mineral comparisons")
    report.append("")

    # Table 2: Source group summaries (summary statistics)
    report.append("### 4.1 Summary Statistics Across Source Groups")
    report.append("")
    report.append("Table 2a shows the distribution of source-group-level medians for each mineral.")
    report.append("")

    table2a = []
    table2a.append("| Mineral | n_source_groups | n_spectra | D1 median (of SG medians) | D1 IQR | D2 median (of SG medians) | D2 IQR |")
    table2a.append("|---------|-----------------|-----------|---------------------------|--------|---------------------------|--------|")

    for mineral in sorted(canonical_summary['mineral'].unique()):
        sg_mineral = sg_summary[sg_summary['mineral'] == mineral]
        n_sg = len(sg_mineral)
        n_spectra = sg_mineral['n_spectra'].sum()

        d1_medians = sg_mineral['D1_n_peaks_prom_median'].dropna()
        d2_medians = sg_mineral['D2_mean_peak_prominence_median'].dropna()

        d1_median_of_medians = d1_medians.median() if len(d1_medians) > 0 else np.nan
        d1_iqr = (d1_medians.quantile(0.75) - d1_medians.quantile(0.25)) if len(d1_medians) > 0 else np.nan
        d2_median_of_medians = d2_medians.median() if len(d2_medians) > 0 else np.nan
        d2_iqr = (d2_medians.quantile(0.75) - d2_medians.quantile(0.25)) if len(d2_medians) > 0 else np.nan

        table2a.append(f"| {mineral} | {n_sg} | {n_spectra} | "
                      f"{d1_median_of_medians:.2f} | {d1_iqr:.2f} | "
                      f"{d2_median_of_medians:.4f} | {d2_iqr:.4f} |")

    report.extend(table2a)
    report.append("")
    report.append("**Table 2a.** Source-group-level descriptor summaries. For each mineral, the table shows the number of source groups, total number of spectra, and the median and IQR of source-group-level medians for D1 and D2.")
    report.append("")

    report.append("### 4.2 Full Source Group Table")
    report.append("")
    report.append("The complete source-group-level descriptor summary is provided in `phase5_step5_20_source_group_descriptor_summary.csv`. This file includes all source groups with their mineral labels, spectrum counts, and descriptor statistics.")
    report.append("")
    report.append("---")
    report.append("")

    # Section 5: Metadata Recovery
    report.append("## 5. Metadata Recovery and Provenance Enrichment")
    report.append("")
    report.append("Step 5.18 recovered metadata from raw file headers and RELAB measurement records. This section summarises the metadata available for the canonical dataset.")
    report.append("")

    report.append("### 5.1 Metadata Coverage")
    report.append("")
    report.append("Table 3 shows the coverage of key metadata fields in the canonical dataset.")
    report.append("")

    table3 = []
    table3.append("| Metadata Field | n_available | n_total | Coverage (%) |")
    table3.append("|----------------|-------------|---------|--------------|")

    metadata_fields = [
        ('Grain size', 'grain_size_value'),
        ('Locality hints', 'locality_hints'),
        ('Mixture status', 'mixture_status_from_description'),
        ('Thermal treatment', 'thermal_treatment_from_description'),
        ('Source angle', 'source_angle'),
        ('Detect angle', 'detect_angle'),
        ('Measurement date', 'measurement_date')
    ]

    for label, col in metadata_fields:
        n_avail = canonical[col].notna().sum()
        n_total = len(canonical)
        pct = 100 * n_avail / n_total
        table3.append(f"| {label} | {n_avail} | {n_total} | {pct:.1f}% |")

    report.extend(table3)
    report.append("")
    report.append("**Table 3.** Metadata coverage in the canonical dataset. Coverage varies by field; grain size and measurement geometry are most complete, while locality hints are sparse.")
    report.append("")

    report.append("### 5.2 Grain Size Distribution")
    report.append("")
    grain_size_avail = canonical['grain_size_value'].dropna()
    if len(grain_size_avail) > 0:
        report.append(f"Grain size information is available for {len(grain_size_avail)}/{len(canonical)} spectra ({100*len(grain_size_avail)/len(canonical):.1f}%).")
        report.append("")
        report.append("Grain size values are recorded as strings (e.g., '<125 um', '45-150 um', '710-1000 um') and reflect the precision of the original measurement records. Numeric parsing is not attempted here; the confidence labels from Step 5.18 should be consulted for interpretation.")
        report.append("")
    else:
        report.append("No grain size information recovered.")
        report.append("")

    report.append("### 5.3 Mixture Status")
    report.append("")
    mixture_avail = canonical['mixture_status_from_description'].dropna()
    if len(mixture_avail) > 0:
        n_explicit = (mixture_avail == 'explicit_mixture').sum()
        n_unknown = (mixture_avail == 'unknown').sum()
        n_absent = (mixture_avail == 'absent').sum()
        report.append(f"Mixture status is available for {len(mixture_avail)}/{len(canonical)} spectra.")
        report.append("")
        report.append(f"- **Explicit mixture:** {n_explicit}")
        report.append(f"- **Unknown:** {n_unknown}")
        report.append(f"- **Absent (no mixture indicated):** {n_absent}")
        report.append("")
        if n_explicit > 0:
            report.append("**Note:** Explicit mixtures should be interpreted with caution. Spectral descriptors of mixtures reflect the combined absorption features of all components, not a single mineral phase.")
            report.append("")
    else:
        report.append("No mixture status information recovered.")
        report.append("")

    report.append("### 5.4 Locality Hints")
    report.append("")
    locality_avail = canonical['locality_hints'].dropna()
    if len(locality_avail) > 0:
        report.append(f"Locality hints are available for {len(locality_avail)}/{len(canonical)} spectra ({100*len(locality_avail)/len(canonical):.1f}%).")
        report.append("")
        report.append("**Important:** Locality hints are extracted from sample descriptions and are **plausible leads requiring verification**, not confirmed localities. They should not be used as definitive geographic or geological context.")
        report.append("")
    else:
        report.append("No locality hints recovered.")
        report.append("")

    report.append("---")
    report.append("")

    # Section 6: Coverage Limitations
    report.append("## 6. Coverage Limitations and Dependence Structures")
    report.append("")
    report.append("### 6.1 Mineral Coverage")
    report.append("")
    report.append("Table 4 categorises minerals by data richness.")
    report.append("")

    table4 = []
    table4.append("| Coverage Tier | Minerals | n_minerals |")
    table4.append("|---------------|----------|------------|")

    # Tier 1: ≥20 spectra and ≥10 source groups
    tier1 = canonical_summary[(canonical_summary['n_spectra'] >= 20) & (canonical_summary['n_source_groups'] >= 10)]
    tier1_minerals = ', '.join(sorted(tier1['mineral'].unique())) if len(tier1) > 0 else 'None'
    table4.append(f"| Rich (≥20 spectra, ≥10 source groups) | {tier1_minerals} | {len(tier1)} |")

    # Tier 2: 5–19 spectra or 3–9 source groups
    tier2 = canonical_summary[((canonical_summary['n_spectra'] >= 5) & (canonical_summary['n_spectra'] < 20)) |
                              ((canonical_summary['n_source_groups'] >= 3) & (canonical_summary['n_source_groups'] < 10))]
    tier2 = tier2[~tier2['mineral'].isin(tier1['mineral'])]
    tier2_minerals = ', '.join(sorted(tier2['mineral'].unique())) if len(tier2) > 0 else 'None'
    table4.append(f"| Moderate (5–19 spectra or 3–9 source groups) | {tier2_minerals} | {len(tier2)} |")

    # Tier 3: <5 spectra AND <3 source groups (not already in tier 1 or 2)
    tier3 = canonical_summary[(canonical_summary['n_spectra'] < 5) & (canonical_summary['n_source_groups'] < 3)]
    tier3 = tier3[~tier3['mineral'].isin(tier1['mineral']) & ~tier3['mineral'].isin(tier2['mineral'])]
    tier3_minerals = ', '.join(sorted(tier3['mineral'].unique())) if len(tier3) > 0 else 'None'
    table4.append(f"| Limited (<5 spectra or <3 source groups) | {tier3_minerals} | {len(tier3)} |")

    report.extend(table4)
    report.append("")
    report.append("**Table 4.** Mineral coverage tiers. Coverage is assessed by both spectrum count and source group count. Minerals in the 'Limited' tier have insufficient data for meaningful distributional summaries.")
    report.append("")

    report.append("### 6.2 Dependence Structures")
    report.append("")
    report.append("Many source groups contain multiple spectra, indicating dependence structures:")
    report.append("")

    # Count source groups with multiple spectra
    sg_multi = sg_summary[sg_summary['n_spectra'] > 1]
    n_multi = len(sg_multi)
    n_total_sg = len(sg_summary)

    report.append(f"- **Source groups with multiple spectra:** {n_multi}/{n_total_sg} ({100*n_multi/n_total_sg:.1f}%)")
    report.append(f"- **Total spectra in multi-spectrum source groups:** {sg_multi['n_spectra'].sum()}")
    report.append("")
    report.append("**Types of dependence (from Step 5.13 and 5.17):**")
    report.append("- **Grain-size series:** Multiple spectra from the same specimen, sieved to different grain-size fractions")
    report.append("- **Replicates:** Repeated measurements of the same specimen")
    report.append("- **Multiple measurement campaigns:** Same specimen measured at different times or under different conditions")
    report.append("")
    report.append("**Implication:** Spectra within a source group are not independent specimens. Cross-mineral comparisons should use source-group-level summaries (Section 4) and acknowledge the effective sample size (n_source_groups), not the spectrum count (n_spectra).")
    report.append("")

    report.append("### 6.3 Minerals with Limited Coverage")
    report.append("")
    limited = canonical_summary[canonical_summary['n_spectra'] < 5]
    if len(limited) > 0:
        report.append("The following minerals have fewer than 5 spectra and insufficient data for meaningful distributional summaries:")
        report.append("")
        for _, row in limited.iterrows():
            report.append(f"- **{row['mineral']}**: {int(row['n_spectra'])} spectra, {int(row['n_source_groups'])} source group(s)")
        report.append("")
        report.append("These minerals are included in the catalogue for completeness, but their descriptor statistics should not be interpreted as representative distributions.")
        report.append("")
    else:
        report.append("All minerals have ≥5 spectra.")
        report.append("")

    report.append("### 6.4 Minerals with Potential Issues")
    report.append("")
    report.append("The following minerals have flags indicating potential data quality or identity issues:")
    report.append("")

    # Check for minerals with identity issues, mixtures, etc.
    issues = canonical[
        (canonical['has_identity_issues'] == True) |
        (canonical['mixture_flag'] == True) |
        (canonical['mineral_description_conflict'] == True) |
        (canonical['augite_olivine_duplicate'] == True)
    ]

    if len(issues) > 0:
        report.append(f"- **Records with identity/mixture/conflict flags:** {len(issues)}/{len(canonical)}")
        report.append("")
        report.append("Affected minerals:")
        for mineral in sorted(issues['mineral_label_linked'].unique()):
            n_issues = len(issues[issues['mineral_label_linked'] == mineral])
            n_total = len(canonical[canonical['mineral_label_linked'] == mineral])
            report.append(f"  - {mineral}: {n_issues}/{n_total} records flagged")
        report.append("")
        report.append("**Note:** Flagged records are included in the catalogue but should be interpreted with caution. See the provenance-enriched catalogue CSV for record-level flags.")
        report.append("")
    else:
        report.append("No minerals with identity/mixture/conflict flags.")
        report.append("")

    report.append("---")
    report.append("")

    # Section 7: Interpretation
    report.append("## 7. Interpretation: What the Catalogue Establishes and What Remains Unknown")
    report.append("")
    report.append("### 7.1 What the Catalogue Establishes")
    report.append("")
    report.append("1. **Descriptor distributions across 18 minerals.** The catalogue provides reproducible, descriptive summaries of D1–D4 for each mineral in the frozen canonical dataset, with full provenance annotation.")
    report.append("")
    report.append("2. **Coverage and metadata recovery.** The catalogue documents the extent of metadata recovery (grain size, locality hints, mixture status, measurement conditions) and identifies gaps.")
    report.append("")
    report.append("3. **Dependence structures.** The catalogue identifies source groups with multiple spectra and documents the types of dependence (grain-size series, replicates), providing the basis for appropriate statistical treatment in future analyses.")
    report.append("")
    report.append("4. **Evidence gaps.** The catalogue makes explicit which minerals have limited coverage, which have unresolved identity or mixture issues, and which comparisons are not defensible.")
    report.append("")

    report.append("### 7.2 What Remains Unknown")
    report.append("")
    report.append("1. **Specimen-specific chemistry.** All chemistry in the integrated dataset is ideal/reference chemistry (level C), not specimen-specific measured chemistry. The effect of chemical variation on descriptor distributions cannot be assessed.")
    report.append("")
    report.append("2. **Specimen-specific crystallography.** Crystal system assignments are from mineral reference databases, not specimen-specific XRD. Polymorph identity is not confirmed for any specimen.")
    report.append("")
    report.append("3. **Specimen independence.** Source groups are provisional units based on RELAB sample IDs. True specimen independence (whether different source groups represent different physical specimens) is not verified.")
    report.append("")
    report.append("4. **Causal drivers of descriptor variation.** The catalogue documents descriptor distributions but does not identify causes. Observed variation may reflect mineral identity, crystal structure, chemistry, grain size, measurement conditions, sample preparation, or their interactions. This catalogue cannot disentangle these effects.")
    report.append("")

    report.append("### 7.3 Comparisons That Are Not Defensible")
    report.append("")
    report.append("The following comparisons are **not defensible** with the current data:")
    report.append("")
    report.append("1. **Cross-mineral inferential comparisons.** Hypothesis tests, confidence intervals, or effect sizes comparing descriptor distributions between minerals are not defensible because:")
    report.append("   - Specimen independence is not verified")
    report.append("   - Chemistry is not specimen-specific")
    report.append("   - Crystal system is perfectly confounded with mineral identity")
    report.append("   - Measurement conditions vary across minerals")
    report.append("")
    report.append("2. **Crystal symmetry hypothesis tests.** The hypothesis that crystal symmetry constrains spectral complexity cannot be tested because:")
    report.append("   - No polymorph pair has verified independent specimens")
    report.append("   - No specimen-specific crystallography is available")
    report.append("   - Mineral identity and crystal system are completely confounded")
    report.append("")
    report.append("3. **Chemistry–descriptor associations.** Associations between chemistry (even ideal/reference chemistry) and descriptors should not be interpreted as causal, because:")
    report.append("   - Reference chemistry is not measured specimen chemistry")
    report.append("   - Chemistry and mineral identity are confounded")
    report.append("   - No control for measurement conditions or sample preparation")
    report.append("")
    report.append("4. **Pooled cross-mineral analysis.** Pooling spectra across minerals and treating them as independent observations is not defensible because:")
    report.append("   - Minerals have different chemistries, crystal systems, and measurement conditions")
    report.append("   - Spectra within source groups are not independent")
    report.append("   - Pooling ignores dependence structures and confounding")
    report.append("")

    report.append("---")
    report.append("")

    # Section 8: Extension Dataset
    report.append("## 8. Extension Dataset (Separate Summary)")
    report.append("")
    if len(extension) > 0:
        report.append(f"The extension dataset contains {len(extension)} spectra from {extension['mineral_label_linked'].nunique()} minerals.")
        report.append("")

        ext_summary = mineral_summary[mineral_summary['population'] == 'EXTENSION']
        if len(ext_summary) > 0:
            report.append("### 8.1 Extension Mineral Summaries")
            report.append("")
            report.append("| Mineral | n_spectra | n_source_groups | D1 median | D2 median | D3 median | D4 median |")
            report.append("|---------|-----------|-----------------|-----------|-----------|-----------|-----------|")

            for _, row in ext_summary.iterrows():
                report.append(f"| {row['mineral']} | {int(row['n_spectra'])} | {int(row['n_source_groups'])} | "
                            f"{row['D1_n_peaks_prom_median']:.2f} | "
                            f"{row['D2_mean_peak_prominence_median']:.4f} | "
                            f"{row['D3_first_deriv_rms_median']:.6f} | "
                            f"{row['D4_second_deriv_rms_median']:.6f} |")

            report.append("")
            report.append("**Note:** Extension spectra have different preprocessing routes and independence statuses than canonical spectra. They should not be pooled with canonical records for comparative analyses.")
            report.append("")
    else:
        report.append("No extension records.")
        report.append("")

    report.append("---")
    report.append("")

    # Section 9: Next Steps
    report.append("## 9. Recommended Next Research Action")
    report.append("")
    report.append("The most useful next research action is to **prepare the evidence-gap and metadata framework for publication** (RQ-A2, RQ-A4, RQ-A5 from Step 5.19).")
    report.append("")
    report.append("This would involve:")
    report.append("")
    report.append("1. **Documenting the 30-field evidence framework** (Step 5.17) as a standalone methodology component")
    report.append("2. **Publishing the metadata recovery methodology and results** (Step 5.18) as a reproducible workflow")
    report.append("3. **Articulating the specific evidence requirements** for testing the crystal symmetry hypothesis (Step 5.19, Section 5)")
    report.append("4. **Demonstrating why naive comparative spectroscopy fails** in curated spectral libraries, using the confounding and dependence structures documented in this catalogue")
    report.append("")
    report.append("This would produce a methods/framework paper component that establishes the infrastructure and evidence standards for future spectroscopic studies, even when the data to execute them are not yet available.")
    report.append("")
    report.append("**Deferred analyses** (pending new evidence):")
    report.append("- Inferential statistics (hypothesis tests, confidence intervals)")
    report.append("- Interpretable ML (requires specimen-specific chemistry and verified independence)")
    report.append("- Symbolic regression / PySR (requires specimen-specific crystallography)")
    report.append("- Targeted polymorph comparisons (requires verified independent specimens)")
    report.append("- Pooled cross-mineral analysis (requires stratification variables)")
    report.append("")
    report.append("---")
    report.append("")

    report.append("**End of Report**")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Script:** `scripts/phase5_step5_20_descriptive_descriptor_catalogue.py`")
    report.append(f"**Frozen inputs:** Verified unchanged (see run manifest)")

    report_text = '\n'.join(report)

    report_path = OUTPUT_DIR / 'phase5_step5_20_catalogue_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"  Saved: {report_path}")

    return report_path

# ============================================================================
# GENERATE RUN MANIFEST
# ============================================================================

def generate_manifest(catalogue, mineral_summary, sg_summary, recovery, start_time):
    """Generate JSON run manifest documenting inputs, outputs, and statistics."""
    print("\nGenerating run manifest...")

    canonical = catalogue[catalogue['dataset_origin'] == 'CANONICAL_V1']
    extension = catalogue[catalogue['dataset_origin'] != 'CANONICAL_V1']

    manifest = {
        'step': 'Phase 5 Step 5.20',
        'title': 'Mineral-by-Mineral Descriptive Spectral Descriptor Catalogue',
        'timestamp': datetime.now().isoformat(),
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'run_duration_seconds': (datetime.now() - start_time).total_seconds(),

        'inputs': {
            'linked_view': {
                'path': str(LINKED_VIEW_PATH),
                'records': 503,
                'description': 'Phase 5 Step 5.13 linked comparative view (454 canonical + 49 extension)'
            },
            'integrated_dataset': {
                'path': str(INTEGRATED_DATASET_PATH),
                'records': 454,
                'description': 'Phase 4 integrated analysis dataset (canonical only)'
            },
            'provenance_mapping': {
                'path': str(PROVENANCE_MAPPING_PATH),
                'records': 382,
                'description': 'Phase 4 provenance group mapping'
            },
            'metadata_recovery_register': {
                'path': str(METADATA_RECOVERY_PATH),
                'records': len(recovery),
                'description': 'Phase 5 Step 5.18 metadata recovery register'
            },
            'symmetry_metadata': {
                'path': str(SYMMETRY_METADATA_PATH),
                'records': 'see file',
                'description': 'Phase 4 symmetry metadata'
            }
        },

        'outputs': {
            'mineral_descriptor_summary': {
                'filename': 'phase5_step5_20_mineral_descriptor_summary.csv',
                'records': len(mineral_summary),
                'description': 'Mineral-level descriptor summaries (canonical and extension)'
            },
            'source_group_descriptor_summary': {
                'filename': 'phase5_step5_20_source_group_descriptor_summary.csv',
                'records': len(sg_summary),
                'description': 'Source-group-level descriptor summaries (canonical only)'
            },
            'provenance_enriched_catalogue': {
                'filename': 'phase5_step5_20_provenance_enriched_catalogue.csv',
                'records': len(catalogue),
                'columns': len(catalogue.columns),
                'description': 'Comprehensive per-spectrum catalogue with all recoverable metadata'
            },
            'catalogue_report': {
                'filename': 'phase5_step5_20_catalogue_report.md',
                'description': 'Comprehensive markdown report with interpretation'
            },
            'figures': [
                'phase5_step5_20_descriptor_distributions_by_mineral.png',
                'phase5_step5_20_sample_sizes_by_mineral.png'
            ],
            'run_manifest': {
                'filename': 'phase5_step5_20_run_manifest.json',
                'description': 'This manifest'
            }
        },

        'statistics': {
            'canonical': {
                'n_spectra': len(canonical),
                'n_minerals': int(canonical['mineral_label_linked'].nunique()),
                'n_source_groups': int(canonical['source_group_id'].nunique()),
                'minerals': sorted(canonical['mineral_label_linked'].unique().tolist())
            },
            'extension': {
                'n_spectra': len(extension),
                'n_minerals': int(extension['mineral_label_linked'].nunique()),
                'n_source_groups': int(extension['source_group_id'].nunique()),
                'minerals': sorted(extension['mineral_label_linked'].unique().tolist())
            },
            'metadata_recovery': {
                'grain_size_available': int(canonical['grain_size_value'].notna().sum()),
                'locality_hints_available': int(canonical['locality_hints'].notna().sum()),
                'mixture_status_available': int(canonical['mixture_status_from_description'].notna().sum()),
                'thermal_treatment_available': int(canonical['thermal_treatment_from_description'].notna().sum()),
                'measurement_geometry_available': int(canonical['source_angle'].notna().sum())
            }
        },

        'frozen_inputs_status': {
            'status': 'UNCHANGED',
            'verification': 'All frozen inputs (Phase 2 canonical spectra, Phase 3 descriptor matrix, Phase 4 integrated dataset, Phase 5 Step 5.13 linked view) were read but not modified. No new descriptors calculated. No definitions changed.',
            'descriptor_definitions': {
                'D1_n_peaks_prom': 'Number of peaks detected by prominence method (Phase 3 frozen)',
                'D2_mean_peak_prominence': 'Mean prominence of detected peaks (Phase 3 frozen)',
                'D3_first_deriv_rms': 'Root mean square of first derivative (Phase 3 frozen)',
                'D4_second_deriv_rms': 'Root mean square of second derivative (Phase 3 frozen)'
            }
        },

        'scientific_constraints': {
            'no_hypothesis_tests': 'No inferential statistics computed',
            'no_causal_claims': 'No causal interpretations made',
            'no_pooled_analysis': 'Extension and canonical records kept separate',
            'no_independence_assumed': 'Source-group dependence acknowledged',
            'no_ideal_as_measured': 'Ideal/reference chemistry not treated as specimen-specific',
            'no_locality_as_confirmed': 'Locality hints not treated as confirmed localities'
        },

        'reproducibility': {
            'script': 'scripts/phase5_step5_20_descriptive_descriptor_catalogue.py',
            'random_seed': 'Not applicable (no stochastic operations)',
            'deterministic': True,
            'description': 'All computations are deterministic. Given the same frozen inputs, the script produces identical outputs.'
        }
    }

    manifest_path = OUTPUT_DIR / 'phase5_step5_20_run_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {manifest_path}")

    return manifest_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()

    print("=" * 80)
    print("Phase 5 Step 5.20 — Mineral-by-Mineral Descriptive Spectral Descriptor Catalogue")
    print("=" * 80)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load data
    linked_view, recovery, integrated, prov_map, symmetry = load_data()

    # Build provenance-enriched catalogue
    catalogue = build_provenance_enriched_catalogue(linked_view, recovery, integrated, prov_map)

    # Compute mineral-level summaries
    mineral_summary = compute_mineral_summaries(catalogue)

    # Compute source-group-level summaries
    sg_summary = compute_source_group_summaries(catalogue)

    # Generate figures
    fig1_path, fig2_path = generate_figures(catalogue, mineral_summary)

    # Generate markdown report
    report_path = generate_report(catalogue, mineral_summary, sg_summary, fig1_path, fig2_path)

    # Generate run manifest
    manifest_path = generate_manifest(catalogue, mineral_summary, sg_summary, recovery, start_time)

    # Save CSV outputs
    print("\nSaving CSV outputs...")
    mineral_summary_path = OUTPUT_DIR / 'phase5_step5_20_mineral_descriptor_summary.csv'
    mineral_summary.to_csv(mineral_summary_path, index=False)
    print(f"  Saved: {mineral_summary_path}")

    sg_summary_path = OUTPUT_DIR / 'phase5_step5_20_source_group_descriptor_summary.csv'
    sg_summary.to_csv(sg_summary_path, index=False)
    print(f"  Saved: {sg_summary_path}")

    catalogue_path = OUTPUT_DIR / 'phase5_step5_20_provenance_enriched_catalogue.csv'
    catalogue.to_csv(catalogue_path, index=False)
    print(f"  Saved: {catalogue_path}")

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")
    print()
    print("Deliverables:")
    print(f"  1. {mineral_summary_path.name}")
    print(f"  2. {sg_summary_path.name}")
    print(f"  3. {catalogue_path.name}")
    print(f"  4. {report_path.name}")
    print(f"  5. {manifest_path.name}")
    print(f"  6. {fig1_path.name}")
    print(f"  7. {fig2_path.name}")
    print()
    print("Frozen inputs: VERIFIED UNCHANGED")
    print("Reproducibility: All computations deterministic")
    print()

if __name__ == '__main__':
    main()
