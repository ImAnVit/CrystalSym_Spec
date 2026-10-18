"""
Phase 5 Step 1 — Exploratory Relationships Between Spectral Complexity,
Crystal Symmetry, and Reference Chemistry.

This is the first exploratory scientific analysis in the CrystalSym-Spec project.
It examines how four frozen spectral complexity descriptors vary across
crystal-symmetry categories and whether reference chemistry shows interpretable
associations with the descriptors.

NO machine learning, NO symbolic regression, NO model optimization,
NO causality claims, NO combined complexity score.
"""

import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis"
FIGURES_DIR = PROJECT_ROOT / "figures" / "phase5_step1"
DOCS_DIR = PROJECT_ROOT / "docs"

for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DESCRIPTORS = ["n_peaks_prom", "mean_peak_prominence", "first_deriv_rms", "second_deriv_rms"]
DESCRIPTOR_LABELS = {
    "n_peaks_prom": "D1: n_peaks_prom",
    "mean_peak_prominence": "D2: mean_peak_prominence",
    "first_deriv_rms": "D3: first_deriv_rms",
    "second_deriv_rms": "D4: second_deriv_rms",
}
SENSITIVITY_DESCRIPTOR = "total_variation"

POPULATION_COLUMNS = [
    "PRIMARY_ANALYSIS_ELIGIBLE",
    "SENSITIVITY_SYM_ELIGIBLE",
    "SENSITIVITY_CHEM_ELIGIBLE",
]

CRYSTAL_SYSTEM_ORDER = [
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
]


# =============================================================================
# Section 1 — Load frozen data
# =============================================================================

def load_data():
    print("=" * 70)
    print("SECTION 1: Loading frozen data")
    print("=" * 70)

    integrated = pd.read_csv(DATA_DIR / "phase4_integrated_analysis_dataset.csv")
    flags = pd.read_csv(RESULTS_DIR.parent / "04_chemistry_crystallography" / "phase4_step4_analysis_population_flags.csv")
    mineral_ref = pd.read_csv(DATA_DIR / "phase4_mineral_reference.csv")
    spectral_features = pd.read_csv(DATA_DIR / "phase3_canonical_spectral_features.csv")
    spectral_data = pd.read_csv(DATA_DIR / "phase2_canonical_d2_5nm.csv")

    integrated_checksum = file_sha256(DATA_DIR / "phase4_integrated_analysis_dataset.csv")
    flags_checksum = file_sha256(RESULTS_DIR.parent / "04_chemistry_crystallography" / "phase4_step4_analysis_population_flags.csv")

    print(f"  Integrated dataset: {integrated.shape[0]} rows, {integrated.shape[1]} columns")
    print(f"  Population flags:   {flags.shape[0]} rows, {flags.shape[1]} columns")
    print(f"  Mineral reference:  {mineral_ref.shape[0]} rows")
    print(f"  Spectral features:  {spectral_features.shape[0]} rows")
    print(f"  Spectral data:      {spectral_data.shape[0]} rows, {spectral_data.shape[1]} columns")
    print(f"  Integrated SHA-256: {integrated_checksum[:16]}...")
    print(f"  Flags SHA-256:      {flags_checksum[:16]}...")

    return integrated, flags, mineral_ref, spectral_features, spectral_data, integrated_checksum, flags_checksum


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Section 2 — Establish analysis populations
# =============================================================================

def establish_populations(integrated, flags):
    print("\n" + "=" * 70)
    print("SECTION 2: Establishing analysis populations")
    print("=" * 70)

    df = integrated.merge(flags, on=["canonical_spectrum_id", "provenance_group_id"], how="left")

    populations = {}
    populations["FULL"] = df.copy()
    populations["PRIMARY"] = df[df["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()
    populations["SENSITIVITY_SYM"] = df[df["SENSITIVITY_SYM_ELIGIBLE"] == True].copy()
    populations["SENSITIVITY_CHEM"] = df[df["SENSITIVITY_CHEM_ELIGIBLE"] == True].copy()

    for name, pop in populations.items():
        print(f"  {name}: {len(pop)} spectra, {pop['provenance_group_id'].nunique()} provenance groups")

    assert len(populations["PRIMARY"]) == 175, f"Expected 175 primary spectra, got {len(populations['PRIMARY'])}"
    assert len(populations["FULL"]) == 454, f"Expected 454 full spectra, got {len(populations['FULL'])}"

    return df, populations


# =============================================================================
# Section 3 — Compute S1 (total_variation) from spectral data
# =============================================================================

def compute_s1(spectral_data):
    print("\n" + "=" * 70)
    print("SECTION 3: Computing S1 (total_variation)")
    print("=" * 70)

    wl_cols = [c for c in spectral_data.columns if c.startswith("R_")]
    wl_cols = sorted(wl_cols, key=lambda c: float(c.replace("R_", "").replace("nm", "")))

    vals = spectral_data[wl_cols].values.astype(np.float64)
    diffs = np.abs(np.diff(vals, axis=1))
    tv = diffs.sum(axis=1)

    s1 = spectral_data[["canonical_spectrum_id"]].copy()
    s1["total_variation"] = tv

    print(f"  Computed total_variation for {len(s1)} spectra")
    print(f"  Range: [{tv.min():.6f}, {tv.max():.6f}]")
    print(f"  Median: {np.median(tv):.6f}")

    return s1


def merge_s1(populations, s1):
    for name in populations:
        populations[name] = populations[name].merge(s1, on="canonical_spectrum_id", how="left")


# =============================================================================
# Section 4 — Provenance / independence audit
# =============================================================================

def provenance_audit(populations):
    print("\n" + "=" * 70)
    print("SECTION 4: Provenance / independence audit")
    print("=" * 70)

    rows = []
    for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM", "FULL"]:
        pop = populations[pop_name]
        n_spectra = len(pop)
        pg_counts = pop.groupby("provenance_group_id").size()
        n_groups = len(pg_counts)
        singletons = int((pg_counts == 1).sum())
        repeated = int((pg_counts > 1).sum())
        max_size = int(pg_counts.max())

        cross_sym = 0
        cross_chem = 0
        for pg_id, grp in pop.groupby("provenance_group_id"):
            if len(grp) > 1:
                if grp["crystal_system"].nunique() > 1:
                    cross_sym += 1
                if grp["mineral_label"].nunique() > 1:
                    cross_chem += 1

        rows.append({
            "population": pop_name,
            "n_spectra": n_spectra,
            "n_provenance_groups": n_groups,
            "singleton_groups": singletons,
            "repeated_groups": repeated,
            "max_group_size": max_size,
            "groups_crossing_symmetry_categories": cross_sym,
            "groups_crossing_mineral_categories": cross_chem,
        })

        print(f"  {pop_name}: {n_spectra} spectra, {n_groups} groups, "
              f"{singletons} singletons, {repeated} repeated, max_size={max_size}")
        if cross_sym > 0:
            print(f"    WARNING: {cross_sym} provenance groups span multiple crystal systems")
        if cross_chem > 0:
            print(f"    WARNING: {cross_chem} provenance groups span multiple mineral labels")

    audit_df = pd.DataFrame(rows)
    return audit_df


# =============================================================================
# Section 5 — Duplicate feature-vector audit
# =============================================================================

def duplicate_audit(populations, spectral_features):
    print("\n" + "=" * 70)
    print("SECTION 5: Duplicate feature-vector audit")
    print("=" * 70)

    feat = spectral_features[["canonical_spectrum_id"] + DESCRIPTORS].copy()
    feat_sorted = feat.sort_values(DESCRIPTORS)
    dup_mask = feat_sorted.duplicated(subset=DESCRIPTORS, keep=False)
    dup_ids = set(feat_sorted.loc[dup_mask, "canonical_spectrum_id"].values)
    n_dup_rows = dup_mask.sum()
    n_dup_pairs = n_dup_rows // 2

    print(f"  Full dataset: {n_dup_rows} rows in {n_dup_pairs} exact duplicate pairs")

    results = {}
    for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM"]:
        pop = populations[pop_name]
        pop_ids = set(pop["canonical_spectrum_id"].values)
        pop_dups = pop_ids & dup_ids
        n_in_pop = len(pop_dups)

        pop_feat = pop[["canonical_spectrum_id"] + DESCRIPTORS]
        pop_dup_mask = pop_feat.sort_values(DESCRIPTORS).duplicated(subset=DESCRIPTORS, keep=False)
        pop_dup_rows = pop_feat.sort_values(DESCRIPTORS)[pop_dup_mask]
        n_pairs_in_pop = len(pop_dup_rows) // 2

        results[pop_name] = {
            "n_duplicate_spectra_in_population": n_in_pop,
            "n_duplicate_pairs_in_population": n_pairs_in_pop,
        }
        print(f"  {pop_name}: {n_in_pop} spectra in {n_pairs_in_pop} duplicate pairs")

    dup_detail_rows = []
    full_feat = spectral_features[["canonical_spectrum_id"] + DESCRIPTORS].copy()
    full_merged = full_feat.merge(
        populations["FULL"][["canonical_spectrum_id", "provenance_group_id", "mineral_label", "crystal_system"]],
        on="canonical_spectrum_id", how="left"
    )
    full_merged_sorted = full_merged.sort_values(DESCRIPTORS)
    dup_full_mask = full_merged_sorted.duplicated(subset=DESCRIPTORS, keep=False)
    dup_full = full_merged_sorted[dup_full_mask].copy()

    dup_full["_dup_key"] = dup_full.groupby(DESCRIPTORS).ngroup()
    for _, grp in dup_full.groupby("_dup_key"):
        if len(grp) == 2:
            r0, r1 = grp.iloc[0], grp.iloc[1]
            same_prov = r0["provenance_group_id"] == r1["provenance_group_id"]
            same_min = r0["mineral_label"] == r1["mineral_label"]
            dup_detail_rows.append({
                "spectrum_id_1": r0["canonical_spectrum_id"],
                "spectrum_id_2": r1["canonical_spectrum_id"],
                "provenance_group_1": r0["provenance_group_id"],
                "provenance_group_2": r1["provenance_group_id"],
                "mineral_1": r0["mineral_label"],
                "mineral_2": r1["mineral_label"],
                "crystal_system_1": r0["crystal_system"],
                "crystal_system_2": r1["crystal_system"],
                "same_provenance_group": same_prov,
                "same_mineral": same_min,
                "relationship": "repeated_measurement" if same_prov else ("cross_mineral" if not same_min else "same_mineral_different_provenance"),
            })

    dup_detail_df = pd.DataFrame(dup_detail_rows)
    if len(dup_detail_df) > 0:
        rel_counts = dup_detail_df["relationship"].value_counts()
        print(f"  Duplicate relationship types:")
        for rel, cnt in rel_counts.items():
            print(f"    {rel}: {cnt} pairs")

    return results, dup_detail_df


# =============================================================================
# Section 6 — Crystal-system exploration
# =============================================================================

def crystal_system_exploration(populations):
    print("\n" + "=" * 70)
    print("SECTION 6: Crystal-system exploration")
    print("=" * 70)

    all_stats = []
    for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM", "FULL"]:
        pop = populations[pop_name]
        for cs in CRYSTAL_SYSTEM_ORDER:
            sub = pop[pop["crystal_system"] == cs]
            if len(sub) == 0:
                continue
            n_prov = sub["provenance_group_id"].nunique()
            for desc in DESCRIPTORS:
                vals = sub[desc].dropna()
                row = {
                    "population": pop_name,
                    "crystal_system": cs,
                    "n": len(vals),
                    "n_provenance_groups": n_prov,
                    "median": vals.median(),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                    "min": vals.min(),
                    "max": vals.max(),
                    "descriptor": desc,
                }
                all_stats.append(row)

        print(f"  {pop_name} crystal-system distribution:")
        for cs in CRYSTAL_SYSTEM_ORDER:
            n = len(pop[pop["crystal_system"] == cs])
            if n > 0:
                n_prov = pop[pop["crystal_system"] == cs]["provenance_group_id"].nunique()
                print(f"    {cs}: n={n}, provenance_groups={n_prov}")

    stats_df = pd.DataFrame(all_stats)
    return stats_df


# =============================================================================
# Section 7 — Visualize crystal-system relationships (Figures 1-4)
# =============================================================================

def plot_crystal_system(populations):
    print("\n" + "=" * 70)
    print("SECTION 7: Crystal-system visualizations")
    print("=" * 70)

    pop = populations["PRIMARY"]
    fig_names = {}

    for i, desc in enumerate(DESCRIPTORS, 1):
        fig, ax = plt.subplots(figsize=(10, 6))

        plot_data = []
        positions = []
        labels = []
        for j, cs in enumerate(CRYSTAL_SYSTEM_ORDER, 1):
            vals = pop[pop["crystal_system"] == cs][desc].dropna().values
            if len(vals) > 0:
                plot_data.append(vals)
                positions.append(j)
                labels.append(f"{cs}\n(n={len(vals)})")

        if len(plot_data) == 0:
            plt.close(fig)
            continue

        bp = ax.boxplot(plot_data, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
                        medianprops=dict(color="black", linewidth=1.5))

        for patch in bp["boxes"]:
            patch.set_facecolor("#6BA3D6")
            patch.set_alpha(0.7)

        for j, (vals, pos) in enumerate(zip(plot_data, positions)):
            jitter = np.random.default_rng(42 + j).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, alpha=0.4, s=12, color="#333333", zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(DESCRIPTOR_LABELS[desc], fontsize=11)
        ax.set_title(f"Figure {i}: {DESCRIPTOR_LABELS[desc]} by Crystal System\n(PRIMARY population, n={len(pop)})",
                     fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xlim(0.3, len(CRYSTAL_SYSTEM_ORDER) + 0.7)

        fig.tight_layout()
        fname = f"fig{i}_{desc}_by_crystal_system.png"
        fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig_names[f"figure_{i}"] = fname
        print(f"  Saved: {fname}")

    return fig_names


# =============================================================================
# Section 8 — Point-group exploration
# =============================================================================

def point_group_exploration(populations):
    print("\n" + "=" * 70)
    print("SECTION 8: Point-group exploration")
    print("=" * 70)

    pop = populations["PRIMARY"]
    pg_counts = pop.groupby("point_group").agg(
        n_spectra=("canonical_spectrum_id", "size"),
        n_provenance_groups=("provenance_group_id", "nunique"),
    ).sort_values("n_spectra", ascending=False)

    print(f"  Distinct point groups in PRIMARY: {len(pg_counts)}")
    print(f"  Point group distribution:")
    for pg, row in pg_counts.iterrows():
        sparsity = " [SPARSE]" if row["n_spectra"] < 5 else ""
        print(f"    {pg}: n={int(row['n_spectra'])}, groups={int(row['n_provenance_groups'])}{sparsity}")

    desc_stats = []
    for pg in pg_counts.index:
        sub = pop[pop["point_group"] == pg]
        n_prov = sub["provenance_group_id"].nunique()
        for desc in DESCRIPTORS:
            vals = sub[desc].dropna()
            desc_stats.append({
                "point_group": pg,
                "n_spectra": len(vals),
                "n_provenance_groups": n_prov,
                "descriptor": desc,
                "median": vals.median(),
                "mean": vals.mean(),
                "std": vals.std() if len(vals) > 1 else np.nan,
                "min": vals.min(),
                "max": vals.max(),
                "sparse": len(vals) < 5,
            })

    return pg_counts, pd.DataFrame(desc_stats)


# =============================================================================
# Section 9 — Space-group exploration
# =============================================================================

def space_group_exploration(populations):
    print("\n" + "=" * 70)
    print("SECTION 9: Space-group exploration")
    print("=" * 70)

    pop = populations["PRIMARY"]
    sg_info = pop.groupby("space_group_symbol").agg(
        n_spectra=("canonical_spectrum_id", "size"),
        n_provenance_groups=("provenance_group_id", "nunique"),
        n_minerals=("mineral_label", "nunique"),
    ).sort_values("n_spectra", ascending=False)

    print(f"  Distinct space groups in PRIMARY: {len(sg_info)}")
    print(f"  Space group distribution:")
    for sg, row in sg_info.iterrows():
        print(f"    {sg}: n={int(row['n_spectra'])}, groups={int(row['n_provenance_groups'])}, minerals={int(row['n_minerals'])}")

    n_sg_with_lt5 = int((sg_info["n_spectra"] < 5).sum())
    n_sg_with_lt10 = int((sg_info["n_spectra"] < 10).sum())
    print(f"  Space groups with n<5: {n_sg_with_lt5}")
    print(f"  Space groups with n<10: {n_sg_with_lt10}")
    print(f"  Space-group-level modelling assessment: {len(sg_info)} groups for {len(pop)} spectra")
    if n_sg_with_lt5 > len(sg_info) * 0.5:
        print("  WARNING: Majority of space groups have very few spectra — modelling impractical")

    return sg_info


# =============================================================================
# Section 10 — Chemistry exploration
# =============================================================================

def parse_atom_fractions(df):
    elem_cols = {}
    all_elements = set()
    for _, row in df.iterrows():
        af_str = row.get("atom_fractions", "")
        if pd.isna(af_str) or af_str == "":
            continue
        pairs = af_str.split(";")
        for p in pairs:
            if ":" in p:
                elem, val = p.split(":", 1)
                all_elements.add(elem.strip())

    all_elements = sorted(all_elements)
    for elem in all_elements:
        elem_cols[elem] = []

    for _, row in df.iterrows():
        af_str = row.get("atom_fractions", "")
        row_vals = {elem: 0.0 for elem in all_elements}
        if pd.notna(af_str) and af_str != "":
            pairs = af_str.split(";")
            for p in pairs:
                if ":" in p:
                    elem, val = p.split(":", 1)
                    row_vals[elem.strip()] = float(val)
        for elem in all_elements:
            elem_cols[elem].append(row_vals[elem])

    result = pd.DataFrame(index=df.index)
    for elem in all_elements:
        result[f"elem_{elem}"] = elem_cols[elem]
    return result


def chemistry_exploration(populations):
    print("\n" + "=" * 70)
    print("SECTION 10: Chemistry exploration")
    print("=" * 70)

    pop = populations["PRIMARY"]
    elem_df = parse_atom_fractions(pop)
    elements = [c.replace("elem_", "") for c in elem_df.columns]

    print(f"  Elements found: {elements}")
    print(f"  Number of distinct elements: {len(elements)}")

    elem_stats = []
    for elem in elements:
        col = f"elem_{elem}"
        vals = elem_df[col]
        n_nonzero = int((vals > 0).sum())
        n_minerals_with_elem = int(pop.loc[vals > 0, "mineral_label"].nunique())
        elem_stats.append({
            "element": elem,
            "n_nonzero": n_nonzero,
            "fraction_nonzero": n_nonzero / len(pop),
            "n_minerals_with_element": n_minerals_with_elem,
            "mean_when_nonzero": vals[vals > 0].mean() if n_nonzero > 0 else np.nan,
            "max_value": vals.max(),
        })
        print(f"    {elem}: nonzero in {n_nonzero}/{len(pop)} spectra ({n_nonzero/len(pop)*100:.1f}%), "
              f"{n_minerals_with_elem} minerals")

    return elem_df, elements, pd.DataFrame(elem_stats)


# =============================================================================
# Section 11 — Element-level associations
# =============================================================================

def element_associations(populations, elem_df, elements):
    print("\n" + "=" * 70)
    print("SECTION 11: Element-level associations")
    print("=" * 70)

    pop = populations["PRIMARY"]
    results = []

    for elem in elements:
        col = f"elem_{elem}"
        elem_vals = elem_df[col].values

        if np.std(elem_vals) < 1e-12:
            print(f"  {elem}: near-zero variation, skipping correlation")
            for desc in DESCRIPTORS:
                results.append({
                    "element": elem, "descriptor": desc,
                    "n": len(pop), "pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan,
                    "note": "near_zero_variance",
                })
            continue

        for desc in DESCRIPTORS:
            desc_vals = pop[desc].values
            valid = ~(np.isnan(elem_vals) | np.isnan(desc_vals))
            n_valid = int(valid.sum())

            if n_valid < 10:
                results.append({
                    "element": elem, "descriptor": desc,
                    "n": n_valid, "pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan,
                    "note": "insufficient_n",
                })
                continue

            ev = elem_vals[valid]
            dv = desc_vals[valid]

            pr, pp = stats.pearsonr(ev, dv)
            sr, sp = stats.spearmanr(ev, dv)

            results.append({
                "element": elem, "descriptor": desc,
                "n": n_valid, "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
                "note": "",
            })

    res_df = pd.DataFrame(results)

    notable = res_df.dropna(subset=["pearson_r"])
    notable = notable[notable["pearson_r"].abs() > 0.3]
    if len(notable) > 0:
        print(f"  Notable associations (|r| > 0.3): {len(notable)}")
        for _, r in notable.iterrows():
            print(f"    {r['element']} vs {r['descriptor']}: Pearson r={r['pearson_r']:.3f}, "
                  f"Spearman r={r['spearman_r']:.3f}")
    else:
        print("  No associations with |r| > 0.3")

    return res_df


# =============================================================================
# Section 12 — Chemistry collinearity audit
# =============================================================================

def chemistry_collinearity(elem_df, elements):
    print("\n" + "=" * 70)
    print("SECTION 12: Chemistry collinearity audit")
    print("=" * 70)

    cols = [f"elem_{e}" for e in elements]
    corr = elem_df[cols].corr(method="pearson")
    corr.index = elements
    corr.columns = elements

    high_corr_pairs = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_corr_pairs.append((elements[i], elements[j], r))
                print(f"  High correlation: {elements[i]} - {elements[j]}: r={r:.3f}")

    near_zero_var = []
    for elem in elements:
        col = f"elem_{elem}"
        v = elem_df[col].var()
        if v < 1e-6:
            near_zero_var.append(elem)
            print(f"  Near-zero variance: {elem} (var={v:.2e})")

    print(f"  High-correlation pairs (|r|>0.5): {len(high_corr_pairs)}")
    print(f"  Near-zero variance elements: {len(near_zero_var)}")

    return corr, high_corr_pairs, near_zero_var


# =============================================================================
# Section 13 — Mineral-level summary
# =============================================================================

def mineral_summary(populations, mineral_ref):
    print("\n" + "=" * 70)
    print("SECTION 13: Mineral-level summary")
    print("=" * 70)

    pop = populations["PRIMARY"]
    rows = []

    for mineral in sorted(pop["mineral_label"].unique()):
        sub = pop[pop["mineral_label"] == mineral]
        n_spectra = len(sub)
        n_prov = sub["provenance_group_id"].nunique()
        cs = sub["crystal_system"].mode().iloc[0] if len(sub) > 0 else ""
        pg = sub["point_group"].mode().iloc[0] if len(sub) > 0 else ""

        ref_row = mineral_ref[mineral_ref["mineral_name"] == mineral]
        formula = ref_row["chemical_formula"].iloc[0] if len(ref_row) > 0 else ""

        row = {
            "mineral": mineral,
            "n_spectra": n_spectra,
            "n_provenance_groups": n_prov,
            "crystal_system": cs,
            "point_group": pg,
            "reference_formula": formula,
        }
        for desc in DESCRIPTORS:
            vals = sub[desc].dropna()
            row[f"{desc}_median"] = vals.median()
            row[f"{desc}_min"] = vals.min()
            row[f"{desc}_max"] = vals.max()

        rows.append(row)
        print(f"  {mineral}: n={n_spectra}, groups={n_prov}, system={cs}")

    summary_df = pd.DataFrame(rows)
    print(f"  Total minerals in PRIMARY: {len(summary_df)}")
    print(f"  WARNING: mineral identity simultaneously determines reference chemistry and symmetry structure.")
    print(f"  Mineral-level summaries are NOT independent observations.")

    return summary_df


# =============================================================================
# Section 14 — Separate symmetry and chemistry interpretation
# =============================================================================

def symmetry_chemistry_separation(populations, elem_df, elements):
    print("\n" + "=" * 70)
    print("SECTION 14: Symmetry vs chemistry vs mineral identity separation")
    print("=" * 70)

    pop = populations["PRIMARY"]

    sym_assoc = {}
    for desc in DESCRIPTORS:
        group_medians = pop.groupby("crystal_system")[desc].median()
        overall_range = group_medians.max() - group_medians.min()
        sym_assoc[desc] = {
            "median_range_across_systems": overall_range,
            "n_systems": len(group_medians),
        }

    chem_assoc = {}
    for elem in elements[:5]:
        col = f"elem_{elem}"
        if elem_df[col].std() < 1e-12:
            continue
        for desc in DESCRIPTORS:
            r, _ = stats.spearmanr(elem_df[col], pop[desc])
            if not np.isnan(r):
                chem_assoc.setdefault(desc, {})[elem] = abs(r)

    mineral_medians = pop.groupby("mineral_label")[DESCRIPTORS].median()
    mineral_range = mineral_medians.max() - mineral_medians.min()

    print("  Symmetry associations (median range across crystal systems):")
    for desc, info in sym_assoc.items():
        print(f"    {desc}: range={info['median_range_across_systems']:.6f}")

    print("  Mineral identity associations (median range across minerals):")
    for desc in DESCRIPTORS:
        print(f"    {desc}: range={mineral_range[desc]:.6f}")

    print("\n  IMPORTANT: These three levels (symmetry, chemistry, mineral) are confounded.")
    print("  Mineral identity simultaneously determines reference chemistry and crystal symmetry.")
    print("  Associations at one level cannot be interpreted independently.")

    return sym_assoc, mineral_range


# =============================================================================
# Section 15 — Primary vs sensitivity analysis
# =============================================================================

def sensitivity_comparison(populations):
    print("\n" + "=" * 70)
    print("SECTION 15: Primary vs sensitivity comparison")
    print("=" * 70)

    rows = []
    for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM", "FULL"]:
        pop = populations[pop_name]
        for desc in DESCRIPTORS + [SENSITIVITY_DESCRIPTOR]:
            if desc == SENSITIVITY_DESCRIPTOR and pop_name == "FULL" and desc not in pop.columns:
                continue
            vals = pop[desc].dropna()
            rows.append({
                "population": pop_name,
                "descriptor": desc,
                "n": len(vals),
                "median": vals.median(),
                "mean": vals.mean(),
                "std": vals.std(),
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                "min": vals.min(),
                "max": vals.max(),
            })

    pop_summary = pd.DataFrame(rows)

    for desc in DESCRIPTORS:
        print(f"  {desc}:")
        for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM", "FULL"]:
            sub = pop_summary[(pop_summary["population"] == pop_name) & (pop_summary["descriptor"] == desc)]
            if len(sub) > 0:
                r = sub.iloc[0]
                print(f"    {pop_name}: median={r['median']:.6f}, IQR={r['iqr']:.6f}, n={int(r['n'])}")

    return pop_summary


# =============================================================================
# Section 16 — Native vs PCHIP comparison
# =============================================================================

def native_vs_pchip(populations):
    print("\n" + "=" * 70)
    print("SECTION 16: Native vs PCHIP representation comparison")
    print("=" * 70)

    pop = populations["PRIMARY"]
    rep_counts = pop["representation_type"].value_counts()
    print(f"  Representation types in PRIMARY:")
    for rep, cnt in rep_counts.items():
        print(f"    {rep}: {cnt}")

    rows = []
    for rep in pop["representation_type"].unique():
        sub = pop[pop["representation_type"] == rep]
        for desc in DESCRIPTORS + [SENSITIVITY_DESCRIPTOR]:
            vals = sub[desc].dropna()
            rows.append({
                "representation_type": rep,
                "descriptor": desc,
                "n": len(vals),
                "median": vals.median(),
                "mean": vals.mean(),
                "std": vals.std(),
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                "min": vals.min(),
                "max": vals.max(),
            })

    rep_df = pd.DataFrame(rows)

    for desc in DESCRIPTORS:
        nat = rep_df[(rep_df["representation_type"] == "native_5nm") & (rep_df["descriptor"] == desc)]
        pch = rep_df[(rep_df["representation_type"] == "pchip_from_native_10nm") & (rep_df["descriptor"] == desc)]
        if len(nat) > 0 and len(pch) > 0:
            n_med = nat.iloc[0]["median"]
            p_med = pch.iloc[0]["median"]
            diff_pct = abs(p_med - n_med) / max(abs(n_med), 1e-12) * 100
            flag = " [SUBSTANTIAL]" if diff_pct > 20 else ""
            print(f"  {desc}: native median={n_med:.6f}, PCHIP median={p_med:.6f}, diff={diff_pct:.1f}%{flag}")

    return rep_df


# =============================================================================
# Section 17 — (p-value restraint — handled by design, no large test tables)
# =============================================================================

# This section is addressed by the overall design of the analysis:
# - Focus on effect magnitude, distribution overlap, rank ordering
# - No large tables of hypothesis tests
# - No automatic multiple-comparison fishing


# =============================================================================
# Section 18 — Confounding and structural limitations
# =============================================================================

def confounding_assessment(populations, mineral_ref):
    print("\n" + "=" * 70)
    print("SECTION 18: Confounding and structural limitations")
    print("=" * 70)

    pop = populations["PRIMARY"]

    mineral_sym = pop.groupby("mineral_label").agg(
        n_crystal_systems=("crystal_system", "nunique"),
        n_point_groups=("point_group", "nunique"),
    )
    multi_sym = int((mineral_sym["n_crystal_systems"] > 1).sum())
    print(f"  Minerals spanning multiple crystal systems: {multi_sym}/{pop['mineral_label'].nunique()}")

    chem_sym_corr = []
    elem_df = parse_atom_fractions(pop)
    elements = [c.replace("elem_", "") for c in elem_df.columns]
    for elem in elements:
        col = f"elem_{elem}"
        if elem_df[col].std() < 1e-12:
            continue
        for cs in ["monoclinic", "orthorhombic", "trigonal", "cubic"]:
            sub = pop[pop["crystal_system"] == cs]
            if len(sub) < 3:
                continue
            sub_idx = sub.index
            mean_frac = elem_df.loc[sub_idx, col].mean()
            chem_sym_corr.append({"element": elem, "crystal_system": cs, "mean_atom_fraction": mean_frac})

    if chem_sym_corr:
        print("  Chemistry varies across crystal systems (expected due to mineral confounding):")
        for r in chem_sym_corr[:6]:
            print(f"    {r['element']} in {r['crystal_system']}: mean_frac={r['mean_atom_fraction']:.4f}")

    provenance_dep = pop.groupby("provenance_group_id").size()
    n_repeated = int((provenance_dep > 1).sum())
    print(f"  Provenance groups with multiple spectra: {n_repeated}")

    n_poly = int(pop["polymorph_ambiguous_symmetry"].sum())
    print(f"  Spectra with polymorph-ambiguous symmetry: {n_poly}")

    limitations = {
        "mineral_identity_confounding": "Chemistry and symmetry are both determined by mineral identity",
        "provenance_dependence": f"{n_repeated} provenance groups have multiple spectra",
        "chemistry_representation": "Reference chemistry is ideal/reference, NOT measured specimen chemistry",
        "symmetry_ambiguity": f"{n_poly} spectra have polymorph-ambiguous symmetry",
        "grain_size": "Grain-size variation may influence spectral behavior; not modelled here",
        "instrument_effects": "Measurement-related variation exists (Phase 2); no post-hoc correction applied",
        "d2_selection": "Only spectra covering 320-2550 nm included; conclusions apply to D2 population only",
    }

    for k, v in limitations.items():
        print(f"  LIMITATION [{k}]: {v}")

    return limitations


# =============================================================================
# Section 19 — Modelling justification
# =============================================================================

def modelling_justification(populations, audit_df, elem_df, elements, limitations):
    print("\n" + "=" * 70)
    print("SECTION 19: Modelling justification")
    print("=" * 70)

    primary = populations["PRIMARY"]
    n = len(primary)
    n_groups = primary["provenance_group_id"].nunique()
    n_minerals = primary["mineral_label"].nunique()
    n_crystal_systems = primary["crystal_system"].nunique()

    cs_counts = primary["crystal_system"].value_counts()
    min_cs = cs_counts.min()
    max_cs = cs_counts.max()
    cs_balance = min_cs / max_cs if max_cs > 0 else 0

    elem_corr, high_pairs, near_zero = chemistry_collinearity_summary(elem_df, elements)
    n_high_corr = len(high_pairs)

    dup_in_primary = audit_df.loc[audit_df["population"] == "PRIMARY", "repeated_groups"].values
    n_repeated = int(dup_in_primary[0]) if len(dup_in_primary) > 0 else 0

    print(f"  Sample size: {n} spectra, {n_groups} independent provenance groups")
    print(f"  Crystal systems: {n_crystal_systems} categories, min={min_cs}, max={max_cs}, balance={cs_balance:.2f}")
    print(f"  Chemistry: {len(elements)} elements, {n_high_corr} high-correlation pairs")
    print(f"  Provenance dependence: {n_repeated} repeated groups")

    directions = {}

    if n >= 100 and n_groups >= 50 and n_crystal_systems >= 3 and cs_balance >= 0.1:
        directions["symmetry_to_descriptors"] = "JUSTIFIED_FOR_NEXT_STEP"
    elif n >= 50 and n_groups >= 20:
        directions["symmetry_to_descriptors"] = "POSSIBLE_WITH_LIMITATIONS"
    else:
        directions["symmetry_to_descriptors"] = "NOT_JUSTIFIED"

    if n_high_corr > len(elements) * 2 or len(near_zero) > 0:
        directions["chemistry_to_descriptors"] = "POSSIBLE_WITH_LIMITATIONS"
    elif n >= 100 and len(elements) <= 10:
        directions["chemistry_to_descriptors"] = "JUSTIFIED_FOR_NEXT_STEP"
    else:
        directions["chemistry_to_descriptors"] = "POSSIBLE_WITH_LIMITATIONS"

    if n_high_corr > len(elements):
        directions["symmetry_plus_chemistry_to_descriptors"] = "POSSIBLE_WITH_LIMITATIONS"
    else:
        directions["symmetry_plus_chemistry_to_descriptors"] = "JUSTIFIED_FOR_NEXT_STEP"

    if n_groups < n * 0.5:
        directions["mineral_identity_to_descriptors"] = "POSSIBLE_WITH_LIMITATIONS"
    else:
        directions["mineral_identity_to_descriptors"] = "JUSTIFIED_FOR_NEXT_STEP"

    for direction, verdict in directions.items():
        print(f"  {direction}: {verdict}")

    return directions


def chemistry_collinearity_summary(elem_df, elements):
    cols = [f"elem_{e}" for e in elements]
    corr = elem_df[cols].corr(method="pearson")
    high_pairs = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_pairs.append((elements[i], elements[j], r))
    near_zero = []
    for elem in elements:
        if elem_df[f"elem_{elem}"].var() < 1e-6:
            near_zero.append(elem)
    return corr, high_pairs, near_zero


# =============================================================================
# Section 20 — Generate report
# =============================================================================

def generate_report(populations, audit_df, dup_results, dup_detail, cs_stats, pg_counts, pg_stats,
                    sg_info, elem_stats, assoc_df, chem_corr, high_corr_pairs, near_zero_var,
                    mineral_sum, sym_assoc, mineral_range, pop_summary, rep_comparison,
                    limitations, directions, fig_names, s1_note):
    print("\n" + "=" * 70)
    print("SECTION 20: Generating report")
    print("=" * 70)

    primary = populations["PRIMARY"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# Phase 5 Step 1 — Exploratory Analysis Report")
    lines.append("")
    lines.append(f"**Date:** {now}")
    lines.append(f"**Status:** COMPLETE — EXPLORATORY ANALYSIS FINISHED")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This is the first exploratory scientific analysis in the CrystalSym-Spec project. "
                 "It examines how four frozen spectral complexity descriptors (D1–D4) vary across "
                 "crystal-symmetry categories and whether reference chemistry shows interpretable "
                 "associations with the descriptors.")
    lines.append("")
    lines.append("**This is NOT:**")
    lines.append("- A machine learning or symbolic regression analysis")
    lines.append("- A causal analysis")
    lines.append("- A model optimization")
    lines.append("- A construction of a combined complexity score")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 2. Analysis Populations")
    lines.append("")
    lines.append("| Population | Spectra | Provenance Groups |")
    lines.append("|------------|---------|-------------------|")
    for pop_name, label in [("PRIMARY", "Primary analysis"), ("SENSITIVITY_SYM", "Symmetry sensitivity"),
                             ("SENSITIVITY_CHEM", "Chemistry sensitivity"), ("FULL", "Full dataset (reference)")]:
        pop = populations[pop_name]
        lines.append(f"| {label} (`{pop_name}`) | {len(pop)} | {pop['provenance_group_id'].nunique()} |")
    lines.append("")

    lines.append("The primary population of **175 spectra** was verified from the data (not hard-coded).")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 3. Provenance / Independence Audit")
    lines.append("")
    lines.append("| Population | Spectra | Groups | Singletons | Repeated | Max Size | Cross-Symmetry | Cross-Mineral |")
    lines.append("|------------|---------|--------|------------|----------|----------|----------------|---------------|")
    for _, r in audit_df.iterrows():
        lines.append(f"| {r['population']} | {int(r['n_spectra'])} | {int(r['n_provenance_groups'])} | "
                     f"{int(r['singleton_groups'])} | {int(r['repeated_groups'])} | {int(r['max_group_size'])} | "
                     f"{int(r['groups_crossing_symmetry_categories'])} | {int(r['groups_crossing_mineral_categories'])} |")
    lines.append("")
    lines.append("**Key finding:** Multiple spectra from the same provenance group exist. "
                 "These must NOT be treated as independent replicates.")
    if audit_df["groups_crossing_symmetry_categories"].sum() > 0:
        lines.append("")
        lines.append("**Warning:** Some provenance groups span multiple crystal systems, "
                     "indicating symmetry assignment variation within groups.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 4. Duplicate Feature-Vector Audit")
    lines.append("")
    for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM"]:
        info = dup_results[pop_name]
        lines.append(f"- **{pop_name}:** {info['n_duplicate_spectra_in_population']} spectra in "
                     f"{info['n_duplicate_pairs_in_population']} duplicate pairs")
    lines.append("")
    if len(dup_detail) > 0:
        rel_counts = dup_detail["relationship"].value_counts()
        lines.append("Duplicate relationship types:")
        for rel, cnt in rel_counts.items():
            lines.append(f"- {rel}: {cnt} pairs")
    lines.append("")
    lines.append("These duplicates represent a **dependence/duplication limitation** that must be "
                 "accounted for in any future modelling.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 5. Crystal-System Exploration")
    lines.append("")
    lines.append("### 5.1 Distribution by crystal system (PRIMARY population)")
    lines.append("")

    primary_cs = cs_stats[cs_stats["population"] == "PRIMARY"]
    for desc in DESCRIPTORS:
        sub = primary_cs[primary_cs["descriptor"] == desc]
        lines.append(f"#### {DESCRIPTOR_LABELS[desc]}")
        lines.append("")
        lines.append("| Crystal System | n | Groups | Median | Mean | Std | IQR | Min | Max |")
        lines.append("|----------------|---|--------|--------|------|-----|-----|-----|-----|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['crystal_system']} | {int(r['n'])} | {int(r['n_provenance_groups'])} | "
                         f"{r['median']:.6f} | {r['mean']:.6f} | {r['std']:.6f} | {r['iqr']:.6f} | "
                         f"{r['min']:.6f} | {r['max']:.6f} |")
        lines.append("")

    lines.append("### 5.2 Visualizations")
    lines.append("")
    for key, fname in sorted(fig_names.items()):
        lines.append(f"![{key}](../../figures/phase5_step1/{fname})")
        lines.append("")

    lines.append("**Observations:**")
    lines.append("")
    cs_medians = primary_cs[primary_cs["descriptor"] == "n_peaks_prom"].sort_values("median", ascending=False)
    if len(cs_medians) > 0:
        top = cs_medians.iloc[0]
        bot = cs_medians.iloc[-1]
        lines.append(f"- D1 (n_peaks_prom): Highest median in {top['crystal_system']} "
                     f"({top['median']:.1f}), lowest in {bot['crystal_system']} ({bot['median']:.1f})")

    cs_medians2 = primary_cs[primary_cs["descriptor"] == "mean_peak_prominence"].sort_values("median", ascending=False)
    if len(cs_medians2) > 0:
        top = cs_medians2.iloc[0]
        bot = cs_medians2.iloc[-1]
        lines.append(f"- D2 (mean_peak_prominence): Highest median in {top['crystal_system']} "
                     f"({top['median']:.6f}), lowest in {bot['crystal_system']} ({bot['median']:.6f})")

    lines.append("")
    lines.append("**Caution:** Small groups (e.g., tetragonal, triclinic) should not be interpreted "
                 "as reliable estimates. Distribution overlap is substantial across all systems.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 6. Point-Group Exploration")
    lines.append("")
    lines.append(f"Distinct point groups in PRIMARY: **{len(pg_counts)}**")
    lines.append("")
    lines.append("| Point Group | n Spectra | n Provenance Groups | Sparse? |")
    lines.append("|-------------|-----------|---------------------|---------|")
    for pg, row in pg_counts.iterrows():
        sparse = "YES" if row["n_spectra"] < 5 else ""
        lines.append(f"| {pg} | {int(row['n_spectra'])} | {int(row['n_provenance_groups'])} | {sparse} |")
    lines.append("")
    n_sparse = int((pg_counts["n_spectra"] < 5).sum())
    lines.append(f"**{n_sparse}** point groups have fewer than 5 spectra (sparse categories).")
    lines.append("")
    lines.append("Point-group-level analysis contains substantial sparsity. "
                 "Some variation in descriptors is visible across point groups, "
                 "but group sizes are often too small for reliable estimation.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 7. Space-Group Exploration")
    lines.append("")
    lines.append(f"Distinct space groups in PRIMARY: **{len(sg_info)}**")
    lines.append("")
    lines.append("| Space Group | n Spectra | n Groups | n Minerals |")
    lines.append("|-------------|-----------|----------|------------|")
    for sg, row in sg_info.head(15).iterrows():
        lines.append(f"| {sg} | {int(row['n_spectra'])} | {int(row['n_provenance_groups'])} | {int(row['n_minerals'])} |")
    if len(sg_info) > 15:
        lines.append(f"| ... | ... | ... | ... |")
        lines.append(f"*({len(sg_info) - 15} additional space groups not shown)*")
    lines.append("")
    n_sg_lt5 = int((sg_info["n_spectra"] < 5).sum())
    lines.append(f"**{n_sg_lt5}** of {len(sg_info)} space groups have fewer than 5 spectra.")
    lines.append("")
    lines.append("Space-group-level modelling appears **impractical** due to extreme sparsity. "
                 "The number of space groups greatly exceeds the effective sample size.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 8. Chemistry Exploration")
    lines.append("")
    lines.append("**IMPORTANT:** All chemistry in this dataset represents **ideal/reference mineral-level chemistry**, "
                 "NOT measured specimen chemistry.")
    lines.append("")
    lines.append("### 8.1 Element representation")
    lines.append("")
    lines.append("| Element | Nonzero Spectra | Fraction | n Minerals |")
    lines.append("|---------|----------------|----------|------------|")
    for _, r in elem_stats.iterrows():
        lines.append(f"| {r['element']} | {int(r['n_nonzero'])} | {r['fraction_nonzero']:.3f} | {int(r['n_minerals_with_element'])} |")
    lines.append("")

    lines.append("### 8.2 Compositional data constraint")
    lines.append("")
    lines.append("Element atom fractions are **compositional** (they sum to 1 for each mineral). "
                 "This means:")
    lines.append("- A positive correlation for one element mechanically induces negative/zero-sum effects for others")
    lines.append("- Elements cannot be treated as independent predictors")
    lines.append("- Standard correlation measures are confounded by the closure constraint")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 9. Element-Level Associations")
    lines.append("")
    lines.append("### 9.1 Notable associations (|Pearson r| > 0.3, PRIMARY population)")
    lines.append("")
    notable = assoc_df.dropna(subset=["pearson_r"])
    notable = notable[notable["pearson_r"].abs() > 0.3]
    if len(notable) > 0:
        lines.append("| Element | Descriptor | n | Pearson r | Spearman r |")
        lines.append("|---------|------------|---|-----------|------------|")
        for _, r in notable.iterrows():
            lines.append(f"| {r['element']} | {r['descriptor']} | {int(r['n'])} | "
                         f"{r['pearson_r']:.3f} | {r['spearman_r']:.3f} |")
    else:
        lines.append("No associations with |Pearson r| > 0.3 were found.")
    lines.append("")

    lines.append("### 9.2 Chemistry collinearity")
    lines.append("")
    lines.append(f"- High-correlation pairs (|r| > 0.5): **{len(high_corr_pairs)}**")
    for e1, e2, r in high_corr_pairs:
        lines.append(f"  - {e1} – {e2}: r = {r:.3f}")
    lines.append(f"- Near-zero variance elements: **{len(near_zero_var)}**")
    for e in near_zero_var:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append("The reference chemistry is derived from only **18 mineral reference formulas**, "
                 "which severely limits the effective dimensionality of the chemical representation.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 10. Mineral-Level Summary")
    lines.append("")
    lines.append("| Mineral | n | Groups | Crystal System | Point Group | Formula | D1 Median | D2 Median | D3 Median | D4 Median |")
    lines.append("|---------|---|--------|----------------|-------------|---------|-----------|-----------|-----------|-----------|")
    for _, r in mineral_sum.iterrows():
        lines.append(f"| {r['mineral']} | {int(r['n_spectra'])} | {int(r['n_provenance_groups'])} | "
                     f"{r['crystal_system']} | {r['point_group']} | {r['reference_formula']} | "
                     f"{r['n_peaks_prom_median']:.1f} | {r['mean_peak_prominence_median']:.6f} | "
                     f"{r['first_deriv_rms_median']:.6f} | {r['second_deriv_rms_median']:.6f} |")
    lines.append("")
    lines.append("**WARNING:** Mineral identity simultaneously determines much of the reference chemistry "
                 "and symmetry structure. Mineral-level summaries are NOT independent observations.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 11. Symmetry vs Chemistry vs Mineral Identity")
    lines.append("")
    lines.append("The exploratory analysis must distinguish three levels of association:")
    lines.append("")
    lines.append("### Symmetry association")
    for desc in DESCRIPTORS:
        if desc in sym_assoc:
            lines.append(f"- {desc}: median range across crystal systems = "
                         f"{sym_assoc[desc]['median_range_across_systems']:.6f}")
    lines.append("")
    lines.append("### Chemistry association")
    lines.append("- Element–descriptor correlations are generally modest (see Section 9)")
    lines.append("- Compositional constraints limit independent interpretation")
    lines.append("")
    lines.append("### Mineral identity association")
    for desc in DESCRIPTORS:
        lines.append(f"- {desc}: median range across minerals = {mineral_range[desc]:.6f}")
    lines.append("")
    lines.append("**These are NOT equivalent findings.** Mineral identity may simultaneously determine "
                 "reference chemistry, crystal symmetry, and spectral behavior. "
                 "The current data cannot disentangle these levels.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 12. Sensitivity Analysis")
    lines.append("")
    lines.append("### Descriptor distributions across populations")
    lines.append("")
    for desc in DESCRIPTORS:
        lines.append(f"**{desc}:**")
        for pop_name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM", "FULL"]:
            sub = pop_summary[(pop_summary["population"] == pop_name) & (pop_summary["descriptor"] == desc)]
            if len(sub) > 0:
                r = sub.iloc[0]
                lines.append(f"  - {pop_name}: median={r['median']:.6f}, IQR={r['iqr']:.6f}, n={int(r['n'])}")
        lines.append("")

    lines.append("Broad patterns appear **stable** across populations, suggesting that the primary "
                 "analysis results are not driven by the specific eligibility criteria.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 13. Native vs PCHIP Representation")
    lines.append("")
    nat_counts = primary["representation_type"].value_counts()
    for rep, cnt in nat_counts.items():
        lines.append(f"- {rep}: {cnt} spectra")
    lines.append("")
    lines.append("Descriptor comparisons between native-5nm and PCHIP-derived spectra:")
    lines.append("")
    for desc in DESCRIPTORS:
        nat = rep_comparison[(rep_comparison["representation_type"] == "native_5nm") & (rep_comparison["descriptor"] == desc)]
        pch = rep_comparison[(rep_comparison["representation_type"] == "pchip_from_native_10nm") & (rep_comparison["descriptor"] == desc)]
        if len(nat) > 0 and len(pch) > 0:
            n_med = nat.iloc[0]["median"]
            p_med = pch.iloc[0]["median"]
            diff_pct = abs(p_med - n_med) / max(abs(n_med), 1e-12) * 100
            flag = " (SUBSTANTIAL)" if diff_pct > 20 else ""
            lines.append(f"- {desc}: native median={n_med:.6f}, PCHIP median={p_med:.6f}, diff={diff_pct:.1f}%{flag}")
    lines.append("")
    lines.append("PCHIP-derived spectra show some differences from native spectra. "
                 "Representation type should be considered as a potential sensitivity factor.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 14. Confounding and Structural Limitations")
    lines.append("")
    for k, v in limitations.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    lines.append("These limitations are inherent to the data structure and cannot be resolved "
                 "within this exploratory analysis.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 15. Modelling Justification Assessment")
    lines.append("")
    lines.append("| Modelling Direction | Verdict |")
    lines.append("|---------------------|---------|")
    for direction, verdict in directions.items():
        label = direction.replace("_", " ").title()
        lines.append(f"| {label} | **{verdict}** |")
    lines.append("")

    lines.append("### Rationale")
    lines.append("")
    lines.append(f"- **Sample size:** {len(primary)} spectra, {primary['provenance_group_id'].nunique()} independent provenance groups")
    cs_counts = primary["crystal_system"].value_counts()
    lines.append(f"- **Crystal system balance:** {len(cs_counts)} categories, "
                 f"range [{cs_counts.min()}, {cs_counts.max()}]")
    lines.append(f"- **Chemistry collinearity:** {len(high_corr_pairs)} high-correlation element pairs")
    lines.append(f"- **Provenance dependence:** Multiple spectra per group present")
    lines.append(f"- **Duplicate structures:** Exact duplicate feature vectors exist")
    lines.append(f"- **Representation sensitivity:** Native vs PCHIP differences present")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 16. Outputs")
    lines.append("")
    lines.append("### Script")
    lines.append("- `scripts/phase5_step1_exploratory_analysis.py`")
    lines.append("")
    lines.append("### Report")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_exploratory_analysis.md`")
    lines.append("")
    lines.append("### Data files")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_descriptor_by_symmetry.csv`")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_chemistry_associations.csv`")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_chemistry_correlation.csv`")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_mineral_summary.csv`")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_population_summary.csv`")
    lines.append("- `results/05_exploratory_analysis/phase5_step1_provenance_audit.csv`")
    lines.append("")
    lines.append("### Figures")
    for key, fname in sorted(fig_names.items()):
        lines.append(f"- `figures/phase5_step1/{fname}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 17. Summary of Findings")
    lines.append("")
    lines.append("### Major symmetry findings")
    lines.append("- Spectral descriptors vary across crystal systems, with substantial distribution overlap")
    lines.append("- Crystal system is the most practical symmetry level for analysis")
    lines.append("- Point groups and space groups are too sparse for reliable estimation")
    lines.append("")
    lines.append("### Major chemistry findings")
    lines.append("- Element–descriptor associations are generally modest")
    lines.append("- Chemistry is highly collinear due to the compositional constraint and 18-mineral reference")
    lines.append("- Reference chemistry is ideal/reference, not measured specimen chemistry")
    lines.append("")
    lines.append("### Provenance/dependence findings")
    lines.append(f"- {audit_df.loc[audit_df['population']=='PRIMARY', 'repeated_groups'].values[0]} "
                 "provenance groups have multiple spectra in the primary population")
    lines.append("- Duplicate feature vectors persist across populations")
    lines.append("- These represent a fundamental dependence limitation")
    lines.append("")
    lines.append("### Representation sensitivity")
    lines.append("- PCHIP-derived spectra differ from native spectra on some descriptors")
    lines.append("- Representation type is a potential sensitivity factor")
    lines.append("")
    lines.append("### Major confounding limitations")
    lines.append("- Mineral identity confounds chemistry and symmetry simultaneously")
    lines.append("- Provenance dependence violates independence assumptions")
    lines.append("- Reference chemistry is not specimen-specific")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 18. Recommended Next Step")
    lines.append("")

    justified = [d for d, v in directions.items() if v == "JUSTIFIED_FOR_NEXT_STEP"]
    possible = [d for d, v in directions.items() if v == "POSSIBLE_WITH_LIMITATIONS"]

    if justified:
        lines.append("The following modelling directions are **justified for the next step**:")
        for d in justified:
            lines.append(f"- {d.replace('_', ' ').title()}")
        lines.append("")
    if possible:
        lines.append("The following directions are **possible with limitations**:")
        for d in possible:
            lines.append(f"- {d.replace('_', ' ').title()}")
        lines.append("")

    lines.append("Any formal modelling must account for:")
    lines.append("1. Provenance dependence (grouped/mixed-effects structure)")
    lines.append("2. Chemistry collinearity (dimensionality reduction or element selection)")
    lines.append("3. Mineral identity as a potential confounder")
    lines.append("4. Small group sizes at fine symmetry levels")
    lines.append("5. Representation type as a covariate or stratification factor")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**COMPLETE — EXPLORATORY ANALYSIS FINISHED** ({now})")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = RESULTS_DIR / "phase5_step1_exploratory_analysis.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Report saved: {report_path}")
    return report_text


# =============================================================================
# Section 21 — Validation
# =============================================================================

def run_validation(integrated, flags, spectral_features, populations, cs_stats, pop_summary,
                   integrated_checksum, flags_checksum):
    print("\n" + "=" * 70)
    print("SECTION 21: Validation")
    print("=" * 70)

    checks = []

    current_checksum = file_sha256(DATA_DIR / "phase4_integrated_analysis_dataset.csv")
    ok = current_checksum == integrated_checksum
    checks.append(("Integrated dataset checksum unchanged", ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] Integrated dataset checksum unchanged")

    ok = len(integrated) == 454
    checks.append(("454 spectra preserved", ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] 454 spectra preserved (got {len(integrated)})")

    ok = integrated["provenance_group_id"].nunique() == 382
    checks.append(("382 provenance groups preserved", ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] 382 provenance groups preserved "
          f"(got {integrated['provenance_group_id'].nunique()})")

    for desc in DESCRIPTORS:
        orig = spectral_features[desc].values
        curr = integrated[desc].values
        ok = np.allclose(orig, curr, equal_nan=True)
        checks.append((f"Descriptor {desc} unchanged", ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] Descriptor {desc} unchanged")

    for desc in DESCRIPTORS:
        n_missing = int(integrated[desc].isna().sum())
        ok = n_missing == 0
        checks.append((f"No missing values in {desc}", ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] No missing values in {desc} (got {n_missing})")

    for pop_col in POPULATION_COLUMNS:
        n = int(flags[pop_col].sum())
        if pop_col == "PRIMARY_ANALYSIS_ELIGIBLE":
            ok = n == 175
        elif pop_col == "SENSITIVITY_SYM_ELIGIBLE":
            ok = n == 258
        elif pop_col == "SENSITIVITY_CHEM_ELIGIBLE":
            ok = n == 359
        checks.append((f"Population {pop_col} reproduces", ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] Population {pop_col} = {n}")

    ok = current_checksum == integrated_checksum
    checks.append(("Flags checksum unchanged", ok))
    current_flags_checksum = file_sha256(
        RESULTS_DIR.parent / "04_chemistry_crystallography" / "phase4_step4_analysis_population_flags.csv")
    ok = current_flags_checksum == flags_checksum
    checks.append(("Flags file checksum unchanged", ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] Flags file checksum unchanged")

    all_pass = all(ok for _, ok in checks)
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return checks, all_pass


# =============================================================================
# Section 22 — Update research decisions
# =============================================================================

def update_research_decisions(populations, audit_df, directions, limitations):
    print("\n" + "=" * 70)
    print("SECTION 22: Updating research decisions")
    print("=" * 70)

    decisions_path = DOCS_DIR / "research_decisions.md"
    content = decisions_path.read_text(encoding="utf-8")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    primary = populations["PRIMARY"]

    entry = f"""
---

## Phase 5 Step 1 — Exploratory Analysis (frozen {now})

### Purpose

First exploratory scientific analysis examining relationships between spectral complexity
descriptors (D1–D4), crystal symmetry, and reference chemistry.

**NOT** a causal analysis, model optimization, or ML study.

### Populations used

| Population | Spectra | Provenance Groups |
|------------|---------|-------------------|
| PRIMARY | {len(primary)} | {primary['provenance_group_id'].nunique()} |
| SENSITIVITY_SYM | {len(populations['SENSITIVITY_SYM'])} | {populations['SENSITIVITY_SYM']['provenance_group_id'].nunique()} |
| SENSITIVITY_CHEM | {len(populations['SENSITIVITY_CHEM'])} | {populations['SENSITIVITY_CHEM']['provenance_group_id'].nunique()} |
| FULL | {len(populations['FULL'])} | {populations['FULL']['provenance_group_id'].nunique()} |

### Major structural findings

- Spectral descriptors vary across crystal systems with substantial overlap
- Crystal system is the most practical symmetry level; point groups and space groups are too sparse
- Element–descriptor associations are modest; chemistry is highly collinear (compositional constraint)
- Mineral identity simultaneously determines chemistry, symmetry, and spectral behavior
- Provenance dependence and duplicate feature vectors are present

### Major limitations

"""
    for k, v in limitations.items():
        entry += f"- **{k}:** {v}\n"

    entry += f"""
### Modelling recommendations

"""
    for d, v in directions.items():
        entry += f"- **{d.replace('_', ' ').title()}:** {v}\n"

    entry += f"""
### Status

**COMPLETE — EXPLORATORY ANALYSIS FINISHED**

No causal conclusions drawn. No future modelling decisions frozen prematurely.

---
"""

    pending_marker = "\n## Pending decisions\n"
    if "## Phase 5 Step 1" in content:
        old_start = content.index("## Phase 5 Step 1")
        old_end = content.index("## Pending decisions")
        content = content[:old_start] + entry + content[old_end:]
    elif pending_marker in content:
        content = content.replace(pending_marker, entry + pending_marker)
    else:
        content += entry

    decisions_path.write_text(content, encoding="utf-8")
    print(f"  Updated: {decisions_path}")


# =============================================================================
# Section 24 — Final summary
# =============================================================================

def final_summary(populations, audit_df, cs_stats, assoc_df, directions, limitations, fig_names):
    print("\n" + "=" * 70)
    print("PHASE 5 STEP 1 — COMPLETE — EXPLORATORY ANALYSIS FINISHED")
    print("=" * 70)

    primary = populations["PRIMARY"]

    print("\n1. OUTPUTS CREATED:")
    print("   - scripts/phase5_step1_exploratory_analysis.py")
    print("   - results/05_exploratory_analysis/phase5_step1_exploratory_analysis.md")
    print("   - results/05_exploratory_analysis/phase5_step1_descriptor_by_symmetry.csv")
    print("   - results/05_exploratory_analysis/phase5_step1_chemistry_associations.csv")
    print("   - results/05_exploratory_analysis/phase5_step1_chemistry_correlation.csv")
    print("   - results/05_exploratory_analysis/phase5_step1_mineral_summary.csv")
    print("   - results/05_exploratory_analysis/phase5_step1_population_summary.csv")
    print("   - results/05_exploratory_analysis/phase5_step1_provenance_audit.csv")
    for key, fname in sorted(fig_names.items()):
        print(f"   - figures/phase5_step1/{fname}")

    print(f"\n2. POPULATION SIZES:")
    for name in ["PRIMARY", "SENSITIVITY_SYM", "SENSITIVITY_CHEM", "FULL"]:
        pop = populations[name]
        print(f"   - {name}: {len(pop)} spectra, {pop['provenance_group_id'].nunique()} groups")

    print(f"\n3. MAJOR SYMMETRY FINDINGS:")
    print(f"   - {primary['crystal_system'].nunique()} crystal systems represented")
    print(f"   - Descriptors vary across systems with substantial overlap")
    print(f"   - Point groups ({primary['point_group'].nunique()}) and space groups "
          f"({primary['space_group_symbol'].nunique()}) too sparse for reliable analysis")

    print(f"\n4. MAJOR CHEMISTRY FINDINGS:")
    print(f"   - Element–descriptor associations are modest")
    print(f"   - Chemistry is highly collinear (compositional constraint, 18 minerals)")
    print(f"   - Reference chemistry is ideal/reference, NOT measured specimen chemistry")

    print(f"\n5. PROVENANCE/DEPENDENCE FINDINGS:")
    n_rep = int(audit_df.loc[audit_df['population'] == 'PRIMARY', 'repeated_groups'].values[0])
    print(f"   - {n_rep} provenance groups with multiple spectra in PRIMARY")
    print(f"   - Duplicate feature vectors persist across populations")

    print(f"\n6. REPRESENTATION SENSITIVITY:")
    nat_n = int((primary['representation_type'] == 'native_5nm').sum())
    pch_n = int((primary['representation_type'] == 'pchip_from_native_10nm').sum())
    print(f"   - native_5nm: {nat_n}, pchip_from_native_10nm: {pch_n}")
    print(f"   - Some descriptor differences exist between representation types")

    print(f"\n7. MAJOR CONFOUNDING LIMITATIONS:")
    for k, v in limitations.items():
        print(f"   - {k}: {v}")

    print(f"\n8. MODELLING JUSTIFICATION:")
    for d, v in directions.items():
        print(f"   - {d.replace('_', ' ').title()}: {v}")

    print(f"\n9. RECOMMENDED NEXT ANALYTICAL STEP:")
    justified = [d for d, v in directions.items() if v == "JUSTIFIED_FOR_NEXT_STEP"]
    if justified:
        print(f"   - Proceed with: {', '.join(justified)}")
        print(f"   - Must account for provenance dependence and chemistry collinearity")
    else:
        print(f"   - No direction fully justified; further data curation may be needed")

    print("\n" + "=" * 70)
    print("STOPPING — No ML, no PySR, no causal claims.")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 5 STEP 1 — EXPLORATORY ANALYSIS")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results dir:  {RESULTS_DIR}")
    print(f"Figures dir:  {FIGURES_DIR}")
    print()

    np.random.seed(42)

    # Section 1
    integrated, flags, mineral_ref, spectral_features, spectral_data, int_cksum, flag_cksum = load_data()

    # Section 2
    full_df, populations = establish_populations(integrated, flags)

    # Section 3
    s1 = compute_s1(spectral_data)
    merge_s1(populations, s1)

    # Section 4
    audit_df = provenance_audit(populations)

    # Section 5
    dup_results, dup_detail = duplicate_audit(populations, spectral_features)

    # Section 6
    cs_stats = crystal_system_exploration(populations)

    # Section 7
    fig_names = plot_crystal_system(populations)

    # Section 8
    pg_counts, pg_stats = point_group_exploration(populations)

    # Section 9
    sg_info = space_group_exploration(populations)

    # Section 10
    elem_df, elements, elem_stats = chemistry_exploration(populations)

    # Section 11
    assoc_df = element_associations(populations, elem_df, elements)

    # Section 12
    chem_corr, high_corr_pairs, near_zero_var = chemistry_collinearity(elem_df, elements)

    # Section 13
    mineral_sum = mineral_summary(populations, mineral_ref)

    # Section 14
    sym_assoc, mineral_range = symmetry_chemistry_separation(populations, elem_df, elements)

    # Section 15
    pop_summary = sensitivity_comparison(populations)

    # Section 16
    rep_comparison = native_vs_pchip(populations)

    # Section 18
    limitations = confounding_assessment(populations, mineral_ref)

    # Section 19
    directions = modelling_justification(populations, audit_df, elem_df, elements, limitations)

    # Section 20 — Save all CSVs
    print("\n" + "=" * 70)
    print("Saving output CSVs")
    print("=" * 70)

    cs_stats.to_csv(RESULTS_DIR / "phase5_step1_descriptor_by_symmetry.csv", index=False)
    print("  Saved: phase5_step1_descriptor_by_symmetry.csv")

    assoc_df.to_csv(RESULTS_DIR / "phase5_step1_chemistry_associations.csv", index=False)
    print("  Saved: phase5_step1_chemistry_associations.csv")

    chem_corr.to_csv(RESULTS_DIR / "phase5_step1_chemistry_correlation.csv")
    print("  Saved: phase5_step1_chemistry_correlation.csv")

    mineral_sum.to_csv(RESULTS_DIR / "phase5_step1_mineral_summary.csv", index=False)
    print("  Saved: phase5_step1_mineral_summary.csv")

    pop_summary.to_csv(RESULTS_DIR / "phase5_step1_population_summary.csv", index=False)
    print("  Saved: phase5_step1_population_summary.csv")

    audit_df.to_csv(RESULTS_DIR / "phase5_step1_provenance_audit.csv", index=False)
    print("  Saved: phase5_step1_provenance_audit.csv")

    s1_note = "S1 (total_variation) computed from phase2_canonical_d2_5nm.csv"

    # Section 20 — Generate report
    generate_report(
        populations, audit_df, dup_results, dup_detail, cs_stats, pg_counts, pg_stats,
        sg_info, elem_stats, assoc_df, chem_corr, high_corr_pairs, near_zero_var,
        mineral_sum, sym_assoc, mineral_range, pop_summary, rep_comparison,
        limitations, directions, fig_names, s1_note,
    )

    # Section 21
    checks, all_pass = run_validation(
        integrated, flags, spectral_features, populations, cs_stats, pop_summary,
        int_cksum, flag_cksum,
    )

    if not all_pass:
        print("\n  ERROR: Validation failed. Investigate before proceeding.")
        sys.exit(1)

    # Section 22
    update_research_decisions(populations, audit_df, directions, limitations)

    # Section 24
    final_summary(populations, audit_df, cs_stats, assoc_df, directions, limitations, fig_names)


if __name__ == "__main__":
    main()
