"""
Phase 5 Step 4 — Coverage Baseline Computation

Derives a verified baseline of the current dataset from frozen Phase 4 inputs.
All outputs are read-only; no frozen data are modified.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis"

INTEGRATED = DATA_DIR / "phase4_integrated_analysis_dataset.csv"
SYMMETRY = DATA_DIR / "phase4_symmetry_metadata.csv"
MINERAL_REF = DATA_DIR / "phase4_mineral_reference.csv"
PROVENANCE = DATA_DIR / "phase4_provenance_mapping.csv"
CHEMISTRY = DATA_DIR / "phase4_chemistry_metadata.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_baseline():
    integrated = read_csv(INTEGRATED)
    symmetry = read_csv(SYMMETRY)
    mineral_ref = read_csv(MINERAL_REF)
    provenance = read_csv(PROVENANCE)
    chemistry = read_csv(CHEMISTRY)

    n_spectra = len(integrated)
    
    # --- Provenance groups ---
    group_ids = set(row["provenance_group_id"] for row in integrated)
    n_groups = len(group_ids)
    
    # --- Minerals ---
    minerals = set(row["mineral_label"] for row in integrated)
    n_minerals = len(minerals)
    
    # --- Crystal systems ---
    systems = set(row["crystal_system"] for row in integrated if row["crystal_system"])
    n_systems = len(systems)
    
    # --- Minerals per crystal system ---
    mineral_to_system = {}
    for row in integrated:
        m = row["mineral_label"]
        s = row["crystal_system"]
        if m and s:
            mineral_to_system[m] = s
    
    system_to_minerals = defaultdict(set)
    for m, s in mineral_to_system.items():
        system_to_minerals[s].add(m)
    
    minerals_per_system = {s: len(ms) for s, ms in system_to_minerals.items()}
    
    # --- Spectra and groups per mineral ---
    spectra_per_mineral = Counter(row["mineral_label"] for row in integrated)
    groups_per_mineral = defaultdict(set)
    for row in integrated:
        groups_per_mineral[row["mineral_label"]].add(row["provenance_group_id"])
    groups_per_mineral_counts = {m: len(g) for m, g in groups_per_mineral.items()}
    
    # --- Spectra and groups per crystal system ---
    spectra_per_system = Counter(row["crystal_system"] for row in integrated if row["crystal_system"])
    groups_per_system = defaultdict(set)
    for row in integrated:
        if row["crystal_system"]:
            groups_per_system[row["crystal_system"]].add(row["provenance_group_id"])
    groups_per_system_counts = {s: len(g) for s, g in groups_per_system.items()}
    
    # --- Eligibility populations ---
    primary = [r for r in integrated if _is_primary(r)]
    sens_sym = [r for r in integrated if _is_sens_sym(r)]
    sens_chem = [r for r in integrated if _is_sens_chem(r)]
    
    primary_groups = set(r["provenance_group_id"] for r in primary)
    sens_sym_groups = set(r["provenance_group_id"] for r in sens_sym)
    sens_chem_groups = set(r["provenance_group_id"] for r in sens_chem)
    
    # --- Ambiguity and conflict flags ---
    chem_ambiguous = sum(1 for r in integrated if r.get("chemistry_eligibility") == "CHEM_AMBIGUOUS")
    chem_tentative = sum(1 for r in integrated if r.get("chemistry_eligibility") == "CHEM_TENTATIVE")
    chem_conflict = sum(1 for r in integrated if r.get("chemistry_eligibility") == "CHEM_CONFLICT")
    chem_ref = sum(1 for r in integrated if r.get("chemistry_eligibility") == "CHEM_REFERENCE_ELIGIBLE")
    
    sym_ambiguous = sum(1 for r in integrated if r.get("symmetry_eligibility_status") == "SYM_AMBIGUOUS")
    sym_tentative = sum(1 for r in integrated if r.get("symmetry_eligibility_status") == "SYM_TENTATIVE")
    sym_conflict = sum(1 for r in integrated if r.get("symmetry_eligibility_status") == "SYM_CONFLICT")
    sym_high = sum(1 for r in integrated if r.get("symmetry_eligibility_status") == "SYM_HIGH_CONFIDENCE")
    
    polymorph_ambig_sym = sum(1 for r in integrated if r.get("polymorph_ambiguous_symmetry") == "True")
    polymorph_ambig_chem = sum(1 for r in integrated if r.get("polymorph_ambiguous_chemistry") == "True")
    tentative_prov = sum(1 for r in integrated if r.get("tentative_provenance") == "True")
    high_conflict = sum(1 for r in integrated if r.get("high_conflict_flag") == "True")
    solid_solution = sum(1 for r in integrated if r.get("solid_solution_flag") == "True")
    mixture = sum(1 for r in integrated if r.get("mixture_flag") == "True")
    formula_ambig = sum(1 for r in integrated if r.get("formula_ambiguous") == "True")
    
    # --- Native vs PCHIP ---
    native_5nm = sum(1 for r in integrated if r.get("representation_type") == "native_5nm")
    pchip = sum(1 for r in integrated if r.get("representation_type") == "pchip_from_native_10nm")
    
    # --- Duplicate relationships ---
    in_dup = sum(1 for r in integrated if r.get("in_duplicate_relationship") == "True")
    
    # --- Provenance group sizes ---
    group_sizes = Counter(r["provenance_group_id"] for r in integrated)
    singleton_groups = sum(1 for g, c in group_sizes.items() if c == 1)
    repeated_groups = sum(1 for g, c in group_sizes.items() if c > 1)
    max_group_size = max(group_sizes.values()) if group_sizes else 0
    
    # --- Duplicate feature vectors ---
    feature_cols = ["n_peaks_prom", "mean_peak_prominence", "first_deriv_rms", "second_deriv_rms"]
    feature_vecs = defaultdict(list)
    for row in integrated:
        key = tuple(row[c] for c in feature_cols)
        feature_vecs[key].append(row["canonical_spectrum_id"])
    dup_groups = {k: v for k, v in feature_vecs.items() if len(v) > 1}
    n_dup_spectra = sum(len(v) for v in dup_groups.values())
    n_dup_groups = len(dup_groups)
    
    # --- Confounding check: mineral -> system mapping ---
    mineral_system_map = defaultdict(set)
    for row in integrated:
        m = row["mineral_label"]
        s = row["crystal_system"]
        if m and s:
            mineral_system_map[m].add(s)
    
    confounding = {m: len(s) for m, s in mineral_system_map.items()}
    all_one_to_one = all(n == 1 for n in confounding.values())
    
    # --- Per-mineral detail for PRIMARY population ---
    primary_mineral_detail = []
    for m in sorted(minerals):
        m_primary = [r for r in primary if r["mineral_label"] == m]
        if not m_primary:
            continue
        m_groups = set(r["provenance_group_id"] for r in m_primary)
        sys = mineral_to_system.get(m, "")
        n_spec = len(m_primary)
        n_grp = len(m_groups)
        native_ct = sum(1 for r in m_primary if r["representation_type"] == "native_5nm")
        pchip_ct = sum(1 for r in m_primary if r["representation_type"] == "pchip_from_native_10nm")
        primary_mineral_detail.append({
            "mineral": m,
            "crystal_system": sys,
            "n_spectra_primary": n_spec,
            "n_groups_primary": n_grp,
            "n_spectra_native_5nm": native_ct,
            "n_spectra_pchip": pchip_ct,
        })
    
    # --- Chemistry eligibility per system (FULL population) ---
    system_chem_elig = defaultdict(lambda: Counter())
    for row in integrated:
        s = row.get("crystal_system", "")
        c = row.get("chemistry_eligibility", "")
        if s and c:
            system_chem_elig[s][c] += 1
    
    # --- Symmetry eligibility per system (FULL population) ---
    system_sym_elig = defaultdict(lambda: Counter())
    for row in integrated:
        s = row.get("crystal_system", "")
        sym = row.get("symmetry_eligibility_status", "")
        if s and sym:
            system_sym_elig[s][sym] += 1
    
    # --- Mineral reference table summary ---
    mineral_ref_summary = []
    for row in mineral_ref:
        mineral_ref_summary.append({
            "mineral_name": row["mineral_name"],
            "chemical_formula": row["chemical_formula"],
            "crystal_system": row["crystal_system"],
            "space_group_symbol": row["space_group_symbol"],
            "point_group": row["point_group"],
            "polymorph_flag": row["polymorph_flag"],
            "polymorph_details": row["polymorph_details"],
            "chemistry_notes": row["chemistry_notes"],
        })
    
    # --- Build output ---
    baseline = {
        "total_spectra": n_spectra,
        "total_provenance_groups": n_groups,
        "total_minerals": n_minerals,
        "total_crystal_systems": n_systems,
        "crystal_systems_list": sorted(systems),
        "minerals_per_system": dict(sorted(minerals_per_system.items())),
        "minerals_list_per_system": {s: sorted(ms) for s, ms in sorted(system_to_minerals.items())},
        "spectra_per_mineral": dict(sorted(spectra_per_mineral.items())),
        "groups_per_mineral": dict(sorted(groups_per_mineral_counts.items())),
        "spectra_per_system": dict(sorted(spectra_per_system.items())),
        "groups_per_system": dict(sorted(groups_per_system_counts.items())),
        "eligibility_populations": {
            "PRIMARY": {"spectra": len(primary), "groups": len(primary_groups)},
            "SENSITIVITY_SYM": {"spectra": len(sens_sym), "groups": len(sens_sym_groups)},
            "SENSITIVITY_CHEM": {"spectra": len(sens_chem), "groups": len(sens_chem_groups)},
            "FULL": {"spectra": n_spectra, "groups": n_groups},
        },
        "symmetry_eligibility": {
            "SYM_HIGH_CONFIDENCE": sym_high,
            "SYM_AMBIGUOUS": sym_ambiguous,
            "SYM_TENTATIVE": sym_tentative,
            "SYM_CONFLICT": sym_conflict,
        },
        "chemistry_eligibility": {
            "CHEM_REFERENCE_ELIGIBLE": chem_ref,
            "CHEM_AMBIGUOUS": chem_ambiguous,
            "CHEM_TENTATIVE": chem_tentative,
            "CHEM_CONFLICT": chem_conflict,
        },
        "ambiguity_flags": {
            "polymorph_ambiguous_symmetry": polymorph_ambig_sym,
            "polymorph_ambiguous_chemistry": polymorph_ambig_chem,
            "tentative_provenance": tentative_prov,
            "high_conflict_flag": high_conflict,
            "solid_solution_flag": solid_solution,
            "mixture_flag": mixture,
            "formula_ambiguous": formula_ambig,
        },
        "representation": {
            "native_5nm": native_5nm,
            "pchip_from_native_10nm": pchip,
        },
        "duplicates": {
            "spectra_in_duplicate_relationship": in_dup,
            "duplicate_feature_vector_groups": n_dup_groups,
            "spectra_with_duplicate_feature_vectors": n_dup_spectra,
        },
        "provenance_group_structure": {
            "singleton_groups": singleton_groups,
            "repeated_groups": repeated_groups,
            "max_group_size": max_group_size,
        },
        "confounding": {
            "complete_mineral_system_confounding": all_one_to_one,
            "mineral_to_system_mapping": dict(sorted(confounding.items())),
        },
        "primary_mineral_detail": primary_mineral_detail,
        "system_chemistry_eligibility": {s: dict(c) for s, c in sorted(system_chem_elig.items())},
        "system_symmetry_eligibility": {s: dict(c) for s, c in sorted(system_sym_elig.items())},
        "mineral_reference_summary": mineral_ref_summary,
    }
    
    return baseline


def _is_primary(row):
    prov_tentative = row.get("tentative_provenance", "False") == "True"
    sym_elig = row.get("symmetry_eligibility_status", "")
    chem_elig = row.get("chemistry_eligibility", "")
    return (not prov_tentative
            and sym_elig == "SYM_HIGH_CONFIDENCE"
            and chem_elig == "CHEM_REFERENCE_ELIGIBLE")


def _is_sens_sym(row):
    prov_tentative = row.get("tentative_provenance", "False") == "True"
    chem_elig = row.get("chemistry_eligibility", "")
    return (not prov_tentative
            and chem_elig == "CHEM_REFERENCE_ELIGIBLE")


def _is_sens_chem(row):
    prov_tentative = row.get("tentative_provenance", "False") == "True"
    sym_elig = row.get("symmetry_eligibility_status", "")
    return (not prov_tentative
            and sym_elig == "SYM_HIGH_CONFIDENCE")


if __name__ == "__main__":
    baseline = compute_baseline()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase5_step4_coverage_baseline_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    print(f"Baseline computed: {out_path}")
    print(f"Total spectra: {baseline['total_spectra']}")
    print(f"Total provenance groups: {baseline['total_provenance_groups']}")
    print(f"Total minerals: {baseline['total_minerals']}")
    print(f"Total crystal systems: {baseline['total_crystal_systems']}")
    print(f"Minerals per system: {baseline['minerals_per_system']}")
    print(f"Complete confounding: {baseline['confounding']['complete_mineral_system_confounding']}")
    print(f"Primary population: {baseline['eligibility_populations']['PRIMARY']}")
    print(f"Representation: {baseline['representation']}")
    print(f"Duplicates: {baseline['duplicates']}")
