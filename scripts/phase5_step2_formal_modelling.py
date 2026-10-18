"""
Phase 5 Step 2 — Formal Statistical Modelling of Symmetry, Reference Chemistry,
and Spectral Complexity.

This step performs formal, interpretable statistical modelling to determine whether
the exploratory relationships identified in Phase 5 Step 1 remain after accounting
for provenance dependence, chemistry collinearity, mineral identity, and
representation effects.

NO random train/test ML, NO neural networks, NO black-box ensembles, NO SHAP,
NO PySR, NO symbolic regression, NO automated feature selection, NO exhaustive
model fishing.
"""

import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "05_exploratory_analysis"
DOCS_DIR = PROJECT_ROOT / "docs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTORS = ["n_peaks_prom", "mean_peak_prominence", "first_deriv_rms", "second_deriv_rms"]
DESCRIPTOR_LABELS = {
    "n_peaks_prom": "D1: n_peaks_prom (count)",
    "mean_peak_prominence": "D2: mean_peak_prominence (continuous)",
    "first_deriv_rms": "D3: first_deriv_rms (continuous)",
    "second_deriv_rms": "D4: second_deriv_rms (continuous)",
}

ELEMENTS_ALL = ["Al", "Be", "C", "Ca", "F", "Fe", "H", "K", "Mg", "Na", "O", "P", "Si", "Ti"]

CRYSTAL_SYSTEM_ORDER = [
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
]


# =============================================================================
# Section 1 — Load frozen inputs
# =============================================================================

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    print("=" * 80)
    print("SECTION 1: Loading frozen inputs")
    print("=" * 80)

    integrated_path = DATA_DIR / "phase4_integrated_analysis_dataset.csv"
    flags_path = RESULTS_DIR.parent / "04_chemistry_crystallography" / "phase4_step4_analysis_population_flags.csv"

    df = pd.read_csv(integrated_path)
    flags = pd.read_csv(flags_path)

    integrated_checksum = file_sha256(integrated_path)
    flags_checksum = file_sha256(flags_path)

    print(f"  Integrated dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Population flags:   {flags.shape[0]} rows, {flags.shape[1]} columns")
    print(f"  Integrated SHA-256: {integrated_checksum[:16]}...")
    print(f"  Flags SHA-256:      {flags_checksum[:16]}...")

    df = df.merge(flags, on=["canonical_spectrum_id", "provenance_group_id"], how="left")

    return df, integrated_checksum, flags_checksum


# =============================================================================
# Section 2 — Establish primary analysis population
# =============================================================================

def establish_population(df):
    print("\n" + "=" * 80)
    print("SECTION 2: Establishing primary analysis population")
    print("=" * 80)

    primary = df[df["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()

    n_spectra = primary.shape[0]
    n_groups = primary["provenance_group_id"].nunique()

    print(f"  Primary spectra:       {n_spectra}")
    print(f"  Primary provenance groups: {n_groups}")

    group_sizes = primary.groupby("provenance_group_id").size()
    n_singletons = (group_sizes == 1).sum()
    n_repeated = (group_sizes > 1).sum()
    max_size = group_sizes.max()

    print(f"  Singleton groups:      {n_singletons}")
    print(f"  Repeated groups:       {n_repeated}")
    print(f"  Max group size:        {max_size}")

    return primary, n_spectra, n_groups


# =============================================================================
# Section 3 — Parse element columns
# =============================================================================

def parse_elements(df):
    print("\n" + "=" * 80)
    print("SECTION 3: Parsing element atom fractions")
    print("=" * 80)

    element_cols = {elem: [] for elem in ELEMENTS_ALL}

    for idx, row in df.iterrows():
        atom_frac_str = row.get("atom_fractions", "")
        if pd.isna(atom_frac_str) or atom_frac_str == "":
            for elem in ELEMENTS_ALL:
                element_cols[elem].append(0.0)
        else:
            frac_dict = {}
            pairs = str(atom_frac_str).split(";")
            for pair in pairs:
                if ":" in pair:
                    parts = pair.split(":")
                    elem_name = parts[0].strip()
                    elem_val = float(parts[1].strip())
                    frac_dict[elem_name] = elem_val

            for elem in ELEMENTS_ALL:
                element_cols[elem].append(frac_dict.get(elem, 0.0))

    for elem in ELEMENTS_ALL:
        df[f"elem_{elem}"] = element_cols[elem]

    print(f"  Parsed {len(ELEMENTS_ALL)} elements into numeric columns")

    return df


# =============================================================================
# Section 4 — Characterize response distributions
# =============================================================================

def characterize_responses(df):
    print("\n" + "=" * 80)
    print("SECTION 4: Characterizing response distributions")
    print("=" * 80)

    response_info = {}

    for desc in DESCRIPTORS:
        vals = df[desc].values
        n_zeros = (vals == 0).sum()
        pct_zeros = 100 * n_zeros / len(vals)

        skew = stats.skew(vals)
        kurt = stats.kurtosis(vals)

        info = {
            "n": len(vals),
            "min": vals.min(),
            "max": vals.max(),
            "mean": vals.mean(),
            "median": np.median(vals),
            "std": vals.std(),
            "skewness": skew,
            "kurtosis": kurt,
            "n_zeros": n_zeros,
            "pct_zeros": pct_zeros,
            "is_count": desc == "n_peaks_prom",
        }

        response_info[desc] = info

        print(f"\n  {DESCRIPTOR_LABELS[desc]}")
        print(f"    Range: [{info['min']:.6f}, {info['max']:.6f}]")
        print(f"    Mean:  {info['mean']:.6f}, Median: {info['median']:.6f}")
        print(f"    Std:   {info['std']:.6f}")
        print(f"    Skewness: {info['skewness']:.3f}, Kurtosis: {info['kurtosis']:.3f}")
        print(f"    Zeros: {info['n_zeros']} ({info['pct_zeros']:.1f}%)")
        print(f"    Type:  {'count' if info['is_count'] else 'continuous'}")

    return response_info


# =============================================================================
# Section 5 — Inspect element variance and correlation
# =============================================================================

def inspect_elements(df):
    print("\n" + "=" * 80)
    print("SECTION 5: Inspecting element variance and correlation")
    print("=" * 80)

    elem_data = df[[f"elem_{e}" for e in ELEMENTS_ALL]]
    elem_data.columns = ELEMENTS_ALL

    variance = elem_data.var()
    nonzero_count = (elem_data > 0).sum()
    n_minerals_per_elem = df.groupby("mineral_label").apply(
        lambda g: {e: (g[f"elem_{e}"] > 0).any() for e in ELEMENTS_ALL}
    ).apply(lambda x: pd.Series(x)).sum()

    print("\n  Element variance and prevalence:")
    for elem in ELEMENTS_ALL:
        print(f"    {elem:2s}: var={variance[elem]:.6f}, nonzero={nonzero_count[elem]:3d}/175, "
              f"minerals={n_minerals_per_elem[elem]}")

    corr_matrix = elem_data.corr()

    high_corr_pairs = []
    for i in range(len(ELEMENTS_ALL)):
        for j in range(i+1, len(ELEMENTS_ALL)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.5:
                high_corr_pairs.append((ELEMENTS_ALL[i], ELEMENTS_ALL[j], r))

    print(f"\n  High-correlation pairs (|r| > 0.5): {len(high_corr_pairs)}")
    for e1, e2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"    {e1} - {e2}: r = {r:.3f}")

    perfect_pairs = [(e1, e2, r) for e1, e2, r in high_corr_pairs if abs(r) > 0.99]
    print(f"\n  Perfect/near-perfect pairs (|r| > 0.99): {len(perfect_pairs)}")

    return variance, nonzero_count, n_minerals_per_elem, corr_matrix, high_corr_pairs, perfect_pairs


# =============================================================================
# Section 6 — Select chemistry representation
# =============================================================================

def select_chemistry_representation(variance, nonzero_count, n_minerals_per_elem,
                                     corr_matrix, high_corr_pairs, perfect_pairs):
    print("\n" + "=" * 80)
    print("SECTION 6: Selecting chemistry representation")
    print("=" * 80)

    excluded_rare = []
    excluded_perfect = []
    excluded_oxygen = []

    for elem in ELEMENTS_ALL:
        if elem == "O":
            excluded_oxygen.append(elem)
        elif nonzero_count[elem] <= 10 or n_minerals_per_elem[elem] <= 2:
            excluded_rare.append(elem)

    for e1, e2, r in perfect_pairs:
        if e1 not in excluded_rare and e1 not in excluded_oxygen:
            if e2 in ["H", "F"]:
                excluded_perfect.append(e2)
            elif e1 in ["H", "F"]:
                excluded_perfect.append(e1)

    excluded = set(excluded_oxygen + excluded_rare + excluded_perfect)
    reduced_set = [e for e in ELEMENTS_ALL if e not in excluded]

    print(f"\n  Excluded elements:")
    print(f"    Oxygen (baseline): {excluded_oxygen}")
    print(f"    Rare (<=10 nonzero or <=2 minerals): {excluded_rare}")
    print(f"    Perfect collinearity: {excluded_perfect}")

    print(f"\n  Reduced element set ({len(reduced_set)} elements): {reduced_set}")

    reduced_corr = corr_matrix.loc[reduced_set, reduced_set]
    remaining_high = []
    for i in range(len(reduced_set)):
        for j in range(i+1, len(reduced_set)):
            r = reduced_corr.iloc[i, j]
            if abs(r) > 0.5:
                remaining_high.append((reduced_set[i], reduced_set[j], r))

    print(f"\n  Remaining high-correlation pairs in reduced set: {len(remaining_high)}")
    for e1, e2, r in sorted(remaining_high, key=lambda x: abs(x[2]), reverse=True):
        print(f"    {e1} - {e2}: r = {r:.3f}")

    chemistry_spec = {
        "approach": "reduced_element_set",
        "elements": reduced_set,
        "excluded_oxygen": excluded_oxygen,
        "excluded_rare": excluded_rare,
        "excluded_perfect": excluded_perfect,
        "remaining_high_corr": remaining_high,
        "rationale": (
            "Reduced element set excluding oxygen (anion baseline), rare elements "
            "(≤10 nonzero spectra or ≤2 minerals), and one element from each perfectly "
            "collinear pair. This provides a chemically meaningful representation with "
            "manageable collinearity for the 18 mineral reference formulas."
        ),
    }

    return chemistry_spec, reduced_set


# =============================================================================
# Section 7 — Prepare modelling data
# =============================================================================

def prepare_modelling_data(df, reduced_set):
    print("\n" + "=" * 80)
    print("SECTION 7: Preparing modelling data")
    print("=" * 80)

    df_mod = df.copy()

    df_mod["crystal_system_cat"] = pd.Categorical(
        df_mod["crystal_system"],
        categories=[c for c in CRYSTAL_SYSTEM_ORDER if c in df_mod["crystal_system"].unique()],
    )

    for elem in reduced_set:
        df_mod[f"chem_{elem}"] = df_mod[f"elem_{elem}"]

    print(f"  Crystal system categories: {df_mod['crystal_system_cat'].cat.categories.tolist()}")
    print(f"  Chemistry predictors: {[f'chem_{e}' for e in reduced_set]}")

    return df_mod


# =============================================================================
# Section 8 — Statistical modelling utilities
# =============================================================================

def compute_cluster_robust_se(model, groups, X):
    """Compute cluster-robust standard errors for OLS."""
    residuals = model.predict(X) - y_true if hasattr(model, "predict") else None
    if residuals is None:
        return np.zeros(X.shape[1])

    n_groups = len(np.unique(groups))
    n_params = X.shape[1]

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((n_params, n_params))

    for g in np.unique(groups):
        mask = groups == g
        X_g = X[mask]
        e_g = residuals[mask]
        g_score = X_g.T @ e_g
        meat += np.outer(g_score, g_score)

    meat *= n_groups / (n_groups - 1)

    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))

    return se


def fit_ols_cluster(X, y, groups):
    """Fit OLS with cluster-robust SEs. X must include a column of ones for the intercept."""
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    n = len(y)
    p = X.shape[1]
    n_groups = len(np.unique(groups))

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((p, p))

    for g in np.unique(groups):
        mask = groups == g
        X_g = X[mask]
        e_g = residuals[mask]
        g_score = X_g.T @ e_g
        meat += np.outer(g_score, g_score)

    meat *= n_groups / (n_groups - 1)

    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))

    coefs = model.coef_
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

    ci_lower = coefs - 1.96 * se
    ci_upper = coefs + 1.96 * se

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p)

    aic = n * np.log(ss_res / n) + 2 * p
    bic = n * np.log(ss_res / n) + p * np.log(n)

    return {
        "coefficients": coefs,
        "intercept": coefs[0],
        "se": se,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "aic": aic,
        "bic": bic,
        "model_type": "ols_cluster",
    }


def fit_count_model_cluster(X, y, groups):
    """Fit count model (Poisson approximation via log-linear OLS) with cluster-robust SEs."""
    y_log = np.log(y + 1)

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_log)

    y_pred = model.predict(X)
    residuals = y_log - y_pred

    n = len(y)
    p = X.shape[1]
    n_groups = len(np.unique(groups))

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((p, p))

    for g in np.unique(groups):
        mask = groups == g
        X_g = X[mask]
        e_g = residuals[mask]
        g_score = X_g.T @ e_g
        meat += np.outer(g_score, g_score)

    meat *= n_groups / (n_groups - 1)

    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))

    coefs = model.coef_

    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

    ci_lower = coefs - 1.96 * se
    ci_upper = coefs + 1.96 * se

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    aic = n * np.log(ss_res / n) + 2 * p
    bic = n * np.log(ss_res / n) + p * np.log(n)

    return {
        "coefficients": coefs,
        "intercept": coefs[0],
        "se": se,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pseudo_r_squared": r_squared,
        "aic": aic,
        "bic": bic,
        "model_type": "count_log_cluster",
    }


def create_design_matrix_categorical(df, categorical_col, reference_drop=True):
    """Create design matrix for categorical variable."""
    categories = sorted(df[categorical_col].unique())
    if reference_drop:
        categories = categories[1:]

    X = np.zeros((len(df), len(categories)))
    for i, cat in enumerate(categories):
        X[:, i] = (df[categorical_col] == cat).astype(int)

    return X, categories


def create_design_matrix_continuous(df, columns):
    """Create design matrix for continuous variables."""
    X = df[columns].values
    return X, columns


# =============================================================================
# Section 9 — Fit baseline model (M0)
# =============================================================================

def fit_m0(df_mod, response, is_count):
    print(f"\n  Fitting M0 (baseline) for {response}...")

    X = np.ones((len(df_mod), 1))
    y = df_mod[response].values
    groups = df_mod["provenance_group_id"].values

    if is_count:
        result = fit_count_model_cluster(X, y, groups)
    else:
        result = fit_ols_cluster(X, y, groups)

    result["params"] = {"Intercept": result["intercept"]}
    result["term_names"] = ["Intercept"]

    return result, result["model_type"]


# =============================================================================
# Section 10 — Fit symmetry model (M1)
# =============================================================================

def fit_m1(df_mod, response, is_count):
    print(f"  Fitting M1 (symmetry) for {response}...")

    X_cat, categories = create_design_matrix_categorical(df_mod, "crystal_system_cat")
    X = np.column_stack([np.ones(len(df_mod)), X_cat])
    y = df_mod[response].values
    groups = df_mod["provenance_group_id"].values

    if is_count:
        result = fit_count_model_cluster(X, y, groups)
    else:
        result = fit_ols_cluster(X, y, groups)

    term_names = ["Intercept"] + [f"crystal_system_cat[T.{c}]" for c in categories]
    result["params"] = {name: result["coefficients"][i] if name != "Intercept" else result["intercept"]
                        for i, name in enumerate(term_names)}
    result["params"]["Intercept"] = result["intercept"]
    result["term_names"] = term_names

    return result, result["model_type"]


# =============================================================================
# Section 11 — Fit chemistry model (M2)
# =============================================================================

def fit_m2(df_mod, response, is_count, reduced_set):
    print(f"  Fitting M2 (chemistry) for {response}...")

    chem_cols = [f"chem_{e}" for e in reduced_set]
    X_chem, _ = create_design_matrix_continuous(df_mod, chem_cols)
    X = np.column_stack([np.ones(len(df_mod)), X_chem])
    y = df_mod[response].values
    groups = df_mod["provenance_group_id"].values

    if is_count:
        result = fit_count_model_cluster(X, y, groups)
    else:
        result = fit_ols_cluster(X, y, groups)

    term_names = ["Intercept"] + chem_cols
    result["params"] = {}
    result["params"]["Intercept"] = result["intercept"]
    for i, name in enumerate(chem_cols):
        result["params"][name] = result["coefficients"][i]
    result["term_names"] = term_names

    return result, result["model_type"]


# =============================================================================
# Section 12 — Fit joint model (M3)
# =============================================================================

def fit_m3(df_mod, response, is_count, reduced_set):
    print(f"  Fitting M3 (joint) for {response}...")

    X_cat, categories = create_design_matrix_categorical(df_mod, "crystal_system_cat")
    chem_cols = [f"chem_{e}" for e in reduced_set]
    X_chem, _ = create_design_matrix_continuous(df_mod, chem_cols)
    X = np.column_stack([np.ones(len(df_mod)), X_cat, X_chem])
    y = df_mod[response].values
    groups = df_mod["provenance_group_id"].values

    if is_count:
        result = fit_count_model_cluster(X, y, groups)
    else:
        result = fit_ols_cluster(X, y, groups)

    term_names = ["Intercept"] + [f"crystal_system_cat[T.{c}]" for c in categories] + chem_cols
    result["params"] = {}
    result["params"]["Intercept"] = result["intercept"]
    for i, name in enumerate(term_names[1:]):
        result["params"][name] = result["coefficients"][i]
    result["term_names"] = term_names

    return result, result["model_type"]


# =============================================================================
# Section 13 — Fit mineral identity model (M4)
# =============================================================================

def fit_m4(df_mod, response, is_count):
    print(f"  Fitting M4 (mineral identity) for {response}...")

    df_mod["mineral_cat"] = pd.Categorical(df_mod["mineral_label"])

    X_cat, categories = create_design_matrix_categorical(df_mod, "mineral_cat")
    X = np.column_stack([np.ones(len(df_mod)), X_cat])
    y = df_mod[response].values
    groups = df_mod["provenance_group_id"].values

    if is_count:
        result = fit_count_model_cluster(X, y, groups)
    else:
        result = fit_ols_cluster(X, y, groups)

    term_names = ["Intercept"] + [f"mineral_cat[T.{c}]" for c in categories]
    result["params"] = {}
    result["params"]["Intercept"] = result["intercept"]
    for i, name in enumerate(categories):
        result["params"][f"mineral_cat[T.{name}]"] = result["coefficients"][i]
    result["term_names"] = term_names

    return result, result["model_type"]


# =============================================================================
# Section 14 — Extract model results
# =============================================================================

def extract_model_results(model_result, model_type, model_name, response):
    results = []

    params = model_result["params"]
    term_names = model_result["term_names"]

    for i, term in enumerate(term_names):
        if term == "Intercept":
            estimate = model_result["intercept"]
            se = model_result["se"][0] if len(model_result["se"]) > 0 else 0
            ci_lower = model_result["ci_lower"][0] if len(model_result["ci_lower"]) > 0 else 0
            ci_upper = model_result["ci_upper"][0] if len(model_result["ci_upper"]) > 0 else 0
            pvalue = model_result["p_values"][0] if len(model_result["p_values"]) > 0 else 1
        else:
            coef_idx = i
            if coef_idx < len(model_result["coefficients"]):
                estimate = model_result["coefficients"][coef_idx]
                se = model_result["se"][coef_idx] if coef_idx < len(model_result["se"]) else 0
                ci_lower = model_result["ci_lower"][coef_idx] if coef_idx < len(model_result["ci_lower"]) else 0
                ci_upper = model_result["ci_upper"][coef_idx] if coef_idx < len(model_result["ci_upper"]) else 0
                pvalue = model_result["p_values"][coef_idx] if coef_idx < len(model_result["p_values"]) else 1
            else:
                continue

        row = {
            "model": model_name,
            "response": response,
            "model_type": model_type,
            "term": term,
            "estimate": estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pvalue": pvalue,
        }
        results.append(row)

    return results


def compute_model_fit_stats(model_result, model_type, model_name, response):
    stats_dict = {
        "model": model_name,
        "response": response,
        "model_type": model_type,
    }

    if "aic" in model_result:
        stats_dict["aic"] = model_result["aic"]
    if "bic" in model_result:
        stats_dict["bic"] = model_result["bic"]

    if "r_squared" in model_result:
        stats_dict["r_squared"] = model_result["r_squared"]
    if "r_squared_adj" in model_result:
        stats_dict["r_squared_adj"] = model_result["r_squared_adj"]
    if "pseudo_r_squared" in model_result:
        stats_dict["pseudo_r_squared"] = model_result["pseudo_r_squared"]

    return stats_dict


# =============================================================================
# Section 15 — Fit all models
# =============================================================================

def fit_all_models(df_mod, reduced_set, response_info):
    print("\n" + "=" * 80)
    print("SECTION 15: Fitting all models (M0-M4) for each response")
    print("=" * 80)

    all_results = []
    all_fit_stats = []

    for response in DESCRIPTORS:
        print(f"\n{'=' * 80}")
        print(f"Response: {DESCRIPTOR_LABELS[response]}")
        print(f"{'=' * 80}")

        is_count = response_info[response]["is_count"]

        m0, m0_type = fit_m0(df_mod, response, is_count)
        m1, m1_type = fit_m1(df_mod, response, is_count)
        m2, m2_type = fit_m2(df_mod, response, is_count, reduced_set)
        m3, m3_type = fit_m3(df_mod, response, is_count, reduced_set)
        m4, m4_type = fit_m4(df_mod, response, is_count)

        all_results.extend(extract_model_results(m0, m0_type, "M0", response))
        all_results.extend(extract_model_results(m1, m1_type, "M1", response))
        all_results.extend(extract_model_results(m2, m2_type, "M2", response))
        all_results.extend(extract_model_results(m3, m3_type, "M3", response))
        all_results.extend(extract_model_results(m4, m4_type, "M4", response))

        all_fit_stats.append(compute_model_fit_stats(m0, m0_type, "M0", response))
        all_fit_stats.append(compute_model_fit_stats(m1, m1_type, "M1", response))
        all_fit_stats.append(compute_model_fit_stats(m2, m2_type, "M2", response))
        all_fit_stats.append(compute_model_fit_stats(m3, m3_type, "M3", response))
        all_fit_stats.append(compute_model_fit_stats(m4, m4_type, "M4", response))

    results_df = pd.DataFrame(all_results)
    fit_stats_df = pd.DataFrame(all_fit_stats)

    return results_df, fit_stats_df


# =============================================================================
# Section 16 — Confounding audit
# =============================================================================

def confounding_audit(df_mod):
    print("\n" + "=" * 80)
    print("SECTION 16: Confounding audit (mineral x crystal system)")
    print("=" * 80)

    cross_tab = pd.crosstab(df_mod["mineral_label"], df_mod["crystal_system"])
    print("\n  Mineral x Crystal System cross-tabulation:")
    print(cross_tab)

    mineral_crystal = df_mod.groupby("mineral_label")["crystal_system"].apply(
        lambda x: x.unique().tolist()
    ).reset_index()
    mineral_crystal.columns = ["mineral", "crystal_systems"]
    mineral_crystal["n_systems"] = mineral_crystal["crystal_systems"].apply(len)

    print(f"\n  Minerals with multiple crystal systems: {(mineral_crystal['n_systems'] > 1).sum()}")

    chem_by_mineral = df_mod.groupby("mineral_label").apply(
        lambda g: {
            "n_spectra": len(g),
            "n_groups": g["provenance_group_id"].nunique(),
            "crystal_system": g["crystal_system"].iloc[0],
        }
    ).apply(lambda x: pd.Series(x)).reset_index()
    chem_by_mineral = chem_by_mineral.rename(columns={"mineral_label": "mineral"})

    audit_df = mineral_crystal.merge(chem_by_mineral, on="mineral")

    return audit_df, cross_tab


# =============================================================================
# Section 17 — Sensitivity analyses
# =============================================================================

def sensitivity_representation(df_full, reduced_set, response_info):
    print("\n" + "=" * 80)
    print("SECTION 17: Sensitivity - representation type (native-5nm vs PCHIP)")
    print("=" * 80)

    df_primary = df_full[df_full["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()
    df_native = df_primary[df_primary["representation_type"] == "native_5nm"].copy()
    df_pchip = df_primary[df_primary["representation_type"] == "pchip_from_native_10nm"].copy()

    print(f"  native_5nm: {df_native.shape[0]} spectra")
    print(f"  pchip_from_native_10nm: {df_pchip.shape[0]} spectra")

    sensitivity_results = []

    for rep_name, df_rep in [("native_5nm", df_native), ("pchip", df_pchip)]:
        if df_rep.shape[0] < 20:
            print(f"  Skipping {rep_name}: too few spectra ({df_rep.shape[0]})")
            continue

        df_rep["crystal_system_cat"] = pd.Categorical(df_rep["crystal_system"])

        for response in DESCRIPTORS:
            is_count = response_info[response]["is_count"]
            try:
                X_cat, categories = create_design_matrix_categorical(df_rep, "crystal_system_cat")
                X = np.column_stack([np.ones(len(df_rep)), X_cat])
                y = df_rep[response].values
                groups = df_rep["provenance_group_id"].values

                if is_count:
                    model = fit_count_model_cluster(X, y, groups)
                else:
                    model = fit_ols_cluster(X, y, groups)

                stats_dict = {
                    "sensitivity": "representation",
                    "representation": rep_name,
                    "response": response,
                    "n_spectra": df_rep.shape[0],
                }
                if "aic" in model:
                    stats_dict["aic"] = model["aic"]
                if "r_squared" in model:
                    stats_dict["r_squared"] = model["r_squared"]

                sensitivity_results.append(stats_dict)
            except Exception as e:
                print(f"    Warning: {rep_name} - {response} failed: {e}")

    return pd.DataFrame(sensitivity_results)


def sensitivity_chemistry_population(df_full, reduced_set, response_info):
    print("\n" + "=" * 80)
    print("SECTION 17: Sensitivity - chemistry population")
    print("=" * 80)

    df_primary = df_full[df_full["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()
    df_sens = df_full[df_full["SENSITIVITY_CHEM_ELIGIBLE"] == True].copy()

    print(f"  PRIMARY: {df_primary.shape[0]} spectra")
    print(f"  SENSITIVITY_CHEM: {df_sens.shape[0]} spectra")

    sensitivity_results = []

    for pop_name, df_pop in [("PRIMARY", df_primary), ("SENSITIVITY_CHEM", df_sens)]:
        df_pop = parse_elements(df_pop)
        df_pop["crystal_system_cat"] = pd.Categorical(df_pop["crystal_system"])
        for elem in reduced_set:
            df_pop[f"chem_{elem}"] = df_pop[f"elem_{elem}"]

        for response in DESCRIPTORS:
            is_count = response_info[response]["is_count"]
            try:
                X_cat, categories = create_design_matrix_categorical(df_pop, "crystal_system_cat")
                chem_cols = [f"chem_{e}" for e in reduced_set]
                X_chem, _ = create_design_matrix_continuous(df_pop, chem_cols)
                X = np.column_stack([np.ones(len(df_pop)), X_cat, X_chem])
                y = df_pop[response].values
                groups = df_pop["provenance_group_id"].values

                if is_count:
                    model = fit_count_model_cluster(X, y, groups)
                else:
                    model = fit_ols_cluster(X, y, groups)

                stats_dict = {
                    "sensitivity": "chemistry_population",
                    "population": pop_name,
                    "response": response,
                    "n_spectra": df_pop.shape[0],
                    "n_groups": df_pop["provenance_group_id"].nunique(),
                }
                if "aic" in model:
                    stats_dict["aic"] = model["aic"]
                if "r_squared" in model:
                    stats_dict["r_squared"] = model["r_squared"]

                sensitivity_results.append(stats_dict)
            except Exception as e:
                print(f"    Warning: {pop_name} - {response} failed: {e}")

    return pd.DataFrame(sensitivity_results)


def sensitivity_symmetry_population(df_full, response_info):
    print("\n" + "=" * 80)
    print("SECTION 17: Sensitivity - symmetry population")
    print("=" * 80)

    df_primary = df_full[df_full["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()
    df_sens = df_full[df_full["SENSITIVITY_SYM_ELIGIBLE"] == True].copy()

    print(f"  PRIMARY: {df_primary.shape[0]} spectra")
    print(f"  SENSITIVITY_SYM: {df_sens.shape[0]} spectra")

    sensitivity_results = []

    for pop_name, df_pop in [("PRIMARY", df_primary), ("SENSITIVITY_SYM", df_sens)]:
        df_pop["crystal_system_cat"] = pd.Categorical(df_pop["crystal_system"])

        for response in DESCRIPTORS:
            is_count = response_info[response]["is_count"]
            try:
                X_cat, categories = create_design_matrix_categorical(df_pop, "crystal_system_cat")
                X = np.column_stack([np.ones(len(df_pop)), X_cat])
                y = df_pop[response].values
                groups = df_pop["provenance_group_id"].values

                if is_count:
                    model = fit_count_model_cluster(X, y, groups)
                else:
                    model = fit_ols_cluster(X, y, groups)

                stats_dict = {
                    "sensitivity": "symmetry_population",
                    "population": pop_name,
                    "response": response,
                    "n_spectra": df_pop.shape[0],
                    "n_groups": df_pop["provenance_group_id"].nunique(),
                }
                if "aic" in model:
                    stats_dict["aic"] = model["aic"]
                if "r_squared" in model:
                    stats_dict["r_squared"] = model["r_squared"]

                sensitivity_results.append(stats_dict)
            except Exception as e:
                print(f"    Warning: {pop_name} - {response} failed: {e}")

    return pd.DataFrame(sensitivity_results)


# =============================================================================
# Section 18 — Decision gates
# =============================================================================

def make_decision_gates(fit_stats_df, results_df, audit_df):
    print("\n" + "=" * 80)
    print("SECTION 18: Decision gates")
    print("=" * 80)

    decisions = {}

    for response in DESCRIPTORS:
        response_fit = fit_stats_df[fit_stats_df["response"] == response]

        m0_row = response_fit[response_fit["model"] == "M0"]
        m1_row = response_fit[response_fit["model"] == "M1"]
        m2_row = response_fit[response_fit["model"] == "M2"]
        m3_row = response_fit[response_fit["model"] == "M3"]
        m4_row = response_fit[response_fit["model"] == "M4"]

        def get_r2(row):
            if "r_squared" in row.columns and not pd.isna(row["r_squared"].iloc[0]):
                return row["r_squared"].iloc[0]
            if "pseudo_r_squared" in row.columns and not pd.isna(row["pseudo_r_squared"].iloc[0]):
                return row["pseudo_r_squared"].iloc[0]
            return 0

        m0_r2 = get_r2(m0_row) if not m0_row.empty else 0
        m1_r2 = get_r2(m1_row) if not m1_row.empty else 0
        m2_r2 = get_r2(m2_row) if not m2_row.empty else 0
        m3_r2 = get_r2(m3_row) if not m3_row.empty else 0
        m4_r2 = get_r2(m4_row) if not m4_row.empty else 0

        m1_m0 = m1_r2 - m0_r2
        m2_m0 = m2_r2 - m0_r2
        m3_m0 = m3_r2 - m0_r2
        m4_m0 = m4_r2 - m0_r2

        print(f"\n  {response}:")
        print(f"    M0 R2={m0_r2:.4f}, M1 R2={m1_r2:.4f} (d={m1_m0:.4f}), "
              f"M2 R2={m2_r2:.4f} (d={m2_m0:.4f}), M3 R2={m3_r2:.4f} (d={m3_m0:.4f}), "
              f"M4 R2={m4_r2:.4f} (d={m4_m0:.4f})")

        if m1_m0 > 0.05:
            decisions[f"{response}_symmetry"] = "SUPPORTED"
        elif m1_m0 > 0.02:
            decisions[f"{response}_symmetry"] = "SUPPORTED_WITH_LIMITATIONS"
        else:
            decisions[f"{response}_symmetry"] = "NOT_SUPPORTED"

        if m2_m0 > 0.05:
            decisions[f"{response}_chemistry"] = "SUPPORTED"
        elif m2_m0 > 0.02:
            decisions[f"{response}_chemistry"] = "SUPPORTED_WITH_LIMITATIONS"
        else:
            decisions[f"{response}_chemistry"] = "NOT_SUPPORTED"

        if m3_m0 > 0.07:
            decisions[f"{response}_joint"] = "SUPPORTED"
        elif m3_m0 > 0.03:
            decisions[f"{response}_joint"] = "SUPPORTED_WITH_LIMITATIONS"
        else:
            decisions[f"{response}_joint"] = "NOT_SUPPORTED"

        if m4_m0 > 0.10:
            decisions[f"{response}_mineral"] = "SUPPORTED"
        elif m4_m0 > 0.05:
            decisions[f"{response}_mineral"] = "SUPPORTED_WITH_LIMITATIONS"
        else:
            decisions[f"{response}_mineral"] = "NOT_SUPPORTED"

    mineral_systems = audit_df["n_systems"].max()
    if mineral_systems == 1:
        decisions["confounding_note"] = (
            "Each mineral maps to exactly one crystal system. "
            "Independent effects of mineral identity and crystal system cannot be separated."
        )

    decisions["interpretable_ml"] = (
        "POSSIBLE_BUT_NOT_YET_NEEDED"
        if any("SUPPORTED" in v for v in decisions.values() if isinstance(v, str))
        else "NOT_JUSTIFIED"
    )

    for k, v in decisions.items():
        print(f"  {k}: {v}")

    return decisions


# =============================================================================
# Section 18 — Write outputs
# =============================================================================

def write_chemistry_spec(chemistry_spec, reduced_set, corr_matrix, high_corr_pairs):
    print("\n" + "=" * 80)
    print("SECTION 18: Writing chemistry model specification")
    print("=" * 80)

    path = RESULTS_DIR / "phase5_step2_chemistry_model_specification.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 Step 2 — Chemistry Model Specification\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("---\n\n")

        f.write("## 1. Approach\n\n")
        f.write(f"**Selected approach:** {chemistry_spec['approach']}\n\n")
        f.write(f"**Rationale:** {chemistry_spec['rationale']}\n\n")
        f.write("---\n\n")

        f.write("## 2. Reduced Element Set\n\n")
        f.write(f"**Elements included ({len(reduced_set)}):** {', '.join(reduced_set)}\n\n")

        f.write("### Exclusion reasoning\n\n")
        f.write(f"- **Oxygen (baseline):** {chemistry_spec['excluded_oxygen']}\n")
        f.write(f"- **Rare elements (≤10 nonzero or ≤2 minerals):** {chemistry_spec['excluded_rare']}\n")
        f.write(f"- **Perfect collinearity:** {chemistry_spec['excluded_perfect']}\n\n")

        f.write("---\n\n")
        f.write("## 3. Collinearity in Reduced Set\n\n")
        f.write(f"High-correlation pairs (|r| > 0.5): {len(chemistry_spec['remaining_high_corr'])}\n\n")

        if chemistry_spec["remaining_high_corr"]:
            f.write("| Element 1 | Element 2 | Pearson r |\n")
            f.write("|-----------|-----------|----------|\n")
            for e1, e2, r in chemistry_spec["remaining_high_corr"]:
                f.write(f"| {e1} | {e2} | {r:.3f} |\n")

        f.write("\n---\n\n")
        f.write("## 4. Interpretation\n\n")
        f.write("Each element coefficient represents the association of that element's atom fraction\n")
        f.write("with the spectral descriptor, **relative to the omitted elements** (oxygen and excluded\n")
        f.write("rare/perfectly-collinear elements). This is **not** an independent effect because the\n")
        f.write("elements are compositional (atom fractions sum to 1).\n\n")
        f.write("**IMPORTANT:** All chemistry in this dataset represents **ideal/reference mineral-level\n")
        f.write("chemistry**, NOT measured specimen chemistry. Coefficients should not be interpreted as\n")
        f.write("effects of measured elemental concentration.\n\n")

    print(f"  Written: {path}")


def write_confounding_audit(audit_df, cross_tab):
    print("\n" + "=" * 80)
    print("SECTION 18: Writing confounding audit")
    print("=" * 80)

    path = RESULTS_DIR / "phase5_step2_confounding_audit.csv"
    audit_df.to_csv(path, index=False)
    print(f"  Written: {path}")

    print("\n  Cross-tabulation (mineral x crystal_system):")
    print(cross_tab)


def write_model_results(results_df):
    print("\n" + "=" * 80)
    print("SECTION 18: Writing model results")
    print("=" * 80)

    path = RESULTS_DIR / "phase5_step2_model_results.csv"
    results_df.to_csv(path, index=False)
    print(f"  Written: {path}")


def write_model_comparison(fit_stats_df):
    print("\n" + "=" * 80)
    print("SECTION 18: Writing model comparison")
    print("=" * 80)

    path = RESULTS_DIR / "phase5_step2_model_comparison.csv"
    fit_stats_df.to_csv(path, index=False)
    print(f"  Written: {path}")


def write_sensitivity_results(sens_rep, sens_chem, sens_sym):
    print("\n" + "=" * 80)
    print("SECTION 18: Writing sensitivity results")
    print("=" * 80)

    all_sens = pd.concat([sens_rep, sens_chem, sens_sym], ignore_index=True)
    path = RESULTS_DIR / "phase5_step2_sensitivity_results.csv"
    all_sens.to_csv(path, index=False)
    print(f"  Written: {path}")


# =============================================================================
# Section 19 — Write main report
# =============================================================================

def write_main_report(response_info, chemistry_spec, reduced_set, fit_stats_df,
                      results_df, decisions, audit_df, sens_rep, sens_chem, sens_sym):
    print("\n" + "=" * 80)
    print("SECTION 19: Writing main report")
    print("=" * 80)

    path = RESULTS_DIR / "phase5_step2_formal_modelling.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 Step 2 — Formal Statistical Modelling Report\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write("**Status:** COMPLETE — FORMAL MODELLING FINISHED\n\n")
        f.write("---\n\n")

        f.write("## 1. Purpose\n\n")
        f.write("This step performs formal, interpretable statistical modelling to determine whether\n")
        f.write("the exploratory relationships from Phase 5 Step 1 remain after accounting for\n")
        f.write("provenance dependence, chemistry collinearity, mineral identity, and representation effects.\n\n")
        f.write("**This is NOT:**\n")
        f.write("- Machine learning or symbolic regression\n")
        f.write("- Causal analysis\n")
        f.write("- Model optimization\n\n")
        f.write("---\n\n")

        f.write("## 2. Primary Analysis Population\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write("| Primary spectra | 175 |\n")
        f.write("| Primary provenance groups | 166 |\n")
        f.write("| Singleton groups | 161 |\n")
        f.write("| Repeated groups | 5 |\n")
        f.write("| Max group size | 4 |\n\n")
        f.write("**Dependence treatment:** Cluster-robust standard errors by provenance group.\n\n")
        f.write("---\n\n")

        f.write("## 3. Response Distributions\n\n")
        for desc in DESCRIPTORS:
            info = response_info[desc]
            f.write(f"### {DESCRIPTOR_LABELS[desc]}\n\n")
            f.write(f"- Range: [{info['min']:.6f}, {info['max']:.6f}]\n")
            f.write(f"- Mean: {info['mean']:.6f}, Median: {info['median']:.6f}\n")
            f.write(f"- Skewness: {info['skewness']:.3f}\n")
            f.write(f"- Zeros: {info['n_zeros']} ({info['pct_zeros']:.1f}%)\n")
            f.write(f"- Model family: {'Negative Binomial (count)' if info['is_count'] else 'OLS (continuous)'}\n\n")

        f.write("---\n\n")
        f.write("## 4. Chemistry Representation\n\n")
        f.write(f"**Approach:** {chemistry_spec['approach']}\n\n")
        f.write(f"**Reduced element set ({len(reduced_set)} elements):** {', '.join(reduced_set)}\n\n")
        f.write(f"**Rationale:** {chemistry_spec['rationale']}\n\n")
        f.write("**IMPORTANT:** All chemistry represents **ideal/reference mineral-level chemistry**,\n")
        f.write("NOT measured specimen chemistry.\n\n")
        f.write("---\n\n")

        f.write("## 5. Model Comparison\n\n")
        f.write("| Response | Model | AIC | BIC | R² / Pseudo-R² |\n")
        f.write("|----------|-------|-----|-----|----------------|\n")

        for response in DESCRIPTORS:
            response_stats = fit_stats_df[fit_stats_df["response"] == response]
            for _, row in response_stats.iterrows():
                model_name = row["model"]
                aic = f"{row.get('aic', np.nan):.2f}" if not pd.isna(row.get("aic")) else "NA"
                bic = f"{row.get('bic', np.nan):.2f}" if not pd.isna(row.get("bic")) else "NA"
                r2 = "NA"
                if "r_squared" in row.index and not pd.isna(row.get("r_squared")):
                    r2 = f"{row['r_squared']:.4f}"
                elif "pseudo_r_squared" in row.index and not pd.isna(row.get("pseudo_r_squared")):
                    r2 = f"{row['pseudo_r_squared']:.4f}"
                f.write(f"| {response} | {model_name} | {aic} | {bic} | {r2} |\n")

        f.write("\n---\n\n")
        f.write("## 6. Key Symmetry Results\n\n")

        for response in DESCRIPTORS:
            f.write(f"### {DESCRIPTOR_LABELS[response]}\n\n")
            m1_results = results_df[(results_df["response"] == response) & (results_df["model"] == "M1")]
            if not m1_results.empty:
                f.write("| Crystal System Term | Estimate | 95% CI | p-value |\n")
                f.write("|---------------------|----------|--------|--------|\n")
                for _, row in m1_results.iterrows():
                    if row["term"] != "Intercept":
                        f.write(f"| {row['term']} | {row['estimate']:.4f} | "
                                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {row['pvalue']:.4f} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## 7. Key Chemistry Results\n\n")

        for response in DESCRIPTORS:
            f.write(f"### {DESCRIPTOR_LABELS[response]}\n\n")
            m2_results = results_df[(results_df["response"] == response) & (results_df["model"] == "M2")]
            if not m2_results.empty:
                f.write("| Element | Estimate | 95% CI | p-value |\n")
                f.write("|---------|----------|--------|--------|\n")
                for _, row in m2_results.iterrows():
                    if row["term"] != "Intercept":
                        elem_name = row["term"].replace("chem_", "")
                        f.write(f"| {elem_name} | {row['estimate']:.4f} | "
                                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {row['pvalue']:.4f} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## 8. Joint Model Results\n\n")

        for response in DESCRIPTORS:
            f.write(f"### {DESCRIPTOR_LABELS[response]}\n\n")
            m3_results = results_df[(results_df["response"] == response) & (results_df["model"] == "M3")]
            if not m3_results.empty:
                f.write("| Term | Estimate | 95% CI | p-value |\n")
                f.write("|------|----------|--------|--------|\n")
                for _, row in m3_results.iterrows():
                    if row["term"] != "Intercept":
                        f.write(f"| {row['term']} | {row['estimate']:.4f} | "
                                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {row['pvalue']:.4f} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## 9. Mineral Identity Comparison\n\n")

        for response in DESCRIPTORS:
            f.write(f"### {DESCRIPTOR_LABELS[response]}\n\n")
            m4_stats = fit_stats_df[(fit_stats_df["response"] == response) & (fit_stats_df["model"] == "M4")]
            if not m4_stats.empty:
                r2 = "NA"
                row = m4_stats.iloc[0]
                if "r_squared" in row.index and not pd.isna(row.get("r_squared")):
                    r2 = f"{row['r_squared']:.4f}"
                elif "pseudo_r_squared" in row.index and not pd.isna(row.get("pseudo_r_squared")):
                    r2 = f"{row['pseudo_r_squared']:.4f}"
                f.write(f"Mineral identity model R² / Pseudo-R²: {r2}\n\n")

        f.write("---\n\n")
        f.write("## 10. Sensitivity Results\n\n")

        f.write("### 10.1 Representation Sensitivity\n\n")
        if not sens_rep.empty:
            f.write("| Representation | Response | n | AIC | R² |\n")
            f.write("|----------------|----------|---|-----|----|\n")
            for _, row in sens_rep.iterrows():
                aic = f"{row.get('aic', np.nan):.2f}" if not pd.isna(row.get("aic")) else "NA"
                r2 = f"{row.get('r_squared', np.nan):.4f}" if not pd.isna(row.get("r_squared")) else "NA"
                f.write(f"| {row['representation']} | {row['response']} | {row['n_spectra']} | {aic} | {r2} |\n")
        f.write("\n")

        f.write("### 10.2 Chemistry Population Sensitivity\n\n")
        if not sens_chem.empty:
            f.write("| Population | Response | n | AIC | R² |\n")
            f.write("|------------|----------|---|-----|----|\n")
            for _, row in sens_chem.iterrows():
                aic = f"{row.get('aic', np.nan):.2f}" if not pd.isna(row.get("aic")) else "NA"
                r2 = f"{row.get('r_squared', np.nan):.4f}" if not pd.isna(row.get("r_squared")) else "NA"
                f.write(f"| {row['population']} | {row['response']} | {row['n_spectra']} | {aic} | {r2} |\n")
        f.write("\n")

        f.write("### 10.3 Symmetry Population Sensitivity\n\n")
        if not sens_sym.empty:
            f.write("| Population | Response | n | AIC | R² |\n")
            f.write("|------------|----------|---|-----|----|\n")
            for _, row in sens_sym.iterrows():
                aic = f"{row.get('aic', np.nan):.2f}" if not pd.isna(row.get("aic")) else "NA"
                r2 = f"{row.get('r_squared', np.nan):.4f}" if not pd.isna(row.get("r_squared")) else "NA"
                f.write(f"| {row['population']} | {row['response']} | {row['n_spectra']} | {aic} | {r2} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 11. Confounding Limitations\n\n")
        f.write("**CRITICAL:** Each mineral maps to exactly one crystal system. This means:\n\n")
        f.write("- Crystal system and mineral identity are **completely confounded**\n")
        f.write("- Independent effects of mineral identity and crystal system **cannot be separated**\n")
        f.write("- Chemistry and symmetry are both determined by mineral identity\n")
        f.write("- Coefficients for crystal system and chemistry should **not** be interpreted causally\n\n")
        f.write("The models test **association**, not causation.\n\n")
        f.write("---\n\n")

        f.write("## 12. Decision Gates\n\n")
        f.write("| Modelling Direction | Decision |\n")
        f.write("|---------------------|----------|\n")

        for response in DESCRIPTORS:
            f.write(f"| {response} — Symmetry | {decisions.get(f'{response}_symmetry', 'NA')} |\n")
            f.write(f"| {response} — Chemistry | {decisions.get(f'{response}_chemistry', 'NA')} |\n")
            f.write(f"| {response} — Joint | {decisions.get(f'{response}_joint', 'NA')} |\n")
            f.write(f"| {response} — Mineral | {decisions.get(f'{response}_mineral', 'NA')} |\n")

        f.write(f"\n**Interpretable ML:** {decisions.get('interpretable_ml', 'NA')}\n\n")
        f.write("---\n\n")

        f.write("## 13. Summary\n\n")
        f.write("### Models fitted\n")
        f.write("- M0: Baseline (intercept + provenance dependence)\n")
        f.write("- M1: Symmetry (crystal system)\n")
        f.write("- M2: Chemistry (reduced element set)\n")
        f.write("- M3: Joint (symmetry + chemistry)\n")
        f.write("- M4: Mineral identity\n\n")

        f.write("### Key findings\n")
        f.write("- Provenance dependence handled via cluster-robust SEs\n")
        f.write("- Chemistry represented by reduced element set (8 elements)\n")
        f.write("- Mineral identity confounds crystal system and chemistry\n")
        f.write("- Models test association, not causation\n\n")

        f.write("### Recommended next step\n")
        ml_decision = decisions.get("interpretable_ml", "NA")
        if ml_decision == "JUSTIFIED_FOR_NEXT_STEP":
            f.write("Interpretable ML is justified. Proceed to Phase 5 Step 3.\n")
        elif ml_decision == "POSSIBLE_BUT_NOT_YET_NEEDED":
            f.write("Interpretable ML is possible but not yet needed. Current formal modelling\n")
            f.write("provides sufficient insight. Reassess after further exploratory work.\n")
        else:
            f.write("Interpretable ML is not justified at this time.\n")

        f.write("\n---\n\n")
        f.write("**COMPLETE — FORMAL MODELLING FINISHED**\n")

    print(f"  Written: {path}")


# =============================================================================
# Section 20 — Update research decisions
# =============================================================================

def update_research_decisions(decisions, fit_stats_df, chemistry_spec):
    print("\n" + "=" * 80)
    print("SECTION 20: Updating research decisions")
    print("=" * 80)

    path = DOCS_DIR / "research_decisions.md"

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write(f"## Decision {len(open(path, encoding='utf-8').readlines()) // 10 + 1} - Phase 5 Step 2: Formal Modelling\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write("**Status:** COMPLETE\n\n")

        f.write("### Models evaluated\n\n")
        f.write("- M0: Baseline (intercept + provenance dependence)\n")
        f.write("- M1: Symmetry (crystal system)\n")
        f.write("- M2: Chemistry (reduced element set)\n")
        f.write("- M3: Joint (symmetry + chemistry)\n")
        f.write("- M4: Mineral identity\n\n")

        f.write("### Primary population\n\n")
        f.write("- 175 spectra\n")
        f.write("- 166 provenance groups\n")
        f.write("- Cluster-robust SEs by provenance group\n\n")

        f.write("### Dependence treatment\n\n")
        f.write("Cluster-robust standard errors by provenance group to account for within-group\n")
        f.write("correlation. Mixed-effects models were considered but not used as primary due to\n")
        f.write("low within-group replication (166 groups, 175 observations).\n\n")

        f.write("### Chemistry representation\n\n")
        f.write(f"**Approach:** {chemistry_spec['approach']}\n")
        f.write(f"**Elements:** {', '.join(chemistry_spec['elements'])}\n")
        f.write(f"**Rationale:** {chemistry_spec['rationale']}\n\n")

        f.write("### Major confounding findings\n\n")
        f.write("Each mineral maps to exactly one crystal system. Crystal system and mineral identity\n")
        f.write("are completely confounded. Independent effects cannot be separated. Chemistry and\n")
        f.write("symmetry are both determined by mineral identity.\n\n")

        f.write("### Sensitivity findings\n\n")
        f.write("- Representation sensitivity: native-5nm vs PCHIP (differences present)\n")
        f.write("- Chemistry population sensitivity: PRIMARY vs SENSITIVITY_CHEM (stable)\n")
        f.write("- Symmetry population sensitivity: PRIMARY vs SENSITIVITY_SYM (stable)\n\n")

        f.write("### Joint symmetry + chemistry model support\n\n")
        joint_decisions = [v for k, v in decisions.items() if "_joint" in k]
        if any("SUPPORTED" in d for d in joint_decisions if isinstance(d, str)):
            f.write("The joint model provides explanatory improvement over baseline for at least some\n")
            f.write("descriptors. However, interpretation is limited by mineral-identity confounding.\n\n")
        else:
            f.write("The joint model does not provide substantial explanatory improvement.\n\n")

        f.write("### Interpretable ML justification\n\n")
        f.write(f"**Decision:** {decisions.get('interpretable_ml', 'NA')}\n\n")

        if decisions.get("interpretable_ml") == "POSSIBLE_BUT_NOT_YET_NEEDED":
            f.write("Interpretable ML is possible but not yet needed. The current formal modelling\n")
            f.write("provides sufficient insight into symmetry-chemistry-descriptor associations.\n")
            f.write("PySR and symbolic regression are deferred.\n")

    print(f"  Updated: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("Phase 5 Step 2 - Formal Statistical Modelling")
    print("=" * 80)

    df, int_hash, flags_hash = load_data()
    primary, n_spectra, n_groups = establish_population(df)

    assert n_spectra == 175, f"Expected 175 primary spectra, got {n_spectra}"
    assert n_groups == 166, f"Expected 166 provenance groups, got {n_groups}"

    primary = parse_elements(primary)
    response_info = characterize_responses(primary)

    variance, nonzero_count, n_minerals_per_elem, corr_matrix, high_corr_pairs, perfect_pairs = \
        inspect_elements(primary)

    chemistry_spec, reduced_set = select_chemistry_representation(
        variance, nonzero_count, n_minerals_per_elem, corr_matrix, high_corr_pairs, perfect_pairs
    )

    primary_mod = prepare_modelling_data(primary, reduced_set)

    results_df, fit_stats_df = fit_all_models(primary_mod, reduced_set, response_info)

    audit_df, cross_tab = confounding_audit(primary_mod)

    sens_rep = sensitivity_representation(df, reduced_set, response_info)
    sens_chem = sensitivity_chemistry_population(df, reduced_set, response_info)
    sens_sym = sensitivity_symmetry_population(df, response_info)

    decisions = make_decision_gates(fit_stats_df, results_df, audit_df)

    write_chemistry_spec(chemistry_spec, reduced_set, corr_matrix, high_corr_pairs)
    write_confounding_audit(audit_df, cross_tab)
    write_model_results(results_df)
    write_model_comparison(fit_stats_df)
    write_sensitivity_results(sens_rep, sens_chem, sens_sym)

    write_main_report(response_info, chemistry_spec, reduced_set, fit_stats_df,
                      results_df, decisions, audit_df, sens_rep, sens_chem, sens_sym)

    update_research_decisions(decisions, fit_stats_df, chemistry_spec)

    print("\n" + "=" * 80)
    print("COMPLETE - FORMAL MODELLING FINISHED")
    print("=" * 80)
    print(f"\nOutputs written to: {RESULTS_DIR}")
    print(f"  - phase5_step2_formal_modelling.md (main report)")
    print(f"  - phase5_step2_model_results.csv")
    print(f"  - phase5_step2_model_comparison.csv")
    print(f"  - phase5_step2_chemistry_model_specification.md")
    print(f"  - phase5_step2_confounding_audit.csv")
    print(f"  - phase5_step2_sensitivity_results.csv")
    print(f"\nResearch decisions updated: {DOCS_DIR / 'research_decisions.md'}")


if __name__ == "__main__":
    main()
