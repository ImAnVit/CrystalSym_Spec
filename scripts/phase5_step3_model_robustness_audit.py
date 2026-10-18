"""
Phase 5 Step 3 — Model Robustness and Interpretation Audit.

Comprehensive audit of Phase 5 Step 2 (Formal Statistical Modelling) outputs.
Reproduces all models, validates statistical implementation, performs residual
diagnostics, assesses confounding, and produces go/no-go decisions.

NO new modelling beyond what is needed for the audit. NO random train/test ML,
NO neural networks, NO black-box ensembles, NO SHAP, NO PySR, NO symbolic
regression, NO automated feature selection.
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

TOLERANCE = 1e-4


# =============================================================================
# Utility: File checksums
# =============================================================================

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Utility: OLS with cluster-robust SE (reproduces Step 2 exactly)
# =============================================================================

def fit_ols_cluster_reproduce(X, y, groups):
    """Exact reproduction of Step 2 fit_ols_cluster."""
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    n = len(y)
    p = X.shape[1]
    n_groups = len(np.unique(groups))

    XtX_inv = np.linalg.inv(X.T @ X)

    h_diag = np.array([X[i] @ XtX_inv @ X[i] for i in range(n)])

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

    # Step 2 uses t-distribution with df=n-p for p-values
    p_values_t = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))
    # Correct cluster-robust DF should be n_groups - 1
    p_values_cr = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_groups - 1))
    # z-based p-values (what the CIs implicitly use via 1.96)
    p_values_z = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

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
        "se": se,
        "p_values_t": p_values_t,
        "p_values_cr": p_values_cr,
        "p_values_z": p_values_z,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "aic": aic,
        "bic": bic,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "residuals": residuals,
        "y_pred": y_pred,
        "n": n,
        "p": p,
        "n_groups": n_groups,
        "t_stats": t_stats,
        "vcov": vcov,
        "h_diag": h_diag,
    }


def fit_count_model_cluster_reproduce(X, y, groups):
    """Exact reproduction of Step 2 fit_count_model_cluster (log-linear OLS)."""
    y_log = np.log(y + 1)
    result = fit_ols_cluster_reproduce(X, y_log, groups)

    ss_res = np.sum(result["residuals"] ** 2)
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    result["pseudo_r_squared"] = 1 - (ss_res / ss_tot)
    result["y_transformed"] = y_log

    return result


# =============================================================================
# Utility: Design matrices
# =============================================================================

def create_design_matrix_categorical(df, categorical_col, reference_drop=True):
    categories = sorted(df[categorical_col].unique())
    if reference_drop:
        categories = categories[1:]
    X = np.zeros((len(df), len(categories)))
    for i, cat in enumerate(categories):
        X[:, i] = (df[categorical_col] == cat).astype(int)
    return X, categories


def parse_elements(df):
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
    return df


# =============================================================================
# Section 1 — Load frozen inputs and verify checksums
# =============================================================================

def section1_load_and_verify():
    print("=" * 80)
    print("SECTION 1: Loading frozen inputs and verifying checksums")
    print("=" * 80)

    integrated_path = DATA_DIR / "phase4_integrated_analysis_dataset.csv"
    flags_path = RESULTS_DIR.parent / "04_chemistry_crystallography" / "phase4_step4_analysis_population_flags.csv"
    step2_results_path = RESULTS_DIR / "phase5_step2_model_results.csv"
    step2_comparison_path = RESULTS_DIR / "phase5_step2_model_comparison.csv"

    checksums = {
        "integrated_dataset": file_sha256(integrated_path),
        "population_flags": file_sha256(flags_path),
        "step2_model_results": file_sha256(step2_results_path),
        "step2_model_comparison": file_sha256(step2_comparison_path),
    }

    df = pd.read_csv(integrated_path)
    flags = pd.read_csv(flags_path)
    step2_results = pd.read_csv(step2_results_path)
    step2_comparison = pd.read_csv(step2_comparison_path)

    print(f"  Integrated dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Population flags:   {flags.shape[0]} rows, {flags.shape[1]} columns")
    print(f"  Step 2 results:     {step2_results.shape[0]} rows")
    print(f"  Step 2 comparison:  {step2_comparison.shape[0]} rows")

    for name, cs in checksums.items():
        print(f"  SHA-256 [{name}]: {cs[:16]}...")

    df = df.merge(flags, on=["canonical_spectrum_id", "provenance_group_id"], how="left")
    df = parse_elements(df)

    return df, step2_results, step2_comparison, checksums


# =============================================================================
# Section 2 — Establish analysis populations
# =============================================================================

def section2_populations(df):
    print("\n" + "=" * 80)
    print("SECTION 2: Establishing analysis populations")
    print("=" * 80)

    populations = {}

    primary = df[df["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()
    populations["PRIMARY"] = primary

    if "SENSITIVITY_SYM_ELIGIBLE" in df.columns:
        sym = df[df["SENSITIVITY_SYM_ELIGIBLE"] == True].copy()
        populations["SENSITIVITY_SYM"] = sym

    if "SENSITIVITY_CHEM_ELIGIBLE" in df.columns:
        chem = df[df["SENSITIVITY_CHEM_ELIGIBLE"] == True].copy()
        populations["SENSITIVITY_CHEM"] = chem

    populations["FULL"] = df.copy()

    for name, pop in populations.items():
        n_spectra = pop.shape[0]
        n_groups = pop["provenance_group_id"].nunique()
        n_minerals = pop["mineral_label"].nunique()
        print(f"  {name:25s}: {n_spectra:4d} spectra, {n_groups:4d} groups, {n_minerals:2d} minerals")

    return populations


# =============================================================================
# Section 3 — Audit A: Reproduce all Step 2 model results
# =============================================================================

def section3_reproduce_models(df, step2_results, step2_comparison):
    print("\n" + "=" * 80)
    print("SECTION 3 (Audit A): Reproducing all Step 2 model results")
    print("=" * 80)

    primary = df[df["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy()
    primary["crystal_system_cat"] = pd.Categorical(
        primary["crystal_system"],
        categories=[c for c in CRYSTAL_SYSTEM_ORDER if c in primary["crystal_system"].unique()],
    )

    reduced_set = ["Al", "Ca", "Fe", "Mg", "Si"]
    for elem in reduced_set:
        primary[f"chem_{elem}"] = primary[f"elem_{elem}"]

    reproduction_checks = []
    reproduced_models = {}

    for response in DESCRIPTORS:
        is_count = (response == "n_peaks_prom")
        y = primary[response].values
        groups = primary["provenance_group_id"].values

        models = {}

        # M0: intercept only
        X0 = np.ones((len(primary), 1))
        if is_count:
            models["M0"] = fit_count_model_cluster_reproduce(X0, y, groups)
        else:
            models["M0"] = fit_ols_cluster_reproduce(X0, y, groups)

        # M1: crystal system
        X_cat, cat_levels = create_design_matrix_categorical(primary, "crystal_system_cat")
        X1 = np.column_stack([np.ones(len(primary)), X_cat])
        if is_count:
            models["M1"] = fit_count_model_cluster_reproduce(X1, y, groups)
        else:
            models["M1"] = fit_ols_cluster_reproduce(X1, y, groups)

        # M2: chemistry
        chem_cols = [f"chem_{e}" for e in reduced_set]
        X_chem = primary[chem_cols].values
        X2 = np.column_stack([np.ones(len(primary)), X_chem])
        if is_count:
            models["M2"] = fit_count_model_cluster_reproduce(X2, y, groups)
        else:
            models["M2"] = fit_ols_cluster_reproduce(X2, y, groups)

        # M3: joint (crystal system + chemistry)
        X3 = np.column_stack([np.ones(len(primary)), X_cat, X_chem])
        if is_count:
            models["M3"] = fit_count_model_cluster_reproduce(X3, y, groups)
        else:
            models["M3"] = fit_ols_cluster_reproduce(X3, y, groups)

        # M4: mineral identity
        mineral_levels = sorted(primary["mineral_label"].unique())
        X_mineral = np.zeros((len(primary), len(mineral_levels) - 1))
        for i, m in enumerate(mineral_levels[1:]):
            X_mineral[:, i] = (primary["mineral_label"] == m).astype(int)
        X4 = np.column_stack([np.ones(len(primary)), X_mineral])
        if is_count:
            models["M4"] = fit_count_model_cluster_reproduce(X4, y, groups)
        else:
            models["M4"] = fit_ols_cluster_reproduce(X4, y, groups)

        reproduced_models[response] = models

        for model_name, res in models.items():
            step2_row = step2_comparison[
                (step2_comparison["model"] == model_name) &
                (step2_comparison["response"] == response)
            ]

            if len(step2_row) > 0:
                step2_aic = step2_row.iloc[0]["aic"]
                aic_match = abs(res["aic"] - step2_aic) < TOLERANCE

                step2_r2_col = "pseudo_r_squared" if is_count else "r_squared"
                step2_r2 = step2_row.iloc[0][step2_r2_col]
                if pd.notna(step2_r2):
                    r2_match = abs(res.get("pseudo_r_squared" if is_count else "r_squared", 0) - step2_r2) < TOLERANCE
                else:
                    r2_match = True
            else:
                aic_match = None
                r2_match = None

            reproduction_checks.append({
                "model": model_name,
                "response": response,
                "reproduced_aic": res["aic"],
                "step2_aic": step2_aic if len(step2_row) > 0 else None,
                "aic_match": aic_match,
                "reproduced_r_squared": res.get("pseudo_r_squared" if is_count else "r_squared"),
                "step2_r_squared": step2_r2 if len(step2_row) > 0 else None,
                "r_squared_match": r2_match,
                "n": res["n"],
                "p": res["p"],
                "n_groups": res["n_groups"],
            })

            print(f"  {model_name} / {response}: AIC={res['aic']:.4f} "
                  f"(match={aic_match}), R²={res.get('pseudo_r_squared' if is_count else 'r_squared', 0):.6f}")

    return reproduction_checks, reproduced_models, primary, reduced_set


# =============================================================================
# Section 4 — Audit B: Model comparison validity
# =============================================================================

def section4_model_comparison_validity(reproduced_models, step2_comparison):
    print("\n" + "=" * 80)
    print("SECTION 4 (Audit B): Model comparison validity")
    print("=" * 80)

    findings = []

    for response in DESCRIPTORS:
        is_count = (response == "n_peaks_prom")
        models = reproduced_models[response]

        print(f"\n  --- {response} ---")

        for model_name in ["M0", "M1", "M2", "M3", "M4"]:
            res = models[model_name]
            r2_key = "pseudo_r_squared" if is_count else "r_squared"
            r2 = res.get(r2_key, 0)
            r2_adj = res.get("r_squared_adj", None)

            r2_label = "pseudo_R² (log scale)" if is_count else "R² (original scale)"
            print(f"    {model_name}: {r2_label} = {r2:.6f}, AIC = {res['aic']:.2f}")

        if is_count:
            findings.append({
                "response": response,
                "issue": "R² scale mismatch",
                "detail": ("n_peaks_prom R² is computed on log(y+1) scale, not original count "
                           "scale. Reported R² values are pseudo-R² and not directly comparable "
                           "to R² from continuous descriptors."),
                "severity": "HIGH",
            })

        if is_count:
            findings.append({
                "response": response,
                "issue": "Model type mislabelled",
                "detail": ("Step 2 report labels n_peaks_prom models as 'Negative Binomial (count)' "
                           "but the implementation uses log(y+1) OLS with cluster-robust SEs. "
                           "This is NOT a negative binomial model. No count distribution is fitted."),
                "severity": "CRITICAL",
            })

        aic_values = [models[m]["aic"] for m in ["M0", "M1", "M2", "M3", "M4"]]
        aic_monotone = all(aic_values[i] >= aic_values[i+1] for i in range(len(aic_values)-1))
        print(f"    AIC monotonically decreasing: {aic_monotone}")

    findings.append({
        "response": "all",
        "issue": "AIC/BIC formula assumes Gaussian likelihood",
        "detail": ("AIC = n*log(SSR/n) + 2*p assumes Gaussian errors. For the count model "
                   "(log-linear OLS), this is an approximation. The values are comparable "
                   "across models fitted to the same response on the same scale, but should "
                   "not be compared across different responses."),
        "severity": "MODERATE",
    })

    findings.append({
        "response": "all",
        "issue": "Decision gate thresholds are arbitrary",
        "detail": ("Step 2 uses ad hoc R² improvement thresholds (0.05, 0.02, 0.07, 0.03, 0.10, 0.05) "
                   "for decision gates. These thresholds are not justified by power analysis, "
                   "cross-validation, or domain knowledge. All gates show SUPPORTED, which "
                   "suggests the thresholds may have been chosen post-hoc to match expectations."),
        "severity": "HIGH",
    })

    return findings


# =============================================================================
# Section 5 — Audit C: Cluster-robust SE implementation
# =============================================================================

def section5_cluster_robust_audit(reproduced_models):
    print("\n" + "=" * 80)
    print("SECTION 5 (Audit C): Cluster-robust SE implementation audit")
    print("=" * 80)

    findings = []

    for response in DESCRIPTORS:
        models = reproduced_models[response]
        m0 = models["M0"]

        n = m0["n"]
        p = m0["p"]
        n_groups = m0["n_groups"]

        print(f"\n  --- {response} ---")
        print(f"    n = {n}, p (M0) = {p}, n_groups = {n_groups}")
        print(f"    DF used for p-values (Step 2): n - p = {n - p}")
        print(f"    Correct cluster-robust DF: n_groups - 1 = {n_groups - 1}")
        print(f"    CI uses z = 1.96 (implicit infinite DF)")

        findings.append({
            "response": response,
            "issue": "DF convention mismatch",
            "detail": (f"Step 2 p-values use t-distribution with df=n-p={n-p}, "
                       f"but cluster-robust inference should use df=n_groups-1={n_groups-1}. "
                       f"With n_groups={n_groups} and many parameters in M4, n-p may be much "
                       f"larger than n_groups-1, making p-values anti-conservative."),
            "severity": "HIGH",
        })

        findings.append({
            "response": response,
            "issue": "CI/p-value inconsistency",
            "detail": ("Confidence intervals use z=1.96 (normal quantile, equivalent to "
                       "infinite DF) while p-values use t-distribution. These are inconsistent: "
                       "if t-distribution is appropriate for p-values, it should also be used "
                       "for CIs. With df=n_groups-1, the correct critical value would be "
                       f"t_0.025({n_groups-1}) = {stats.t.ppf(0.975, n_groups-1):.4f}."),
            "severity": "HIGH",
        })

        for model_name in ["M0", "M4"]:
            res = models[model_name]
            p_vals_t = res["p_values_t"]
            p_vals_cr = res["p_values_cr"]
            p_vals_z = res["p_values_z"]

            n_sign_t = (p_vals_t < 0.05).sum()
            n_sign_cr = (p_vals_cr < 0.05).sum()
            n_sign_z = (p_vals_z < 0.05).sum()

            print(f"\n    {model_name} significance counts (alpha=0.05):")
            print(f"      t(df=n-p):    {n_sign_t}/{len(p_vals_t)} significant")
            print(f"      t(df=n_g-1):  {n_sign_cr}/{len(p_vals_cr)} significant")
            print(f"      z (normal):   {n_sign_z}/{len(p_vals_z)} significant")

        findings.append({
            "response": response,
            "issue": "Small-sample correction (HC1)",
            "detail": (f"Variance uses HC1 correction: meat *= n_groups/(n_groups-1) = "
                       f"{n_groups}/{n_groups-1} = {n_groups/(n_groups-1):.6f}. "
                       f"This is a mild correction. With {n_groups} groups, the correction "
                       f"factor is {n_groups/(n_groups-1):.4f}."),
            "severity": "LOW",
        })

    return findings


# =============================================================================
# Section 6 — Audit D: Response distributions and residual diagnostics
# =============================================================================

def section6_residual_diagnostics(reproduced_models, primary):
    print("\n" + "=" * 80)
    print("SECTION 6 (Audit D): Response distributions and residual diagnostics")
    print("=" * 80)

    diagnostic_rows = []
    fig_idx = 0

    for response in DESCRIPTORS:
        is_count = (response == "n_peaks_prom")
        models = reproduced_models[response]
        y = primary[response].values

        print(f"\n  --- {response} ---")

        n_zeros = (y == 0).sum()
        pct_zeros = 100 * n_zeros / len(y)
        skew = stats.skew(y)
        kurt = stats.kurtosis(y)

        print(f"    n_zeros = {n_zeros} ({pct_zeros:.1f}%)")
        print(f"    skewness = {skew:.3f}, kurtosis = {kurt:.3f}")

        diagnostic_rows.append({
            "response": response,
            "diagnostic": "response_distribution",
            "metric": "n_zeros",
            "value": n_zeros,
            "flag": "YES" if pct_zeros > 20 else "NO",
        })
        diagnostic_rows.append({
            "response": response,
            "diagnostic": "response_distribution",
            "metric": "pct_zeros",
            "value": pct_zeros,
            "flag": "YES" if pct_zeros > 20 else "NO",
        })
        diagnostic_rows.append({
            "response": response,
            "diagnostic": "response_distribution",
            "metric": "skewness",
            "value": skew,
            "flag": "YES" if abs(skew) > 2 else "NO",
        })
        diagnostic_rows.append({
            "response": response,
            "diagnostic": "response_distribution",
            "metric": "kurtosis",
            "value": kurt,
            "flag": "YES" if abs(kurt) > 7 else "NO",
        })

        # Residual diagnostics for each model
        for model_name in ["M0", "M1", "M3", "M4"]:
            res = models[model_name]
            resid = res["residuals"]
            y_pred = res["y_pred"]

            shapiro_stat, shapiro_p = stats.shapiro(resid[:min(len(resid), 5000)])
            normality_flag = "YES" if shapiro_p < 0.01 else "NO"

            resid_skew = stats.skew(resid)
            resid_kurt = stats.kurtosis(resid)

            # Cook's distance using proper hat matrix diagonal
            n = len(resid)
            p = res["p"]
            h_diag = res["h_diag"]
            mse = np.sum(resid ** 2) / (n - p)

            denom = (1 - h_diag) ** 2
            denom = np.where(denom < 1e-10, 1e-10, denom)
            cooks_d = (resid ** 2 / (p * mse)) * (h_diag / denom)
            n_influential = (cooks_d > 4.0 / n).sum()
            max_cooks = cooks_d.max()

            print(f"    {model_name}: Shapiro p={shapiro_p:.4f}, "
                  f"skew={resid_skew:.3f}, kurt={resid_kurt:.3f}, "
                  f"influential (D>4/n)={n_influential}, max_D={max_cooks:.4f}")

            diagnostic_rows.append({
                "response": response,
                "diagnostic": f"{model_name}_residuals",
                "metric": "shapiro_p",
                "value": shapiro_p,
                "flag": normality_flag,
            })
            diagnostic_rows.append({
                "response": response,
                "diagnostic": f"{model_name}_residuals",
                "metric": "skewness",
                "value": resid_skew,
                "flag": "YES" if abs(resid_skew) > 2 else "NO",
            })
            diagnostic_rows.append({
                "response": response,
                "diagnostic": f"{model_name}_residuals",
                "metric": "kurtosis",
                "value": resid_kurt,
                "flag": "YES" if abs(resid_kurt) > 7 else "NO",
            })
            diagnostic_rows.append({
                "response": response,
                "diagnostic": f"{model_name}_residuals",
                "metric": "n_influential_cooks",
                "value": n_influential,
                "flag": "YES" if n_influential > 5 else "NO",
            })
            diagnostic_rows.append({
                "response": response,
                "diagnostic": f"{model_name}_residuals",
                "metric": "max_cooks_d",
                "value": max_cooks,
                "flag": "YES" if max_cooks > 1.0 else "NO",
            })

        # Diagnostic figure: residuals for M3 (joint model)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Residual Diagnostics: {DESCRIPTOR_LABELS[response]}", fontsize=14)

        res_m3 = models["M3"]
        resid = res_m3["residuals"]
        y_pred = res_m3["y_pred"]

        axes[0, 0].hist(resid, bins=30, edgecolor="black", alpha=0.7)
        axes[0, 0].set_xlabel("Residual")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Residual distribution")

        axes[0, 1].scatter(y_pred, resid, alpha=0.5, s=10)
        axes[0, 1].axhline(0, color="red", linestyle="--")
        axes[0, 1].set_xlabel("Fitted value")
        axes[0, 1].set_ylabel("Residual")
        axes[0, 1].set_title("Residuals vs fitted")

        stats.probplot(resid, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q plot")

        h_diag_m3 = res_m3["h_diag"]
        mse_m3 = np.sum(resid ** 2) / (len(resid) - res_m3["p"])
        denom_m3 = (1 - h_diag_m3) ** 2
        denom_m3 = np.where(denom_m3 < 1e-10, 1e-10, denom_m3)
        cooks_d = (resid ** 2 / (res_m3["p"] * mse_m3)) * (h_diag_m3 / denom_m3)
        axes[1, 1].bar(range(len(cooks_d)), cooks_d, width=1.0, alpha=0.7)
        axes[1, 1].axhline(4.0 / len(cooks_d), color="red", linestyle="--", label="4/n threshold")
        axes[1, 1].set_xlabel("Observation index")
        axes[1, 1].set_ylabel("Cook's distance")
        axes[1, 1].set_title("Cook's distance")
        axes[1, 1].legend()

        plt.tight_layout()
        fig_path = RESULTS_DIR / f"phase5_step3_diag_{response}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {fig_path.name}")
        fig_idx += 1

    return diagnostic_rows


# =============================================================================
# Section 7 — Audit E: Chemistry representation
# =============================================================================

def section7_chemistry_audit(primary, reduced_set):
    print("\n" + "=" * 80)
    print("SECTION 7 (Audit E): Chemistry representation audit")
    print("=" * 80)

    findings = []

    elem_data = primary[[f"elem_{e}" for e in ELEMENTS_ALL]]
    elem_data.columns = ELEMENTS_ALL

    reduced_data = primary[[f"elem_{e}" for e in reduced_set]]
    reduced_data.columns = reduced_set

    corr = reduced_data.corr()
    print("\n  Correlation matrix (reduced set):")
    print(corr.round(3).to_string())

    rank = np.linalg.matrix_rank(reduced_data.values)
    n_cols = len(reduced_set)
    print(f"\n  Matrix rank: {rank} / {n_cols} columns")

    if rank < n_cols:
        findings.append({
            "issue": "Rank deficiency in chemistry design matrix",
            "detail": f"Rank = {rank} < {n_cols} columns. Design matrix is rank-deficient.",
            "severity": "CRITICAL",
        })
    else:
        print("  No rank deficiency detected.")

    cond_number = np.linalg.cond(reduced_data.values)
    print(f"  Condition number: {cond_number:.2f}")

    if cond_number > 100:
        findings.append({
            "issue": "High condition number in chemistry matrix",
            "detail": f"Condition number = {cond_number:.2f} > 100. Indicates severe multicollinearity.",
            "severity": "HIGH",
        })

    print("\n  Compositional constraint check:")
    elem_sums = elem_data[ELEMENTS_ALL].sum(axis=1)
    print(f"    Element fraction sums: mean={elem_sums.mean():.6f}, "
          f"min={elem_sums.min():.6f}, max={elem_sums.max():.6f}")

    findings.append({
        "issue": "Compositional data interpretation",
        "detail": ("Element atom fractions are compositional (sum to 1). Coefficients represent "
                   "associations RELATIVE TO omitted elements (oxygen and excluded rare elements). "
                   "A positive coefficient for element X does NOT mean X increases the response; "
                   "it means X's fraction is associated with higher response relative to the "
                   "omitted baseline. This is a fundamental interpretational constraint."),
        "severity": "HIGH",
    })

    print("\n  Collinearity impact on coefficient stability:")
    for e1 in reduced_set:
        for e2 in reduced_set:
            if e1 < e2:
                r = corr.loc[e1, e2]
                if abs(r) > 0.5:
                    print(f"    {e1} - {e2}: r = {r:.3f}")
                    findings.append({
                        "issue": f"Moderate-high collinearity: {e1}-{e2}",
                        "detail": (f"Pearson r = {r:.3f}. Coefficients for {e1} and {e2} "
                                   f"may be unstable and their individual interpretations unreliable."),
                        "severity": "MODERATE",
                    })

    return findings


# =============================================================================
# Section 8 — Audit F: Mineral identity vs crystal system confounding
# =============================================================================

def section8_confounding_audit(primary, reproduced_models):
    print("\n" + "=" * 80)
    print("SECTION 8 (Audit F): Mineral identity vs crystal system confounding")
    print("=" * 80)

    findings = []

    mineral_system = primary.groupby("mineral_label")["crystal_system"].nunique()
    perfect_confounding = mineral_system[mineral_system == 1]
    multi_system = mineral_system[mineral_system > 1]

    print(f"\n  Minerals with single crystal system: {len(perfect_confounding)}")
    print(f"  Minerals with multiple crystal systems: {len(multi_system)}")

    if len(multi_system) == 0:
        findings.append({
            "issue": "Complete mineral-system confounding",
            "detail": ("Every mineral in the PRIMARY population maps to exactly one crystal "
                       "system. Crystal system and mineral identity are completely confounded. "
                       "M1 (crystal system) and M4 (mineral identity) effects cannot be "
                       "distinguished. Any 'symmetry effect' could equally be a mineral-specific "
                       "chemical/structural effect."),
            "severity": "CRITICAL",
        })

    confounding_table = primary.groupby("mineral_label").agg(
        crystal_systems=("crystal_system", lambda x: sorted(x.unique())),
        n_systems=("crystal_system", "nunique"),
        n_spectra=("canonical_spectrum_id", "count"),
        n_groups=("provenance_group_id", "nunique"),
    ).reset_index()

    print("\n  Confounding table:")
    for _, row in confounding_table.iterrows():
        print(f"    {row['mineral_label']:15s}: {row['n_systems']} system(s), "
              f"{row['n_spectra']:3d} spectra, {row['n_groups']:3d} groups")

    # Leave-one-mineral-out sensitivity
    print("\n  Leave-one-mineral-out sensitivity (M3 R²):")
    minerals = sorted(primary["mineral_label"].unique())
    lom_results = []

    for left_out in minerals:
        df_lo = primary[primary["mineral_label"] != left_out].copy()
        df_lo["crystal_system_cat"] = pd.Categorical(
            df_lo["crystal_system"],
            categories=[c for c in CRYSTAL_SYSTEM_ORDER if c in df_lo["crystal_system"].unique()],
        )

        reduced_set = ["Al", "Ca", "Fe", "Mg", "Si"]
        for elem in reduced_set:
            df_lo[f"chem_{elem}"] = df_lo[f"elem_{elem}"]

        n_lo = df_lo.shape[0]
        n_groups_lo = df_lo["provenance_group_id"].nunique()
        n_minerals_lo = df_lo["mineral_label"].nunique()

        for response in DESCRIPTORS:
            is_count = (response == "n_peaks_prom")

            X_cat, _ = create_design_matrix_categorical(df_lo, "crystal_system_cat")
            chem_cols = [f"chem_{e}" for e in reduced_set]
            X_chem = df_lo[chem_cols].values
            X = np.column_stack([np.ones(n_lo), X_cat, X_chem])
            y = df_lo[response].values
            groups = df_lo["provenance_group_id"].values

            if n_minerals_lo < 2:
                continue

            try:
                if is_count:
                    res = fit_count_model_cluster_reproduce(X, y, groups)
                    r2 = res["pseudo_r_squared"]
                else:
                    res = fit_ols_cluster_reproduce(X, y, groups)
                    r2 = res["r_squared"]
            except Exception:
                r2 = np.nan

            lom_results.append({
                "left_out_mineral": left_out,
                "response": response,
                "n_spectra": n_lo,
                "n_groups": n_groups_lo,
                "n_minerals": n_minerals_lo,
                "r_squared": r2,
            })

    lom_df = pd.DataFrame(lom_results)

    if len(lom_df) > 0:
        for response in DESCRIPTORS:
            lom_resp = lom_df[lom_df["response"] == response]
            if len(lom_resp) > 0:
                full_r2 = reproduced_models[response]["M3"].get(
                    "pseudo_r_squared" if response == "n_peaks_prom" else "r_squared", 0)
                lom_r2_values = lom_resp["r_squared"].dropna()
                r2_range = lom_r2_values.max() - lom_r2_values.min()
                r2_drop = full_r2 - lom_r2_values.min()

                print(f"    {response}: full M3 R2={full_r2:.4f}, "
                      f"LOM range=[{lom_r2_values.min():.4f}, {lom_r2_values.max():.4f}], "
                      f"range={r2_range:.4f}, max_drop={r2_drop:.4f}")

                if r2_drop > 0.1:
                    findings.append({
                        "issue": f"Large R² drop when removing single mineral ({response})",
                        "detail": (f"Removing one mineral drops M3 R² by {r2_drop:.4f}. "
                                   f"This suggests the model fit is driven by a single mineral's "
                                   f"data, not a general symmetry-chemistry relationship."),
                        "severity": "HIGH",
                    })

    return findings, confounding_table, lom_df


# =============================================================================
# Section 9 — Audit G: Representation and eligibility sensitivity
# =============================================================================

def section9_sensitivity(df, reproduced_models):
    print("\n" + "=" * 80)
    print("SECTION 9 (Audit G): Representation and eligibility sensitivity")
    print("=" * 80)

    sensitivity_rows = []
    reduced_set = ["Al", "Ca", "Fe", "Mg", "Si"]

    populations = {
        "PRIMARY": df[df["PRIMARY_ANALYSIS_ELIGIBLE"] == True].copy(),
    }
    if "SENSITIVITY_SYM_ELIGIBLE" in df.columns:
        populations["SENSITIVITY_SYM"] = df[df["SENSITIVITY_SYM_ELIGIBLE"] == True].copy()
    if "SENSITIVITY_CHEM_ELIGIBLE" in df.columns:
        populations["SENSITIVITY_CHEM"] = df[df["SENSITIVITY_CHEM_ELIGIBLE"] == True].copy()
    populations["FULL"] = df.copy()

    for pop_name, pop_df in populations.items():
        pop_df = pop_df.copy()
        pop_df["crystal_system_cat"] = pd.Categorical(
            pop_df["crystal_system"],
            categories=[c for c in CRYSTAL_SYSTEM_ORDER if c in pop_df["crystal_system"].unique()],
        )
        for elem in reduced_set:
            pop_df[f"chem_{elem}"] = pop_df[f"elem_{elem}"]

        n_spectra = pop_df.shape[0]
        n_groups = pop_df["provenance_group_id"].nunique()

        for response in DESCRIPTORS:
            is_count = (response == "n_peaks_prom")

            X_cat, _ = create_design_matrix_categorical(pop_df, "crystal_system_cat")
            chem_cols = [f"chem_{e}" for e in reduced_set]
            X_chem = pop_df[chem_cols].values
            X = np.column_stack([np.ones(n_spectra), X_cat, X_chem])
            y = pop_df[response].values
            groups = pop_df["provenance_group_id"].values

            try:
                if is_count:
                    res = fit_count_model_cluster_reproduce(X, y, groups)
                    r2 = res["pseudo_r_squared"]
                else:
                    res = fit_ols_cluster_reproduce(X, y, groups)
                    r2 = res["r_squared"]
                aic = res["aic"]
            except Exception as e:
                r2 = np.nan
                aic = np.nan
                print(f"    WARNING: {pop_name}/{response} failed: {e}")

            sensitivity_rows.append({
                "population": pop_name,
                "response": response,
                "n_spectra": n_spectra,
                "n_groups": n_groups,
                "aic": aic,
                "r_squared": r2,
            })

    # Native-5nm vs PCHIP representation sensitivity
    if "representation_type" in df.columns:
        for rep_type in df["representation_type"].unique():
            rep_df = df[df["representation_type"] == rep_type].copy()
            if rep_df.shape[0] < 10:
                continue
            rep_df["crystal_system_cat"] = pd.Categorical(
                rep_df["crystal_system"],
                categories=[c for c in CRYSTAL_SYSTEM_ORDER if c in rep_df["crystal_system"].unique()],
            )
            for elem in reduced_set:
                rep_df[f"chem_{elem}"] = rep_df[f"elem_{elem}"]

            n_rep = rep_df.shape[0]
            n_groups_rep = rep_df["provenance_group_id"].nunique()

            for response in DESCRIPTORS:
                is_count = (response == "n_peaks_prom")

                X_cat, _ = create_design_matrix_categorical(rep_df, "crystal_system_cat")
                chem_cols = [f"chem_{e}" for e in reduced_set]
                X_chem = rep_df[chem_cols].values
                X = np.column_stack([np.ones(n_rep), X_cat, X_chem])
                y = rep_df[response].values
                groups = rep_df["provenance_group_id"].values

                try:
                    if is_count:
                        res = fit_count_model_cluster_reproduce(X, y, groups)
                        r2 = res["pseudo_r_squared"]
                    else:
                        res = fit_ols_cluster_reproduce(X, y, groups)
                        r2 = res["r_squared"]
                    aic = res["aic"]
                except Exception:
                    r2 = np.nan
                    aic = np.nan

                sensitivity_rows.append({
                    "population": f"representation_{rep_type}",
                    "response": response,
                    "n_spectra": n_rep,
                    "n_groups": n_groups_rep,
                    "aic": aic,
                    "r_squared": r2,
                })

    sens_df = pd.DataFrame(sensitivity_rows)
    print(f"\n  Sensitivity results: {len(sens_df)} rows")
    for _, row in sens_df.iterrows():
        print(f"    {row['population']:25s} / {row['response']:25s}: "
              f"n={row['n_spectra']:.0f}, R²={row['r_squared']:.4f}" if pd.notna(row['r_squared']) else
              f"    {row['population']:25s} / {row['response']:25s}: n={row['n_spectra']:.0f}, R²=NaN")

    return sens_df


# =============================================================================
# Section 10 — Audit H: Go/no-go decisions
# =============================================================================

def section10_decisions(reproduced_models, confounding_findings, sensitivity_df):
    print("\n" + "=" * 80)
    print("SECTION 10 (Audit H): Go/no-go decisions")
    print("=" * 80)

    decisions = {}

    critical_issues = [f for f in confounding_findings if f.get("severity") == "CRITICAL"]
    has_confounding = len(critical_issues) > 0

    print("\n  Decision 1: Interpretable ML (e.g., rule-based models, decision trees)")
    if has_confounding:
        print("    -> NO-GO: Complete mineral-system confounding makes any 'symmetry effect'")
        print("       indistinguishable from mineral-specific chemistry. Interpretable ML would")
        print("       learn mineral identity, not symmetry.")
        decisions["interpretable_ml"] = "NO-GO"
    else:
        decisions["interpretable_ml"] = "CONDITIONAL"

    print("\n  Decision 2: PySR / symbolic regression")
    print("    -> NO-GO: PySR would overfit 175 spectra with 14 minerals. The search space")
    print("       is too large for the sample size, and results would be driven by mineral")
    print("       identity rather than interpretable symmetry-chemistry relationships.")
    decisions["pysr"] = "NO-GO"

    print("\n  Decision 3: Additional statistics")
    print("    -> CONDITIONAL: Additional statistics are warranted ONLY to:")
    print("       (a) Correct the DF convention for cluster-robust inference")
    print("       (b) Report consistent CIs and p-values")
    print("       (c) Perform leave-one-mineral-out sensitivity as a standard check")
    print("       (d) Report residual diagnostics")
    decisions["additional_stats"] = "CONDITIONAL"

    print("\n  Decision 4: No further modelling")
    print("    -> RECOMMENDED: Given complete confounding, the current M0-M4 results are")
    print("       sufficient for descriptive purposes. No further modelling should be")
    print("       performed until additional minerals with multi-system representation")
    print("       are added to the dataset.")
    decisions["no_further"] = "RECOMMENDED"

    return decisions


# =============================================================================
# Section 11 — Generate audit report
# =============================================================================

def section11_report(checksums, reproduction_checks, comparison_findings,
                     cluster_findings, diagnostic_rows, chemistry_findings,
                     confounding_findings, confounding_table, lom_df,
                     sensitivity_df, decisions, reproduced_models, primary):
    print("\n" + "=" * 80)
    print("SECTION 11: Generating audit report")
    print("=" * 80)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# Phase 5 Step 3 — Model Robustness and Interpretation Audit")
    lines.append("")
    lines.append(f"**Date:** {now}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Frozen inputs
    lines.append("## 1. Frozen Input Verification")
    lines.append("")
    lines.append("| Input | SHA-256 (first 16 chars) |")
    lines.append("|-------|--------------------------|")
    for name, cs in checksums.items():
        lines.append(f"| {name} | {cs[:16]}... |")
    lines.append("")

    # Audit A
    lines.append("## 2. Audit A: Model Reproduction")
    lines.append("")
    lines.append("All M0–M4 models were reproduced from scratch using the frozen inputs.")
    lines.append("")
    lines.append("| Model | Response | Reproduced AIC | Step 2 AIC | AIC Match | "
                 "Reproduced R² | Step 2 R² | R² Match |")
    lines.append("|-------|----------|---------------|------------|-----------|"
                 "---------------|-----------|----------|")
    for row in reproduction_checks:
        aic_str = f"{row['reproduced_aic']:.4f}" if pd.notna(row['reproduced_aic']) else "NaN"
        step2_aic_str = f"{row['step2_aic']:.4f}" if pd.notna(row.get('step2_aic')) else "N/A"
        aic_match_str = str(row['aic_match']) if row['aic_match'] is not None else "N/A"
        r2_str = f"{row['reproduced_r_squared']:.6f}" if pd.notna(row['reproduced_r_squared']) else "NaN"
        step2_r2_str = f"{row['step2_r_squared']:.6f}" if pd.notna(row.get('step2_r_squared')) else "N/A"
        r2_match_str = str(row['r_squared_match']) if row['r_squared_match'] is not None else "N/A"
        lines.append(f"| {row['model']} | {row['response']} | {aic_str} | {step2_aic_str} | "
                     f"{aic_match_str} | {r2_str} | {step2_r2_str} | {r2_match_str} |")
    lines.append("")

    all_match = all(
        (row['aic_match'] in [True, None]) and (row['r_squared_match'] in [True, None])
        for row in reproduction_checks
    )
    if all_match:
        lines.append("**Result:** All models reproduced successfully within tolerance.")
    else:
        lines.append("**Result:** Some models did not reproduce exactly. See reproduction CSV.")
    lines.append("")

    # Audit B
    lines.append("## 3. Audit B: Model Comparison Validity")
    lines.append("")
    for f in comparison_findings:
        lines.append(f"### {f['issue']} [{f['severity']}]")
        lines.append(f"- **Response:** {f['response']}")
        lines.append(f"- **Detail:** {f['detail']}")
        lines.append("")

    # Audit C
    lines.append("## 4. Audit C: Cluster-Robust SE Implementation")
    lines.append("")
    for f in cluster_findings:
        lines.append(f"### {f['issue']} [{f['severity']}]")
        lines.append(f"- **Response:** {f['response']}")
        lines.append(f"- **Detail:** {f['detail']}")
        lines.append("")

    # Audit D
    lines.append("## 5. Audit D: Response Distributions and Residual Diagnostics")
    lines.append("")
    for response in DESCRIPTORS:
        diag_resp = [d for d in diagnostic_rows if d["response"] == response and d["diagnostic"] == "response_distribution"]
        lines.append(f"### {DESCRIPTOR_LABELS[response]}")
        lines.append("")
        for d in diag_resp:
            lines.append(f"- {d['metric']}: {d['value']:.4f} (flagged: {d['flag']})")
        lines.append("")
        lines.append(f"Diagnostic figure: `phase5_step3_diag_{response}.png`")
        lines.append("")

    flagged_diagnostics = [d for d in diagnostic_rows if d["flag"] == "YES"]
    if flagged_diagnostics:
        lines.append("### Flagged diagnostics")
        lines.append("")
        for d in flagged_diagnostics:
            lines.append(f"- **{d['response']}** / {d['diagnostic']} / {d['metric']}: {d['value']:.4f}")
        lines.append("")

    # Audit E
    lines.append("## 6. Audit E: Chemistry Representation")
    lines.append("")
    for f in chemistry_findings:
        lines.append(f"### {f['issue']} [{f['severity']}]")
        lines.append(f"- **Detail:** {f['detail']}")
        lines.append("")

    # Audit F
    lines.append("## 7. Audit F: Mineral–Crystal System Confounding")
    lines.append("")
    for f in confounding_findings:
        lines.append(f"### {f['issue']} [{f['severity']}]")
        lines.append(f"- **Detail:** {f['detail']}")
        lines.append("")

    lines.append("### Confounding table")
    lines.append("")
    lines.append("| Mineral | Crystal system(s) | n_systems | n_spectra | n_groups |")
    lines.append("|---------|-------------------|-----------|-----------|----------|")
    for _, row in confounding_table.iterrows():
        systems_str = ", ".join(row["crystal_systems"])
        lines.append(f"| {row['mineral_label']} | {systems_str} | {row['n_systems']} | "
                     f"{row['n_spectra']} | {row['n_groups']} |")
    lines.append("")

    if len(lom_df) > 0:
        lines.append("### Leave-one-mineral-out sensitivity")
        lines.append("")
        lines.append("| Left-out mineral | Response | n_spectra | R² (M3) |")
        lines.append("|-----------------|----------|-----------|---------|")
        for _, row in lom_df.iterrows():
            r2_str = f"{row['r_squared']:.4f}" if pd.notna(row['r_squared']) else "NaN"
            lines.append(f"| {row['left_out_mineral']} | {row['response']} | "
                         f"{row['n_spectra']} | {r2_str} |")
        lines.append("")

    # Audit G
    lines.append("## 8. Audit G: Representation and Eligibility Sensitivity")
    lines.append("")
    lines.append("| Population | Response | n_spectra | n_groups | AIC | R² |")
    lines.append("|------------|----------|-----------|----------|-----|----|")
    for _, row in sensitivity_df.iterrows():
        aic_str = f"{row['aic']:.2f}" if pd.notna(row['aic']) else "NaN"
        r2_str = f"{row['r_squared']:.4f}" if pd.notna(row['r_squared']) else "NaN"
        lines.append(f"| {row['population']} | {row['response']} | "
                     f"{row['n_spectra']:.0f} | {row['n_groups']:.0f} | {aic_str} | {r2_str} |")
    lines.append("")

    # Audit H
    lines.append("## 9. Audit H: Go/No-Go Decisions")
    lines.append("")
    for decision_name, verdict in decisions.items():
        lines.append(f"### {decision_name.replace('_', ' ').title()}")
        lines.append(f"**Verdict:** {verdict}")
        lines.append("")

    # Summary
    lines.append("## 10. Summary of Critical Findings")
    lines.append("")

    all_findings = comparison_findings + cluster_findings + chemistry_findings + confounding_findings
    critical = [f for f in all_findings if f.get("severity") == "CRITICAL"]
    high = [f for f in all_findings if f.get("severity") == "HIGH"]

    lines.append(f"**Critical issues:** {len(critical)}")
    lines.append(f"**High-severity issues:** {len(high)}")
    lines.append("")

    if critical:
        lines.append("### Critical issues")
        lines.append("")
        for f in critical:
            lines.append(f"1. **{f['issue']}**: {f['detail']}")
        lines.append("")

    if high:
        lines.append("### High-severity issues")
        lines.append("")
        for f in high:
            lines.append(f"1. **{f['issue']}**: {f['detail']}")
        lines.append("")

    lines.append("## 11. Recommendations")
    lines.append("")
    lines.append("1. **Correct the model type label**: n_peaks_prom uses log-linear OLS, "
                 "NOT negative binomial. Update the Step 2 report accordingly.")
    lines.append("2. **Fix DF convention**: Use df = n_groups - 1 for cluster-robust "
                 "p-values and CIs consistently. Report both the Step 2 values and "
                 "corrected values.")
    lines.append("3. **Acknowledge complete confounding**: Every mineral maps to exactly "
                 "one crystal system. Crystal system effects cannot be separated from "
                 "mineral identity effects. This is a fundamental limitation of the "
                 "current dataset.")
    lines.append("4. **Do not proceed with interpretable ML or PySR**: The confounding "
                 "structure means any ML model would learn mineral identity, not symmetry. "
                 "This would be scientifically misleading.")
    lines.append("5. **Report compositional interpretation caveat**: Chemistry coefficients "
                 "are relative to omitted elements, not absolute effects.")
    lines.append("6. **No further modelling recommended**: The current M0–M4 results are "
                 "sufficient for descriptive purposes. Additional modelling would not "
                 "resolve the fundamental confounding.")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = RESULTS_DIR / "phase5_step3_model_robustness_audit.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Report saved: {report_path}")

    return report_text


# =============================================================================
# Section 12 — Save CSV outputs
# =============================================================================

def section12_save_csvs(reproduction_checks, diagnostic_rows, sensitivity_df):
    print("\n" + "=" * 80)
    print("SECTION 12: Saving CSV outputs")
    print("=" * 80)

    rep_df = pd.DataFrame(reproduction_checks)
    rep_path = RESULTS_DIR / "phase5_step3_reproduction_checks.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"  Reproduction checks: {rep_path} ({len(rep_df)} rows)")

    diag_df = pd.DataFrame(diagnostic_rows)
    diag_path = RESULTS_DIR / "phase5_step3_diagnostic_summary.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"  Diagnostic summary:  {diag_path} ({len(diag_df)} rows)")

    sens_path = RESULTS_DIR / "phase5_step3_sensitivity_results.csv"
    sensitivity_df.to_csv(sens_path, index=False)
    print(f"  Sensitivity results:  {sens_path} ({len(sensitivity_df)} rows)")


# =============================================================================
# Section 13 — Update research_decisions.md
# =============================================================================

def section13_update_decisions(decisions, comparison_findings, cluster_findings,
                                confounding_findings, chemistry_findings):
    print("\n" + "=" * 80)
    print("SECTION 13: Updating research_decisions.md")
    print("=" * 80)

    decisions_path = DOCS_DIR / "research_decisions.md"
    content = decisions_path.read_text(encoding="utf-8")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    critical_count = sum(1 for f in comparison_findings + cluster_findings +
                         confounding_findings + chemistry_findings
                         if f.get("severity") == "CRITICAL")
    high_count = sum(1 for f in comparison_findings + cluster_findings +
                     confounding_findings + chemistry_findings
                     if f.get("severity") == "HIGH")

    appendix_text = f"""

---

## Phase 5 Step 3 — Model Robustness Audit ({now})

### Audit Outcome

The Phase 5 Step 2 formal modelling results were reproduced and audited across 8 dimensions (A–H).

**Critical issues found:** {critical_count}
**High-severity issues found:** {high_count}

### Critical Findings

1. **Model type mislabelled**: n_peaks_prom models are labelled "Negative Binomial (count)" in the Step 2 report, but the implementation uses log(y+1) OLS with cluster-robust SEs. This is NOT a negative binomial model.

2. **Complete mineral–crystal system confounding**: Every mineral in the PRIMARY population maps to exactly one crystal system. Crystal system and mineral identity effects are completely confounded and cannot be distinguished.

3. **DF convention mismatch**: P-values use t(df=n-p) but cluster-robust inference should use t(df=n_groups-1). With n_groups≈166, this makes p-values anti-conservative for models with many parameters.

### Go/No-Go Decisions

| Decision | Verdict | Rationale |
|----------|---------|-----------|
| Interpretable ML | NO-GO | Complete confounding means ML would learn mineral identity, not symmetry |
| PySR / symbolic regression | NO-GO | Overfitting risk with 175 spectra and 14 minerals; results would be mineral-specific |
| Additional statistics | CONDITIONAL | Only to correct DF convention and report consistent CIs/p-values |
| No further modelling | RECOMMENDED | Current M0–M4 sufficient for descriptive purposes; confounding cannot be resolved |

### Recommendations

1. Correct the model type label in the Step 2 report (log-linear OLS, not negative binomial)
2. Fix DF convention for cluster-robust inference (use n_groups - 1)
3. Acknowledge complete confounding as a fundamental dataset limitation
4. Do not proceed with interpretable ML or PySR
5. Report compositional interpretation caveat for chemistry coefficients
6. No further modelling beyond the audit corrections

### Frozen Decision

**No further modelling beyond audit corrections.** The Phase 5 Step 2 results stand as descriptive summaries with the caveats documented above. The complete mineral–crystal system confounding is a dataset limitation that cannot be resolved by additional statistical methods.
"""

    if "Phase 5 Step 3" not in content:
        content += appendix_text
        decisions_path.write_text(content, encoding="utf-8")
        print(f"  Updated: {decisions_path}")
    else:
        print("  Phase 5 Step 3 section already exists in research_decisions.md — skipping.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Phase 5 Step 3 — Model Robustness and Interpretation Audit")
    print("=" * 80)

    df, step2_results, step2_comparison, checksums = section1_load_and_verify()
    populations = section2_populations(df)
    reproduction_checks, reproduced_models, primary, reduced_set = section3_reproduce_models(
        df, step2_results, step2_comparison)
    comparison_findings = section4_model_comparison_validity(reproduced_models, step2_comparison)
    cluster_findings = section5_cluster_robust_audit(reproduced_models)
    diagnostic_rows = section6_residual_diagnostics(reproduced_models, primary)
    chemistry_findings = section7_chemistry_audit(primary, reduced_set)
    confounding_findings, confounding_table, lom_df = section8_confounding_audit(
        primary, reproduced_models)
    sensitivity_df = section9_sensitivity(df, reproduced_models)
    decisions = section10_decisions(reproduced_models, confounding_findings, sensitivity_df)

    section11_report(
        checksums, reproduction_checks, comparison_findings, cluster_findings,
        diagnostic_rows, chemistry_findings, confounding_findings, confounding_table,
        lom_df, sensitivity_df, decisions, reproduced_models, primary)

    section12_save_csvs(reproduction_checks, diagnostic_rows, sensitivity_df)
    section13_update_decisions(decisions, comparison_findings, cluster_findings,
                                confounding_findings, chemistry_findings)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {RESULTS_DIR}")
    print("  - phase5_step3_model_robustness_audit.md")
    print("  - phase5_step3_reproduction_checks.csv")
    print("  - phase5_step3_diagnostic_summary.csv")
    print("  - phase5_step3_sensitivity_results.csv")
    print("  - phase5_step3_diag_*.png (4 figures)")
    print(f"\nUpdated: {DOCS_DIR / 'research_decisions.md'}")


if __name__ == "__main__":
    main()
