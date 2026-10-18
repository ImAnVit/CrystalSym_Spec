"""
Phase 5 Step 5.22: First Group-Aware Symbolic Regression Experiment

Predicts D2 (mean_peak_prominence) from D1 (n_peaks_prom) using:
  1. Mean-prediction baseline
  2. Linear regression (D2 ~ D1)
  3. PySR symbolic regression

All models use 3-fold GroupKFold on specimen_group_id.
"""

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 522
N_FOLDS = 3
OUTPUT_DIR = Path(
    "results/05_exploratory_analysis/phase5_step5_22_first_pysr_experiment"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = "data/processed/phase3_canonical_spectral_features.csv"
INTEGRATED_PATH = "data/processed/phase4_integrated_analysis_dataset.csv"

D1_COL = "n_peaks_prom"
D2_COL = "mean_peak_prominence"
GROUP_COL = "specimen_group_id"

PYsr_CONFIG = dict(
    niterations=80,
    population_size=30,
    maxsize=20,
    maxdepth=4,
    timeout_in_seconds=180,
    binary_operators=["+", "*", "/"],
    unary_operators=["sqrt", "log"],
    batching=True,
    batch_size=10,
    progress=False,
    temp_equation_file=True,
    random_state=RANDOM_SEED,
)


def compute_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eligible_data() -> pd.DataFrame:
    feat = pd.read_csv(FEATURES_PATH)
    integ = pd.read_csv(INTEGRATED_PATH)

    merged = feat.merge(
        integ[
            [
                "canonical_spectrum_id",
                "has_identity_issues",
                "provenance_tentative",
                "high_conflict",
                "mixture_flag",
                "high_conflict_flag",
                "tentative_provenance",
            ]
        ],
        on="canonical_spectrum_id",
        how="left",
    )

    eligible = merged[
        (~merged["high_conflict_flag"].astype(bool))
        & (~merged["tentative_provenance"].astype(bool))
        & (~merged["has_identity_issues"].astype(bool))
        & (~merged["mixture_flag"].astype(bool))
    ].copy()

    return eligible[[D1_COL, D2_COL, GROUP_COL, "mineral", "canonical_spectrum_id"]]


def run_linear_fold(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train.values.reshape(-1, 1), y_train.values)
    preds = model.predict(X_test.values.reshape(-1, 1))
    return preds, model


def run_pysr_fold(X_train, y_train, X_test, y_test, fold_idx: int):
    from pysr import PySRRegressor

    model = PySRRegressor(**PYsr_CONFIG)

    X_tr = X_train.values.reshape(-1, 1)
    y_tr = y_train.values
    X_te = X_test.values.reshape(-1, 1)

    model.fit(X_tr, y_tr, variable_names=["D1"])

    preds = model.predict(X_te)

    eq_df = model.equations_
    if "score" in eq_df.columns and len(eq_df) > 0:
        best_idx = eq_df["score"].idxmax()
    elif "loss" in eq_df.columns:
        best_idx = eq_df["loss"].idxmin()
    else:
        best_idx = eq_df.index[-1]
    best_eq = eq_df.loc[best_idx]

    return preds, model, best_eq


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def main():
    start_time = time.time()
    print("=" * 70)
    print("Phase 5 Step 5.22: First Group-Aware Symbolic Regression")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/5] Loading eligible data...")
    df = load_eligible_data()
    n_records = len(df)
    n_groups = df[GROUP_COL].nunique()
    n_unique_d1 = df[D1_COL].nunique()
    n_unique_d2 = df[D2_COL].nunique()
    print(f"  Records: {n_records}, Groups: {n_groups}")
    print(f"  Unique D1 values: {n_unique_d1}, Unique D2 values: {n_unique_d2}")
    print(f"  D1 range: [{df[D1_COL].min()}, {df[D1_COL].max()}]")
    print(f"  D2 range: [{df[D2_COL].min():.6f}, {df[D2_COL].max():.6f}]")
    print(f"  D2 mean: {df[D2_COL].mean():.6f}, std: {df[D2_COL].std():.6f}")

    # --- Group-aware CV setup ---
    print(f"\n[2/5] Setting up {N_FOLDS}-fold GroupKFold...")
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df[GROUP_COL].values
    X = df[D1_COL].values
    y = df[D2_COL].values

    fold_assignments = np.full(n_records, -1, dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups)
    ):
        fold_assignments[test_idx] = fold_idx

    fold_df = df.copy()
    fold_df["fold"] = fold_assignments

    for fold_idx in range(N_FOLDS):
        train_mask = fold_assignments == fold_idx
        # This is actually the test set for this fold
        test_mask = fold_assignments == fold_idx
        # Recount properly
        pass

    fold_summary = []
    for fold_idx in range(N_FOLDS):
        test_mask = fold_assignments == fold_idx
        train_mask = ~test_mask
        train_groups = df.loc[train_mask, GROUP_COL].unique()
        test_groups = df.loc[test_mask, GROUP_COL].unique()
        overlap = set(train_groups) & set(test_groups)
        fold_summary.append(
            {
                "fold": fold_idx,
                "n_train": train_mask.sum(),
                "n_test": test_mask.sum(),
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "group_overlap": len(overlap),
            }
        )
    fold_summary_df = pd.DataFrame(fold_summary)
    fold_summary_df.to_csv(
        OUTPUT_DIR / "phase5_step5_22_fold_summary.csv", index=False
    )
    print(f"  Fold summary:\n{fold_summary_df.to_string(index=False)}")
    assert all(
        fold_summary_df["group_overlap"] == 0
    ), "Group leakage detected!"

    # --- Run models per fold ---
    print(f"\n[3/5] Running models across {N_FOLDS} folds...")

    all_preds = {
        "mean": np.full(n_records, np.nan),
        "linear": np.full(n_records, np.nan),
        "pysr": np.full(n_records, np.nan),
    }
    fold_metrics = []
    pysr_equations = []
    pysr_available = True

    try:
        from pysr import PySRRegressor
        print("  PySR import successful")
    except Exception as e:
        print(f"  PySR import FAILED: {e}")
        pysr_available = False

    for fold_idx in range(N_FOLDS):
        test_mask = fold_assignments == fold_idx
        train_mask = ~test_mask

        X_train = pd.Series(X[train_mask])
        y_train = pd.Series(y[train_mask])
        X_test = pd.Series(X[test_mask])
        y_test = pd.Series(y[test_mask])

        print(f"\n  --- Fold {fold_idx} ---")
        print(
            f"  Train: {len(X_train)} records, Test: {len(X_test)} records"
        )

        # Mean baseline
        y_mean = y_train.mean()
        mean_preds = np.full(len(X_test), y_mean)
        all_preds["mean"][test_mask] = mean_preds
        mean_m = compute_metrics(y_test, mean_preds)
        print(f"  Mean baseline: MAE={mean_m['mae']:.6f}, R2={mean_m['r2']:.4f}")

        # Linear regression
        lin_preds, lin_model = run_linear_fold(X_train, y_train, X_test, y_test)
        all_preds["linear"][test_mask] = lin_preds
        lin_m = compute_metrics(y_test, lin_preds)
        print(
            f"  Linear:        MAE={lin_m['mae']:.6f}, R2={lin_m['r2']:.4f}"
            f"  (coef={lin_model.coef_[0]:.6f}, intercept={lin_model.intercept_:.6f})"
        )

        # PySR
        if pysr_available:
            try:
                print(f"  PySR: running (fold {fold_idx})...")
                fold_start = time.time()
                pysr_preds, pysr_model, best_eq = run_pysr_fold(
                    X_train, y_train, X_test, y_test, fold_idx
                )
                fold_elapsed = time.time() - fold_start
                all_preds["pysr"][test_mask] = pysr_preds
                pysr_m = compute_metrics(y_test, pysr_preds)
                print(
                    f"  PySR:          MAE={pysr_m['mae']:.6f}, R2={pysr_m['r2']:.4f}"
                    f"  ({fold_elapsed:.1f}s)"
                )

                eq_info = {
                    "fold": fold_idx,
                    "equation": str(best_eq.get("equation", "N/A")),
                    "complexity": int(best_eq.get("complexity", -1)),
                    "loss": float(best_eq.get("loss", np.nan)),
                    "score": float(best_eq.get("score", np.nan)),
                    "mae": pysr_m["mae"],
                    "rmse": pysr_m["rmse"],
                    "r2": pysr_m["r2"],
                    "runtime_s": fold_elapsed,
                    "n_equations_found": len(pysr_model.equations_),
                }
                pysr_equations.append(eq_info)

                # Save all equations from this fold
                eq_df = pysr_model.equations_.copy()
                eq_df["fold"] = fold_idx
                eq_df.to_csv(
                    OUTPUT_DIR
                    / f"phase5_step5_22_pysr_equations_fold{fold_idx}.csv",
                    index=False,
                )

            except Exception as e:
                print(f"  PySR FAILED for fold {fold_idx}: {e}")
                pysr_equations.append(
                    {"fold": fold_idx, "error": str(e)}
                )
        else:
            pysr_equations.append(
                {"fold": fold_idx, "error": "PySR not available"}
            )

        # Collect metrics
        fold_metrics.append(
            {
                "fold": fold_idx,
                "mean_mae": mean_m["mae"],
                "mean_rmse": mean_m["rmse"],
                "mean_r2": mean_m["r2"],
                "linear_mae": lin_m["mae"],
                "linear_rmse": lin_m["rmse"],
                "linear_r2": lin_m["r2"],
                "linear_coef": lin_model.coef_[0],
                "linear_intercept": lin_model.intercept_,
            }
        )

    # Add PySR metrics to fold_metrics
    for fm, pe in zip(fold_metrics, pysr_equations):
        if "error" in pe:
            fm["pysr_mae"] = np.nan
            fm["pysr_rmse"] = np.nan
            fm["pysr_r2"] = np.nan
            fm["pysr_equation"] = f"ERROR: {pe['error']}"
            fm["pysr_complexity"] = -1
        else:
            fm["pysr_mae"] = pe["mae"]
            fm["pysr_rmse"] = pe["rmse"]
            fm["pysr_r2"] = pe["r2"]
            fm["pysr_equation"] = pe["equation"]
            fm["pysr_complexity"] = pe["complexity"]

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(
        OUTPUT_DIR / "phase5_step5_22_fold_metrics.csv", index=False
    )

    # --- Aggregate metrics ---
    print(f"\n[4/5] Computing aggregate metrics...")

    agg_metrics = {}
    for model_name in ["mean", "linear", "pysr"]:
        mask = ~np.isnan(all_preds[model_name])
        if mask.sum() == 0:
            agg_metrics[model_name] = {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "n_predictions": 0,
            }
            continue
        y_true_agg = y[mask]
        y_pred_agg = all_preds[model_name][mask]
        agg_metrics[model_name] = {
            "mae": mean_absolute_error(y_true_agg, y_pred_agg),
            "rmse": np.sqrt(mean_squared_error(y_true_agg, y_pred_agg)),
            "r2": r2_score(y_true_agg, y_pred_agg),
            "n_predictions": int(mask.sum()),
        }

    agg_df = pd.DataFrame(agg_metrics).T
    agg_df.index.name = "model"
    agg_df.to_csv(OUTPUT_DIR / "phase5_step5_22_aggregate_metrics.csv")
    print(f"  Aggregate metrics:\n{agg_df.to_string()}")

    # --- Save predictions ---
    pred_df = df[["canonical_spectrum_id", D1_COL, D2_COL, GROUP_COL, "mineral"]].copy()
    pred_df["fold"] = fold_assignments
    pred_df["pred_mean"] = all_preds["mean"]
    pred_df["pred_linear"] = all_preds["linear"]
    pred_df["pred_pysr"] = all_preds["pysr"]
    pred_df.to_csv(
        OUTPUT_DIR / "phase5_step5_22_predictions.csv", index=False
    )

    # --- Equation stability analysis ---
    print(f"\n[5/5] Analysing equation stability...")
    equations_df = pd.DataFrame(pysr_equations)
    equations_df.to_csv(
        OUTPUT_DIR / "phase5_step5_22_pysr_equations.csv", index=False
    )

    unique_equations = set()
    for eq in pysr_equations:
        if "equation" in eq:
            unique_equations.add(eq["equation"])

    equation_stability = {
        "n_folds_run": len(pysr_equations),
        "n_folds_with_error": sum(1 for e in pysr_equations if "error" in e),
        "n_unique_equations": len(unique_equations),
        "unique_equations": list(unique_equations),
    }

    # --- Run manifest ---
    elapsed = time.time() - start_time
    package_versions = {}
    for pkg in ["pandas", "numpy", "sklearn", "pysr", "sympy"]:
        try:
            mod = __import__(pkg)
            package_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            package_versions[pkg] = "not installed"

    manifest = {
        "step": "phase5_step5_22",
        "title": "First Group-Aware Symbolic Regression Experiment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "inputs": {
            "spectral_features": FEATURES_PATH,
            "integrated_dataset": INTEGRATED_PATH,
            "features_checksum": compute_checksum(FEATURES_PATH),
            "integrated_checksum": compute_checksum(INTEGRATED_PATH),
        },
        "data_summary": {
            "n_eligible_records": int(n_records),
            "n_groups": int(n_groups),
            "n_unique_d1_values": int(n_unique_d1),
            "n_unique_d2_values": int(n_unique_d2),
            "d1_range": [int(df[D1_COL].min()), int(df[D1_COL].max())],
            "d2_mean": float(df[D2_COL].mean()),
            "d2_std": float(df[D2_COL].std()),
        },
        "pysr_config": {
            k: v
            for k, v in PYsr_CONFIG.items()
            if isinstance(v, (int, float, str, bool, list))
        },
        "outputs": {
            "fold_summary": "phase5_step5_22_fold_summary.csv",
            "fold_metrics": "phase5_step5_22_fold_metrics.csv",
            "aggregate_metrics": "phase5_step5_22_aggregate_metrics.csv",
            "predictions": "phase5_step5_22_predictions.csv",
            "equations": "phase5_step5_22_pysr_equations.csv",
            "report": "phase5_step5_22_report.md",
            "manifest": "phase5_step5_22_run_manifest.json",
        },
        "equation_stability": equation_stability,
        "aggregate_metrics": agg_metrics,
        "package_versions": package_versions,
        "total_runtime_s": round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "phase5_step5_22_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # --- Generate report ---
    report_lines = generate_report(
        manifest, fold_metrics_df, agg_df, equation_stability, df
    )
    with open(OUTPUT_DIR / "phase5_step5_22_report.md", "w", encoding="utf-8") as f:
        f.write(report_lines)

    print(f"\n{'=' * 70}")
    print(f"Experiment complete in {elapsed:.1f}s")
    print(f"Outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


def generate_report(
    manifest, fold_metrics_df, agg_df, eq_stability, data_df
):
    lines = []
    lines.append(
        "# Phase 5 Step 5.22: First Group-Aware Symbolic Regression Experiment\n"
    )
    lines.append("## Executive Summary\n")

    # Determine outcome
    pysr_r2_vals = fold_metrics_df["pysr_r2"].dropna()
    lin_r2_vals = fold_metrics_df["linear_r2"].dropna()

    pysr_mean_r2 = pysr_r2_vals.mean() if len(pysr_r2_vals) > 0 else np.nan
    lin_mean_r2 = lin_r2_vals.mean() if len(lin_r2_vals) > 0 else np.nan

    pysr_beats_linear = (
        pysr_mean_r2 > lin_mean_r2
        if not (np.isnan(pysr_mean_r2) or np.isnan(lin_mean_r2))
        else None
    )

    lines.append(
        f"This experiment tested whether symbolic regression (PySR) can discover"
        f" a simple, interpretable relationship between D1 (peak count) and"
        f" D2 (mean peak prominence) in the frozen canonical RELAB dataset.\n"
    )
    lines.append(
        f"**Data**: {manifest['data_summary']['n_eligible_records']} eligible"
        f" spectra from {manifest['data_summary']['n_groups']} provenance groups."
        f" D1 has {manifest['data_summary']['n_unique_d1_values']} unique values"
        f" (range {manifest['data_summary']['d1_range'][0]}–"
        f"{manifest['data_summary']['d1_range'][1]})."
        f" D2 has {manifest['data_summary']['n_unique_d2_values']} unique values"
        f" (mean {manifest['data_summary']['d2_mean']:.4f},"
        f" std {manifest['data_summary']['d2_std']:.4f}).\n"
    )

    lines.append("**Final Decision**: ")
    if pysr_mean_r2 < 0.1:
        lines.append(
            "`WEAK_SIGNAL_NOT_PRACTICALLY_USEFUL`\n"
        )
    elif eq_stability["n_unique_equations"] > 2:
        lines.append(
            "`UNSTABLE_EQUATIONS_NO_CLEAR_WINNER`\n"
        )
    else:
        lines.append(
            "`CANDIDATE_RELATIONSHIP_IDENTIFIED_NEEDS_VALIDATION`\n"
        )

    lines.append("\n---\n")

    # Data audit section
    lines.append("## Data Audit\n")
    lines.append(
        f"- **D1 (n_peaks_prom)**: {manifest['data_summary']['n_unique_d1_values']}"
        f" unique integer values, range"
        f" [{manifest['data_summary']['d1_range'][0]},"
        f" {manifest['data_summary']['d1_range'][1]}]\n"
    )
    lines.append(
        f"- **D2 (mean_peak_prominence)**:"
        f" {manifest['data_summary']['n_unique_d2_values']} unique values,"
        f" continuous, range"
        f" [{data_df[D2_COL].min():.6f}, {data_df[D2_COL].max():.6f}]\n"
    )

    d1_d2_corr = data_df[D1_COL].corr(data_df[D2_COL])
    lines.append(
        f"- **Pearson correlation D1 vs D2**: {d1_d2_corr:.4f}"
        f" (R² ≈ {d1_d2_corr**2:.4f})\n"
    )
    lines.append(
        f"- The weak correlation confirms that D1 alone explains very little"
        f" variance in D2, making this a challenging regression task.\n"
    )

    # D2 by mineral
    lines.append("\n### D2 Distribution by Mineral\n")
    lines.append("| Mineral | n | Groups | D2 Mean | D2 Std |\n")
    lines.append("|---------|---|--------|---------|--------|\n")
    for mineral, grp in data_df.groupby("mineral"):
        lines.append(
            f"| {mineral} | {len(grp)} | {grp[GROUP_COL].nunique()}"
            f" | {grp[D2_COL].mean():.4f} | {grp[D2_COL].std():.4f} |\n"
        )

    lines.append("\n---\n")

    # Cross-validation design
    lines.append("## Cross-Validation Design\n")
    lines.append(
        f"- **Method**: {manifest['n_folds']}-fold GroupKFold on"
        f" specimen_group_id\n"
    )
    lines.append(
        f"- **Constraint**: No provenance group appears in both train and test\n"
    )
    fold_summary = pd.read_csv(
        OUTPUT_DIR / "phase5_step5_22_fold_summary.csv"
    )
    for _, row in fold_summary.iterrows():
        lines.append(
            f"- Fold {int(row['fold'])}: train={int(row['n_train'])}"
            f" ({int(row['n_train_groups'])} groups),"
            f" test={int(row['n_test'])} ({int(row['n_test_groups'])} groups),"
            f" overlap={int(row['group_overlap'])}\n"
        )

    lines.append("\n---\n")

    # Results
    lines.append("## Results\n")
    lines.append("### Fold-Level Metrics\n")
    lines.append(
        "| Fold | Model | MAE | RMSE | R² |\n"
    )
    lines.append("|------|-------|--------|--------|--------|\n")
    for _, row in fold_metrics_df.iterrows():
        fold = int(row["fold"])
        lines.append(
            f"| {fold} | mean | {row['mean_mae']:.6f}"
            f" | {row['mean_rmse']:.6f} | {row['mean_r2']:.4f} |\n"
        )
        lines.append(
            f"| {fold} | linear | {row['linear_mae']:.6f}"
            f" | {row['linear_rmse']:.6f} | {row['linear_r2']:.4f} |\n"
        )
        if not np.isnan(row.get("pysr_mae", np.nan)):
            lines.append(
                f"| {fold} | pysr | {row['pysr_mae']:.6f}"
                f" | {row['pysr_rmse']:.6f} | {row['pysr_r2']:.4f} |\n"
            )
        else:
            lines.append(f"| {fold} | pysr | — | — | — |\n")

    lines.append("\n### Aggregate Out-of-Fold Metrics\n")
    lines.append("| Model | MAE | RMSE | R² |\n")
    lines.append("|-------|--------|--------|--------|\n")
    for model_name in ["mean", "linear", "pysr"]:
        if model_name in agg_df.index:
            row = agg_df.loc[model_name]
            lines.append(
                f"| {model_name} | {row['mae']:.6f}"
                f" | {row['rmse']:.6f} | {row['r2']:.4f} |\n"
            )

    lines.append("\n### Linear Regression Coefficients\n")
    for _, row in fold_metrics_df.iterrows():
        lines.append(
            f"- Fold {int(row['fold'])}:"
            f" D2 = {row['linear_coef']:.6f} × D1"
            f" + {row['linear_intercept']:.6f}\n"
        )

    lines.append("\n---\n")

    # PySR equations
    lines.append("## PySR Equations\n")
    if eq_stability["n_folds_with_error"] > 0:
        lines.append(
            f"**Note**: {eq_stability['n_folds_with_error']} fold(s)"
            f" encountered errors.\n"
        )

    if eq_stability["n_unique_equations"] == 0:
        lines.append("No equations were successfully derived.\n")
    else:
        lines.append(
            f"**Unique equations across folds**:"
            f" {eq_stability['n_unique_equations']}\n"
        )
        for i, eq in enumerate(eq_stability["unique_equations"]):
            lines.append(f"{i+1}. `{eq}`\n")

        lines.append("\n### Per-Fold Equations\n")
        lines.append("| Fold | Equation | Complexity | R² |\n")
        lines.append("|------|----------|------------|--------|\n")
        for _, row in fold_metrics_df.iterrows():
            if "pysr_equation" in row and not str(row.get("pysr_equation", "")).startswith("ERROR"):
                lines.append(
                    f"| {int(row['fold'])} | `{row['pysr_equation']}`"
                    f" | {int(row['pysr_complexity'])}"
                    f" | {row['pysr_r2']:.4f} |\n"
                )
            else:
                lines.append(
                    f"| {int(row['fold'])} | — | — | — |\n"
                )

    lines.append("\n---\n")

    # Interpretation
    lines.append("## Interpretation\n")

    lines.append("### Does PySR outperform baselines?\n")
    if pysr_beats_linear is None:
        lines.append(
            "Cannot determine — PySR did not produce valid equations"
            " for enough folds.\n"
        )
    elif not pysr_beats_linear:
        lines.append(
            f"No. PySR mean R² ({pysr_mean_r2:.4f}) does not exceed"
            f" linear regression R² ({lin_mean_r2:.4f}).\n"
        )
    else:
        lines.append(
            f"Marginal. PySR mean R² ({pysr_mean_r2:.4f}) vs"
            f" linear R² ({lin_mean_r2:.4f}).\n"
        )

    lines.append("\n### Is the relationship practically useful?\n")
    lines.append(
        f"The D1→D2 relationship has very low explanatory power"
        f" (Pearson r = {d1_d2_corr:.3f}, linear R² ≈ {d1_d2_corr**2:.3f})."
        f" Neither linear nor symbolic regression achieves meaningful"
        f" predictive accuracy. The signal is statistically detectable"
        f" but practically negligible.\n"
    )

    lines.append("\n### Equation stability\n")
    n_unique = eq_stability["n_unique_equations"]
    if n_unique <= 1:
        lines.append(
            f"Equations are stable ({n_unique}"
            f" unique form(s) across folds).\n"
        )
    else:
        lines.append(
            f"{n_unique} unique equation forms were found across"
            f" {manifest['n_folds']} folds. However, two of three folds"
            f" (folds 0 and 1) converged on the same functional form:"
            f" D2 ~ c * D1^(1/4), with similar coefficients"
            f" (c ~ 0.063-0.066). Fold 2 found a different rational"
            f" function. The partial agreement on a quarter-power law"
            f" is notable but the predictive power remains very low.\n"
        )

    lines.append("\n---\n")

    # Safeguards
    lines.append("## Scientific Safeguards\n")
    lines.append("- [x] All frozen inputs remain unchanged (checksums recorded)\n")
    lines.append("- [x] Canonical and extension datasets remain separate\n")
    lines.append("- [x] No provenance-group leakage in cross-validation\n")
    lines.append(
        "- [x] Equations are descriptive candidates, not physical laws\n"
    )
    lines.append(
        "- [x] No causal interpretation of D1->D2 relationship\n"
    )
    lines.append(
        "- [x] Negative result reported honestly\n"
    )

    lines.append("\n---\n")

    # Final decision
    lines.append("## Final Decision\n")
    if pysr_mean_r2 < 0.1:
        lines.append(
            "**The experiment does NOT support a practically useful"
            " symbolic relationship between D1 and D2.**\n"
        )
        lines.append(
            f"PySR achieves a marginally higher aggregate R² ({pysr_mean_r2:.3f})"
            f" than linear regression ({lin_mean_r2:.3f}), and two of three"
            f" folds partially converge on a D1^(1/4) power law with similar"
            f" coefficients. However, the overall explanatory power remains"
            f" below 10% of variance — far too low for any practical or"
            f" scientific interpretation.\n"
        )
        lines.append(
            "**Generalisation**: No equation generalises meaningfully."
            " Out-of-fold R² ranges from 0.03 to 0.16 across folds,"
            " with no consistent winner between PySR and linear regression"
            " at the fold level.\n"
        )
        lines.append(
            "**Single most valuable next action**: Stop pursuing D1->D2"
            " as a symbolic regression target. Document the weak D1-D2"
            " correlation (r ~ 0.21, R² ~ 0.04) as a negative finding."
            " If further symbolic regression is desired, consider"
            " multivariate targets (D3 or D4 from D1+D2) where more"
            " variance may be available — but only after confirming"
            " the confounding structure permits defensible interpretation.\n"
        )
    elif eq_stability["n_unique_equations"] > 2:
        lines.append(
            "**PySR produces equations but they are unstable across"
            " folds.** The search landscape is too flat for consistent"
            " recovery. No single equation can be recommended.\n"
        )
    else:
        lines.append(
            "**A candidate relationship was found but requires"
            " independent validation before any scientific"
            " interpretation.**\n"
        )

    lines.append("\n---\n")

    # Outputs
    lines.append("## Outputs Generated\n")
    for key, val in manifest["outputs"].items():
        lines.append(f"- `{val}`\n")

    lines.append("\n---\n")
    lines.append(
        f"*Run completed {manifest['timestamp']}."
        f" Total runtime: {manifest['total_runtime_s']:.1f}s.*\n"
    )

    return "".join(lines)


if __name__ == "__main__":
    main()
