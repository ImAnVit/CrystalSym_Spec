"""
Phase 5 Step 5.23: Multivariate Descriptor Prediction — Feasibility and Target Selection

Assesses whether D3 (first-derivative RMS) or D4 (second-derivative RMS) can be
predicted from D1 (peak count) and D2 (mean peak prominence), using group-aware
cross-validation. Runs PySR only if evidence justifies it.

Inputs (frozen):
  - data/processed/phase3_canonical_spectral_features.csv
  - data/processed/phase4_integrated_analysis_dataset.csv

Outputs → results/05_exploratory_analysis/phase5_step5_23_multivariate_descriptor_pysr/
"""

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

OUT_DIR = Path("results/05_exploratory_analysis/phase5_step5_23_multivariate_descriptor_pysr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 523
N_FOLDS = 3
RNG = np.random.default_rng(SEED)

# ── 1. Load and reconcile population ──────────────────────────────────────────

features = pd.read_csv("data/processed/phase3_canonical_spectral_features.csv")
integrated = pd.read_csv("data/processed/phase4_integrated_analysis_dataset.csv")

feat_checksum = hashlib.sha256(
    Path("data/processed/phase3_canonical_spectral_features.csv").read_bytes()
).hexdigest()
integ_checksum = hashlib.sha256(
    Path("data/processed/phase4_integrated_analysis_dataset.csv").read_bytes()
).hexdigest()

print(f"Phase 3 features: {len(features)} rows, checksum={feat_checksum[:16]}...")
print(f"Phase 4 integrated: {len(integrated)} rows, checksum={integ_checksum[:16]}...")

# Merge to get eligibility flags
df = features.merge(
    integrated[[
        "canonical_spectrum_id", "high_conflict_flag", "tentative_provenance",
        "has_identity_issues", "mixture_flag", "provenance_group_id"
    ]],
    on="canonical_spectrum_id",
    how="left",
)

# Apply same ML eligibility filter as Step 5.22
excluded = df["high_conflict_flag"] | df["tentative_provenance"] | df["has_identity_issues"] | df["mixture_flag"]
eligible = df[~excluded].copy().reset_index(drop=True)

print(f"\nEligible population: {len(eligible)} spectra, {eligible['specimen_group_id'].nunique()} groups")

# Verify checksums match Step 5.22
assert feat_checksum == "a761544a2c959db83ce5ee5acc7c537ad876283c733ee08ace3f081e2af6efda"
assert integ_checksum == "ed055581c1dac9eefa816ac8acb3ee2d8618ec0ea8a8b37de8f1d87aacab2382"

# ── 2. Descriptor statistics ──────────────────────────────────────────────────

D1 = eligible["n_peaks_prom"]
D2 = eligible["mean_peak_prominence"]
D3 = eligible["first_deriv_rms"]
D4 = eligible["second_deriv_rms"]

print("\n" + "=" * 70)
print("DESCRIPTOR DISTRIBUTIONS")
print("=" * 70)

desc_stats = pd.DataFrame({
    "descriptor": ["D1 (n_peaks_prom)", "D2 (mean_peak_prominence)",
                   "D3 (first_deriv_rms)", "D4 (second_deriv_rms)"],
    "n": [D1.count(), D2.count(), D3.count(), D4.count()],
    "n_missing": [D1.isna().sum(), D2.isna().sum(), D3.isna().sum(), D4.isna().sum()],
    "unique": [D1.nunique(), D2.nunique(), D3.nunique(), D4.nunique()],
    "mean": [D1.mean(), D2.mean(), D3.mean(), D4.mean()],
    "std": [D1.std(), D2.std(), D3.std(), D4.std()],
    "min": [D1.min(), D2.min(), D3.min(), D4.min()],
    "median": [D1.median(), D2.median(), D3.median(), D4.median()],
    "max": [D1.max(), D2.max(), D3.max(), D4.max()],
})
print(desc_stats.to_string(index=False))

# ── 3. Correlation structure ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("CORRELATION MATRIX (Pearson)")
print("=" * 70)

desc_cols = ["n_peaks_prom", "mean_peak_prominence", "first_deriv_rms", "second_deriv_rms"]
desc_labels = ["D1", "D2", "D3", "D4"]

corr_pearson = eligible[desc_cols].corr(method="pearson")
corr_spearman = eligible[desc_cols].corr(method="spearman")

corr_pearson.index = desc_labels
corr_pearson.columns = desc_labels
corr_spearman.index = desc_labels
corr_spearman.columns = desc_labels

print(corr_pearson.round(4).to_string())
print("\nCORRELATION MATRIX (Spearman)")
print(corr_spearman.round(4).to_string())

# ── 4. Target-specific assessment ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("TARGET FEASIBILITY ASSESSMENT")
print("=" * 70)

# D3 from D1+D2
r_d3_d1 = eligible["n_peaks_prom"].corr(eligible["first_deriv_rms"])
r_d3_d2 = eligible["mean_peak_prominence"].corr(eligible["first_deriv_rms"])
print(f"\nD3 (first_deriv_rms):")
print(f"  Pearson r(D1, D3) = {r_d3_d1:.4f}")
print(f"  Pearson r(D2, D3) = {r_d3_d2:.4f}")
print(f"  D3 unique values: {D3.nunique()}")
print(f"  D3 missing: {D3.isna().sum()}")

# D4 from D1+D2
r_d4_d1 = eligible["n_peaks_prom"].corr(eligible["second_deriv_rms"])
r_d4_d2 = eligible["mean_peak_prominence"].corr(eligible["second_deriv_rms"])
print(f"\nD4 (second_deriv_rms):")
print(f"  Pearson r(D1, D4) = {r_d4_d1:.4f}")
print(f"  Pearson r(D2, D4) = {r_d4_d2:.4f}")
print(f"  D4 unique values: {D4.nunique()}")
print(f"  D4 missing: {D4.isna().sum()}")

# ── 5. Group-level variation analysis ─────────────────────────────────────────

print("\n" + "=" * 70)
print("GROUP-LEVEL VARIATION")
print("=" * 70)

group_stats = eligible.groupby("specimen_group_id").agg(
    n_spectra=("canonical_spectrum_id", "count"),
    D3_mean=("first_deriv_rms", "mean"),
    D3_std=("first_deriv_rms", "std"),
    D4_mean=("second_deriv_rms", "mean"),
    D4_std=("second_deriv_rms", "std"),
).reset_index()

# Most groups have only 1 spectrum, so within-group std is NaN
multi_spectrum_groups = group_stats[group_stats["n_spectra"] > 1]
print(f"Total groups: {len(group_stats)}")
print(f"Groups with >1 spectrum: {len(multi_spectrum_groups)}")

# Between-group variation (using group means)
print(f"\nBetween-group variation (group means):")
print(f"  D3 mean-of-means: {group_stats['D3_mean'].mean():.6f}, "
      f"std-of-means: {group_stats['D3_mean'].std():.6f}")
print(f"  D4 mean-of-means: {group_stats['D4_mean'].mean():.6f}, "
      f"std-of-means: {group_stats['D4_mean'].std():.6f}")

# Intraclass correlation: proportion of total variance between groups
# Use proper one-way random effects ICC(1,1)
overall_d3_var = eligible["first_deriv_rms"].var()
overall_d4_var = eligible["second_deriv_rms"].var()

def compute_icc(df, value_col, group_col):
    """Compute ICC(1,1) using one-way random effects model."""
    grouped = df.groupby(group_col)[value_col]
    k = grouped.size()
    n_groups = len(k)
    n_total = k.sum()
    grand_mean = df[value_col].mean()
    group_means = grouped.mean()

    msb = (grouped.size() * (group_means - grand_mean) ** 2).sum() / (n_groups - 1)
    msw = sum(
        ((df.loc[grouped.groups[g], value_col] - group_means[g]) ** 2).sum()
        for g in grouped.groups
    ) / (n_total - n_groups)

    k_bar = n_total / n_groups
    if msb == 0:
        return 0.0
    icc = (msb - msw) / (msb + (k_bar - 1) * msw)
    return max(0.0, icc)

d3_icc = compute_icc(eligible, "first_deriv_rms", "specimen_group_id")
d4_icc = compute_icc(eligible, "second_deriv_rms", "specimen_group_id")

print(f"\nVariance decomposition:")
print(f"  D3: total_var={overall_d3_var:.8f}, ICC(1,1)={d3_icc:.3f}")
print(f"  D4: total_var={overall_d4_var:.12f}, ICC(1,1)={d4_icc:.3f}")

# ── 6. Mineral-level descriptor summary ───────────────────────────────────────

mineral_desc = eligible.groupby("mineral").agg(
    n=("canonical_spectrum_id", "count"),
    n_groups=("specimen_group_id", "nunique"),
    D3_mean=("first_deriv_rms", "mean"),
    D3_std=("first_deriv_rms", "std"),
    D4_mean=("second_deriv_rms", "mean"),
    D4_std=("second_deriv_rms", "std"),
).sort_values("n", ascending=False)

print("\n" + "=" * 70)
print("DESCRIPTOR SUMMARY BY MINERAL")
print("=" * 70)
print(mineral_desc.round(6).to_string())

# ── 7. Save intermediate artifacts ────────────────────────────────────────────

desc_stats.to_csv(OUT_DIR / "phase5_step5_23_descriptor_statistics.csv", index=False)
corr_pearson.to_csv(OUT_DIR / "phase5_step5_23_pearson_correlation.csv")
corr_spearman.to_csv(OUT_DIR / "phase5_step5_23_spearman_correlation.csv")
mineral_desc.round(6).to_csv(OUT_DIR / "phase5_step5_23_mineral_descriptor_summary.csv")
group_stats.to_csv(OUT_DIR / "phase5_step5_23_group_statistics.csv", index=False)

# ── 8. Decide which targets to model ─────────────────────────────────────────

print("\n" + "=" * 70)
print("TARGET SELECTION DECISION")
print("=" * 70)

# Decision criteria:
# 1. No missing values in target or predictors
# 2. Sufficient correlation (|r| > 0.15 for at least one predictor)
# 3. Sufficient group-level variation (ICC > 0.1 means grouping matters)
# 4. Enough data points (n >= 100)

d3_missing = D3.isna().sum() + D1.isna().sum() + D2.isna().sum()
d4_missing = D4.isna().sum() + D1.isna().sum() + D2.isna().sum()

d3_max_r = max(abs(r_d3_d1), abs(r_d3_d2))
d4_max_r = max(abs(r_d4_d1), abs(r_d4_d2))

print(f"\nD3 from D1+D2:")
print(f"  Missing values: {d3_missing}")
print(f"  Max |r| with predictors: {d3_max_r:.4f}")
print(f"  ICC (group-level variation): {d3_icc:.3f}")
print(f"  n: {len(eligible)}")

print(f"\nD4 from D1+D2:")
print(f"  Missing values: {d4_missing}")
print(f"  Max |r| with predictors: {d4_max_r:.4f}")
print(f"  ICC (group-level variation): {d4_icc:.3f}")
print(f"  n: {len(eligible)}")
D3_FEASIBLE = d3_missing == 0 and d3_max_r > 0.15
D4_FEASIBLE = d4_missing == 0 and d4_max_r > 0.15

print(f"\nD3 feasible for modelling: {D3_FEASIBLE}")
print(f"D4 feasible for modelling: {D4_FEASIBLE}")

# ── 9. Group-aware predictive comparison ──────────────────────────────────────

def run_predictive_comparison(eligible_df, target_col, predictor_cols, group_col,
                               model_name_prefix, seed=523):
    """Run mean baseline, Ridge, and PySR with GroupKFold."""
    from pysr import PySRRegressor

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = eligible_df[group_col].values
    X = eligible_df[predictor_cols].values
    y = eligible_df[target_col].values

    fold_results = []
    pred_records = []
    all_equations = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        train_groups_set = set(groups_train)
        test_groups_set = set(groups_test)
        overlap = train_groups_set & test_groups_set
        assert len(overlap) == 0, f"Fold {fold_idx}: group overlap detected!"

        fold_result = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_groups": len(train_groups_set),
            "n_test_groups": len(test_groups_set),
            "group_overlap": len(overlap),
        }

        ss_tot = np.sum((y_test - y_test.mean()) ** 2)

        # ── Mean baseline ──
        y_mean_pred = np.full(len(test_idx), y_train.mean())
        mean_mae = np.mean(np.abs(y_test - y_mean_pred))
        mean_rmse = np.sqrt(np.mean((y_test - y_mean_pred) ** 2))
        ss_res = np.sum((y_test - y_mean_pred) ** 2)
        mean_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fold_result.update({"mean_mae": mean_mae, "mean_rmse": mean_rmse, "mean_r2": mean_r2})

        # ── Ridge regression ──
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_sc, y_train)
        y_ridge_pred = ridge.predict(X_test_sc)
        ridge_mae = np.mean(np.abs(y_test - y_ridge_pred))
        ridge_rmse = np.sqrt(np.mean((y_test - y_ridge_pred) ** 2))
        ss_res = np.sum((y_test - y_ridge_pred) ** 2)
        ridge_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fold_result.update({
            "ridge_mae": ridge_mae, "ridge_rmse": ridge_rmse, "ridge_r2": ridge_r2,
            "ridge_coefs": str(ridge.coef_.tolist()),
            "ridge_intercept": ridge.intercept_,
        })

        # ── PySR ──
        X_train_df = pd.DataFrame(X_train, columns=predictor_cols)
        X_test_df = pd.DataFrame(X_test, columns=predictor_cols)
        pysr = PySRRegressor(
            niterations=80, population_size=30, maxsize=20, maxdepth=4,
            timeout_in_seconds=180, binary_operators=["+", "*", "/"],
            unary_operators=["sqrt", "log"], batching=True, batch_size=10,
            progress=False, temp_equation_file=True, random_state=seed + fold_idx,
        )

        pysr_r2 = np.nan
        pysr_equation = "FAILED"
        pysr_complexity = -1
        y_pysr_pred = np.full(len(test_idx), np.nan)

        try:
            pysr.fit(X_train_df, y_train)
            y_pysr_pred = pysr.predict(X_test_df)
            pysr_mae = np.mean(np.abs(y_test - y_pysr_pred))
            pysr_rmse = np.sqrt(np.mean((y_test - y_pysr_pred) ** 2))
            ss_res = np.sum((y_test - y_pysr_pred) ** 2)
            pysr_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            eq_df = pysr.equations_
            best_eq = eq_df.iloc[-1]
            pysr_equation = str(best_eq["equation"])
            pysr_complexity = int(best_eq["complexity"])

            for _, eq_row in eq_df.iterrows():
                all_equations.append({
                    "fold": fold_idx,
                    "equation": str(eq_row["equation"]),
                    "complexity": int(eq_row["complexity"]),
                    "loss": float(eq_row["loss"]),
                    "score": float(eq_row["score"]) if "score" in eq_row.index else np.nan,
                })
        except Exception as e:
            print(f"  [WARNING] PySR fold {fold_idx} failed: {e}")

        fold_result.update({
            "pysr_mae": float(np.mean(np.abs(y_test - y_pysr_pred[~np.isnan(y_pysr_pred)])))
            if (~np.isnan(y_pysr_pred)).any() else np.nan,
            "pysr_rmse": float(np.sqrt(np.mean((y_test[~np.isnan(y_pysr_pred)] - y_pysr_pred[~np.isnan(y_pysr_pred)]) ** 2)))
            if (~np.isnan(y_pysr_pred)).any() else np.nan,
            "pysr_r2": pysr_r2,
            "pysr_equation": pysr_equation,
            "pysr_complexity": pysr_complexity,
        })

        # Store per-record predictions (all models at once)
        for j, idx in enumerate(test_idx):
            pred_records.append({
                "canonical_spectrum_id": eligible_df.iloc[idx]["canonical_spectrum_id"],
                "mineral": eligible_df.iloc[idx]["mineral"],
                "target_actual": float(y[idx]),
                "fold": fold_idx,
                "pred_mean": float(y_mean_pred[j]),
                "pred_ridge": float(y_ridge_pred[j]),
                "pred_pysr": float(y_pysr_pred[j]) if not np.isnan(y_pysr_pred[j]) else np.nan,
            })

        fold_results.append(fold_result)
        print(f"  Fold {fold_idx}: mean R2={mean_r2:.4f}, Ridge R2={ridge_r2:.4f}, "
              f"PySR R2={pysr_r2:.4f}")

    return fold_results, pred_records, all_equations


# ── 10. Execute modelling for feasible targets ────────────────────────────────

t0 = time.time()

target_results = {}

if D3_FEASIBLE:
    print("\n" + "=" * 70)
    print("MODELLING D3 (first_deriv_rms) from D1 + D2")
    print("=" * 70)

    d3_folds, d3_preds, d3_eqs = run_predictive_comparison(
        eligible, "first_deriv_rms", ["n_peaks_prom", "mean_peak_prominence"],
        "specimen_group_id", "D3", seed=SEED,
    )

    d3_folds_df = pd.DataFrame(d3_folds)
    d3_preds_df = pd.DataFrame(d3_preds)
    d3_eqs_df = pd.DataFrame(d3_eqs) if d3_eqs else pd.DataFrame()

    # Aggregate out-of-fold metrics
    d3_agg = {}
    for model in ["mean", "ridge", "pysr"]:
        pred_col = f"pred_{model}"
        actual = d3_preds_df["target_actual"].values.astype(float)
        predicted = d3_preds_df[pred_col].values.astype(float)
        mask = ~np.isnan(predicted)
        if mask.sum() > 0:
            mae = np.mean(np.abs(actual[mask] - predicted[mask]))
            rmse = np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))
            ss_res = np.sum((actual[mask] - predicted[mask]) ** 2)
            ss_tot = np.sum((actual[mask] - actual[mask].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            d3_agg[model] = {"mae": mae, "rmse": rmse, "r2": r2, "n_predictions": int(mask.sum())}

    d3_agg_df = pd.DataFrame(d3_agg).T
    d3_agg_df.index.name = "model"

    print(f"\nD3 Aggregate metrics:")
    print(d3_agg_df.round(4).to_string())

    # Save
    d3_folds_df.to_csv(OUT_DIR / "phase5_step5_23_d3_fold_metrics.csv", index=False)
    d3_preds_df.to_csv(OUT_DIR / "phase5_step5_23_d3_predictions.csv", index=False)
    if not d3_eqs_df.empty:
        d3_eqs_df.to_csv(OUT_DIR / "phase5_step5_23_d3_pysr_equations.csv", index=False)
    d3_agg_df.round(6).to_csv(OUT_DIR / "phase5_step5_23_d3_aggregate_metrics.csv")

    target_results["D3"] = {
        "feasible": True,
        "aggregate_r2": {m: d3_agg[m]["r2"] for m in ["mean", "ridge", "pysr"]},
        "n_equations": len(d3_eqs_df),
    }

if D4_FEASIBLE:
    print("\n" + "=" * 70)
    print("MODELLING D4 (second_deriv_rms) from D1 + D2")
    print("=" * 70)

    d4_folds, d4_preds, d4_eqs = run_predictive_comparison(
        eligible, "second_deriv_rms", ["n_peaks_prom", "mean_peak_prominence"],
        "specimen_group_id", "D4", seed=SEED,
    )

    d4_folds_df = pd.DataFrame(d4_folds)
    d4_preds_df = pd.DataFrame(d4_preds)
    d4_eqs_df = pd.DataFrame(d4_eqs) if d4_eqs else pd.DataFrame()

    d4_agg = {}
    for model in ["mean", "ridge", "pysr"]:
        pred_col = f"pred_{model}"
        actual = d4_preds_df["target_actual"].values.astype(float)
        predicted = d4_preds_df[pred_col].values.astype(float)
        mask = ~np.isnan(predicted)
        if mask.sum() > 0:
            mae = np.mean(np.abs(actual[mask] - predicted[mask]))
            rmse = np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))
            ss_res = np.sum((actual[mask] - predicted[mask]) ** 2)
            ss_tot = np.sum((actual[mask] - actual[mask].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            d4_agg[model] = {"mae": mae, "rmse": rmse, "r2": r2, "n_predictions": int(mask.sum())}

    d4_agg_df = pd.DataFrame(d4_agg).T
    d4_agg_df.index.name = "model"

    print(f"\nD4 Aggregate metrics:")
    print(d4_agg_df.round(4).to_string())

    d4_folds_df.to_csv(OUT_DIR / "phase5_step5_23_d4_fold_metrics.csv", index=False)
    d4_preds_df.to_csv(OUT_DIR / "phase5_step5_23_d4_predictions.csv", index=False)
    if not d4_eqs_df.empty:
        d4_eqs_df.to_csv(OUT_DIR / "phase5_step5_23_d4_pysr_equations.csv", index=False)
    d4_agg_df.round(6).to_csv(OUT_DIR / "phase5_step5_23_d4_aggregate_metrics.csv")

    target_results["D4"] = {
        "feasible": True,
        "aggregate_r2": {m: d4_agg[m]["r2"] for m in ["mean", "ridge", "pysr"]},
        "n_equations": len(d4_eqs_df),
    }

# ── 11. Save fold assignments ─────────────────────────────────────────────────

gkf = GroupKFold(n_splits=N_FOLDS)
fold_assignments = pd.DataFrame({
    "canonical_spectrum_id": eligible["canonical_spectrum_id"],
    "specimen_group_id": eligible["specimen_group_id"],
    "mineral": eligible["mineral"],
})
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(
    eligible[["n_peaks_prom", "mean_peak_prominence"]].values,
    eligible["first_deriv_rms"].values,
    eligible["specimen_group_id"].values,
)):
    fold_assignments[f"fold_{fold_idx}_test"] = False
    fold_assignments.loc[test_idx, f"fold_{fold_idx}_test"] = True

fold_assignments.to_csv(OUT_DIR / "phase5_step5_23_fold_assignments.csv", index=False)

elapsed = time.time() - t0

# ── 12. Build feasibility table ───────────────────────────────────────────────

feasibility_rows = []
for target_name, target_col, r_d1, r_d2, missing, icc, feasible in [
    ("D3 (first_deriv_rms)", "first_deriv_rms", r_d3_d1, r_d3_d2, d3_missing, d3_icc, D3_FEASIBLE),
    ("D4 (second_deriv_rms)", "second_deriv_rms", r_d4_d1, r_d4_d2, d4_missing, d4_icc, D4_FEASIBLE),
]:
    row = {
        "target": target_name,
        "n": len(eligible),
        "n_missing": missing,
        "pearson_r_D1": round(r_d1, 4),
        "pearson_r_D2": round(r_d2, 4),
        "max_abs_r": round(max(abs(r_d1), abs(r_d2)), 4),
        "ICC": round(icc, 3),
        "modelled": feasible,
    }
    if feasible and target_name.split()[0] in target_results:
        tr = target_results[target_name.split()[0]]
        row["mean_R2"] = round(tr["aggregate_r2"]["mean"], 4)
        row["ridge_R2"] = round(tr["aggregate_r2"]["ridge"], 4)
        row["pysr_R2"] = round(tr["aggregate_r2"]["pysr"], 4)
        row["n_pysr_equations"] = tr["n_equations"]
    feasibility_rows.append(row)

feasibility_df = pd.DataFrame(feasibility_rows)
feasibility_df.to_csv(OUT_DIR / "phase5_step5_23_target_feasibility.csv", index=False)

print("\n" + "=" * 70)
print("TARGET FEASIBILITY SUMMARY")
print("=" * 70)
print(feasibility_df.to_string(index=False))

# ── 13. Save run manifest ─────────────────────────────────────────────────────

manifest = {
    "step": "phase5_step5_23",
    "title": "Multivariate Descriptor Prediction: Feasibility and Target Selection",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "random_seed": SEED,
    "n_folds": N_FOLDS,
    "inputs": {
        "spectral_features": "data/processed/phase3_canonical_spectral_features.csv",
        "integrated_dataset": "data/processed/phase4_integrated_analysis_dataset.csv",
        "features_checksum": feat_checksum,
        "integrated_checksum": integ_checksum,
    },
    "population": {
        "n_eligible_spectra": len(eligible),
        "n_groups": int(eligible["specimen_group_id"].nunique()),
        "n_minerals": int(eligible["mineral"].nunique()),
        "eligibility_filter": {
            "high_conflict_excluded": int(df["high_conflict_flag"].sum()),
            "tentative_provenance_excluded": int(df["tentative_provenance"].sum()),
            "identity_issues_excluded": int(df["has_identity_issues"].sum()),
            "mixture_flag_excluded": int(df["mixture_flag"].sum()),
            "total_excluded": int(excluded.sum()),
        },
        "singleton_minerals": ["Beryl", "Grossular"],
    },
    "descriptor_statistics": {
        "D3": {
            "mean": float(D3.mean()),
            "std": float(D3.std()),
            "unique": int(D3.nunique()),
            "missing": int(D3.isna().sum()),
        },
        "D4": {
            "mean": float(D4.mean()),
            "std": float(D4.std()),
            "unique": int(D4.nunique()),
            "missing": int(D4.isna().sum()),
        },
    },
    "correlations": {
        "r_D1_D3": round(float(r_d3_d1), 4),
        "r_D2_D3": round(float(r_d3_d2), 4),
        "r_D1_D4": round(float(r_d4_d1), 4),
        "r_D2_D4": round(float(r_d4_d2), 4),
    },
    "target_decisions": {
        "D3_feasible": bool(D3_FEASIBLE),
        "D4_feasible": bool(D4_FEASIBLE),
        "D3_max_abs_r": round(float(d3_max_r), 4),
        "D4_max_abs_r": round(float(d4_max_r), 4),
        "D3_ICC": round(float(d3_icc), 3),
        "D4_ICC": round(float(d4_icc), 3),
    },
    "pysr_config": {
        "niterations": 80,
        "population_size": 30,
        "maxsize": 20,
        "maxdepth": 4,
        "timeout_in_seconds": 180,
        "binary_operators": ["+", "*", "/"],
        "unary_operators": ["sqrt", "log"],
    },
    "target_results": target_results,
    "total_runtime_s": round(elapsed, 1),
}

manifest_path = OUT_DIR / "phase5_step5_23_run_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nTotal runtime: {elapsed:.1f}s")
print(f"Outputs saved to: {OUT_DIR}")
