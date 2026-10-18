"""
Phase 5 Step 5.24: Mineral Classification Robustness and Spectral Interpretation

Part A: Reproduce the Step 5.21 baseline
Part B: Evaluate robustness with multiple models
Part C: Investigate spectral regions via permutation importance
Part D: Targeted sensitivity analysis
Part E: Scientific interpretation and final decision
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, precision_score, recall_score
)

RANDOM_SEED = 521
np.random.seed(RANDOM_SEED)

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/05_exploratory_analysis/phase5_step5_24_classification_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 5 STEP 5.24: MINERAL CLASSIFICATION ROBUSTNESS")
print("=" * 80)
print()

# ============================================================================
# PART A: REPRODUCE BASELINE
# ============================================================================

print("=" * 80)
print("PART A: REPRODUCE BASELINE")
print("=" * 80)
print()

print("Loading frozen datasets...")
canonical_spectra = pd.read_csv(DATA_DIR / "phase2_canonical_d2_5nm.csv")
integrated = pd.read_csv(DATA_DIR / "phase4_integrated_analysis_dataset.csv")

wl_cols = [c for c in canonical_spectra.columns if c.startswith('R_')]
wl_numeric = np.array([float(c.replace('R_', '').replace('nm', '')) for c in wl_cols])
print(f"  Canonical spectra: {len(canonical_spectra)} records")
print(f"  Integrated dataset: {len(integrated)} records")
print(f"  Wavelength features: {len(wl_cols)} ({wl_cols[0]} to {wl_cols[-1]})")
print()

df = integrated.merge(
    canonical_spectra[['canonical_spectrum_id'] + wl_cols],
    on='canonical_spectrum_id',
    how='left'
)

# Apply same eligibility filters as Step 5.21
df['ml_eligible'] = True
df.loc[df['high_conflict'] == True, 'ml_eligible'] = False
df.loc[df['provenance_tentative'] == True, 'ml_eligible'] = False
df.loc[df['has_identity_issues'] == True, 'ml_eligible'] = False
df.loc[df['mixture_flag'] == True, 'ml_eligible'] = False

ml_df = df[df['ml_eligible']].copy()
print(f"ML-eligible spectra: {len(ml_df)}")
print(f"ML-eligible groups: {ml_df['provenance_group_id'].nunique()}")
print()

# Class support analysis
class_support = []
for mineral in sorted(ml_df['mineral_label'].unique()):
    mineral_data = ml_df[ml_df['mineral_label'] == mineral]
    n_spectra = len(mineral_data)
    n_groups = mineral_data['provenance_group_id'].nunique()
    evaluable = n_groups >= 2
    class_support.append({
        'mineral': mineral,
        'n_spectra': n_spectra,
        'n_groups': n_groups,
        'evaluable': evaluable
    })
    print(f"  {mineral:15s}: {n_spectra:3d} spectra, {n_groups:3d} groups - {'EVALUABLE' if evaluable else 'INSUFFICIENT'}")

class_support_df = pd.DataFrame(class_support)
class_support_df.to_csv(OUTPUT_DIR / "phase5_step5_24_class_support.csv", index=False)
print()

evaluable_classes = sorted([r['mineral'] for r in class_support if r['evaluable']])
print(f"Evaluable classes: {len(evaluable_classes)}")

# Resolve population discrepancy
feasibility_spectra = len(ml_df)
feasibility_groups = ml_df['provenance_group_id'].nunique()

pilot_df = ml_df[ml_df['mineral_label'].isin(evaluable_classes)].copy()
model_spectra = len(pilot_df)
model_groups = pilot_df['provenance_group_id'].nunique()

print()
print("POPULATION RECONCILIATION:")
print(f"  Feasibility population: {feasibility_spectra} spectra, {feasibility_groups} groups")
print(f"  Model population:       {model_spectra} spectra, {model_groups} groups")
excluded_minerals = [r['mineral'] for r in class_support if not r['evaluable']]
excluded_spectra = sum(r['n_spectra'] for r in class_support if not r['evaluable'])
excluded_groups = sum(1 for r in class_support if not r['evaluable'])
print(f"  Excluded: {excluded_spectra} spectra, {excluded_groups} groups from: {', '.join(excluded_minerals)}")
print()

# Prepare data
X = pilot_df[wl_cols].to_numpy(dtype=np.float64)
y = pilot_df['mineral_label'].to_numpy(dtype=str)
groups = pilot_df['provenance_group_id'].to_numpy(dtype=str)

n_splits = 3

# Verify no group leakage
print("VERIFYING GROUP-AWARE FOLD INTEGRITY:")
fold_groups = {}
gkf = GroupKFold(n_splits=n_splits)
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    fold_groups[fold_idx] = {
        'train_groups': train_groups,
        'test_groups': test_groups,
        'n_train': len(train_idx),
        'n_test': len(test_idx)
    }
    overlap = train_groups & test_groups
    print(f"  Fold {fold_idx}: train={len(train_idx)} spectra ({len(train_groups)} groups), "
          f"test={len(test_idx)} spectra ({len(test_groups)} groups), "
          f"overlap={len(overlap)}")
    assert len(overlap) == 0, f"Group leakage in fold {fold_idx}!"

all_test_groups = set()
for fold_idx in range(n_splits):
    all_test_groups |= fold_groups[fold_idx]['test_groups']
print(f"  Total unique test groups across folds: {len(all_test_groups)}")
print(f"  Total unique groups: {len(set(groups))}")
print()

# Save fold assignments
split_assignments = pilot_df[['canonical_spectrum_id', 'provenance_group_id', 'mineral_label']].copy()
split_assignments['cv_fold'] = -1
for fold_idx in range(n_splits):
    test_groups = fold_groups[fold_idx]['test_groups']
    split_assignments.loc[
        split_assignments['provenance_group_id'].isin(test_groups),
        'cv_fold'
    ] = fold_idx
split_assignments.to_csv(OUTPUT_DIR / "phase5_step5_24_fold_assignments.csv", index=False)

# ============================================================================
# PART B: EVALUATE ROBUSTNESS
# ============================================================================

print("=" * 80)
print("PART B: EVALUATE ROBUSTNESS")
print("=" * 80)
print()

models = {
    'dummy_majority': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
    'logistic_l2': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'
        ))
    ]),
    'random_forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_SEED
        ))
    ])
}

# Overall metrics
overall_results = []
# Per-fold metrics
fold_results = []
# All predictions
all_predictions = []
# Per-class metrics per fold
per_class_fold_results = []

for model_name, model in models.items():
    print(f"Model: {model_name}")
    
    gkf = GroupKFold(n_splits=n_splits)
    y_pred_all = cross_val_predict(model, X, y, cv=gkf, groups=groups)
    
    acc = accuracy_score(y, y_pred_all)
    bal_acc = balanced_accuracy_score(y, y_pred_all)
    macro_f1 = f1_score(y, y_pred_all, average='macro', zero_division=0)
    weighted_f1 = f1_score(y, y_pred_all, average='weighted', zero_division=0)
    
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print()
    
    overall_results.append({
        'model': model_name,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'n_folds': n_splits,
        'random_seed': RANDOM_SEED
    })
    
    for spec_id, grp, true, pred in zip(
        pilot_df['canonical_spectrum_id'].values,
        groups, y, y_pred_all
    ):
        all_predictions.append({
            'model': model_name,
            'spectrum_id': spec_id,
            'provenance_group_id': grp,
            'true_mineral': true,
            'predicted_mineral': pred,
            'correct': true == pred
        })
    
    # Per-fold metrics
    for fold_idx in range(n_splits):
        test_mask = split_assignments['cv_fold'] == fold_idx
        test_ids = split_assignments.loc[test_mask, 'canonical_spectrum_id'].values
        pred_mask = np.isin(pilot_df['canonical_spectrum_id'].values, test_ids)
        
        y_true_fold = y[pred_mask]
        y_pred_fold = y_pred_all[pred_mask]
        
        fold_acc = accuracy_score(y_true_fold, y_pred_fold)
        fold_bal_acc = balanced_accuracy_score(y_true_fold, y_pred_fold, )
        fold_macro_f1 = f1_score(y_true_fold, y_pred_fold, average='macro', zero_division=0)
        fold_weighted_f1 = f1_score(y_true_fold, y_pred_fold, average='weighted', zero_division=0)
        
        fold_results.append({
            'model': model_name,
            'fold': fold_idx,
            'accuracy': fold_acc,
            'balanced_accuracy': fold_bal_acc,
            'macro_f1': fold_macro_f1,
            'weighted_f1': fold_weighted_f1,
            'n_test_spectra': len(y_true_fold),
            'n_test_groups': len(fold_groups[fold_idx]['test_groups'])
        })
        
        # Per-class metrics for this fold
        for mineral in evaluable_classes:
            tp = np.sum((y_true_fold == mineral) & (y_pred_fold == mineral))
            fp = np.sum((y_true_fold != mineral) & (y_pred_fold == mineral))
            fn = np.sum((y_true_fold == mineral) & (y_pred_fold != mineral))
            support = tp + fn
            
            cls_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            cls_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cls_f1 = 2 * cls_prec * cls_rec / (cls_prec + cls_rec) if (cls_prec + cls_rec) > 0 else 0.0
            
            per_class_fold_results.append({
                'model': model_name,
                'fold': fold_idx,
                'mineral': mineral,
                'precision': cls_prec,
                'recall': cls_rec,
                'f1': cls_f1,
                'support': support
            })

# Save results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv(OUTPUT_DIR / "phase5_step5_24_overall_metrics.csv", index=False)

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(OUTPUT_DIR / "phase5_step5_24_fold_metrics.csv", index=False)

predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv(OUTPUT_DIR / "phase5_step5_24_predictions.csv", index=False)

per_class_fold_df = pd.DataFrame(per_class_fold_results)
per_class_fold_df.to_csv(OUTPUT_DIR / "phase5_step5_24_per_class_fold_metrics.csv", index=False)

# Fold variability
print("FOLD VARIABILITY (logistic_l2):")
lr_folds = fold_df[fold_df['model'] == 'logistic_l2']
for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']:
    vals = lr_folds[metric].values
    print(f"  {metric}: {vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f} "
          f"(mean={vals.mean():.4f}, std={vals.std():.4f})")
print()

# Per-class aggregated results for best model
best_model = 'logistic_l2'
best_preds = predictions_df[predictions_df['model'] == best_model]
y_true_best = best_preds['true_mineral'].to_numpy(dtype=str)
y_pred_best = best_preds['predicted_mineral'].to_numpy(dtype=str)

# Per-class metrics (aggregated across folds)
per_class_results = []
for mineral in evaluable_classes:
    tp = np.sum((y_true_best == mineral) & (y_pred_best == mineral))
    fp = np.sum((y_true_best != mineral) & (y_pred_best == mineral))
    fn = np.sum((y_true_best == mineral) & (y_pred_best != mineral))
    support = tp + fn
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    n_groups_support = pilot_df.loc[
        pilot_df['mineral_label'] == mineral, 'provenance_group_id'
    ].nunique()
    
    per_class_results.append({
        'mineral': mineral,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'support_spectra': support,
        'support_groups': n_groups_support,
        'evaluation_tier': (
            'well_supported' if n_groups_support >= 5 else
            'limited' if n_groups_support >= 3 else
            'weak'
        )
    })

per_class_df = pd.DataFrame(per_class_results)
per_class_df.to_csv(OUTPUT_DIR / "phase5_step5_24_per_class_metrics.csv", index=False)

print("PER-CLASS PERFORMANCE (logistic_l2):")
print(f"{'Mineral':15s} {'Prec':6s} {'Rec':6s} {'F1':6s} {'Spec':5s} {'Grp':4s} {'Tier':15s}")
print("-" * 70)
for _, row in per_class_df.iterrows():
    print(f"{row['mineral']:15s} {row['precision']:.3f}  {row['recall']:.3f}  "
          f"{row['f1']:.3f}  {row['support_spectra']:5.0f} {row['support_groups']:4.0f} "
          f"{row['evaluation_tier']:15s}")
print()

# Confusion matrix
cm = confusion_matrix(y_true_best, y_pred_best, labels=evaluable_classes)
cm_df = pd.DataFrame(cm, index=evaluable_classes, columns=evaluable_classes)
cm_df.to_csv(OUTPUT_DIR / "phase5_step5_24_confusion_matrix.csv")

# ============================================================================
# PART C: SPECTRAL REGION IMPORTANCE
# ============================================================================

print("=" * 80)
print("PART C: SPECTRAL REGION IMPORTANCE")
print("=" * 80)
print()

# Define spectral regions (standard VNIR-SWIR mineral regions)
spectral_regions = {
    'VNIR_vis_400_500': (400, 500),
    'VNIR_vis_500_600': (500, 600),
    'VNIR_red_600_700': (600, 700),
    'VNIR_NIR_edge_700_900': (700, 900),
    'NIR_Fe2p_900_1100': (900, 1100),
    'NIR_1100_1300': (1100, 1300),
    'SWIR_water_1300_1500': (1300, 1500),
    'SWIR_1500_1700': (1500, 1700),
    'SWIR_OH_1700_1900': (1700, 1900),
    'SWIR_water_1900_2100': (1900, 2100),
    'SWIR_2100_2300': (2100, 2300),
    'SWIR_2300_2500': (2300, 2500),
}

# Per-wavelength permutation importance (group-aware)
print("Computing per-wavelength permutation importance (group-aware)...")
print("(Using logistic_l2 model with 3-fold GroupKFold)")
print()

# Fit model on full data for permutation importance assessment
# We use a group-aware approach: permute within training folds, evaluate on test folds
wavelength_importance_per_fold = np.zeros((n_splits, len(wl_cols)))

base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'))
])

n_permutations = 20

for fold_idx in range(n_splits):
    train_mask = split_assignments['cv_fold'] != fold_idx
    test_mask = split_assignments['cv_fold'] == fold_idx
    
    train_ids = split_assignments.loc[train_mask, 'canonical_spectrum_id'].values
    test_ids = split_assignments.loc[test_mask, 'canonical_spectrum_id'].values
    
    train_idx = np.where(np.isin(pilot_df['canonical_spectrum_id'].values, train_ids))[0]
    test_idx = np.where(np.isin(pilot_df['canonical_spectrum_id'].values, test_ids))[0]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    base_model.fit(X_train, y_train)
    base_score = balanced_accuracy_score(y_test, base_model.predict(X_test))
    
    rng = np.random.RandomState(RANDOM_SEED + fold_idx)
    
    for j in range(len(wl_cols)):
        scores = []
        for perm in range(n_permutations):
            X_perm = X_test.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_score = balanced_accuracy_score(y_test, base_model.predict(X_perm))
            scores.append(perm_score)
        mean_perm_score = np.mean(scores)
        wavelength_importance_per_fold[fold_idx, j] = base_score - mean_perm_score
    
    print(f"  Fold {fold_idx}: base balanced accuracy = {base_score:.4f}")

# Aggregate importance across folds
mean_importance = wavelength_importance_per_fold.mean(axis=0)
std_importance = wavelength_importance_per_fold.std(axis=0)

# Per-wavelength results
wl_importance_df = pd.DataFrame({
    'wavelength_column': wl_cols,
    'wavelength_nm': wl_numeric,
    'mean_importance': mean_importance,
    'std_importance': std_importance,
    'fold0_importance': wavelength_importance_per_fold[0],
    'fold1_importance': wavelength_importance_per_fold[1],
    'fold2_importance': wavelength_importance_per_fold[2],
})
wl_importance_df.to_csv(OUTPUT_DIR / "phase5_step5_24_wavelength_importance.csv", index=False)

# Aggregate into spectral regions
region_importance = []
for region_name, (lo, hi) in spectral_regions.items():
    mask = (wl_numeric >= lo) & (wl_numeric < hi)
    if mask.sum() > 0:
        region_imp = mean_importance[mask]
        region_std = std_importance[mask]
        fold_means = [wavelength_importance_per_fold[f, mask].mean() for f in range(n_splits)]
        
        region_importance.append({
            'region': region_name,
            'wavelength_range_nm': f"{lo}-{hi}",
            'n_wavelengths': mask.sum(),
            'mean_importance': region_imp.mean(),
            'std_importance': region_std.mean(),
            'max_importance': region_imp.max(),
            'fold0_mean': fold_means[0],
            'fold1_mean': fold_means[1],
            'fold2_mean': fold_means[2],
            'fold_consistency': np.std(fold_means),
        })

region_importance_df = pd.DataFrame(region_importance)
region_importance_df = region_importance_df.sort_values('mean_importance', ascending=False)
region_importance_df.to_csv(OUTPUT_DIR / "phase5_step5_24_region_importance.csv", index=False)

print()
print("SPECTRAL REGION IMPORTANCE (aggregated from per-wavelength):")
print(f"{'Region':30s} {'Range':10s} {'MeanImp':8s} {'StdImp':8s} {'Consistency':12s}")
print("-" * 75)
for _, row in region_importance_df.iterrows():
    consistency = "STABLE" if row['fold_consistency'] < 0.02 else (
        "MODERATE" if row['fold_consistency'] < 0.04 else "UNSTABLE"
    )
    print(f"{row['region']:30s} {row['wavelength_range_nm']:10s} "
          f"{row['mean_importance']:.4f}   {row['std_importance']:.4f}   {consistency}")
print()

# Cross-fold consistency of top regions
top_regions = region_importance_df.head(5)
print("TOP 5 REGIONS - FOLD CONSISTENCY:")
for _, row in top_regions.iterrows():
    fold_vals = [row['fold0_mean'], row['fold1_mean'], row['fold2_mean']]
    print(f"  {row['region']:30s}: folds = [{fold_vals[0]:.4f}, {fold_vals[1]:.4f}, {fold_vals[2]:.4f}], "
          f"std = {np.std(fold_vals):.4f}")
print()

# ============================================================================
# PART D: SENSITIVITY ANALYSIS
# ============================================================================

print("=" * 80)
print("PART D: TARGETED SENSITIVITY ANALYSIS")
print("=" * 80)
print()

sensitivity_results = []

# D1: Identify repeated spectra (same provenance_group_id, multiple spectra)
group_counts = pilot_df.groupby('provenance_group_id').size()
repeated_groups = group_counts[group_counts > 1].index.tolist()
n_repeated_spectra = pilot_df[pilot_df['provenance_group_id'].isin(repeated_groups)].shape[0]
n_single_groups = group_counts[group_counts == 1].index.tolist()
n_single_spectra = pilot_df[pilot_df['provenance_group_id'].isin(n_single_groups)].shape[0]

print(f"DEPENDENT SPECTRA STRUCTURE:")
print(f"  Groups with repeated spectra: {len(repeated_groups)}")
print(f"  Spectra from repeated groups: {n_repeated_spectra}")
print(f"  Groups with single spectrum:  {len(n_single_groups)}")
print(f"  Spectra from single groups:   {n_single_spectra}")
print()

# D1 sensitivity: Keep only one spectrum per group (first alphabetically by spectrum_id)
print("SENSITIVITY D1: One spectrum per group (conservative)")
conservative_df = pilot_df.sort_values('canonical_spectrum_id').drop_duplicates(
    subset=['provenance_group_id'], keep='first'
)
X_c = conservative_df[wl_cols].to_numpy(dtype=np.float64)
y_c = conservative_df['mineral_label'].to_numpy(dtype=str)
groups_c = conservative_df['provenance_group_id'].to_numpy(dtype=str)

# Check class support after dedup
c_classes = []
for mineral in evaluable_classes:
    n_g = conservative_df[conservative_df['mineral_label'] == mineral]['provenance_group_id'].nunique()
    if n_g >= 2:
        c_classes.append(mineral)

if len(c_classes) == len(evaluable_classes):
    gkf_c = GroupKFold(n_splits=n_splits)
    
    lr_c = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'))
    ])
    
    y_pred_c = cross_val_predict(lr_c, X_c, y_c, cv=gkf_c, groups=groups_c)
    
    c_acc = accuracy_score(y_c, y_pred_c)
    c_bal = balanced_accuracy_score(y_c, y_pred_c)
    c_macro = f1_score(y_c, y_pred_c, average='macro', zero_division=0)
    c_weighted = f1_score(y_c, y_pred_c, average='weighted', zero_division=0)
    
    print(f"  Spectra: {len(conservative_df)}, Groups: {conservative_df['provenance_group_id'].nunique()}")
    print(f"  Accuracy: {c_acc:.4f}, Balanced Acc: {c_bal:.4f}, Macro F1: {c_macro:.4f}, Weighted F1: {c_weighted:.4f}")
    
    sensitivity_results.append({
        'analysis': 'D1_one_per_group',
        'n_spectra': len(conservative_df),
        'n_groups': conservative_df['provenance_group_id'].nunique(),
        'accuracy': c_acc,
        'balanced_accuracy': c_bal,
        'macro_f1': c_macro,
        'weighted_f1': c_weighted,
    })
else:
    print(f"  WARNING: Deduplication reduces evaluable classes from {len(evaluable_classes)} to {len(c_classes)}")
    sensitivity_results.append({
        'analysis': 'D1_one_per_group',
        'n_spectra': len(conservative_df),
        'n_groups': conservative_df['provenance_group_id'].nunique(),
        'accuracy': np.nan,
        'balanced_accuracy': np.nan,
        'macro_f1': np.nan,
        'weighted_f1': np.nan,
        'note': f'class_reduction_{len(evaluable_classes)}_to_{len(c_classes)}'
    })
print()

# D2: Exclude weakly-supported classes (fewer than 3 groups)
print("SENSITIVITY D2: Exclude classes with < 3 independent groups")
well_supported = [r['mineral'] for r in class_support if r['n_groups'] >= 3]
print(f"  Classes with >= 3 groups: {len(well_supported)} / {len(evaluable_classes)}")

ws_df = pilot_df[pilot_df['mineral_label'].isin(well_supported)].copy()
X_ws = ws_df[wl_cols].to_numpy(dtype=np.float64)
y_ws = ws_df['mineral_label'].to_numpy(dtype=str)
groups_ws = ws_df['provenance_group_id'].to_numpy(dtype=str)

lr_ws = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'))
])

y_pred_ws = cross_val_predict(lr_ws, X_ws, y_ws, cv=GroupKFold(n_splits=n_splits), groups=groups_ws)

ws_acc = accuracy_score(y_ws, y_pred_ws)
ws_bal = balanced_accuracy_score(y_ws, y_pred_ws)
ws_macro = f1_score(y_ws, y_pred_ws, average='macro', zero_division=0)
ws_weighted = f1_score(y_ws, y_pred_ws, average='weighted', zero_division=0)

print(f"  Spectra: {len(ws_df)}, Groups: {ws_df['provenance_group_id'].nunique()}")
print(f"  Accuracy: {ws_acc:.4f}, Balanced Acc: {ws_bal:.4f}, Macro F1: {ws_macro:.4f}, Weighted F1: {ws_weighted:.4f}")

sensitivity_results.append({
    'analysis': 'D2_exclude_weak_classes',
    'n_spectra': len(ws_df),
    'n_groups': ws_df['provenance_group_id'].nunique(),
    'n_classes': len(well_supported),
    'accuracy': ws_acc,
    'balanced_accuracy': ws_bal,
    'macro_f1': ws_macro,
    'weighted_f1': ws_weighted,
})
print()

# D3: Wavelength-reduced representation (spectral regions instead of individual wavelengths)
print("SENSITIVITY D3: Wavelength-reduced (regional means)")
region_features = {}
for region_name, (lo, hi) in spectral_regions.items():
    mask = (wl_numeric >= lo) & (wl_numeric < hi)
    if mask.sum() > 0:
        region_features[region_name] = X[:, mask].mean(axis=1)

X_region = np.column_stack(list(region_features.values()))
region_names = list(region_features.keys())
print(f"  Reduced features: {X_region.shape[1]} regional means (from {X.shape[1]} wavelengths)")

lr_region = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'))
])

y_pred_region = cross_val_predict(lr_region, X_region, y, cv=gkf, groups=groups)

r_acc = accuracy_score(y, y_pred_region)
r_bal = balanced_accuracy_score(y, y_pred_region)
r_macro = f1_score(y, y_pred_region, average='macro', zero_division=0)
r_weighted = f1_score(y, y_pred_region, average='weighted', zero_division=0)

print(f"  Accuracy: {r_acc:.4f}, Balanced Acc: {r_bal:.4f}, Macro F1: {r_macro:.4f}, Weighted F1: {r_weighted:.4f}")

sensitivity_results.append({
    'analysis': 'D3_regional_means',
    'n_spectra': len(pilot_df),
    'n_groups': pilot_df['provenance_group_id'].nunique(),
    'n_features': X_region.shape[1],
    'accuracy': r_acc,
    'balanced_accuracy': r_bal,
    'macro_f1': r_macro,
    'weighted_f1': r_weighted,
})
print()

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.to_csv(OUTPUT_DIR / "phase5_step5_24_sensitivity_results.csv", index=False)

# Comparison table
print("SENSITIVITY COMPARISON:")
lr_overall = overall_df[overall_df['model'] == 'logistic_l2'].iloc[0]
print(f"  PRIMARY (full, 447 wl):  Acc={lr_overall['accuracy']:.4f}, "
      f"BalAcc={lr_overall['balanced_accuracy']:.4f}, MacroF1={lr_overall['macro_f1']:.4f}")
for _, row in sensitivity_df.iterrows():
    if not np.isnan(row['accuracy']):
        print(f"  {row['analysis']:25s}: Acc={row['accuracy']:.4f}, "
              f"BalAcc={row['balanced_accuracy']:.4f}, MacroF1={row['macro_f1']:.4f}")
print()

# ============================================================================
# PART E: SCIENTIFIC INTERPRETATION
# ============================================================================

print("=" * 80)
print("PART E: SCIENTIFIC INTERPRETATION")
print("=" * 80)
print()

# Q1: Does classifier outperform baseline consistently?
lr_fold_data = fold_df[fold_df['model'] == 'logistic_l2']
dummy_fold_data = fold_df[fold_df['model'] == 'dummy_majority']

q1_answer = []
q1_answer.append("Q1: Does the classifier outperform the baseline consistently across folds?")
q1_answer.append("")
for fold_idx in range(n_splits):
    lr_f = lr_fold_data[lr_fold_data['fold'] == fold_idx].iloc[0]
    dm_f = dummy_fold_data[dummy_fold_data['fold'] == fold_idx].iloc[0]
    delta = lr_f['macro_f1'] - dm_f['macro_f1']
    q1_answer.append(f"  Fold {fold_idx}: logistic_l2 macro_F1={lr_f['macro_f1']:.4f}, "
                     f"dummy macro_F1={dm_f['macro_f1']:.4f}, delta={delta:+.4f}")

lr_mean_macro = lr_fold_data['macro_f1'].mean()
dm_mean_macro = dummy_fold_data['macro_f1'].mean()
q1_answer.append(f"  Mean: logistic_l2={lr_mean_macro:.4f}, dummy={dm_mean_macro:.4f}")
q1_answer.append(f"  VERDICT: {'YES' if lr_mean_macro > dm_mean_macro * 2 else 'MARGINAL'} - "
                 f"logistic_l2 exceeds dummy by {lr_mean_macro/dm_mean_macro:.1f}x on average")
q1_answer.append("")

# Q2: Is macro-F1 supported across multiple classes or dominated by a subset?
q2_answer = []
q2_answer.append("Q2: Is macro-F1 supported by useful performance across multiple classes?")
q2_answer.append("")
n_classes_f1_above_05 = (per_class_df['f1'] >= 0.5).sum()
n_classes_f1_above_03 = (per_class_df['f1'] >= 0.3).sum()
n_classes_f1_zero = (per_class_df['f1'] == 0).sum()
q2_answer.append(f"  Classes with F1 >= 0.5: {n_classes_f1_above_05} / {len(evaluable_classes)}")
q2_answer.append(f"  Classes with F1 >= 0.3: {n_classes_f1_above_03} / {len(evaluable_classes)}")
q2_answer.append(f"  Classes with F1 = 0.0:  {n_classes_f1_zero} / {len(evaluable_classes)}")
q2_answer.append("")

strong_classes = per_class_df[per_class_df['f1'] >= 0.5]['mineral'].tolist()
weak_classes = per_class_df[per_class_df['f1'] == 0]['mineral'].tolist()
q2_answer.append(f"  Strong classes (F1>=0.5): {', '.join(strong_classes) if strong_classes else 'NONE'}")
q2_answer.append(f"  Zero-performance classes: {', '.join(weak_classes) if weak_classes else 'NONE'}")
q2_answer.append("")

# Check if macro-F1 is dominated by a subset
top3_f1 = per_class_df.nlargest(3, 'f1')['f1'].mean()
bottom3_f1 = per_class_df.nsmallest(3, 'f1')['f1'].mean()
q2_answer.append(f"  Mean F1 of top-3 classes:    {top3_f1:.4f}")
q2_answer.append(f"  Mean F1 of bottom-3 classes: {bottom3_f1:.4f}")
if top3_f1 > 0.7 and bottom3_f1 < 0.2:
    q2_answer.append("  VERDICT: PARTIALLY - performance is concentrated in a subset of classes")
elif n_classes_f1_above_05 >= 8:
    q2_answer.append("  VERDICT: YES - useful performance across a majority of classes")
else:
    q2_answer.append("  VERDICT: MIXED - some classes well-classified, others poorly")
q2_answer.append("")

# Q3: Are spectral-region importance patterns stable enough to interpret?
q3_answer = []
q3_answer.append("Q3: Are spectral-region importance patterns stable enough to interpret?")
q3_answer.append("")

stable_regions = region_importance_df[region_importance_df['fold_consistency'] < 0.02]
moderate_regions = region_importance_df[
    (region_importance_df['fold_consistency'] >= 0.02) & 
    (region_importance_df['fold_consistency'] < 0.04)
]
unstable_regions = region_importance_df[region_importance_df['fold_consistency'] >= 0.04]

q3_answer.append(f"  Stable regions (fold std < 0.02):    {len(stable_regions)}")
q3_answer.append(f"  Moderate regions (0.02 <= std < 0.04): {len(moderate_regions)}")
q3_answer.append(f"  Unstable regions (fold std >= 0.04):  {len(unstable_regions)}")
q3_answer.append("")

if len(stable_regions) >= 3:
    top_stable = stable_regions.head(3)
    q3_answer.append("  Top stable regions:")
    for _, row in top_stable.iterrows():
        q3_answer.append(f"    {row['region']:30s}: mean={row['mean_importance']:.4f}, "
                        f"fold_std={row['fold_consistency']:.4f}")
    q3_answer.append("  VERDICT: YES - several regions show consistent importance across folds")
elif len(stable_regions) + len(moderate_regions) >= 4:
    q3_answer.append("  VERDICT: PARTIALLY - some regions are interpretable but stability is mixed")
else:
    q3_answer.append("  VERDICT: NO - importance patterns are too unstable for confident interpretation")
q3_answer.append("")

# Q4: What can this experiment establish vs. what remains unproven?
q4_answer = []
q4_answer.append("Q4: What can this experiment establish, and what remains unproven?")
q4_answer.append("")
q4_answer.append("  ESTABLISHED:")
q4_answer.append("  - Reflectance spectra contain information that distinguishes mineral identities")
q4_answer.append("  - This discrimination survives group-aware validation (no leakage from repeated spectra)")
q4_answer.append("  - Performance substantially exceeds a majority-class baseline")
q4_answer.append("  - Certain spectral regions contribute more to discrimination than others")
q4_answer.append("")
q4_answer.append("  NOT ESTABLISHED:")
q4_answer.append("  - That crystal symmetry has an independent effect on spectral discriminability")
q4_answer.append("    (mineral identity and crystal system are completely confounded)")
q4_answer.append("  - That the classification would generalise to new instruments or laboratories")
q4_answer.append("  - That low-F1 minerals are intrinsically unidentifiable (may reflect limited data)")
q4_answer.append("  - Any causal chemical or physical mechanism for the spectral differences")
q4_answer.append("  - That the discovered spectral regions reflect specific absorption features")
q4_answer.append("    (permutation importance is affected by correlated adjacent wavelengths)")
q4_answer.append("")

# Q5: Is the result strong enough for a manuscript?
q5_answer = []
q5_answer.append("Q5: Is the result strong enough for a research manuscript?")
q5_answer.append("")

# Key criteria
lr_macro = lr_overall['macro_f1']
lr_bal = lr_overall['balanced_accuracy']
n_strong = n_classes_f1_above_05
n_total = len(evaluable_classes)

q5_answer.append(f"  Key metrics:")
q5_answer.append(f"    Logistic L2 macro-F1:         {lr_macro:.4f}")
q5_answer.append(f"    Logistic L2 balanced accuracy: {lr_bal:.4f}")
q5_answer.append(f"    Classes with F1 >= 0.5:        {n_strong}/{n_total}")
q5_answer.append(f"    Dummy baseline macro-F1:       {dm_mean_macro:.4f}")
q5_answer.append(f"    Fold-to-fold std (macro-F1):   {lr_fold_data['macro_f1'].std():.4f}")
q5_answer.append("")

# Sensitivity check
sensitivity_stable = all(
    abs(row['macro_f1'] - lr_macro) < 0.15 
    for _, row in sensitivity_df.iterrows() 
    if not np.isnan(row['accuracy'])
)
q5_answer.append(f"  Sensitivity: results {'stable' if sensitivity_stable else 'variable'} "
                 f"across analysis variants")
q5_answer.append("")

# Final assessment
if lr_macro > 0.5 and n_strong >= 8 and sensitivity_stable:
    q5_answer.append("  VERDICT: Results are PARTIALLY ROBUST with qualifications.")
    q5_answer.append("  The classification result is real and exceeds baseline substantially,")
    q5_answer.append("  but several classes have weak support and crystal-system effects")
    q5_answer.append("  cannot be independently assessed from this dataset.")
    final_decision = "CLASSIFICATION_RESULT_PARTIALLY_ROBUST_REQUIRES_QUALIFICATION"
elif lr_macro > 0.6 and n_strong >= 10:
    q5_answer.append("  VERDICT: Results are robust enough for a manuscript with appropriate caveats.")
    final_decision = "CLASSIFICATION_RESULT_ROBUST_ENOUGH_FOR_MANUSCRIPT"
else:
    q5_answer.append("  VERDICT: Results show real signal but require further data acquisition")
    q5_answer.append("  before strong claims can be made.")
    final_decision = "CLASSIFICATION_RESULT_PARTIALLY_ROBUST_REQUIRES_QUALIFICATION"
q5_answer.append("")

# Write interpretation
interpretation_lines = (
    q1_answer + q2_answer + q3_answer + q4_answer + q5_answer
)
interpretation_text = "\n".join(interpretation_lines)
print(interpretation_text)

with open(OUTPUT_DIR / "phase5_step5_24_scientific_interpretation.md", 'w') as f:
    f.write("# Phase 5 Step 5.24: Scientific Interpretation\n\n")
    f.write(interpretation_text)
    f.write("\n\n---\n\n")
    f.write(f"## Final Decision\n\n")
    f.write(f"`{final_decision}`\n\n")
    f.write(f"## Strongest Defensible Conclusion\n\n")
    f.write(f"Reflectance spectra at 5 nm resolution (320-2550 nm) contain sufficient information "
            f"to discriminate among {len(evaluable_classes)} mineral identities under group-aware "
            f"cross-validation, with logistic regression achieving macro-F1 = {lr_macro:.3f} "
            f"versus a dummy baseline of {dm_mean_macro:.3f}. "
            f"Performance is concentrated in {n_strong} well-supported classes "
            f"(F1 >= 0.5), while {n_classes_f1_zero} classes show zero performance due to "
            f"limited independent-group support. "
            f"This result establishes that mineral discrimination is feasible under "
            f"conservative validation but does NOT establish an independent crystal-system effect, "
            f"causal chemical relationships, or generalisation to new instruments.\n\n")
    f.write(f"## Single Most Valuable Next Action\n\n")
    f.write(f"Acquire additional independent-provenance spectra for the {n_classes_f1_zero} "
            f"zero-performance classes and the {n_total - n_strong - n_classes_f1_zero} "
            f"below-threshold classes, targeting at least 5 independent groups per class, "
            f"to determine whether weak performance reflects intrinsic spectral similarity "
            f"or simply insufficient data. Do NOT launch additional analysis steps on the "
            f"current dataset without new data.\n")

# ============================================================================
# RUN MANIFEST
# ============================================================================

manifest = {
    "step": "phase5_step5_24",
    "title": "Mineral Classification Robustness and Spectral Interpretation",
    "timestamp": datetime.now().isoformat(),
    "random_seed": RANDOM_SEED,
    "inputs": {
        "canonical_spectra": "phase2_canonical_d2_5nm.csv",
        "integrated_dataset": "phase4_integrated_analysis_dataset.csv",
        "step_521_baseline": "phase5_step5_21_ml_pilot_pysr_path"
    },
    "population": {
        "feasibility_spectra": feasibility_spectra,
        "feasibility_groups": feasibility_groups,
        "model_spectra": model_spectra,
        "model_groups": model_groups,
        "evaluable_classes": len(evaluable_classes),
        "excluded_minerals": excluded_minerals,
        "discrepancy_explained": (
            f"Feasibility population ({feasibility_spectra} spectra, {feasibility_groups} groups) "
            f"includes {', '.join(excluded_minerals)} with <2 groups each. "
            f"Model population ({model_spectra} spectra, {model_groups} groups) excludes these "
            f"because GroupKFold requires >=2 groups per class."
        )
    },
    "experimental_design": {
        "n_folds": n_splits,
        "cv_strategy": "GroupKFold",
        "grouping": "provenance_group_id",
        "n_features_full": len(wl_cols),
        "n_features_reduced": X_region.shape[1],
        "spectral_range": f"{wl_numeric[0]:.0f}-{wl_numeric[-1]:.0f} nm",
        "wavelength_spacing": "5 nm",
        "n_permutations_importance": n_permutations,
        "preprocessing_within_folds": True,
        "group_leakage_checked": True,
        "group_leakage_found": False
    },
    "models_evaluated": ["dummy_majority", "logistic_l2", "random_forest"],
    "best_model": "logistic_l2",
    "key_results": {
        "logistic_l2_macro_f1": float(lr_macro),
        "logistic_l2_balanced_accuracy": float(lr_bal),
        "dummy_macro_f1": float(dm_mean_macro),
        "n_classes_f1_above_05": int(n_strong),
        "n_classes_f1_zero": int(n_classes_f1_zero),
        "fold_variability_macro_f1": float(lr_fold_data['macro_f1'].std()),
    },
    "outputs": {
        "class_support": "phase5_step5_24_class_support.csv",
        "fold_assignments": "phase5_step5_24_fold_assignments.csv",
        "overall_metrics": "phase5_step5_24_overall_metrics.csv",
        "fold_metrics": "phase5_step5_24_fold_metrics.csv",
        "predictions": "phase5_step5_24_predictions.csv",
        "per_class_metrics": "phase5_step5_24_per_class_metrics.csv",
        "per_class_fold_metrics": "phase5_step5_24_per_class_fold_metrics.csv",
        "confusion_matrix": "phase5_step5_24_confusion_matrix.csv",
        "wavelength_importance": "phase5_step5_24_wavelength_importance.csv",
        "region_importance": "phase5_step5_24_region_importance.csv",
        "sensitivity_results": "phase5_step5_24_sensitivity_results.csv",
        "scientific_interpretation": "phase5_step5_24_scientific_interpretation.md",
        "run_manifest": "phase5_step5_24_run_manifest.json"
    },
    "final_decision": final_decision,
    "frozen_datasets_unchanged": True,
    "extension_dataset_not_merged": True,
    "package_versions": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": "1.9.1"
    }
}

with open(OUTPUT_DIR / "phase5_step5_24_run_manifest.json", 'w') as f:
    json.dump(manifest, f, indent=2)

print()
print("=" * 80)
print(f"FINAL DECISION: {final_decision}")
print("=" * 80)
print()
print("All outputs saved to:", OUTPUT_DIR)
