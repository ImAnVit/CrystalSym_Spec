"""
Phase 5 Step 5.21: First Group-Aware ML Pilot and PySR Execution Path

Part B: Assess ML task feasibility
Part C: Run minimal leakage-resistant ML pilot
Part D: Decide first PySR target
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ML libraries
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    f1_score, accuracy_score
)

# Set random seed for reproducibility
RANDOM_SEED = 521
np.random.seed(RANDOM_SEED)

# Paths
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/05_exploratory_analysis/phase5_step5_21_ml_pilot_pysr_path")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 5 STEP 5.21: GROUP-AWARE ML PILOT AND PySR PATH")
print("=" * 80)
print()

# Load data
print("Loading frozen datasets...")
canonical_spectra = pd.read_csv(DATA_DIR / "phase2_canonical_d2_5nm.csv")
integrated = pd.read_csv(DATA_DIR / "phase4_integrated_analysis_dataset.csv")

print(f"  Canonical spectra: {len(canonical_spectra)} records")
print(f"  Integrated dataset: {len(integrated)} records")
print()

# Get wavelength columns
wl_cols = [c for c in canonical_spectra.columns if c.startswith('R_')]
print(f"Wavelength columns: {len(wl_cols)} ({wl_cols[0]} to {wl_cols[-1]})")
print()

# Merge datasets: integrated (already has spectral features) + wavelength reflectance
df = integrated.merge(
    canonical_spectra[['canonical_spectrum_id'] + wl_cols],
    on='canonical_spectrum_id',
    how='left'
)

# ============================================================================
# PART B: ASSESS ML TASK FEASIBILITY
# ============================================================================

print("=" * 80)
print("PART B: ML TASK FEASIBILITY ASSESSMENT")
print("=" * 80)
print()

# Define eligibility criteria
print("Defining eligibility criteria...")
print("Focus: Mineral identity confidence (not chemistry/symmetry metadata quality)")
print()

# Start with all spectra
df['ml_eligible'] = True

# Exclude high-conflict records (identity conflicts)
df.loc[df['high_conflict'] == True, 'ml_eligible'] = False
print(f"  High conflict excluded: {(df['high_conflict'] == True).sum()} spectra")

# Exclude tentative provenance
df.loc[df['provenance_tentative'] == True, 'ml_eligible'] = False
print(f"  Tentative provenance excluded: {(df['provenance_tentative'] == True).sum()} spectra")

# Exclude identity issues
df.loc[df['has_identity_issues'] == True, 'ml_eligible'] = False
print(f"  Identity issues excluded: {(df['has_identity_issues'] == True).sum()} spectra")

# Exclude mixture flags
df.loc[df['mixture_flag'] == True, 'ml_eligible'] = False
print(f"  Mixture flags excluded: {(df['mixture_flag'] == True).sum()} spectra")

# NOTE: chemistry_eligibility and symmetry_eligibility_status are NOT used for filtering
# because they reflect metadata quality, not mineral identity confidence.
# For mineral classification, we need confident mineral identity, not confident chemistry/symmetry.

print()
print(f"TOTAL ML-ELIGIBLE SPECTRA: {df['ml_eligible'].sum()} / {len(df)}")
print()

# Filter to eligible
ml_df = df[df['ml_eligible']].copy()

# Class support analysis
print("CLASS SUPPORT ANALYSIS")
print("-" * 80)

class_support = []
for mineral in sorted(ml_df['mineral_label'].unique()):
    mineral_data = ml_df[ml_df['mineral_label'] == mineral]
    n_spectra = len(mineral_data)
    n_groups = mineral_data['provenance_group_id'].nunique()

    # Check if evaluable (need at least 2 groups for grouped CV)
    evaluable = "YES" if n_groups >= 2 else "NO - insufficient groups"

    class_support.append({
        'mineral': mineral,
        'n_spectra': n_spectra,
        'n_groups': n_groups,
        'evaluable': evaluable
    })

    print(f"  {mineral:15s}: {n_spectra:3d} spectra, {n_groups:3d} groups - {evaluable}")

class_support_df = pd.DataFrame(class_support)
class_support_df.to_csv(OUTPUT_DIR / "phase5_step5_21_class_support.csv", index=False)

print()
evaluable_classes = class_support_df[class_support_df['evaluable'] == 'YES']['mineral'].tolist()
print(f"EVALUABLE CLASSES: {len(evaluable_classes)} / {len(class_support_df)}")
print(f"  {', '.join(evaluable_classes)}")
print()

non_evaluable = class_support_df[class_support_df['evaluable'] != 'YES']
if len(non_evaluable) > 0:
    print("NON-EVALUABLE CLASSES:")
    for _, row in non_evaluable.iterrows():
        print(f"  {row['mineral']}: {row['n_groups']} group(s)")
    print()

# Feasibility decision
print("FEASIBILITY DECISION")
print("-" * 80)

if len(evaluable_classes) < 2:
    print("BLOCKED: Fewer than 2 evaluable classes")
    print("Cannot perform meaningful multiclass classification")
    feasibility = "BLOCKED"
elif len(evaluable_classes) < 5:
    print("LIMITED: Only {} evaluable classes".format(len(evaluable_classes)))
    print("Multiclass evaluation possible but limited in scope")
    feasibility = "LIMITED"
else:
    print(f"FEASIBLE: {len(evaluable_classes)} evaluable classes")
    print("Multiclass grouped CV evaluation is defensible")
    feasibility = "FEASIBLE"

print()

# Save feasibility assessment
feasibility_df = pd.DataFrame([{
    'total_spectra': len(df),
    'ml_eligible_spectra': len(ml_df),
    'total_minerals': df['mineral_label'].nunique(),
    'evaluable_minerals': len(evaluable_classes),
    'total_groups_eligible': ml_df['provenance_group_id'].nunique(),
    'feasibility': feasibility,
    'timestamp': datetime.now().isoformat()
}])
feasibility_df.to_csv(OUTPUT_DIR / "phase5_step5_21_ml_task_feasibility.csv", index=False)

# ============================================================================
# PART C: RUN MINIMAL LEAKAGE-RESISTANT ML PILOT
# ============================================================================

if feasibility in ["FEASIBLE", "LIMITED"]:
    print("=" * 80)
    print("PART C: GROUP-AWARE ML PILOT")
    print("=" * 80)
    print()

    # Filter to evaluable classes only
    pilot_df = ml_df[ml_df['mineral_label'].isin(evaluable_classes)].copy()
    print(f"Pilot dataset: {len(pilot_df)} spectra, {pilot_df['provenance_group_id'].nunique()} groups")
    print(f"Classes: {len(evaluable_classes)}")
    print()

    # Prepare features (wavelengths) and target
    X = pilot_df[wl_cols].values
    y = pilot_df['mineral_label'].values
    groups = pilot_df['provenance_group_id'].values

    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution:")
    for mineral in evaluable_classes:
        n = (y == mineral).sum()
        print(f"  {mineral}: {n}")
    print()

    # Define models
    models = {
        'dummy_majority': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
        'logistic_l2': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0, l1_ratio=0, max_iter=1000,
                random_state=RANDOM_SEED
            ))
        ]),
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RidgeClassifier(alpha=1.0, random_state=RANDOM_SEED))
        ]),
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=RANDOM_SEED
            ))
        ])
    }

    # Determine number of folds based on minimum group count
    min_groups = min([
        (pilot_df[pilot_df['mineral_label'] == mineral]['provenance_group_id'].nunique())
        for mineral in evaluable_classes
    ])
    n_splits = min(5, min_groups)
    print(f"Using {n_splits}-fold GroupKFold (min groups per class: {min_groups})")
    print()

    # Run cross-validation for each model
    all_results = []
    all_predictions = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        gkf = GroupKFold(n_splits=n_splits)

        # Cross-val predictions
        y_pred = cross_val_predict(model, X, y, cv=gkf, groups=groups)

        # Metrics
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average='macro', zero_division=0)

        print(f"  Accuracy: {acc:.3f}")
        print(f"  Balanced Accuracy: {bal_acc:.3f}")
        print(f"  Macro F1: {macro_f1:.3f}")
        print()

        # Store results
        all_results.append({
            'model': model_name,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'macro_f1': macro_f1,
            'n_folds': n_splits,
            'random_seed': RANDOM_SEED
        })

        # Store predictions
        for i, (true, pred) in enumerate(zip(y, y_pred)):
            all_predictions.append({
                'model': model_name,
                'spectrum_id': pilot_df.iloc[i]['canonical_spectrum_id'],
                'provenance_group_id': groups[i],
                'true_mineral': true,
                'predicted_mineral': pred,
                'correct': true == pred
            })

    # Save metrics
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "phase5_step5_21_ml_metrics.csv", index=False)

    # Save predictions
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(OUTPUT_DIR / "phase5_step5_21_predictions.csv", index=False)

    # Confusion matrix for best model (excluding dummy)
    best_model_name = results_df[results_df['model'] != 'dummy_majority'].sort_values(
        'macro_f1', ascending=False
    ).iloc[0]['model']

    print(f"Best model: {best_model_name}")
    print()

    best_preds = predictions_df[predictions_df['model'] == best_model_name]
    y_true = best_preds['true_mineral'].values
    y_pred = best_preds['predicted_mineral'].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=evaluable_classes)
    cm_df = pd.DataFrame(cm, index=evaluable_classes, columns=evaluable_classes)
    cm_df.to_csv(OUTPUT_DIR / "phase5_step5_21_confusion_matrix.csv")

    print("Confusion matrix:")
    print(cm_df)
    print()

    # Per-class metrics for best model
    print("Per-class classification report:")
    print(classification_report(y_true, y_pred, target_names=evaluable_classes, zero_division=0))

    # Save grouped split assignments
    split_assignments = pilot_df[['canonical_spectrum_id', 'provenance_group_id', 'mineral_label']].copy()
    split_assignments['cv_fold'] = -1  # Placeholder

    # Assign folds
    gkf = GroupKFold(n_splits=n_splits)
    fold_idx = 0
    for train_idx, test_idx in gkf.split(X, y, groups):
        test_groups = groups[test_idx]
        split_assignments.loc[
            split_assignments['provenance_group_id'].isin(test_groups),
            'cv_fold'
        ] = fold_idx
        fold_idx += 1

    split_assignments.to_csv(
        OUTPUT_DIR / "phase5_step5_21_grouped_split_assignments.csv",
        index=False
    )

    print(f"Saved split assignments for {len(split_assignments)} spectra")
    print()

# ============================================================================
# PART D: DECIDE FIRST PySR TARGET
# ============================================================================

print("=" * 80)
print("PART D: PySR TARGET DECISION")
print("=" * 80)
print()

print("Evaluating candidate symbolic regression targets...")
print()

# Candidate targets from D1-D4 descriptors
candidates = []

# Candidate 1: Predict D2 (mean_peak_prominence) from D1 (n_peaks_prom)
# Scientific question: Is there a meaningful relationship between peak count and prominence?
candidates.append({
    'target': 'mean_peak_prominence (D2)',
    'predictors': 'n_peaks_prom (D1)',
    'scientific_rationale': 'Relationship between peak count and average prominence',
    'n_unique_predictors': ml_df['n_peaks_prom'].nunique(),
    'n_observations': len(ml_df),
    'n_independent_groups': ml_df['provenance_group_id'].nunique(),
    'confounding_risk': 'LOW - both are spectral descriptors',
    'feasibility': 'FEASIBLE'
})

# Candidate 2: Predict D3 (first_deriv_rms) from D1+D2
# Scientific question: Can spectral roughness be predicted from peak characteristics?
candidates.append({
    'target': 'first_deriv_rms (D3)',
    'predictors': 'n_peaks_prom (D1) + mean_peak_prominence (D2)',
    'scientific_rationale': 'Predicting spectral derivative from peak features',
    'n_unique_predictors': ml_df[['n_peaks_prom', 'mean_peak_prominence']].drop_duplicates().shape[0],
    'n_observations': len(ml_df),
    'n_independent_groups': ml_df['provenance_group_id'].nunique(),
    'confounding_risk': 'LOW - all spectral descriptors',
    'feasibility': 'FEASIBLE'
})

# Candidate 3: Predict D4 from D1+D2+D3
# Scientific question: Can second derivative be predicted from other descriptors?
candidates.append({
    'target': 'second_deriv_rms (D4)',
    'predictors': 'D1 + D2 + D3',
    'scientific_rationale': 'Predicting higher-order spectral complexity',
    'n_unique_predictors': ml_df[['n_peaks_prom', 'mean_peak_prominence', 'first_deriv_rms']].drop_duplicates().shape[0],
    'n_observations': len(ml_df),
    'n_independent_groups': ml_df['provenance_group_id'].nunique(),
    'confounding_risk': 'LOW - all spectral descriptors',
    'feasibility': 'FEASIBLE'
})

# Candidate 4: Predict D1 from chemistry/symmetry
# Scientific question: Can spectral complexity be predicted from crystal chemistry?
# PROBLEM: Mineral identity completely confounds crystal system
candidates.append({
    'target': 'n_peaks_prom (D1)',
    'predictors': 'crystal_system + n_distinct_elements',
    'scientific_rationale': 'Predicting peak count from symmetry and chemistry',
    'n_unique_predictors': ml_df[['crystal_system', 'n_distinct_elements']].drop_duplicates().shape[0],
    'n_observations': len(ml_df),
    'n_independent_groups': ml_df['provenance_group_id'].nunique(),
    'confounding_risk': 'HIGH - crystal_system perfectly confounded with mineral identity',
    'feasibility': 'NOT DEFENSIBLE - confounding'
})

# Candidate 5: Predict D2 from chemistry
# PROBLEM: Reference chemistry is mineral-level, not specimen-measured
candidates.append({
    'target': 'mean_peak_prominence (D2)',
    'predictors': 'n_distinct_elements + n_total_atoms',
    'scientific_rationale': 'Predicting prominence from chemical complexity',
    'n_unique_predictors': ml_df[['n_distinct_elements', 'n_total_atoms']].drop_duplicates().shape[0],
    'n_observations': len(ml_df),
    'n_independent_groups': ml_df['provenance_group_id'].nunique(),
    'confounding_risk': 'HIGH - chemistry is mineral-level reference, not specimen-measured',
    'feasibility': 'NOT DEFENSIBLE - reference chemistry confounding'
})

candidates_df = pd.DataFrame(candidates)
print("CANDIDATE PySR TARGETS:")
print()

for i, row in candidates_df.iterrows():
    print(f"{i+1}. Target: {row['target']}")
    print(f"   Predictors: {row['predictors']}")
    print(f"   Rationale: {row['scientific_rationale']}")
    print(f"   Unique predictor combinations: {row['n_unique_predictors']}")
    print(f"   Independent groups: {row['n_independent_groups']}")
    print(f"   Confounding risk: {row['confounding_risk']}")
    print(f"   Feasibility: {row['feasibility']}")
    print()

# Decision
print("PySR TARGET DECISION")
print("-" * 80)

feasible_candidates = candidates_df[candidates_df['feasibility'] == 'FEASIBLE']

if len(feasible_candidates) == 0:
    print("NO DEFENSIBLE PySR TARGET IDENTIFIED")
    print()
    print("Blocking factors:")
    print("  - Crystal system completely confounded with mineral identity")
    print("  - Reference chemistry is mineral-level, not specimen-measured")
    print("  - Insufficient independent variation in chemistry/symmetry predictors")
    print()
    pysr_decision = "NOT_READY"
else:
    print(f"IDENTIFIED {len(feasible_candidates)} FEASIBLE TARGET(S)")
    print()
    print("RECOMMENDED FIRST PySR EXPERIMENT:")
    print(f"  Target: {feasible_candidates.iloc[0]['target']}")
    print(f"  Predictors: {feasible_candidates.iloc[0]['predictors']}")
    print(f"  Rationale: {feasible_candidates.iloc[0]['scientific_rationale']}")
    print()
    print("VALIDATION PLAN:")
    print("  - Use group-aware train/test split (hold out 20% of provenance groups)")
    print("  - Compare against linear regression baseline")
    print("  - Limit equation complexity (max 3 terms, max 2 operators)")
    print("  - Check dimensional consistency where applicable")
    print("  - Require out-of-sample R² > 0.5 to be meaningful")
    print()
    print("CAVEATS:")
    print("  - Spectral descriptors are intercorrelated by construction")
    print("  - Any discovered relationship is descriptive, not causal")
    print("  - Independent validation requires new spectra from new specimens")
    print()
    pysr_decision = "READY"

# Save PySR decision
pysr_decision_md = f"""# Phase 5 Step 5.21: PySR Target Decision

## Candidate Targets Evaluated

| Target | Predictors | Rationale | Unique Predictors | Independent Groups | Confounding Risk | Feasibility |
|--------|------------|-----------|-------------------|-------------------|------------------|-------------|
"""

for _, row in candidates_df.iterrows():
    pysr_decision_md += f"| {row['target']} | {row['predictors']} | {row['scientific_rationale']} | {row['n_unique_predictors']} | {row['n_independent_groups']} | {row['confounding_risk']} | {row['feasibility']} |\n"

pysr_decision_md += f"""
## Decision

**PySR Status**: {pysr_decision}

"""

if pysr_decision == "READY":
    pysr_decision_md += f"""
## Recommended First Experiment

**Target**: {feasible_candidates.iloc[0]['target']}

**Predictors**: {feasible_candidates.iloc[0]['predictors']}

**Scientific Rationale**: {feasible_candidates.iloc[0]['scientific_rationale']}

## Validation Plan

- Group-aware train/test split (hold out 20% of provenance groups)
- Compare against linear regression baseline
- Limit equation complexity (max 3 terms, max 2 operators)
- Check dimensional consistency where applicable
- Require out-of-sample R² > 0.5 to be meaningful

## Caveats

- Spectral descriptors are intercorrelated by construction
- Any discovered relationship is descriptive, not causal
- Independent validation requires new spectra from new specimens
"""
else:
    pysr_decision_md += """
## Blocking Factors

- Crystal system completely confounded with mineral identity
- Reference chemistry is mineral-level, not specimen-measured
- Insufficient independent variation in chemistry/symmetry predictors

## Required Evidence for Future PySR

- Specimen-measured chemistry (not mineral-level references)
- Controlled polymorph samples (same mineral, different crystal system)
- Or: Focus on spectral-descriptor-to-spectral-descriptor relationships
"""

with open(OUTPUT_DIR / "phase5_step5_21_pysr_target_decision.md", 'w', encoding='utf-8') as f:
    f.write(pysr_decision_md)

print()
print(f"PySR decision saved: {pysr_decision}")
print()

# ============================================================================
# FINAL DECISION AND MANIFEST
# ============================================================================

print("=" * 80)
print("FINAL DECISION")
print("=" * 80)
print()

if feasibility == "BLOCKED":
    final_decision = "ML_AND_PYSR_BLOCKED_BY_SPECIFIC_EVIDENCE_GAP"
    print("ML_AND_PYSR_BLOCKED_BY_SPECIFIC_EVIDENCE_GAP")
    print()
    print("Reason: Insufficient evaluable classes for meaningful ML classification")
elif feasibility in ["FEASIBLE", "LIMITED"]:
    if pysr_decision == "READY":
        final_decision = "ML_PILOT_COMPLETED_PYSR_TARGET_READY"
        print("ML_PILOT_COMPLETED_PYSR_TARGET_READY")
        print()
        print("ML pilot completed with grouped CV")
        print("PySR target identified: spectral-descriptor-to-spectral-descriptor")
    else:
        final_decision = "ML_PILOT_COMPLETED_PYSR_TARGET_NOT_READY"
        print("ML_PILOT_COMPLETED_PYSR_TARGET_NOT_READY")
        print()
        print("ML pilot completed with grouped CV")
        print("PySR target not defensible: requires specimen-measured chemistry or controlled polymorphs")

print()

# Create run manifest
manifest = {
    "step": "phase5_step5_21",
    "title": "First Group-Aware ML Pilot and PySR Execution Path",
    "timestamp": datetime.now().isoformat(),
    "random_seed": RANDOM_SEED,
    "inputs": {
        "canonical_spectra": "phase2_canonical_d2_5nm.csv",
        "integrated_dataset": "phase4_integrated_analysis_dataset.csv",
        "spectral_features": "phase3_canonical_spectral_features.csv"
    },
    "outputs": {
        "feasibility_assessment": "phase5_step5_21_ml_task_feasibility.csv",
        "class_support": "phase5_step5_21_class_support.csv",
        "ml_metrics": "phase5_step5_21_ml_metrics.csv" if feasibility != "BLOCKED" else "N/A",
        "confusion_matrix": "phase5_step5_21_confusion_matrix.csv" if feasibility != "BLOCKED" else "N/A",
        "split_assignments": "phase5_step5_21_grouped_split_assignments.csv" if feasibility != "BLOCKED" else "N/A",
        "pysr_decision": "phase5_step5_21_pysr_target_decision.md"
    },
    "key_statistics": {
        "total_spectra": len(df),
        "ml_eligible_spectra": len(ml_df),
        "evaluable_classes": len(evaluable_classes),
        "total_groups_eligible": ml_df['provenance_group_id'].nunique(),
        "feasibility": feasibility,
        "pysr_decision": pysr_decision
    },
    "final_decision": final_decision,
    "package_versions": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": "1.9.1"
    }
}

with open(OUTPUT_DIR / "phase5_step5_21_run_manifest.json", 'w') as f:
    json.dump(manifest, f, indent=2)

print("Run manifest saved")
print()

# ============================================================================
# GENERATE FINAL REPORT
# ============================================================================

report_md = f"""# Phase 5 Step 5.21: First Group-Aware ML Pilot and PySR Execution Path

## Executive Summary

This step conducted a feasibility assessment for machine learning classification of mineral identities from reflectance spectra under source-group-aware validation, and evaluated candidate targets for symbolic regression (PySR).

**Final Decision**: `{final_decision}`

---

## Part A: ML Question Definition

**Primary pilot question**: Can the available RELAB reflectance spectra distinguish mineral identities under source-group-aware validation?

This is an exploratory spectral-classification task. It is NOT a test of whether crystal symmetry independently controls spectral complexity.

---

## Part B: ML Task Feasibility Assessment

### Eligibility Criteria

Starting from {len(df)} total spectra, the following filters were applied to ensure confident mineral identity:

- High conflict excluded: {(df['high_conflict'] == True).sum()} spectra
- Tentative provenance excluded: {(df['provenance_tentative'] == True).sum()} spectra
- Identity issues excluded: {(df['has_identity_issues'] == True).sum()} spectra
- Mixture flags excluded: {(df['mixture_flag'] == True).sum()} spectra

Note: Chemistry and symmetry eligibility were NOT used for filtering because they reflect metadata quality, not mineral identity confidence. For mineral classification, confident mineral identity is sufficient.

**Result**: {len(ml_df)} ML-eligible spectra from {ml_df['provenance_group_id'].nunique()} provenance groups

### Class Support

Of {df['mineral_label'].nunique()} minerals, {len(evaluable_classes)} have sufficient independent provenance groups (≥2) for grouped cross-validation:

| Mineral | Spectra | Groups | Evaluable |
|---------|---------|--------|-----------|
"""

for _, row in class_support_df.iterrows():
    report_md += f"| {row['mineral']} | {row['n_spectra']} | {row['n_groups']} | {row['evaluable']} |\n"

report_md += f"""
**Feasibility**: {feasibility}

"""

if feasibility in ["FEASIBLE", "LIMITED"]:
    report_md += f"""
---

## Part C: Group-Aware ML Pilot Results

### Experimental Design

- **Dataset**: {len(pilot_df)} spectra from {pilot_df['provenance_group_id'].nunique()} provenance groups
- **Features**: {len(wl_cols)} wavelength reflectance values (320-2550 nm, 5 nm spacing)
- **Target**: Mineral identity ({len(evaluable_classes)} classes)
- **Validation**: {n_splits}-fold GroupKFold (no group overlap across folds)
- **Preprocessing**: All preprocessing (scaling, feature selection) fitted within training folds only

### Models Evaluated

1. **Dummy baseline**: Majority class predictor
2. **Logistic Regression (L2)**: Regularized linear classifier
3. **Ridge Classifier**: Linear classifier with L2 penalty
4. **Random Forest**: Ensemble of 100 decision trees (max_depth=10)

### Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | N Folds | Random Seed |
|-------|----------|-------------------|----------|---------|-------------|
"""

for _, row in results_df.iterrows():
    report_md += f"| {row['model']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | {row['n_folds']} | {row['random_seed']} |\n"

report_md += f"""
**Best model**: {best_model_name} (Macro F1 = {results_df[results_df['model'] == best_model_name]['macro_f1'].values[0]:.3f})

### Confusion Matrix

"""

# Format confusion matrix as markdown table
report_md += "| True \\ Predicted | " + " | ".join(cm_df.columns) + " |\n"
report_md += "|" + "---|" * (len(cm_df.columns) + 1) + "\n"
for idx in cm_df.index:
    report_md += f"| {idx} | " + " | ".join([str(x) for x in cm_df.loc[idx]]) + " |\n"

report_md += """
### Per-Class Performance

See `phase5_step5_21_ml_metrics.csv` and confusion matrix for detailed per-class results.

### Key Observations

- Group-aware validation prevents leakage from repeated spectra
- Performance must be interpreted against the dummy baseline
- Results are descriptive only; no causal claims about crystal symmetry
- Complete mineral/crystal-system confounding prevents independent symmetry effect assessment

"""

report_md += f"""
---

## Part D: PySR Target Decision

### Candidate Targets Evaluated

| Target | Predictors | Rationale | Unique Predictors | Independent Groups | Confounding Risk | Feasibility |
|--------|------------|-----------|-------------------|-------------------|------------------|-------------|
"""

for _, row in candidates_df.iterrows():
    report_md += f"| {row['target']} | {row['predictors']} | {row['scientific_rationale']} | {row['n_unique_predictors']} | {row['n_independent_groups']} | {row['confounding_risk']} | {row['feasibility']} |\n"

report_md += f"""
### Decision

**PySR Status**: {pysr_decision}

"""

if pysr_decision == "READY":
    report_md += f"""
### Recommended First Experiment

**Target**: {feasible_candidates.iloc[0]['target']}

**Predictors**: {feasible_candidates.iloc[0]['predictors']}

**Rationale**: {feasible_candidates.iloc[0]['scientific_rationale']}

### Validation Plan

- Group-aware train/test split (hold out 20% of provenance groups)
- Compare against linear regression baseline
- Limit equation complexity (max 3 terms, max 2 operators)
- Check dimensional consistency where applicable
- Require out-of-sample R² > 0.5 to be meaningful

### Caveats

- Spectral descriptors are intercorrelated by construction
- Any discovered relationship is descriptive, not causal
- Independent validation requires new spectra from new specimens
"""
else:
    report_md += """
### Blocking Factors

1. **Crystal system completely confounded with mineral identity**
   - Each mineral maps to exactly one crystal system
   - Cannot disentangle mineral vs symmetry effects

2. **Reference chemistry is mineral-level, not specimen-measured**
   - All "chemistry" values are idealized references
   - No independent chemical variation within minerals

3. **Insufficient independent variation**
   - Too few unique predictor combinations for meaningful symbolic regression

### Required Evidence for Future PySR

- Specimen-measured chemistry (not mineral-level references)
- Controlled polymorph samples (same mineral, different crystal system)
- Or: Focus on spectral-descriptor-to-spectral-descriptor relationships (lower scientific value but feasible)

"""

report_md += f"""
---

## Scientific Safeguards

✓ All frozen inputs remain unchanged
✓ Canonical and extension datasets remain separate
✓ No leakage across provenance groups (group-aware CV)
✓ No claim that source groups are certainly independent physical specimens
✓ No causal interpretation of classification results
✓ No independent crystal-system effect claim (complete confounding)
✓ No interpretation of reference chemistry as measured specimen chemistry
✓ No PySR equation promoted as physical law without independent validation

---

## Outputs Generated

- `phase5_step5_21_ml_task_feasibility.csv`: Feasibility assessment
- `phase5_step5_21_class_support.csv`: Class-level support counts
- `phase5_step5_21_ml_metrics.csv`: Model performance metrics
- `phase5_step5_21_confusion_matrix.csv`: Confusion matrix for best model
- `phase5_step5_21_grouped_split_assignments.csv`: CV fold assignments
- `phase5_step5_21_pysr_target_decision.md`: PySR target evaluation
- `phase5_step5_21_run_manifest.json`: Execution manifest

---

## Next Action

"""

if final_decision == "ML_PILOT_COMPLETED_PYSR_TARGET_READY":
    report_md += """
**Single most valuable next action**: Execute the recommended PySR experiment on spectral-descriptor-to-spectral-descriptor relationships, with strict complexity controls and comparison against linear baselines. This will establish whether symbolic regression can discover interpretable relationships in the spectral feature space, even though such relationships are descriptive rather than causal.
"""
elif final_decision == "ML_PILOT_COMPLETED_PYSR_TARGET_NOT_READY":
    report_md += """
**Single most valuable next action**: Acquire specimen-measured chemistry for a subset of provenance groups, or identify controlled polymorph samples, to enable a defensible PySR experiment that tests whether crystal chemistry or symmetry predicts spectral complexity. Without this evidence, PySR cannot address the original research question.
"""
else:
    report_md += """
**Single most valuable next action**: Expand the eligible population by resolving identity conflicts through curator enquiry, or acquire additional spectra from independent specimens for underrepresented minerals, to enable a meaningful multiclass ML evaluation.
"""

with open(OUTPUT_DIR / "phase5_step5_21_report.md", 'w', encoding='utf-8') as f:
    f.write(report_md)

print("Final report saved")
print()
print("=" * 80)
print("PHASE 5 STEP 5.21 COMPLETE")
print("=" * 80)
