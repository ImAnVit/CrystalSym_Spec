# Methodology — CrystalSym-Spec

**Last updated:** 2026-09-22

---

## Leakage-control principle

> Future predictive-model evaluation must prevent measurements belonging to the same RELAB sample/specimen group, and exact duplicate spectra, from being distributed across training and evaluation sets.

### Rationale

The RELAB dataset contains:

- ~300 sample IDs with multiple measurements (repeated spectra of the same specimen under different conditions)
- 84 byte-identical duplicate files distributed across mineral directories
- Cross-mineral duplicates where the same physical spectrum appears in two different mineral classes

If measurements from the same specimen or exact duplicate spectra are split across training and test sets, the model will appear to perform better than it actually is. This is a form of data leakage that invalidates performance estimates and undermines the scientific conclusions.

### Implementation principle

When train/test splitting is implemented:

1. Splitting unit = specimen/sample group (not individual measurement)
2. All measurements sharing a `specimen_group_id` must appear in the same fold
3. All measurements sharing an `exact_duplicate_group_id` must appear in the same fold
4. Cross-mineral duplicates require special handling (the same spectrum cannot be simultaneously "Augite" in training and "Hypersthene" in testing)

### Status

No ML splitting has been implemented yet. This principle is established as a binding constraint for all future predictive modeling in this project.

---

## General methodological constraints

1. Raw RELAB data are immutable. No spectral values are altered.
2. Provenance is preserved at every stage.
3. No derived feature is considered validated until its computation is documented and verified.
4. All exclusions are recorded with reasons, never silently dropped.
