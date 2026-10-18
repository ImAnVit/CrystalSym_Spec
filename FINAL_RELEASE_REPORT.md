# Final Release Report

**Project:** CrystalSym-Spec
**Date:** 2026-10-18
**Status:** Ready for submission.

---

## 1. Final Manuscript

| Item | Path | Status |
|------|------|--------|
| Final manuscript | `manuscript_final.md` | Complete |
| Supplementary material | `supplementary_material.md` | Complete |
| Cover letter | `cover_letter.md` | Complete (placeholders remain) |
| Verified references (BibTeX) | `references.bib` | Complete (21 entries) |

The manuscript is a clean publication version derived from v0.7. Internal sections (changelogs, blocker lists, claim-tracking) have been removed. Editorial notes are retained only where genuine verification is required before submission.

---

## 2. Submission Files

### Main-text figures (5, in PNG/PDF/SVG)

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `figures/phase5_step5_25F_main_figures/figure1_dataset_composition.*` | Dataset composition and provenance structure |
| Figure 2 | `figures/phase5_step5_25F_main_figures/figure2_classification_performance.*` | Confusion matrix and per-class F1 |
| Figure 3 | `figures/phase5_step5_25F_main_figures/figure3_support_vs_performance.*` | Class support vs performance scatter |
| Figure 4 | `figures/phase5_step5_25F_main_figures/figure4_symbolic_regression.*` | PySR predicted vs actual (D3/D4) |
| Figure 5 | `figures/phase5_step5_25F_main_figures/figure5_evidence_framework.*` | Evidence framework and analysis gates |

### Supplementary figures (2, in PNG/PDF/SVG)

| Figure | File | Description |
|--------|------|-------------|
| Figure S1 | `figures/supplementary/figureS1_symbolic_regression.*` | Symbolic regression with fold stability |
| Figure S2 | `figures/supplementary/figureS2_region_importance.*` | Regional permutation importance |

### Supplementary material

- `supplementary_material.md` — Supplementary text S1--S5, tables S1--S4, figure captions S1--S2, supplementary code description.

### Data

- `data/processed/phase2_canonical_d2_5nm.csv` — 454 spectra, D2 domain, 5 nm spacing
- `data/processed/phase3_canonical_spectral_features.csv` — 4 spectral descriptors
- `data/processed/phase4_integrated_analysis_dataset.csv` — Integrated analysis dataset
- `metadata/` — RELAB metadata tables and field dictionary

### Software

- `evidence_auditor.py` — Configurable metadata-coverage auditor
- `tests/test_evidence_auditor.py` — 39 synthetic tests
- `example_audit.py` — Reproducible example
- `requirements.txt` — Python dependencies
- `LICENSE` — BSD 3-Clause
- `README.md` — Installation, usage, schema documentation
- `CITATION.cff` — Citation metadata
- `CHANGELOG.md` — Release notes
- `.gitignore` — Excludes RELAB raw data and temporary files

---

## 3. Test Status

```
39 passed in 5.32s
```

All 39 synthetic tests pass. Test categories:
- Complete and missing fields (5 tests)
- Ambiguous and conflicting metadata (5 tests)
- Reference vs specimen chemistry (6 tests)
- Multiple spectra per group (4 tests)
- Missing provenance links (3 tests)
- Gate status (4 tests)
- Report generation (4 tests)
- NOT_OPERATIONALIZED fields (6 tests)
- CSV export (2 tests)

---

## 4. Checks Completed

### Data consistency

| Check | Result |
|-------|--------|
| Canonical dataset: 454 spectra | Verified |
| 18 mineral labels | Verified |
| 382 provenance groups | Verified |
| Classification subset: 309 spectra, 284 groups, 16 classes | Verified |
| Logistic regression macro-F1 = 0.6136 | Verified |
| Dummy baseline macro-F1 = 0.0248 | Verified |
| 10 of 16 classes with F1 >= 0.5 | Verified |
| 11 of 12 regions with negative permutation importance | Verified |

### Figure validation

| Check | Result |
|-------|--------|
| All 5 main figures exist in 3 formats | Verified |
| All 2 supplementary figures generated in 3 formats | Verified |
| Figure data matches source CSVs | Verified (from prior audit) |
| Colorblind accessibility | Verified (from prior audit) |

### Reference consistency

| Check | Result |
|-------|--------|
| All in-text citations have matching BibTeX entries | Verified |
| Balmer/Fasnacht error corrected | Fixed |
| Jahoda author list expanded to 6 authors | Fixed |
| Duplicate Hao entry removed | Fixed |
| Unused BibTeX entries remain (not cited in manuscript) | Noted |

### Manuscript consistency

| Check | Result |
|-------|--------|
| Sample counts match frozen data | Verified |
| Table 1 matches descriptor summary CSV | Verified |
| Tables 2--3 match classification metrics CSVs | Verified |
| Table 4 matches sensitivity results CSV | Verified |
| No unsupported novelty claims | Verified |
| No causal claims | Verified |
| No external validation claims | Verified |
| Within-library framing throughout | Verified |
| Limitations section complete | Verified |

---

## 5. Remaining Blockers

| ID | Blocker | Severity | Blocks submission? |
|----|---------|----------|--------------------|
| B1 | RELAB acknowledgement wording unverified against RELAB data-use policy | MODERATE | No — can submit with editorial note, but should verify before camera-ready |
| B2 | Fasnacht et al. (2019) DOI not verified (authors and article number confirmed) | MINOR | No — abstract-level verification is sufficient for initial submission |
| B3 | Code repository URL not determined | MINOR | No — can add at submission time |
| B4 | Author names, affiliations, funding, acknowledgements not provided | MODERATE | Yes — required for submission |
| B5 | PySR version not recorded in run manifests | MINOR | No — noted in manuscript |
| B6 | Normalization discrepancy: manuscript says "unit area" but phase2 report says "none applied" | LOW | No — may refer to different pipeline stages; verify before camera-ready |

---

## 6. Actions Required Before Submission

### Required (author must perform)

1. **Complete author metadata.** Fill in author names, affiliations, corresponding author email, funding statement, and author contributions in `manuscript_final.md`, `cover_letter.md`, `CITATION.cff`, and `supplementary_material.md`.

2. **Verify RELAB acknowledgement.** Check the acknowledgement text in Section 7 of `manuscript_final.md` against RELAB's current data-use policy. Contact Dr. T. Hiroi or check the RELAB website if necessary.

3. **Create GitHub repository.** Initialize a git repository, add all files, commit, and push to GitHub. Update `manuscript_final.md` (Data Availability), `cover_letter.md`, `CITATION.cff`, and `CHANGELOG.md` with the repository URL.

4. **Suggest reviewers.** Fill in 2--3 suggested reviewers in `cover_letter.md` with name, affiliation, email, and expertise.

5. **Verify Fasnacht DOI.** Confirm the DOI for Fasnacht et al. (2019) *Scientific Data* 6:268 via the publisher's website. Update `references.bib` with the verified DOI.

### Recommended (before camera-ready)

6. **Resolve normalization discrepancy.** Verify whether unit-area normalization was applied during the classification pipeline (Section 2.4) or not. If not, correct the manuscript text.

7. **Full-text reference check.** Verify Jahoda et al. (2021) and Király and Tóth (2025) via full text, not just abstracts.

8. **Convert supplementary material to PDF.** The journal likely requires a single PDF for supplementary material. Convert `supplementary_material.md` to PDF with figures embedded.

9. **Format manuscript per journal guidelines.** Convert `manuscript_final.md` to the journal's required format (likely LaTeX or Word). Check *Computers & Geosciences* author instructions for specific requirements.

10. **Remove the wrongly-named `supplementary_material.pdf`** file (contains markdown text with .pdf extension from an earlier error).

---

## 7. Novelty and Scope Risk (Honest Assessment)

The manuscript's contribution is **incremental in computational novelty** but **substantial in domain-specific rigor**:

- Group-aware cross-validation is established practice, not novel.
- Evidence auditing follows standard data-quality principles.
- The contribution is the systematic application of these principles to a real spectral library, revealing what becomes visible.

**Risk for *Computers & Geosciences*:** The paper is predominantly a "negative results" paper (what the data cannot support). This may be seen as insufficiently novel for the journal. However, the practical implications for spectral library practice are significant, and the evidence-auditor tool provides a concrete computational contribution.

**Mitigation:** The cover letter frames the contribution as orthogonal to prior ML-for-minerals work: not "can we classify?" but "what does defensible classification require?"

---

## 8. Preserved Provenance

All frozen datasets, historical versions, analysis scripts, and result documents are preserved in their original locations:

- `data/processed/` — All frozen canonical datasets
- `data/raw/RELAB/` — Raw RELAB spectra (not for distribution)
- `results/` — Complete audit trail from Phase 1 through Phase 5
- `scripts/` — All analysis scripts
- `metadata/` — RELAB metadata tables and field dictionary
- `docs/` — Methodology and research decision logs
- `results/05_exploratory_analysis/phase5_step5_25_manuscript_readiness/` — All manuscript drafts (v0.2--v0.7), audit reports, discrepancy resolutions

No research provenance has been deleted.

---

**End of Final Release Report**
