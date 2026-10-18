# Cover Letter Draft -- Computers & Geosciences

**Date:** 2026-10-18

---

Vitaly Andriyko
Independent Researcher, Nanjing, China

andriykovit@gmail.com

2026-10-18

Editor-in-Chief
Computers & Geosciences

Dear Editor,

We submit for consideration the manuscript "What Can Curated Mineral Spectral Libraries Support? A Provenance-Aware Assessment of Dependence Structures, Metadata Gaps, and Inferential Limits in RELAB."

Curated spectral libraries such as RELAB are essential infrastructure for mineral identification, yet their metadata were designed for human curation and spectral matching, not for the requirements of modern machine-learning workflows. Individual spectra are typically treated as independent observations without accounting for provenance structures, the distinction between reference and specimen-specific chemistry, or the dependence relationships among spectra from the same physical source.

We ask a practical question: what can researchers reliably infer from a curated mineral reflectance library when spectral records, provenance, mineral identity, chemistry, and crystallography are incompletely linked? Using the RELAB library (454 spectra, 18 mineral labels, 382 provenance groups), we conducted a 30-field evidence audit, applied group-aware cross-validation to prevent dependence-structure leakage, and explicitly separated what the data can support at different inferential levels.

Our key findings are:
1. Mineral classification is feasible under group-aware validation (macro-F1 = 0.6136 vs dummy baseline 0.0248), but results are within-library and not externally validated.
2. The evidence audit reveals that only descriptive analysis is supported; all gates requiring specimen-specific chemistry or verified independence are closed.
3. The hypothesis that crystal symmetry independently constrains spectral complexity cannot be tested because mineral identity and crystal system are completely confounded, no specimen-specific chemistry exists, and specimen independence is unverified.

We believe this manuscript fits the scope of Computers & Geosciences because it demonstrates the application of established computational methods (group-aware cross-validation, evidence auditing, permutation importance, symbolic regression) to a domain-specific problem where standard practices produce invisible but fundamental inferential limitations. The contribution is not algorithmic novelty but the systematic application of computational rigor to a real-world dataset, revealing what becomes visible when provenance and evidence structure are taken seriously.

The manuscript includes a configurable evidence-auditor tool (Supplementary Code) that demonstrates the framework's applicability to other spectral libraries through configuration. The tool has been tested with synthetic data only; no external library validation has been performed.

This manuscript has not been published previously and is not under consideration elsewhere. All authors have approved the manuscript and agree with its submission to Computers & Geosciences.


Thank you for your consideration.

Sincerely,

Vitaly Andriyko
