# CrystalSym-Spec: Provenance-Aware Assessment of Mineral Spectral Library Inferential Limits

A reproducible assessment of what curated mineral spectral libraries can and cannot support for inferential analysis, using the RELAB library as a case study.

## Overview

This repository accompanies the manuscript:

> [Vitaly Andriyko]. *What Can Curated Mineral Spectral Libraries Support? A Provenance-Aware Assessment of Dependence Structures, Metadata Gaps, and Inferential Limits in RELAB.* Computers & Geosciences [in preparation].

The project demonstrates that standard ML workflows applied to curated spectral libraries can produce invisible but fundamental inferential limitations when provenance structures, metadata gaps, and dependence relationships are not accounted for.

**Key results:**
- 454 RELAB spectra (18 minerals, 382 provenance groups) assembled into a frozen canonical dataset
- 30-field evidence audit quantifying metadata completeness
- Group-aware cross-validation achieving macro-F1 = 0.6136 for 16-class mineral classification
- Crystal symmetry hypothesis shown to be untestable with current data (complete confounding, zero specimen-specific chemistry)

## Repository Structure

```
crystalsym-spec/
  manuscript_final.md          Final manuscript (publication version)
  supplementary_material.md    Supplementary text, tables, figures
  cover_letter.md              Cover letter for journal submission
  evidence_auditor.py          Configurable metadata-coverage auditor
  example_audit.py             Reproducible example with synthetic data
  requirements.txt             Python dependencies
  LICENSE                      BSD 3-Clause
  CITATION.cff                 Citation metadata
  CHANGELOG.md                 Release notes
  .gitignore
  tests/
    test_evidence_auditor.py   39 synthetic tests
  figures/
    phase5_step5_25F_main_figures/   5 main-text figures (PNG/PDF/SVG)
    supplementary/                    2 supplementary figures (PNG/PDF/SVG)
  data/
    processed/                 Frozen canonical datasets and derived tables
    raw/RELAB/                 Raw RELAB spectra (NOT included in release)
  metadata/                    RELAB metadata tables and field dictionary
  scripts/                     Analysis pipeline scripts (Phase 1--5)
  results/                     Frozen analysis results and audit reports
  config/                      Pipeline configuration
  docs/                        Methodology and research decision logs
```

## Installation

```bash
# Clone the repository
git clone [repository URL]
cd crystalsym-spec

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

Requires Python >= 3.10.

## Evidence Auditor Tool

The evidence auditor is a configurable metadata-coverage tool for mineral spectral libraries. It accepts a library's metadata table and a field-mapping configuration, then produces an evidence coverage report against a 30-field, 9-category framework.

### Quick Start

```bash
python example_audit.py
```

This runs a minimal audit on synthetic data and prints a Markdown report.

### Usage

```python
import pandas as pd
from evidence_auditor import AuditConfig, FieldMapping, audit_evidence, generate_report

# 1. Prepare your record table (one row per spectral record)
records = pd.read_csv("your_library_records.csv")

# 2. Prepare your provenance group table
groups = pd.read_csv("your_provenance_groups.csv")

# 3. Configure field mappings
config = AuditConfig(
    library_name="YourLibrary",
    library_version="1.0",
    field_mappings={
        "mineral_name": FieldMapping("your_mineral_column"),
        "specimen_id": FieldMapping("your_sample_id_column"),
        "spectral_range": FieldMapping(["wl_min_col", "wl_max_col"]),
    },
    ambiguous_values={
        "identity_confidence": ["TENTATIVE", "UNCERTAIN"],
    },
)

# 4. Run the audit
result = audit_evidence(records, groups, config)

# 5. View the report
print(generate_report(result))

# 6. Export CSV results
result.export_csvs("output_directory")
```

### Input Schema

| Input | Required columns | Description |
|-------|-----------------|-------------|
| Record table | `record_id`, `mineral_label` | One row per spectral record |
| Group table | `record_id`, `provenance_group_id` | Record-to-group mapping |
| Field mapping | N/A | Dict mapping evidence fields to library columns |

### Output Schema

| Output | Shape | Description |
|--------|-------|-------------|
| `field_coverage` | 30 rows | Per-field coverage statistics |
| `record_evidence` | n_records rows | Per-record evidence status |
| `gate_status` | 4 rows | Analysis gate open/closed status |
| `validation_warnings` | list | Unmapped fields, missing links |

## Testing

```bash
python -m pytest tests/test_evidence_auditor.py -v
```

39 synthetic tests covering: complete/missing fields, ambiguous metadata, reference vs. specimen chemistry, multiple spectra per group, missing provenance links, gate computation, report generation, and CSV export.

## Data

### Frozen Canonical Dataset

The processed canonical dataset is available in `data/processed/`:

| File | Description |
|------|-------------|
| `phase2_canonical_d2_5nm.csv` | 454 spectra, D2 domain, 5 nm spacing |
| `phase3_canonical_spectral_features.csv` | 4 spectral descriptors per spectrum |
| `phase4_integrated_analysis_dataset.csv` | Integrated dataset with chemistry and symmetry metadata |

### RELAB Raw Data

Raw RELAB spectral data (.tab files) are publicly available through the RELAB facility at Brown University. These data are **not included** in this repository due to redistribution restrictions.

## Limitations

- **Single library.** All analyses use RELAB data. Results may not generalise.
- **No external validation.** All classification results are within-library cross-validation.
- **Evidence auditor tested with synthetic data only.** No external library has been audited.
- **Provenance groups are provisional.** True specimen independence is unverified.
- **PySR version unrecorded.** The symbolic regression tool version was not captured in run manifests.

## License

BSD 3-Clause. See `LICENSE`.

## Citation

If you use this software, please cite the associated manuscript:

> [Vitaly Andriyko]. What Can Curated Mineral Spectral Libraries Support? A Provenance-Aware Assessment of Dependence Structures, Metadata Gaps, and Inferential Limits in RELAB. *Computers & Geosciences* [in preparation].

A `CITATION.cff` file is provided for automated citation extraction.
