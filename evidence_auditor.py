"""
Phase 5 Step 5.25I -- Minimal Reusable Evidence-Audit Prototype.

A configurable metadata-coverage auditor that accepts a spectral library's
metadata table and a field-mapping configuration, then produces an evidence
coverage report against the 30-field, 9-category framework.

This is a minimal prototype for feasibility assessment, not a production
software package. It demonstrates that the CrystalSym-Spec evidence framework
can be generalised beyond RELAB through configurable field mappings.

Inputs:
    1. Record table (CSV): one row per spectral record
    2. Provenance group table (CSV): record-to-group mapping
    3. Field mapping (dict or YAML): library column -> evidence field

Outputs:
    1. Field coverage summary (DataFrame)
    2. Per-record evidence status (DataFrame)
    3. Analysis gate status (DataFrame)
    4. Markdown report (string)

Author: CrystalSym-Spec Step 5.25I
Date: 2026-10-18
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Framework schema (30 fields, 9 categories, 4 tiers)
# ---------------------------------------------------------------------------

TIER_COLUMNS = [
    "tier_descriptive",
    "tier_source_group_comparison",
    "tier_specimen_chemistry_modelling",
    "tier_ml_symbolic_regression",
]

EVIDENCE_SCHEMA: list[dict[str, Any]] = [
    {"evidence_category": "mineral_identity", "evidence_field": "mineral_name",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "mineral_identity", "evidence_field": "identity_confidence",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "mineral_identity", "evidence_field": "identification_method",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "specimen_identifiers", "evidence_field": "specimen_id",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "specimen_identifiers", "evidence_field": "source_group_id",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "source_group_level"},
    {"evidence_category": "specimen_identifiers", "evidence_field": "unique_specimen_identifier",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "provenance", "evidence_field": "source_collection",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "provenance", "evidence_field": "geographic_provenance",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "provenance", "evidence_field": "independence_status",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "source_group_level"},
    {"evidence_category": "provenance", "evidence_field": "n_verified_independent_specimens",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "source_group_level"},
    {"evidence_category": "preparation_treatment", "evidence_field": "grain_size",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "preparation_treatment", "evidence_field": "sample_preparation",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "preparation_treatment", "evidence_field": "thermal_treatment",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "preparation_treatment", "evidence_field": "mixture_status",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "measurement_conditions", "evidence_field": "measurement_geometry",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "measurement_conditions", "evidence_field": "instrument_configuration",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "measurement_conditions", "evidence_field": "calibration_details",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "spectral_properties", "evidence_field": "spectral_range",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "spectral_properties", "evidence_field": "sampling_interval",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "spectral_properties", "evidence_field": "preprocessing_history",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "chemistry", "evidence_field": "specimen_specific_chemistry",
     "tier_descriptive": "NOT_REQUIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "chemistry", "evidence_field": "reference_formula",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "reference_level"},
    {"evidence_category": "chemistry", "evidence_field": "chemistry_evidence_level",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "crystallography", "evidence_field": "crystal_system",
     "tier_descriptive": "REQUIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "reference_level"},
    {"evidence_category": "crystallography", "evidence_field": "space_group",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "REQUIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "reference_level"},
    {"evidence_category": "crystallography", "evidence_field": "unit_cell_parameters",
     "tier_descriptive": "NOT_REQUIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "crystallography", "evidence_field": "crystallographic_evidence_source",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "REQUIRED", "tier_ml_symbolic_regression": "REQUIRED",
     "evidence_level_expected": "specimen_level"},
    {"evidence_category": "uncertainty_metadata", "evidence_field": "measurement_uncertainty",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "uncertainty_metadata", "evidence_field": "descriptor_uncertainty",
     "tier_descriptive": "NOT_REQUIRED", "tier_source_group_comparison": "NOT_REQUIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "record_level"},
    {"evidence_category": "uncertainty_metadata", "evidence_field": "reproducibility_evidence",
     "tier_descriptive": "DESIRED", "tier_source_group_comparison": "DESIRED",
     "tier_specimen_chemistry_modelling": "DESIRED", "tier_ml_symbolic_regression": "DESIRED",
     "evidence_level_expected": "specimen_level"},
]

ANALYSIS_GATES = [
    {
        "gate": "GATE_DESCRIPTIVE",
        "tier": "tier_descriptive",
        "description": "Descriptive analysis of available data",
    },
    {
        "gate": "GATE_SOURCE_GROUP_COMPARISON",
        "tier": "tier_source_group_comparison",
        "description": "Source-group-level comparison",
    },
    {
        "gate": "GATE_SPECIMEN_CHEMISTRY",
        "tier": "tier_specimen_chemistry_modelling",
        "description": "Specimen-specific chemistry or structure modelling",
    },
    {
        "gate": "GATE_INTERPRETABLE_ML",
        "tier": "tier_ml_symbolic_regression",
        "description": "Interpretable ML or symbolic regression",
    },
]

NOT_OPERATIONALIZED_FIELDS: dict[str, dict[str, str]] = {
    "identification_method": {
        "classification": "HUMAN_REVIEW",
        "rationale": "Most libraries do not record how identity was determined; flag if absent but cannot auto-resolve.",
    },
    "unique_specimen_identifier": {
        "classification": "OUT_OF_SCOPE",
        "rationale": "Persistent specimen-level IDs are rare; report as NOT_ASSESSED unless explicitly mapped.",
    },
    "geographic_provenance": {
        "classification": "HUMAN_REVIEW",
        "rationale": "Locality data are inconsistently recorded; flag if absent but cannot auto-verify.",
    },
    "specimen_specific_chemistry": {
        "classification": "AUTOMATED",
        "rationale": "If a column is mapped, coverage is checkable automatically like any other field.",
    },
    "unit_cell_parameters": {
        "classification": "OUT_OF_SCOPE",
        "rationale": "Rarely available in spectral libraries; report as NOT_ASSESSED unless explicitly mapped.",
    },
    "descriptor_uncertainty": {
        "classification": "OUT_OF_SCOPE",
        "rationale": "Not library metadata; it is a derived quantity from computed spectral descriptors.",
    },
}


# ---------------------------------------------------------------------------
# 2. Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldMapping:
    """Maps a library-specific column to an evidence field."""
    source_column: str | list[str]
    transform: str | None = None
    value_map: dict[str, str] | None = None


@dataclass
class AuditConfig:
    """Configuration for an evidence audit run."""
    library_name: str
    library_version: str
    field_mappings: dict[str, FieldMapping]
    ambiguous_values: dict[str, list[str]] = field(default_factory=dict)
    reference_derived_indicators: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class AuditResult:
    """Complete result of an evidence audit."""
    config: AuditConfig
    field_coverage: pd.DataFrame
    record_evidence: pd.DataFrame
    gate_status: pd.DataFrame
    validation_warnings: list[str]
    n_records: int
    n_groups: int
    n_minerals: int

    def export_csvs(self, directory: str | Path) -> dict[str, str]:
        """Write field_coverage, record_evidence, and gate_status to CSV.

        Returns a dict mapping logical name to written file path.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, df in [
            ("field_coverage", self.field_coverage),
            ("record_evidence", self.record_evidence),
            ("gate_status", self.gate_status),
        ]:
            p = directory / f"{name}.csv"
            df.to_csv(p, index=False)
            paths[name] = str(p)
        return paths


# ---------------------------------------------------------------------------
# 3. Core audit engine
# ---------------------------------------------------------------------------

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return s in ("", "na", "NA", "N/A", "n/a", "nan", "NaN", "null", "NULL", "none", "None")


def _classify_evidence_status(
    value: Any,
    field_name: str,
    config: AuditConfig,
) -> str:
    if _is_empty(value):
        return "MISSING"
    ambiguous = config.ambiguous_values.get(field_name, [])
    if str(value).strip() in ambiguous:
        return "AMBIGUOUS"
    ref_indicators = config.reference_derived_indicators.get(field_name, [])
    if str(value).strip() in ref_indicators:
        return "REFERENCE_DERIVED"
    return "PRESENT"


def _resolve_value(row: pd.Series, mapping: FieldMapping) -> Any:
    if isinstance(mapping.source_column, list):
        parts = []
        for col in mapping.source_column:
            if col in row.index and not _is_empty(row[col]):
                parts.append(str(row[col]).strip())
        return "; ".join(parts) if parts else None
    col = mapping.source_column
    if col not in row.index:
        return None
    value = row[col]
    if mapping.transform == "map_values" and mapping.value_map is not None:
        return mapping.value_map.get(str(value).strip(), value)
    return value


def audit_evidence(
    records: pd.DataFrame,
    provenance_groups: pd.DataFrame,
    config: AuditConfig,
) -> AuditResult:
    schema_df = pd.DataFrame(EVIDENCE_SCHEMA)
    warnings: list[str] = []

    mapped_fields = set(config.field_mappings.keys())
    unmapped = set(schema_df["evidence_field"]) - mapped_fields
    for f in sorted(unmapped):
        warnings.append(f"Field '{f}' has no source column configured (NOT_ASSESSED)")

    orphan_check = provenance_groups.groupby("provenance_group_id")["record_id"].count()
    zero_record_groups = orphan_check[orphan_check == 0]
    if len(zero_record_groups) > 0:
        warnings.append(
            f"{len(zero_record_groups)} provenance group(s) have zero records"
        )

    missing_group_links = records[
        ~records["record_id"].isin(provenance_groups["record_id"])
    ]
    if len(missing_group_links) > 0:
        warnings.append(
            f"{len(missing_group_links)} record(s) have no provenance group assignment"
        )

    record_rows = []
    for _, row in records.iterrows():
        record_entry: dict[str, Any] = {"record_id": row.get("record_id", None)}
        for _, schema_row in schema_df.iterrows():
            field_name = schema_row["evidence_field"]
            if field_name in config.field_mappings:
                mapping = config.field_mappings[field_name]
                if isinstance(mapping.source_column, list):
                    cols_exist = all(c in row.index for c in mapping.source_column)
                else:
                    cols_exist = mapping.source_column in row.index
                if not cols_exist:
                    record_entry[field_name] = "NOT_ASSESSED"
                else:
                    value = _resolve_value(row, mapping)
                    record_entry[field_name] = _classify_evidence_status(
                        value, field_name, config
                    )
            else:
                record_entry[field_name] = "NOT_ASSESSED"
        record_rows.append(record_entry)

    record_evidence_df = pd.DataFrame(record_rows)

    coverage_rows = []
    for _, schema_row in schema_df.iterrows():
        field_name = schema_row["evidence_field"]
        if field_name not in record_evidence_df.columns:
            continue
        statuses = record_evidence_df[field_name]
        n_total = len(statuses)
        n_present = (statuses == "PRESENT").sum()
        n_missing = (statuses == "MISSING").sum()
        n_ambiguous = (statuses == "AMBIGUOUS").sum()
        n_ref_derived = (statuses == "REFERENCE_DERIVED").sum()
        n_not_assessed = (statuses == "NOT_ASSESSED").sum()
        n_assessed = n_total - n_not_assessed
        coverage_frac = n_present / n_assessed if n_assessed > 0 else 0.0
        coverage_rows.append({
            "evidence_category": schema_row["evidence_category"],
            "evidence_field": field_name,
            "n_present": int(n_present),
            "n_missing": int(n_missing),
            "n_ambiguous": int(n_ambiguous),
            "n_reference_derived": int(n_ref_derived),
            "n_not_assessed": int(n_not_assessed),
            "n_assessed": int(n_assessed),
            "coverage_fraction": round(coverage_frac, 4),
            "evidence_level_expected": schema_row["evidence_level_expected"],
            **{tier: schema_row[tier] for tier in TIER_COLUMNS},
        })

    field_coverage_df = pd.DataFrame(coverage_rows)

    gate_rows = []
    for gate_def in ANALYSIS_GATES:
        tier_col = gate_def["tier"]
        required_fields = field_coverage_df[
            field_coverage_df[tier_col] == "REQUIRED"
        ]
        n_required = len(required_fields)
        n_required_met = (required_fields["coverage_fraction"] >= 0.8).sum()
        all_met = n_required_met == n_required
        blocking = required_fields[required_fields["coverage_fraction"] < 0.8]
        gate_rows.append({
            "gate": gate_def["gate"],
            "description": gate_def["description"],
            "status": "OPEN" if all_met else "CLOSED",
            "n_required_fields": int(n_required),
            "n_fields_meeting_threshold": int(n_required_met),
            "blocking_fields": "; ".join(blocking["evidence_field"].tolist())
            if len(blocking) > 0 else "none",
        })
        if not all_met and len(blocking) > 0:
            for _, b in blocking.iterrows():
                warnings.append(
                    f"{gate_def['gate']}: field '{b['evidence_field']}' "
                    f"coverage {b['coverage_fraction']:.2%} < 80% threshold"
                )

    gate_status_df = pd.DataFrame(gate_rows)

    n_groups = (
        provenance_groups["provenance_group_id"].nunique()
        if len(provenance_groups) > 0
        else 0
    )
    n_minerals = (
        records["mineral_label"].nunique()
        if "mineral_label" in records.columns
        else 0
    )

    return AuditResult(
        config=config,
        field_coverage=field_coverage_df,
        record_evidence=record_evidence_df,
        gate_status=gate_status_df,
        validation_warnings=warnings,
        n_records=len(records),
        n_groups=n_groups,
        n_minerals=n_minerals,
    )


# ---------------------------------------------------------------------------
# 4. Report generation
# ---------------------------------------------------------------------------

def generate_report(result: AuditResult) -> str:
    buf = io.StringIO()
    c = result.config

    buf.write(f"# Evidence Audit Report: {c.library_name} v{c.library_version}\n\n")
    buf.write(f"**Records:** {result.n_records}  \n")
    buf.write(f"**Provenance groups:** {result.n_groups}  \n")
    buf.write(f"**Mineral labels:** {result.n_minerals}  \n\n")

    buf.write("---\n\n## Field Coverage\n\n")
    buf.write("| Category | Field | Coverage | Assessed | Present | Missing | Ambiguous | Ref-derived |\n")
    buf.write("|----------|-------|----------|----------|---------|---------|-----------|-------------|\n")
    for _, row in result.field_coverage.iterrows():
        buf.write(
            f"| {row['evidence_category']} | {row['evidence_field']} "
            f"| {row['coverage_fraction']:.1%} "
            f"| {row['n_assessed']} | {row['n_present']} | {row['n_missing']} "
            f"| {row['n_ambiguous']} | {row['n_reference_derived']} |\n"
        )

    buf.write("\n---\n\n## Analysis Gates\n\n")
    buf.write("| Gate | Status | Required | Met | Blocking |\n")
    buf.write("|------|--------|----------|-----|----------|\n")
    for _, row in result.gate_status.iterrows():
        buf.write(
            f"| {row['gate']} | {row['status']} "
            f"| {row['n_required_fields']} | {row['n_fields_meeting_threshold']} "
            f"| {row['blocking_fields']} |\n"
        )

    if result.validation_warnings:
        buf.write("\n---\n\n## Validation Warnings\n\n")
        for w in result.validation_warnings:
            buf.write(f"- {w}\n")

    not_assessed = result.field_coverage[
        result.field_coverage["n_assessed"] == 0
    ]
    if len(not_assessed) > 0:
        buf.write("\n---\n\n## Out-of-Scope / Not Assessed Fields\n\n")
        buf.write("The following fields have no mapped source column and were not assessed:\n\n")
        for _, row in not_assessed.iterrows():
            fname = row["evidence_field"]
            info = NOT_OPERATIONALIZED_FIELDS.get(fname, {})
            rationale = info.get("rationale", "No mapping configured.")
            classification = info.get("classification", "NOT_ASSESSED")
            buf.write(f"- **{fname}** [{classification}]: {rationale}\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# 5. CLI entry point
# ---------------------------------------------------------------------------

def _cli_example() -> None:
    """Run a minimal audit on synthetic data to demonstrate the interface."""
    records = pd.DataFrame([
        {
            "record_id": "SPEC_001", "mineral_label": "Quartz",
            "sample_id": "QZ-01", "confidence": "STRONG",
            "collection": "LIB-A", "locality": "Brazil",
            "indep": "INDEPENDENT", "grain_um": "125",
            "pure": "TRUE", "geometry": "30/0",
            "wl_min": 350, "wl_max": 2500, "spacing": 5.0,
            "ref_formula": "SiO2", "crystal_sys": "trigonal",
        },
        {
            "record_id": "SPEC_002", "mineral_label": "Calcite",
            "sample_id": "CC-02", "confidence": "TENTATIVE",
            "collection": "LIB-B", "locality": None,
            "indep": "UNKNOWN", "grain_um": "unknown",
            "pure": "TRUE", "geometry": "45/0",
            "wl_min": 400, "wl_max": 2400, "spacing": 10.0,
            "ref_formula": "CaCO3", "crystal_sys": "trigonal",
        },
    ])
    groups = pd.DataFrame([
        {"record_id": "SPEC_001", "provenance_group_id": "PG_QZ01"},
        {"record_id": "SPEC_002", "provenance_group_id": "PG_CC02"},
    ])
    config = AuditConfig(
        library_name="ExampleLib",
        library_version="1.0",
        field_mappings={
            "mineral_name": FieldMapping("mineral_label"),
            "identity_confidence": FieldMapping("confidence"),
            "specimen_id": FieldMapping("sample_id"),
            "source_collection": FieldMapping("collection"),
            "geographic_provenance": FieldMapping("locality"),
            "independence_status": FieldMapping("indep"),
            "grain_size": FieldMapping("grain_um"),
            "mixture_status": FieldMapping("pure"),
            "measurement_geometry": FieldMapping("geometry"),
            "spectral_range": FieldMapping(["wl_min", "wl_max"]),
            "sampling_interval": FieldMapping("spacing"),
            "reference_formula": FieldMapping("ref_formula"),
            "crystal_system": FieldMapping("crystal_sys"),
        },
        ambiguous_values={
            "identity_confidence": ["TENTATIVE"],
            "independence_status": ["UNKNOWN"],
            "grain_size": ["unknown"],
        },
    )
    result = audit_evidence(records, groups, config)
    print(generate_report(result))


if __name__ == "__main__":
    _cli_example()
