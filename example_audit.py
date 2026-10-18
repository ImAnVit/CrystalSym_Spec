"""
Reproducible example: audit a synthetic spectral library.

This script demonstrates the evidence-auditor interface using only
synthetic data. No external library data are required.

Run:
    python example_audit.py
"""

from __future__ import annotations

import pandas as pd

from evidence_auditor import (
    AuditConfig,
    FieldMapping,
    audit_evidence,
    generate_report,
)


def main() -> None:
    records = pd.DataFrame([
        {
            "record_id": "SPEC_001", "mineral_label": "Quartz",
            "sample_id": "QZ-01", "confidence": "STRONG",
            "collection": "LIB-A", "locality": "Brazil",
            "indep": "INDEPENDENT", "grain_um": "125",
            "pure": "TRUE", "geometry": "30/0",
            "wl_min": 350, "wl_max": 2500, "spacing": 5.0,
            "preprocessing": "interpolated+normalised",
            "ref_formula": "SiO2", "chem_level": "A",
            "crystal_sys": "trigonal", "spacegroup": "P3121",
            "crystal_source": "XRD",
        },
        {
            "record_id": "SPEC_002", "mineral_label": "Quartz",
            "sample_id": "QZ-01", "confidence": "STRONG",
            "collection": "LIB-A", "locality": "Brazil",
            "indep": "INDEPENDENT", "grain_um": "250",
            "pure": "TRUE", "geometry": "30/0",
            "wl_min": 350, "wl_max": 2500, "spacing": 5.0,
            "preprocessing": "interpolated+normalised",
            "ref_formula": "SiO2", "chem_level": "A",
            "crystal_sys": "trigonal", "spacegroup": "P3121",
            "crystal_source": "XRD",
        },
        {
            "record_id": "SPEC_003", "mineral_label": "Calcite",
            "sample_id": "CC-02", "confidence": "TENTATIVE",
            "collection": "LIB-B", "locality": None,
            "indep": "UNKNOWN", "grain_um": "unknown",
            "pure": "TRUE", "geometry": "45/0",
            "wl_min": 400, "wl_max": 2400, "spacing": 10.0,
            "preprocessing": "normalised",
            "ref_formula": "CaCO3", "chem_level": "C",
            "crystal_sys": "trigonal", "spacegroup": "R-3c",
            "crystal_source": "reference",
        },
    ])

    groups = pd.DataFrame([
        {"record_id": "SPEC_001", "provenance_group_id": "PG_QZ01"},
        {"record_id": "SPEC_002", "provenance_group_id": "PG_QZ01"},
        {"record_id": "SPEC_003", "provenance_group_id": "PG_CC02"},
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
            "preprocessing_history": FieldMapping("preprocessing"),
            "reference_formula": FieldMapping("ref_formula"),
            "chemistry_evidence_level": FieldMapping(
                "chem_level", transform="map_values",
                value_map={"A": "A", "B": "B", "C": "C"},
            ),
            "crystal_system": FieldMapping("crystal_sys"),
            "space_group": FieldMapping("spacegroup"),
            "crystallographic_evidence_source": FieldMapping("crystal_source"),
        },
        ambiguous_values={
            "identity_confidence": ["TENTATIVE"],
            "independence_status": ["UNKNOWN"],
            "grain_size": ["unknown"],
        },
        reference_derived_indicators={
            "chemistry_evidence_level": ["C"],
            "crystallographic_evidence_source": ["reference"],
        },
    )

    result = audit_evidence(records, groups, config)

    print(generate_report(result))

    paths = result.export_csvs("example_output")
    print("CSV exports:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
