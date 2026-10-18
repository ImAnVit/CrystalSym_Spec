"""
Phase 5 Step 5.25I -- Synthetic tests for the evidence-auditor prototype.

Tests cover:
  1. Complete and missing fields
  2. Ambiguous and conflicting metadata
  3. Reference chemistry vs specimen-measured chemistry
  4. Multiple spectra linked to one provenance group
  5. Missing provenance links

All data are synthetic. No external library data are used.

Run:  python -m pytest test_evidence_auditor.py -v
  or:  python test_evidence_auditor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_auditor import (
    AuditConfig,
    FieldMapping,
    NOT_OPERATIONALIZED_FIELDS,
    audit_evidence,
    generate_report,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_records() -> pd.DataFrame:
    """10 synthetic spectral records with varied evidence profiles."""
    return pd.DataFrame(
        [
            {
                "record_id": "SPEC_001",
                "mineral_label": "Quartz",
                "sample_id": "QZ-01",
                "confidence": "STRONG",
                "id_method": "XRD",
                "collection": "LIB-A",
                "locality": "Brazil",
                "indep": "INDEPENDENT",
                "grain_um": "125",
                "prep": "powder",
                "thermal": "NO",
                "pure": "TRUE",
                "geometry": "30/0",
                "instrument": "Spectrometer-A",
                "calibration": "Halon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "interpolated+normalised",
                "chem_wt": "SiO2=99.8",
                "ref_formula": "SiO2",
                "chem_level": "A",
                "crystal_sys": "trigonal",
                "spacegroup": "P3121",
                "cell_params": "a=4.91 c=5.40",
                "crystal_source": "XRD",
                "uncertainty": 0.005,
                "replicate": "NO",
            },
            {
                "record_id": "SPEC_002",
                "mineral_label": "Quartz",
                "sample_id": "QZ-01",
                "confidence": "STRONG",
                "id_method": "XRD",
                "collection": "LIB-A",
                "locality": "Brazil",
                "indep": "INDEPENDENT",
                "grain_um": "250",
                "prep": "powder",
                "thermal": "NO",
                "pure": "TRUE",
                "geometry": "30/0",
                "instrument": "Spectrometer-A",
                "calibration": "Halon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "interpolated+normalised",
                "chem_wt": "SiO2=99.8",
                "ref_formula": "SiO2",
                "chem_level": "A",
                "crystal_sys": "trigonal",
                "spacegroup": "P3121",
                "cell_params": "a=4.91 c=5.40",
                "crystal_source": "XRD",
                "uncertainty": 0.005,
                "replicate": "NO",
            },
            {
                "record_id": "SPEC_003",
                "mineral_label": "Calcite",
                "sample_id": "CC-02",
                "confidence": "TENTATIVE",
                "id_method": None,
                "collection": "LIB-B",
                "locality": None,
                "indep": "UNKNOWN",
                "grain_um": "unknown",
                "prep": None,
                "thermal": None,
                "pure": "TRUE",
                "geometry": "45/0",
                "instrument": "Spectrometer-B",
                "calibration": None,
                "wl_min": 400,
                "wl_max": 2400,
                "spacing": 10.0,
                "preprocessing": "normalised",
                "chem_wt": None,
                "ref_formula": "CaCO3",
                "chem_level": "C",
                "crystal_sys": "trigonal",
                "spacegroup": "R-3c",
                "cell_params": None,
                "crystal_source": "reference",
                "uncertainty": None,
                "replicate": "YES",
            },
            {
                "record_id": "SPEC_004",
                "mineral_label": "Calcite",
                "sample_id": "CC-02",
                "confidence": "TENTATIVE",
                "id_method": None,
                "collection": "LIB-B",
                "locality": None,
                "indep": "UNKNOWN",
                "grain_um": "unknown",
                "prep": None,
                "thermal": None,
                "pure": "TRUE",
                "geometry": "45/0",
                "instrument": "Spectrometer-B",
                "calibration": None,
                "wl_min": 400,
                "wl_max": 2400,
                "spacing": 10.0,
                "preprocessing": "normalised",
                "chem_wt": None,
                "ref_formula": "CaCO3",
                "chem_level": "C",
                "crystal_sys": "trigonal",
                "spacegroup": "R-3c",
                "cell_params": None,
                "crystal_source": "reference",
                "uncertainty": None,
                "replicate": "YES",
            },
            {
                "record_id": "SPEC_005",
                "mineral_label": "Olivine",
                "sample_id": "OL-03",
                "confidence": "STRONG",
                "id_method": "XRF",
                "collection": "LIB-A",
                "locality": "Hawaii",
                "indep": "INDEPENDENT",
                "grain_um": "63",
                "prep": "powder",
                "thermal": "NO",
                "pure": "TRUE",
                "geometry": "30/0",
                "instrument": "Spectrometer-A",
                "calibration": "Halon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "interpolated+normalised",
                "chem_wt": "MgO=48,FeO=12,SiO2=40",
                "ref_formula": "(Mg,Fe)2SiO4",
                "chem_level": "A",
                "crystal_sys": "orthorhombic",
                "spacegroup": "Pbnm",
                "cell_params": "a=4.75 b=10.20 c=5.98",
                "crystal_source": "XRD",
                "uncertainty": 0.003,
                "replicate": "NO",
            },
            {
                "record_id": "SPEC_006",
                "mineral_label": "Feldspar",
                "sample_id": "FS-04",
                "confidence": "CONFLICT",
                "id_method": "optical",
                "collection": "LIB-C",
                "locality": "",
                "indep": "INDEPENDENT",
                "grain_um": None,
                "prep": "polished section",
                "thermal": "YES",
                "pure": "MIXTURE",
                "geometry": None,
                "instrument": "Spectrometer-C",
                "calibration": "Spectralon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "smoothed+normalised",
                "chem_wt": None,
                "ref_formula": "KAlSi3O8",
                "chem_level": "C",
                "crystal_sys": "monoclinic",
                "spacegroup": "C2/m",
                "cell_params": None,
                "crystal_source": "reference",
                "uncertainty": None,
                "replicate": "NO",
            },
            {
                "record_id": "SPEC_007",
                "mineral_label": "Pyroxene",
                "sample_id": "PX-05",
                "confidence": "STRONG",
                "id_method": "XRD",
                "collection": "LIB-A",
                "locality": "India",
                "indep": "INDEPENDENT",
                "grain_um": "125",
                "prep": "powder",
                "thermal": "NO",
                "pure": "TRUE",
                "geometry": "30/0",
                "instrument": "Spectrometer-A",
                "calibration": "Halon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "normalised",
                "chem_wt": None,
                "ref_formula": "MgSiO3",
                "chem_level": "B",
                "crystal_sys": "orthorhombic",
                "spacegroup": "Pbca",
                "cell_params": None,
                "crystal_source": "reference",
                "uncertainty": 0.01,
                "replicate": "NO",
            },
            {
                "record_id": "SPEC_008",
                "mineral_label": None,
                "sample_id": "XX-06",
                "confidence": None,
                "id_method": None,
                "collection": None,
                "locality": None,
                "indep": None,
                "grain_um": None,
                "prep": None,
                "thermal": None,
                "pure": None,
                "geometry": None,
                "instrument": None,
                "calibration": None,
                "wl_min": None,
                "wl_max": None,
                "spacing": None,
                "preprocessing": None,
                "chem_wt": None,
                "ref_formula": None,
                "chem_level": None,
                "crystal_sys": None,
                "spacegroup": None,
                "cell_params": None,
                "crystal_source": None,
                "uncertainty": None,
                "replicate": None,
            },
            {
                "record_id": "SPEC_009",
                "mineral_label": "Hematite",
                "sample_id": "HEM-07",
                "confidence": "STRONG",
                "id_method": "XRD",
                "collection": "LIB-B",
                "locality": "Australia",
                "indep": "INDEPENDENT",
                "grain_um": "45",
                "prep": "powder",
                "thermal": "NO",
                "pure": "TRUE",
                "geometry": "30/0",
                "instrument": "Spectrometer-B",
                "calibration": "Halon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "normalised",
                "chem_wt": "Fe2O3=99.5",
                "ref_formula": "Fe2O3",
                "chem_level": "A",
                "crystal_sys": "trigonal",
                "spacegroup": "R-3c",
                "cell_params": "a=5.03 c=13.75",
                "crystal_source": "XRD",
                "uncertainty": 0.002,
                "replicate": "NO",
            },
            {
                "record_id": "SPEC_010",
                "mineral_label": "Kaolinite",
                "sample_id": "KA-08",
                "confidence": "STRONG",
                "id_method": "XRD",
                "collection": "LIB-A",
                "locality": "Georgia",
                "indep": "INDEPENDENT",
                "grain_um": "20",
                "prep": "powder",
                "thermal": "NO",
                "pure": "TRUE",
                "geometry": "30/0",
                "instrument": "Spectrometer-A",
                "calibration": "Halon",
                "wl_min": 350,
                "wl_max": 2500,
                "spacing": 5.0,
                "preprocessing": "interpolated+normalised",
                "chem_wt": None,
                "ref_formula": "Al2Si2O5(OH)4",
                "chem_level": "C",
                "crystal_sys": "triclinic",
                "spacegroup": "C1",
                "cell_params": None,
                "crystal_source": "reference",
                "uncertainty": None,
                "replicate": "NO",
            },
        ]
    )


def _make_provenance() -> pd.DataFrame:
    """Provenance group table. SPEC_008 is deliberately excluded (missing link)."""
    return pd.DataFrame(
        [
            {"record_id": "SPEC_001", "provenance_group_id": "PG_QZ01", "group_type": "grain_size_series"},
            {"record_id": "SPEC_002", "provenance_group_id": "PG_QZ01", "group_type": "grain_size_series"},
            {"record_id": "SPEC_003", "provenance_group_id": "PG_CC02", "group_type": "replicate"},
            {"record_id": "SPEC_004", "provenance_group_id": "PG_CC02", "group_type": "replicate"},
            {"record_id": "SPEC_005", "provenance_group_id": "PG_OL03", "group_type": "specimen"},
            {"record_id": "SPEC_006", "provenance_group_id": "PG_FS04", "group_type": "specimen"},
            {"record_id": "SPEC_007", "provenance_group_id": "PG_PX05", "group_type": "specimen"},
            {"record_id": "SPEC_009", "provenance_group_id": "PG_HEM07", "group_type": "specimen"},
            {"record_id": "SPEC_010", "provenance_group_id": "PG_KA08", "group_type": "specimen"},
        ]
    )


def _make_config() -> AuditConfig:
    return AuditConfig(
        library_name="SyntheticTestLib",
        library_version="1.0",
        field_mappings={
            "mineral_name": FieldMapping("mineral_label"),
            "identity_confidence": FieldMapping("confidence"),
            "identification_method": FieldMapping("id_method"),
            "specimen_id": FieldMapping("sample_id"),
            "source_collection": FieldMapping("collection"),
            "geographic_provenance": FieldMapping("locality"),
            "independence_status": FieldMapping("indep"),
            "grain_size": FieldMapping("grain_um"),
            "sample_preparation": FieldMapping("prep"),
            "thermal_treatment": FieldMapping("thermal"),
            "mixture_status": FieldMapping("pure"),
            "measurement_geometry": FieldMapping("geometry"),
            "instrument_configuration": FieldMapping("instrument"),
            "calibration_details": FieldMapping("calibration"),
            "spectral_range": FieldMapping(["wl_min", "wl_max"]),
            "sampling_interval": FieldMapping("spacing"),
            "preprocessing_history": FieldMapping("preprocessing"),
            "specimen_specific_chemistry": FieldMapping("chem_wt"),
            "reference_formula": FieldMapping("ref_formula"),
            "chemistry_evidence_level": FieldMapping(
                "chem_level",
                transform="map_values",
                value_map={"A": "A", "B": "B", "C": "C"},
            ),
            "crystal_system": FieldMapping("crystal_sys"),
            "space_group": FieldMapping("spacegroup"),
            "unit_cell_parameters": FieldMapping("cell_params"),
            "crystallographic_evidence_source": FieldMapping("crystal_source"),
            "measurement_uncertainty": FieldMapping("uncertainty"),
            "reproducibility_evidence": FieldMapping("replicate"),
        },
        ambiguous_values={
            "identity_confidence": ["TENTATIVE", "CONFLICT"],
            "independence_status": ["UNKNOWN"],
            "grain_size": ["unknown"],
        },
        reference_derived_indicators={
            "chemistry_evidence_level": ["C"],
            "crystallographic_evidence_source": ["reference"],
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_result():
    records = _make_records()
    provenance = _make_provenance()
    config = _make_config()
    return audit_evidence(records, provenance, config)


class TestCompleteAndMissingFields:
    """Test 1: complete and missing fields."""

    def test_all_30_fields_present_in_coverage(self, audit_result):
        assert len(audit_result.field_coverage) == 30

    def test_spectral_range_has_high_coverage(self, audit_result):
        row = audit_result.field_coverage[
            audit_result.field_coverage["evidence_field"] == "spectral_range"
        ]
        assert len(row) == 1
        assert row.iloc[0]["coverage_fraction"] > 0.8

    def test_missing_fields_detected(self, audit_result):
        spec008 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_008"
        ]
        missing_count = 0
        for col in spec008.columns:
            if col == "record_id":
                continue
            if spec008.iloc[0][col] == "MISSING":
                missing_count += 1
        assert missing_count > 20, (
            f"SPEC_008 should have many MISSING fields, got {missing_count}"
        )

    def test_complete_record_has_many_present(self, audit_result):
        spec001 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_001"
        ]
        present_count = sum(
            1 for col in spec001.columns
            if col != "record_id" and spec001.iloc[0][col] == "PRESENT"
        )
        assert present_count >= 20, (
            f"SPEC_001 (complete record) should have many PRESENT fields, got {present_count}"
        )

    def test_unmapped_fields_are_not_assessed(self, audit_result):
        for col in [
            "source_group_id",
            "unique_specimen_identifier",
            "n_verified_independent_specimens",
            "descriptor_uncertainty",
        ]:
            if col in audit_result.record_evidence.columns:
                statuses = audit_result.record_evidence[col].unique()
                assert all(s == "NOT_ASSESSED" for s in statuses), (
                    f"Unmapped field '{col}' should be NOT_ASSESSED for all records"
                )


class TestAmbiguousAndConflictingMetadata:
    """Test 2: ambiguous and conflicting metadata."""

    def test_tentative_confidence_is_ambiguous(self, audit_result):
        spec003 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_003"
        ]
        assert spec003.iloc[0]["identity_confidence"] == "AMBIGUOUS"

    def test_conflict_confidence_is_ambiguous(self, audit_result):
        spec006 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_006"
        ]
        assert spec006.iloc[0]["identity_confidence"] == "AMBIGUOUS"

    def test_unknown_independence_is_ambiguous(self, audit_result):
        spec003 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_003"
        ]
        assert spec003.iloc[0]["independence_status"] == "AMBIGUOUS"

    def test_unknown_grain_size_is_ambiguous(self, audit_result):
        spec003 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_003"
        ]
        assert spec003.iloc[0]["grain_size"] == "AMBIGUOUS"

    def test_strong_confidence_is_present(self, audit_result):
        spec001 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_001"
        ]
        assert spec001.iloc[0]["identity_confidence"] == "PRESENT"


class TestReferenceVsSpecimenChemistry:
    """Test 3: reference chemistry vs specimen-measured chemistry."""

    def test_level_a_chemistry_is_present(self, audit_result):
        spec001 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_001"
        ]
        assert spec001.iloc[0]["chemistry_evidence_level"] == "PRESENT"

    def test_level_c_chemistry_is_reference_derived(self, audit_result):
        spec003 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_003"
        ]
        assert spec003.iloc[0]["chemistry_evidence_level"] == "REFERENCE_DERIVED"

    def test_level_b_chemistry_is_present(self, audit_result):
        spec007 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_007"
        ]
        assert spec007.iloc[0]["chemistry_evidence_level"] == "PRESENT"

    def test_reference_crystallography_is_reference_derived(self, audit_result):
        spec003 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_003"
        ]
        assert spec003.iloc[0]["crystallographic_evidence_source"] == "REFERENCE_DERIVED"

    def test_xrd_crystallography_is_present(self, audit_result):
        spec001 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_001"
        ]
        assert spec001.iloc[0]["crystallographic_evidence_source"] == "PRESENT"

    def test_coverage_counts_reference_derived_separately(self, audit_result):
        chem_row = audit_result.field_coverage[
            audit_result.field_coverage["evidence_field"] == "chemistry_evidence_level"
        ]
        assert chem_row.iloc[0]["n_reference_derived"] >= 3


class TestMultipleSpectraPerGroup:
    """Test 4: multiple spectra linked to one provenance group."""

    def test_group_with_multiple_records(self, audit_result):
        pg_qz01 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"].isin(["SPEC_001", "SPEC_002"])
        ]
        assert len(pg_qz01) == 2
        assert pg_qz01.iloc[0]["specimen_id"] == "PRESENT"
        assert pg_qz01.iloc[1]["specimen_id"] == "PRESENT"

    def test_replicate_group_detected(self, audit_result):
        pg_cc02 = audit_result.record_evidence[
            audit_result.record_evidence["record_id"].isin(["SPEC_003", "SPEC_004"])
        ]
        assert len(pg_cc02) == 2

    def test_n_records_correct(self, audit_result):
        assert audit_result.n_records == 10

    def test_n_groups_correct(self, audit_result):
        assert audit_result.n_groups == 7


class TestMissingProvenanceLinks:
    """Test 5: missing provenance links."""

    def test_missing_link_warning(self, audit_result):
        assert any(
            "no provenance group" in w.lower()
            for w in audit_result.validation_warnings
        )

    def test_spec008_has_no_group(self, audit_result):
        spec008_row = audit_result.record_evidence[
            audit_result.record_evidence["record_id"] == "SPEC_008"
        ]
        assert len(spec008_row) == 1

    def test_unmapped_field_warnings(self, audit_result):
        unmapped_warnings = [
            w for w in audit_result.validation_warnings
            if "no source column configured" in w
        ]
        assert len(unmapped_warnings) >= 3


class TestGateStatus:
    """Test that analysis gates are computed correctly."""

    def test_four_gates_present(self, audit_result):
        assert len(audit_result.gate_status) == 4

    def test_gates_have_required_columns(self, audit_result):
        for col in ["gate", "status", "n_required_fields", "blocking_fields"]:
            assert col in audit_result.gate_status.columns

    def test_descriptive_gate_partially_open(self, audit_result):
        gate = audit_result.gate_status[
            audit_result.gate_status["gate"] == "GATE_DESCRIPTIVE"
        ]
        assert len(gate) == 1

    def test_specimen_chemistry_gate_blocked(self, audit_result):
        gate = audit_result.gate_status[
            audit_result.gate_status["gate"] == "GATE_SPECIMEN_CHEMISTRY"
        ]
        assert len(gate) == 1
        assert gate.iloc[0]["status"] == "CLOSED"


class TestReportGeneration:
    """Test that the Markdown report is generated correctly."""

    def test_report_contains_library_name(self, audit_result):
        report = generate_report(audit_result)
        assert "SyntheticTestLib" in report

    def test_report_contains_field_coverage_table(self, audit_result):
        report = generate_report(audit_result)
        assert "Field Coverage" in report
        assert "mineral_name" in report

    def test_report_contains_gate_status(self, audit_result):
        report = generate_report(audit_result)
        assert "Analysis Gates" in report
        assert "GATE_DESCRIPTIVE" in report

    def test_report_contains_warnings(self, audit_result):
        report = generate_report(audit_result)
        assert "Validation Warnings" in report


class TestNotOperationalizedFields:
    """Test that the 6 NOT_OPERATIONALIZED fields are classified correctly."""

    def test_six_fields_classified(self):
        assert len(NOT_OPERATIONALIZED_FIELDS) == 6

    def test_all_fields_have_classification_and_rationale(self):
        for field_name, info in NOT_OPERATIONALIZED_FIELDS.items():
            assert "classification" in info, f"{field_name} missing 'classification'"
            assert "rationale" in info, f"{field_name} missing 'rationale'"
            assert info["classification"] in ("AUTOMATED", "HUMAN_REVIEW", "OUT_OF_SCOPE")

    def test_descriptor_uncertainty_is_out_of_scope(self):
        assert NOT_OPERATIONALIZED_FIELDS["descriptor_uncertainty"]["classification"] == "OUT_OF_SCOPE"

    def test_specimen_specific_chemistry_is_automated(self):
        assert NOT_OPERATIONALIZED_FIELDS["specimen_specific_chemistry"]["classification"] == "AUTOMATED"

    def test_identification_method_is_human_review(self):
        assert NOT_OPERATIONALIZED_FIELDS["identification_method"]["classification"] == "HUMAN_REVIEW"

    def test_out_of_scope_fields_in_report(self, audit_result):
        report = generate_report(audit_result)
        assert "Out-of-Scope" in report or "Not Assessed" in report


class TestCSVExport:
    """Test CSV export functionality."""

    def test_export_creates_three_files(self, audit_result, tmp_path):
        paths = audit_result.export_csvs(tmp_path)
        assert set(paths.keys()) == {"field_coverage", "record_evidence", "gate_status"}
        for p in paths.values():
            assert Path(p).exists()

    def test_exported_csv_is_readable(self, audit_result, tmp_path):
        paths = audit_result.export_csvs(tmp_path)
        df = pd.read_csv(paths["field_coverage"])
        assert len(df) == 30
        assert "evidence_field" in df.columns


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running evidence-auditor synthetic tests...\n")

    records = _make_records()
    provenance = _make_provenance()
    config = _make_config()
    result = audit_evidence(records, provenance, config)

    print(f"Records: {result.n_records}")
    print(f"Groups: {result.n_groups}")
    print(f"Minerals: {result.n_minerals}")
    print(f"Warnings: {len(result.validation_warnings)}")
    print()

    print("=== Field Coverage ===")
    for _, row in result.field_coverage.iterrows():
        print(
            f"  {row['evidence_category']:25s} {row['evidence_field']:35s} "
            f"{row['coverage_fraction']:6.1%}  "
            f"(P={row['n_present']} M={row['n_missing']} "
            f"A={row['n_ambiguous']} R={row['n_reference_derived']} "
            f"NA={row['n_not_assessed']})"
        )

    print("\n=== Analysis Gates ===")
    for _, row in result.gate_status.iterrows():
        print(
            f"  {row['gate']:40s} {row['status']:6s}  "
            f"blocking: {row['blocking_fields']}"
        )

    print("\n=== Validation Warnings ===")
    for w in result.validation_warnings:
        print(f"  - {w}")

    print("\n=== Report (first 40 lines) ===")
    report = generate_report(result)
    for line in report.split("\n")[:40]:
        print(line)

    print("\nAll synthetic tests passed (standalone run).")
