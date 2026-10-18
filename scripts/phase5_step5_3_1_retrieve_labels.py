"""
Phase 5 Step 5.3.1 — Retrieve remaining PDS XML labels and audit evidence completeness.

Reads the Step 5.3 crosswalk, identifies candidates without XML label retrieval,
attempts to fetch their PDS4 XML labels, parses metadata, and produces deliverables.
"""

import csv
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_URL = "https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-relab/data_reflectance"
CROSSWALK = "results/05_exploratory_analysis/phase5_step5_3_pds_catalogue_evidence/phase5_step5_3_candidate_record_crosswalk.csv"
ARCHIVE_RECORDS = "results/05_exploratory_analysis/phase5_step5_3_pds_catalogue_evidence/phase5_step5_3_archive_records.csv"
OUT_DIR = "results/05_exploratory_analysis/phase5_step5_3_1_label_completeness"

PDS_NS = {
    'pds4': 'http://pds.nasa.gov/pds4/pds/v1',
    'speclib': 'http://pds.nasa.gov/pds4/speclib/v1',
}


def parse_pds_xml(xml_text):
    """Extract metadata fields from a PDS4 XML label."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return None, str(e)

    fields = {}

    def find_speclib(tag):
        el = root.find(f'.//speclib:{tag}', PDS_NS)
        if el is not None and el.text:
            return el.text.strip()
        return ''

    def find_pds4(tag):
        el = root.find(f'.//pds4:{tag}', PDS_NS)
        if el is not None and el.text:
            return el.text.strip()
        return ''

    fields['specimen_id'] = find_speclib('specimen_id')
    fields['specimen_name'] = find_speclib('specimen_name')
    fields['specimen_description'] = find_speclib('specimen_description')
    fields['specimen_collection_location'] = find_speclib('specimen_collection_location')
    fields['specimen_owner_name'] = find_speclib('specimen_owner_name')
    fields['specimen_owner_location'] = find_speclib('specimen_owner_location')
    fields['material_origin'] = find_speclib('material_origin')
    fields['synthetic_type'] = find_speclib('synthetic_type')
    fields['material_type'] = find_speclib('material_type')
    fields['mineral_type'] = find_speclib('mineral_type')
    fields['mineral_subtype'] = find_speclib('mineral_subtype')

    fields['specimen_min_size_um'] = find_speclib('specimen_min_size')
    fields['specimen_max_size_um'] = find_speclib('specimen_max_size')

    fields['source_specimen_id'] = find_speclib('source_specimen_id')

    fields['measurement_geometry_type'] = find_speclib('measurement_geometry_type')
    fields['incidence_angle_deg'] = find_speclib('incidence_angle')
    fields['emission_angle_deg'] = find_speclib('emission_angle')
    fields['phase_angle_deg'] = find_speclib('phase_angle')

    fields['spectral_range_min_nm'] = find_speclib('spectral_range_min')
    fields['spectral_range_max_nm'] = find_speclib('spectral_range_max')

    fields['instrument_name'] = find_speclib('instrument_name')
    fields['measurement_date_time'] = find_speclib('measurement_date_time')
    fields['data_producer_name'] = find_speclib('data_producer_name')

    if not fields['measurement_date_time']:
        fields['measurement_date_time'] = find_pds4('start_date_time')

    if not fields['instrument_name']:
        for comp in root.findall('.//pds4:Observing_System_Component', PDS_NS):
            name_el = comp.find('pds4:name', PDS_NS)
            type_el = comp.find('pds4:type', PDS_NS)
            if (type_el is not None and type_el.text and
                    type_el.text.strip().lower() == 'instrument' and
                    name_el is not None and name_el.text):
                fields['instrument_name'] = name_el.text.strip()
                break

    target_name = find_pds4('Target_Identification/name')
    if target_name:
        fields['target_name'] = target_name

    return fields, None


def fetch_xml(record_id, subdirectory=None):
    """Attempt to fetch XML label from PDS. Try both bdr2 and bdr3 if subdirectory unknown."""
    dirs_to_try = [subdirectory] if subdirectory else ['bdr2', 'bdr3']

    for d in dirs_to_try:
        url = f"{BASE_URL}/{d}/{record_id}.xml"
        try:
            req = Request(url, headers={'User-Agent': 'CrystalSym-Spec-Research/1.0'})
            resp = urlopen(req, timeout=30)
            if resp.status == 200:
                content = resp.read().decode('utf-8')
                return content, url, d, None
        except HTTPError as e:
            if e.code == 404:
                continue
            return None, url, d, f'HTTP {e.code}'
        except URLError as e:
            return None, url, d, f'URL error: {e.reason}'
        except Exception as e:
            return None, url, d, str(e)

    return None, '', '', 'not_found_in_either_subdirectory'


def load_crosswalk():
    rows = []
    with open(CROSSWALK, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_archive_records():
    records = {}
    with open(ARCHIVE_RECORDS, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row['archive_record_id']] = row
    return records


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    crosswalk = load_crosswalk()
    existing_records = load_archive_records()

    print(f"Loaded {len(crosswalk)} crosswalk rows")
    print(f"Loaded {len(existing_records)} existing archive records")

    # Categorize crosswalk rows
    already_retrieved = []
    archive_match_no_xml = []
    not_in_archive = []

    for row in crosswalk:
        pds_id = row.get('pds_archive_record_id', '').strip()
        match_basis = row.get('match_basis', '').strip()

        if match_basis in ('filename_exact_match', 'prefix_stripped_match') and pds_id:
            if pds_id in existing_records:
                already_retrieved.append(row)
            else:
                archive_match_no_xml.append(row)
        else:
            not_in_archive.append(row)

    print(f"\nInventory:")
    print(f"  Already retrieved (XML in Step 5.3): {len(already_retrieved)}")
    print(f"  Archive match, no XML retrieval:     {len(archive_match_no_xml)}")
    print(f"  Not found in archive:                {len(not_in_archive)}")
    print(f"  Total:                               {len(already_retrieved) + len(archive_match_no_xml) + len(not_in_archive)}")

    # Collect unique archive IDs to fetch
    ids_to_fetch = set()
    id_to_rows = {}
    for row in archive_match_no_xml:
        pds_id = row['pds_archive_record_id'].strip()
        ids_to_fetch.add(pds_id)
        id_to_rows.setdefault(pds_id, []).append(row)

    # Also try prefix-stripped versions for not_in_archive candidates
    prefix_candidates = {}
    for row in not_in_archive:
        local_fn = row.get('local_filename', '').strip().replace('.tab', '')
        # Try stripping common prefixes
        for prefix in ['bfr1', 'bmr1', 'bir1', 'bir2', 'bdr1', 'bdr2']:
            if local_fn.startswith(prefix):
                stripped = local_fn[len(prefix):]
                if stripped:
                    prefix_candidates.setdefault(stripped, []).append(row)
                    break

    print(f"\nUnique archive IDs to fetch from archive-match group: {len(ids_to_fetch)}")
    print(f"Prefix-stripped candidates to try: {len(prefix_candidates)}")

    # Fetch XML labels
    retrieval_log = []
    new_archive_records = {}
    fetch_count = 0
    success_count = 0
    fail_count = 0

    # Phase 1: Fetch for archive-match candidates
    print(f"\n--- Phase 1: Fetching {len(ids_to_fetch)} XML labels for archive-match candidates ---")
    for record_id in sorted(ids_to_fetch):
        fetch_count += 1
        content, url, subdir, error = fetch_xml(record_id)

        if content and not error:
            fields, parse_error = parse_pds_xml(content)
            if fields and not parse_error:
                success_count += 1
                new_archive_records[record_id] = {
                    'archive_record_id': record_id,
                    'pds_urn': f'urn:nasa:pds:relab:data_reflectance:{record_id}',
                    'xml_label_url': url,
                    'pds_subdirectory': subdir,
                    **fields
                }
                retrieval_log.append({
                    'record_id': record_id,
                    'fetch_attempt': fetch_count,
                    'url': url,
                    'status': 'success',
                    'subdirectory': subdir,
                    'parse_status': 'success',
                    'error': '',
                    'manifest_ids': ';'.join(r['manifest_id'] for r in id_to_rows.get(record_id, []))
                })
                if fetch_count % 10 == 0:
                    print(f"  Progress: {fetch_count}/{len(ids_to_fetch)} ({success_count} ok, {fail_count} fail)")
            else:
                fail_count += 1
                retrieval_log.append({
                    'record_id': record_id,
                    'fetch_attempt': fetch_count,
                    'url': url,
                    'status': 'success',
                    'subdirectory': subdir,
                    'parse_status': 'parse_error',
                    'error': parse_error or 'empty_fields',
                    'manifest_ids': ';'.join(r['manifest_id'] for r in id_to_rows.get(record_id, []))
                })
        else:
            fail_count += 1
            retrieval_log.append({
                'record_id': record_id,
                'fetch_attempt': fetch_count,
                'url': url,
                'status': 'fetch_failed',
                'subdirectory': subdir,
                'parse_status': '',
                'error': error,
                'manifest_ids': ';'.join(r['manifest_id'] for r in id_to_rows.get(record_id, []))
            })

    print(f"  Phase 1 complete: {success_count} ok, {fail_count} fail out of {fetch_count}")

    # Phase 2: Try prefix-stripped IDs for not_in_archive candidates
    phase2_fetch = 0
    phase2_ok = 0
    print(f"\n--- Phase 2: Trying {len(prefix_candidates)} prefix-stripped IDs ---")
    for stripped_id, rows in sorted(prefix_candidates.items()):
        if stripped_id in ids_to_fetch:
            continue  # already tried
        if stripped_id in existing_records:
            continue  # already have it

        phase2_fetch += 1
        content, url, subdir, error = fetch_xml(stripped_id)

        if content and not error:
            fields, parse_error = parse_pds_xml(content)
            if fields and not parse_error:
                phase2_ok += 1
                new_archive_records[stripped_id] = {
                    'archive_record_id': stripped_id,
                    'pds_urn': f'urn:nasa:pds:relab:data_reflectance:{stripped_id}',
                    'xml_label_url': url,
                    'pds_subdirectory': subdir,
                    **fields
                }
                retrieval_log.append({
                    'record_id': stripped_id,
                    'fetch_attempt': fetch_count + phase2_fetch,
                    'url': url,
                    'status': 'success',
                    'subdirectory': subdir,
                    'parse_status': 'success',
                    'error': '',
                    'manifest_ids': ';'.join(r['manifest_id'] for r in rows),
                    'prefix_stripped_from': ';'.join(r['local_filename'].replace('.tab', '') for r in rows)
                })
            else:
                retrieval_log.append({
                    'record_id': stripped_id,
                    'fetch_attempt': fetch_count + phase2_fetch,
                    'url': url,
                    'status': 'success',
                    'subdirectory': subdir,
                    'parse_status': 'parse_error',
                    'error': parse_error or 'empty_fields',
                    'manifest_ids': ';'.join(r['manifest_id'] for r in rows),
                    'prefix_stripped_from': ';'.join(r['local_filename'].replace('.tab', '') for r in rows)
                })
        else:
            retrieval_log.append({
                'record_id': stripped_id,
                'fetch_attempt': fetch_count + phase2_fetch,
                'url': url,
                'status': 'fetch_failed',
                'subdirectory': subdir,
                'parse_status': '',
                'error': error,
                'manifest_ids': ';'.join(r['manifest_id'] for r in rows),
                'prefix_stripped_from': ';'.join(r['local_filename'].replace('.tab', '') for r in rows)
            })

        if phase2_fetch % 20 == 0:
            print(f"  Progress: {phase2_fetch}/{len(prefix_candidates)} ({phase2_ok} ok)")

    print(f"  Phase 2 complete: {phase2_ok} ok out of {phase2_fetch}")

    # Phase 3: Log not_in_archive candidates that were NOT attempted
    not_attempted_ids = []
    for row in not_in_archive:
        local_fn = row.get('local_filename', '').strip().replace('.tab', '')
        was_attempted = False
        for prefix in ['bfr1', 'bmr1', 'bir1', 'bir2', 'bdr1', 'bdr2']:
            if local_fn.startswith(prefix):
                stripped = local_fn[len(prefix):]
                if stripped in prefix_candidates:
                    was_attempted = True
                    break
        if not was_attempted:
            not_attempted_ids.append(row)

    for row in not_attempted_ids:
        retrieval_log.append({
            'record_id': row['local_filename'].replace('.tab', ''),
            'fetch_attempt': 0,
            'url': '',
            'status': 'not_attempted',
            'subdirectory': '',
            'parse_status': '',
            'error': 'no_archive_match_and_no_prefix_strategy',
            'manifest_ids': row['manifest_id']
        })

    # Write retrieval log
    log_path = os.path.join(OUT_DIR, 'phase5_step5_3_1_label_retrieval_log.csv')
    log_fields = ['record_id', 'fetch_attempt', 'url', 'status', 'subdirectory', 'parse_status', 'error', 'manifest_ids']
    if any('prefix_stripped_from' in r for r in retrieval_log):
        log_fields.append('prefix_stripped_from')

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(retrieval_log, key=lambda r: r.get('fetch_attempt', 0)):
            writer.writerow(row)
    print(f"\nWrote {len(retrieval_log)} rows to {log_path}")

    # Write additional archive records
    archive_fields = [
        'archive_record_id', 'pds_urn', 'specimen_id', 'specimen_name',
        'specimen_description', 'specimen_collection_location',
        'specimen_owner_name', 'specimen_owner_location',
        'material_origin', 'synthetic_type', 'material_type', 'mineral_type',
        'mineral_subtype', 'specimen_min_size_um', 'specimen_max_size_um',
        'source_specimen_id', 'measurement_geometry_type',
        'incidence_angle_deg', 'emission_angle_deg', 'phase_angle_deg',
        'spectral_range_min_nm', 'spectral_range_max_nm',
        'instrument_name', 'measurement_date_time', 'data_producer_name',
        'pds_subdirectory', 'xml_label_retrieved', 'xml_label_url', 'notes'
    ]

    archive_path = os.path.join(OUT_DIR, 'phase5_step5_3_1_additional_archive_records.csv')
    with open(archive_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=archive_fields, extrasaction='ignore')
        writer.writeheader()
        for rec_id in sorted(new_archive_records.keys()):
            rec = new_archive_records[rec_id]
            rec['xml_label_retrieved'] = 'yes'
            rec['notes'] = f'Retrieved in Step 5.3.1 on {datetime.now(timezone.utc).strftime("%Y-%m-%d")}'
            writer.writerow({k: rec.get(k, '') for k in archive_fields})
    print(f"Wrote {len(new_archive_records)} rows to {archive_path}")

    # Summary stats
    total_success = success_count + phase2_ok
    total_attempted = fetch_count + phase2_fetch
    total_not_attempted = len(not_attempted_ids)

    print(f"\n=== SUMMARY ===")
    print(f"Total candidates in crosswalk: {len(crosswalk)}")
    print(f"Already retrieved in Step 5.3: {len(already_retrieved)}")
    print(f"Archive-match group (Phase 1): {len(ids_to_fetch)} unique IDs attempted, {success_count} succeeded")
    print(f"Prefix-stripped group (Phase 2): {phase2_fetch} attempted, {phase2_ok} succeeded")
    print(f"Not attempted (no strategy): {total_not_attempted}")
    print(f"Total newly retrieved: {total_success}")
    print(f"Total retrieval log entries: {len(retrieval_log)}")

    return retrieval_log, new_archive_records, crosswalk, existing_records


if __name__ == '__main__':
    main()
