"""
Phase 5 Step 5.3.1 — Phase 3: Try fetching all 'not attempted' candidates from PDS.
"""

import csv
import os
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_URL = "https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-relab/data_reflectance"
OUT_DIR = "results/05_exploratory_analysis/phase5_step5_3_1_label_completeness"
LOG_FILE = os.path.join(OUT_DIR, "phase5_step5_3_1_label_retrieval_log.csv")

PDS_NS = {
    'pds4': 'http://pds.nasa.gov/pds4/pds/v1',
    'speclib': 'http://pds.nasa.gov/pds4/speclib/v1',
}


def parse_pds_xml(xml_text):
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

    return fields, None


def fetch_xml(record_id):
    for d in ['bdr2', 'bdr3']:
        url = f"{BASE_URL}/{d}/{record_id}.xml"
        try:
            req = Request(url, headers={'User-Agent': 'CrystalSym-Spec-Research/1.0'})
            resp = urlopen(req, timeout=30)
            if resp.status == 200:
                return resp.read().decode('utf-8'), url, d, None
        except HTTPError as e:
            if e.code == 404:
                continue
            return None, url, d, f'HTTP {e.code}'
        except URLError as e:
            return None, url, d, f'URL error: {e.reason}'
        except Exception as e:
            return None, url, d, str(e)
    return None, '', '', 'not_found_in_either_subdirectory'


def main():
    existing_log = []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_log.append(row)

    not_attempted = [r for r in existing_log if r['status'] == 'not_attempted']
    print(f"Found {len(not_attempted)} not-attempted entries")

    new_records_path = os.path.join(OUT_DIR, "phase5_step5_3_1_additional_archive_records.csv")
    existing_new = []
    with open(new_records_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        arch_fields = reader.fieldnames
        for row in reader:
            existing_new.append(row)

    new_found = 0
    still_missing = 0

    for entry in not_attempted:
        record_id = entry['record_id']
        content, url, subdir, error = fetch_xml(record_id)

        if content and not error:
            fields, parse_error = parse_pds_xml(content)
            if fields and not parse_error and any(v for v in fields.values()):
                new_found += 1
                entry['status'] = 'success'
                entry['url'] = url
                entry['subdirectory'] = subdir
                entry['parse_status'] = 'success'
                entry['error'] = ''

                rec = {
                    'archive_record_id': record_id,
                    'pds_urn': f'urn:nasa:pds:relab:data_reflectance:{record_id}',
                    'xml_label_url': url,
                    'pds_subdirectory': subdir,
                    'xml_label_retrieved': 'yes',
                    'notes': f'Retrieved in Step 5.3.1 Phase 3 on 2026-10-14'
                }
                for k, v in fields.items():
                    if k in arch_fields:
                        rec[k] = v
                existing_new.append(rec)
                print(f"  FOUND: {record_id} in {subdir}")
            else:
                still_missing += 1
                entry['status'] = 'fetch_failed'
                entry['error'] = 'parse_error_or_empty'
        else:
            still_missing += 1
            entry['status'] = 'not_found_in_archive'
            entry['error'] = error or 'not_found'

    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in existing_log:
            writer.writerow(row)

    with open(new_records_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=arch_fields, extrasaction='ignore')
        writer.writeheader()
        for row in existing_new:
            writer.writerow({k: row.get(k, '') for k in arch_fields})

    print(f"\nPhase 3 results: {new_found} found, {still_missing} still missing")
    print(f"Total new records now: {len(existing_new)}")


if __name__ == '__main__':
    main()
