"""Validate frozen components and audit compact JSON before publication."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sha = lambda b: hashlib.sha256(b).hexdigest()
components = [
    ('compliant-continuous-full4D-v01', 'publication-inventory.json'),
    ('compliant-continuous-full4D-v01', 'completed-publication-inventory.json'),
]
paths = {str(Path(__file__).relative_to(ROOT)),
         'runs/operator-dev/c39w-reference-progress-v01/publication-receipt.json',
         'runs/operator-dev/c39x-fixture-reference-v01/REPORT.md'}
inventories = []
for directory, filename in components:
    p = ROOT / 'runs/operator-dev' / directory / filename
    inventories.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p.read_bytes())))
    for row in json.loads(p.read_text())['files']:
        b = (ROOT / row['path']).read_bytes()
        if len(b) != row['bytes'] or sha(b) != row['sha256']:
            raise ValueError('Changed frozen artifact: ' + row['path'])
        paths.add(row['path'])
    paths.add(str(p.relative_to(ROOT)))

RAW_KEYS = {'rest_m', 'displacement_m', 'velocity_m_s', 'probe_q', 'probe_v',
            'positions_m', 'observations_m', 'measurement_m', 'noise_m',
            'actual_target_m', 'predicted_target_m', 'posterior_samples',
            'truth_points_m', 'canonical_q_m', 'canonical_v_m_s'}

def audit(value, file, trail=''):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in RAW_KEYS and isinstance(child, (list, dict)):
                raise ValueError(f'Raw physical JSON field: {file}:{trail}/{key}')
            audit(child, file, trail + '/' + key)
    elif isinstance(value, list):
        # Compact metadata may contain declared forces, mesh cells, clock times,
        # PID lists, provenance and hashes, but never physical coordinate grids.
        if len(value) > 8 and all(isinstance(x, list) and len(x) == 3 and
                                 all(isinstance(y, (int, float)) for y in x)
                                 for x in value):
            raise ValueError(f'Coordinate-like raw matrix: {file}:{trail}')
        for index, child in enumerate(value):
            audit(child, file, trail + '/' + str(index))

files = []
for name in sorted(paths):
    p = ROOT / name
    b = p.read_bytes()
    b.decode('utf-8')
    if p.is_symlink() or p.suffix not in ['.py', '.md', '.json', '.txt', '.log']:
        raise ValueError('Source and compact UTF8 text only')
    if p.suffix == '.json':
        audit(json.loads(b), name)
    files.append(dict(path=name, bytes=len(b), sha256=sha(b),
                      git_blob_sha1=hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()))
manifest = dict(schema='C39X-fixture-reference-v01',
                base_commit='3caae8c3236a5bdcd5f4a2556b5c233a5fbea15c',
                base_tree='af40c9894b080c62b0629629d11a90520f4af8a7',
                component_inventories=inventories, files=files, file_count=len(files),
                total_bytes=sum(r['bytes'] for r in files),
                local_preupload_JSON_raw_field_audit_passed=True,
                live_outputs_included=False, raw_fields_or_media_uploaded=False,
                dataset_admitted=False, campaign_complete=False)
p = ROOT / 'runs/campaign-c39x-paths.json'
p.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(files=len(files), bytes=manifest['total_bytes'], manifest_sha256=sha(p.read_bytes()))))
