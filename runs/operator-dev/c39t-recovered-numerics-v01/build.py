"""Validate frozen components and audit compact JSON before publication."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sha = lambda b: hashlib.sha256(b).hexdigest()
components = [
    ('long-supported-static-pilot-v01', 'completed-publication-inventory.json'),
    ('long-midpoint-capacity-v01', 'completed-publication-inventory.json'),
    ('long-smooth-trajectory-design-v01', 'publication-inventory.json'),
    ('long-smooth-trajectory-v02', 'preparation-publication-inventory.json'),
    ('resolved-public-inference-v01', 'completed-publication-inventory.json'),
    ('resolved-CDF-failure-trace-v01', 'publication-inventory.json'),
    ('resolved-CDF-domain-guard-v02', 'publication-inventory.json'),
    ('resolved-guard-repair-v02', 'preparation-publication-inventory.json'),
    ('resolved-guard-repair-v02', 'compact-completed-publication-inventory.json'),
    ('resolved-support-repair-v03', 'prepared-publication-inventory.json'),
    ('resolved-private-numerical-evaluation-v01', 'publication-inventory.json'),
    ('resolved-private-numerical-results-v01', 'publication-inventory.json'),
]
paths = {str(Path(__file__).relative_to(ROOT)),
         'runs/operator-dev/long-smooth-trajectory-v02/execution.json',
         'runs/operator-dev/long-smooth-trajectory-v02/terminal-technical-proof.json',
         'runs/operator-dev/resolved-support-repair-v03/completed-summary.json',
         'runs/operator-dev/c39t-recovered-numerics-v01/REPORT.md'}
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
manifest = dict(schema='C39T-recovered-numerics-and-dynamic-source-v01',
                base_commit='db2f06939a313c174b5c720727705670410cbdec',
                base_tree='7728979334faf615a948d13a9c8d1f48dc46fa1c',
                component_inventories=inventories, files=files, file_count=len(files),
                total_bytes=sum(r['bytes'] for r in files),
                local_preupload_JSON_raw_field_audit_passed=True,
                live_outputs_included=False, raw_fields_or_media_uploaded=False,
                dataset_admitted=False, campaign_complete=False)
p = ROOT / 'runs/campaign-c39t-paths.json'
p.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(files=len(files), bytes=manifest['total_bytes'], manifest_sha256=sha(p.read_bytes()))))
