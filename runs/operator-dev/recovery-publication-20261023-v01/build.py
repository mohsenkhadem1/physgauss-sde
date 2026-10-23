"""Audit component public inventories; never transfer raw simulation arrays."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = '2ccdc348fd21ed8276d9f7431e116d405bd7c3d2'
TREE = 'c735f75fe42c7eb736b0742a826926adb064a9ed'
sha = lambda b: hashlib.sha256(b).hexdigest()

def audit(v, trail=''):
    forbidden = {'rest_m', 'displacement_m', 'velocity_m_s', 'probe_q', 'probe_v',
                 'positions_m', 'observations_m', 'measurement_m', 'noise_m',
                 'posterior_samples', 'truth_points_m', 'canonical_q_m', 'canonical_v_m_s'}
    if isinstance(v, dict):
        for k, x in v.items():
            if k in forbidden and isinstance(x, (dict, list)):
                raise ValueError('Raw array ' + trail + '/' + k)
            audit(x, trail + '/' + k)
    elif isinstance(v, list):
        if len(v) > 8 and all(isinstance(x, list) and len(x) == 3 and
                             all(isinstance(y, (int, float)) for y in x) for x in v):
            raise ValueError('Coordinate-like matrix ' + trail)
        for i, x in enumerate(v):
            audit(x, trail + '/' + str(i))

def build(inventories, extras):
    paths = set(extras)
    components = []
    for name in inventories:
        p = ROOT / name
        components.append(dict(path=name, sha256=sha(p.read_bytes())))
        inv = json.loads(p.read_text())
        for row in inv.get('public_files', inv.get('files', [])):
            b = (ROOT / row['path']).read_bytes()
            if sha(b) != row['sha256']:
                raise ValueError('Changed frozen file: ' + row['path'])
            paths.add(row['path'])
    files = []
    for name in sorted(paths):
        p = ROOT / name
        b = p.read_bytes()
        b.decode('utf-8')
        if p.is_symlink() or p.suffix not in {'.py', '.json', '.txt'}:
            raise ValueError('Only source and compact text: ' + name)
        if p.suffix == '.json':
            audit(json.loads(b), name)
        files.append(dict(path=name, bytes=len(b), sha256=sha(b),
                          git_blob_sha1=hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()))
    manifest = dict(base_commit=BASE, base_tree=TREE, component_inventories=components,
                    files=files, file_count=len(files), total_bytes=sum(x['bytes'] for x in files),
                    raw_field_audit_passed=True, raw_datasets_uploaded=False,
                    existing_user_deletion_preserved=True, campaign_complete=False)
    out = ROOT / 'runs/operator-dev/recovery-publication-20261023-v01/manifest.json'
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(path=str(out.relative_to(ROOT)), files=len(files), bytes=manifest['total_bytes'], sha256=sha(out.read_bytes()))

if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument('--inventory', action='append', default=[])
    a.add_argument('--extra', action='append', default=[])
    args = a.parse_args()
    print(json.dumps(build(args.inventory, args.extra)))
