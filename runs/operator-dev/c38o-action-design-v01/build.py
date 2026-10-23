"""Publish completed controls and frozen next-phase source, never live outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sha = lambda b: hashlib.sha256(b).hexdigest()
components = [('c38-initial-public-inference-v03', 'completed-publication-inventory.json'), ('c38-initial-public-inference-v02', 'failure-publication-inventory.json'), ('c38-static-policy-execution-v02', 'preparation-inventory.json'), ('c38-static-policy-execution-v02', 'completion-inventory.json'), ('c38-postprobe-source-v02', 'publication-inventory.json'), ('c38-acquired-probe-release-v01', 'completed-publication-inventory.json'), ('c38-postprobe-inference-v01', 'preparation-publication-inventory.json'), ('c38-full-evaluation-source-v01', 'publication-inventory.json'), ('c38-static-inverse-video-v01', 'publication-inventory.json'), ('c38-postprobe-profile-v01', 'publication-inventory.json'), ('p2-contact-rigid-elastic-diagnosis-v01', 'completed-publication-inventory.json'), ('p2-soft-floor-screen-v01', 'completed-publication-inventory.json'), ('c39-coarse-linear-screen-v01', 'compact-publication-inventory.json'), ('c39-fine-linear-screen-v01', 'completed-publication-inventory.json')]
paths = set()
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
paths.update(['docs/GAUSSIAN_C38_PROBING_CHOICE_CHECKPOINT.md', 'runs/operator-dev/c39-controlled-extension-design-v01/design.json', 'runs/operator-dev/c38n-inverse-controls-v01/publication-receipt.json', str(Path(__file__).relative_to(ROOT))])
files = []
for name in sorted(paths):
    p = ROOT / name
    b = p.read_bytes()
    b.decode('utf-8')
    if p.is_symlink() or p.suffix not in ['.py', '.md', '.json', '.txt', '.log']:
        raise ValueError('Source and compact UTF8 text only')
    files.append(dict(path=name, bytes=len(b), sha256=sha(b),
                      git_blob_sha1=hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()))
manifest = dict(schema='C38O-action-design-v01',
                base_commit='0109bdcdf98423cafb3930aab5274e55dd605ef1',
                base_tree='dc3a65096632a33c17092a7ca84173d6cf7d16d3',
                component_inventories=inventories, files=files, file_count=len(files),
                total_bytes=sum(r['bytes'] for r in files),
                live_outputs_included=False, raw_fields_or_media_uploaded=False,
                dataset_admitted=False, campaign_complete=False)
p = ROOT / 'runs/campaign-c38o-paths.json'
p.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(files=len(files), bytes=manifest['total_bytes'], manifest_sha256=sha(p.read_bytes()))))
