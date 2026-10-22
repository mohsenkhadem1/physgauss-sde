"""Publish completed controls and frozen next-phase source, never live outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sha = lambda b: hashlib.sha256(b).hexdigest()
components = [
    ('p2-torch-warp-hvp-benchmark-v02', 'publication-inventory.json'),
    ('c38-coarse-identity-v01', 'publication-inventory.json'),
    ('c38-graded-family-warp-v01', 'preparation-inventory.json'),
    ('p2-contact-root-integration-v01', 'publication-inventory.json'),
    ('p2-contact-root-fixed-v01', 'publication-inventory.json'),
    ('p2-contact-root-fixed-v02', 'publication-inventory.json'),
    ('p2-warp-root-contact-v01', 'publication-inventory.json'),
    ('c38-completed-figure-v01', 'publication-inventory.json'),
]
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
paths.update(['docs/GAUSSIAN_C38_SOLVER_UPDATE.md',
              'docs/GAUSSIAN_C38_FULL14_GRADED_DESIGN.md',
              str(Path(__file__).relative_to(ROOT))])
files = []
for name in sorted(paths):
    p = ROOT / name
    b = p.read_bytes()
    b.decode('utf-8')
    if p.is_symlink() or p.suffix not in ['.py', '.md', '.json', '.txt', '.log']:
        raise ValueError('Source and compact UTF8 text only')
    files.append(dict(path=name, bytes=len(b), sha256=sha(b),
                      git_blob_sha1=hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()))
manifest = dict(schema='C38I-solver-controls-v01',
                base_commit='8ec472a75784b48bd610848ec01b5a7c8bae1b5a',
                base_tree='cd6025ecb0e54d51156cf6ac1ca3f9424f128709',
                component_inventories=inventories, files=files, file_count=len(files),
                total_bytes=sum(r['bytes'] for r in files),
                live_outputs_included=False, raw_fields_or_media_uploaded=False,
                dataset_admitted=False, campaign_complete=False)
p = ROOT / 'runs/campaign-c38i-paths.json'
p.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(files=len(files), bytes=manifest['total_bytes'], manifest_sha256=sha(p.read_bytes()))))
