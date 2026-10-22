"""Publish completed controls and frozen next-phase source, never live outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sha = lambda b: hashlib.sha256(b).hexdigest()
components = [('p2-corrected-contact-video-v01', 'publication-inventory.json'), ('c38-graded-family-warp-v01', 'completed-publication-inventory.json'), ('p2-warp-root-contact-gpu-v02', 'publication-inventory.json'), ('p2-contact-bulk-quadrature-v01', 'publication-inventory.json'), ('p2-warp-reference-force-v01', 'publication-inventory.json'), ('p2-corrected-contact-reference-v01', 'preparation-publication-inventory.json'), ('p2-corrected-contact-reference-v01', 'result-publication-inventory.json'), ('p2-historical-reference-test-v01', 'publication-inventory.json'), ('c38-prediction-pipeline-audit-v01', 'publication-inventory.json'), ('c38-static-study-source-v01', 'publication-inventory.json'), ('c38-static-export-source-v01', 'publication-inventory.json'), ('c38-static-export-source-v02', 'publication-inventory.json'), ('c38-static-linear-source-v01', 'publication-inventory.json'), ('c38-static-linear-predictions-v01', 'preparation-inventory.json'), ('c38-nonlinear-table-draft-v01', 'source-design-inventory.json'), ('p2-supported-gravity-fine-v01', 'completed-publication-inventory.json'), ('p2-corrected-contact-order32-v01', 'publication-inventory.json'), ('p2-corrected-contact-order128-v01', 'publication-inventory.json')]
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
paths.update(['docs/GAUSSIAN_C38_FINITE_REFERENCE_RESULTS.md',
              'docs/GAUSSIAN_C38_NONLINEAR_TABLE_DESIGN.md', 'docs/GAUSSIAN_C38_STATIC_LINEAR_BASELINE.md',
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
manifest = dict(schema='C38J-finite-reference-v01',
                base_commit='9e041c989d2fe36ca143c973c27148f88836219b',
                base_tree='84a4c521673f040a88a5d4da3fab3ef0b1be8452',
                component_inventories=inventories, files=files, file_count=len(files),
                total_bytes=sum(r['bytes'] for r in files),
                live_outputs_included=False, raw_fields_or_media_uploaded=False,
                dataset_admitted=False, campaign_complete=False)
p = ROOT / 'runs/campaign-c38j-paths.json'
p.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(files=len(files), bytes=manifest['total_bytes'], manifest_sha256=sha(p.read_bytes()))))
