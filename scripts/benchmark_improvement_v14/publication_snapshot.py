"""Snapshot campaign-only publication artifacts; never include local raw runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def prepare(stage):
    if not stage or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in stage):
        raise ValueError('Use a simple stage name')
    destination = ROOT / 'runs/benchmark-improvement-v14/publication' / stage
    if destination.exists():
        raise ValueError('Publication snapshots are immutable; use a new stage')
    candidates = list((ROOT / 'scripts/benchmark_improvement_v14').glob('*.py'))
    candidates += list((ROOT / 'reports/benchmark-improvement-v14').rglob('*'))
    candidates += list((ROOT / 'tests').glob('*v14*.py'))
    candidates += [ROOT / 'configs/benchmark_improvement_v14.json']
    selected = {}
    compressed = []
    for path in candidates:
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.tmp':
            continue
        relative = str(path.relative_to(ROOT))
        data = path.read_bytes()
        if path.suffix == '.json' and len(data) > 2_000_000:
            # Only derived evaluation reports may be compressed. Keep every
            # frozen protocol/inventory at its original path and exact bytes.
            if any(word in path.name for word in ('freeze', 'protocol', 'inventory')):
                selected[relative] = data
            else:
                selected[relative + '.gz'] = gzip.compress(data, compresslevel=9, mtime=0)
                compressed.append(dict(path=relative, published=relative + '.gz',
                                       sha256=hashlib.sha256(data).hexdigest()))
        else:
            selected[relative] = data
    selected['reports/benchmark-improvement-v14/compressed-reports.json'] = (
        json.dumps(dict(reports=compressed,
                        restore='gzip -dk <report.json.gz>; verify decompressed SHA-256 before auditing'),
                   indent=2) + '\n').encode()
    files = []
    for relative, data in sorted(selected.items()):
        previous = subprocess.run(['git', 'show', 'HEAD:' + relative], cwd=ROOT, capture_output=True)
        if previous.returncode == 0 and previous.stdout == data:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(dict(path=relative, snapshot=str(target), bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(),
                          git_sha=hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()))
    manifest = dict(parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    base_tree=subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
                    files=files)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(destination / 'manifest.json')
    print(len(files), 'changed files;', sum(f['bytes'] for f in files), 'bytes; raw datasets excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    prepare(parser.parse_args().stage)
