"""Fixed-case secondary-search videos using the unchanged world-scale renderer."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'runs/coupled-friction-v13'
REPORT = ROOT/'reports/coupled-friction-v13'
PARENT = ROOT/'runs/coupled-surface-v13'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Routes:
    def __truediv__(self, key):
        return (PARENT if str(key) == 'reference' else BASE)/key


def main():
    freeze = json.loads((REPORT/'method-freeze.json').read_text())
    if digest(REPORT/'protocol.json') != freeze['protocol_sha256']:
        raise RuntimeError('Secondary protocol changed')
    for path, expected in freeze['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Secondary source changed: '+path)
    audit_path = REPORT/'independent-audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit.get('passed') or audit.get('errors') or audit.get('pending'):
        raise RuntimeError('Completed independent secondary audit required')
    for path, expected in audit['evidence_sha256'].items():
        if digest(Path(path)) != expected:
            raise RuntimeError('Secondary audit evidence changed: '+path)
    source = ROOT/'scripts/coupled_contact_v13/surface_render.py'
    name = 'scripts.coupled_contact_v13._fine_grid_surface_renderer'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.BASE = Routes()
    module.REPORT = REPORT
    module.main()
    provenance = REPORT/'video-provenance.json'
    rows = json.loads(provenance.read_text())
    for row in rows:
        row['study'] = ('Secondary uniform friction-grid refinement; unchanged physics, '
                        'shared world scale and camera; quarter-speed playback')
        row['fine_grid_renderer_sha256'] = digest(Path(__file__))
        row['independent_audit_sha256'] = digest(audit_path)
        row['secondary_protocol_sha256'] = digest(REPORT/'protocol.json')
        row['secondary_method_freeze_sha256'] = digest(REPORT/'method-freeze.json')
    provenance.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
