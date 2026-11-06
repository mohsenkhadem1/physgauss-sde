"""Audit matched budgets, implementation identity and saved field states."""
from pathlib import Path
import hashlib
import json
import shutil
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/pidg-dense-mechanics-v01'
SCENES = ['balls_reaction', 'rubber_duck', 'motion_kuro']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def count_points(p):
    with p.open('rb') as f:
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                return int(line.split()[-1])
            if line == 'end_header':
                raise ValueError('Missing vertex count')

records = []
for scene in SCENES:
    folder = ROOT / 'runs/pidg-dense-mechanics-v01' / scene
    result = json.loads((REPORT / scene / 'results.json').read_text())
    records_scene = []
    for arm, weight in [('native-refined', 0), ('dense-1e-5', 1e-5), ('dense-1e-4', 1e-4), ('dense-1e-3', 1e-3)]:
        run = folder / arm
        metadata = json.loads((run / 'run.json').read_text())
        args = metadata['arguments']
        assert metadata['status'] == 'training_complete'
        assert args['steps'] == 5000 and args['width'] == 320 and args['dense_weight'] == weight
        assert args['scene'] == scene and args['flow'] == (scene != 'rubber_duck')
        count = count_points(run / 'model/point_cloud/iteration_5000/point_cloud.ply')
        record = dict(arm=arm, steps=args['steps'], points=count, warm_from=str((ROOT / args['warm_from']).resolve()), weight=weight)
        if weight:
            expected = metadata['source_sha256']['scripts/pidg_dense_mechanics/field.py']
            assert expected == sha(ROOT / 'scripts/pidg_dense_mechanics/field.py')
            state = torch.load(run / 'dense-mechanics.pt', map_location='cpu', weights_only=False)
            assert state['last']['calls'] == 5000
            assert all(torch.isfinite(t).all() for t in state['state_dict'].values())
            record.update(field_sha256=expected, field_finite=True, calls=5000,
                          load_rms=float(state['state_dict']['load'].square().mean().sqrt()),
                          last_balance=state['last']['balance'],
                          last_inversion_fraction=state['last']['inversion_fraction'])
        dest = REPORT / scene / 'run-metadata'
        dest.mkdir(exist_ok=True)
        shutil.copy2(run / 'run.json', dest / (arm + '.json'))
        records_scene.append(record)
    assert len({r['points'] for r in records_scene}) == 1
    assert len({r['warm_from'] for r in records_scene}) == 1
    base = json.loads((Path(records_scene[0]['warm_from']) / 'run.json').read_text())
    assert base['status'] == 'training_complete' and base['arguments']['steps'] == 50000
    assert base['arguments']['width'] == 320
    records.append(dict(scene=scene, matched_capacity=True, matched_base=True, base_steps=50000, arms=records_scene))
out = dict(passed=True, scenes=records, scope='Implementation, budgets, capacity, finite learned fields. Does not certify physical identification or residual convergence.')
(REPORT / 'campaign-audit.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'passed': True, 'scenes': SCENES}))
