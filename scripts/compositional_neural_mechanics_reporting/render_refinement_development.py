"""Visualize both exposed action-pilot objects without refitting or selecting a winner.

The first held command is prospectively used for both objects. Static camera
orbit, not a dynamic trajectory. Fresh dense queries are visualization only.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build, pack
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.render_comparison import render

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'reports/compositional-neural-mechanics-v16'
B = ROOT/'runs/compositional-neural-mechanics-v16'


def read(path):
    return json.loads(path.read_text())


def dense(problem, parameters, pose, points):
    model = make_model(problem.scene, **problem.kw)
    problem.bind(model, torch.tensor(parameters, dtype=torch.float64))
    u = None
    for fraction in problem.continuation:
        model.set_controls(pose*fraction)
        u = model.solve(initial=u, force_tolerance_N=1e-10, maximum_iterations=60)['displacement_local_m']
    return model.sample_material(u, points)


def main():
    torch.set_num_threads(1)
    audit = read(R/'local-action-final-independent-audit.json')
    if not (audit.get('passed') or audit.get('status') == 'passed'):
        raise ValueError('Completed independent action audit required')
    folder = B/'refinement-development-visualization'
    folder.mkdir(exist_ok=False)
    q = np.stack(np.meshgrid(*([np.linspace(0, 1, 7)]*3), indexing='ij'), -1).reshape(-1, 3)
    manifest = {'scope': __doc__, 'command_index': 0, 'worlds': [0, 1], 'inputs': {}}
    for world in (0, 1):
        pilot = B/f'local-action-development/world{world}'
        paths = [pilot/name for name in ('public.json', 'initial-fit.json', 'fits.json', 'selection.json')]
        paths += [R/f'full-continuous-inference-pilot-world{world}.json']
        for path in paths:
            manifest['inputs'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        public, initial, fits, selection, original = map(read, paths)
        geometry = GeometrySpec(**public['geometry'])
        pose = np.asarray(public['held_poses'][0])
        dummy = [dict(pose=pose, points=q, displacement=np.zeros_like(q))]
        action = selection['policies']['target_task_variance']
        problems = [build(geometry, public['categories'], dummy, learned) for learned in (False, True, True)]
        parameters = [pack(original['scene']), initial['parameters'], fits[action]['parameters']]
        fields = []
        for name, problem, theta in zip(('reference', 'initial', 'refined'), problems, parameters):
            path = folder/f'world{world}-{name}.npz'
            np.savez_compressed(path, field=dense(problem, theta, pose, q))
            fields.append(path)
        render(geometry, fields,
               ['Actual simulation (n=2)', 'Prediction before extra observation', f'Prediction after {action}'],
               R/f'videos/observation-refinement-development-world{world}.mp4', frames=120)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
