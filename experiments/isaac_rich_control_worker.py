"""Explicit TRAIN-only off-axis curriculum using the unchanged Isaac worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import isaac_calibrated_probes_v03 as base

ORIGINAL_MENU = base.calibrated_menu


def rich_menu(config, *args, **kwargs):
    menu = ORIGINAL_MENU(*args, **kwargs)
    additions = []
    for direction in config['raw_direction_vectors']:
        unit = np.asarray(direction, dtype=float)
        unit /= np.linalg.norm(unit)
        for magnitude in config['resultant_force_norms_N']:
            action = {**menu['candidates'][0], 'action_id': f'rich_{len(additions):02d}',
                'kind': 'train_force_angle_curriculum', 'peak_force_N': (float(magnitude)*unit).tolist()}
            if np.allclose(action['peak_force_N'], menu['evaluation_query']['peak_force_N'], atol=1e-12, rtol=0):
                raise ValueError('Independent evaluation force must not be a fitting control')
            additions.append(action)
    additions.append({**menu['candidates'][0], 'action_id': 'zero_force',
                      'kind': 'train_zero_force_curriculum', 'peak_force_N': [0., 0., 0.]})
    if [a['action_id'] for a in additions] != config['new_fitting_ids']:
        raise ValueError('Rich menu differs from frozen fitting IDs')
    menu['candidates'] += additions
    menu['protocol'] = config['protocol']
    menu['qualification'] = config['physics_policy']
    menu['fitting_ids'] = config['new_fitting_ids']
    menu['evaluation_remains_excluded'] = True
    return menu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/gaussian_rich_controls_v01.json'))
    p.add_argument('--episode', required=True)
    p.add_argument('--gpu', type=int, default=3)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    a.dataset, a.output = Path(config['dataset']), Path(config['output'])
    case = next(c for c in json.loads((a.dataset/'plan.json').read_text())['episodes'] if c['episode_id'] == a.episode)
    if case['split'] != 'train' or a.episode not in config['train_ids']:
        raise ValueError('Only frozen TRAIN curriculum layouts allowed')
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != config['base_worker_sha256']:
        raise ValueError('Verified worker source changed')
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = {'config_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest(),
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config': config, 'change': 'Public control menu only; physics and QC unchanged'}
    path = a.output/'rich-protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Rich curriculum source/configuration changed')
    else:
        with path.open('x') as f:
            json.dump(protocol, f, indent=2)
            f.write('\n')
    sources = a.output/'rich-provenance'
    sources.mkdir(exist_ok=True)
    target = sources/Path(__file__).name
    if target.exists() and target.read_bytes() != Path(__file__).read_bytes():
        raise ValueError('Rich curriculum wrapper changed')
    shutil.copy2(__file__, target)
    base.calibrated_menu = lambda *args, **kwargs: rich_menu(config, *args, **kwargs)
    return base.run_worker(a)


if __name__ == '__main__':
    raise SystemExit(main())
