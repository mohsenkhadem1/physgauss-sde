"""TRAIN-only audit of neural label conventions against actual logged commands."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from gaussian_operator import physical_vector, category_labels, force_envelope
from diagnose_gaussian_forward import control_physical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(case, spec, rest, trajectory, action=None, handles=None):
    physical = physical_vector(case, spec, rest)
    bc, family = category_labels(case)
    if action:
        physical = control_physical(physical, action, handles)
        family = int(action['force_type_index'])
    with np.load(trajectory) as raw:
        midpoint = raw['force_step_midpoints_s']
        applied = raw['force_step_commands_N']
        times = raw['times_s']
        targets = raw['boundary_target_positions_m']
        simulated = raw['simulated_times_s']
    envelope = force_envelope(torch.tensor([family]), torch.tensor(midpoint, dtype=torch.float64), case['seconds']).numpy()[0]
    expected = envelope[:, None, None]*physical[16:22].reshape(2, 3)
    force_error = float(np.max(np.abs(expected-applied)))
    if bc != 3 and np.any(physical[22:25]):
        raise ValueError('Inactive support motion encoded as active')
    shift = np.sin(np.pi*times/case['seconds'])[:, None]**2*physical[22:25]
    expected_targets = rest[np.asarray(spec['fixture_nodes'])][None]+shift[:, None]
    target_error = float(np.max(np.abs(expected_targets-targets)))
    dt_error = float(np.max(np.abs(np.diff(midpoint)-1/case['physics_hz'])))
    clock_error = float(np.max(np.abs(simulated-times)))
    handles_error = float(np.max(np.abs(physical[10:16].reshape(2, 3)-rest[spec['load_nodes']])))
    return {'force_max_error_N': force_error, 'prescribed_target_max_error_m': target_error,
        'step_size_max_error_s': dt_error, 'sensor_clock_max_error_s': clock_error,
        'encoded_handle_max_error_m': handles_error,
        'passed': force_error < 1e-7 and target_error < 1e-7 and dt_error < 1e-10 and clock_error < 1e-9 and handles_error < 1e-7,
        'trajectory_sha256': sha(trajectory)}


def run(campaign, output):
    config = json.loads(campaign.read_text())
    rows = []
    for root in [Path(config['original_dataset']), Path(config['dataset'])]:
        for case in json.loads((root/'plan.json').read_text())['episodes']:
            # Assert eligibility BEFORE any private label or command log read.
            if case['split'] != 'train':
                continue
            episode = root/'episodes'/case['episode_id']
            if not json.loads((episode/'validation.json').read_text())['passed']:
                raise ValueError('Source capture unaudited')
            with np.load(episode/'reference.npz') as raw:
                rest = raw['rest_points_m']
            spec = json.loads((episode/'truth/boundaries.json').read_text())
            rows.append({'dataset': str(root), 'episode_id': case['episode_id'], 'kind': 'passive',
                         **check(case, spec, rest, episode/'truth/trajectory.npz')})
    root, pool = Path(config['dataset']), Path(config['old_train_pool'])
    cases = {c['episode_id']: c for c in json.loads((root/'plan.json').read_text())['episodes']}
    roles = json.loads(Path('configs/gaussian_probe_roles_v01.json').read_text())
    for eid in roles['teacher_fit_ids']:
        case = cases[eid]
        if case['split'] != 'train':
            raise ValueError('TRAIN controls only')
        with np.load(root/'episodes'/eid/'reference.npz') as raw:
            rest = raw['rest_points_m']
        private = pool/'private'/eid
        spec = json.loads((private/'truth/boundaries.json').read_text())
        menu = json.loads((pool/'public'/eid/'action-menu.json').read_text())
        for action in menu['candidates']:
            if action['action_id'] not in roles['eligible_control_ids']:
                raise ValueError('Only teacher candidate controls')
            rows.append({'dataset': str(pool), 'episode_id': eid+'/'+action['action_id'], 'kind': 'control',
                **check(case, spec, rest, private/action['action_id']/'trajectory.npz', action, menu['known_handle_centers_m'])})
    if len(rows) != 200:
        raise ValueError('Expected152passive plus48control TRAIN records')
    result = {'campaign_sha256': sha(campaign), 'records': len(rows), 'passed': all(r['passed'] for r in rows),
        'test_labels_opened': False, 'validation_labels_opened': False, 'rows': rows,
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in
            ['audit_campaign_commands.py', 'gaussian_operator.py', 'diagnose_gaussian_forward.py']},
        'qualification': 'Command, label, hardware and time-convention agreement only; no guarantee of accurate learned mechanics or transmitted handle forces'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({key: result[key] for key in ['records', 'passed', 'test_labels_opened', 'validation_labels_opened']}))
    return not result['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    p.add_argument('--output', type=Path, default=Path('runs/gaussian-campaign-v01/command-audit.json'))
    args = p.parse_args()
    raise SystemExit(run(args.campaign, args.output))
