"""Explicit fitting/selection roles for asynchronously captured campaign controls.

No gate, calibration or TEST response is accessible through this training API.
Snapshots use fixed prefixes of the predeclared TRAIN queue, never whichever
cases happen to finish successfully first.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from gaussian_operator import physical_vector, category_labels
from diagnose_gaussian_forward import control_physical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_role(config, episode_id, purpose):
    if purpose == 'fit' and episode_id in config['new_train_control_ids']:
        return 'train', Path(config['expanded_train_pool'])
    if purpose == 'select' and episode_id in config['new_validation_roles']['selection']:
        return 'validation', Path(config['expanded_validation_pool'])
    raise ValueError('Episode/purpose is not an authorized fitting or selection role')


def load_control_snapshot(config, episode_ids, purpose):
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Distinct nonempty fixed episode list required')
    # Establish ALL role eligibility before any case-private path is opened.
    roles = [require_role(config, eid, purpose) for eid in episode_ids]
    dataset = Path(config['dataset'])
    cases = {c['episode_id']: c for c in json.loads((dataset/'plan.json').read_text())['episodes']}
    result = []
    expected_force = np.array([[.02, 0, 0], [0, .02, 0], [0, 0, .02], [0, 0, .02]])
    for eid, (split, pool) in zip(episode_ids, roles):
        case = cases[eid]
        if case['split'] != split:
            raise ValueError('Case split differs from declared role')
        private = pool/'private'/eid
        for name in ['validation.json', 'summary.json']:
            if not (private/name).is_file():
                raise FileNotFoundError('Fixed snapshot incomplete: '+eid)
        if not json.loads((private/'validation.json').read_text()).get('passed'):
            raise ValueError('Control capture failed audit: '+eid)
        summary = json.loads((private/'summary.json').read_text())
        if not summary.get('completed') or (private/'failure.json').exists():
            raise ValueError('Control generation incomplete/failed')
        source = summary['source_sha256']
        if source.get('isaac_calibrated_probes_v03.py') != config['simulation']['worker_sha256']:
            raise ValueError('Unexpected control generation source')
        for name, expected in source.items():
            if Path(name).name != name or sha(private/'provenance'/name) != expected:
                raise ValueError('Control source provenance mismatch')
        if json.loads((private/'truth/recipe.json').read_text()) != case:
            raise ValueError('Control recipe changed')
        spec = json.loads((private/'truth/boundaries.json').read_text())
        if spec != json.loads((dataset/'episodes'/eid/'truth/boundaries.json').read_text()):
            raise ValueError('Control support/hardware geometry differs from original')
        with np.load(dataset/'episodes'/eid/'reference.npz') as raw:
            rest, tets = raw['rest_points_m'].astype(np.float32), raw['tetrahedra']
        physical = physical_vector(case, spec, rest)
        bc, _ = category_labels(case)
        menu = json.loads((pool/'public'/eid/'action-menu.json').read_text())
        actions = menu['candidates']
        if [a['action_id'] for a in actions] != ['poke_0', 'poke_1', 'poke_2', 'poke_3']:
            raise ValueError('Only original four candidate controls may fit/select')
        for index, action in enumerate(actions):
            np.testing.assert_allclose(action['peak_force_N'], expected_force[index], atol=1e-12)
            if action['force_type_index'] != (0 if index < 3 else 2):
                raise ValueError('Wrong control temporal family')
            path = private/action['action_id']
            if json.loads((path/'control.json').read_text()) != action:
                raise ValueError('Private control differs from public menu')
            with np.load(path/'trajectory.npz') as raw:
                q = raw['positions_m'].astype(np.float32)
                times = raw['times_s'].astype(np.float32)
            if q.shape != (31, len(rest), 3) or not np.isfinite(q).all():
                raise ValueError('Invalid dense control trajectory')
            np.testing.assert_allclose(times, np.arange(31)/10, atol=1e-6)
            certificates = json.loads((path/'capture-audit.json').read_text())
            if len(certificates) != 31 or any(not c['passed'] or
                    hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() != c['positions_sha256']
                    for frame, c in zip(q, certificates)):
                raise ValueError('Saved trajectory differs from audited sensor state')
            result.append({'path': str(path), 'split': split, 'role': 'teacher_fit_control' if purpose == 'fit' else 'selection_control',
                'episode_id': eid+'/'+action['action_id'], 'group_id': case['group_id'],
                'physical': control_physical(physical, action, menu['known_handle_centers_m']),
                'bc': bc, 'force': action['force_type_index'], 'rest': rest, 'tets': tets, 'q': q, 'times': times,
                'scale_m': max(float(np.sqrt(np.mean((q-rest)**2))), .00004),
                'trajectory_sha256': sha(path/'trajectory.npz'), 'recipe_sha256': case['recipe_sha256'],
                'generation_validation_sha256': sha(private/'validation.json')})
    return result
