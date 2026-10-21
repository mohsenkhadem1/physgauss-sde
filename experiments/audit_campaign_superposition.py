"""TRAIN-only representability check for force-linear trajectory predictors.

Independent diagonal TRAIN responses are opened ONLY for this physical/model
class diagnostic, never as teacher weight-fitting targets. Moving supports are
excluded: without a zero-force response four controls cannot identify an affine
linearity residual independently of the unknown support-motion offset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linearity_diagnostics(force_vectors, displacement):
    """Exact least-squares lower bound for a force-linear map, at fixed context.

    The first three vectors span3D and the fourth is a withheld combination.
    Fits independent response coefficients at every time/node/coordinate,
    imposing no neural capacity, smoothness or physics restrictions.
    """
    force = np.asarray(force_vectors, dtype=float)
    y = np.asarray(displacement, dtype=float)
    if force.shape != (4, 3) or y.shape[0] != 4 or not np.isfinite(y).all():
        raise ValueError('Four finite matched responses and3Dforces required')
    if np.linalg.matrix_rank(force[:3]) != 3:
        raise ValueError('Firstthreeforcevectors must span3D')
    coefficients = np.linalg.solve(force[:3].T, force[3])
    predicted = np.tensordot(coefficients, y[:3], axes=1)
    flattened = y.reshape(4, -1)
    reconstructed = force@np.linalg.lstsq(force, flattened, rcond=None)[0]
    residual = reconstructed-flattened
    rmse = lambda a: float(np.sqrt(np.mean(np.asarray(a)**2)))
    static = rmse(y[3])
    return {'combination_coefficients': coefficients.tolist(),
        'diagonal_from_three_axis_rmse_m': rmse(predicted-y[3]),
        'diagonal_static_rmse_m': static,
        'diagonal_prediction_rms_motion_m': rmse(predicted),
        'diagonal_amplitude_ratio': rmse(predicted)/max(static, 1e-30),
        'diagonal_residual_to_static': rmse(predicted-y[3])/max(static, 1e-30),
        'best_joint_force_linear_pooled_rmse_m': rmse(residual),
        'joint_static_pooled_rmse_m': rmse(y),
        'best_joint_residual_fraction': rmse(residual)/max(rmse(y), 1e-30)}


def run(campaign, output):
    config = json.loads(campaign.read_text())
    dataset, pool = Path(config['dataset']), Path(config['old_train_pool'])
    cases = {c['episode_id']: c for c in json.loads((dataset/'plan.json').read_text())['episodes']}
    roles = json.loads(Path('configs/gaussian_probe_roles_v01.json').read_text())
    rows, skipped, hashes = [], [], {str(campaign): sha(campaign)}
    for eid in roles['teacher_fit_ids']:
        case = cases[eid]
        if case['split'] != 'train':
            raise ValueError('Only TRAIN can enter the representability diagnostic')
        if case['boundary_type'] == 'moving_patch':
            skipped.append({'episode_id': eid, 'reason': 'Needs separately measured zero-force support-motion response'})
            continue
        private = pool/'private'/eid
        if not json.loads((private/'validation.json').read_text())['passed']:
            raise ValueError('Unaudited simulator responses')
        with np.load(dataset/'episodes'/eid/'reference.npz') as raw:
            rest = raw['rest_points_m']
        menu_path = pool/'public'/eid/'action-menu.json'
        menu = json.loads(menu_path.read_text())
        actions = menu['candidates'][:3]+[menu['evaluation_query']]
        if any(a['force_type_index'] != 0 or a.get('start_s', 0) != 0 for a in actions):
            raise ValueError('Identical pulse envelope and start time required')
        displacement, times, force, step_commands = [], None, [], []
        for action in actions:
            path = private/action['action_id']/'trajectory.npz'
            with np.load(path) as raw:
                if times is None:
                    times = raw['times_s']
                np.testing.assert_array_equal(raw['times_s'], times)
                np.testing.assert_allclose(raw['boundary_target_positions_m'],
                    np.broadcast_to(rest[np.array(json.loads((private/'truth/boundaries.json').read_text())['fixture_nodes'])], raw['boundary_target_positions_m'].shape), atol=1e-7)
                displacement.append(raw['positions_m']-rest)
                step_commands.append(raw['force_step_commands_N'])
            force.append(action['peak_force_N'])
            hashes[str(path)] = sha(path)
        result = linearity_diagnostics(force, displacement)
        combined_command = np.tensordot(result['combination_coefficients'], step_commands[:3], axes=1)
        command_error = float(np.max(np.abs(combined_command-step_commands[3])))
        if command_error > 1e-8:
            raise ValueError('Actual logged forcecommands do not superpose')
        rows.append({'episode_id': eid, 'boundary_type': case['boundary_type'],
                     'force_combination_max_command_error_N': command_error, **result})
        hashes[str(menu_path)] = sha(menu_path)
    if len(rows) != 9:
        raise ValueError('Expected9nonmoving TRAIN layouts')
    result = {'campaign_sha256': sha(campaign), 'train_layouts': len(rows), 'rows': rows, 'skipped': skipped,
        'diagonal_used_for': 'TRAIN-only model-class diagnosis, no weight fitting or calibration',
        'test_labels_opened': False, 'validation_labels_opened': False, 'weights_fitted': False,
        'mean_diagonal_combination_rmse_m': float(np.mean([r['diagonal_from_three_axis_rmse_m'] for r in rows])),
        'mean_diagonal_static_rmse_m': float(np.mean([r['diagonal_static_rmse_m'] for r in rows])),
        'mean_best_joint_residual_fraction': float(np.mean([r['best_joint_residual_fraction'] for r in rows])),
        'input_sha256': hashes, 'source_sha256': sha(__file__),
        'qualification': 'Tests compatibility of this fixed-discretization Isaac data with exact force linearity. Residual may include finite-deformation effects, floating-point/solver artefacts or reset discrepancies; this audit does not identify which. Joint floor is pooled coordinate MSE, not a lower bound on each separate control RMSE.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['train_layouts', 'mean_diagonal_combination_rmse_m', 'mean_diagonal_static_rmse_m', 'mean_best_joint_residual_fraction']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    p.add_argument('--output', type=Path, default=Path('runs/gaussian-campaign-v01/superposition-audit.json'))
    a = p.parse_args()
    run(a.campaign, a.output)
