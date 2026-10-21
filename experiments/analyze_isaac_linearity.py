"""Analyze only independently verified zero-force diagnostic cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def affine_diagnostics(forces, trajectories, zero_index=3, target_index=4):
    f = np.asarray(forces, dtype=float)
    q = np.asarray(trajectories, dtype=float)
    if f.shape != (5, 3) or q.shape[0] != 5:
        raise ValueError('Expected3axis, zero and diagonal responses')
    if not np.array_equal(f[zero_index], np.zeros(3)):
        raise ValueError('Measured zero response required')
    coefficients = np.linalg.solve(f[:3].T, f[target_index])
    zero = q[zero_index]
    combined = zero+np.tensordot(coefficients, q[:3]-zero, axes=1)
    design = np.c_[np.ones(5), f]
    y = (q-zero).reshape(5, -1)
    residual = design@np.linalg.lstsq(design, y, rcond=None)[0]-y
    rmse = lambda a: float(np.sqrt(np.mean(np.asarray(a)**2)))
    return {'diagonal_zero_corrected_combination_rmse_m': rmse(combined-q[target_index]),
        'force_induced_diagonal_rms_m': rmse(q[target_index]-zero),
        'best_affine_pooled_rmse_m': rmse(residual),
        'pooled_force_induced_rms_m': rmse(y),
        'best_affine_residual_fraction': rmse(residual)/max(rmse(y), 1e-30),
        'combination_coefficients': coefficients.tolist()}, combined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, default=Path('runs/gaussian-campaign-v01/linearity-diagnostic'))
    p.add_argument('--original', type=Path, default=Path('runs/isaac-calibrated-probes-train-v01'))
    p.add_argument('--episodes', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows, inputs = [], {}
    for eid in a.episodes:
        private = a.pool/'private'/eid
        if not json.loads((private/'validation.json').read_text())['passed']:
            raise ValueError('Case not fully independently verified')
        case = json.loads((private/'truth/recipe.json').read_text())
        if case['split'] != 'train':
            raise ValueError('TRAIN diagnostic only')
        with np.load(private/'reference.npz') as raw:
            rest = raw['rest_points_m']
        menu = json.loads((a.pool/'public'/eid/'action-menu.json').read_text())
        actions = menu['candidates']+[menu['evaluation_query']]
        if [x['action_id'] for x in actions] != ['poke_0', 'poke_1', 'poke_2', 'zero_force', 'evaluation']:
            raise ValueError('Unexpected zero-force diagnostic menu')
        q, commands, times = [], [], None
        for action in actions:
            path = private/action['action_id']/'trajectory.npz'
            with np.load(path) as raw:
                if times is None:
                    times = raw['times_s']
                np.testing.assert_array_equal(raw['times_s'], times)
                q.append(raw['positions_m'])
                commands.append(raw['force_step_commands_N'])
            inputs[str(path)] = sha(path)
        q = np.asarray(q)
        metrics, combined = affine_diagnostics([x['peak_force_N'] for x in actions], q)
        zero_rms = float(np.sqrt(np.mean((q[3]-rest)**2)))
        original_path = a.original/'private'/eid/'poke_0/trajectory.npz'
        with np.load(original_path) as raw:
            old_q = raw['positions_m']
        command_difference = np.tensordot(metrics['combination_coefficients'], commands[:3], axes=1)-commands[4]
        row = {'episode_id': eid, 'boundary_type': case['boundary_type'], **metrics,
            'zero_force_coordinate_rms_from_rest_m': zero_rms,
            'old_vs_new_axis_replay_coordinate_rmse_m': float(np.sqrt(np.mean((q[0]-old_q)**2))),
            'old_vs_new_axis_replay_max_abs_m': float(np.max(np.abs(q[0]-old_q))),
            'old_vs_new_axis_bitwise_equal': bool(np.array_equal(q[0], old_q)),
            'logged_force_combination_max_error_N': float(np.max(np.abs(command_difference)))}
        if row['logged_force_combination_max_error_N'] > 1e-8:
            raise ValueError('Forcing itself fails superposition')
        rows.append(row)
        inputs[str(original_path)] = sha(original_path)
        for path in [private/'validation.json', private/'reference.npz', private/'truth/recipe.json', a.pool/'diagnostic-protocol.json']:
            inputs[str(path)] = sha(path)
        # Save exact curves used for visualization without magnifying motion.
        curves = {'times_s': times,
            'diagonal_rms_motion_mm': np.sqrt(np.mean((q[4]-rest)**2, axis=(1, 2)))*1000,
            'zero_corrected_combination_rms_motion_mm': np.sqrt(np.mean((combined-rest)**2, axis=(1, 2)))*1000,
            'zero_force_rms_motion_mm': np.sqrt(np.mean((q[3]-rest)**2, axis=(1, 2)))*1000}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.output.parent/(eid+'-linearity-curves.npz'), **curves)
    result = {'rows': rows, 'input_sha256': inputs, 'source_sha256': sha(__file__),
        'test_labels_opened': False, 'weights_fitted': False,
        'qualification': 'Measured zero response separates a constant drift/offset from the force-linearity residual; it does not distinguish finite-deformation physics from solver/discretization effects. Diagonal TRAIN data are diagnostic only.'}
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
