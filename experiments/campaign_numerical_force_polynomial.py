"""TRAIN-only force-map diagnosis; never establishes continuum accuracy.

Fit independent linear/quadratic maps at every saved time/node/coordinate using
only matched pulse controls. The independently captured diagonal is opened only
after both coefficient arrays and predictions have been frozen and hashed.
No network, material inference, reserved control gate or calibration is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_train_ids(ids, declared):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Distinct nonempty TRAIN IDs required')
    if any(not re.fullmatch(r't\d{4}', eid) or eid not in declared for eid in ids):
        raise ValueError('Only the frozen rich-curriculum TRAIN IDs may be opened')


def force_features(force, degree, scale_N=.02):
    f = np.asarray(force, dtype=np.float64)/scale_N
    if f.shape[-1] != 3 or not np.isfinite(f).all() or scale_N <= 0:
        raise ValueError('Finite three-component force and positive scale required')
    if degree == 1:
        return f
    if degree != 2:
        raise ValueError('Only prespecified linear and quadratic maps supported')
    x, y, z = np.moveaxis(f, -1, 0)
    return np.concatenate((f, np.stack((x*x, y*y, z*z, x*y, x*z, y*z), -1)), -1)


def fit_response_map(forces, responses, degree):
    design = force_features(forces, degree)
    y = np.asarray(responses, dtype=np.float64)
    if design.ndim != 2 or len(y) != len(design) or not np.isfinite(y).all():
        raise ValueError('Finite matched control responses required')
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError('Force design does not identify this polynomial: rank '+str(rank))
    coefficients = np.linalg.lstsq(design, y.reshape(len(y), -1), rcond=None)[0]
    prediction = (design@coefficients).reshape(y.shape)
    return coefficients.reshape((design.shape[1],)+y.shape[1:]), {
        'rank': int(rank), 'columns': int(design.shape[1]),
        'condition_number': float(np.linalg.cond(design)),
        'fit_coordinate_rmse_m': rms(prediction-y),
        'qualification': 'Unregularized fixed per-layout least squares; no held-diagonal selection'}


def predict_response(coefficients, force, degree):
    return np.tensordot(force_features(force, degree), coefficients, axes=([-1], [0]))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def finite_strain(rest, tets, q):
    rest, q = np.asarray(rest, dtype=float), np.asarray(q, dtype=float)
    dm = rest[tets[:, 1:]]-rest[tets[:, :1]]
    ds = q[..., tets[:, 1:], :]-q[..., tets[:, :1], :]
    # Row-edge convention: deformation gradient is transpose(inv(Dm)@Ds).
    f = np.swapaxes(np.linalg.inv(dm)@ds, -1, -2)
    green = .5*(np.swapaxes(f, -1, -2)@f-np.eye(3))
    principal = np.linalg.svd(f, compute_uv=False)
    return {'minimum_volume_ratio': float(np.linalg.det(f).min()),
            'maximum_absolute_principal_stretch_minus_one': float(np.abs(principal-1).max()),
            'green_strain_frobenius_p95': float(np.quantile(np.linalg.norm(green, axis=(-1,-2)), .95)),
            'green_strain_frobenius_max': float(np.linalg.norm(green, axis=(-1,-2)).max()),
            'qualification': 'Kinematic strain, not a bound on nonlinear force-response error'}


def output_precision_audit(q, velocity, physics_hz):
    q32 = np.asarray(q, dtype=np.float32)
    ulp = np.abs(np.spacing(q32)).astype(float)
    # Exclude subnormal/near-origin spacing from this scale diagnostic.
    moving = np.abs(velocity) > 1e-12
    return {'saved_dtype': str(q.dtype),
            'float32_output_ulp_max_m': float(ulp.max()),
            'float32_output_ulp_median_m': float(np.median(ulp)),
            'sampled_velocity_dt_below_one_output_ulp_fraction':
                float(np.mean(np.abs(velocity[moving])/physics_hz < ulp[moving])) if moving.any() else None,
            'qualification': 'Output representation only. Velocity*dt is a scale estimate at sampled times; it does not expose integration arithmetic or prove accumulated roundoff.'}


def run(pool, dataset, episodes, output):
    pool, dataset, output = map(Path, (pool, dataset, output))
    protocol_path = pool/'rich-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    require_train_ids(episodes, protocol['config']['train_ids'])
    plan_path = dataset/'plan.json'
    cases = {c['episode_id']: c for c in json.loads(plan_path.read_text())['episodes']}
    # Check ALL roles before opening any private response path.
    if any(cases[eid]['split'] != 'train' for eid in episodes):
        raise ValueError('Dataset plan split differs from TRAIN role')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): sha(p) for p in [protocol_path, plan_path]}
    rows, skipped = [], []
    for eid in episodes:
        private = pool/'private'/eid
        if not all((private/name).is_file() for name in ['summary.json', 'validation.json']):
            skipped.append({'episode_id': eid, 'reason': 'Not fully completed and validated; no trajectories opened'})
            continue
        summary = json.loads((private/'summary.json').read_text())
        validation = json.loads((private/'validation.json').read_text())
        if not summary.get('completed') or not validation.get('passed') or (private/'failure.json').exists():
            raise ValueError('Completed case has failed/partial validation: '+eid)
        for name, expected in summary['source_sha256'].items():
            source = private/'provenance'/name
            if Path(name).name != name or sha(source) != expected:
                raise ValueError('Simulator provenance mismatch')
            hashes[str(source)] = expected
        for name in ['summary.json', 'validation.json']:
            path = private/name; hashes[str(path)] = sha(path)
        menu_path = pool/'public'/eid/'action-menu.json'
        menu = json.loads(menu_path.read_text()); hashes[str(menu_path)] = sha(menu_path)
        candidates = {a['action_id']: a for a in menu['candidates']}
        names = ['poke_0', 'poke_1', 'poke_2']+[f'rich_{i:02d}' for i in range(8)]
        if any(candidates[n]['force_type'] != 'push_pulse' or candidates[n]['handle_index'] != 0 for n in names+['zero_force']):
            raise ValueError('Only matched pulse family/handle may share a polynomial')
        def load(name, action):
            branch = private/name
            if not (branch/'summary.json').is_file():
                raise ValueError('Missing completed action summary')
            control_path = branch/'control.json'
            if json.loads(control_path.read_text()) != action:
                raise ValueError('Private/public command mismatch')
            path = branch/'trajectory.npz'
            with np.load(path) as raw:
                data = {k: raw[k].copy() for k in ['times_s','positions_m','velocities_m_s',
                    'force_step_midpoints_s','force_step_commands_N','force_step_net_moments_Nm']}
            if not all(np.isfinite(x).all() for x in data.values()):
                raise ValueError('Nonfinite recorded data')
            cert_path = branch/'capture-audit.json'
            certs = json.loads(cert_path.read_text())
            if len(certs) != len(data['positions_m']) or any(not c['passed'] or
                hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest() != c['positions_sha256']
                for q,c in zip(data['positions_m'], certs)):
                raise ValueError('Trajectory differs from independently audited sensor state')
            for p in [path, control_path, cert_path, branch/'summary.json']:
                hashes[str(p)] = sha(p)
            return data
        zero = load('zero_force', candidates['zero_force'])
        if np.any(zero['force_step_commands_N']):
            raise ValueError('Nonzero logged zero-force command')
        controls = [load(n, candidates[n]) for n in names]
        forces = np.array([candidates[n]['peak_force_N'] for n in names])
        for data in controls:
            np.testing.assert_array_equal(data['times_s'], zero['times_s'])
            np.testing.assert_array_equal(data['force_step_midpoints_s'], zero['force_step_midpoints_s'])
        commands = controls[0]['force_step_commands_N'][:,0,0]/forces[0,0]
        for force,data in zip(forces, controls):
            np.testing.assert_allclose(data['force_step_commands_N'][:,0], commands[:,None]*force, atol=5e-9, rtol=0)
            np.testing.assert_array_equal(data['force_step_commands_N'][:,1], 0)
        response = np.array([c['positions_m'].astype(float)-zero['positions_m'].astype(float) for c in controls])
        query = menu['evaluation_query']
        if query['force_type'] != 'push_pulse':
            raise ValueError('Held diagonal is a different temporal family')
        predicted, model_info = {}, {}
        frozen = {}
        for degree in [1,2]:
            coefficients, info = fit_response_map(forces, response, degree)
            predicted[degree] = zero['positions_m']+predict_response(coefficients, query['peak_force_N'], degree)
            model_info[str(degree)] = info
            frozen[f'coefficients_degree{degree}'] = coefficients
            frozen[f'diagonal_prediction_degree{degree}'] = predicted[degree]
        pred_path = output/(eid+'-frozen-predictions.npz')
        np.savez_compressed(pred_path, **frozen, times_s=zero['times_s'])
        before = sha(pred_path)
        # Only now open the held TRAIN diagonal. It never chooses coefficients.
        diagonal = load('evaluation', query)
        np.testing.assert_allclose(diagonal['force_step_commands_N'][:,0], commands[:,None]*query['peak_force_N'], atol=5e-9, rtol=0)
        target = diagonal['positions_m']; zero_q = zero['positions_m']
        reference_path = dataset/'episodes'/eid/'reference.npz'
        with np.load(reference_path) as ref:rest,tets = ref['rest_points_m'],ref['tetrahedra']
        hashes[str(reference_path)] = sha(reference_path)
        homogeneity = []
        for pair in range(4):
            low, high = 3+2*pair,4+2*pair
            np.testing.assert_allclose(forces[high], 3*forces[low], atol=1e-12)
            defect = response[high]-3*response[low]
            homogeneity.append({'low_action': names[low], 'high_action': names[high],
                'high_to_low_force_ratio': 3., 'response_defect_coordinate_rms_m': rms(defect),
                'high_response_coordinate_rms_m': rms(response[high]),
                'relative_defect': rms(defect)/max(rms(response[high]), 1e-30)})
        models = {str(k): {**model_info[str(k)], 'held_diagonal_coordinate_rmse_m': rms(v-target),
            'predicted_force_induced_rms_m': rms(v-zero_q),
            'minimum_predicted_volume_ratio': finite_strain(rest,tets,v)['minimum_volume_ratio']} for k,v in predicted.items()}
        strain = {n: finite_strain(rest,tets,data['positions_m']) for n,data in zip(names,controls)}
        strain['evaluation'] = finite_strain(rest,tets,target)
        row = {'episode_id': eid, 'boundary_type': cases[eid]['boundary_type'],
            'physics_hz': summary['physics_hz'], 'numerical_convergence_established': summary.get('numerical_convergence_established',False),
            'fit_actions': names, 'fit_response_count': len(names), 'zero_response_as_fixed_offset': True,
            'held_diagonal_opened_after_fit': True, 'frozen_predictions_sha256': before,
            'models': models, 'zero_baseline_coordinate_rmse_m': rms(target-zero_q),
            'static_coordinate_rmse_m': rms(target-rest), 'homogeneity': homogeneity,
            'actual_strain': strain, 'precision': output_precision_audit(target,diagonal['velocities_m_s'],summary['physics_hz'])}
        if sha(pred_path) != before:
            raise RuntimeError('Prediction changed during evaluation')
        rows.append(row)
    result = {'rows': rows, 'skipped': skipped, 'input_sha256': hashes,
        'source_sha256': sha(__file__), 'test_or_validation_responses_opened': False,
        'calibration_responses_opened': False, 'material_inference_performed': False,
        'qualification': 'TRAIN model-class diagnosis at fixed simulator discretization. Smooth polynomial fit success/failure cannot establish a physical cause; convergence, attachment/contact and floating-point effects remain separate hypotheses.'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',default='runs/gaussian-campaign-v01/rich-controls')
    p.add_argument('--dataset',default='runs/gaussian-boundary-neural-v01')
    p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(**vars(a));print(json.dumps({'completed':len(r['rows']),'skipped':r['skipped']}))
