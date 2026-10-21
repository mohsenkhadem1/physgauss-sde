"""Privileged forward checks on frozen passive and controlled TEST responses."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from gaussian_operator import physical_vector, category_labels
from train_gaussian_inverse import load_frozen_operator
from gaussian_study_statistics import paired_layout_bootstrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path):
    config = json.loads(Path(path).read_text())
    for name in ['passive', 'controlled', 'inverse']:
        if sha(config[name+'_checkpoint']) != config[name+'_sha256']:
            raise ValueError('Frozen checkpoint changed: '+name)
    return config


def query(model, rest, times, physical, bc, force, device):
    physical = np.asarray(physical, dtype=np.float32)
    if physical.ndim == 1:
        physical = physical[None]
    with torch.no_grad():
        return model(torch.as_tensor(rest, dtype=torch.float32, device=device),
                     torch.as_tensor(times, dtype=torch.float32, device=device),
                     torch.as_tensor(physical, device=device),
                     torch.full((len(physical),), bc, device=device, dtype=torch.long),
                     torch.full((len(physical),), force, device=device, dtype=torch.long)).cpu().numpy()


def control_physical(physical, action, known_handles):
    """Evaluation-only oracle: retain supports/material; replace every old load."""
    result = np.asarray(physical).copy()
    result[10:16] = np.asarray(known_handles).ravel()
    result[16:22] = 0
    result[16:19] = action['peak_force_N']
    return result


def motion_metrics(predicted, actual, rest, times):
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError('Finite matched trajectories required')
    output = {}
    for name, mask in [('all', np.ones(len(times), dtype=bool)),
                       ('gap', (times > 1) & (times < 2)),
                       ('late', (times >= 2) & (times <= 3))]:
        if not mask.any():
            continue
        p, a = predicted[mask]-rest, actual[mask]-rest
        output[name+'_coordinate_rmse_m'] = float(np.sqrt(np.mean((p-a)**2)))
        output[name+'_static_rmse_m'] = float(np.sqrt(np.mean(a*a)))
        # Amplitude and alignment help distinguish under-response from phase/
        # direction errors. This is descriptive, not an additive error split.
        output[name+'_predicted_rms_motion_m'] = float(np.sqrt(np.mean(p*p)))
        output[name+'_actual_rms_motion_m'] = float(np.sqrt(np.mean(a*a)))
        output[name+'_displacement_cosine'] = float(np.sum(p*a)/max(np.linalg.norm(p)*np.linalg.norm(a), 1e-30))
    return output


def run(protocol, output, device):
    cfg = load_protocol(protocol)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    models = {key: load_frozen_operator(cfg[key+'_checkpoint'], device) for key in ['passive', 'controlled']}
    rows, inputs = [], {str(protocol): sha(protocol)}
    for root_key in ['primary_dataset', 'secondary_dataset']:
        root = Path(cfg[root_key])
        plan = json.loads((root/'plan.json').read_text())
        inputs[str(root/'plan.json')] = sha(root/'plan.json')
        learned = {}
        for context in ['full', 'prefix']:
            source = Path('runs')/('gaussian-final-test-'+('fresh64' if root_key == 'primary_dataset' else 'old12')+'-25d-rgbd-'+context+'-v01')/'results.json'
            inputs[str(source)] = sha(source)
            learned[context] = {r['episode_id']: r for r in json.loads(source.read_text())['episodes']}
        for case in plan['episodes']:
            if case['split'] != 'test':
                continue
            episode = root/'episodes'/case['episode_id']
            if not json.loads((episode/'validation.json').read_text())['passed']:
                raise ValueError('Unaudited source')
            with np.load(episode/'reference.npz') as raw:
                rest = raw['rest_points_m']
            spec = json.loads((episode/'truth/boundaries.json').read_text())
            physical = physical_vector(case, spec, rest)
            bc, family = category_labels(case)
            with np.load(episode/'truth/trajectory.npz') as raw:
                actual, times = raw['positions_m'], raw['times_s']
            pred = query(models['passive'], rest, times, physical, bc, family, device)[0]
            row = {'dataset': root_key, 'episode_id': case['episode_id'], 'group_id': case['group_id'],
                   'boundary_type': case['boundary_type'], 'force_type': case['force_type'],
                   'oracle': motion_metrics(pred, actual, rest, times),
                   'learned_rgbd_full': learned['full'][case['episode_id']],
                   'learned_rgbd_prefix': learned['prefix'][case['episode_id']]}
            rows.append(row)
            for p in [episode/'reference.npz', episode/'validation.json', episode/'truth/boundaries.json', episode/'truth/trajectory.npz']:
                inputs[str(p)] = sha(p)
        print(f'{root_key}: oracle done', flush=True)
    controls = []
    root, pool = Path(cfg['primary_dataset']), Path(cfg['probe_pool'])
    original_path = Path('runs/gaussian-final-active-probe-25d-rgbd-test-v01/results.json')
    inputs[str(original_path)] = sha(original_path)
    original = json.loads(original_path.read_text())
    original = {r['episode_id']: r for r in original['episodes']}
    for i in range(8):
        eid = f'z{i:04d}'
        private = pool/'private'/eid
        case = json.loads((private/'truth/recipe.json').read_text())
        spec = json.loads((private/'truth/boundaries.json').read_text())
        menu_path = pool/'public'/eid/'action-menu.json'
        menu = json.loads(menu_path.read_text())
        if not json.loads((private/'validation.json').read_text())['passed']:
            raise ValueError('Unaudited probe source')
        with np.load(root/'episodes'/eid/'reference.npz') as raw:
            rest = raw['rest_points_m']
        physical = physical_vector(case, spec, rest)
        bc, _ = category_labels(case)
        for action in menu['candidates']+[menu['evaluation_query']]:
            control = control_physical(physical, action, menu['known_handle_centers_m'])
            path = private/action['action_id']/'trajectory.npz'
            with np.load(path) as raw:
                actual, times = raw['positions_m'], raw['times_s']
            row = {'episode_id': eid, 'group_id': case['group_id'], 'boundary_type': case['boundary_type'],
                   'action_id': action['action_id'], 'force_N': action['peak_force_N'], 'models': {}}
            for key, model in models.items():
                pred = query(model, rest, times, control, bc, int(action['force_type_index']), device)[0]
                row['models'][key] = motion_metrics(pred, actual, rest, times)
            if action['action_id'] == 'evaluation':
                row['learned_control_policies'] = original[eid]['policies']
            controls.append(row)
            inputs[str(path)] = sha(path)
        for p in [private/'truth/recipe.json', private/'truth/boundaries.json', private/'validation.json', menu_path]:
            inputs[str(p)] = sha(p)
        print(f'{eid}: five true-parameter control responses done', flush=True)
    comparisons = {}
    for dataset in ['primary_dataset', 'secondary_dataset']:
        selected = [r for r in rows if r['dataset'] == dataset]
        groups = [r['group_id'] for r in selected]
        for metric, oldmetric in [('gap', 'gap_coordinate_rmse_m'), ('late', 'late_2to3_coordinate_rmse_m')]:
            oracle = [r['oracle'][metric+'_coordinate_rmse_m'] for r in selected]
            comparisons[dataset+'_'+metric+'_oracle_vs_static'] = paired_layout_bootstrap(oracle, [r['oracle'][metric+'_static_rmse_m'] for r in selected], groups)
            comparisons[dataset+'_'+metric+'_oracle_vs_inferred'] = paired_layout_bootstrap(oracle, [r['learned_rgbd_full'][oldmetric] for r in selected], groups)
            if metric == 'late':
                comparisons[dataset+'_late_oracle_vs_prefix_inferred'] = paired_layout_bootstrap(oracle, [r['learned_rgbd_prefix'][oldmetric] for r in selected], groups)
    evals = [r for r in controls if r['action_id'] == 'evaluation']
    for model in models:
        oracle = [r['models'][model]['all_coordinate_rmse_m'] for r in evals]
        for baseline in ['static', 'passive_inference', 'risk_after_poke']:
            values = [r['models'][model]['all_static_rmse_m'] if baseline == 'static' else
                r['learned_control_policies']['passive' if baseline == 'passive_inference' else 'predictive_risk']['withheld_control_coordinate_rmse_m'] for r in evals]
            comparisons[f'control_evaluation_{model}_oracle_vs_{baseline}'] = paired_layout_bootstrap(oracle, values)
    for key in ['passive', 'controlled', 'inverse']:
        inputs[cfg[key+'_checkpoint']] = sha(cfg[key+'_checkpoint'])
    inputs[str(Path(__file__))] = sha(__file__)
    result = {'protocol': cfg, 'protocol_sha256': sha(protocol), 'passive': rows,
              'controls': controls, 'comparisons': comparisons, 'input_sha256': inputs,
              'qualification': 'Privileged post-test diagnosis. Oracle is a true-label query, not a guarantee of lowest surrogate prediction error. Forward and inference errors do not add independently.'}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/gaussian_diagnosis_v01.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/gaussian-diagnosis-oracle-v03'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.protocol, args.output, args.device)
