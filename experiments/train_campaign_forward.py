"""Paired loss/sampling ablations on the unchanged frozen Green architecture.

Only declared TRAIN controls can fit weights. Old VALIDATION candidate controls
are a pilot selection set; fresh validation gates and all TEST data stay closed.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from train_gaussian_control_forward import collect_control_records, decimate_passive_training
from train_gaussian_forward import collect_records, make_model
from gaussian_operator import physical_vector, category_labels
from diagnose_gaussian_forward import control_physical, motion_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validation_controls(dataset, pool):
    """Only VALIDATION candidate labels; never read independent evaluation."""
    cases = {c['episode_id']: c for c in json.loads((dataset/'plan.json').read_text())['episodes']}
    records = []
    for private in sorted((pool/'private').iterdir()):
        if not private.is_dir():
            continue
        eid = private.name
        case = cases[eid]
        if case['split'] != 'validation':
            raise ValueError('Nonvalidation layout in pilot control selection')
        if not json.loads((private/'validation.json').read_text())['passed']:
            raise ValueError('Unaudited validation control')
        recipe = json.loads((private/'truth/recipe.json').read_text())
        if recipe != case:
            raise ValueError('Validation recipe differs')
        spec = json.loads((private/'truth/boundaries.json').read_text())
        with np.load(dataset/'episodes'/eid/'reference.npz') as raw:
            rest, tets = raw['rest_points_m'].astype(np.float32), raw['tetrahedra']
        physical = physical_vector(case, spec, rest)
        bc, _ = category_labels(case)
        menu = json.loads((pool/'public'/eid/'action-menu.json').read_text())
        for action in menu['candidates']:
            if action['action_id'] not in {'poke_0', 'poke_1', 'poke_2', 'poke_3'}:
                raise ValueError('Independent evaluation control cannot select pilot')
            path = private/action['action_id']/'trajectory.npz'
            with np.load(path) as raw:
                q, times = raw['positions_m'].astype(np.float32), raw['times_s'].astype(np.float32)
            records.append({'episode_id': eid+'/'+action['action_id'], 'group_id': case['group_id'],
                'role': 'pilot_validation_control', 'split': 'validation', 'physical': control_physical(physical, action, menu['known_handle_centers_m']),
                'bc': bc, 'force': action['force_type_index'], 'rest': rest, 'tets': tets, 'q': q, 'times': times,
                'path': str(path.parent), 'trajectory_sha256': sha(path), 'recipe_sha256': case['recipe_sha256'],
                'scale_m': max(float(np.sqrt(np.mean((q-rest)**2))), .00004)})
    if len(records) != 16 or len({r['bc'] for r in records}) != 4:
        raise ValueError('Expected four audited pilot validation layouts covering four support types')
    return records


def normalized_scales(records, control_floor):
    return np.array([max(float(np.sqrt(np.mean((r['q']-r['rest'])**2))),
                         control_floor if r['role'] == 'teacher_fit_control' else .0004) for r in records], dtype=np.float32)


def sample_ids(rng, records, batch, probe_fraction, balanced):
    if any(r.get('split') != 'train' for r in records):
        raise ValueError('Sampling accepts TRAIN records only')
    is_control = np.array([r['role'] == 'teacher_fit_control' for r in records])
    bc = np.array([r['bc'] for r in records])
    output = []
    for _ in range(batch):
        selected = is_control == (rng.random() < probe_fraction)
        if balanced:
            selected &= bc == rng.integers(4)
        choices = np.flatnonzero(selected)
        if not len(choices):
            raise ValueError('Requested sampling stratum empty')
        output.append(rng.choice(choices))
    return np.array(output)


def evaluate(model, records, device, volumes=False):
    model.eval()
    rows = []
    with torch.no_grad():
        for r in records:
            rest = torch.as_tensor(r['rest'], device=device)
            predicted = model(rest, torch.as_tensor(r['times'], device=device),
                torch.as_tensor(r['physical'][None], device=device),
                torch.tensor([r['bc']], device=device), torch.tensor([r['force']], device=device))[0].cpu().numpy()
            metrics = motion_metrics(predicted, r['q'], r['rest'], r['times'])
            metrics['normalized_rmse'] = metrics['all_coordinate_rmse_m']/max(metrics['all_actual_rms_motion_m'], .00004)
            metrics['amplitude_ratio'] = metrics['all_predicted_rms_motion_m']/max(metrics['all_actual_rms_motion_m'], 1e-12)
            if volumes:
                tets = r['tets']
                actual_det = np.linalg.det(predicted[:, tets[:, 1:]]-predicted[:, tets[:, :1]])
                base_det = np.linalg.det(r['rest'][tets[:, 1:]]-r['rest'][tets[:, :1]])
                metrics['minimum_volume_ratio'] = float((actual_det/base_det).min())
            rows.append({'episode_id': r['episode_id'], 'group_id': r['group_id'], 'bc': r['bc'], **metrics})
    return {'mean_rmse_m': float(np.mean([r['all_coordinate_rmse_m'] for r in rows])),
            'mean_static_rmse_m': float(np.mean([r['all_static_rmse_m'] for r in rows])),
            'mean_normalized_rmse': float(np.mean([r['normalized_rmse'] for r in rows])),
            'episodes': rows}


def run(campaign, arm_name, output, device):
    config = json.loads(campaign.read_text())
    arm = next(a for a in config['initial_arms'] if a['name'] == arm_name)
    train_cfg = config['initial_training']
    torch.set_num_threads(2)
    torch.manual_seed(arm['seed'])
    rng = np.random.default_rng(arm['seed'])
    roots = [config['original_dataset'], config['dataset']]
    passive = collect_records(roots)
    controls = collect_control_records(Path(config['dataset']), Path(config['old_train_pool']), Path('configs/gaussian_probe_roles_v01.json'))
    passive = decimate_passive_training(passive, controls[0]['times'])
    train = [r for r in passive if r['split'] == 'train']+controls
    val_passive = [r for r in passive if r['split'] == 'validation']
    val_control = load_validation_controls(Path(config['original_dataset']), Path(config['old_validation_pool']))
    if len(train) != 200 or len(val_passive) != 44:
        raise ValueError('Initial pilot record set changed')
    checkpoint = torch.load(config['initial_resume'], map_location=device, weights_only=False)
    if checkpoint['model_class'] != 'NeuralGreenOperator':
        raise ValueError('Initial ablations require unchanged architecture')
    model = make_model(checkpoint['model_class'], checkpoint['model_kwargs'], train[0]['rest']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=1e-5)
    output.mkdir(parents=True, exist_ok=False)
    names = ['train_campaign_forward.py', 'train_gaussian_control_forward.py', 'train_gaussian_forward.py',
             'gaussian_operator.py', 'gaussian_probe_roles.py', 'diagnose_gaussian_forward.py']
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    (output/'provenance').mkdir()
    for name, data in sources.items():
        (output/'provenance'/name).write_bytes(data)
    shutil.copy2(campaign, output/'campaign.json')
    catalog = [{k: r.get(k) for k in ['episode_id', 'group_id', 'path', 'split', 'role', 'trajectory_sha256', 'recipe_sha256']} for r in train+val_passive+val_control]
    (output/'catalog.json').write_text(json.dumps(catalog, indent=2)+'\n')
    protocol = {'campaign_sha256': sha(campaign), 'arm': arm, 'training': train_cfg,
        'source_sha256': hashes, 'resume_sha256': sha(config['initial_resume']),
        'training_records': len(train), 'validation_passive': len(val_passive), 'validation_controls': len(val_control),
        'selection': 'Minimum pilot control normalizedRMSE(.04mmfloor) +0.25*passive coordinateRMSE / initial passiveRMSE',
        'test_labels_opened': False, 'fresh_gate_labels_opened': False,
        'expanded_controls_used': False, 'independent_diagonal_labels_opened': False}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    scales = normalized_scales(train, arm['control_scale_floor_m'])
    rest = torch.tensor(np.stack([r['rest'] for r in train]), device=device)
    target = torch.tensor(np.stack([r['q'] for r in train]), device=device)
    physical = torch.tensor(np.stack([r['physical'] for r in train]), device=device)
    bc = torch.tensor([r['bc'] for r in train], device=device)
    force = torch.tensor([r['force'] for r in train], device=device)
    scale = torch.tensor(scales, device=device)
    times = torch.tensor(train[0]['times'], device=device)
    for r in train:
        np.testing.assert_allclose(r['times'], train[0]['times'], atol=1e-6)
    base_passive = evaluate(model, val_passive, device)
    base_controls = evaluate(model, val_control, device)
    (output/'initial-validation.json').write_text(json.dumps({'passive': base_passive, 'controls': base_controls}, indent=2)+'\n')
    print(json.dumps({'arm': arm_name, 'started': True, 'initial_passive_rmse_m': base_passive['mean_rmse_m'],
                      'initial_controls_rmse_m': base_controls['mean_rmse_m'], 'control_loss_scale_min_m': float(scales[-48:].min())}), flush=True)
    started, best, history = time.monotonic(), float('inf'), []
    for step in range(1, train_cfg['steps']+1):
        model.train()
        ids = torch.tensor(sample_ids(rng, train, train_cfg['batch'], train_cfg['probe_fraction'], arm['bc_balanced']), device=device)
        nodes = torch.tensor(rng.choice(rest.shape[1], min(train_cfg['query_nodes'], rest.shape[1]), replace=False), device=device)
        tids = torch.tensor(np.sort(rng.choice(np.arange(1, len(times)), min(train_cfg['query_times'], len(times)-1), replace=False)), device=device)
        predicted = model(rest[ids][:, nodes], times[tids], physical[ids], bc[ids], force[ids])
        loss = ((predicted-target[ids][:, tids][:, :, nodes])/scale[ids, None, None, None]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach()), 'elapsed_s': time.monotonic()-started}), flush=True)
        if step == 1 or step % train_cfg['evaluation_every'] == 0 or step == train_cfg['steps']:
            pv, cv = evaluate(model, val_passive, device), evaluate(model, val_control, device)
            score = cv['mean_normalized_rmse']+.25*pv['mean_rmse_m']/base_passive['mean_rmse_m']
            payload = {'model_class': checkpoint['model_class'], 'model_kwargs': checkpoint['model_kwargs'], 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'step': step, 'source_sha256': hashes, 'protocol': protocol,
                'validation': {'passive': pv, 'controls': cv, 'selection_score': score}}
            torch.save(payload, output/'latest.pt')
            if score < best:
                best = score
                torch.save(payload, output/'best.pt')
                (output/'validation.json').write_text(json.dumps(payload['validation'], indent=2)+'\n')
            history.append({'step': step, 'passive_rmse_m': pv['mean_rmse_m'], 'control_rmse_m': cv['mean_rmse_m'],
                'control_normalized_rmse': cv['mean_normalized_rmse'], 'selection_score': score, 'best_score': best,
                'elapsed_s': time.monotonic()-started})
            (output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
            print(json.dumps(history[-1]), flush=True)
            if any(Path(__file__).with_name(name).read_bytes() != content for name, content in sources.items()):
                raise RuntimeError('Frozen experiment source changed')
    torch.save(payload, output/'final.pt')
    best_checkpoint = torch.load(output/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state'])
    (output/'best-validation-volume-audit.json').write_text(json.dumps(
        evaluate(model, val_control, device, volumes=True), indent=2)+'\n')
    (output/'completed.json').write_text(json.dumps({'completed': True, 'best_score': best, 'steps': train_cfg['steps'],
        'latest_sha256': sha(output/'latest.pt'), 'best_sha256': sha(output/'best.pt')}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    p.add_argument('--arm', choices=['same_loss', 'small_response_loss', 'balanced_small_response_loss'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    run(args.campaign, args.arm, args.output, args.device)
