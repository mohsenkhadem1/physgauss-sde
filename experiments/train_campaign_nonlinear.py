"""Matched from-scratch Green/nonlinear/zero-referenced forward experiments."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from train_campaign_forward import load_validation_controls, normalized_scales, sample_ids, evaluate, sha
from train_gaussian_control_forward import collect_control_records, decimate_passive_training
from train_gaussian_forward import collect_records
from train_gaussian_inverse import load_frozen_operator
from campaign_control_data import load_control_snapshot
from campaign_forward_models import CLASSES


def dataset_snapshot(config, wave):
    controls = collect_control_records(Path(config['dataset']), Path(config['old_train_pool']), Path('configs/gaussian_probe_roles_v01.json'))
    passive = decimate_passive_training(collect_records([config['original_dataset'], config['dataset']]), controls[0]['times'])
    train = [r for r in passive if r['split'] == 'train']+controls
    count = {'pilot': 0, 'first8': 8, 'first16': 16, 'full32': 32}[wave]
    if count:
        train += load_control_snapshot(config, config['new_train_control_ids'][:count], 'fit')
    val_passive = [r for r in passive if r['split'] == 'validation']
    val_control = load_validation_controls(Path(config['original_dataset']), Path(config['old_validation_pool'])) if wave in ['pilot', 'first8'] else \
        load_control_snapshot(config, config['new_validation_roles']['selection'], 'select')
    if len(train) != 200+count*4 or len(val_passive) != 44:
        raise ValueError('Snapshot record count differs')
    return train, val_passive, val_control


def run(phase_path, name, wave, output, device):
    phase = json.loads(phase_path.read_text())
    config = json.loads(Path(phase['campaign']).read_text())
    if name not in phase['models']:
        raise ValueError('Model not in frozen comparison')
    train_cfg = phase['training']
    train, val_passive, val_control = dataset_snapshot(config, wave)
    torch.set_num_threads(2)
    torch.manual_seed(phase['seed'])
    rng = np.random.default_rng(phase['seed'])
    kwargs = {'hidden': 128, 'rank': 24, 'depth': 3, 'seconds': 3., 'displacement_scale_m': .03}
    model = CLASSES[name](**kwargs, reference_points=train[0]['rest']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=1e-5)
    output.mkdir(parents=True, exist_ok=False)
    source_names = ['train_campaign_nonlinear.py', 'campaign_forward_models.py', 'campaign_control_data.py',
        'train_campaign_forward.py', 'gaussian_operator.py', 'train_gaussian_forward.py', 'train_gaussian_control_forward.py',
        'gaussian_probe_roles.py', 'diagnose_gaussian_forward.py']
    sources = {n: Path(__file__).with_name(n).read_bytes() for n in source_names}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}
    (output/'provenance').mkdir()
    for n, b in sources.items():
        (output/'provenance'/n).write_bytes(b)
    (output/'phase.json').write_bytes(phase_path.read_bytes())
    (output/'campaign.json').write_bytes(Path(phase['campaign']).read_bytes())
    catalog = [{k: r.get(k) for k in ['episode_id', 'group_id', 'path', 'split', 'role', 'trajectory_sha256', 'recipe_sha256']} for r in train+val_passive+val_control]
    (output/'catalog.json').write_text(json.dumps(catalog, indent=2)+'\n')
    protocol = {'phase_sha256': sha(phase_path), 'campaign_sha256': sha(phase['campaign']), 'model': name, 'wave': wave,
        'seed': phase['seed'], 'training': train_cfg, 'source_sha256': hashes, 'training_records': len(train),
        'validation_passive': len(val_passive), 'validation_controls': len(val_control),
        'selection': phase['selection'], 'test_labels_opened': False, 'fresh_gate_labels_opened': False,
        'training_independent_diagonal_targets': False, 'training_zero_force_diagnostic_targets': False}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rest = torch.tensor(np.stack([r['rest'] for r in train]), device=device)
    target = torch.tensor(np.stack([r['q'] for r in train]), device=device)
    physical = torch.tensor(np.stack([r['physical'] for r in train]), device=device)
    bc = torch.tensor([r['bc'] for r in train], device=device)
    force = torch.tensor([r['force'] for r in train], device=device)
    scale = torch.tensor(normalized_scales(train, train_cfg['control_scale_floor_m']), device=device)
    times = torch.tensor(train[0]['times'], device=device)
    baseline = load_frozen_operator(config['original_passive_forward'], device)
    passive_normalizer = evaluate(baseline, val_passive, device)['mean_rmse_m']
    del baseline
    best, history, started = float('inf'), [], time.monotonic()
    def validation(step):
        nonlocal best
        pv, cv = evaluate(model, val_passive, device), evaluate(model, val_control, device)
        score = cv['mean_normalized_rmse']+.25*pv['mean_rmse_m']/passive_normalizer
        payload = {'model_class': type(model).__name__, 'model_kwargs': kwargs, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'step': step, 'source_sha256': hashes, 'protocol': protocol,
            'validation': {'passive': pv, 'controls': cv, 'selection_score': score}, 'passive_score_normalizer_m': passive_normalizer}
        torch.save(payload, output/'latest.pt')
        if step == 0:
            torch.save(payload, output/'initial.pt')
        if score < best:
            best = score
            torch.save(payload, output/'best.pt')
            (output/'validation.json').write_text(json.dumps(payload['validation'], indent=2)+'\n')
        row = {'step': step, 'passive_rmse_m': pv['mean_rmse_m'], 'control_rmse_m': cv['mean_rmse_m'],
            'control_normalized_rmse': cv['mean_normalized_rmse'], 'selection_score': score, 'best_score': best,
            'elapsed_s': time.monotonic()-started}
        history.append(row)
        (output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        if any(Path(__file__).with_name(n).read_bytes() != b for n, b in sources.items()):
            raise RuntimeError('Frozen experiment source changed')
        return payload
    validation(0)
    for step in range(1, train_cfg['steps']+1):
        model.train()
        ids = torch.tensor(sample_ids(rng, train, train_cfg['batch'], train_cfg['probe_fraction'], train_cfg['bc_balanced']), device=device)
        nodes = torch.tensor(rng.choice(rest.shape[1], min(train_cfg['query_nodes'], rest.shape[1]), replace=False), device=device)
        tid = torch.tensor(np.sort(rng.choice(np.arange(1, len(times)), min(train_cfg['query_times'], len(times)-1), replace=False)), device=device)
        pred = model(rest[ids][:, nodes], times[tid], physical[ids], bc[ids], force[ids])
        loss = ((pred-target[ids][:, tid][:, :, nodes])/scale[ids, None, None, None]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach()), 'elapsed_s': time.monotonic()-started}), flush=True)
        if step % train_cfg['evaluation_every'] == 0 or step == train_cfg['steps']:
            payload = validation(step)
    torch.save(payload, output/'final.pt')
    selected = torch.load(output/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model_state'])
    (output/'best-volume-audit.json').write_text(json.dumps(evaluate(model, val_control, device, volumes=True), indent=2)+'\n')
    (output/'completed.json').write_text(json.dumps({'completed': True, 'steps': train_cfg['steps'], 'best_score': best,
        'best_sha256': sha(output/'best.pt'), 'final_sha256': sha(output/'final.pt')}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', type=Path, default=Path('configs/gaussian_campaign_nonlinear_v01.json'))
    p.add_argument('--model', choices=['green', 'nonlinear', 'zero_referenced'], required=True)
    p.add_argument('--wave', choices=['pilot', 'first8', 'first16', 'full32'], default='pilot')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    run(a.phase, a.model, a.wave, a.output, a.device)
