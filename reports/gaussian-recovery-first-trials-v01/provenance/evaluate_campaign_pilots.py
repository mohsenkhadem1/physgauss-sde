"""Separate training-fit capacity from validation transfer for frozen pilots."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_campaign_forward import load_validation_controls, evaluate, sha
from train_gaussian_control_forward import collect_control_records
from train_gaussian_inverse import load_frozen_operator


def summarize(scores):
    rows = scores['episodes']
    eligible = [r for r in rows if r['all_actual_rms_motion_m'] >= .00005]
    return {'mean_coordinate_rmse_m': scores['mean_rmse_m'],
        'mean_static_coordinate_rmse_m': scores['mean_static_rmse_m'],
        'relative_mean_improvement_over_static': 1-scores['mean_rmse_m']/scores['mean_static_rmse_m'],
        'mean_normalized_rmse_0p04mm_floor': scores['mean_normalized_rmse'],
        'amplitude_eligible_controls': len(eligible),
        'median_amplitude_ratio_eligible': float(np.median([r['amplitude_ratio'] for r in eligible])) if eligible else None,
        'boundary_types': {str(bc): {'coordinate_rmse_m': float(np.mean([r['all_coordinate_rmse_m'] for r in rows if r['bc'] == bc])),
                                   'static_rmse_m': float(np.mean([r['all_static_rmse_m'] for r in rows if r['bc'] == bc]))} for bc in range(4)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    p.add_argument('--root', type=Path, default=Path('runs/gaussian-campaign-v01/pilots'))
    p.add_argument('--output', type=Path, default=Path('runs/gaussian-campaign-v01/pilot-fit-transfer.json'))
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    cfg = json.loads(a.campaign.read_text())
    execution = json.loads((a.root/'execution.json').read_text())
    if not execution['completed'] or any(r['state'] != 'completed' for r in execution['arms']):
        raise ValueError('All frozen pilot arms must finish first')
    torch.set_num_threads(2)
    train = collect_control_records(Path(cfg['dataset']), Path(cfg['old_train_pool']), Path('configs/gaussian_probe_roles_v01.json'))
    validation = load_validation_controls(Path(cfg['original_dataset']), Path(cfg['old_validation_pool']))
    models = {'original_passive': Path(cfg['original_passive_forward']),
              'original_controlled': Path(cfg['original_control_forward'])}
    for arm in cfg['initial_arms']:
        folder = a.root/arm['name']
        complete = json.loads((folder/'completed.json').read_text())
        if sha(folder/'best.pt') != complete['best_sha256']:
            raise ValueError('Selected checkpoint changed')
        for which in ['best', 'final']:
            models[arm['name']+'_'+which] = folder/(which+'.pt')
    rows = {}
    for name, path in models.items():
        model = load_frozen_operator(path, a.device)
        fit = evaluate(model, train, a.device)
        val = evaluate(model, validation, a.device)
        rows[name] = {'checkpoint': str(path), 'checkpoint_sha256': sha(path),
                      'training': {'summary': summarize(fit), **fit},
                      'validation': {'summary': summarize(val), **val}}
        print(json.dumps({'model': name, 'training': summarize(fit), 'validation': summarize(val)}), flush=True)
    result = {'campaign_sha256': sha(a.campaign), 'models': rows, 'train_controls': len(train), 'validation_controls': len(validation),
        'fresh_validation_gate_opened': False, 'test_labels_opened': False,
        'selection_policy': 'No new selection: evaluate both preselected best and fixed12000-step final for each arm',
        'qualification': 'Pilot fits on48TRAIN controls, transfer on16candidatecontrols from4oldVALIDATION layouts; no independent diagonal controls here',
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in
            ['evaluate_campaign_pilots.py', 'train_campaign_forward.py', 'train_gaussian_inverse.py', 'gaussian_operator.py']}}
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
