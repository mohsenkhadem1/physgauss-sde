"""Known-parameter forward comparison on eight declared selection layouts.

This is development model selection, not the reserved control-response gate or
fresh confirmation. Freeze the complete checkpoint list before opening labels.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from campaign_control_data import load_control_snapshot
from campaign_forward_models import load_campaign_operator
from train_campaign_forward import evaluate, sha
from gaussian_study_statistics import paired_layout_bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    p.add_argument('--phase-root', type=Path, default=Path('runs/gaussian-campaign-v01/nonlinear-pilot'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    cfg = json.loads(a.campaign.read_text())
    phase = json.loads((a.phase_root/'execution.json').read_text())
    if not phase['completed']:
        raise ValueError('All predefined candidates must complete')
    models = {'original_passive': Path(cfg['original_passive_forward']),
              'original_controlled': Path(cfg['original_control_forward'])}
    if phase['wave'] == 'pilot':
        for which in ['best', 'final']:
            models['balanced_loss_'+which] = Path('runs/gaussian-campaign-v01/pilots/balanced_small_response_loss')/(which+'.pt')
    for model in ['green', 'nonlinear', 'zero_referenced']:
        for which in ['best', 'final']:
            models[model+'_'+which] = a.phase_root/model/(which+'.pt')
    a.output.mkdir(parents=True, exist_ok=False)
    candidate_list = {'campaign_sha256': sha(a.campaign), 'selection_ids': cfg['new_validation_roles']['selection'],
        'models': {name: {'path': str(path), 'sha256': sha(path)} for name, path in models.items()},
        'metric': 'Control macro mean normalized coordinateRMSE with0.04mmfloor; no reservedgate orTEST labels'}
    # Save candidate identities before any new predictive selection label read.
    (a.output/'candidates-before-labels.json').write_text(json.dumps(candidate_list, indent=2)+'\n')
    records = load_control_snapshot(cfg, cfg['new_validation_roles']['selection'], 'select')
    torch.set_num_threads(2)
    results = {}
    for name, path in models.items():
        model = load_campaign_operator(path, a.device)
        scores = evaluate(model, records, a.device, volumes=True)
        eligible = [r for r in scores['episodes'] if r['all_actual_rms_motion_m'] >= .00005]
        scores['summary'] = {'mean_rmse_m': scores['mean_rmse_m'], 'static_mean_rmse_m': scores['mean_static_rmse_m'],
            'improvement_over_static': 1-scores['mean_rmse_m']/scores['mean_static_rmse_m'],
            'mean_normalized_rmse': scores['mean_normalized_rmse'],
            'minimum_volume_ratio': min(r['minimum_volume_ratio'] for r in scores['episodes']),
            'amplitude_eligible_controls': len(eligible),
            'eligible_amplitude_fraction_in_0p5_to_1p5': float(np.mean([.5 <= r['amplitude_ratio'] <= 1.5 for r in eligible])) if eligible else None}
        if sha(path) != candidate_list['models'][name]['sha256']:
            raise ValueError('Frozen candidate changed')
        results[name] = scores
        print(json.dumps({'model': name, **scores['summary']}), flush=True)
    for name, scores in results.items():
        baseline = results['original_controlled']['episodes']
        np.testing.assert_array_equal([r['episode_id'] for r in scores['episodes']], [r['episode_id'] for r in baseline])
        scores['versus_old_teacher'] = paired_layout_bootstrap(
            [r['all_coordinate_rmse_m'] for r in scores['episodes']], [r['all_coordinate_rmse_m'] for r in baseline],
            [r['group_id'] for r in baseline])
    result = {'candidates': candidate_list, 'models': results, 'layouts': 8, 'controls': len(records),
        'reserved_control_gate_opened': False, 'fresh_confirmation_opened': False,
        'qualification': 'Development selection evidence only. True physical parameters supplied to audit the forward map; not image-conditioned inference.',
        'input_trajectory_sha256': {r['path']+'/trajectory.npz': r['trajectory_sha256'] for r in records},
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in
            ['evaluate_campaign_selection.py', 'campaign_control_data.py', 'campaign_forward_models.py', 'train_campaign_forward.py', 'gaussian_operator.py']}}
    (a.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
