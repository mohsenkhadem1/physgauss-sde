"""Verify restored nonlinear weights reproduce live TRAIN-control predictions."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from campaign_forward_models import load_campaign_operator
from gaussian_operator import physical_vector, category_labels
from diagnose_gaussian_forward import control_physical, query


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restored-root', type=Path, required=True)
    p.add_argument('--original-root', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-relative', type=Path,
                   default=Path('runs/gaussian-campaign-v01/nonlinear-pilot/zero_referenced/final.pt'))
    a = p.parse_args()
    relative = a.checkpoint_relative
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Checkpoint must be relative to both archive roots')
    live_path, restored_path = a.original_root/relative, a.restored_root/relative
    if sha(live_path) != sha(restored_path):
        raise ValueError('Restored checkpoint bytes differ')
    torch.set_num_threads(2)
    live = load_campaign_operator(live_path)
    restored = load_campaign_operator(restored_path)
    records = []
    for eid in ['t0001', 't0003']:
        episode = a.original_root/'runs/gaussian-boundary-neural-v01/episodes'/eid
        case = json.loads((episode/'truth/recipe.json').read_text())
        if case['split'] != 'train':
            raise ValueError('TRAIN-only replay proof')
        spec = json.loads((episode/'truth/boundaries.json').read_text())
        with np.load(episode/'reference.npz') as raw:
            rest = raw['rest_points_m'].astype(np.float32)
        menu = json.loads((a.original_root/'runs/isaac-calibrated-probes-train-v01/public'/eid/'action-menu.json').read_text())
        physical = control_physical(physical_vector(case, spec, rest), menu['candidates'][2], menu['known_handle_centers_m'])
        bc, _ = category_labels(case)
        times = np.array(menu['sensor']['times_s'], dtype=np.float32)
        old = query(live, rest, times, physical, bc, 0, 'cpu')
        new = query(restored, rest, times, physical, bc, 0, 'cpu')
        if not np.array_equal(old, new):
            raise ValueError('Restored forward predictions do not reproduce')
        zero = physical.copy()
        zero[16:25] = 0
        zero_q = query(restored, rest, times, zero, bc, 0, 'cpu')
        if not np.array_equal(zero_q, np.broadcast_to(rest, zero_q.shape)):
            raise ValueError('Zero-input invariant lost')
        records.append({'episode_id': eid, 'shape': list(new.shape), 'bitwise_equal': True,
            'prediction_sha256': hashlib.sha256(np.ascontiguousarray(new).tobytes()).hexdigest(),
            'coordinate_rms_motion_m': float(np.sqrt(np.mean((new-rest)**2))),
            'zero_force_and_zero_motion_exact_rest': True})
    result = {'checkpoint_relative': str(relative), 'restored_checkpoint_sha256': sha(restored_path), 'records': records,
        'test_labels_opened': False, 'source_sha256': sha(__file__),
        'qualification': 'Portability/numerical replay proof on TRAIN controls, not a new predictive quality or posterior calibration result'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
