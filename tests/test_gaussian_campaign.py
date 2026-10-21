import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from train_campaign_forward import normalized_scales, sample_ids
from isaac_dataset import digest_json
from audit_campaign_commands import quantized_clock_residual
import json


def test_clock_audit_handles_float32_dt_but_rejects_real_time_offset():
    times = np.arange(61)/20
    simulated = np.rint(times*960)*float(np.float32(1/960))
    assert np.max(np.abs(simulated-times)) > 1e-7
    assert quantized_clock_residual(simulated, times, 960) == 0
    assert quantized_clock_residual(simulated+.001, times, 960) > .00099


def test_small_control_loss_floor_leaves_passive_scales_unchanged():
    rest = np.zeros((1, 3), dtype=np.float32)
    rows = [dict(rest=rest, q=rest[None]+.00001, role=role) for role in ['passive', 'teacher_fit_control']]
    np.testing.assert_allclose(normalized_scales(rows, .00004), [.0004, .00004])
    np.testing.assert_allclose(normalized_scales(rows, .0004), [.0004, .0004])


def test_balanced_sampling_covers_roles_and_boundary_types_without_test_labels():
    rows = [{'role': role, 'bc': bc, 'split': 'train'} for role in ['passive', 'teacher_fit_control'] for bc in range(4)]
    ids = sample_ids(np.random.default_rng(1), rows, 10000, .75, True)
    frac = np.mean([rows[i]['role'] == 'teacher_fit_control' for i in ids])
    assert .72 < frac < .78
    for bc in range(4):
        assert .22 < np.mean([rows[i]['bc'] == bc for i in ids]) < .28
    rows[-1]['split'] = 'test'
    with pytest.raises(ValueError, match='TRAIN'):
        sample_ids(np.random.default_rng(1), rows, 8, .75, True)


def test_campaign_roles_and_fresh_recipe_hashes_are_disjoint():
    base = Path(__file__).resolve().parents[1]
    cfg = json.loads((base/'configs/gaussian_campaign_v01.json').read_text())
    assert not set(cfg['new_train_control_ids']) & set(cfg['new_train_calibration_only_ids']+cfg['old_calibration_only_ids'])
    roles = cfg['new_validation_roles']
    assert not set(roles['selection']) & set(roles['forward_gate'])
    assert set(roles['selection']+roles['forward_gate']) == set(cfg['new_validation_control_ids'])
    plan = json.loads((base/cfg['heldout_plan']).read_text())
    assert digest_json({k:v for k,v in plan.items() if k != 'plan_sha256'}) == plan['plan_sha256']
    assert len(plan['episodes']) == 32
    assert len({c['seed'] for c in plan['episodes']}) == 32
    assert all(c['split'] == 'test' and c['seed'] >= 861000 for c in plan['episodes'])
    for case in plan['episodes']:
        assert digest_json({k:v for k,v in case.items() if k != 'recipe_sha256'}) == case['recipe_sha256']
