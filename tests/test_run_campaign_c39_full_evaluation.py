"""Exercise all64 joins and saved posterior codecs without mechanics/noise draws."""
import copy
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments import run_campaign_c39_full_evaluation as f
from experiments import campaign_c38_controlled_observations as obs
from experiments.campaign_c38_prediction_metrics import POINTS, POINT_IDS


def build(root, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(f.os, 'sched_getaffinity', lambda pid: {34, 35})
    for name in f.source_records(Path(f.ROOT)):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(f.ROOT) / name, dest)
    design = root / obs.DESIGN_PATH
    design.parent.mkdir(parents=True)
    shutil.copyfile(Path(f.ROOT) / obs.DESIGN_PATH, design)
    (root / 'fake').mkdir()
    write = lambda p, x: f.io.write_new(root / p, x)
    rec = lambda p: f.io.record(root, root / p)
    write('fake/design.json', dict(frozen_before_any_outcome=True))
    monkeypatch.setattr(f, 'DESIGN', rec('fake/design.json'))
    target = POINTS.copy()
    target[:, 2] += .001
    def prediction(token, error, side=False):
        q = target.copy()
        q[:, 2] += error
        intervals = np.stack([q[POINT_IDS] - .0001, q[POINT_IDS], q[POINT_IDS] + .0001], -1)
        result = dict(object_id=token, completed=True, numerical_controls_passed=True,
                      methods={'adaptive': dict(completed=True, canonical_mean_positions_m=q,
                          category_probabilities=[.8, .2], material_equal_tail_Pa=[90000., 100000., 110000.],
                          material_quantile_details=[dict(unit_bracket=dict(lower=np.log(1.2)/np.log(2), upper=.3)), {},
                                                     dict(unit_bracket=dict(lower=.4, upper=np.log(110000/75000)/np.log(2)))],
                          canonical_position_equal_tail_m=intervals,
                          canonical_position_interval95_numerical_enclosure_m=intervals[:, :, [0, 2]],
                          canonical_coordinate_variance_m2=np.ones((63, 3))*1e-9,
                          canonical_variance_integration_resolution_estimate_m2=np.ones((63, 3))*1e-15)})
        if side:
            result.update(acquired_probe_action_id=f.metrics.SIDE, acquired_probe_force_world_N=np.array([0., 2., 0.]),
                          acquired_coordinate_count=54, fresh_prior=True, initial_posterior_weights_reused=False)
        return result
    def bundle(token, tag, error):
        models = {}
        for name in f.io.MODELS:
            folder = root / 'fake' / (token + '-' + tag + '-' + name)
            folder.mkdir()
            f.io.save_full_result(folder, prediction(token, error, tag == 'side'))
            models[name] = dict(execution_completed=True, files={n: f.io.record(root, folder / n) for n in f.FILES})
        path = 'fake/' + token + '-' + tag + '.json'
        state = dict(completed=True, object_id=token, models=models)
        if tag == 'down':
            state['force_Fz_N'] = -2.
        if tag == 'side':
            state.update(action_id=f.metrics.SIDE, force_world_N=[0., 2., 0.])
        write(path, state)
        return rec(path)
    initial, down, side, choices, private = [], [], [], [], []
    oldrows = {name: dict(initial=[], down=[]) for name in f.io.MODELS}
    original_design = obs.load_frozen_design(root)
    for c in range(4):
        for n in range(16):
            token = f'{c*16+n:032x}'
            initial.append(dict(object_id=token, result=bundle(token, 'before', 2e-6)))
            down.append(dict(object_id=token, force_Fz_N=-2., result=bundle(token, 'down', 1e-6)))
            side.append(dict(object_id=token, action_id=f.metrics.SIDE, result=bundle(token, 'side', 3e-6)))
            cp = 'fake/' + token + '-choices.json'
            write(cp, dict(completed=True))
            choices.append(dict(object_id=token, result=rec(cp), choices=dict(information=f.metrics.DOWN,
                                target_risk=f.metrics.DOWN, fixed_down=f.metrics.DOWN,
                                fixed_side=f.metrics.SIDE, random=f.metrics.SIDE)))
            pub = obs.public_prefix_payload(token, target[POINT_IDS], design=original_design)
            prefix = root / 'fake' / (token + '-prefix.npz')
            f.evaluation.codec.exclusive_npz(prefix, f.evaluation.codec.encode_public(pub, design=original_design))
            truth = dict(object_id=np.asarray(token), private_configuration_index=np.asarray(c, dtype=np.int64),
                         repeat_index=np.asarray(n, dtype=np.int64), young_Pa=np.asarray(100000.),
                         support_category=np.asarray('both_ends'), canonical_target_Fz_N=np.asarray(1.),
                         canonical_target_point_ids=np.arange(63), canonical_target_positions_m=target)
            archive = root / 'fake' / (token + '-private.npz')
            np.savez(archive, **truth, candidate_probe_observations_m=np.array([object()], dtype=object))
            private.append(dict(object_id=token, private_future=f.io.record(root, archive), public_prefix=f.io.record(root, prefix)))
            decoded_truth = {k: (v.item() if np.asarray(v).shape == () else v) for k, v in truth.items()}
            for name in f.io.MODELS:
                oldrows[name]['initial'].append(f.evaluation.metrics.score_prediction(prediction(token, 2e-6), decoded_truth, pub))
                oldrows[name]['down'].append(f.evaluation.metrics.score_prediction(prediction(token, 1e-6), decoded_truth, pub))
    write('fake/initial.json', dict(completed=True, count=64, model_executions=128, objects=initial))
    ir = rec('fake/initial.json')
    write('fake/oldchoices.json', {})
    oldcr = rec('fake/oldchoices.json')
    write('fake/oldpost.json', dict(completed=True, initial_prediction_seal=ir, choices_seal=oldcr, objects=down))
    write('fake/oldsummary.json', dict(completed=True, source_sha256={}, initial_prediction_seal=ir,
                                     choices_seal=oldcr, postprobe_prediction_seal=rec('fake/oldpost.json')))
    monkeypatch.setattr(f, 'C38_SUMMARY', rec('fake/oldsummary.json'))
    for name in f.io.MODELS:
        base = root / 'runs/gaussian-campaign-v01/c38-full-evaluation-v01'
        (base / 'initial').mkdir(parents=True, exist_ok=True)
        f.io.write_new(base / 'initial' / (name + '-private-metrics.json'), f.io.jsonable(oldrows[name]['initial']))
        f.io.write_new(base / (name + '-down-private-metrics.json'), f.io.jsonable(oldrows[name]['down']))
    f.freeze_reference('fake/reference.json', root=root)
    write('fake/choices.json', dict(schema='C39-typed-static-policy-choices-seal-v01', completed=True, count=64,
          model_executions=128, extension_design=f.DESIGN, private_SIDE_outcomes_access=False,
          acquired_DOWN_observations_or_posteriors_access=False, private_target_or_latent_access=False,
          all_choices_sealed_before_SIDE_noise=True, initial_prediction_seal=ir,
          choices=choices, model_choices={name: choices for name in f.io.MODELS}))
    cr = rec('fake/choices.json')
    write('fake/post.json', dict(completed=True, count=64, model_executions=128,
          private_target_or_latent_access=False, initial_prediction_seal=ir, choices_seal=cr, objects=side))
    write('fake/observations.json', dict(completed=True, records=private))
    return dict(reference_record=rec('fake/reference.json'), choices_record=cr,
                post_record=rec('fake/post.json'), observation_execution_record=rec('fake/observations.json'))


def test_full_saved_cohort_reuse_and_poisoned_unused_observation(tmp_path, monkeypatch):
    records = build(tmp_path, monkeypatch)
    result = f.run(**records, output='result', root=tmp_path)
    assert result['completed'] and result['actual_probe_observations_extracted_by_evaluation'] is False
    assert result['C38_targets_previously_analyzed'] is True
    for model in f.io.MODELS:
        comparison = result['models'][model]['canonical_RMSE_advantage_over_baseline_m']['information']
        assert comparison['fixed_down']['conditional_noise_bootstrap95'] == [0., 0.]
        assert comparison['fixed_side']['mean_improvement'] > 0
    # allow_pickle=False would fail if evaluation touched the poisoned probeYs.


def test_one_missing_SIDE_hash_rejects_before_any_private_archive(tmp_path, monkeypatch):
    records = build(tmp_path, monkeypatch)
    post = f.io.read_json(tmp_path, records['post_record'])
    state = f.io.read_json(tmp_path, post['objects'][-1]['result'])
    (tmp_path / state['models']['nonlinear_table']['files']['posterior-arrays.npz']['path']).unlink()
    monkeypatch.setattr(f.np, 'load', lambda *a, **k: pytest.fail('Private array access before complete hash barrier'))
    with pytest.raises((ValueError, FileNotFoundError)):
        f.run(**records, output='reject', root=tmp_path)
