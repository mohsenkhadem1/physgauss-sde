"""Synthetic role barriers, stable moments and phase-order integration checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_point_prior_analysis as runner
from experiments.censored_range_observation import CensoredRanges


def plan():
    objects = [dict(index=i,object_id=f'point_prior_{i:03d}',
        role='CALIBRATION' if i < 16 else 'RESERVED_EVALUATION',
        realized_hypothesis=dict(path=f'PRIVATE/{i}.json',sha256='not-opened')) for i in range(48)]
    return dict(objects=objects, roles=dict(CALIBRATION=list(range(16)),
        RESERVED_EVALUATION=list(range(16,48))),cells=[])


def test_calibration_role_barrier_exposes_exact_first16_without_private_reads(monkeypatch):
    def forbidden(*a,**kw):
        raise AssertionError('Role barrier must use metadata only')
    monkeypatch.setattr(runner,'read',forbidden)
    monkeypatch.setattr(np,'load',forbidden)
    selected = runner.eligible_objects(plan(),'CALIBRATION',execution_sha256='execution',method_sha256='method')
    assert [o['index'] for o in selected] == list(range(16))
    assert all(o['role'] == 'CALIBRATION' for o in selected)
    with pytest.raises(ValueError,match='explicit reserved release'):
        runner.eligible_objects(plan(),'RESERVED_EVALUATION',execution_sha256='execution',method_sha256='method')


def test_reserved_role_requires_exact_independent_release_and_calibration_freeze():
    freeze_record = dict(path='calibration.json',sha256='hash')
    freeze = dict(completed=True,method_sha256='method',execution_protocol_sha256='execution',
        calibration_object_indices=list(range(16)))
    release = dict(schema='point-prior-reserved-release-v0.1',
        status='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE',method_sha256='method',
        execution_protocol_sha256='execution',object_indices=list(range(16,48)),
        calibration_freeze=freeze_record)
    kw = dict(execution_sha256='execution',method_sha256='method',release=release,
        calibration_freeze=freeze,calibration_freeze_record=freeze_record)
    selected = runner.eligible_objects(plan(),'RESERVED_EVALUATION',**kw)
    assert [o['index'] for o in selected] == list(range(16,48))
    for key,value in [('completed',False),('calibration_object_indices',list(range(15))),
                      ('execution_protocol_sha256','another')]:
        with pytest.raises(ValueError,match='explicit reserved release'):
            runner.eligible_objects(plan(),'RESERVED_EVALUATION',**{**kw,'calibration_freeze':{**freeze,key:value}})
    with pytest.raises(ValueError,match='reserved release'):
        runner.eligible_objects(plan(),'CALIBRATION',**kw)


def test_all16_pair_completeness_before_arrays_and_reserved_files_never_open(tmp_path,monkeypatch):
    p = plan()
    for obj in p['objects']:
        for label in ('A','B'):
            cell_id = obj['object_id']+'__'+label
            output = 'captures/'+cell_id
            protocol_path = tmp_path/(cell_id+'.json')
            protocol_path.write_text(json.dumps(dict(cell_id=cell_id,output=output,
                evaluation_role=obj['role'],integration_hz=120)))
            p['cells'].append(dict(id=cell_id,object_id=obj['object_id'],role=obj['role'],
                output=output,protocol=runner.record(protocol_path,tmp_path)))
            # No reserved capture directory exists at all. Any accidental open
            # of one therefore fails even if the metadata filter is bypassed.
            if obj['index'] < 16:
                directory = tmp_path/output
                directory.mkdir(parents=True)
                (directory/'trajectory.npz').write_bytes(b'NOT AN ARRAY FILE')
                runner.write(directory/'execution.json',dict(terminal=True,completed=True))
                runner.write(directory/'summary.json',dict(completed=True,frames=61,steps=360,
                    trajectory_sha256=runner.sha(directory/'trajectory.npz')))
    selected = runner.eligible_objects(p,'CALIBRATION',execution_sha256='x',method_sha256='m')
    monkeypatch.setattr(np,'load',lambda *a,**k: pytest.fail('Pair preflight must not open arrays'))
    pairs = runner.capture_pairs(p,selected,root=tmp_path)
    assert sorted(pairs) == list(range(16)) and all(set(v) == {'A','B'} for v in pairs.values())
    failed = tmp_path/'captures/point_prior_015__B/execution.json'
    runner.write(failed,dict(terminal=True,completed=False))
    with pytest.raises(ValueError,match='ALL selected-role'):
        runner.capture_pairs(p,selected,root=tmp_path)


def test_weighted_displacement_moments_match_closed_form_and_constant_field():
    u = np.arange(4*3*2*3,dtype=float).reshape(4,3,2,3)*1e-4
    w = np.array([.1,.2,.3,.4])
    mean,std = runner.weighted_displacement_moments(u,w,chunk=2)
    expected = np.average(u,axis=0,weights=w)
    np.testing.assert_allclose(mean,expected,atol=1e-18,rtol=1e-14)
    np.testing.assert_allclose(std**2,np.average((u-expected)**2,axis=0,weights=w),atol=1e-20)
    constant = np.broadcast_to(np.array([1e-3,-2e-3,5e-3]),(4,3,2,3)).copy()
    mean,std = runner.weighted_displacement_moments(constant,w)
    np.testing.assert_array_equal(mean,constant[0])
    np.testing.assert_array_equal(std,np.zeros_like(std))
    # A common large displacement does not enter E[u²]-E[u]² cancellation.
    shifted_mean,shifted_std = runner.weighted_displacement_moments(u+1e6,w,chunk=1)
    np.testing.assert_allclose(shifted_mean-1e6,expected,atol=2e-10,rtol=0)
    np.testing.assert_allclose(shifted_std,np.sqrt(np.average((u-expected)**2,axis=0,weights=w)),rtol=1e-7)
    with pytest.raises(ValueError,match='normalized'):
        runner.weighted_displacement_moments(u,[.1,.2,.3,.3])


def test_band_score_target_union_and_common_fair_width_window():
    truth_a = np.zeros((61,1,3)); truth_b = truth_a.copy()
    mean = truth_a.copy(); mean[25] = 2.
    std_a = np.ones_like(mean); std_a[25] = 999.
    std_b = np.ones_like(mean)*2
    base = runner.target_metrics(truth_a,truth_b,mean,np.zeros_like(mean),std_a,std_b,after_view=False)
    view = runner.target_metrics(truth_a,truth_b,mean,np.zeros_like(mean),std_a,std_b,after_view=True)
    assert base['individual_band_score'] == pytest.approx(2./999)
    assert view['individual_band_score'] == 0
    assert base['A_after1s_MSE_m2'] > 0 and base['A_after1p5s_MSE_m2'] == 0
    assert base['mean_target_scale_m'] == view['mean_target_scale_m'] == pytest.approx((30+61*2)/91)
    assert base['latent_target_std_RMS_m'] == view['latent_target_std_RMS_m'] == pytest.approx(np.sqrt((30+61*4)/91))


def test_episode_freezes_choice_before_A24_and_states_before_any_future_or_labels(tmp_path,monkeypatch):
    import torch
    from experiments import campaign_sparse_depth_queries as rays_module
    index = 3
    rest = np.zeros((1,3))
    public = dict(rest_points_m=rest.tolist(),surface_faces=[])
    queries = [dict(xs=list(range(36)),frame_index=i) for i in (10,20)]
    candidates = [dict(xs=list(range(36)),frame_index=24) for _ in range(4)]
    destination = tmp_path/'point_prior_003'
    events = []
    original_read = runner.read

    def rec(path,*a,**k):
        return dict(path=str(path),sha256=runner.sha(path))
    monkeypatch.setattr(runner,'record',rec)
    monkeypatch.setattr(runner,'checked',lambda r,*a,**k: Path(r['path']))
    spec_path = tmp_path/'PRIVATE-truth.json'
    runner.write(spec_path,dict(fixture_nodes=[4],material=dict(youngs_modulus_Pa=1000.,
        density_kg_m3=500.,poissons_ratio=.3,body_linear_drag_s_inverse=.5)))

    def guarded_read(path):
        if Path(path) == spec_path:
            assert (destination/'posterior-states-frozen.json').is_file()
            events.append('parameter-truth')
        return original_read(path)
    monkeypatch.setattr(runner,'read',guarded_read)

    def actual(trajectory,frames,*args):
        if frames == [10,20]:
            assert not (destination/'active-choice-frozen.json').exists()
            events.append('earlyA')
            return CensoredRanges(np.ones(72,bool),np.zeros(72))
        assert frames == [24] and (destination/'active-choice-frozen.json').is_file()
        assert 'choose' in events and 'future-A' not in events
        events.append('actualA24')
        return CensoredRanges(np.ones(36,bool),np.zeros(36))
    monkeypatch.setattr(runner,'_actual_observation',actual)

    def subset(trajectory,frames,_rest):
        assert frames == list(range(61))
        freeze = destination/'posterior-states-frozen.json'
        assert freeze.is_file()
        assert original_read(freeze)['future_A_or_B_truth_opened'] is False
        with np.load(destination/'predictive-states.npz') as data:
            assert all(name+'__weights' in data for name in runner.STATE_NAMES)
        events.append('future-'+trajectory['label'])
        return np.ones((1,61,1,3))*1e-3
    monkeypatch.setattr(runner,'_load_positions_subset',subset)

    summary = dict(seed=20261112,count_per_mask=3,support_categories=[
        dict(node_id=4,prior_probability=.8,posterior_probability=.9),
        dict(node_id=9,prior_probability=.2,posterior_probability=.1)],
        material=[dict(name=n,quantile05=lo,quantile95=hi) for n,lo,hi in (
            ('E_Pa',900.,1100.),('rho_kg_m3',450.,550.),('nu',.2,.4),('body_drag_s_inverse',.4,.6))])
    empirical = dict(reference_indices=np.zeros(2048,int),geometry_ids=np.full(2048,4),
        physical_coordinates=np.tile([1000.,500.,.3,.5],(2048,1)),weights=np.full(2048,1./2048))
    monkeypatch.setattr(runner,'condition_reference',lambda *a: dict(empirical_posterior=empirical,
        summaries=[summary],stability={'synthetic':True},seeds=[dict(seed=20261112,
        log_likelihood=np.zeros(6),prefixes=[dict(count_per_mask=3,indices=np.arange(6),weights=np.ones(6)/6)])]))

    class Forward:
        def predict(self,clock,geometry,physical,device):
            events.append('predict')
            return dict(positions_m=torch.zeros((len(geometry),61,1,3),dtype=torch.float64))
    monkeypatch.setattr(rays_module,'predict_ranges',lambda positions,faces,queries,**kw:
        CensoredRanges(np.ones((len(positions),36),bool),np.zeros((len(positions),36))))

    def choose(features,predictions,model,device):
        assert features.shape == (2048,90) and len(predictions) == 4
        assert events.count('predict') == 8 and 'actualA24' not in events
        events.append('choose')
        return dict(selected_index=2)
    monkeypatch.setattr(runner,'select_point_active_view',choose)
    monkeypatch.setattr(runner,'update_after_view',lambda *a: (
        np.full(2048,1./2048),dict(effective_sample_size=2048.)))
    obj = dict(index=index,role='CALIBRATION',object_id='point_prior_003',realized_hypothesis=rec(spec_path))
    pair = {label:dict(trajectory=dict(label=label)) for label in ('A','B')}
    sensor = dict(range_std_m=.0002,visibility_error_probability=.005,background_std_m=.05,outlier_probability=0.)
    context = (Forward(),public,SimpleNamespace(primary_seed=20261112,counts_per_mask=(1,3)),
        queries,candidates,dict(working_sensor=sensor))
    protocol = dict(device='cpu',source_sha256={})
    result = runner.analyze_episode(obj,pair,context,protocol,'executionSHA',destination)
    assert result['posterior_frozen_before_future_truth'] is True
    assert result['execution_protocol_sha256'] == 'executionSHA'
    assert result['selected_active_view'] == 2
    assert events.count('actualA24') == 4
    assert events.index('choose') < events.index('actualA24') < events.index('future-A') < events.index('parameter-truth')
    assert result['joint_band_score'] == max(r['individual_band_score'] for r in result['states'].values())
    assert set(result['states']) == set(runner.STATE_NAMES)
    assert result['supporttruthprobability'] == .9


def test_execution_validation_rejects_partial_source_map_but_cpu_report_can_ignore_CVD(monkeypatch):
    method = dict(cohort_plan=dict(path='plan.json',sha256='plan'),base_inference=dict(library_references=[]))
    p = plan()
    p.update(public_manifest=dict(path='public.json',sha256='public'),source_sha256={})
    cache = dict(completed=True,source_sha256={})
    expected_sources = {'experiments/required.py':'exact-source-hash'}
    execution = dict(schema=runner.SCHEMA,status='FROZEN_POINT_PRIOR_ANALYSIS_EXECUTION',
        method=dict(path=runner.METHOD_PATH,sha256=runner.METHOD_SHA256),cpu_threads=2,
        forward_particle_chunk=512,particle_count=2048,support_prior=runner.SUPPORT_PRIOR,
        cohort_plan=method['cohort_plan'],prior_libraries=[],cache=dict(path='cache.json',sha256='cache'),
        source_sha256={},public_manifest=p['public_manifest'],roles=p['roles'],
        analysis_output='synthetic-output',device='cuda:0',CUDA_VISIBLE_DEVICES='2')
    values = {'execution.json':execution,runner.METHOD_PATH:method,'plan.json':p,'cache.json':cache}
    monkeypatch.setattr(runner,'checked',lambda r: r['path'])
    monkeypatch.setattr(runner,'read',lambda p: values[p])
    monkeypatch.setattr(runner,'_expected_source_pins',lambda *a: expected_sources)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError,match='dependency closure'):
        runner.validate_execution('execution.json','hash',check_environment=False)
    execution['source_sha256'] = expected_sources
    runner.validate_execution('execution.json','hash',check_environment=False)
    with pytest.raises(ValueError,match='environment mismatch'):
        runner.validate_execution('execution.json','hash')
