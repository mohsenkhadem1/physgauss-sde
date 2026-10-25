"""Synthetic data-binding/retention controls, never actual fresh data."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_fresh64_inner_resolution_control_v01 as m


def test_saved_response_and_both_independent_noises_are_hash_bound():
    x=np.arange(24,dtype=float)*1e-5;pn=np.full(24,2e-6);qn=np.zeros((2,3,24));qn[0,0]=-3e-6
    expected=[(m.PREFIX,x+pn),(m.PREFIX,x+qn[0,0])]
    digest=m.fresh.oldstudy.observation_fingerprint(expected)
    actual=m.reconstruct(x,pn,qn,digest)
    for (_,a),(_,b)in zip(actual,expected):np.testing.assert_array_equal(a,b)
    for field in('truth','prefix','probe'):
        xx=x.copy();pp=pn.copy();qq=qn.copy()
        if field=='truth':xx[0]+=1e-9
        elif field=='prefix':pp[0]+=1e-9
        else:qq[0,0,0]+=1e-9
        with pytest.raises(ValueError,match='EXACT'):m.reconstruct(xx,pp,qq,digest)


def test_load_only_permitted_members_never_unit_material_or_held(monkeypatch):
    x=np.arange(24,dtype=float);pn=np.ones(24);qn=np.full((2,3,24),2.)
    accesses=[]
    class Archive:
        def __init__(self,kind):self.kind=kind
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def __getitem__(self,key):
            accesses.append((self.kind,key));allowed={'truth':{'response_m':x},'draws':{'prefix_noise':pn,'probe_noise':qn}}
            if key not in allowed[self.kind]:raise AssertionError('Forbidden scientific member')
            return allowed[self.kind][key]
    monkeypatch.setattr(m.np,'load',lambda path,**kwargs:Archive('truth'if str(path).endswith('truth.npz')else'draws'))
    expected=m.fresh.oldstudy.observation_fingerprint([(m.PREFIX,x+pn),(m.PREFIX,x+qn[0,0])])
    origin=dict(truth_responses={m.PREFIX:dict(archive=dict(path='truth.npz'))},draws=dict(path='draws.npz'),context=dict(observations_SHA256=expected))
    m.load_observations(origin)
    assert accesses==[('truth','response_m'),('draws','prefix_noise'),('draws','probe_noise')]


def failed():
    return dict(qualified=False,physical_valid=True,kind='nonlinear',numerical_backend='nonlinear_conditional_reference_v03_MIS',raw_SHA256='hash',context=dict(protocol_SHA256=m.PRIMARY_SHA,index=5,phase='normal-r0',draws_SHA256='d',observations_SHA256='o',model_fingerprint='m',recipe_SHA256='r'),diagnostics=dict(qualified=False,schema='nonlinear-conditional-reference-v03',levels=[dict(status='computed')for _ in range(5)]))


def test_failed_context_not_replaced_or_scientific_fields_read():
    class Forbidden(dict):
        def __getitem__(self,key):
            if key in('PIT','covered','true_unit','target_RMS_m'):raise AssertionError('Scientific field read')
            return super().__getitem__(key)
    row=Forbidden(failed());m.failed_context(row)
    for name,value in [('qualified',True),('physical_valid',False),('kind','linear')]:
        bad=failed();bad[name]=value
        with pytest.raises(ValueError):m.failed_context(bad)
    bad=failed();bad['context']['index']=6
    with pytest.raises(ValueError):m.failed_context(bad)


def test_new_root_release_required_before_archives(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LOCAL',tmp_path)
    p=dict(source_SHA256={'s':'h'},origin={'fixed':'obs'})
    for release in [dict(approved=False),dict(approved=True,protocol_SHA256='wrong'),dict(approved=True,protocol_SHA256='digest',source_SHA256={'s':'wrong'},origin=p['origin'])]:
        (tmp_path/'release.json').write_text(json.dumps(release))
        with pytest.raises(ValueError):m.release_gate(p,'digest')
    (tmp_path/'release.json').write_text(json.dumps(dict(approved=True,protocol_SHA256='digest',source_SHA256=p['source_SHA256'],origin=p['origin'])))
    m.release_gate(p,'digest')


def diagnostics():
    metric=dict(passed=True,log_evidence_difference=0.,maximum_CDF_difference=0.,maximum_unit_quantile_difference=0.,maximum_unit_mean_difference=0.,maximum_predictive_mean_difference_m=0.,maximum_predictive_SD_difference_m=0.)
    return dict(schema='nonlinear-conditional-reference-v04',gates=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01),levels=[dict(outer=o,inner=i)for o,i in[(512,128),(512,256),(512,512),(512,1024),(128,1024),(256,1024)]],successive={a:[dict(coarse=c,fine=f,**metric)for c,f in pairs]for a,pairs in [('inner',[(128,256),(256,512),(512,1024)]),('outer',[(128,256),(256,512)])]},independent_adaptive=dict(passed=True,epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6,log_evidence_difference=0.,maximum_CDF_error=0.,adaptive_refinement_log_evidence_difference=0.,adaptive_refinement_maximum_CDF_difference=0.,maximum_QUADPACK_relative_error_estimates={'1e-06':0.,'1e-08':0.}))


def test_extended_numerical_gate_rejects_relabel_or_threshold_change():
    d=diagnostics();r=dict(qualified=True,diagnostics=d)
    assert m.numerical_checks(r)['passed']
    d['successive']['inner'][-1]['maximum_CDF_difference']=.00501
    assert not m.numerical_checks(r)['passed']
    d=diagnostics();d['levels'][3]['inner']=512
    with pytest.raises(ValueError):m.numerical_checks(dict(qualified=True,diagnostics=d))
    d=diagnostics();d['gates']['CDF']=.006
    with pytest.raises(ValueError):m.numerical_checks(dict(qualified=True,diagnostics=d))


def test_controller_failure_never_completed_or_admitted():
    assert m.cost.phase_completion(dict(failure='monitor',worker_exit_code=0,cleanup_exit_code=0),dict(completed=True,passed=True))==dict(completed=False,qualified=False)
    assert m.RESOURCES['controller_CPU']==18 and m.RESOURCES['worker_CPU']==17
    assert m.PROJECTION['maximum_hypotheses']==1536*4096
    assert m.PROJECTION['target_chunk_bytes']==1024*96*8
    rows=m.PROJECTION['per_case']
    assert rows[m.CASES[0]['id']]['final_outer_rows']==1536
    assert rows[m.CASES[1]['id']]['final_outer_rows']==3072
    assert rows[m.CASES[1]['id']]['atoms_per_row']==2048
    assert rows[m.CASES[2]['id']]['maximum_hypotheses']==3145728


def test_outer_both_models_and_axial_observation_identity():
    x=np.arange(24,dtype=float);a=-x;pn=np.ones(24);qn=np.full((2,3,24),2.)
    expected=[(m.PREFIX,x+pn),('paired_axial_x180',a+qn[1,0])]
    digest=m.fresh.oldstudy.observation_fingerprint(expected)
    m.reconstruct(x,pn,qn,digest,probe_response=a,probe_label='paired_axial_x180',family=1)
    with pytest.raises(ValueError,match='EXACT'):m.reconstruct(x,pn,qn,'wrong',probe_response=a,probe_label='paired_axial_x180',family=1)
    for case in m.CASES[1:]:
        plan=m.case_plan(case)
        assert plan.inner==(128,256,512)and plan.outer==(128,256,512,1024)
        row=failed();row['kind']=case['kind'];row['context'].update(index=case['index'],phase=case['phase'])
        if case['kind']=='linear':row.update(numerical_backend='conditional_amplitude_PWL_v02');row['diagnostics']['schema']='conditional-amplitude-v02'
        m.failed_context(row,case)
    assert m.RESOURCES['wall_cap_s']==2400


def test_old512_sufficient_statistics_reproduction_detects_tamper():
    current=dict(predictive_mean=np.zeros(96),predictive_SD=np.ones(96),unit_CDF=np.tile(np.linspace(0,1,33),(2,1)))
    saved=dict(mean=current['predictive_mean'].copy(),SD=current['predictive_SD'].copy(),unit_CDF=current['unit_CDF'].copy())
    assert m.old512_parity(current,saved)['passed']
    saved['mean'][4]=3e-12
    assert not m.old512_parity(current,saved)['passed']
    saved['mean'][4]=0;saved['unit_CDF'][0,4]+=2e-8
    assert not m.old512_parity(current,saved)['passed']
    saved['unit_CDF']=np.zeros((2,32))
    with pytest.raises(ValueError):m.old512_parity(current,saved)
