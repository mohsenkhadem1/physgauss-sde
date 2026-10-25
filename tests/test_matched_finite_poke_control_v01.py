import copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from experiments import run_matched_finite_poke_control_v01 as m


def catalogs():
    a=np.zeros((8,60));a[:,12:]=np.arange(8)[:,None]*1e-5
    return dict(NH=a,linear=np.column_stack([a[:,:12],2*a[:,12:]]))


def good_choice():return m.model(catalogs()['NH']).choose(np.zeros(4))


def test_same_uniform8_and_original_fivelevel_choice_predicates():
    a=catalogs()['NH'];ref=m.model(a);base=ref.prefix_posterior(np.zeros(4));np.testing.assert_allclose(np.exp(base),np.full(8,1/8),rtol=0,atol=1e-16)
    c=good_choice();assert m.choice_check(c)and c['selected_action']==0
    bad=copy.deepcopy(c);bad['actions'][0]['qualified']=False
    with pytest.raises(ValueError,match='flags'):m.choice_check(bad)
    bad=copy.deepcopy(c);bad['actions'].pop()
    with pytest.raises(ValueError,match='All8'):m.choice_check(bad)
    with pytest.raises(ValueError):m.model(a[:-1])


def test_continuous32_barrier_precedes_catalog_or_future_array(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'LOCAL',tmp_path);monkeypatch.setattr(m,'guard',lambda start:None);monkeypatch.setattr(m,'sources',lambda:{'source':'pin'})
    m.write(tmp_path/'release.json',dict(approved=True,parent_protocol_SHA256='parent',source_SHA256={'source':'pin'},resources=m.LIMITS));seen=[]
    monkeypatch.setattr(m.continuous,'validate',lambda digest:seen.append('validate')or{})
    def blocked(digest):seen.append('all32');raise ValueError('Continuous choice rootclosure missing')
    monkeypatch.setattr(m.continuous,'all_seals_gate',blocked);monkeypatch.setattr(m.continuous,'inputs_gate',lambda digest:pytest.fail('No inputs before all32'))
    monkeypatch.setattr(m,'finite_catalog',lambda *args:pytest.fail('No models before all32'));monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No future arrays'))
    with pytest.raises(ValueError,match='rootclosure'):m.choose('parent')
    assert seen==['validate','all32']


def test_full_NH_and_linear60_projection_uses_all88_same_captures(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'guard',lambda start:None)
    points=m.continuous.design.physics.measurement_points();loads=[s['id']for s in m.continuous.design.physics.load_specs()];captures={};checked=[]
    for state in range(8):
        for j,label in enumerate(loads):
            q=np.arange(48.).reshape(16,3)*1e-8+(state*100+j)*1e-6;path=tmp_path/f'{state}-{j}.npz';np.savez_compressed(path,query_points_m=points,query_displacement_m=q,linear_displacement_m=2*q);captures[state,label]=dict(path=str(path),sha256=m.sha(path))
    monkeypatch.setattr(m.continuous,'capture',lambda p,i,label:captures[i,label]);monkeypatch.setattr(m.continuous,'verify',lambda rows:checked.append(len(rows)))
    monkeypatch.setattr(m.continuous.physics_stage.geometry,'field_queries',lambda n,u,p:u.copy())
    real=np.load
    def load(*args,**kwargs):assert checked==[88];return real(*args,**kwargs)
    monkeypatch.setattr(m.np,'load',load)
    values,records,labels=m.finite_catalog(dict(objects=m.continuous.design.objects()),0.)
    assert len(records)==88 and len(labels)==60 and values['NH'].shape==(8,60);np.testing.assert_array_equal(values['linear'],2*values['NH'])
    for state in range(8):
        for k,spec in enumerate(m.continuous.design.physics.measurement_specs()):
            j=loads.index(spec['load']);i=next(i for i,p in enumerate(points)if np.array_equal(p,spec['point_m']));q=np.arange(48.).reshape(16,3)*1e-8+(state*100+j)*1e-6
            assert values['NH'][state,k]==q[i]@spec['projection']


def shared_files(monkeypatch,tmp_path):
    local=tmp_path/'finite';parent=tmp_path/'parent';local.mkdir();parent.mkdir();monkeypatch.setattr(m,'LOCAL',local);monkeypatch.setattr(m.continuous,'LOCAL',parent);monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'guard',lambda start:None)
    monkeypatch.setattr(m,'record',lambda p:dict(path=str(p),sha256=m.sha(p)));monkeypatch.setattr(m,'sources',lambda:{'procedure':'pin'})
    (local/'release.json').write_text('{}')
    for name in('protocol.json','release.json','execution-choices.json','root-closure-choices.json','inputs.json'):(parent/name).write_text('{}')
    prefixes=[]
    for index in range(32):
        path=parent/f'prefix-{index}.npz';np.savez_compressed(path,observed_prefix=np.zeros(4),prefix_noise=np.zeros(4));prefixes.append(dict(index=index,record=m.record(path),observation_fingerprint=m.continuous.design.fingerprint(np.zeros(4))))
    noise=parent/'noise.npz';np.savez_compressed(noise,poke_noise=np.zeros((4,8,8)))
    inputs=dict(prefixes=prefixes,poke_noise=m.record(noise));return local,parent,inputs


def test_all64_choices_only_prefix_no_trial_identity_or_counterfactual_reads(monkeypatch,tmp_path):
    local,parent,inputs=shared_files(monkeypatch,tmp_path);seals=[dict(index=i,seal={'bound':i})for i in range(32)];seen=[]
    class NoTruth(dict):
        def __getitem__(self,key):pytest.fail('Finite choice cannotaccess generatingtrialidentity')
    monkeypatch.setattr(m,'barrier',lambda *a:(NoTruth(),seals,inputs));monkeypatch.setattr(m,'finite_catalog',lambda *a:(catalogs(),[],['label']*60));choice=good_choice()
    class Ref:
        def choose(self,obs):seen.append(obs.copy());return copy.deepcopy(choice)
    monkeypatch.setattr(m,'model',lambda array:Ref());m.choose('parent');value=json.loads((local/'finite-choice-seals.json').read_text())
    assert value['completed']and len(value['choices'])==64 and len(seen)==64 and all(np.array_equal(v,np.zeros(4))for v in seen)
    assert[(v['model'],v['index'])for v in value['choices']]==[(name,i)for name in m.MODELS for i in range(32)]


@pytest.mark.parametrize('negative',[False,True])
def test_evaluator_all512_freshprefix_updates_and_failedchoice_blocks_fullcomparison(monkeypatch,tmp_path,negative):
    local,parent,inputs=shared_files(monkeypatch,tmp_path);arrays=catalogs();catalog=local/'catalog.npz';np.savez_compressed(catalog,**arrays);(local/'finite-choice-seals.json').write_text('{}')
    choice=good_choice();choices=[]
    for name in m.MODELS:
        for index in range(32):choices.append(dict(model=name,index=index,choice=copy.deepcopy(choice),observation_fingerprint=inputs['prefixes'][index]['observation_fingerprint']))
    if negative:choices[5]['choice'].update(qualified=False,selected_action=None)
    sealed=dict(catalog=m.record(catalog),choices=choices,wall_s=0.);p=dict(trials=m.continuous.design.trials());barrier=[]
    def gate(*args):barrier.append('all64');return p,inputs,sealed
    monkeypatch.setattr(m,'sealed_choices',gate);real_load=np.load
    def load(*args,**kwargs):assert barrier==['all64'];return real_load(*args,**kwargs)
    monkeypatch.setattr(m.np,'load',load);real_model=m.model;updates=[]
    class Checked:
        def __init__(self,a):self.ref=real_model(a)
        def prefix_posterior(self,obs):return self.ref.prefix_posterior(obs)
        def after_one_poke(self,base,action,outcome):
            np.testing.assert_array_equal(base,self.ref.prefix_posterior(np.zeros(4)));updates.append(action);return self.ref.after_one_poke(base,action,outcome)
        def moments(self,weights):return self.ref.moments(weights)
    monkeypatch.setattr(m,'model',Checked);m.evaluate('parent');result=json.loads((local/'evaluation-results-LOCAL.json').read_text());summary=result['summary']
    assert len(result['rows'])==64 and len(updates)==512 and summary['all64_retained']and summary['retained_counterfactuals_per_model']==256 and summary['comparison_blocked']is negative
    assert all(len(r['counterfactuals'])==8 for r in result['rows'])
    if negative:assert 'equal_object_mean_MSE_m2'not in summary
    else:
        for row in result['rows']:assert row['policies']['uniform_random']['MSE_m2']==np.mean([v['MSE_m2']for v in row['counterfactuals']])


def synthetic_rows():
    rows=[]
    for name in m.MODELS:
        for r in range(4):
            for i in range(8):
                counter=[dict(action=a,MSE_m2=100*i+r*r+a+(10 if name=='linear'else 0),posterior_variance_m2=1.)for a in range(8)];before=dict(MSE_m2=999.,posterior_variance_m2=2.);rows.append(dict(model=name,index=8*r+i,object_index=i,replicate=r,choice_qualified=True,selected_action=0,fixed_action=5,before=before,counterfactuals=counter,policies=m.policy_metrics(before,counter,0,True)))
    return rows


def test_equal4noise_then8objects_exactrandom_MSE_and_no_missingdenominator():
    rows=synthetic_rows();s=m.summarize(rows);assert s['equal_object_mean_MSE_m2']['NH']['uniform_random']==357. and s['equal_object_mean_MSE_m2']['NH']['fixed']==358.5 and s['NH_minus_linear_MSE_m2']['adaptive']==-10.
    with pytest.raises(ValueError,match='64'):m.summarize(rows[:-1])
    rows[0]['policies']['uniform_random']['MSE_m2']+=1
    with pytest.raises(ValueError,match='Recomputed'):m.summarize(rows)


def test_incomplete64_or_unrefined_choice_refuses_before_evaluation_array(monkeypatch,tmp_path):
    local,parent,inputs=shared_files(monkeypatch,tmp_path);seals=[dict(index=i)for i in range(32)];monkeypatch.setattr(m,'barrier',lambda *a:({},seals,inputs));monkeypatch.setattr(m.continuous.base,'absent',lambda pid:None);monkeypatch.setattr(m.continuous,'verify',lambda v:None)
    value=dict(completed=True,pid=10,source_SHA256=m.sources(),parent_protocol=m.record(parent/'protocol.json'),parent_release=m.record(parent/'release.json'),root_release=m.record(local/'release.json'),continuous32_choice_seals=seals,continuous_choice_execution=m.record(parent/'execution-choices.json'),continuous_choice_root_closure=m.record(parent/'root-closure-choices.json'),inputs=m.record(parent/'inputs.json'),no_trial_identity_or_mapped_outcome_used=True,source_captures=[],catalog={},choices=[])
    m.write(local/'finite-choice-seals.json',value);monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('No outcomesbefore64seals'))
    with pytest.raises(ValueError,match='64'):m.evaluate('parent')


def test_real_CPU17_guard_only_no_data_calls():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=['taskset','-c','17',sys.executable,'-c','import time; from experiments.run_matched_finite_poke_control_v01 import guard; guard(time.monotonic()); print("guard-PASS")'];v=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30);assert v.returncode==0 and v.stdout.strip()=='guard-PASS',v.stderr


def test_honest_failed_quadrature_retained_without_relaxation():
    c=good_choice();prior=c['prior_variance_m2'];row=c['actions'][0]
    row['levels'][-2]['risk_m2']=.9*prior;row['levels'][-1]['risk_m2']=.5*prior
    row.update(risk_m2=.5*prior,refinement_indicator_m2=abs(.5*prior-.9*prior),qualified=False)
    c.update(qualified=False,numerical_ties=[0],selected_action=None)
    assert m.choice_check(c)is False and len(c['actions'])==8
