import copy,json,os,sys,time
from pathlib import Path
import numpy as np
import pytest
from experiments import audit_matched_finite_poke_control_v01 as a
from experiments import run_matched_finite_poke_control_v01 as producer


def fixture(monkeypatch,tmp_path):
    local=tmp_path/'finite';parent=tmp_path/'parent';local.mkdir();parent.mkdir()
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'LOCAL',local);monkeypatch.setattr(a.c,'LOCAL',parent);monkeypatch.setattr(a,'guard',lambda start:None)
    monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=a.run.sha(p)))
    monkeypatch.setattr(a,'sources',lambda:{'auditor':'pin'});monkeypatch.setattr(a.run,'sources',lambda:{'producer':'pin'})
    for name in('protocol.json','release.json'):(parent/name).write_text('{}')
    for name in('release.json','finite-choice-seals.json','evaluation-results-LOCAL.json'):(local/name).write_text('{}')
    rng=np.random.default_rng(123);nh=rng.normal(0,15e-6,(8,60));linear=nh.copy();linear[:,:12]*=.8;linear[:,12:]*=1.2
    arrays=dict(NH=nh,linear=linear);noise=rng.normal(0,20e-6,(4,8,8));n=parent/'noise.npz';np.savez(n,poke_noise=noise)
    catalog=local/'catalogs-LOCAL.npz';labels=[str(i)for i in range(60)];np.savez(catalog,**arrays,labels=labels)
    inputs=dict(poke_noise=a.run.record(n),prefixes=[]);observations=[]
    for i in range(32):
        eps=rng.normal(0,20e-6,4);obs=nh[i%8,:4]+eps;observations.append(obs);path=parent/f'prefix-{i}.npz';np.savez(path,observed_prefix=obs,prefix_noise=eps)
        inputs['prefixes'].append(dict(index=i,record=a.run.record(path),observation_fingerprint=a.c.design.fingerprint(obs)))
    return local,parent,arrays,labels,noise,inputs,observations


def test_independent_gaussian_update_analytic_odds_and_no_chained_poke():
    catalog=np.zeros((8,60));catalog[4:,:4]=1e-5;catalog[4:,4]=2e-5;catalog[4:,5]=-2e-5;catalog[4:,12:]=1e-3;obs=np.zeros(4)
    first=a.direct_posterior(catalog,obs,0,2e-5);expected=1/(1+np.exp(0.)) # prefix odds exp(-.5), action odds exp(+.5)
    assert first['weights'][4:].sum()==pytest.approx(expected,abs=2e-16)
    second=a.direct_posterior(catalog,obs,1,2e-5);expected_second=1/(1+np.exp(2.))
    assert second['weights'][4:].sum()==pytest.approx(expected_second,abs=2e-16)
    np.testing.assert_allclose(first['mean'],.001*.5,rtol=0,atol=1e-18)
    np.testing.assert_allclose(first['variance'],.001**2*.25,rtol=0,atol=1e-21)
    with pytest.raises(ValueError):a.direct_posterior(catalog[:-1],obs)


def test_momentbands_are_descriptive_and_finite_not_coverage_level():
    m=dict(mean=np.zeros(48),variance=np.full(48,4e-12));truth=np.r_[np.full(24,3e-6),np.full(24,5e-6)]
    values=a.moment_metrics(m,truth);assert values['moment_band_2SD_inclusion_fraction']==.5
    assert values['moment_band_2SD_mean_width_m']==pytest.approx(8e-6)


def test_continuous_choice_closure_stops_before_finite_data(monkeypatch,tmp_path):
    local,*_=fixture(monkeypatch,tmp_path);a.run.write(local/'release.json',dict(approved=True,parent_protocol_SHA256='p',source_SHA256=a.run.sources(),resources=a.run.LIMITS))
    seen=[];monkeypatch.setattr(a.c,'validate',lambda p:seen.append('validate')or{})
    def fail(p):seen.append('continuous32');raise ValueError('missing closed32')
    monkeypatch.setattr(a.c,'all_seals_gate',fail);monkeypatch.setattr(a.c,'inputs_gate',lambda p:pytest.fail('No inputs before choiceclosure'));monkeypatch.setattr(a.np,'load',lambda *x,**k:pytest.fail('No arrays before barriers'))
    with pytest.raises(ValueError,match='closed32'):a.metadata_barrier('p')
    assert seen==['validate','continuous32']


def closure_fixture(monkeypatch,tmp_path):
    local=tmp_path/'finite';op=tmp_path/'operator';local.mkdir();op.mkdir();monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'LOCAL',local);monkeypatch.setattr(a,'OPERATOR',op)
    monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(Path(p).relative_to(tmp_path)),sha256=a.run.sha(p)))
    files={'audit.py':'a','test.py':'t','producer.py':'p','parent.py':'c'}
    for p,s in files.items():(tmp_path/p).write_text(s)
    pins={p:a.run.sha(tmp_path/p)for p in files};monkeypatch.setattr(a,'sources',lambda:{k:pins[k]for k in('audit.py','test.py')});monkeypatch.setattr(a.run,'sources',lambda:{'producer.py':pins['producer.py']});monkeypatch.setattr(a.c,'sources',lambda:{'parent.py':pins['parent.py']})
    approval=op/'root-preapproval.json';a.run.write(approval,dict(approved=True,schema='noisy-poke-operator-preapproval-v01',source_SHA256=pins,stage_caps={'finite_choose':[17,1800,4*1024**3],'finite_audit':[16,1800,4*1024**3]}));result=local/'finite-choice-seals.json';result.write_text('{}');r=a.run.record(result)
    identity=dict(pid=7654321,PGID=7654321,PPID=100,starttime=123,state='R',command='bootstrap')
    stage=op/'stages/finite_choose/execution.json';stage.parent.mkdir(parents=True)
    value=dict(phase='finite_choose',parent_protocol_SHA256='p',completed=True,reaped=True,exit_code=0,cleanup_exit_code=0,child_identity=identity,source_SHA256=pins,operator_preapproval=a.run.record(approval),resources=dict(CPU=17,wall_s=1800,RSS_bytes=4*1024**3),command=['taskset','-c','17',sys.executable,'-u','-m',a.run.__name__,'choose','--parent-sha','p'],outputs=[r],result=r,observed_owned=[identity],wall_s=1.,peak_RSS_bytes=1024)
    a.run.write(stage,value);closure=local/'root-closure-choose.json';v=dict(root_reaped=True,root_exit_code=0,root_owned_supervisor=True,external_subprocess_wait=True,all_owned_PID_and_groups_absent=True,parent_protocol_SHA256='p',source_SHA256=a.run.sources(),owned_pids=[identity['pid']],operator_preapproval=a.run.record(approval),operator_stage_execution=a.run.record(stage),result=r);a.run.write(closure,v);monkeypatch.setattr(a.c.base,'absent',lambda pid:None)
    return r,stage,value,closure,v,approval


def test_external_wait_writer_and_prepinned_auditor_not_self_asserted(monkeypatch,tmp_path):
    r,stage,value,path,v,approval=closure_fixture(monkeypatch,tmp_path)
    assert a.closure('choose','p',r)['pid']==7654321
    value['observed_owned'].append(dict(value['child_identity'],pid=7654322));a.run.write(stage,value);v['operator_stage_execution']=a.run.record(stage);a.run.write(path,v)
    with pytest.raises(ValueError,match='unexpected descendants'):a.closure('choose','p',r)
    value['observed_owned']=value['observed_owned'][:1];a.run.write(stage,value);v['operator_stage_execution']=a.run.record(stage);a.run.write(path,v)
    approval_value=a.run.read(approval);approval_value['source_SHA256']['audit.py']='0'*64;a.run.write(approval,approval_value);v['operator_preapproval']=a.run.record(approval);a.run.write(path,v)
    with pytest.raises(ValueError,match='prepinned'):a.closure('choose','p',r)


def test_unreaped_or_changed_result_is_rejected_before_npz(monkeypatch,tmp_path):
    r,stage,value,path,v,_=closure_fixture(monkeypatch,tmp_path);v['root_reaped']=False;a.run.write(path,v)
    monkeypatch.setattr(a.np,'load',lambda *x,**k:pytest.fail('No raw arrays'))
    with pytest.raises(ValueError,match='root closure'):a.closure('choose','p',r)
    v['root_reaped']=True;a.run.write(path,v);(a.ROOT/r['path']).write_text('tampered')
    with pytest.raises(ValueError,match='bytes'):a.closure('choose','p',r)


def make_full_result(monkeypatch,tmp_path,negative=False):
    local,parent,arrays,labels,noise,inputs,obs=fixture(monkeypatch,tmp_path);choice=producer.model(np.zeros((8,60))).choose(np.zeros(4));choices=[];rows=[];rawdir=local/'posteriors-LOCAL';rawdir.mkdir()
    for name in producer.MODELS:
        ref=producer.model(arrays[name])
        for index in range(32):
            obj,rep=index%8,index//8;selected=copy.deepcopy(choice)
            if negative and name=='linear'and index==3:
                selected['actions'][0]['levels'][-1]['risk_m2']=1e-20;selected['actions'][0].update(risk_m2=1e-20,refinement_indicator_m2=1e-20,qualified=False);selected.update(qualified=False,selected_action=None,numerical_ties=list(range(8)))
            choices.append(dict(model=name,index=index,choice=selected,observation_fingerprint=inputs['prefixes'][index]['observation_fingerprint']))
            base=ref.prefix_posterior(obs[index]);before=producer.mse_metrics(ref.moments(base),arrays['NH'][obj,12:]);counter=[];moments=[]
            for action in range(8):
                posterior=ref.after_one_poke(base,action,float(arrays['NH'][obj,4+action]+noise[rep,obj,action]));moment=ref.moments(posterior);moments.append((posterior,moment));counter.append(dict(action=action,**producer.mse_metrics(moment,arrays['NH'][obj,12:]),fresh_prefix_plus_ONE_scalar=True))
            raw=rawdir/f'{name}-{index:02d}.npz';np.savez(raw,mean_m=np.array([m['mean']for w,m in moments]),weights=np.array([np.exp(w)for w,m in moments]),before_mean_m=ref.moments(base)['mean'])
            qualified=selected['qualified'];rows.append(dict(model=name,index=index,object_index=obj,replicate=rep,choice_qualified=qualified,selected_action=selected['selected_action'],fixed_action=5,before=before,counterfactuals=counter,policies=producer.policy_metrics(before,counter,selected['selected_action'],qualified),posterior_archive=a.run.record(raw)))
    result=dict(rows=rows,summary=producer.summarize(rows));a.run.write(local/'compact-summary.json',result['summary']);seal=dict(catalog=a.run.record(local/'catalogs-LOCAL.npz'),choices=choices,source_captures=[])
    p=dict(trials=a.c.design.trials());monkeypatch.setattr(a,'metadata_barrier',lambda sha:(p,inputs,seal,result,{}));monkeypatch.setattr(a,'reconstruct_catalog',lambda p,check:(arrays,labels))
    return local,result,arrays


@pytest.mark.parametrize('negative',[False,True])
def test_full512_independent_updates_equalobject_metrics_and_negative_denominator(monkeypatch,tmp_path,negative):
    local,result,_=make_full_result(monkeypatch,tmp_path,negative);value=a.audit('p')
    assert value['audit_passed']and value['independently_recomputed_posterior_updates']==512 and value['summary']['comparison_blocked']is negative
    assert value['summary']['retained_counterfactuals_per_model']==256
    assert value['maximum_reproduction_errors']['mean_absolute_m']<1e-18
    if negative:assert not value['descriptive_momentband_diagnostics']and'equal_object_mean_MSE_m2'not in value['summary']
    else:
        for model in producer.MODELS:
            for policy in('before','fixed','adaptive','uniform_random'):
                expected=np.mean([np.mean([r['policies'][policy]['MSE_m2']for r in result['rows']if r['model']==model and r['object_index']==obj])for obj in range(8)])
                assert value['summary']['equal_object_mean_MSE_m2'][model][policy]==pytest.approx(expected,rel=1e-12)
        assert value['momentband_scope'].startswith('Mean±2SD descriptive')


def test_raw_weight_tamper_and_wrong_random_aggregation_detected(monkeypatch,tmp_path):
    local,result,_=make_full_result(monkeypatch,tmp_path);row=result['rows'][0];path=a.bound(row['posterior_archive'])
    with np.load(path)as z:values={k:z[k].copy()for k in z.files}
    values['weights'][0]=np.roll(values['weights'][0],1);np.savez(path,**values);row['posterior_archive']=a.run.record(path)
    with pytest.raises(ValueError,match='numerical'):a.audit('p')
    values['weights'][0]=np.roll(values['weights'][0],-1);np.savez(path,**values);row['posterior_archive']=a.run.record(path);row['policies']['uniform_random']['MSE_m2']*=2
    with pytest.raises(ValueError,match='numerical'):a.audit('p')


def test_own_guard_cpu16_no_gpu_and_exact_threads(monkeypatch):
    monkeypatch.setattr(a.os,'sched_getaffinity',lambda pid:{16});monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(k,'1')
    monkeypatch.setattr(a.shutil,'disk_usage',lambda p:type('Disk',(),{'free':100*1024**3})());a.guard(time.monotonic())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CPU16'):a.guard(time.monotonic())


def test_metadata_all64_and_last_archive_hash_before_any_deserialization(monkeypatch,tmp_path):
    local,parent,arrays,labels,noise,inputs,obs=fixture(monkeypatch,tmp_path)
    # File bytes here are deliberately NOT NPZ: the metadata barrier must only
    # hash them. Missing last membership/hash must fail without opening arrays.
    monkeypatch.setattr(a.c,'validate',lambda p:{})
    continuous_seals=[dict(index=i,seal={'commit':i})for i in range(32)]
    monkeypatch.setattr(a.c,'all_seals_gate',lambda p:continuous_seals);monkeypatch.setattr(a.c,'inputs_gate',lambda p:inputs)
    for name in('execution-choices.json','root-closure-choices.json','inputs.json'):(parent/name).write_text('{}')
    for name in('parent-audit.py','parent-audit-test.py'):(tmp_path/name).write_text('source')
    monkeypatch.setattr(a.c,'AUDIT_SOURCE','parent-audit.py');monkeypatch.setattr(a.c,'AUDIT_TEST','parent-audit-test.py')
    approval=tmp_path/'approval.json';approval.write_text('{}');approval_rec=a.run.record(approval)
    a.run.write(local/'release.json',dict(approved=True,parent_protocol_SHA256='p',source_SHA256=a.run.sources(),resources=a.run.LIMITS,root_owned_operator_preapproval=approval_rec))
    monkeypatch.setattr(a,'closure',lambda phase,*x:dict(pid=123,preapproval=approval_rec))
    loads=[v['id']for v in a.c.design.physics.load_specs()];captures={}
    for state in range(8):
        for j,label in enumerate(loads):
            p=tmp_path/f'capture-{state}-{j}';p.write_text('raw bytes');captures[state,label]=a.run.record(p)
    monkeypatch.setattr(a.c,'capture',lambda p,i,label:captures[i,label])
    capture_rows=[dict(state=i,label=label,capture=captures[i,label])for i in range(8)for label in loads]
    good=producer.model(np.zeros((8,60))).choose(np.zeros(4))
    choices=[dict(model=m,index=i,choice=good,prefix_input=inputs['prefixes'][i]['record'],observation_fingerprint=inputs['prefixes'][i]['observation_fingerprint'],fixed_action=5)for m in producer.MODELS for i in range(32)]
    seal=dict(completed=True,pid=123,source_SHA256=a.run.sources(),parent_protocol=a.run.record(parent/'protocol.json'),parent_release=a.run.record(parent/'release.json'),root_release=a.run.record(local/'release.json'),continuous32_choice_seals=continuous_seals,continuous_choice_execution=a.run.record(parent/'execution-choices.json'),continuous_choice_root_closure=a.run.record(parent/'root-closure-choices.json'),inputs=a.run.record(parent/'inputs.json'),no_trial_identity_or_mapped_outcome_used=True,candidate_labels=list(producer.CANDIDATE_LABELS),fixed_action=5,fixed_action_label=producer.CANDIDATE_LABELS[5],choices=choices,source_captures=capture_rows,catalog=a.run.record(local/'catalogs-LOCAL.npz'),wall_s=1.)
    a.run.write(local/'finite-choice-seals.json',seal);rows=[];(local/'posteriors-LOCAL').mkdir()
    for m in producer.MODELS:
        for i in range(32):
            raw=local/'posteriors-LOCAL'/f'{m}-{i:02d}.npz';raw.write_text('no NPZ deserialization allowed');rows.append(dict(model=m,index=i,posterior_archive=a.run.record(raw),choice_seal=a.run.record(local/'finite-choice-seals.json')))
    value=dict(completed=True,source_SHA256=a.run.sources(),parent_protocol=a.run.record(parent/'protocol.json'),finite_choices=a.run.record(local/'finite-choice-seals.json'),rows=rows,wall_s=2.,combined_owned_wall_s=3.)
    a.run.write(local/'evaluation-results-LOCAL.json',value)
    a.run.write(parent/'independent-terminal-review.json',dict(audit_passed=True,protocol=a.run.record(parent/'protocol.json'),source_SHA256={p:a.run.sha(tmp_path/p)for p in('parent-audit.py','parent-audit-test.py')}))
    def verify(v):
        if isinstance(v,dict):
            if set(v)=={'path','sha256'}:a.bound(v)
            else:
                for x in v.values():verify(x)
        elif isinstance(v,list):
            for x in v:verify(x)
    monkeypatch.setattr(a.c,'verify',verify);monkeypatch.setattr(a.np,'load',lambda *x,**k:pytest.fail('No arrays within metadata barrier'))
    assert len(a.metadata_barrier('p')[3]['rows'])==64
    value['rows'].pop();a.run.write(local/'evaluation-results-LOCAL.json',value)
    with pytest.raises(ValueError,match='All64'):a.metadata_barrier('p')
    value['rows'].append(rows[-1]if len(rows)==64 else dict(model='linear',index=31,posterior_archive=a.run.record(local/'posteriors-LOCAL/linear-31.npz'),choice_seal=a.run.record(local/'finite-choice-seals.json')));a.run.write(local/'evaluation-results-LOCAL.json',value)
    # Last archive byte tamper must fail BEFORE ANY NPZ, even with allmembers.
    (local/'posteriors-LOCAL/linear-31.npz').write_text('tampered')
    with pytest.raises(ValueError,match='Canonical everyposterior'):a.metadata_barrier('p')
