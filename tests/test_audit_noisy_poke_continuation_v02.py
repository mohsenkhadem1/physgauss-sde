import copy
from pathlib import Path
import numpy as np
import pytest
from experiments import audit_noisy_poke_continuation_v02 as a


def paths(monkeypatch,tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a.run,'ROOT',tmp_path);monkeypatch.setattr(a.parent,'ROOT',tmp_path)
    monkeypatch.setattr(a.run,'LOCAL',new);monkeypatch.setattr(a.parent,'LOCAL',old)
    def record(path):return dict(path=str(path.relative_to(tmp_path)),sha256=a.run.sha(path))
    monkeypatch.setattr(a.run,'record',record);monkeypatch.setattr(a.parent,'record',record)
    for directory,names in((old,('protocol.json','release.json','execution-choices.json','execution-checkpoint.json')),(new,('protocol.json','release.json','execution-remaining.json'))):
        for name in names:a.run.write(directory/name,{})
    context=dict(execution_protocol=record(new/'protocol.json'),execution_release=record(new/'release.json'),execution_source_SHA256={'new':'execution'})
    monkeypatch.setattr(a.run,'execution_context',lambda:copy.deepcopy(context));return old,new,context


def test_explicit_old0_new31_context_without_rebranding(monkeypatch,tmp_path):
    old,new,context=paths(monkeypatch,tmp_path)
    a.check_execution_context({'protocol_SHA256':'old'},0);a.check_execution_context(dict(context,protocol_SHA256='old'),1)
    assert a.run.evaluation_directory(0)==old/'trial-00/evaluation'and a.run.evaluation_directory(31)==new/'trial-31/evaluation'
    with pytest.raises(ValueError,match='Original trial0'):a.check_execution_context(dict(context),0)
    for key in context:
        bad=copy.deepcopy(context);bad.pop(key)
        with pytest.raises(ValueError,match='NEW execution'):a.check_execution_context(bad,1)


def test_unclosed_new_phase_stops_before_any_array_or_result_reads(monkeypatch,tmp_path):
    paths(monkeypatch,tmp_path)
    def fail(digest):raise ValueError('newremaining rootclosure missing')
    monkeypatch.setattr(a.run,'metadata_barrier',fail);monkeypatch.setattr(a.np,'load',lambda *x,**k:pytest.fail('No arrays before BOTH roots'))
    with pytest.raises(ValueError,match='rootclosure'):a.metadata_barrier('parent')


def metadata_fixture(monkeypatch,tmp_path):
    old,new,context=paths(monkeypatch,tmp_path);pins={a.SOURCE:'audit',a.TEST:'test'}
    monkeypatch.setattr(a,'sources',lambda:pins.copy());monkeypatch.setattr(a.run,'sources',lambda:pins.copy());monkeypatch.setattr(a.parent,'sources',lambda:{'parent':'original'})
    a.run.write(new/'protocol.json',dict(source_SHA256=pins))
    # Rebind context after the actual newprotocol bytes change.
    context['execution_protocol']=a.run.record(new/'protocol.json')
    p=dict(source_SHA256={'parent':'original'},origin={},trials=a.parent.design.trials());phases={name:dict(wall_s=1,workers=[])for name in('choices','checkpoint','remaining')};entries=[];seals=[]
    for index in range(32):
        prefix=old/f'trial-{index:02d}/prefix/result-LOCAL.json';a.run.write(prefix,{})
        seal=old/f'trial-{index:02d}/choice-seal.json';a.run.write(seal,dict(prefix_result=a.run.record(prefix)));seals.append(dict(index=index,seal=a.run.record(seal)))
        directory=a.run.evaluation_directory(index);raw=directory/'snapshot.npz';raw.parent.mkdir(parents=True,exist_ok=True);raw.write_text('Hash bytes only; never open as NPZ')
        posts=[]
        for action in range(8):
            post=directory/f'action-{action}/result-LOCAL.json';a.run.write(post,dict((context if index else{}),index=index,action=action,protocol_SHA256='parent',prefix_plus_ONE_reset_poke=True,snapshot=a.run.record(raw)));posts.append(dict(action=action,record=a.run.record(post)))
        result=directory/'result-LOCAL.json';a.run.write(result,dict((context if index else{}),index=index,object_index=index%8,replicate=index//8,posteriors=posts))
        entries.append(dict(index=index,qualified=True,record=a.run.record(result)))
    def verify(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:a.bound(value)
            else:
                for v in value.values():verify(v)
        elif isinstance(value,list):
            for v in value:verify(v)
    monkeypatch.setattr(a.parent,'verify',verify);monkeypatch.setattr(a.run,'metadata_barrier',lambda digest:(p,phases,seals,{},entries))
    return p,phases,seals,entries,context


def test_all32_and_last_snapshot_hash_before_npz_with_mixed_source_pins(monkeypatch,tmp_path):
    p,phases,seals,entries,context=metadata_fixture(monkeypatch,tmp_path);monkeypatch.setattr(a.np,'load',lambda *x,**k:pytest.fail('Metadata barrier cannot openNPZ'))
    assert len(a.metadata_barrier('parent')[-1])==32
    raw=a.run.evaluation_directory(31)/'snapshot.npz';raw.write_text('tampered')
    with pytest.raises(ValueError,match='byte commitment'):a.metadata_barrier('parent')


def test_missing_trial0_wrong_namespace_or_newsource_claim_on_old_are_rejected(monkeypatch,tmp_path):
    p,phases,seals,entries,context=metadata_fixture(monkeypatch,tmp_path)
    first=entries.pop(0)
    with pytest.raises(ValueError,match='All32 mixed'):a.metadata_barrier('parent')
    entries.insert(0,first);wrong=a.run.LOCAL/'trial-00/evaluation/result-LOCAL.json';a.run.write(wrong,dict(posteriors=[]));original=entries[0]['record'];entries[0]['record']=a.run.record(wrong)
    with pytest.raises(ValueError,match='canonical path'):a.metadata_barrier('parent')
    entries[0]['record']=original;value=a.run.read(a.bound(original));value.update(context);a.run.write(a.bound(original),value);entries[0]['record']=a.run.record(a.run.evaluation_directory(0)/'result-LOCAL.json')
    with pytest.raises(ValueError,match='Original trial0'):a.metadata_barrier('parent')


def test_new_procedure_precommit_and_parent_attestation_are_distinct(monkeypatch,tmp_path):
    p,phases,seals,entries,context=metadata_fixture(monkeypatch,tmp_path)
    value=a.run.read(a.run.LOCAL/'protocol.json');value['source_SHA256'][a.TEST]='wrong';a.run.write(a.run.LOCAL/'protocol.json',value)
    with pytest.raises(ValueError,match='prospectively pinned'):a.metadata_barrier('parent')


def test_original_numerical_helper_is_reused_and_relaxation_still_fails(monkeypatch,tmp_path):
    from tests.test_audit_noisy_poke_benchmark_v01 import posterior_fixture
    assert a.check_posterior is a.legacy.check_posterior and a.snapshot is a.legacy.snapshot
    value,observations,p=posterior_fixture(monkeypatch,tmp_path)
    assert a.check_posterior(value,tmp_path,observations,p)['density_qualified']
    path=tmp_path/'category-0-LOCAL.json';v=a.parent.read(path);v['diagnostics']['gates']['CDF']=.01;a.parent.write(path,v);value['categories'][0]['record']=a.parent.record(path)
    with pytest.raises(ValueError,match='gates'):a.check_posterior(value,tmp_path,observations,p)


def test_full32_mixed_source_replay_keeps_old0_and_all256_reset_joints(monkeypatch,tmp_path):
    from tests.test_layout_grouped_allactions_v01 import rows as utility_rows
    old,new,context=paths(monkeypatch,tmp_path);monkeypatch.setattr(a,'guard',lambda start:None);monkeypatch.setattr(a.run,'sources',lambda:{'new':'execution'})
    monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(p),sha256=a.run.sha(p)));monkeypatch.setattr(a.run,'continuation_gate',lambda d:None);monkeypatch.setattr(a.run.base,'load_models',lambda m:[None]*4)
    monkeypatch.setattr(a,'check_posterior',lambda *args,**kwargs:dict(density_qualified=True,references=4,banks=5))
    for name in('protocol.json','release.json','execution-choices.json','execution-checkpoint.json','execution-remaining.json'):a.run.write(a.run.LOCAL/name,{})
    for name in(a.SOURCE,a.TEST):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('toy source commitment')
    names=[s['id']for s in a.run.design.physics.measurement_specs()];p=dict(objects=a.run.design.objects(),trials=a.run.design.trials(),origin=dict(models=[{}]*4),source_SHA256={'toy':'source'});inputs=dict(prefixes=[]);seals=[];entries=[];evalrows=[];noise=np.arange(256.).reshape(4,8,8)*1e-8
    np.savez_compressed(a.run.LOCAL/'poke.npz',poke_noise=noise);inputs['poke_noise']=a.run.record(a.run.LOCAL/'poke.npz')
    def actual(i):
        values=np.arange(60.)*1e-6+i*1e-7;return dict(values=values,warm=values[12:]+1e-6,oracle=values[12:]+2e-6,parameters=np.array([p['objects'][i]['E_Pa'],p['objects'][i]['K_N_m']]),captures=[])
    monkeypatch.setattr(a,'actual_truth',lambda p,i,models:actual(i))
    def snap(path,mean,sd):
        q=np.tile([.05,.95],(2,1));np.savez_compressed(path,mean_m=mean,SD_m=np.ones(48)*sd,unit_mean=np.ones(2)*.5,layout_probability=np.ones(4)*.25,material_unit_quantiles=q,material_quantiles=np.exp(np.log(a.run.design.BOUNDS[:,0,None])+a.run.design.LOG_WIDTH[:,None]*q),material_unit_CDF=np.tile(np.linspace(0,1,33),(2,1)));return a.run.record(path)
    for trial in p['trials']:
        index,i,r=trial['index'],trial['object_index'],trial['replicate'];ac=actual(i);truth=ac['values'][12:];parameters=ac['parameters'];directory=a.parent.LOCAL/f'trial-{index:02d}';pd=directory/'prefix';ed=a.run.evaluation_directory(index);pd.mkdir(parents=True);ed.mkdir(parents=True);prefix_noise=np.ones(4)*r*1e-7;observed=ac['values'][:4]+prefix_noise
        inp=a.run.LOCAL/f'input-{index}.npz';np.savez_compressed(inp,observed_prefix=observed,prefix_noise=prefix_noise);inputs['prefixes'].append(dict(index=index,record=a.run.record(inp),observation_fingerprint=a.run.design.fingerprint(observed)))
        obs=list(zip(names[:4],observed[:,None]));jobs=utility_rows();ranking=a.run.base.old.classify(jobs,dict(outcome_relative=1e-4,latent_relative=1e-3));prefix=dict(index=index,qualified=True,observations_fingerprint=a.run.observations_fingerprint(obs),utility_jobs=jobs,ranking=ranking,snapshot=snap(pd/'posterior-LOCAL.npz',truth+1e-5,2e-5));a.run.write(pd/'result-LOCAL.json',prefix)
        seal=dict(qualified=True,adaptive_action=ranking['unique_best_action'],fixed_action=5,prefix_result=a.run.record(pd/'result-LOCAL.json'));a.run.write(directory/'choice-seal.json',seal);seals.append(dict(index=index,seal=a.run.record(directory/'choice-seal.json')))
        truthpath=ed/'truth-LOCAL.npz';np.savez_compressed(truthpath,NH_target_m=truth,n12_linear_target_m=ac['warm'],n6_true_parameter_oracle_m=ac['oracle'],clean_candidate_m=ac['values'][4:12]);before=a.metrics(a.snapshot(pd/'posterior-LOCAL.npz'),truth,parameters);posts=[];after=[]
        for action in range(8):
            ad=ed/f'action-{action}';ad.mkdir();joint=obs+[(names[4+action],np.array([ac['values'][4+action]+noise[r,i,action]]))];np.savez_compressed(ad/'observations-LOCAL.npz',labels=np.array([x[0]for x in joint]),values=np.array([x[1]for x in joint]));snapshot=snap(ad/'posterior-LOCAL.npz',truth+(action-3.5)*1e-6,1e-5);result=dict(**(context if index else{}),index=index,action=action,qualified=True,protocol_SHA256='toy',choice_seal=seals[-1]['seal'],prefix_plus_ONE_reset_poke=True,observations=a.run.record(ad/'observations-LOCAL.npz'),observations_fingerprint=a.run.observations_fingerprint(joint),snapshot=snapshot);a.run.write(ad/'result-LOCAL.json',result);m=a.metrics(a.snapshot(ad/'posterior-LOCAL.npz'),truth,parameters);after.append(m);posts.append(dict(action=action,qualified=True,record=a.run.record(ad/'result-LOCAL.json'),metrics=m))
        policy=a.run.design.compare_policies(before['field'],[m['field']for m in after],seal['adaptive_action'],5);material=a.run.design.compare_material(before['material'],[m['material']for m in after],seal['adaptive_action'],5)
        value=dict(**(context if index else{}),index=index,object_index=i,replicate=r,qualified=True,protocol_SHA256='toy',choice_seal=seals[-1]['seal'],prefix_result=seal['prefix_result'],source_captures=[],truth=a.run.record(truthpath),oracle_metrics=dict(n12_linear_vs_n12_NH_MSE_m2=float(np.mean((ac['warm']-truth)**2)),n6_modal_vs_n12_linear_MSE_m2=float(np.mean((ac['oracle']-ac['warm'])**2)),n6_modal_vs_n12_NH_MSE_m2=float(np.mean((ac['oracle']-truth)**2))),before=before,posteriors=posts,policies=policy,material_policies=material);a.run.write(ed/'result-LOCAL.json',value);entries.append(dict(index=index,qualified=True,record=a.run.record(ed/'result-LOCAL.json')));evalrows.append(value)
    phases={name:{}for name in('choices','checkpoint','remaining')};monkeypatch.setattr(a,'metadata_barrier',lambda digest:(p,phases,seals,inputs,entries))
    phase_order=[0]+[i for slot in range(3)for i in range(1,32)if i%3==slot];summary=dict(**context,parent_protocol=a.run.record(a.parent.LOCAL/'protocol.json'),protocol_SHA256='toy',summary=a.run.design.aggregate([evalrows[i]for i in phase_order]),all32_trial_records=[entries[i]for i in phase_order],phases=a.phase_records());a.run.write(a.run.LOCAL/'summary-LOCAL.json',summary)
    result=a.audit('toy');assert result['audit_passed']and result['completed_category_references']==1152 and result['qualified_joint_posteriors']==256 and result['qualified_prefixes']==32 and result['completed_utility_jobs']==1536 and not result['summary']['comparison_blocked']

    assert result['parent_protocol']==a.run.record(old/'protocol.json') and result['protocol']==a.run.record(new/'protocol.json')
    assert result['reused_trial_indices']==[0] and result['new_trial_indices']==list(range(1,32))
    assert result['parent_source_SHA256']==p['source_SHA256'] and result['execution_source_SHA256']=={'new':'execution'}
    assert result['all32_trial_records'][0]==a.run.record(old/'trial-00/evaluation/result-LOCAL.json')
    assert all(result['all32_trial_records'][i]==a.run.record(new/f'trial-{i:02d}/evaluation/result-LOCAL.json') for i in range(1,32))
