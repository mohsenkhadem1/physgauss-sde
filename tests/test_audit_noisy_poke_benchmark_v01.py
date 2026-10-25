import copy
from pathlib import Path
import numpy as np
import pytest
from experiments import audit_noisy_poke_benchmark_v01 as a


def reference_value(context):
    check=dict(passed=True,log_evidence_difference=0.,maximum_CDF_difference=0.,maximum_unit_quantile_difference=0.,maximum_unit_mean_difference=0.,maximum_predictive_mean_difference_m=0.,maximum_predictive_SD_difference_m=0.)
    diagnostics=dict(schema='conditional-amplitude-v02',gates=a.numerical.GATES,independent_adaptive_per_posterior_required=True,levels=[dict(outer=o,inner=i,status='computed',log_evidence=0.)for o,i in((512,128),(512,256),(512,512),(128,512),(256,512))],successive={axis:[dict(coarse=c,fine=f,**check)for c,f in((128,256),(256,512))]for axis in('inner','outer')},independent_adaptive=dict(passed=True,epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6,log_evidence_difference=0.,maximum_CDF_error=0.,adaptive_refinement_log_evidence_difference=0.,adaptive_refinement_maximum_CDF_difference=0.,maximum_QUADPACK_relative_error_estimates={'1e-06':0.,'1e-08':0.}))
    return dict(qualified=True,strict=dict(passed=True),diagnostics=diagnostics,reference_context=context)


def posterior_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(a.run,'ROOT',tmp_path);monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(p),sha256=a.run.sha(p)))
    observations=[('prefix',np.array([2e-5]))];p=dict(sigma_m=20e-6,origin=dict(models=[dict(model_fingerprint=str(k)*64)for k in range(4)]));value=dict(qualified=True,categories=[],decision_levels=[])
    for k in range(4):
        context=dict(observations_fingerprint=a.run.observations_fingerprint(observations),reference_plan=a.run.CONTEXT_PLAN,target_labels=[s['id']for s in a.run.design.physics.measurement_specs()[12:]],sigma_m=20e-6,model_fingerprint=p['origin']['models'][k]['model_fingerprint']);path=tmp_path/f'category-{k}-LOCAL.json';a.run.write(path,reference_value(context));value['categories'].append(dict(layout=k,qualified=True,record=a.run.record(path)))
    check=dict(passed=True,per_category=[dict(passed=True,log_evidence=0.,mean_m=0.,SD_m=0.)for _ in range(4)],mixture_mean_error_m=0.,mixture_SD_error_m=0.,layout_TV=0.)
    for o,n in a.run.base.DECISION_LEVELS:
        path=tmp_path/f'decision-{o}-{n}-LOCAL.json';a.run.write(path,dict(checks=check));value['decision_levels'].append(dict(outer=o,amplitude=n,record=a.run.record(path),checks=check))
    value['mixture_interval_checks']=dict(passed=True,levels=[[512,512],[512,256],[256,512]],weights_from_continuous_category_logZ=True,CDF_gate=.005,unit_quantile_gate=.002,maximum_CDF_difference=0.,maximum_unit_quantile_difference=0.);path=tmp_path/'posterior-LOCAL.npz';path.write_bytes(b'Not deserialized by scalar numerical parser');value['snapshot']=a.run.record(path)
    return value,observations,p


def test_actual_model_observation_context_and_all_original_numeric_gates(monkeypatch,tmp_path):
    value,obs,p=posterior_fixture(monkeypatch,tmp_path);check=a.check_posterior(value,tmp_path,obs,p);assert check==dict(density_qualified=True,references=4,banks=5)
    path=tmp_path/'category-0-LOCAL.json';v=a.run.read(path);v['reference_context']['model_fingerprint']='wrong-but-consistent';a.run.write(path,v);value['categories'][0]['record']=a.run.record(path)
    with pytest.raises(ValueError,match='fingerprint'):a.check_posterior(value,tmp_path,obs,p)


def test_relaxed_gate_or_missing_adaptive_flag_refused(monkeypatch,tmp_path):
    value,obs,p=posterior_fixture(monkeypatch,tmp_path);path=tmp_path/'category-0-LOCAL.json';v=a.run.read(path);v['diagnostics']['gates']['CDF']=.01;a.run.write(path,v);value['categories'][0]['record']=a.run.record(path)
    with pytest.raises(ValueError,match='gates'):a.check_posterior(value,tmp_path,obs,p)


def test_wrong_observation_or_mixture_refinement_flag_refused(monkeypatch,tmp_path):
    value,obs,p=posterior_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='fingerprint'):a.check_posterior(value,tmp_path,[('prefix',np.array([3e-5]))],p)
    value['mixture_interval_checks']['maximum_CDF_difference']=.006
    with pytest.raises(ValueError,match='refinement scalar'):a.check_posterior(value,tmp_path,obs,p)


def test_posterior_snapshot_material_physical_unit_conversion_and_sd(tmp_path):
    path=tmp_path/'x.npz';q=np.tile([.05,.95],(2,1));values=dict(mean_m=np.zeros(48),SD_m=np.ones(48)*1e-5,unit_mean=np.ones(2)*.5,layout_probability=np.ones(4)*.25,material_unit_quantiles=q,material_quantiles=np.exp(np.log(a.run.design.BOUNDS[:,0,None])+a.run.design.LOG_WIDTH[:,None]*q),material_unit_CDF=np.tile(np.linspace(0,1,33),(2,1)));np.savez_compressed(path,**values);s=a.snapshot(path);assert s['mean_m'].shape==(48,)
    values['material_quantiles']=values['material_quantiles']+1.;np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='quantiles derived'):a.snapshot(path)


def test_independent_metrics_and_uniformrandom_meanloss_not_ensemble():
    truth=np.zeros(48);parameters=np.array([2500.,np.sqrt(150.)]);q=np.array([[2200.,2800.],[10.,14.]])
    def snap(sign,sd):return dict(mean_m=np.ones(48)*sign,SD_m=np.ones(48)*sd,unit_mean=np.ones(2)*.5,layout_probability=np.ones(4)*.25,material_quantiles=q)
    before=a.metrics(snap(1,2),truth,parameters);after=[a.metrics(snap(1 if i%2 else-1,i+1),truth,parameters)for i in range(8)]
    assert a.equal(before['field'],a.run.design.prediction_metrics(snap(1,2)['mean_m'],snap(1,2)['SD_m'],truth))
    assert a.equal(before['material'],a.run.design.material_metrics(dict(physical_quantiles=q),snap(1,2)['unit_mean'],parameters))
    policies=a.policies(before['field'],[m['field']for m in after],2,2);assert policies['uniform_random']['MSE_m2']==1. and policies['adaptive']==policies['fixed']
    assert a.equal(policies,a.run.design.compare_policies(before['field'],[m['field']for m in after],2,2))
    assert a.equal(a.material_policies(before['material'],[m['material']for m in after],2,2),a.run.design.compare_material(before['material'],[m['material']for m in after],2,2))


def test_all32_exactidentities_and_equalobjects_fournoise_means():
    rows=[dict(**v,qualified=True,policies=dict(paired_adaptive_minus_fixed_MSE_m2=float(v['object_index']),paired_adaptive_minus_random_MSE_m2=-1.))for v in a.run.design.trials()]
    assert a.aggregate(rows)==a.run.design.aggregate(rows)and a.aggregate(rows)['equal_object_mean_adaptive_minus_fixed_MSE_m2']==3.5
    rows[0]['qualified']=False;assert a.aggregate(rows)==a.run.design.aggregate(rows)and a.aggregate(rows)['comparison_blocked']
    rows[0]['object_index']=7
    with pytest.raises(ValueError,match='identities'):a.aggregate(rows)


def test_allclosedphases_and_32seals_before_target_score_deserialization(monkeypatch,tmp_path):
    monkeypatch.setattr(a.run,'ROOT',tmp_path);monkeypatch.setattr(a.run,'LOCAL',tmp_path/'run');monkeypatch.setattr(a.run,'record',lambda p:dict(path=str(p),sha256=a.run.sha(p)));a.run.write(a.run.LOCAL/'protocol.json',{})
    p=dict(source_SHA256={a.SOURCE:'s',a.TEST:'t'},resources=dict(global_wall_s=1000,aggregate_RSS_bytes=1000,worker_RSS_bytes=1000,owned_VRAM_bytes=1000));monkeypatch.setattr(a.run,'validate',lambda d:p)
    phases={};seen=[]
    for name in('choices','checkpoint','remaining'):
        ids=[]if name=='choices'else([0]if name=='checkpoint'else list(range(1,32)));entries=[]
        for i in ids:
            target=a.run.LOCAL/f'trial-{i:02d}'/'evaluation'/'result-LOCAL.json';a.run.write(target,dict(SCORE_MUST_NOT_BE_READ=True));entries.append(dict(index=i,record=a.run.record(target)))
        wp=tmp_path/f'{name}-worker.json';a.run.write(wp,dict(trials=entries));phases[name]=dict(wall_s=1.,workers=[]if name=='choices'else[dict(result=a.run.record(wp))])
    monkeypatch.setattr(a.run,'completed_phase',lambda name,d:seen.append(name)or phases[name]);monkeypatch.setattr(a.run,'all_seals_gate',lambda d:seen.append('ALL32SEALED')or[dict(index=i)for i in range(32)]);monkeypatch.setattr(a.run,'inputs_gate',lambda d:{})
    real=a.run.read
    def read(path):
        if path.name=='result-LOCAL.json':pytest.fail('Target-derived score read before barrier returns')
        return real(path)
    monkeypatch.setattr(a.run,'read',read);monkeypatch.setattr(a.np,'load',lambda *a,**k:pytest.fail('No NPZ before whole barrier'))
    p,d,ph,seals,inputs,entries=a.metadata_barrier();assert len(entries)==32 and seen==['choices','checkpoint','remaining','ALL32SEALED']
    phases['remaining']['workers']=[]
    with pytest.raises(ValueError,match='32 evaluation'):a.metadata_barrier()


def test_full32_toy_metric_replay_includes_all256_joints_and_unsorted_phase_records(monkeypatch,tmp_path):
    from tests.test_layout_grouped_allactions_v01 import rows as utility_rows
    monkeypatch.setattr(a.run,'ROOT',tmp_path);monkeypatch.setattr(a.run,'LOCAL',tmp_path/'run');monkeypatch.setattr(a,'guard',lambda start:None)
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
        index,i,r=trial['index'],trial['object_index'],trial['replicate'];ac=actual(i);truth=ac['values'][12:];parameters=ac['parameters'];directory=a.run.LOCAL/f'trial-{index:02d}';pd=directory/'prefix';ed=directory/'evaluation';pd.mkdir(parents=True);ed.mkdir();prefix_noise=np.ones(4)*r*1e-7;observed=ac['values'][:4]+prefix_noise
        inp=a.run.LOCAL/f'input-{index}.npz';np.savez_compressed(inp,observed_prefix=observed,prefix_noise=prefix_noise);inputs['prefixes'].append(dict(index=index,record=a.run.record(inp),observation_fingerprint=a.run.design.fingerprint(observed)))
        obs=list(zip(names[:4],observed[:,None]));jobs=utility_rows();ranking=a.run.base.old.classify(jobs,dict(outcome_relative=1e-4,latent_relative=1e-3));prefix=dict(index=index,qualified=True,observations_fingerprint=a.run.observations_fingerprint(obs),utility_jobs=jobs,ranking=ranking,snapshot=snap(pd/'posterior-LOCAL.npz',truth+1e-5,2e-5));a.run.write(pd/'result-LOCAL.json',prefix)
        seal=dict(qualified=True,adaptive_action=ranking['unique_best_action'],fixed_action=5,prefix_result=a.run.record(pd/'result-LOCAL.json'));a.run.write(directory/'choice-seal.json',seal);seals.append(dict(index=index,seal=a.run.record(directory/'choice-seal.json')))
        truthpath=ed/'truth-LOCAL.npz';np.savez_compressed(truthpath,NH_target_m=truth,n12_linear_target_m=ac['warm'],n6_true_parameter_oracle_m=ac['oracle'],clean_candidate_m=ac['values'][4:12]);before=a.metrics(a.snapshot(pd/'posterior-LOCAL.npz'),truth,parameters);posts=[];after=[]
        for action in range(8):
            ad=ed/f'action-{action}';ad.mkdir();joint=obs+[(names[4+action],np.array([ac['values'][4+action]+noise[r,i,action]]))];np.savez_compressed(ad/'observations-LOCAL.npz',labels=np.array([x[0]for x in joint]),values=np.array([x[1]for x in joint]));snapshot=snap(ad/'posterior-LOCAL.npz',truth+(action-3.5)*1e-6,1e-5);result=dict(index=index,action=action,qualified=True,protocol_SHA256='toy',choice_seal=seals[-1]['seal'],prefix_plus_ONE_reset_poke=True,observations=a.run.record(ad/'observations-LOCAL.npz'),observations_fingerprint=a.run.observations_fingerprint(joint),snapshot=snapshot);a.run.write(ad/'result-LOCAL.json',result);m=a.metrics(a.snapshot(ad/'posterior-LOCAL.npz'),truth,parameters);after.append(m);posts.append(dict(action=action,qualified=True,record=a.run.record(ad/'result-LOCAL.json'),metrics=m))
        policy=a.run.design.compare_policies(before['field'],[m['field']for m in after],seal['adaptive_action'],5);material=a.run.design.compare_material(before['material'],[m['material']for m in after],seal['adaptive_action'],5)
        value=dict(index=index,object_index=i,replicate=r,qualified=True,protocol_SHA256='toy',choice_seal=seals[-1]['seal'],prefix_result=seal['prefix_result'],source_captures=[],truth=a.run.record(truthpath),oracle_metrics=dict(n12_linear_vs_n12_NH_MSE_m2=float(np.mean((ac['warm']-truth)**2)),n6_modal_vs_n12_linear_MSE_m2=float(np.mean((ac['oracle']-ac['warm'])**2)),n6_modal_vs_n12_NH_MSE_m2=float(np.mean((ac['oracle']-truth)**2))),before=before,posteriors=posts,policies=policy,material_policies=material);a.run.write(ed/'result-LOCAL.json',value);entries.append(dict(index=index,qualified=True,record=a.run.record(ed/'result-LOCAL.json')));evalrows.append(value)
    phases={name:{}for name in('choices','checkpoint','remaining')};monkeypatch.setattr(a,'metadata_barrier',lambda:(p,'toy',phases,seals,inputs,entries))
    phase_order=[0]+[i for slot in range(3)for i in range(1,32)if i%3==slot];summary=dict(protocol_SHA256='toy',summary=a.run.design.aggregate([evalrows[i]for i in phase_order]),all32_trial_records=[entries[i]for i in phase_order],phases=[a.run.record(a.run.LOCAL/f'execution-{name}.json')for name in phases]);a.run.write(a.run.LOCAL/'summary-LOCAL.json',summary)
    result=a.audit();assert result['audit_passed']and result['completed_category_references']==1152 and result['qualified_joint_posteriors']==256 and result['qualified_prefixes']==32 and result['completed_utility_jobs']==1536 and not result['summary']['comparison_blocked']
