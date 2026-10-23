import pytest
from experiments.run_campaign_c38_full_evaluation import validate_post_seal


def fixture():
    initial=dict(path='initial.json',sha256='a'*64);choices=dict(path='choices.json',sha256='b'*64)
    ids={f'{i:032x}' for i in range(64)}
    rows=[dict(object_id=t,force_Fz_N=f,result=dict(path='result.json',sha256='c'*64)) for t in ids for f in (-2.,2.)]
    seal=dict(completed=True,count=128,model_executions=256,private_target_or_latent_access=False,
              initial_prediction_seal=initial,choices_seal=choices,objects=rows)
    return seal,initial,choices,ids


def test_postseal_requires_all_counterfactual_predictions_before_truth():
    s,i,c,ids=fixture();assert len(validate_post_seal(s,i,c,ids))==128
    s['objects']=s['objects'][:-1]
    with pytest.raises(ValueError,match='All64x2'):validate_post_seal(s,i,c,ids)


def test_seals_must_bind_identical_initial_and_choices_and_no_latent_role():
    s,i,c,ids=fixture();s['choices_seal']=dict(path='other.json',sha256='b'*64)
    with pytest.raises(ValueError,match='bound'):validate_post_seal(s,i,c,ids)
    s,i,c,ids=fixture();s['private_target_or_latent_access']=True
    with pytest.raises(ValueError):validate_post_seal(s,i,c,ids)
    s,i,c,ids=fixture();s['objects'][0]['force_Fz_N']=0.
    with pytest.raises(ValueError):validate_post_seal(s,i,c,ids)


def test_full_fake_cohort_roundtrip_with_poisoned_unused_probe_member(tmp_path,monkeypatch):
    """Exercise actual codecs/files/metric joins without physical/noise draws."""
    import shutil
    from pathlib import Path
    import numpy as np
    from experiments import run_campaign_c38_full_evaluation as f
    from experiments import campaign_c38_controlled_observations as obs
    from experiments.campaign_c38_prediction_metrics import POINTS,POINT_IDS
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(f.os,'sched_getaffinity',lambda pid:{34,35})
    d=tmp_path/obs.DESIGN_PATH;d.parent.mkdir(parents=True);shutil.copyfile(Path(f.ROOT)/obs.DESIGN_PATH,d)
    for name in ('run_campaign_c38_full_evaluation.py','run_campaign_c38_prediction_evaluation.py','campaign_c38_prediction_metrics.py','campaign_c38_policy_metrics.py'):
        destination=tmp_path/'experiments'/name;destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(f.ROOT)/'experiments'/name,destination)
    design=obs.load_frozen_design(tmp_path);root=tmp_path;data=root/'fake';data.mkdir()
    write=lambda path,value:f.io.write_new(root/path,value)
    rec=lambda path:f.io.record(root,root/path)
    target=POINTS.copy();target[:,2]+=.001
    def prediction(token,error,force=None):
        q=target.copy();q[:,2]+=error
        intervals=np.stack([q[POINT_IDS]-.0001,q[POINT_IDS],q[POINT_IDS]+.0001],-1)
        row=dict(completed=True,canonical_mean_positions_m=q,category_probabilities=[.8,.2],
                 material_equal_tail_Pa=[90000.,100000.,110000.],
                 material_quantile_details=[dict(unit_bracket=dict(lower=np.log(1.2)/np.log(2),upper=.3)),{},dict(unit_bracket=dict(lower=.4,upper=np.log(110000/75000)/np.log(2)))],
                 canonical_position_equal_tail_m=intervals,canonical_position_interval95_numerical_enclosure_m=intervals[:,:,[0,2]],
                 canonical_coordinate_variance_m2=np.ones((63,3))*1e-9,canonical_variance_integration_resolution_estimate_m2=np.ones((63,3))*1e-15)
        r=dict(object_id=token,completed=True,numerical_controls_passed=True,methods={'adaptive':row})
        if force is not None:r.update(acquired_probe_force_Fz_N=force,acquired_coordinate_count=54,fresh_prior=True,initial_posterior_weights_reused=False)
        return r
    def save_models(token,tag,error,force=None):
        models={}
        for model in f.io.MODELS:
            folder=data/(token+'-'+tag+'-'+model);folder.mkdir()
            f.io.save_full_result(folder,prediction(token,error,force))
            files={n:f.io.record(root,folder/n) for n in ('posterior-arrays.npz','posterior-structure.json','numerics.json')}
            models[model]=dict(execution_completed=True,files=files)
        path='fake/'+token+'-'+tag+'.json'
        value=dict(completed=True,object_id=token,models=models)
        if force is not None:value['force_Fz_N']=force
        write(path,value);return rec(path)
    initial_rows=[];post_rows=[];choices=[];private_rows=[]
    for c in range(4):
        for repeat in range(16):
            token=f'{c*16+repeat:032x}'
            initial_rows.append(dict(object_id=token,result=save_models(token,'before',2e-6)))
            for force,error in ((-2.,1e-6),(2.,3e-6)):
                post_rows.append(dict(object_id=token,force_Fz_N=force,result=save_models(token,'down' if force<0 else 'up',error,force)))
            cp='fake/'+token+'-choice.json';write(cp,dict(completed=True))
            choices.append(dict(object_id=token,choices=dict(information=-2.,target_risk=-2.,fixed_down=-2.,fixed_up=2.,random=2.),result=rec(cp)))
            pub=obs.public_prefix_payload(token,target[POINT_IDS],design=design)
            pubpath=data/(token+'-prefix.npz')
            f.evaluation.codec.exclusive_npz(pubpath,f.evaluation.codec.encode_public(pub,design=design))
            truth=dict(object_id=np.asarray(token),private_configuration_index=np.asarray(c,dtype=np.int64),repeat_index=np.asarray(repeat,dtype=np.int64),
                       young_Pa=np.asarray(100000.),support_category=np.asarray('both_ends'),canonical_target_Fz_N=np.asarray(1.),
                       canonical_target_point_ids=np.arange(63),canonical_target_positions_m=target,
                       candidate_probe_observations_m=np.array([object()],dtype=object))
            privatepath=data/(token+'-private.npz');np.savez(privatepath,**truth)
            private_rows.append(dict(object_id=token,public_prefix=f.io.record(root,pubpath),private_future=f.io.record(root,privatepath)))
    write('fake/initial.json',dict(completed=True,count=64,model_executions=128,private_truth_access=False,objects=initial_rows))
    ir=rec('fake/initial.json')
    write('fake/choices.json',dict(completed=True,count=64,private_probe_outcomes_access=False,initial_prediction_seal=ir,choices=choices,model_choices={'linear_rest_tangent':choices}))
    cr=rec('fake/choices.json')
    write('fake/post.json',dict(completed=True,count=128,model_executions=256,private_target_or_latent_access=False,initial_prediction_seal=ir,choices_seal=cr,objects=post_rows))
    write('fake/production.json',dict(completed=True,records=private_rows))
    result=f.run(ir,cr,rec('fake/post.json'),rec('fake/production.json'),'evaluation',root=root)
    assert result['completed'] and result['actual_probe_observations_extracted_by_evaluation'] is False
    for model in f.io.MODELS:
        comparison=result['postprobe'][model]['canonical_RMSE_advantage_over_fixed_m']['information']
        assert comparison['fixed_down']['conditional_noise_bootstrap95']==[0.,0.]
        assert comparison['fixed_up']['mean_improvement']>0
    # In particular, np.load(allow_pickle=False) would raise if the poisoned
    # future measurement member had been materialized anywhere by evaluation.
