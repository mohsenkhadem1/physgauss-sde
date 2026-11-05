import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.gaussian_boundary_flow_v02 import isaac_prediction_cohort as cohort


def test_roster_uses_only_committed_identity_metadata(tmp_path):
    datasets=[]
    for name,count in [('neural',64),('orbit',12)]:
        datasets.append(dict(root=str(tmp_path/name),cases=[dict(episode_id=f'e{i:04d}',split='test',group_id=f'g{i}') for i in range(count)]))
    path=tmp_path/'seal.json';path.write_text(json.dumps(dict(datasets=datasets)))
    roster=cohort.roster_from_seal(path)
    assert len(roster)==76 and all(set(r)=={'case','dataset','group','episode_id','episode'} for r in roster)
    datasets[0]['cases'][0]['episode_id']='../private'
    path.write_text(json.dumps(dict(datasets=datasets)))
    with pytest.raises(ValueError,match='identity'):cohort.roster_from_seal(path)


def test_public_cache_hashes_only_five_prefix_inputs(tmp_path,monkeypatch):
    episode=tmp_path/'episode';episode.mkdir();frames=np.arange(5)*5
    for relative in ['reference.npz','observations/cameras.npz']:
        p=episode/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'public')
    for frame in frames:
        for folder,ext in [('rgb','png'),('masks','png'),('depth','npz')]:
            p=episode/'observations'/folder/f'{frame:05d}.{ext}';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'prefix')
    calls=[]
    def load(path,mode,**kwargs):
        calls.append(kwargs)
        return dict(cameras=dict(times_s=np.arange(5)*.25,frame_ids=frames),reference=dict(rest_points_m=np.zeros((455,3),np.float32)))
    monkeypatch.setattr(cohort,'load_public_observations',load)
    monkeypatch.setattr(cohort,'extract_features',lambda x:dict(features=np.zeros((5,119),np.float32)))
    arrays,seal=cohort.public_case_cache(dict(case='testcase',episode=str(episode)),tmp_path/'output','protocol')
    assert calls==[dict(size=(192,160),max_time_s=1.)] and arrays['features'].shape==(5,119)
    assert len(seal['public_input_sha256'])==17 and all('/truth/' not in p for p in seal['public_input_sha256'])
    # Changing an admitted public source cannot silently reuse old cached features.
    (episode/'reference.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Public inputs changed'):
        cohort.public_case_cache(dict(case='testcase',episode=str(episode)),tmp_path/'output','protocol')


def test_ridge_prefix_fills_future_with_zero_only():
    training_times=np.array([0,.25,.5,.75,1,2,2.25,2.5,2.75,3])
    prefix=np.ones((5,119),np.float32)
    out=cohort.align_prefix_features((10,119),training_times,1.,prefix,training_times[:5])
    assert np.array_equal(out[:5],prefix) and np.all(out[5:]==0)
    with pytest.raises(ValueError,match='Future timestamp'):
        cohort.align_prefix_features((10,119),training_times,1.,prefix,np.array([0,.25,.5,.75,2]))


def test_barrier_requires_every_model_case_and_retains_failures(tmp_path,monkeypatch):
    cases=[dict(case=f'case{i:02d}') for i in range(76)]
    protocol=dict(cases=cases);(tmp_path/'protocol.json').write_text(json.dumps(protocol));ph=cohort.sha(tmp_path/'protocol.json')
    monkeypatch.setattr(cohort,'verify_protocol',lambda path:protocol)
    for row in cases:
        for kind in cohort.KINDS:
            path=tmp_path/row['case']/kind/'status.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(case=row['case'],model=kind,protocol_sha256=ph,state='failed',error_type='DeclaredFailure',error='preserved')))
    missing=tmp_path/'case75'/'ridge'/'status.json';saved=missing.read_text();missing.unlink()
    with pytest.raises(ValueError,match='1/380 missing'):cohort.barrier(tmp_path)
    assert not(tmp_path/'barrier.json').exists()
    missing.write_text(saved);b=cohort.barrier(tmp_path)
    assert b['terminal']==380 and b['failed']==380 and b['complete']==0
    # Deliberately all failed is a valid completed accounting barrier, never success.
    assert cohort.barrier(tmp_path)==b


def test_new_and_archived_inference_select_batch_dimension():
    class Fake:
        def sample(self,features,samples,steps,generator):return self.infer(dict(features=features),samples,generator)
        def infer(self,batch,samples,generator):
            assert batch['features'].shape==(1,5,119)
            return dict(physical=torch.zeros(1,samples,25),effective_physical=torch.ones(1,samples,25),boundary_type=torch.zeros(1,samples,dtype=torch.long),force_type=torch.ones(1,samples,dtype=torch.long))
    for kind in ['flow','mdn','25d','chart']:
        out=cohort.infer_particles(kind,Fake(),dict(features=np.zeros((5,119),np.float32)),dict(samples=3),'cpu',42)
        assert out['physical'].shape==(3,25) and out['force_type'].shape==(3,)


@pytest.mark.parametrize('mutation',['forecast','checkpoint'])
def test_terminal_seal_rejects_changed_forecast_or_checkpoint(tmp_path,mutation):
    case=tmp_path/'case';run=case/'flow';run.mkdir(parents=True)
    public=case/'public-prefix.npz';public.write_bytes(b'bound public cache')
    (case/'public-prefix-seal.json').write_text(json.dumps(dict(case='case',protocol_sha256='protocol',cache_sha256=cohort.sha(public))))
    forecast=run/'predictions.npz';forecast.write_bytes(b'bound forecast')
    protocol=dict(models=dict(flow=dict(checkpoint_sha256='weights',samples=128)),teacher_sha256='teacher')
    seal=dict(case='case',model='flow',protocol_sha256='protocol',checkpoint_sha256='weights',teacher_sha256='teacher',samples=128,prediction_sha256=cohort.sha(forecast),public_prefix_seal_sha256=cohort.sha(case/'public-prefix-seal.json'))
    seal_path=run/'prediction-seal.json';seal_path.write_text(json.dumps(seal))
    status=dict(case='case',model='flow',protocol_sha256='protocol',state='complete',seal_sha256=cohort.sha(seal_path))
    status_path=run/'status.json';status_path.write_text(json.dumps(status))
    assert cohort.verify_terminal(case,'flow',protocol,'protocol')['state']=='complete'
    if mutation=='forecast':forecast.write_bytes(b'changed forecast')
    else:
        seal['checkpoint_sha256']='different weights';seal_path.write_text(json.dumps(seal));status['seal_sha256']=cohort.sha(seal_path);status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError,match='seal changed|identity binding'):
        cohort.verify_terminal(case,'flow',protocol,'protocol')


def test_archived_teacher_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(cohort,'load_observation_posterior',lambda *a:(None,dict(operator_checkpoint_sha256='wrong')))
    with pytest.raises(ValueError,match='teacher mismatch'):
        cohort.load_predictor('chart',dict(checkpoint='unopened',teacher_sha256='right'),'cpu')


@pytest.mark.parametrize('changed',['samples','steps','clock','nodes','prefix','teacher'])
def test_declared_budgets_are_enforced(changed):
    recipe=dict(evaluation=dict(samples=128,flow_steps=32,seed=61033),teacher_sha256='teacher')
    protocol=dict(models={k:dict(samples=1 if k=='ridge' else 128,teacher_sha256='teacher') for k in cohort.KINDS},
                  forecast=dict(steps=32,seed=61033,times_s=(np.arange(61)/20).tolist(),all_nodes=455,primary_window_s=[2.,3.]),
                  input=dict(frames=5,times_s=[0,.25,.5,.75,1.],feature_shape=[5,119],size=[192,160]))
    cohort.validate_budget(protocol,recipe)
    if changed=='samples':protocol['models']['flow']['samples']=32
    elif changed=='steps':protocol['forecast']['steps']=0
    elif changed=='clock':protocol['forecast']['times_s']=[2.,2.5,3.]
    elif changed=='nodes':protocol['forecast']['all_nodes']=64
    elif changed=='prefix':protocol['input']['frames']=10
    else:protocol['models']['chart']['teacher_sha256']='different'
    with pytest.raises(ValueError):cohort.validate_budget(protocol,recipe)
