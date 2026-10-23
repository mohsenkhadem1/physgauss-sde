"""Synthetic64 identities/measurements only; no actual payload or entropy."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from experiments import release_campaign_c38_acquired_probes as m
from experiments.campaign_c38_public_probe_codec import encode_probe,decode_probe
from test_campaign_c38_public_prefix_inference_v02 import design,payload
from test_campaign_c38_postprobe_inference import probe


def put(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def fixture(root,monkeypatch):
 monkeypatch.setattr(m,'sources',lambda:['source.py']);put(root,'source.py',{})
 initial=[];choices=[];public=[];private=[]
 for i in range(64):
  token=format(i+1,'032x');p=payload(token);prefix=root/'PUBLIC'/(token+'.npz');prefix.parent.mkdir(exist_ok=True)
  m.prefix_codec.exclusive_npz(prefix,m.prefix_codec.encode_public(p,design=design()));pref=m.io.record(root,prefix)
  public.append(dict(object_id=token,prefix=dict(path=prefix.name,sha256=pref['sha256'])))
  data=dict(object_id=np.asarray(token),candidate_probe_Fz_N=np.array([-2.,2.]),candidate_probe_observations_m=np.stack([p['reference_points_m']-.0001,p['reference_points_m']+.0001]),young_Pa=np.asarray(123456.),canonical_target_positions_m=np.full((63,3),999.))
  target=root/'PRIVATE'/(token+'.npz');target.parent.mkdir(exist_ok=True);m.prefix_codec.exclusive_npz(target,data)
  private.append(dict(object_id=token,public_prefix=pref,private_future=m.io.record(root,target)))
  models={}
  for name in m.io.MODELS:
   files={f:put(root,'initial/'+token+'/'+name+'/'+f,{'synthetic':True})for f in ('posterior-arrays.npz','posterior-structure.json','numerics.json')}
   models[name]=dict(execution_completed=True,files=files)
  result=put(root,'initial/'+token+'/result.json',dict(completed=True,object_id=token,models=models));initial.append(dict(object_id=token,result=result))
  c=dict(information=-2.,target_risk=2.,fixed_down=-2.,fixed_up=2.,random=-2.)
  models={name:dict(choices=c,result=put(root,'policy/'+token+'/'+name+'.json',{'synthetic':True}))for name in m.io.MODELS}
  result=put(root,'policy/'+token+'/result.json',dict(completed=True,object_id=token,models=models,private_probe_outcomes_access=False,private_target_or_latent_access=False));choices.append(dict(object_id=token,choices=c,result=result))
 initial_rec=put(root,'initial-seal.json',dict(schema='C38-initial-public-inference-seal-v03',completed=True,count=64,model_executions=128,private_truth_access=False,all_predictions_sealed_before_private_evaluation=True,objects=initial))
 choice_rec=put(root,'choices.json',dict(schema='C38-static-policy-choices-seal-v02',completed=True,count=64,model='nonlinear_table',model_executions=128,private_probe_outcomes_access=False,private_target_or_latent_access=False,all_choices_sealed_before_private_evaluation=True,initial_prediction_seal=initial_rec,choices=choices))
 pub=put(root,'PUBLIC/manifest.json',dict(schema='C38-public-controlled-ideal-prefixes-v01',completed=True,information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',records=public,count=64,physical_configurations=4,noise_repeats_per_configuration=16,independent_population_objects=False,design_sha256=m.obs.DESIGN_SHA256,is_RGBD_perception=False,is_temporal_forecast=False))
 priv=put(root,'PRIVATE/execution.json',dict(completed=True,identity_mapping_persisted_before_noise=True,records=private))
 return dict(initial_seal=initial_rec['path'],choices_seal=choice_rec['path'],public_manifest=pub['path'],observation_execution=priv['path'])


def test_codec_strict_scalar_shape_and_no_extra_private_or_other_action():
 p=payload();b=probe(p);a=encode_probe(p,b,design=design());r=decode_probe(p,a,design=design());assert type(r['acquisition_index'])is int and type(r['force_Fz_N'])is float
 bad=dict(a,young_Pa=np.array(90000.))
 with pytest.raises(ValueError,match='allowlist'):decode_probe(p,bad,design=design())
 bad=copy.deepcopy(a);bad['acquisition_index']=np.asarray(1.,dtype=np.float64)
 with pytest.raises(ValueError,match='int64'):decode_probe(p,bad,design=design())
 bad=copy.deepcopy(a);bad['observed_positions_m']=np.zeros((2,9,3))
 with pytest.raises(ValueError):decode_probe(p,bad,design=design())


def test_prepare_metadata_only_and_all64_choices_required_before_any_measurement(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch)
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Measurement during prepare')))
 h=m.prepare('protocol.json','released',root=tmp_path,**args);assert m.validate('protocol.json',h,root=tmp_path)['rows']and not(tmp_path/'released').exists()
 bad=json.loads((tmp_path/args['choices_seal']).read_text());bad['choices'].pop();(tmp_path/args['choices_seal']).write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='All64'):m.prepare('other.json','other',root=tmp_path,**args)
 assert not(tmp_path/'other.json').exists()


def test_fullsynthetic_release_reads_only3_savedmembers_preserves_same_actual_values(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);h=m.prepare('protocol.json','released',root=tmp_path,**args)
 monkeypatch.setattr(m,'guard',lambda root:None);monkeypatch.setattr(m.obs,'load_frozen_design',lambda root:design())
 monkeypatch.setattr(np.random,'default_rng',lambda *a,**k:(_ for _ in ()).throw(AssertionError('New noise')))
 real=np.load;reads=[]
 class PrivateSpy:
  def __init__(self,z):self.z=z
  def __enter__(self):return self
  def __exit__(self,*a):self.z.close()
  def __getitem__(self,key):
   assert key in {'object_id','candidate_probe_Fz_N','candidate_probe_observations_m'};reads.append(key);return self.z[key]
 def loader(path,*a,**k):
  z=real(path,*a,**k)
  return PrivateSpy(z)if Path(path).parent.name=='PRIVATE'else z
 monkeypatch.setattr(np,'load',loader)
 state=m.run('protocol.json',h,root=tmp_path)
 assert state['completed']and len(reads)==64*3 and not state['target_or_latent_members_read']
 manifest=json.loads((tmp_path/'released/manifest.json').read_text());assert manifest['count']==128 and manifest['object_count']==64 and manifest['one_acquisition_per_policy']
 for row in manifest['records']:
  p=payload(row['object_id'])
  for a in row['acquisitions']:
   with real(tmp_path/'released'/a['acquisition']['path'],allow_pickle=False)as z:r=decode_probe(p,{k:z[k]for k in z.files},design=design())
   np.testing.assert_array_equal(r['observed_positions_m'],p['reference_points_m']+(-.0001 if a['force_Fz_N']<0 else .0001))
 with pytest.raises(FileExistsError):m.run('protocol.json',h,root=tmp_path)


def test_private_member_spy_wrong_id_and_nonfinite_fail_without_truth_access():
 p=payload();values=dict(object_id=np.asarray('b'*32),candidate_probe_Fz_N=np.array([-2.,2.]),candidate_probe_observations_m=np.zeros((2,9,3)))
 class Spy:
  def __getitem__(self,key):assert key in values;return values[key]
 with pytest.raises(ValueError,match='identity'):m.extract_saved_measurements(Spy(),p['object_id'])
 values['object_id']=np.asarray(p['object_id']);values['candidate_probe_observations_m'][0,0,0]=np.nan
 with pytest.raises(ValueError):m.extract_saved_measurements(Spy(),p['object_id'])
