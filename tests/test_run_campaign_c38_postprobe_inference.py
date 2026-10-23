"""Manufactured public acquisition manifests, spy workers, no actualdata."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_postprobe_inference as m
from test_campaign_c38_public_prefix_inference_v02 import design,payload
from test_campaign_c38_postprobe_inference import probe
from experiments.campaign_c38_public_probe_codec import encode_probe


def put(root,name,value=None,raw=None):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if raw is None:p.write_text(json.dumps(value))
 else:p.write_bytes(raw)
 return m.io.record(root,p)


def fixture(root,monkeypatch):
 put(root,'source.py',{});monkeypatch.setattr(m,'sources',lambda:['source.py'])
 prefixes=[];acquired=[]
 for i in range(64):
  token=format(i+1,'032x');r=put(root,'PUBLIC/'+token+'.npz',raw=b'fake prefix, no arrays');prefixes.append(dict(object_id=token,prefix=dict(path=Path(r['path']).name,sha256=r['sha256'])))
  acqs=[]
  for F,tag in [(-2.,'down2'),(2.,'up2')]:
   a=put(root,'acquired/'+token+'__'+tag+'.npz',raw=b'fake acquiredoneprobe');acqs.append(dict(force_Fz_N=F,acquisition=dict(path=Path(a['path']).name,sha256=a['sha256'])))
  acquired.append(dict(object_id=token,acquisitions=acqs))
 pub=put(root,'PUBLIC/manifest.json',dict(schema='C38-public-controlled-ideal-prefixes-v01',completed=True,information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',records=prefixes,count=64,physical_configurations=4,noise_repeats_per_configuration=16,independent_population_objects=False,design_sha256=m.obs.DESIGN_SHA256,is_RGBD_perception=False,is_temporal_forecast=False))
 initial=put(root,'initial-seal.json',{'manufactured':True});choices=put(root,'choices.json',dict(completed=True,count=64,initial_prediction_seal=initial,private_probe_outcomes_access=False,private_target_or_latent_access=False,all_choices_sealed_before_private_evaluation=True))
 manifest=put(root,'acquired/manifest.json',dict(schema='C38-public-acquired-probes-v01',completed=True,count=128,object_count=64,records=acquired,initial_prediction_seal=initial,choices_seal=choices,one_acquisition_per_policy=True,counterfactual_fixed_policy_actions_share_same_saved_measurement=True,new_noise_draws=False,target_or_latent_members_included=False,is_RGBD_perception=False,is_temporal_forecast=False))
 inputs=dict(initial_prediction_seal=initial,public_manifest=pub,candidate_archive=put(root,'candidate.npz',raw=b'fakecandidate'),linear_library=put(root,'linear.npz',raw=b'fakelinear'))
 ip=dict(taskset=dict(path='/usr/bin/taskset',sha256=m.io.sha('/usr/bin/taskset')))
 monkeypatch.setattr(m,'initial_ready',lambda *a,**k:(ip,inputs))
 return manifest,inputs


def test_prepare_metadataonly_all128join_exclusive_and_private_extra_rejected(tmp_path,monkeypatch):
 manifest,inputs=fixture(tmp_path,monkeypatch)
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Arrayread duringprepare')))
 h=m.prepare('opdev/p.json','out',initial_protocol='initial.json',initial_protocol_sha256='a'*64,acquisitions_manifest=manifest['path'],root=tmp_path)
 p=m.validate('opdev/p.json',h,root=tmp_path);assert len(p['jobs'])==128 and set(p['inputs'])==set(inputs)|{'choices_seal','acquisitions_manifest'}and not(tmp_path/'out').exists()
 assert all(set(j)=={'id','object_id','force_Fz_N','prefix','acquisition'}for j in p['jobs'])
 with pytest.raises(FileExistsError):m.prepare('opdev/p.json','out',initial_protocol='initial.json',initial_protocol_sha256='a'*64,acquisitions_manifest=manifest['path'],root=tmp_path)
 bad=json.loads((tmp_path/manifest['path']).read_text());bad['private_target']=dict(path='PRIVATE/future.npz',sha256='f'*64);(tmp_path/manifest['path']).write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='allowlist'):m.public_jobs(tmp_path,m.io.record(tmp_path,tmp_path/manifest['path']),inputs)


def test_missing_other_action_or_basename_escape_rejected(tmp_path,monkeypatch):
 manifest,inputs=fixture(tmp_path,monkeypatch);path=tmp_path/manifest['path'];original=json.loads(path.read_text());bad=copy.deepcopy(original);bad['records'][0]['acquisitions'].pop();path.write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='Exactly2'):m.public_jobs(tmp_path,m.io.record(tmp_path,path),inputs)
 bad=copy.deepcopy(original);bad['records'][0]['acquisitions'][0]['acquisition']['path']='../PRIVATE/future.npz';path.write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='basename'):m.public_jobs(tmp_path,m.io.record(tmp_path,path),inputs)


def test_worker_reads_onlyprefix_oneprobe_andcalls_bothmodels_fresh(tmp_path,monkeypatch):
 manifest,inputs=fixture(tmp_path,monkeypatch);jobs,choices=m.public_jobs(tmp_path,manifest,inputs);job=jobs[0];p=dict(output='out',jobs=jobs,inputs=inputs)
 monkeypatch.setattr(m,'validate',lambda *a,**k:p);monkeypatch.setattr(m.io,'guard',lambda *a:None);monkeypatch.setattr(m.obs,'load_frozen_design',lambda root:design())
 public=payload(job['object_id']);acquired=probe(public,job['force_Fz_N']);encoded=encode_probe(public,acquired,design=design());reads=[]
 monkeypatch.setattr(m.prefix_codec,'load_public_prefix',lambda root,r,**k:public)
 from experiments import campaign_c38_candidate_point_forward as cf
 candidate=object();linear=object();monkeypatch.setattr(cf,'CandidatePointForward',lambda *a,**k:candidate);monkeypatch.setattr(m.io,'load_linear_library',lambda *a,**k:linear)
 class Z:
  def __init__(self,data):self.data=data;self.files=list(data)
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def __getitem__(self,k):return self.data[k]
 def load(path,**kw):
  name=Path(path).relative_to(tmp_path).as_posix();reads.append(name)
  if name==job['acquisition']['path']:return Z(encoded)
  if name==inputs['candidate_archive']['path']:return Z({k:np.zeros(1)for k in cf.ARCHIVE_KEYS})
  raise AssertionError('Otheraction/private read:'+name)
 monkeypatch.setattr(np,'load',load);calls=[]
 def infer(model,prefix,one,*,design):
  assert model in(candidate,linear)and prefix is public and one['force_Fz_N']==job['force_Fz_N'];calls.append(model)
  return dict(completed=True,numerical_controls_passed=True,object_id=job['object_id'],acquired_probe_force_Fz_N=job['force_Fz_N'],acquired_coordinate_count=54,fresh_prior=True,initial_posterior_weights_reused=False,methods={},failures=[],canonical_test_array=np.zeros((63,3)))
 monkeypatch.setattr(m,'infer_public_probe',infer)
 r=m.worker('p.json','a'*64,job['id'],0,root=tmp_path)
 assert r['completed']and calls==[candidate,linear]and r['force_Fz_N']==job['force_Fz_N']and len(reads)==2
 assert set(r['models'])==set(m.io.MODELS)and not r['private_target_or_latent_access']


def fake_queue(root,monkeypatch,fail=False,numeric=False):
 jobs=[dict(id=format(i+1,'032x')+'__'+tag,object_id=format(i+1,'032x'),force_Fz_N=F)for i in range(64)for F,tag in [(-2.,'down2'),(2.,'up2')]]
 p=dict(output='out',jobs=jobs,taskset={'path':'fake-taskset'},inputs={'choices_seal':{'path':'choices','sha256':'a'*64},'initial_prediction_seal':{'path':'initial','sha256':'b'*64}})
 monkeypatch.setattr(m,'validate',lambda *a,**k:p);monkeypatch.setattr(m.io,'guard',lambda *a:None);monkeypatch.setattr(m.io,'current_rss',lambda *a:0);monkeypatch.setattr(m.time,'sleep',lambda *a:None);calls=[]
 class Process:
  def __init__(self,cmd,**kw):
   assert kw['start_new_session']is True and kw['cwd']==root;assert not any('PRIVATE'in str(x)for x in cmd)
   self.pid=800000+len(calls);calls.append(self);self.code=1 if fail and len(calls)==1 else 0
   jid=cmd[cmd.index('--job-id')+1];j=next(x for x in jobs if x['id']==jid);d=root/'out/objects'/jid;d.mkdir(parents=True);models={}
   for name in m.io.MODELS:
    files={f:put(root,str((d/name/f).relative_to(root)),{'fake':True})for f in ('posterior-arrays.npz','posterior-structure.json','numerics.json')};models[name]=dict(execution_completed=True,files=files)
   put(root,str((d/'result.json').relative_to(root)),dict(completed=True,pid=self.pid,object_id=j['object_id'],force_Fz_N=j['force_Fz_N'],protocol_sha256='c'*64,models=models,private_target_or_latent_access=False,all_models_numerical_passed=not(numeric and len(calls)==1)))
  def poll(self):return self.code
  def wait(self,timeout=None):return self.code
 monkeypatch.setattr(m.subprocess,'Popen',Process)
 return calls


def test_full128pairs256modelseal_retains_numeric_negative_no_evaluation(tmp_path,monkeypatch):
 calls=fake_queue(tmp_path,monkeypatch,numeric=True);s=m.run('p.json','c'*64,root=tmp_path);assert s['completed']and len(calls)==128
 a=json.loads((tmp_path/'out/prediction-seal.json').read_text());assert a['count']==128 and a['model_executions']==256 and not a['all_numerical_controls_passed']and not a['private_target_or_latent_access']
 assert all(set(r)=={'object_id','force_Fz_N','result'}for r in a['objects'])and len({(r['object_id'],r['force_Fz_N'])for r in a['objects']})==128
 assert a['all_postpredictions_sealed_before_private_evaluation']and s['all_owned_workers_reaped']


def test_technical_failure_stopsnew_drains_inflight_keeps116queued(tmp_path,monkeypatch):
 calls=fake_queue(tmp_path,monkeypatch,fail=True);s=m.run('p.json','c'*64,root=tmp_path)
 assert not s['completed']and s['technical_failure']and len(calls)==12 and len(s['records'])==12 and len(s['queued_job_ids'])==116 and s['all_owned_workers_reaped']


def test_declared12_disjoint_CPU2_slots_stay_outside_other_roles():
 assert m.RESOURCES['max_workers']==12
 assert m.RESOURCES['worker_slots']==[[i,i+1]for i in range(6,30,2)]
 assert len({x for pair in m.RESOURCES['worker_slots']for x in pair})==24
 assert not(set(range(0,6))|{30,31,32,33,34,35})&{x for pair in m.RESOURCES['worker_slots']for x in pair}
 assert m.RESOURCES['controller_cores']==[32,33]and m.RESOURCES['worker_RSS_bytes']==2*1024**3
