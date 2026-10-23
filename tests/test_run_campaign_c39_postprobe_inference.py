"""Synthetic sanitizedSIDE manifests/spyworkers/queues; no actual C39 values."""
import copy,json,os,signal,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c39_postprobe_inference as m
from test_campaign_c38_public_prefix_inference_v02 import design,payload
from test_campaign_c39_postprobe_inference import probe
from experiments.campaign_c39_side_observations import encode


def put(root,name,value=None,raw=None):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if raw is None:p.write_text(json.dumps(value))
 else:p.write_bytes(raw)
 return m.io.record(root,p)


def fixture(root,monkeypatch):
 put(root,'source.py',{});monkeypatch.setattr(m,'sources',lambda:['source.py']);prefixes=[];acquired=[]
 for i in range(64):
  token=format(i+1,'032x');r=put(root,'PUBLIC/'+token+'.npz',raw=b'fakeprefix');prefixes.append(dict(object_id=token,prefix=dict(path=Path(r['path']).name,sha256=r['sha256'])))
  a=put(root,'acquired/'+token+'__side2.npz',raw=b'fakeSIDE');acquired.append(dict(object_id=token,acquisition=dict(path=Path(a['path']).name,sha256=a['sha256'])))
 pub=put(root,'PUBLIC/manifest.json',dict(schema='C38-public-controlled-ideal-prefixes-v01',completed=True,information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',records=prefixes,count=64,physical_configurations=4,noise_repeats_per_configuration=16,independent_population_objects=False,design_sha256=m.obs.DESIGN_SHA256,is_RGBD_perception=False,is_temporal_forecast=False))
 initial=put(root,'initial-seal.json',{'manufactured':True});cp=dict(path='policyprotocol',sha256='a'*64);choices=put(root,'choices.json',dict(protocol=cp));put(root,'policy/execution.json',dict(completed=True,owned_worker_reaped=True,choices_seal=choices,pid=9999999,worker_pid=9999998))
 manifest=put(root,'acquired/manifest.json',dict(schema='C39-public-acquired-SIDE-probes-v01',completed=True,count=64,records=acquired,extension_design=m.common.DESIGN,choices_seal=choices,action_id='middle_side_2N',force_vector_N=[0.,2.,0.],same_original_identities_and_prefixes=True,new_SIDE_noise_only=True,no_new_DOWN_noise=True,private_labels_or_targets_included=False,is_RGBD_perception=False,is_temporal_forecast=False))
 side=dict(forward_point_validation=put(root,'validation.json',{}));inputs=dict(initial_prediction_seal=initial,public_manifest=pub,candidate_archive=put(root,'candidate.npz',raw=b'fakecandidate'),linear_library=put(root,'linear.npz',raw=b'fakelinear'),side_inputs=side);ip=dict(taskset=dict(path='/usr/bin/taskset',sha256=m.io.sha('/usr/bin/taskset')))
 monkeypatch.setattr(m,'initial_ready',lambda*a,**k:(ip,inputs.copy()));monkeypatch.setattr(m.common,'side_plan',lambda*a:side);monkeypatch.setattr(m.common,'validate_choices',lambda*a,**k:dict(protocol=cp));monkeypatch.setattr(m.choice_runtime,'validate',lambda*a,**k:({'output':'policy'},{}));return manifest,inputs


def test_prepare_exact64_source_only_gate_and_private_extra_rejected(tmp_path,monkeypatch):
 manifest,inputs=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No arrays in gate')))
 h=m.prepare('op/p.json','out',initial_protocol='initial.json',initial_protocol_sha256='a'*64,acquisitions_manifest=manifest['path'],forward_validation='validation.json',root=tmp_path);p=m.validate('op/p.json',h,root=tmp_path)
 assert len(p['jobs'])==64 and not(tmp_path/'out').exists();assert all(set(j)=={'id','object_id','action_id','force_world_N','prefix','acquisition'}for j in p['jobs'])
 q=tmp_path/manifest['path'];raw=json.loads(q.read_text());bad=copy.deepcopy(raw);bad['private_target']={};q.write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='allowlist'):m.public_jobs(tmp_path,m.io.record(tmp_path,q),inputs)
 bad=copy.deepcopy(raw);bad['records'].pop();q.write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='All64'):m.public_jobs(tmp_path,m.io.record(tmp_path,q),inputs)
 bad=copy.deepcopy(raw);bad['records'][0]['acquisition']['path']='../PRIVATE/field.npz';q.write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='basename'):m.public_jobs(tmp_path,m.io.record(tmp_path,q),inputs)


def test_worker_only_oneSIDE_plus_prefix_bothmodels_no_oldDOWN_reads(tmp_path,monkeypatch):
 manifest,inputs=fixture(tmp_path,monkeypatch);jobs,choices=m.public_jobs(tmp_path,manifest,inputs);j=jobs[0];p=dict(output='out',jobs=jobs,inputs=inputs);monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m.io,'guard',lambda*a:None);monkeypatch.setattr(m.obs,'load_frozen_design',lambda*a:design());public=payload(j['object_id']);one=probe(public);members=encode(public,one,design=design());reads=[];monkeypatch.setattr(m.prefix_codec,'load_public_prefix',lambda*a,**k:public);models=dict(zip(m.io.MODELS,(object(),object())));monkeypatch.setattr(m.common,'load_composites',lambda*a:models)
 class Z:
  files=list(members)
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def __getitem__(self,k):return members[k]
 def load(path,**kw):
  reads.append(Path(path).relative_to(tmp_path).as_posix());assert reads[-1]==j['acquisition']['path'];return Z()
 monkeypatch.setattr(np,'load',load);calls=[]
 def infer(model,prefix,one,*,design):
  assert prefix is public and one['action_id']=='middle_side_2N';calls.append(model);return dict(completed=True,numerical_controls_passed=True,object_id=j['object_id'],acquired_probe_action_id=j['action_id'],acquired_probe_force_world_N=np.array([0.,2.,0.]),methods={},failures=[])
 monkeypatch.setattr(m,'infer_public_probe',infer);r=m.worker('p','c'*64,j['id'],0,root=tmp_path);assert r['completed']and calls==list(models.values())and len(reads)==1
 assert r['action_id']=='middle_side_2N'and r['force_world_N']==[0.,2.,0.]and 'force_Fz_N'not in r


def fake_queue(root,monkeypatch,fail=False,numeric=False):
 jobs=[dict(id=format(i+1,'032x')+'__side2',object_id=format(i+1,'032x'),action_id='middle_side_2N',force_world_N=[0.,2.,0.])for i in range(64)];p=dict(output='out',jobs=jobs,taskset={'path':'fake-taskset'},inputs={'choices_seal':{'path':'choices','sha256':'a'*64},'initial_prediction_seal':{'path':'initial','sha256':'b'*64}});monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m.io,'guard',lambda*a:None);monkeypatch.setattr(m.io,'current_rss',lambda*a:0);monkeypatch.setattr(m.time,'sleep',lambda*a:None);calls=[]
 class Process:
  def __init__(self,cmd,**kw):
   assert kw['start_new_session']is True;self.pid=800000+len(calls);calls.append(self);self.code=1 if fail and len(calls)==1 else 0;jid=cmd[cmd.index('--job-id')+1];j=next(x for x in jobs if x['id']==jid);d=root/'out/objects'/jid;d.mkdir(parents=True);models={}
   for name in m.io.MODELS:
    files={f:put(root,str((d/name/f).relative_to(root)),{'fake':True})for f in ('posterior-arrays.npz','posterior-structure.json','numerics.json')};models[name]=dict(execution_completed=True,files=files)
   put(root,str((d/'result.json').relative_to(root)),dict(completed=True,pid=self.pid,object_id=j['object_id'],action_id=j['action_id'],force_world_N=j['force_world_N'],protocol_sha256='c'*64,models=models,private_target_or_latent_access=False,all_models_numerical_passed=not(numeric and len(calls)==1)))
  def poll(self):return self.code
  def wait(self,timeout=None):return self.code
 monkeypatch.setattr(m.subprocess,'Popen',Process);return calls


def test_full64_128modelseal_retains_numeric_failure(tmp_path,monkeypatch):
 calls=fake_queue(tmp_path,monkeypatch,numeric=True);s=m.run('p','c'*64,root=tmp_path);assert s['completed']and len(calls)==64 and s['all_owned_workers_reaped'];seal=json.loads((tmp_path/'out/prediction-seal.json').read_text());assert seal['count']==64 and seal['model_executions']==128 and not seal['all_numerical_controls_passed'];assert all(set(r)=={'object_id','action_id','result'}for r in seal['objects'])


def test_first_technical_failure_stops_new_drains_twelve(tmp_path,monkeypatch):
 calls=fake_queue(tmp_path,monkeypatch,fail=True);s=m.run('p','c'*64,root=tmp_path);assert not s['completed']and s['technical_failure']and len(calls)==12 and len(s['records'])==12 and len(s['queued_job_ids'])==52 and s['all_owned_workers_reaped']


def test_owned_cleanup_and_worker_resources():
 proc=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],start_new_session=True);assert m.io.cleanup_owned(proc,.1)is not None
 with pytest.raises(ProcessLookupError):os.kill(proc.pid,0)
 assert m.RESOURCES['worker_slots']==[[i,i+1]for i in range(6,30,2)]and m.RESOURCES['controller_cores']==[32,33]
 assert m.RESOURCES['worker_wall_s']==300 and m.RESOURCES['global_wall_s']==3600
