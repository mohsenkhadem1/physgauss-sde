"""Temporary metadata/poison barriers; NEVER call actual generator or inference."""
import json,copy
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_surrogate_sbc as m


def put(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def closure(root):
 pr=put(root,'protocol.json',dict(fixture=True));p=dict(output='sbc');model={n:put(root,'model/'+n,{})for n in m.FILES};rows=[];laws=[];executed=[]
 for i in range(64):
  public={n:put(root,'public/'+str(i)+'/'+n,{})for n in m.FILES};private=put(root,'PRIVATE/'+str(i)+'.npz',dict(poison_unused=True));rows.append(dict(object_index=i,public=public,private=private));value=dict(object_index=i,private_labels_read=False,observations={k:{}for k in m.sbc.PHASES},coefficients={c:{}for c in m.sbc.CATEGORIES},phases={k:dict(phase=k,completed=False,numerical_controls_passed=False,failures=[])for k in m.sbc.PHASES});files={n:put(root,'laws/'+str(i)+'/'+n,value if n=='posterior-structure.json'else {})for n in m.FILES};laws.append(dict(object_index=i,files=files));executed.append(dict(object_index=i,pid=9100001+i,exit_code=0,reaped=True,files=files))
 g=put(root,'sbc/generation/seal.json',dict(schema='SURROGATE-SBC-generation-seal-v01',completed=True,count=64,protocol=pr,model=model,objects=rows,generator_design_sealed_before_draws=True));e=put(root,'sbc/inference/execution.json',dict(completed=True,pid=9100000,protocol=pr,stage='inference',all_owned_workers_reaped=True,records=executed));seal=put(root,'sbc/inference/seal.json',dict(schema='SURROGATE-SBC-all-laws-seal-v01',completed=True,count=64,phase_laws=128,protocol=pr,private_labels_read=False,generation_seal=g,execution=e,objects=laws));return p,pr,laws


def test_all64_128law_barrier_missing_LAST_bundle_blocks_every_private_access(tmp_path,monkeypatch):
 p,pr,laws=closure(tmp_path);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No array before barrier')));g,s,proof=m.barrier(tmp_path,p,pr);assert proof['all65_inference_PIDs_absent']
 (tmp_path/laws[-1]['files']['posterior-arrays.npz']['path']).unlink();original=m.io.checked
 def check(root,r):
  if r['path'].startswith('PRIVATE/'):raise AssertionError('Private bytes before all laws')
  return original(root,r)
 monkeypatch.setattr(m.io,'checked',check)
 with pytest.raises(FileNotFoundError):m.barrier(tmp_path,p,pr)


def test_selective_poisoned_unused_arrays_and_exact_tree():
 values=dict(young_Pa=np.asarray(100000.),support_category=np.asarray('both_ends'),target_displacement_m=np.zeros((63,3)))
 class Poison:
  def __getitem__(self,k):
   if k not in m.PRIVATE_KEYS:raise AssertionError('Unused measurement/V/noise array access')
   return values[k]
 t=m.extract_truth(Poison());assert set(t)==m.PRIVATE_KEYS
 assert m.exact_tree_identity(values,values)
 b=copy.deepcopy(values);b['target_displacement_m'][0,0]=1e-30;assert not m.exact_tree_identity(values,b)


def test_complete_metadata_rejects_live_PID_and_missing_object(tmp_path,monkeypatch):
 p,pr,laws=closure(tmp_path);sfile=tmp_path/'sbc/inference/seal.json';s=json.loads(sfile.read_text());s['objects'].pop();sfile.write_text(json.dumps(s))
 with pytest.raises(ValueError):m.barrier(tmp_path,p,pr)


def test_prepare_validate_no_rng_no_arrays_and_seed_tamper_guard(tmp_path,monkeypatch):
 names=['source.py'];(tmp_path/names[0]).write_text('fixture')
 monkeypatch.setattr(m,'sources',lambda:names);a=put(tmp_path,'modelprefix.json',{});b=put(tmp_path,'modelSIDE.json',{});ref=put(tmp_path,'ref.json',dict(inputs=dict(model_inputs=dict(candidate_archive=a,side_inputs=dict(candidate_side_archive=b)))))
 monkeypatch.setattr(m,'rng',lambda*a,**k:(_ for _ in()).throw(AssertionError('No pre-freeze random draw')));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('No model/truth arrays in prepare')))
 h=m.prepare('op/protocol.json','new-sbc',ref['path'],ref['sha256'],root=tmp_path);m.validate('op/protocol.json',h,root=tmp_path);assert not(tmp_path/'new-sbc').exists()
 path=tmp_path/'op/protocol.json';p=json.loads(path.read_text());p['design']['seeds']['latent']+=1;path.write_text(json.dumps(p));changed=m.io.sha(path)
 with pytest.raises(ValueError):m.validate('op/protocol.json',changed,root=tmp_path)


def test_actual_owned_controller_sigterm_and_timeout_drain_reap(tmp_path):
 import os,signal,subprocess,sys,time
 for mode in ('interrupt','timeout'):
  out=tmp_path/mode;out.mkdir();code="from experiments.run_campaign_surrogate_sbc import owned_jobs;from pathlib import Path;import sys;owned_jobs(lambda i,s:[sys.executable,'-c','import time;time.sleep(30)'],[0],1,Path(sys.argv[1]),worker_wall_s=float(sys.argv[2]),global_wall_s=15,RSS_bytes=2147483648)"
  child=subprocess.Popen([sys.executable,'-c',code,str(out),'10'if mode=='interrupt'else'.15'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  deadline=time.monotonic()+10;journal=None
  while time.monotonic()<deadline:
   try:
    journal=json.loads((out/'journal.json').read_text())
    if journal['live']:break
   except(FileNotFoundError,json.JSONDecodeError):pass
   time.sleep(.03)
  assert journal and journal['live'];pid=journal['live'][0]['pid']
  if mode=='interrupt':child.send_signal(signal.SIGTERM)
  child.communicate(timeout=10);assert child.returncode!=0
  final=json.loads((out/'journal.json').read_text());assert final['status']=='FAILED_ALL_REAPED'and final['all_owned_workers_reaped']and not final['live']
  assert not Path('/proc',str(pid)).exists()and any(r['pid']==pid and r['reaped']for r in final['records'])


def test_evaluation_reconstruction_mismatch_blocks_ALL_truth_members(tmp_path,monkeypatch):
 p=dict(output='out');pr=dict(path='recipe',sha256='a'*64);g=dict(model={'fixture':'model'},objects=[dict(object_index=0,public={'fixture':'public'},private={'fixture':'PRIVATE'})]);seal=dict(objects=[dict(object_index=0,files={'fixture':'saved'})]);coeff={'fixture':np.zeros(3)};obs={'fixture':np.zeros(3)};saved=dict(coefficients=coeff,observations=obs,phases={k:dict(log_normalizer=0.)for k in m.sbc.PHASES});public=dict(object_index=0,observed_displacements=obs)
 monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m,'guard',lambda*a:None);monkeypatch.setattr(m,'barrier',lambda*a:(g,seal,{}))
 monkeypatch.setattr(m,'read_bundle',lambda root,f:dict(coefficients=coeff)if f['fixture']=='model'else public if f['fixture']=='public'else saved)
 monkeypatch.setattr(m.sbc,'compile_inference',lambda*a:(dict(log_normalizer=1.),{'sealed-post':object()}));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('Private labels before frozen posterior verification')))
 with pytest.raises(ValueError,match='BEFORE private'):m.worker('recipe','a'*64,'evaluation',0,0,root=tmp_path)


def test_real_composite_typed_API_with_synthetic_forward_fixture(tmp_path,monkeypatch):
 from experiments import campaign_c38_candidate_point_forward as prefix
 from experiments import campaign_c39_side_point_forward as side
 calls=[]
 class Initial:
  categories=side.CATEGORIES;young_nodes=side.YOUNG.copy()
  def observation_coefficients(self,c,force,ids):calls.append(('prefix',force,len(ids)));return np.zeros((3,3*len(ids)))
 class Side:
  categories=side.CATEGORIES;young_nodes=side.YOUNG.copy()
  def observation_coefficients(self,c,action,ids):assert action=='middle_side_2N';calls.append(('SIDE',action,len(ids)));return np.zeros((3,3*len(ids)))
 monkeypatch.setattr(prefix,'CandidatePointForward',lambda*a:Initial());monkeypatch.setattr(side,'SidePointForward',lambda*a:Side());monkeypatch.setattr(m.io,'checked',lambda root,r:r['path'])
 class Archive:
  def __init__(self,keys):self.files=keys
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def __getitem__(self,k):return np.zeros(1)
 monkeypatch.setattr(np,'load',lambda path,**kw:Archive(prefix.ARCHIVE_KEYS if path=='prefix'else side.ARCHIVE_KEYS))
 coeff=m.frozen_coefficients(tmp_path,dict(model_inputs=dict(prefix=dict(path='prefix'),SIDE=dict(path='SIDE'))));m.sbc.validate_coefficients(coeff)
 assert ('prefix',0.,9)in calls and ('prefix',1.,63)in calls and ('SIDE','middle_side_2N',9)in calls


def test_manufactured_worker_verifies_both_sealed_laws_BEFORE_poisoned_truth_then_reuses_posts(tmp_path,monkeypatch):
 from test_campaign_surrogate_sbc import coefficients
 coeff=coefficients();E=100000.;category=m.sbc.CATEGORIES[0];obs={k:m.sbc.evaluate_polynomial(coeff[category][k],E)+np.linspace(-1e-4,1e-4,27)for k in m.sbc.PHASES};phases={k:m.sbc.infer(coeff,obs,k)for k in m.sbc.PHASES};previous=dict(coefficients=coeff,observations=obs,phases=phases)
 # A real serialized NPZ/structure roundtrip must preserve reconstruction identity.
 files=m.bundle(tmp_path,tmp_path/'sealed-synthetic',previous);restored=m.read_bundle(tmp_path,files);assert m.exact_tree_identity(previous,restored)
 g=dict(model={'fixture':'model'},objects=[dict(object_index=0,public={'fixture':'public'},private={'path':'PRIVATE.npz','sha256':'a'*64})]);seal=dict(objects=[dict(object_index=0,files={'fixture':'saved'})]);public=dict(object_index=0,observed_displacements=obs)
 monkeypatch.setattr(m,'validate',lambda*a,**k:dict(output='new-output'));monkeypatch.setattr(m,'guard',lambda*a:None);monkeypatch.setattr(m,'barrier',lambda*a:(g,seal,{}));monkeypatch.setattr(m,'read_bundle',lambda root,f:dict(coefficients=coeff)if f['fixture']=='model'else public if f['fixture']=='public'else restored)
 original=m.sbc.compile_inference;events=[];postids=set()
 def compile(*args):
  assert 'truth'not in events
  r,p=original(*args);events.append(args[2]);postids.update(id(v)for v in p.values());return r,p
 monkeypatch.setattr(m.sbc,'compile_inference',compile)
 values=dict(young_Pa=np.asarray(E),support_category=np.asarray(category),target_displacement_m=m.sbc.evaluate_polynomial(coeff[category]['target'],E).reshape(63,3))
 class Private:
  def __enter__(self):assert events==list(m.sbc.PHASES);events.append('truth');return self
  def __exit__(self,*a):pass
  def __getitem__(self,k):
   if k not in m.PRIVATE_KEYS:raise AssertionError('Poison unused sensor noise/future measurement')
   return values[k]
 monkeypatch.setattr(np,'load',lambda*a,**k:Private());monkeypatch.setattr(m.io,'checked',lambda root,r:tmp_path/r['path'])
 class Uniforms:
  def random(self,n):return np.full(n,.3)
 monkeypatch.setattr(m,'rng',lambda*a:Uniforms())
 original_score=m.sbc.score_laws
 def score(*args,**kw):
  assert events[-1]=='truth'and all(id(p)in postids for p in kw['posts'].values());return original_score(*args,**kw)
 monkeypatch.setattr(m.sbc,'score_laws',score);m.worker('recipe','a'*64,'evaluation',0,0,root=tmp_path)
 assert events==['prefix','SIDE','truth']
