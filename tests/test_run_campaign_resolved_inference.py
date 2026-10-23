"""Temporary manufactured metadata/results only; no campaign values or draws."""
import copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_resolved_inference as m
from test_campaign_c38_public_prefix_inference_v02 import design,payload
from test_campaign_c39_postprobe_inference import probe


def put(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def plan():
 jobs=[dict(id=format(i+1,'032x')+'__'+kind,kind=kind,object_id=format(i+1,'032x'),prefix={'path':'prefix','sha256':'a'*64},old_result={'path':'old','sha256':'b'*64},**({'acquisition':{'path':'SIDE','sha256':'c'*64}}if kind=='SIDE'else{}))for kind in m.KINDS for i in range(64)]
 return dict(initial_protocol={'path':'initial','sha256':'a'*64},SIDE_protocol={'path':'SIDE','sha256':'b'*64},initial_prediction_seal={'path':'initialseal','sha256':'c'*64},SIDE_prediction_seal={'path':'SIDEseal','sha256':'d'*64},jobs=jobs,taskset={'path':'/usr/bin/taskset','sha256':m.io.sha('/usr/bin/taskset')},model_inputs={'candidate_archive':{'path':'candidate','sha256':'a'*64},'linear_library':{'path':'linear','sha256':'b'*64},'side_inputs':{}})


def core():
 methods={}
 for name in m.prefix.METHODS:
  methods[name]={k:(np.array([.5,-0.])if k in {'canonical_mean_displacement_m','canonical_coordinate_variance_m2'}else .25)for k in m.CORE_KEYS};methods[name]['canonical_quantile_details']={'old':True}
 return dict(methods=methods,object_id='a'*32,categories=list(m.prefix.CATEGORIES),material_CDF_grid_Pa=np.array([75000.,150000.]),canonical_point_ids=np.arange(63),canonical_interval_point_ids=m.prefix.POINT_IDS.copy(),equal_tail_probabilities=[.025,.5,.975],gates=m.prefix.GATES.copy())


def test_core_comparison_is_bitexact_and_does_not_treat_new_quantiles_as_margin():
 a=core();b=copy.deepcopy(a);b['methods']['adaptive']['canonical_quantile_details']={'new':True};b['quantile_resolution']={'new':True};assert m.core_identity(a,b)['passed']
 b['methods']['GL128']['canonical_coordinate_variance_m2'][0]=np.nextafter(.5,1.)
 r=m.core_identity(a,b);assert not r['passed']and r['bit_mismatch_paths']==['GL128/canonical_coordinate_variance_m2']
 b=copy.deepcopy(a);b['methods']['adaptive']['canonical_mean_displacement_m'][1]=0.
 assert not m.core_identity(a,b)['passed']


def test_saved_bundle_array_closure_and_tamper_before_member_access(tmp_path,monkeypatch):
 d=tmp_path/'saved';d.mkdir();m.io.save_full_result(d,core());files={n:m.io.record(tmp_path,d/n)for n in m.FILES};r=m.load_saved_result(tmp_path,files);assert m.core_identity(core(),r)['passed']
 (d/'numerics.json').write_text('changed')
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('member read before ALLbundle hashes')))
 with pytest.raises(ValueError,match='Changed input'):m.load_saved_result(tmp_path,files)


def test_prepare_only_metadata_then_exact_source_job_closure(tmp_path,monkeypatch):
 put(tmp_path,'source.py',{});monkeypatch.setattr(m,'sources',lambda:['source.py']);i=plan();monkeypatch.setattr(m,'completed_inputs',lambda*a,**k:copy.deepcopy(i));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('actual array during prepare')))
 h=m.prepare('op/p.json','out',initial_protocol='initial',initial_protocol_sha256='a'*64,SIDE_protocol='SIDE',SIDE_protocol_sha256='b'*64,root=tmp_path);p=m.validate('op/p.json',h,root=tmp_path);assert len(p['inputs']['jobs'])==128 and not(tmp_path/'out').exists()
 assert len([j for j in p['inputs']['jobs']if j['kind']=='prefix'and 'acquisition'not in j])==64
 with pytest.raises(FileExistsError):m.prepare('op/p.json','out',initial_protocol='initial',initial_protocol_sha256='a'*64,SIDE_protocol='SIDE',SIDE_protocol_sha256='b'*64,root=tmp_path)
 i['jobs'][0]['object_id']='bad'
 with pytest.raises(ValueError,match='closure'):m.validate('op/p.json',h,root=tmp_path)


def test_unfinished_or_live_C39_blocks_before_any_prediction_array(tmp_path,monkeypatch):
 initial={'path':'initial','sha256':'a'*64};SIDE={'path':'SIDE','sha256':'b'*64};monkeypatch.setattr(m.oldside,'initial_ready',lambda*a,**k:({},{}));monkeypatch.setattr(m.oldside,'validate',lambda*a,**k:dict(initial_protocol=initial,output='SIDEout'));put(tmp_path,'SIDEout/execution.json',dict(completed=False));monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('array before completed C39')))
 with pytest.raises(ValueError,match='terminal/reaped'):m.completed_inputs(tmp_path,initial,SIDE)
 put(tmp_path,'SIDEout/execution.json',dict(completed=True,technical_failure=False,all_owned_workers_reaped=True,protocol=SIDE,records=[dict(pid=99999999)for _ in range(64)],queued_job_ids=[],live=[],pid=os.getpid()))
 with pytest.raises(ValueError,match='absent'):m.completed_inputs(tmp_path,initial,SIDE)


def test_complete64_seal_creates128_jobs_and_rejects_relabelled_old_action(tmp_path,monkeypatch):
 initial={'path':'initial','sha256':'a'*64};SIDE={'path':'SIDE','sha256':'b'*64};ids=[format(i+1,'032x')for i in range(64)];oldrows=[dict(object_id=t,result=put(tmp_path,'old/'+t+'.json',{}))for t in ids];ir=put(tmp_path,'initialseal.json',dict(objects=oldrows));sr=put(tmp_path,'SIDEseal.json',dict(schema='C39-public-SIDE-postprobe-prediction-seal-v01',completed=True,count=64,model_executions=128,protocol=SIDE,private_target_or_latent_access=False,all_postpredictions_sealed_before_private_evaluation=True,initial_prediction_seal=ir,objects=[dict(r,action_id='middle_side_2N')for r in oldrows]));records=[dict(r,pid=99999990+i,reaped=True,completed=True,exit_code=0)for i,r in enumerate(oldrows)];execution=dict(completed=True,technical_failure=False,all_owned_workers_reaped=True,protocol=SIDE,records=records,queued_job_ids=[],live=[],pid=99999989,prediction_seal=sr);put(tmp_path,'SIDEout/execution.json',execution)
 ip=dict(taskset={'path':'taskset','sha256':'c'*64});monkeypatch.setattr(m.oldside,'initial_ready',lambda*a,**k:(ip,dict(initial_prediction_seal=ir)));sp=dict(initial_protocol=initial,output='SIDEout',inputs={},jobs=[dict(object_id=t,prefix={'path':'p/'+t,'sha256':'d'*64},acquisition={'path':'s/'+t,'sha256':'e'*64})for t in ids]);monkeypatch.setattr(m.oldside,'validate',lambda*a,**k:sp);monkeypatch.setattr(m.oldside,'output_result',lambda*a:None);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('NPZ during complete barrier')))
 p=m.completed_inputs(tmp_path,initial,SIDE);assert len(p['jobs'])==128 and all('acquisition'not in r for r in p['jobs']if r['kind']=='prefix')
 bad=m.io.read_json(tmp_path,sr);bad['objects'][-1]['action_id']='middle_down_2N';execution['prediction_seal']=put(tmp_path,'badseal.json',bad);put(tmp_path,'SIDEout/execution.json',execution)
 with pytest.raises(ValueError,match='identities'):m.completed_inputs(tmp_path,initial,SIDE)


@pytest.mark.parametrize('kind',['prefix','SIDE'])
def test_worker_only_original_public_inputs_and_retains_numeric_negative(tmp_path,monkeypatch,kind):
 i=plan();j=next(x for x in i['jobs']if x['kind']==kind);j['old_result']=put(tmp_path,'old.json',dict(completed=True,object_id=j['object_id'],models={n:{'files':{}}for n in m.io.MODELS}));p=dict(output='out',inputs=i);monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m.io,'guard',lambda*a:None);monkeypatch.setattr(m.oldside.obs,'load_frozen_design',lambda*a:design());public=payload(j['object_id']);monkeypatch.setattr(m.oldside.prefix_codec,'load_public_prefix',lambda*a,**k:public);one=probe(public)
 from experiments.campaign_c39_side_observations import encode
 data=encode(public,one,design=design());seen=[]
 class Z:
  files=list(data)
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def __getitem__(self,k):return data[k]
 def load(path,**kw):seen.append(path);assert path==tmp_path/'SIDE';return Z()
 (tmp_path/'SIDE').write_bytes(b'fake');j['acquisition']=m.io.record(tmp_path,tmp_path/'SIDE')if kind=='SIDE'else None
 if kind=='prefix':j.pop('acquisition')
 monkeypatch.setattr(np,'load',load);calls=[]
 class Model:
  def __init__(self):self.initial_candidate=object()
 models={n:Model()for n in m.io.MODELS};monkeypatch.setattr(m.oldside.common,'load_composites',lambda*a:models)
 def infer(model,prefix,*args,**kw):
  calls.append(model);r=core();r.update(object_id=j['object_id'],completed=True,numerical_controls_passed=len(calls)!=1);return r
 monkeypatch.setattr(m.prefix,'infer_public_prefix',infer);monkeypatch.setattr(m.side,'infer_public_probe',infer);monkeypatch.setattr(m,'load_saved_result',lambda*a:core())
 state=m.worker('p','d'*64,j['id'],0,root=tmp_path);assert state['completed']and not state['all_models_numerical_passed']and len(calls)==2
 assert calls==([x.initial_candidate for x in models.values()]if kind=='prefix'else list(models.values()))
 assert len(seen)==(1 if kind=='SIDE'else 0)


def queue(root,monkeypatch,*,technical=False,numeric=False):
 i=plan();p=dict(output='out',inputs=i);monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m.io,'guard',lambda*a:None);monkeypatch.setattr(m.io,'current_rss',lambda*a:0);monkeypatch.setattr(m.time,'sleep',lambda*a:None);calls=[]
 class Process:
  def __init__(self,cmd,**kw):
   assert kw['start_new_session']is True;self.pid=9000000+len(calls);calls.append(self);self.code=1 if technical and len(calls)==1 else 0;j=next(j for j in i['jobs']if j['id']==cmd[cmd.index('--job-id')+1]);models={};base='out/objects/'+j['id']
   for name in m.io.MODELS:
    files={f:put(root,base+'/'+name+'/'+f,{'fake':True})for f in m.FILES};c=put(root,base+'/'+name+'/core-bit-identity.json',{'passed':True});models[name]=dict(execution_completed=True,files=files,core_bit_identity=c,core_bit_identity_passed=True)
   put(root,base+'/result.json',dict(completed=True,pid=self.pid,object_id=j['object_id'],kind=j['kind'],protocol_sha256='e'*64,models=models,private_target_or_latent_access=False,all_models_numerical_passed=not(numeric and len(calls)==1)))
  def poll(self):return self.code
  def wait(self,timeout=None):return self.code
 monkeypatch.setattr(m.subprocess,'Popen',Process);return calls


def test_all128_bundles_256_models_seal_without_discarding_numeric_failure(tmp_path,monkeypatch):
 calls=queue(tmp_path,monkeypatch,numeric=True);s=m.run('p','e'*64,root=tmp_path);seal=json.loads((tmp_path/'out/prediction-seal.json').read_text());assert s['completed']and s['all_owned_workers_reaped']and len(calls)==128 and seal['count']==128 and seal['model_executions']==256 and not seal['all_numerical_controls_passed']and seal['not_new_population_test']


def test_technical_failure_stops_new_launches_and_drains_all_twelve(tmp_path,monkeypatch):
 calls=queue(tmp_path,monkeypatch,technical=True);s=m.run('p','e'*64,root=tmp_path);assert not s['completed']and s['technical_failure']and len(calls)==12 and len(s['records'])==12 and len(s['queued_job_ids'])==116 and s['all_owned_workers_reaped']


def test_owned_timeout_cleanup_and_declared_resources():
 p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],start_new_session=True);m.io.cleanup_owned(p,.1)
 with pytest.raises(ProcessLookupError):os.kill(p.pid,0)
 assert m.RESOURCES['worker_slots']==[[i,i+1]for i in range(6,30,2)]and m.RESOURCES['max_workers']==12 and m.RESOURCES['controller_cores']==[32,33]
 assert m.RESOURCES['worker_wall_s']==600 and m.RESOURCES['global_wall_s']==1800 and m.RESOURCES['worker_RSS_bytes']==2*1024**3
