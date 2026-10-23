"""Synthetic CDFs and temp-root reuse/repair bundles; no actual data/noise."""
import ast,copy,inspect,json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_resolved_repair_v03 as m
from experiments import campaign_resolved_quantiles as q1
from experiments import campaign_resolved_quantiles_v02 as q2
from experiments import campaign_resolved_prefix_inference as prefix1
from experiments import campaign_resolved_SIDE_inference as side1
from test_run_campaign_resolved_inference import core
from test_campaign_c38_public_prefix_inference_v02 import design,payload


def put(root,name,value):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.io.record(root,p)


def strip_projection(x):
 if isinstance(x,dict):return{k:strip_projection(v)for k,v in x.items()if k not in ('unprojected_CDF_estimate','CDF_domain_estimate_projection')}
 if isinstance(x,list):return[strip_projection(v)for v in x]
 return x


@pytest.mark.parametrize('probability',[.025,.5,.975])
def test_valid_v01_query_path_values_bounds_and_callsequence_are_identical(probability):
 calls=[[],[]]
 def cdf(i):
  def query(x):calls[i].append(x);return dict(cdf=x,lower=x,upper=x)
  return query
 a=q1.bounded_continuous_quantile(cdf(0),0.,1.,probability);b=q2.bounded_continuous_quantile(cdf(1),0.,1.,probability)
 assert a==strip_projection(b)and calls[0]==calls[1]


def test_exact_observed_endpoint_tuple_is_only_centralprojection_bounds_unchanged():
 observed=dict(cdf=1.0000000000000002,lower=.9999999999999336,upper=1.);saved=observed.copy()
 def cdf(x):return observed if x==1. else dict(cdf=x,lower=x,upper=x)
 with pytest.raises(FloatingPointError,match='Finite ordered'):q1.bounded_continuous_quantile(cdf,0.,1.,.5)
 r=q2.bounded_continuous_quantile(cdf,0.,1.,.5);assert abs(r['value']-.5)<1e-8 and observed==saved
 with pytest.raises(FloatingPointError,match='Interior'):q2.bounded_continuous_quantile(lambda x:dict(cdf=.6,lower=.4,upper=.5),0.,1.,.5)


def test_both_compiler_function_bodies_unchanged_only_helper_import_changes():
 for before,after in [(prefix1,m.prefix),(side1,m.side)]:
  a=ast.parse(inspect.getsource(before));b=ast.parse(inspect.getsource(after));aa={n.name:ast.dump(n)for n in a.body if isinstance(n,ast.FunctionDef)};bb={n.name:ast.dump(n)for n in b.body if isinstance(n,ast.FunctionDef)};assert aa==bb
  imports=[n.module for n in b.body if isinstance(n,ast.ImportFrom)];assert 'experiments.campaign_resolved_quantiles_v03'in imports and 'experiments.campaign_resolved_quantiles'not in imports


def fixture(root,monkeypatch,failed_count=2):
 monkeypatch.setattr(m,'v02_lineage',lambda root:dict(synthetic=True))
 original=put(root,'original-protocol.json',{});jobs=[];rows=[];sealed=[]
 for kind in ('prefix','SIDE'):
  for i in range(64):
   token=format(i+1,'032x');jid=token+'__'+kind;pid=9000000+len(jobs);base='v01/objects/'+jid;models={}
   for name in m.io.MODELS:
    files={f:put(root,base+'/'+name+'/'+f,{'synthetic':True})for f in m.old.FILES};proof=put(root,base+'/'+name+'/core.json',dict(passed=True,bit_mismatch_paths=[]));models[name]=dict(execution_completed=True,completed=True,numerical_controls_passed=not(kind=='prefix'and i<failed_count and name=='linear_rest_tangent'),files=files,core_bit_identity=proof,core_bit_identity_passed=True)
   result=put(root,base+'/result.json',dict(completed=True,pid=pid,kind=kind,object_id=token,protocol_sha256=original['sha256'],models=models,private_target_or_latent_access=False));rows.append(dict(job_id=jid,kind=kind,object_id=token,pid=pid,result=result,completed=True,reaped=True,exit_code=0));sealed.append(dict(object_id=token,kind=kind,result=result));jobs.append(dict(id=jid,kind=kind,object_id=token,prefix={'path':'PUBLIC/'+token,'sha256':'a'*64},old_result=result,**({'acquisition':{'path':'SIDE/'+token,'sha256':'b'*64}}if kind=='SIDE'else{})))
 seal=put(root,'v01/prediction-seal.json',dict(schema='RESOLVED-prefix-SIDE-prediction-seal-v01',completed=True,count=128,model_executions=256,protocol=original,private_target_or_latent_access=False,not_new_population_test=True,objects=sealed));put(root,'v01/execution.json',dict(completed=True,technical_failure=False,all_owned_workers_reaped=True,protocol=original,queued_job_ids=[],live=[],records=rows,pid=9999998,prediction_seal=seal));p=dict(output='v01',inputs=dict(jobs=jobs,model_inputs={},taskset={'path':'/usr/bin/taskset','sha256':m.io.sha('/usr/bin/taskset')}));monkeypatch.setattr(m.old,'validate',lambda*a,**k:p);return original,p


def test_every_success_reused_only_failed_singlemodel_jobs_and_tamper_barrier(tmp_path,monkeypatch):
 original,_=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('array during reuse_plan')));p=m.reuse_plan(tmp_path,original);assert len(p['reused_models'])==254 and len(p['jobs'])==2 and all(r['model']=='linear_rest_tangent'and r['kind']=='prefix'for r in p['jobs']);assert p['denominators']['v01_failed_model_arms']==2
 assert len({(r['kind'],r['object_id'],r['model'])for r in p['reused_models']})==254
 f=p['reused_models'][-1]['files']['posterior-arrays.npz'];(tmp_path/f['path']).write_bytes(b'changed')
 with pytest.raises(ValueError,match='Changed input'):m.reuse_plan(tmp_path,original)


def test_v01_must_be_fully_terminal_before_repair_plan(tmp_path,monkeypatch):
 original,_=fixture(tmp_path,monkeypatch);p=tmp_path/'v01/execution.json';e=json.loads(p.read_text());e['completed']=False;p.write_text(json.dumps(e))
 with pytest.raises(ValueError,match='ALLv01'):m.reuse_plan(tmp_path,original)


def test_single_failed_model_worker_not_both_models(tmp_path,monkeypatch):
 original,_=fixture(tmp_path,monkeypatch);plan=m.reuse_plan(tmp_path,original);j=plan['jobs'][0];p=dict(output='repair',inputs=plan);monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m.io,'guard',lambda*a:None);monkeypatch.setattr(m.old.oldside.obs,'load_frozen_design',lambda*a:design());public=payload(j['object_id']);monkeypatch.setattr(m.old.oldside.prefix_codec,'load_public_prefix',lambda*a,**k:public)
 class Model:
  def __init__(self):self.initial_candidate=object()
 models={n:Model()for n in m.io.MODELS};plan['model_inputs']=dict(candidate_archive={},linear_library={},side_inputs={});monkeypatch.setattr(m.old.oldside.common,'load_composites',lambda*a:models);calls=[]
 def infer(model,*a,**k):calls.append(model);r=core();r.update(completed=True,numerical_controls_passed=False,object_id=j['object_id']);return r
 monkeypatch.setattr(m.prefix,'infer_public_prefix',infer);monkeypatch.setattr(m.old,'load_saved_result',lambda*a:core());s=m.worker('p','d'*64,j['id'],0,root=tmp_path)
 assert s['completed']and not s['numerical_controls_passed']and s['core_bit_identity_passed']and calls==[models['linear_rest_tangent'].initial_candidate]
 assert s['origin']=='REPAIRED_v03'


def queue(root,monkeypatch,technical=False):
 original,_=fixture(root,monkeypatch,failed_count=20 if technical else 2);plan=m.reuse_plan(root,original);p=dict(output='repair',inputs=plan);monkeypatch.setattr(m,'validate',lambda*a,**k:p);monkeypatch.setattr(m.io,'guard',lambda*a:None);monkeypatch.setattr(m.io,'current_rss',lambda*a:0);monkeypatch.setattr(m.time,'sleep',lambda*a:None);calls=[]
 class Proc:
  def __init__(self,cmd,**kw):
   assert kw['start_new_session'];self.pid=9500000+len(calls);calls.append(self);self.code=1 if technical and len(calls)==1 else 0;j=next(j for j in plan['jobs']if j['id']==cmd[cmd.index('--job-id')+1]);base='repair/objects/'+j['id'];files={f:put(root,base+'/'+f,{'synthetic':True})for f in m.old.FILES};proof=put(root,base+'/core.json',dict(passed=True));put(root,base+'/result.json',dict(completed=True,pid=self.pid,kind=j['kind'],object_id=j['object_id'],model=j['model'],protocol_sha256=cmd[cmd.index('--protocol-sha256')+1],origin='REPAIRED_v03',private_target_or_latent_access=False,files=files,core_bit_identity=proof,core_bit_identity_passed=True,model_completed=True,numerical_controls_passed=len(calls)!=1,all_models_numerical_passed=len(calls)!=1))
  def poll(self):return self.code
  def wait(self,timeout=None):return self.code
 monkeypatch.setattr(m.subprocess,'Popen',Proc);return calls


def test_merged256_hashbarrier_origins_denominators_and_failed_repair_retained(tmp_path,monkeypatch):
 calls=queue(tmp_path,monkeypatch);s=m.run('original-protocol.json',m.io.sha(tmp_path/'original-protocol.json'),root=tmp_path);assert s['completed']and len(calls)==2;seal=json.loads((tmp_path/'repair/prediction-seal.json').read_text());assert seal['model_executions']==256 and len(seal['model_records'])==256 and not seal['all_numerical_controls_passed'];assert sum(r['origin']=='REUSED_v01'for r in seal['model_records'])==254 and sum(r['origin']=='REPAIRED_v03'for r in seal['model_records'])==2 and seal['denominators']['merged_numerical_negative_arms']==1
 assert seal['repair_execution']['path']=='repair/terminal-executions.json'and seal['original_v01_execution']['path']=='v01/execution.json'


def test_first_technical_failure_stops_new_repairs_and_drains_owned_twelve(tmp_path,monkeypatch):
 calls=queue(tmp_path,monkeypatch,technical=True);s=m.run('p','e'*64,root=tmp_path);assert not s['completed']and s['technical_failure']and len(calls)==2 and len(s['records'])==2 and len(s['queued_job_ids'])==18 and s['all_owned_workers_reaped']and not(tmp_path/'repair/prediction-seal.json').exists()


def test_v02_lineage_retains_negative_hashes_and_requires_reaped_barrier(tmp_path,monkeypatch):
 base='runs/gaussian-campaign-v01/resolved-guard-repair-v02/'
 negative=[]
 for i in range(2):
  f=put(tmp_path,f'v02-negative-{i}.json',dict(failure='GL128 support-tail'))
  negative.append(dict(kind='prefix',model='linear_rest_tangent',completed=True,numerical_controls_passed=False,core_bit_identity_passed=True,files={'numerics.json':f}))
 put(tmp_path,base+'terminal-executions.json',dict(completed=True))
 put(tmp_path,base+'prediction-seal.json',dict(model_executions=256,denominators=dict(merged_numerical_negative_arms=2),model_records=negative))
 put(tmp_path,base+'execution.json',dict(completed=True,all_owned_workers_reaped=True,technical_failure=False,live=[],pid=9999998,records=[dict(pid=9999999)]))
 monkeypatch.setattr(np,'load',lambda*a,**k:(_ for _ in()).throw(AssertionError('array access during lineage')))
 assert m.v02_lineage(tmp_path)['negative_model_records']==negative
 p=tmp_path/(base+'execution.json');e=json.loads(p.read_text());e['all_owned_workers_reaped']=False;p.write_text(json.dumps(e))
 with pytest.raises(ValueError,match='v02 must'):m.v02_lineage(tmp_path)
