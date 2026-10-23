import copy,os,signal,subprocess,sys,json
import numpy as np
import pytest
from experiments import run_campaign_c38_static_policy_v02 as run


def fixtures():
 ids=[format(i,'032x') for i in range(64)];rows=[dict(object_id=i,result=dict(path='out/'+i+'.json',sha256='a'*64),all_models_numerical_passed=True) for i in ids]
 seal=dict(schema='C38-initial-public-inference-seal-v02',completed=True,count=64,model_executions=128,protocol_sha256='b'*64,objects=rows,private_truth_access=False,all_predictions_sealed_before_private_evaluation=True)
 execution=dict(completed=True,all_owned_workers_reaped=True,private_truth_access=False,technical_failure=False,records=copy.deepcopy(rows))
 return ids,seal,execution


def test_exact_initial_seal_and_terminal_gate_before_any_values(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No array reads in gate')))
 ids,s,e=fixtures();assert len(run.validate_initial_seal(s,e,protocol_digest='b'*64,ids=ids))==64
 for field,value in [('count',63),('model_executions',127),('private_truth_access',True),('completed',False),('protocol_sha256','c'*64)]:
  bad=dict(s,**{field:value})
  with pytest.raises(ValueError):run.validate_initial_seal(bad,e,protocol_digest='b'*64,ids=ids)
 for field,value in [('all_owned_workers_reaped',False),('completed',False),('technical_failure',True)]:
  with pytest.raises(ValueError):run.validate_initial_seal(s,dict(e,**{field:value}),protocol_digest='b'*64,ids=ids)
 bad=copy.deepcopy(e);bad['records'][-1]['object_id']=ids[0]
 with pytest.raises(ValueError):run.validate_initial_seal(s,bad,protocol_digest='b'*64,ids=ids)
 bad=copy.deepcopy(s);bad['objects'][0]['result']['sha256']='f'*64
 with pytest.raises(ValueError):run.validate_initial_seal(bad,e,protocol_digest='b'*64,ids=ids)


def test_runtime_is_hash_bound_data_not_versioned_code_import(tmp_path,monkeypatch):
 monkeypatch.setattr(run.initial,'runtime',lambda:dict(manufactured_runtime=True));monkeypatch.setattr(run.obs,'load_frozen_design',lambda root:None)
 monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No array loading in runtime gate')))
 names={run.initial.WORKER,run.initial.SOURCE,run.codec.SOURCE,'experiments/campaign_c38_controlled_observations.py','experiments/campaign_c38_polynomial_likelihood_v02.py','experiments/campaign_c38_polynomial_cache.py','experiments/campaign_c38_candidate_point_forward.py','experiments/campaign_c38_linear_candidate_forward.py'}
 for n in names:p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SOURCE FIXTURE; NOT EXECUTABLE')
 protocol=dict(schema='C38-initial-public-inference-v73',status='FROZEN_BEFORE_PUBLIC_PREFIX_INFERENCE',models=list(run.MODELS),candidate_mesh=[40,20,20],private_target_or_latent_access=False,runtime=run.initial.runtime(),design=dict(path=run.obs.DESIGN_PATH,sha256=run.obs.DESIGN_SHA256),model_inputs=dict(candidate_archive={},public_manifest={},linear={}),source_sha256={n:run.sha(tmp_path/n) for n in names},output='out')
 p=tmp_path/'protocol.json';p.write_text(json.dumps(protocol));record=run.record(tmp_path,p)
 assert run.validate_initial_protocol(tmp_path,record,full_inputs=False)['schema']==protocol['schema']
 (tmp_path/run.initial.WORKER).write_text('CHANGED')
 with pytest.raises(ValueError):run.validate_initial_protocol(tmp_path,record,full_inputs=False)


def test_numerical_failure_retains_object_with_explicit_fixed_down_fallback():
 c,passed=run.choices_from_result(dict(completed=False,failures=['cloud numerical failure']),'a'*32)
 assert not passed and c['information']==c['target_risk']==c['fixed_down']==-2 and c['fixed_up']==2 and c['random'] in (-2,2)
 result=dict(completed=True,numerical_cloud_controls_passed=True,information=dict(selected_force_Fz_N=2.),target_risk=dict(selected_force_Fz_N=-2.))
 d,passed=run.choices_from_result(result,'a'*32);assert passed and d['information']==2 and d['target_risk']==-2 and d['random']==c['random']


def test_portable_hash_record_rejects_parent_escape_and_tamper(tmp_path):
 f=tmp_path/'file.json';f.write_text('{}');record=run.record(tmp_path,f);assert record['path']=='file.json';assert run.checked(tmp_path,record)==f
 f.write_text('{"changed":true}')
 with pytest.raises(ValueError):run.checked(tmp_path,record)
 for path in ['../private.json','/tmp/private.json','a/../../private.json']:
  with pytest.raises(ValueError):run.local(tmp_path,path)


def test_owned_cleanup_reaps_child_and_does_not_touch_parent():
 child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],start_new_session=True)
 code=run.initial.cleanup_owned(child,.1);assert code is not None and child.poll() is not None
 with pytest.raises(ProcessLookupError):os.kill(child.pid,0)
 assert os.getpid()>0


def test_guard_enforces_CPU_threads_RSS_and_filesystem_reserve(monkeypatch,tmp_path):
 monkeypatch.setattr(os,'sched_getaffinity',lambda pid:{30,31});monkeypatch.setattr(run,'current_rss',lambda pid:1)
 for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:monkeypatch.setenv(name,'2')
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(run.shutil,'disk_usage',lambda p:type('D',(),dict(free=100*1024**3))());run.guard(tmp_path)
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
 with pytest.raises(ValueError):run.guard(tmp_path)
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(run.shutil,'disk_usage',lambda p:type('D',(),dict(free=1024**3))())
 with pytest.raises(OSError):run.guard(tmp_path)
 monkeypatch.setattr(run.shutil,'disk_usage',lambda p:type('D',(),dict(free=100*1024**3))());monkeypatch.setattr(run,'current_rss',lambda pid:3*1024**3)
 with pytest.raises(MemoryError):run.guard(tmp_path)
