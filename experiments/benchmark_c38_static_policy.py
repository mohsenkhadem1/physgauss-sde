"""Finite manufactured policy runtime only; no actual archives or prefix reads."""
import argparse,hashlib,json,os,signal,time,resource
from pathlib import Path
import numpy as np
from experiments.campaign_c38_static_policy import design_static_policy
from experiments.benchmark_c38_public_prefix_inference_v02 import manufactured_inputs,design
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c38-static-policy-v01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 def convert(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=convert)+'\n')
def prepare():
 path=OP/'manufactured-protocol.json'
 if path.exists():raise FileExistsError('Preserve frozen benchmark')
 files=['experiments/campaign_c38_static_policy.py','tests/test_campaign_c38_static_policy.py','experiments/benchmark_c38_static_policy.py','experiments/benchmark_c38_public_prefix_inference_v02.py','experiments/campaign_c38_public_prefix_inference_v02.py','experiments/campaign_c38_candidate_point_forward.py','experiments/campaign_c38_linear_candidate_forward.py','experiments/campaign_c38_controlled_observations.py','experiments/campaign_c38_polynomial_likelihood_v02.py','experiments/campaign_c38_polynomial_cache.py','experiments/campaign_c38_table_validation.py','experiments/campaign_c38_nonlinear_table_catalog.py','experiments/run_campaign_c38_static_prequalification.py']
 p=dict(schema='C38-static-policy-manufactured-source-control-v01',status='FROZEN_BEFORE_MANUFACTURED_RUNTIME',source_sha256={f:sha(ROOT/f) for f in files},CPU_affinity_cores=[30,31],CUDA_VISIBLE_DEVICES='',hard_per_candidate_s=300,candidates=['nonlinear','linear'],public_data='Manufactured same48/28 forward fields and deterministic fake prefix; no actual prefix/archive/private records read',cloud_orders=[32,64,128],fantasy_counts=[64,128,256],fantasy_seed=2026102302,random_seed=2026102303,decision='Same raw winner across all orders/budgets; paired advantage >2MCSE each; final-order advantage >2*range(order advantages), otherwise fixed-down. Exact ties down. Same128cloud fantasies reused by all orders and counts. No additional fantasies.',physical_admission=False,actual_execution_authorized=False)
 write(path,p);return sha(path)
class Cap(BaseException):pass
def run(digest):
 p=json.loads((OP/'manufactured-protocol.json').read_text())
 if sha(OP/'manufactured-protocol.json')!=digest or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changed frozen source/protocol')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31} or any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2/thread2 only')
 models,payload=manufactured_inputs();result=dict(completed=False,protocol_sha256=digest,actual_data_read=False,candidates={})
 for name in p['candidates']:
  t=time.monotonic();signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(Cap('Frozen300s manufactured cap')));signal.alarm(300)
  try:
   r=design_static_policy(models[name],payload,design=design());write(OP/f'{name}-manufactured-result.json',r)
   summary={k:r.get(k) for k in ['completed','numerical_cloud_controls_passed','failures','cloud_checks','information','target_risk','fixed_down','fixed_up','random','prior_canonical_mean_coordinate_variance_m2']}
   summary['choices_frozen']=r.get('choices_frozen');summary['fantasy_outcomes_shared_all_orders']=r.get('fantasy_stream',{}).get('same_fantasy_outcomes_all_cloud_orders',False)
  except BaseException as error:summary=dict(completed=False,error=repr(error))
  finally:signal.alarm(0)
  summary['elapsed_s']=time.monotonic()-t;summary['process_peak_RSS_bytes_cumulative']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;result['candidates'][name]=summary;write(OP/'manufactured-result.json',result)
 result['completed']=all(r['completed'] for r in result['candidates'].values());result['all_cloud_controls_passed']=all(r.get('numerical_cloud_controls_passed',False) for r in result['candidates'].values());write(OP/'manufactured-result.json',result);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--sha');args=a.parse_args();r=prepare() if args.action=='prepare' else run(args.sha)
 print(json.dumps(r if isinstance(r,str) else dict(completed=r['completed'],all_cloud_controls_passed=r['all_cloud_controls_passed'],times_s={k:v['elapsed_s'] for k,v in r['candidates'].items()})))
