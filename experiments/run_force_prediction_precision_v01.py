"""Frozen saved-only predictive precision; no inference/forward-model calls."""
from pathlib import Path
import argparse,hashlib,json,os,resource,time
import numpy as np
from experiments import campaign_force_prediction_precision_v01 as h
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/force-prediction-precision-v01'
OLD=ROOT/'runs/gaussian-campaign-v01/compliant-force-challenge-v01'
SOURCES=['experiments/campaign_force_prediction_precision_v01.py','experiments/run_force_prediction_precision_v01.py','tests/test_force_prediction_precision_v01.py','experiments/supervise_surrogate_sbc_continuation.py','experiments/campaign_owned_proc.py']
DESIGN=dict(method='Self-normalized ratio delta MCSE, centered variance and SD delta',blocks=16,chain_clusters=8,
 gate='max(16half and8chain RMS_MCSE)<=max(2e-6,.03predictiveRMS_SD), bothmean andSD; blockweightESS>=8; positiveallmasses',
 old_failed_gates_preserved=True,old_pass_not_overwritten=True,production_saved_read_requires_root_review=True,
 block_weight_ESS_is='Mass concentration effective count only; not autocorrelation-adjusted ESS',
 qualified_forecast_requires_original_acquisition_and_independent_SMC_gates=True,
 independent_batch_assumption='Eight chains treated independent and stationary; adjacent halfchain blocks may remain dependent, hence both estimators reported',
 limitations='Only16 long batches and8 independent chains; asymptotic delta diagnostics, stationarity assumed; no finite-sample bound, calibrated interval or policy superiority claim',
 CPU=[2,3],RSS_bytes=4*1024**3,wall_s=900,no_GPU=True,no_new_draws=True,no_new_physics=True)
def sha(path):
 x=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):x.update(b)
 return x.hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,'x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(np.integer,np.floating,np.bool_)):return x.item()
 return x
def prepare():
 # Hash bytes only, without interpreting metrics; binds all previous failures.
 names=['MH-target.npy','MH-updates.npz','summary.json']
 value=dict(design=DESIGN,sources={n:sha(ROOT/n) for n in SOURCES},inputs={n:sha(OLD/n) for n in names})
 write(OP/'protocol.json',value);print(sha(OP/'protocol.json'))
def pre_blocks(target):
 if target.shape!=(512000,1620):raise ValueError('Exact saved target shape')
 means=[];variances=[]
 for b in range(16):
  start=b*32000;anchor=np.array(target[start]);total=np.zeros(1620);second=np.zeros(1620)
  for first in range(start,start+32000,1024):
   x=np.asarray(target[first:min(first+1024,start+32000)])-anchor
   total+=x.sum(0);second+=(x*x).sum(0)
   if isinstance(target,np.memmap):target._mmap.madvise(4)
  mu=total/32000;v=second/32000-mu*mu
  if (v<0).any():raise FloatingPointError('Negative anchored variance')
  means.append(anchor+mu);variances.append(v)
 return np.full((16,1),1/16),np.asarray(means)[:,None],np.asarray(variances)[:,None]
def compact(result):
 return {k:plain(v) for k,v in result.items() if k not in ('sixteen_half_batches','eight_chain_clusters')}
def run(digest):
 started=time.monotonic();p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest or p['design']!=DESIGN or p['sources']!={n:sha(ROOT/n) for n in SOURCES}:raise ValueError('Frozen protocol mismatch')
 if set(os.sched_getaffinity(0))!={2,3} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU2,3 only')
 for n,s in p['inputs'].items():
  if sha(OLD/n)!=s:raise ValueError('Saved input changed')
 write(OP/'STARTED',dict(pid=os.getpid(),protocol_SHA256=digest,no_retry=True))
 target=np.load(OLD/'MH-target.npy',mmap_mode='r',allow_pickle=False)
 before=h.assess(*pre_blocks(target));del target
 with np.load(OLD/'MH-updates.npz',allow_pickle=False) as z:
  after=h.assess(z['masses'],z['halves'],z['halfvariance'])
  mismatch=float(np.max(abs(after['sixteen_half_batches']['mean']-z['means'])))
  vmismatch=float(np.max(abs(after['sixteen_half_batches']['variance']-z['variance'])))
  if mismatch>1e-14 or vmismatch>1e-18:raise ValueError('Saved pooled-moment equality failed')
 old=json.loads((OLD/'summary.json').read_text())
 old_gates={k:old[k] for k in ('reference_agreement_passed','pre_split_half_passed','importance_and_split_half_passed','pre_acquisition_stable')}
 smc=old['SMC_comparisons']
 independent_smc_passed=bool(len(smc)==3 and all(r['passed'] for r in smc))
 acquisition_passed=bool(old['pre_acquisition_stable'])
 new_precision_passed=bool(np.all(before['passed']) and np.all(after['passed']))
 result=dict(completed=True,design=DESIGN,protocol_SHA256=digest,before=compact(before),after=compact(after),
  saved_mean_max_difference_m=mismatch,saved_variance_max_difference_m2=vmismatch,old_gates_unchanged=old_gates,
  all_new_precision_diagnostics_passed=new_precision_passed,
  saved_independent_SMC_all_three_passed=independent_smc_passed,
  saved_pre_acquisition_stability_passed=acquisition_passed,
  new_diagnostic_with_independent_reference_and_acquisition_passed=bool(new_precision_passed and independent_smc_passed and acquisition_passed),
  older_failed_qualification_remains_failed=True,no_policy_advantage_claim=True,
  wall_s=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 write(OP/'summary.json',result)
 write(ROOT/'reports/force-prediction-precision-v01/summary.json',result)
 print(json.dumps({k:result[k] for k in ('completed','all_new_precision_diagnostics_passed','wall_s','peak_RSS_bytes')}))
def supervise(digest):
 from experiments.supervise_surrogate_sbc_continuation import supervise as owned
 command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c','2,3',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_force_prediction_precision_v01','run','--digest',digest]
 receipt=owned(command,OP/'actual.log',OP/'receipt.json',wall_s=900,RSS_bytes=4*1024**3)
 print(json.dumps(receipt));return receipt
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','supervise']);a.add_argument('--digest');q=a.parse_args()
 if q.mode=='prepare':prepare()
 elif q.mode=='run':run(q.digest)
 else:raise SystemExit(0 if supervise(q.digest)['completed'] else 1)
