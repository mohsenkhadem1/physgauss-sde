"""Paired manufactured54-coordinate OLD/plain vs NEW/exact-memo benchmark.

Zero actual-data inputs; no profiler overhead. Both complete numerical results
and every root bracket must match bit-for-bit, excluding only cache counters.
"""
import argparse,json,os,signal,subprocess,sys,time,resource
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_initial_inference as io
from experiments import campaign_c38_postprobe_inference as old
from experiments import campaign_c38_postprobe_memo as new
from experiments.benchmark_c38_public_prefix_inference_v02 import manufactured_inputs,coefficients,design
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c38-postprobe-memo-v01';OUT=ROOT/'runs/gaussian-campaign-v01/c38-postprobe-memo-v01';SOURCE='experiments/benchmark_c38_postprobe_memo.py'
CAP=dict(cores=[30,31],threads=2,wall_s=300,RSS_bytes=2*1024**3,no_GPU=True,cleanup_s=3)


def sources():return sorted(set(io.source_names())|{SOURCE,'experiments/campaign_c38_postprobe_inference.py','experiments/campaign_c38_postprobe_memo.py','experiments/campaign_c38_exact_posterior_memo.py','tests/test_campaign_c38_exact_posterior_memo.py','experiments/benchmark_c38_public_prefix_inference_v02.py'})
def prepare():
 if (OP/'protocol.json').exists() or OUT.exists():raise FileExistsError('Fresh benchmark only')
 OP.mkdir(parents=True,exist_ok=True);p=dict(schema='C38-exact-memo-manufactured-pair-v01',resources=CAP,source_sha256={n:io.sha(ROOT/n)for n in sources()},runtime=io.runtime(),methods=['old_plain','new_exact_memo'],comparison='All complete numericoutputs/rootbrackets bit-identical; ONLY topcache_stats counters excluded',profiling_enabled=False,actual_data_read=False,order_scope='Fixed old-then-new in onefreshprocess; sharedgenericlibrarywarming disclosed; no repeated-trial speed confidence claim')
 io.write_new(OP/'protocol.json',p);return io.sha(OP/'protocol.json')
def validate(digest):
 p=io.read_json(ROOT,dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest))
 if p.get('source_sha256')!={n:io.sha(ROOT/n)for n in sources()} or p.get('resources')!=CAP or p.get('runtime')!=io.runtime():raise ValueError('Changed frozen source/runtime/caps')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31} or any(os.environ.get(k)!='2'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2/noGPU')
 return p


def compare(a,b,path='root'):
 if path=='root':a={k:v for k,v in a.items()if k!='cache_stats'};b={k:v for k,v in b.items()if k!='cache_stats'}
 if isinstance(a,np.ndarray)or isinstance(b,np.ndarray):
  return [] if isinstance(a,np.ndarray)and isinstance(b,np.ndarray)and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes() else [path]
 if isinstance(a,np.generic):a=a.item()
 if isinstance(b,np.generic):b=b.item()
 if isinstance(a,dict)and isinstance(b,dict):
  if set(a)!=set(b):return [path+'/KEYS']
  return [d for k in a for d in compare(a[k],b[k],path+'/'+str(k))]
 if isinstance(a,(tuple,list))and isinstance(b,(tuple,list)):
  if len(a)!=len(b):return [path+'/LENGTH']
  return [d for i,(x,y)in enumerate(zip(a,b))for d in compare(x,y,path+'/'+str(i))]
 if isinstance(a,float)and isinstance(b,float):return []if a.hex()==b.hex()else[path]
 return []if type(a)is type(b)and a==b else[path]


class Stop(BaseException):pass
def worker(digest):
 validate(digest);OUT.mkdir(parents=True,exist_ok=False);np.load=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No actual array inputs'))
 models,public=manufactured_inputs();a=coefficients('both_ends',-2.,True);x=.7;q=old.POINTS+a[0]+x*(a[1]+x*a[2]);probe=dict(information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_PROBE',object_id=public['object_id'],observation_kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',acquisition_index=1,force_Fz_N=-2.,qualified_point_ids=old.POINT_IDS.copy(),reference_points_m=old.POINTS[old.POINT_IDS].copy(),observed_positions_m=q[old.POINT_IDS]+.03*.0002*np.cos(np.arange(27)).reshape(9,3),sensor_covariance_m2=np.eye(27)*.0002**2,reset_from_common_rest=True,is_RGBD_perception=False,is_temporal_forecast=False)
 results={};summary=dict(completed=False,bit_identical=False,actual_data_read=False,methods={})
 signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(Stop('Owned bounded benchmark termination')))
 try:
  for name,module in [('old_plain',old),('new_exact_memo',new)]:
   t=time.monotonic();r=module.infer_public_probe(models['nonlinear'],public,probe,design=design());results[name]=r;summary['methods'][name]=dict(completed=r['completed'],numerical_controls_passed=r['numerical_controls_passed'],failures=r['failures'],comparisons=r.get('comparisons'),elapsed_s=time.monotonic()-t,process_peak_RSS_bytes_cumulative=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,cache_stats=r.get('cache_stats'));io.write_new(OUT/(name+'.json'),io.jsonable(r));io.atomic(OP/'result-summary.json',io.jsonable(summary))
  differences=compare(results['old_plain'],results['new_exact_memo']);summary.update(completed=True,bit_identical=not differences,all_numerical_controls_passed=all(r['completed']and r['numerical_controls_passed']for r in results.values()),mismatched_paths= differences[:30],mismatched_path_count=len(differences),new_over_old_time=summary['methods']['new_exact_memo']['elapsed_s']/summary['methods']['old_plain']['elapsed_s'])
 except BaseException as error:summary['error']=repr(error)
 finally:io.atomic(OP/'result-summary.json',io.jsonable(summary))
 return summary


def run(digest):
 validate(digest);t=time.monotonic();s=dict(completed=False,pid=os.getpid(),protocol_sha256=digest,actual_data_read=False);proc=None;log=None
 def stop(*a):raise Stop('Owned controller stopped')
 previous={n:signal.signal(n,stop)for n in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]}
 try:
  log=(OP/'worker.log').open('xb');proc=subprocess.Popen([sys.executable,'-m','experiments.benchmark_c38_postprobe_memo','worker','--sha',digest],cwd=ROOT,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);s.update(worker_pid=proc.pid,PGID=proc.pid);io.atomic(OP/'execution.json',s)
  while proc.poll()is None:
   if time.monotonic()-t>CAP['wall_s']:raise TimeoutError('300s global benchmark cap')
   try:rss=next((int(x.split()[1])*1024 for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines()if x.startswith('VmRSS:')),0)
   except FileNotFoundError:rss=0
   if rss>CAP['RSS_bytes']:raise MemoryError('2GiB RSS cap')
   time.sleep(.2)
  code=proc.wait();s['exit_code']=code
  if code!=0:raise RuntimeError('Benchmark worker failed')
  r=json.loads((OP/'result-summary.json').read_text());s.update(completed=r['completed'],bit_identical=r['bit_identical'],all_numerical_controls_passed=r.get('all_numerical_controls_passed',False),summary=io.record(ROOT,OP/'result-summary.json'))
 except BaseException as e:s['error']=repr(e)
 finally:
  if proc is not None:s['exit_code']=io.cleanup_owned(proc,3);s['owned_worker_reaped']=True
  if log is not None:log.close()
  s['elapsed_s']=time.monotonic()-t;io.atomic(OP/'execution.json',s)
  for n,h in previous.items():signal.signal(n,h)
 return s


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker','run']);a.add_argument('--sha');v=a.parse_args()
 if v.mode=='prepare':print(prepare())
 elif v.mode=='worker':r=worker(v.sha);print(json.dumps(dict(completed=r['completed'],bit_identical=r['bit_identical'])));sys.exit(0 if r['completed'] else 1)
 else:r=run(v.sha);print(json.dumps(r));sys.exit(0 if r['completed'] else 1)
