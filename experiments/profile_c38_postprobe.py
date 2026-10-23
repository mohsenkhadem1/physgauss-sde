"""Bounded cProfile on a MANUFACTURED54-coordinate postprobe case only.

No actual archive/prefix/private data inputs exist. The unchanged frozen worker
is instrumented; this does not modify or restart any current inference job.
"""
from __future__ import annotations
import argparse,cProfile,json,os,pstats,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_initial_inference as io
from experiments import campaign_c38_postprobe_inference as post
from experiments.benchmark_c38_public_prefix_inference_v02 import manufactured_inputs,coefficients,design
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c38-postprobe-profile-v01';OUT=ROOT/'runs/gaussian-campaign-v01/c38-postprobe-profile-v01'
SOURCE='experiments/profile_c38_postprobe.py';CAP=dict(cores=[30,31],threads=2,wall_s=180,RSS_bytes=2*1024**3,no_GPU=True,cleanup_s=3)


def sources():return sorted(set(io.source_names())|{SOURCE,'experiments/campaign_c38_postprobe_inference.py','experiments/benchmark_c38_public_prefix_inference_v02.py'})
def prepare():
 p=OP/'protocol.json'
 if p.exists() or OUT.exists():raise FileExistsError('Preserve all prior profile phases')
 OP.mkdir(parents=True,exist_ok=True);r=dict(schema='C38-manufactured-postprobe-cProfile-v01',source_sha256={n:io.sha(ROOT/n)for n in sources()},runtime=io.runtime(),resources=CAP,manufactured_case='Known polynomial syntheticnonlinear48table +syntheticprefix +ONEsynthetic−2N probe54coordinates; deterministicfakeerrors; no actualdraws/archives',worker_unchanged=True,actual_data_read=False,goal='Instrument full3methods/63moments/27coordinateintervals without changing numerical gates; source-only future optimization evidence')
 io.write_new(p,r);return io.sha(p)


def validate(digest):
 p=io.read_json(ROOT,dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest))
 if p.get('source_sha256')!={n:io.sha(ROOT/n)for n in sources()} or p.get('runtime')!=io.runtime() or p.get('resources')!=CAP:raise ValueError('Frozen source/runtime/resources changed')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31} or any(os.environ.get(k)!='2'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2/noGPU contract')
 return p


def _entry(key,value):
 file,line,name=key;cc,nc,tt,ct,callers=value
 try:file=Path(file).resolve().relative_to(ROOT).as_posix()
 except ValueError:file=str(file)
 return dict(file=file,line=line,function=name,primitive_calls=cc,total_calls=nc,self_s=tt,cumulative_s=ct)


class StopProfile(BaseException):pass
def worker(digest):
 validate(digest);OUT.mkdir(parents=True,exist_ok=False)
 # Enforce no ndarray archive reading even accidentally in this synthetic case.
 np.load=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO ACTUAL ARRAY FILE INPUTS'))
 models,public=manufactured_inputs();a=coefficients('both_ends',-2.,True);x=.7;known=post.POINTS+a[0]+x*(a[1]+x*a[2]);fake_error=.03*.0002*np.cos(np.arange(27)).reshape(9,3)
 probe=dict(information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_PROBE',object_id=public['object_id'],observation_kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',acquisition_index=1,force_Fz_N=-2.,qualified_point_ids=post.POINT_IDS.copy(),reference_points_m=post.POINTS[post.POINT_IDS].copy(),observed_positions_m=known[post.POINT_IDS]+fake_error,sensor_covariance_m2=np.eye(27)*.0002**2,reset_from_common_rest=True,is_RGBD_perception=False,is_temporal_forecast=False)
 profile=cProfile.Profile();r=None;error=None;t=time.monotonic()
 def terminate(*args):raise StopProfile('Owned bounded profiling termination; incomplete profile retained')
 signal.signal(signal.SIGTERM,terminate)
 try:profile.enable();r=post.infer_public_probe(models['nonlinear'],public,probe,design=design())
 except BaseException as e:error=repr(e)
 finally:
  profile.disable();elapsed=time.monotonic()-t;profile.dump_stats(OUT/'profile.pstats');stats=pstats.Stats(profile).stats
  wanted={'leggauss','roots','intervals','rational','integrate_x','coordinate_cdf','coordinate_quantile','bounded_quantile','target_moments','_get_interval'}
  summary=dict(completed=r is not None and r.get('completed')is True,numerical_controls_passed=r is not None and r.get('numerical_controls_passed')is True,error=error,elapsed_s=elapsed,peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,actual_data_read=False,source_worker_sha256=io.sha(ROOT/'experiments/campaign_c38_postprobe_inference.py'),comparisons=None if r is None else r.get('comparisons'),failures=None if r is None else r.get('failures'),top_cumulative=[_entry(k,v)for k,v in sorted(stats.items(),key=lambda kv:kv[1][3],reverse=True)[:50]],top_self=[_entry(k,v)for k,v in sorted(stats.items(),key=lambda kv:kv[1][2],reverse=True)[:40]],selected_functions=[_entry(k,v)for k,v in sorted(stats.items(),key=lambda kv:kv[1][3],reverse=True)if k[2]in wanted])
  io.write_new(OP/'result-summary.json',io.jsonable(summary))
  if r is not None:io.write_new(OUT/'manufactured-result.json',io.jsonable(r))
 return summary


def run(digest):
 validate(digest);start=time.monotonic();state=dict(completed=False,protocol_sha256=digest,pid=os.getpid(),actual_data_read=False);proc=None;log=None
 def stop(*args):raise StopProfile('Owned supervisor termination')
 old={s:signal.signal(s,stop)for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]}
 try:
  log=(OP/'worker.log').open('xb');proc=subprocess.Popen([sys.executable,'-m','experiments.profile_c38_postprobe','worker','--sha',digest],cwd=ROOT,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state.update(worker_pid=proc.pid,PGID=proc.pid);io.atomic(OP/'execution.json',state)
  while proc.poll() is None:
   if time.monotonic()-start>CAP['wall_s']:raise TimeoutError('180s profile wall cap')
   try:
    status=Path(f'/proc/{proc.pid}/status').read_text();rss=next((int(x.split()[1])*1024 for x in status.splitlines()if x.startswith('VmRSS:')),0)
   except FileNotFoundError:rss=0
   if rss>CAP['RSS_bytes']:raise MemoryError('2GiB profile RSS cap')
   time.sleep(.2)
  code=proc.wait();state['exit_code']=code
  if code!=0:raise RuntimeError('Manufactured profile worker failed')
  summary=json.loads((OP/'result-summary.json').read_text());state.update(completed=summary['completed'],numerical_controls_passed=summary['numerical_controls_passed'],summary=io.record(ROOT,OP/'result-summary.json'))
 except BaseException as e:state['error']=repr(e)
 finally:
  if proc is not None:state['exit_code']=io.cleanup_owned(proc,CAP['cleanup_s']);state['owned_worker_reaped']=True
  if log is not None:log.close()
  state['elapsed_s']=time.monotonic()-start;io.atomic(OP/'execution.json',state)
  for s,h in old.items():signal.signal(s,h)
 return state


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker']);a.add_argument('--sha');v=a.parse_args()
 if v.mode=='prepare':print(prepare())
 elif v.mode=='worker':s=worker(v.sha);print(json.dumps(dict(completed=s['completed'],numerical_controls_passed=s['numerical_controls_passed'])));sys.exit(0 if s['completed'] else 1)
 else:s=run(v.sha);print(json.dumps(s));sys.exit(0 if s['completed']else 1)
