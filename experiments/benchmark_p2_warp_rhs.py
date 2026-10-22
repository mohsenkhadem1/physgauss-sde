"""Bounded GPU1 same-state parity/timing; never changes trajectory backend."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_contact_dop853 import ROOT,GPU,sha,write
OP=ROOT/'runs/operator-dev/p2-warp-rhs-v01';REFERENCE=ROOT/'runs/gaussian-campaign-v01/p2-contact-dop853-v03';ENV_SITE=ROOT/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'
SOURCES=['experiments/campaign_p2_warp_rhs.py','tests/test_campaign_p2_warp_rhs.py','experiments/benchmark_p2_warp_rhs.py','tests/test_benchmark_p2_warp_rhs.py']

def compare(reference,actual,M):
 d=actual['gradient']-reference['gradient'];da=actual['acceleration']-reference['acceleration']
 r=dict(force_difference_N=float(np.linalg.norm(d)),energy_difference_J=abs(actual['energy']-reference['energy']),minimum_J_difference=abs(actual['minimum_J']-reference['minimum_J']),mass_weighted_acceleration_difference_N=float(np.linalg.norm(M@da)),maximum_acceleration_difference_m_s2=float(np.linalg.norm(da,axis=1).max()),maximum_masssolve_R_N=max(reference['masssolve_R_N'],actual['masssolve_R_N']))
 r['passed']=r['force_difference_N']<=1e-10 and r['energy_difference_J']<=1e-12 and r['minimum_J_difference']<=2e-14 and r['mass_weighted_acceleration_difference_N']<=1e-10 and r['maximum_masssolve_R_N']<=1e-8
 return r

def prepare():
 p=OP/'gpu-protocol.json'
 if p.exists():raise FileExistsError('Frozen benchmark')
 cp=REFERENCE/'accuracy3.json';summary=REFERENCE/'summary.json';execution=ROOT/'runs/operator-dev/p2-contact-dop853-v03/execution.json';case=json.loads(cp.read_text());s=json.loads(summary.read_text());e=json.loads(execution.read_text());capture=ROOT/case['capture']['path'];prior=ROOT/'runs/operator-dev/p2-contact-dop853-v03/protocol.json';pins=json.loads(prior.read_text())['source_sha256']
 if not s.get('completed') or not s.get('time_reference_qualified') or not e.get('terminal') or not e.get('completed') or sha(capture)!=case['capture']['sha256']:raise ValueError('Completed qualified reference required')
 for key in ('worker_pid','controller_pid'):
  try:os.kill(e[key],0)
  except ProcessLookupError:pass
  else:raise ValueError('Prior process still exists')
 installed=['warp/__init__.py','warp/_src/context.py','warp/_src/types.py','warp/_src/codegen.py','warp/config.py']
 p_data=dict(schema='Same-state-Warp-Torch-FP64-P2-RHS-parity-timing-v0.1',status='FROZEN_BEFORE_GPU_CALLS',GPU_UUID=GPU,wall_cap_s=180,CPU_threads=2,source_sha256={f:sha(ROOT/f) for f in SOURCES},physical_source_sha256=pins,input_sha256={str(f.relative_to(ROOT)):sha(f) for f in [cp,summary,execution,capture,prior]},installed_warp_source_sha256={f:sha(ENV_SITE/f) for f in installed},selection='All9savedcommonoutputs plus acceptedstate nearestfixedt=.102s (stable earliesttie)',gates=dict(force_L2_N=1e-10,energy_J=1e-12,minimum_J_absolute=2e-14,mass_weighted_acceleration_L2_N=1e-10,masssolve_R_N=1e-8),repeated_state_evaluations=4,timing=dict(rounds=5,repetitions_per_state=4,alternate_backend_order=True,exclude_and_report_warmJIT=True,include_transfers_and_all_output_telemetry=True),mass='Reuse identical originalTorch fullM GPUCholesky factor forbothbackends; no mass approximation',nonclaims=['No trajectory backend change','No speed claim unless same-state parity and repeated-state envelope pass','Runtime bridge appends isolatedprefix only in freshchild sys.path; no installedpackage/venvchanges'])
 write(p,p_data);return sha(p)

def run(digest):
 if sha(OP/'gpu-protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Changed protocol/GPU')
 p=json.loads((OP/'gpu-protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('physical_source_sha256',ROOT),('input_sha256',ROOT),('installed_warp_source_sha256',ENV_SITE)]:
  if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changed source/input '+group)
 result=dict(completed=False,passed=False,pid=os.getpid(),protocol_sha256=digest);started=time.monotonic()
 try:
  import torch
  torch.set_num_threads(2);torch.ones(1,device='cuda:0').sum().item()
  own=[s for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True).splitlines() if s.split(',')[0].strip()==str(os.getpid())]
  if not own or any(s.split(',')[1].strip()!=GPU for s in own):raise ValueError('Wrong physical GPU')
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0' or not Path(wp.__file__).resolve().is_relative_to(ENV_SITE):raise ValueError('Wrong pinned Warp')
  wp.config.kernel_cache_dir=str(OP/'warp-cache');from experiments.campaign_p2_warp_rhs import WarpP2Force
  from experiments.campaign_p2_torch_contact import PlaneContactP2
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.run_p2_contact_validation import build
  m=build([4,2,2],300000.,9.81);t=PlaneContactP2(m,'cuda:0',kappa_A=1e9,surface_order=8);a={k:getattr(t,k).cpu().numpy().copy() for k in WarpP2Force.REQUIRED};w=WarpP2Force(a,young=m.young,poisson=m.poisson,kappa_A=1e9,device='cuda:0');Mnp=scalar_mass(m).toarray();M=torch.tensor(Mnp,dtype=torch.float64,device='cuda:0');L=torch.linalg.cholesky(M)
  cp=json.loads((REFERENCE/'accuracy3.json').read_text());z=np.load(ROOT/cp['capture']['path'],allow_pickle=False);states=[q-z['rest_m'] for q in z['positions_m']];j=int(np.argmin(abs(z['accepted_times_s']-.102)));states.append(z['accepted_states'][:3*len(m.rest),j].reshape(-1,3));selected=list(map(float,z['times_s']))+[float(z['accepted_times_s'][j])]
  def full(cache):
   g=torch.as_tensor(cache['gradient'],dtype=torch.float64,device='cuda:0');net=t.force-g;acc=torch.cholesky_solve(net,L);R=float(torch.linalg.vector_norm(M@acc-net));return dict(energy=float(cache['energy']),gradient=g.cpu().numpy().copy(),minimum_J=float(cache['minimum_J']),acceleration=acc.cpu().numpy().copy(),masssolve_R_N=R)
  def original(u):return full(t.evaluate(torch.tensor(u,dtype=torch.float64,device='cuda:0')))
  def fused(u):return full(w.evaluate(u))
  before=time.monotonic();fused(states[0]);torch.cuda.synchronize();warm=time.monotonic()-before;original(states[0]);parity=[];envelopes=[];activity=[]
  for i,u in enumerate(states):
   ref=original(u);actual=fused(u);parity.append(dict(state=i,time_s=selected[i],**compare(ref,actual,Mnp)));c=t.evaluate(torch.tensor(u,dtype=torch.float64,device='cuda:0'));activity.append(int(c['contact_active'].sum()))
   for rep in range(p['repeated_state_evaluations']):envelopes.append(dict(state=i,repeat=rep,**compare(actual,fused(u),Mnp)))
  if not any(activity) or not any(v==0 for v in activity):raise ValueError('Both active and inactive contact states required')
  timings=[]
  for r in range(p['timing']['rounds']):
   order=['torch','warp'] if r%2==0 else ['warp','torch'];row={'round':r,'order':order}
   for name in order:
    fn=original if name=='torch' else fused;torch.cuda.synchronize();now=time.perf_counter()
    for _ in range(p['timing']['repetitions_per_state']):
     for u in states:fn(u)
    torch.cuda.synchronize();row[name+'_seconds_per_evaluation']=(time.perf_counter()-now)/(len(states)*p['timing']['repetitions_per_state'])
   timings.append(row)
  med={name:float(np.median([row[name+'_seconds_per_evaluation'] for row in timings])) for name in ['torch','warp']};passed=all(v['passed'] for v in parity+envelopes)
  result.update(completed=True,passed=passed,versions=dict(torch=torch.__version__,warp=wp.__version__,numpy=np.__version__),own_GPU_records=own,states_times_s=selected,active_quadrature_counts=activity,parity=parity,repeat_envelopes=envelopes,warp_initial_JIT_and_first_evaluation_s=warm,timings=timings,median_seconds_per_evaluation=med,inclusive_speed_ratio=med['torch']/med['warp'] if passed else None,maximum_torch_allocated_bytes=torch.cuda.max_memory_allocated(),trajectory_backend_qualified=False)
 except Exception as exc:result.update(error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
 finally:result['elapsed_s']=time.monotonic()-started;write(OP/'gpu-result.json',result)
 return result

def launch(digest):
 if sha(OP/'gpu-protocol.json')!=digest or (OP/'gpu-execution.json').exists():raise ValueError('Changed/reused protocol')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(s.split(',')[1].strip()==GPU for s in apps.splitlines() if ',' in s):raise ValueError('GPU1 notidle')
 e=dict(terminal=False,completed=False,controller_pid=os.getpid(),protocol_sha256=digest,wall_cap_s=180);write(OP/'gpu-execution.json',e);start=time.monotonic()
 with (OP/'gpu-worker.log').open('x') as f:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.benchmark_p2_warp_rhs','run','--sha',digest],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});e['worker_pid']=child.pid;write(OP/'gpu-execution.json',e)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;e['timeout']=True
 r=json.loads((OP/'gpu-result.json').read_text()) if (OP/'gpu-result.json').exists() else {};e.update(terminal=True,completed=code==0 and r.get('completed') is True,passed=r.get('passed',False),elapsed_s=time.monotonic()-start,exit_code=code);write(OP/'gpu-execution.json',e);return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
