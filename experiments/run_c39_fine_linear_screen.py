"""Prospective two-support graded32/40 linear down/side confirmation only."""
import argparse,ast,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_c39_coarse_linear_screen import ROOT,sha,write,fisher,POINT_IDS
OP=ROOT/'runs/operator-dev/c39-fine-linear-screen-v01';OUT=ROOT/'runs/gaussian-campaign-v01/c39-fine-linear-screen-v01';SOURCE='experiments/run_c39_fine_linear_screen.py';TEST='tests/test_run_c39_fine_linear_screen.py'
JOBS=[dict(support='both_ends',GPU_uuid='GPU-5b432ce5-a24a-68ba-150f-682265260a9b',cores=[0,1]),dict(support='separated_bottom_patches',GPU_uuid='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb',cores=[2,3])];MESHES=((32,16,16),(40,20,20));ACTIONS=('gravity','middle_down_2N','middle_side_2N')

def closure():
 seen=set();pending=[SOURCE,TEST]
 while pending:
  f=pending.pop()
  if f in seen:continue
  seen.add(f)
  for n in ast.walk(ast.parse((ROOT/f).read_text())):
   if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments.'):
    dep='experiments/'+n.module.split('.')[1]+'.py'
    if (ROOT/dep).exists():pending.append(dep)
 return sorted(seen)
def canonical(gravity,down):return 1.5*gravity-.5*down

def prepare():
 from experiments.campaign_owned_gpu_budget_v02 import LIMITS
 coarse=ROOT/'runs/operator-dev/c39-coarse-linear-screen-v01/result.json';c=json.loads(coarse.read_text())
 if not c['completed'] or not c['promising_for_fine_linear_screen']:raise ValueError('Prospective coarse criterion prerequisite')
 old=ROOT/'runs/operator-dev/c38-static-linear-predictions-v01/protocol-v01.json';pold=json.loads(old.read_text())
 p=dict(schema='C39-linear-down-side-fine-confirmation-v01',status='FROZEN_BEFORE_NEW_LINEAR_RESPONSES',source_sha256={f:sha(ROOT/f) for f in closure()},evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [coarse,coarse.with_name('protocol.json'),old]},installed_warp_API_sha256=pold['installed_warp_API_sha256'],jobs=JOBS,meshes=[list(x) for x in MESHES],basis_actions=list(ACTIONS),canonical='u_up1=1.5*u_gravity-.5*u_middle_down2; exact same public uniform band',young_Pa=75000.,phase_cap_s=650,model_cap_s=300,global_cap_s=700,resources=LIMITS,projected_write_reserve_bytes=2*1024**3,CPU_threads=2,solver=pold['solver'],gates=dict(original_Torch_HVP_free_R_N=1e-10,exact_mask_m=0.,q63_conditional_actual_h_indicator_m=20e-6,winner_Fisher_ratio_at_least=1.05),scope='Rest-tangent model screening only; gravity/down/side and canonical ideal9top Fisher at sigma200um. No nonlinear labels/observations/draws. LEFT excluded only from this FINE design after coarse third-place result; coarse rows preserved. All4 q63 responses required for spatial gate, not combined continuum bound.',no_nonlinear_or_dataset_admission=True)
 f=OP/'protocol.json'
 if f.exists():raise FileExistsError(f)
 write(f,p);return sha(f)
def validate(digest):
 p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed protocol')
 for key in ['source_sha256','evidence_sha256']:
  if any(sha(ROOT/f)!=h for f,h in p[key].items()):raise ValueError('Changed '+key)
 site=ROOT/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'
 if any(sha(site/f)!=h for f,h in p['installed_warp_API_sha256'].items()):raise ValueError('Changed installed Warp API')
 return p

def worker(digest,support):
 p=validate(digest);job=next(j for j in JOBS if j['support']==support);out=OUT/support;out.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,passed=False,models=[],protocol_sha256=digest,pid=os.getpid())
 def stopped(*_):raise TimeoutError('Frozen model/phase resource interruption')
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped)
 try:
  import torch
  sys.path.append(str(ROOT/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'));import warp as wp
  from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
  from experiments.campaign_c38_verified_warp_solver import IndependentlyCheckedWarpCGMixin
  from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
  from experiments.campaign_c39_known_actions import compile_known_action
  from experiments.campaign_c38_table_validation import POINTS
  from experiments.campaign_owned_gpu_budget_v02 import collect_resources,validate_budget,validate_cpu
  if wp.__version__!='1.18.0' or os.environ.get('CUDA_VISIBLE_DEVICES')!=job['GPU_uuid']:raise ValueError('Frozen device/runtime')
  torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(job['cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());torch.cuda.set_per_process_memory_fraction(p['resources']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.zeros(1,device='cuda:0').sum().item();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);own=[line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
  if set(own)!={job['GPU_uuid']}:raise ValueError('Own actual GPU UUID mismatch')
  s['own_GPU_UUIDs']=own;wp.config.kernel_cache_dir=str(out/'warp-cache')
  def guard():
   r=collect_resources(gpu_uuid=job['GPU_uuid'],pid=os.getpid(),roots=[out],torch=torch);validate_budget(r,reserve_write_bytes=p['projected_write_reserve_bytes'],limits=p['resources']);write(out/'resource-latest.json',r)
  class Solver(IndependentlyCheckedWarpCGMixin,StreamingSupportedP2):
   def tangent(self,c,v):
    r=super().tangent(c,v);self.calls=getattr(self,'calls',0)+1
    if self.calls%16==0:guard()
    return r
  guard()
  for cells in MESHES:
   signal.alarm(300);clock=time.monotonic();m=SupportedBoundaryGradedGeometry(cells,75000.,support,0.,rule='keast24');op=Solver(m,'cuda:0',p['solver']['CG_cap'],p['solver']['quadrature_chunk_size'],support_kind=support);cache=op.evaluate(torch.zeros_like(op.rest));responses={};row=dict(cells=list(cells),hmax_m=m.maximum_reference_tet_edge_m,responses=[]);s['models'].append(row);guard()
   for action in ACTIONS:
    f=compile_known_action(m.rest,m.tets,m.faces,m.elements,action);rhs=torch.tensor(f['total_N'],dtype=torch.float64,device='cuda:0');u,it=op.cg(cache,rhs,0.,rtol=0.,atol=2.5e-11);applied=StreamingSupportedP2.tangent(op,cache,u);R=float(torch.linalg.vector_norm((applied-rhs).masked_fill(op.mask,0.)))
    if R>1e-10 or bool((u[op.mask]!=0).any()) or not bool(torch.isfinite(u).all()):raise ValueError('Original residual/mask/finite gate')
    responses[action]=u.detach().cpu().numpy();u63=m.query(responses[action],POINTS);row['responses'].append(dict(action=action,CG_iterations=it,original_Torch_HVP_free_R_N=R,exact_constraint_error_m=0.,logE_Fisher_9top=fisher(u63[POINT_IDS]),displacement63_m=u63.tolist(),load_metadata=f['metadata']));write(out/'result.json',s);guard()
   responses['canonical_up_1N']=canonical(responses['gravity'],responses['middle_down_2N']);u63=m.query(responses['canonical_up_1N'],POINTS);row['responses'].append(dict(action='canonical_up_1N',exact_linear_combination=True,logE_Fisher_9top=fisher(u63[POINT_IDS]),displacement63_m=u63.tolist()));row['CG_original_HVP_checks']=op.independent_linear_checks;row['seconds']=time.monotonic()-clock;rank=sorted([r for r in row['responses'] if r['action'] in ACTIONS[1:]],key=lambda r:-r['logE_Fisher_9top']);row['ranking']=[r['action'] for r in rank];row['winner_ratio']=rank[0]['logE_Fisher_9top']/rank[1]['logE_Fisher_9top'];capture=out/('mesh'+str(cells[0])+'.npz');np.savez(capture,rest_m=m.rest,material_probes_m=POINTS,**responses);row['capture']=dict(path=str(capture.relative_to(ROOT)),sha256=sha(capture));del u,applied,rhs,cache,op,m;torch.cuda.empty_cache();guard();write(out/'result.json',s);signal.alarm(0)
   if time.monotonic()-begin>650:raise TimeoutError('Phase cap')
  a,b=s['models'];factor=1/(a['hmax_m']/b['hmax_m']-1);indicators={ra['action']:factor*float(np.linalg.norm(np.array(ra['displacement63_m'])-np.array(rb['displacement63_m']),axis=1).max()) for ra,rb in zip(a['responses'],b['responses'])};s.update(completed=True,passed=True,conditional_actual_h_factor=factor,conditional_q63_indicators_m=indicators,q63_gate_passed=all(x<=20e-6 for x in indicators.values()),mesh_consistent_winner=a['ranking'][0]==b['ranking'][0],rank_margin_passed=all(r['winner_ratio']>=1.05 for r in s['models']))
 except BaseException as e:s.update(error=repr(e),traceback=traceback.format_exc())
 finally:signal.alarm(0);s['seconds']=time.monotonic()-begin;write(out/'result.json',s)
 return s

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('No retries')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(j['GPU_uuid'] in apps for j in JOBS):raise RuntimeError('Declared GPUs must be idle')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,terminal=False,controller_pid=os.getpid(),records=[],protocol_sha256=digest);active={};logs=[]
 from experiments.campaign_owned_gpu_budget_v02 import terminate_owned
 def stopped(*_):raise InterruptedError('Owned fine controller stop')
 old={sig:signal.signal(sig,stopped) for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]};signal.alarm(700)
 try:
  for j in JOBS:
   f=(OP/(j['support']+'.log')).open('x');logs.append(f);blocked=signal.pthread_sigmask(signal.SIG_BLOCK,set(old))
   try:
    child=subprocess.Popen(['/usr/bin/taskset','-c',','.join(map(str,j['cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_c39_fine_linear_screen','worker','--sha',digest,'--support',j['support']],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':j['GPU_uuid'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});row=dict(support=j['support'],pid=child.pid);s['records'].append(row);active[j['support']]=(child,row,time.monotonic())
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
   write(OP/'execution.json',s)
  while active:
   for name,(child,row,t) in list(active.items()):
    if time.monotonic()-t>650 and child.poll() is None:row['timeout']=True;terminate_owned(child)
    if child.poll() is None:continue
    row['exit_code']=child.wait();f=OUT/name/'result.json';r=json.loads(f.read_text()) if f.exists() else {};row['completed']=r.get('completed',False);row['passed']=r.get('passed',False);row['result']=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)) if f.exists() else None
    if f.exists():
     h=sha(f);cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/f'{h}.json';cache.parent.mkdir(parents=True,exist_ok=True)
     if cache.exists():assert cache.read_bytes()==f.read_bytes()
     else:cache.write_bytes(f.read_bytes())
     row['external_cache']=str(cache.relative_to(ROOT))
    del active[name];write(OP/'execution.json',s)
   if active:time.sleep(.2)
  s['completed']=len(s['records'])==2 and all(r['completed'] and r['exit_code']==0 for r in s['records'])
  if s['completed']:
   results=[json.loads((ROOT/r['result']['path']).read_text()) for r in s['records']];s.update(all_numeric_passed=all(r['passed'] for r in results),all_q63_spatial_passed=all(r['q63_gate_passed'] for r in results),fine_opposite_winner_passed=all(r['mesh_consistent_winner'] and r['rank_margin_passed'] for r in results) and results[0]['models'][0]['ranking'][0]!=results[1]['models'][0]['ranking'][0])
 except BaseException as e:s['error']=repr(e)
 finally:
  signal.alarm(0)
  for child,row,t in active.values():row.update(cleanup=True,exit_code=terminate_owned(child))
  for f in logs:f.close()
  for sig,h in old.items():signal.signal(sig,h)
  s.update(terminal=True,seconds=time.monotonic()-start,no_nonlinear_or_dataset_admission=True);write(OP/'execution.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','launch']);p.add_argument('--sha');p.add_argument('--support');a=p.parse_args();r=prepare() if a.action=='prepare' else worker(a.sha,a.support) if a.action=='worker' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','seconds','error','all_q63_spatial_passed','fine_opposite_winner_passed']}));raise SystemExit(0 if isinstance(r,str) or r['completed'] else 1)
