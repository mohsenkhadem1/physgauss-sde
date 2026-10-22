"""One fixed physical transverse-load case, prospectively finer static meshes."""
import argparse,json,os,signal,subprocess,time,traceback,multiprocessing,shutil
from pathlib import Path
import numpy as np
from experiments.run_p2_torch_benchmark import sha,write,ROOT
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-transverse-spatial-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-transverse-spatial-v01'

def indicator(delta,coarse_nx,fine_nx):
 if not np.isfinite(delta) or delta<0 or not 0<coarse_nx<fine_nx:raise ValueError('Finite nonnegative field difference and refined spacing required')
 return float(delta*coarse_nx/(fine_nx-coarse_nx))
def disk_guard():
 if shutil.disk_usage(ROOT).free<30*1024**3:raise RuntimeError('Frozen30GiBfree disk reserve breached; no deletion/retry')
def prepare():
 prior=ROOT/'runs/gaussian-campaign-v01/p2-varied-static-v01/case0.json';row=json.loads(prior.read_text())
 if row['E_Pa']!=4500 or row['support']!='left_face' or row['force_pattern']!='transverse_end' or len(row['meshes'])!=2 or not all(r['completed'] for r in row['meshes']):raise ValueError('Exact two completed unchanged original fields required')
 sources=['experiments/run_p2_transverse_spatial_continuation.py','experiments/campaign_p2_block_preconditioner.py','experiments/campaign_p2_torch.py','experiments/campaign_block_surface_controls.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py','experiments/run_static_p2_axial_continuation.py']
 p=dict(schema='C38-P2-transverse-spatial-continuation-v0.1',status='FROZEN_BEFORE_NEW_FINER_FIELDS',source_sha256={f:sha(ROOT/f) for f in sources},prior_case=dict(path=str(prior.relative_to(ROOT)),sha256=sha(prior)),prior_fields=[r['capture'] for r in row['meshes']],jobs=[dict(cells=[24,12,12],tetrahedra=20736),dict(cells=[32,16,16],tetrahedra=49152)],all_from_REST=True,order=3,physics=dict(E_Pa=4500,nu=.35,rho=625,gravity=0,BC='left_face',force_pattern='transverse_end',right_face_force_world_N=[0,0,-.12]),solver=dict(class_name='BlockJacobiP2',CG_cap=4000,nonlinear_cap=25,original_solve_R_N=1e-10,no_physical_regularization=True),GPU=2,CPU_threads=2,hard_cap_s=1800,GPU_allocated_guard_bytes=8*1024**3,actual_RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,forecast_s=[600,1200],gates=dict(original_R_N=1e-9,global_force_N=1e-8,actual_moment_Nm=1e-8,exact_clamp_m=0,minimum_J=.1,positive_global_J_certificate=True,conditional_fine_indicator_m=20e-6),indicator='delta24to32/(32/24-1)=3*delta under assumedp>=1; all63 original fixed probes; NOT rigorous continuum bound',stages='Attempt both new meshes fromREST; stop on numerical/resource failure preserving completed/current/queued status; no rerun completed fields',nonclaims=['Old6144to12000 failure preserved','QuadratureQ4 is separate prospective stage only after spatialpass','No time/contact/training/adaptiveforce claims','Report strain peak/99th/RMS/nonrigid separately; no query trimming or threshold relaxation'])
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen protocol')
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Changed protocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 for r in [p['prior_case']]+p['prior_fields']:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed prior field/result')
 disk_guard();OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,meshes=[],all_from_REST=True,prior_failure_preserved=True,spatial_qualified=False,quadrature_or_dynamic_or_training_qualified=False,planned=[dict(j,status='QUEUED') for j in p['jobs']]);fields={};levels=[]
 def save():s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
 def interrupt(sig,frame):raise TimeoutError('Frozen1800s/12GiBRSS resource guard')
 signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+1800,12*1024**3,OUT/'resource-guard.json'),daemon=True);guard.start()
 try:
  import torch
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_block_surface_controls import surface_control,support_groups
  from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
  torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction((8*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
  class BudgetBlock(BlockJacobiP2):
   def check(self):
    if torch.cuda.max_memory_allocated()>8*1024**3:raise MemoryError('Frozen8GiBGPUallocated guard')
   def evaluate(self,u):
    a=super().evaluate(u);self.check();self.evaluated_history.append(dict(original_free_R_N=float(torch.linalg.vector_norm((a['gradient']-self.force)[~self.mask])),minimum_J=float(a['minimum_J']),potential_J=float(a['energy']-(self.force*u).sum())));self.last_positive_evaluated_u=u.detach().cpu().numpy();return a
   def tangent(self,cache,v):
    a=super().tangent(cache,v);self.check();return a
  for n,r in zip([16,20],p['prior_fields']):
   with np.load(ROOT/r['path'],allow_pickle=False) as a:fields[n]=a['probe_positions_m'].copy();points=a['material_probes_m'].copy()
  for job in p['jobs']:
   disk_guard();begin=time.monotonic();slot=next(r for r in s['planned'] if r['cells']==job['cells']);slot['status']='RUNNING';save();row=dict(job,completed=False);t=None
   try:
    m=StableP2Block(tuple(job['cells']),young=4500,poisson=.35,density=625,gravity=0,traction=1,order=3);groups,active=support_groups(m.rest,'left_face');m.mask=np.repeat(groups.any(0)[:,None],3,1);m.free=np.flatnonzero(~m.mask.ravel());m.load,control=surface_control(m.rest,m.tets,m.faces,'transverse_end',m.elements);t=BudgetBlock(m,'cuda:0',4000);t.evaluated_history=[];t.check();answer=t.solve();q=answer['positions'].cpu().numpy();residual=(answer['cache']['gradient']-t.force).cpu().numpy();diag=m.diagnostics(dict(positions=q,residual=residual,J=answer['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=answer['iterations']));queries=m.query(q,points);f=OUT/f'transverse-tets{len(m.tets)}.npz';disk_guard();np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=queries)
    physical=diag['original_free_residual_N']<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and (diag['global_affine_F_positive_J_sufficient_lower_bound'] or 0)>0
    row.update(completed=True,physics_passed=physical,diagnostics=diag,history=answer['history'],linear_records=t.linear_records,known_control=control,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)));fields[job['cells'][0]]=queries;slot['status']='COMPLETED'
   except Exception as e:
    row.update(error=repr(e),traceback=traceback.format_exc(),linear_records=getattr(t,'linear_records',[]),evaluated_history=getattr(t,'evaluated_history',[]));slot['status']='FAILED'
    if t is not None and hasattr(t,'last_positive_evaluated_u'):np.savez_compressed(OUT/f'failure-tets{job["tetrahedra"]}-last-positive-evaluated.npz',u=t.last_positive_evaluated_u,semantics='last positive-J evaluated trial, NOT necessarily accepted')
    raise
   finally:row['elapsed_s']=time.monotonic()-begin;s['meshes'].append(row);write(OUT/f'tets{job["tetrahedra"]}.json',row);save()
   if not physical:raise RuntimeError('Original physics gate failed; later mesh remains queued')
   del answer,t,m;torch.cuda.empty_cache()
  levels=[16,20,24,32];differences=[float(np.linalg.norm(fields[b]-fields[a],axis=-1).max()) for a,b in zip(levels[:-1],levels[1:])];est=indicator(differences[-1],24,32);s.update(completed=True,all_actual_nx=levels,all_adjacent_max63query_difference_m=differences,all_adjacent_conditional_p1_fine_indicators_m=[indicator(d,a,b) for d,a,b in zip(differences,levels[:-1],levels[1:])],conditional_finest_indicator_m=est,finest_multiplier=3.,spatial_qualified=est<=20e-6 and all(x['physics_passed'] for x in s['meshes']),passed=est<=20e-6 and all(x['physics_passed'] for x in s['meshes']))
 except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc())
 finally:save();guard.terminate();guard.join(timeout=2)
 return s
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':
  s=run(a.sha);print(json.dumps({k:s.get(k) for k in ['completed','passed','elapsed_s','error','all_adjacent_max63query_difference_m','conditional_finest_indicator_m']}));raise SystemExit(0 if s['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
  if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changed protocol')
  old=json.loads((ROOT/'runs/operator-dev/p2-torsion-blockpcg-v01/execution.json').read_text())
  if not old['terminal'] or any(Path('/proc',str(old[k])).exists() for k in ['pid','worker_pid']):raise ValueError('GPU2 torsion stage not drained')
  disk_guard();r=dict(terminal=False,pid=os.getpid(),protocol_sha256=a.sha,hard_cap_s=1800);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_transverse_spatial_continuation','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);begin=time.monotonic()
   try:code=child.wait(timeout=1800)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
