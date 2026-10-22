"""New FP64 two-level solver controls,12k parity, then fixed49k bending field."""
import argparse,json,os,signal,subprocess,time,multiprocessing,shutil,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
from experiments.run_static_p2_axial_continuation import watchdog
OP=ROOT/'runs/operator-dev/p2-two-level-static-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-two-level-static-v01'

def original_physics(diag):
 return bool(diag['original_free_residual_N']<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and (diag['global_affine_F_positive_J_sufficient_lower_bound'] or 0)>0)
def disk():
 if shutil.disk_usage(ROOT).free<30*1024**3:raise RuntimeError('Frozen30GiBdiskreserve')
def prepare():
 prior=ROOT/'runs/gaussian-campaign-v01/p2-varied-static-v01/case0.json';old=json.loads(prior.read_text());r12=next(r for r in old['meshes'] if r['tetrahedra']==12000);r24path=ROOT/'runs/gaussian-campaign-v01/p2-transverse-spatial-v01/tets20736.json';r24=json.loads(r24path.read_text())
 if not r12['completed'] or not r24['completed'] or not r24['physics_passed']:raise ValueError('Immutable completed parity/spatial references required')
 sources=['experiments/run_p2_two_level_static.py','experiments/campaign_p2_coarse_preconditioner.py','experiments/campaign_p2_block_preconditioner.py','experiments/campaign_p2_torch.py','experiments/campaign_p2_consistent_transient.py','experiments/campaign_block_surface_controls.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py','experiments/run_static_p2_axial_continuation.py']
 p=dict(schema='C38-P2-two-level-static-v0.1',status='FROZEN_BEFORE_GPUCONTROLS_AND_PHYSICS',source_sha256={f:sha(ROOT/f) for f in sources},old_case0=dict(path=str(prior.relative_to(ROOT)),sha256=sha(prior)),reference12=r12['capture'],reference24_result=dict(path=str(r24path.relative_to(ROOT)),sha256=sha(r24path)),reference24=r24['capture'],old_fields=[r['capture'] for r in old['meshes']],jobs=[dict(cells=[20,10,10],tets=12000,role='SAME_DISCRETE_MODEL_PARITY'),dict(cells=[32,16,16],tets=49152,role='NEW_FINE_SPATIAL_FIELD')],all_from_REST=True,physics=dict(E_Pa=4500,nu=.35,rho=625,gravity=0,full_left_clamp=True,force_pattern='transverse_end',force_N=[0,0,-.12],quadrature=3),preconditioner='FP64 additiveexactnodalblock+coarse4x2x2restP2(K0+mfMc); originalfullFP64operator',CG_cap=4000,nonlinear_cap=25,GPU=0,CPU_threads=2,hard_cap_s=900,GPU_allocated_guard_bytes=8*1024**3,actual_RSS_guard_bytes=12*1024**3,minimum_disk_free_bytes=30*1024**3,gates=dict(original_R_N=1e-9,global_force_N=1e-8,actual_moment_Nm=1e-8,clamp_m=0,minimum_J=.1,positive_global_J=True,parity12k_max_allnode_q_m=1e-8,coarse_P_affine_max_m=1e-12,coarse_P_constant_max=1e-11,adjoint_and_Bsymmetry_relative=1e-10,conditional_fine_indicator_m=20e-6),conditional_fine_error='3*max63point(delta24to32), derived1/(32/24-1) underassumedp>=1;notrigorousbound',conditional_resource_preemption='Afternew49152physicalgatespass, rootauthorizedresourcepreemptONLYoldincomplete49152; separateoperatorreceipt; nooldcompletedfieldsremoved',limits=['No change to old runtime or failed gates','No physicalretuning or FP32 arithmetic','Q4/time/contact/training qualification separate','Peakstrain and auxiliarybulkmetrics reportedwithoutdiscardingedgepoints'])
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenprotocol')
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedprotocol/GPU')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changedsource '+f)
 for r in [p['old_case0'],p['reference12'],p['reference24_result'],p['reference24']]+p['old_fields']:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changedcompletedreference')
 disk();OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,controls=[],cases=[],parity_passed=False,physical_fine_passed=False,spatial_passed=False,quadrature_or_dynamic_or_training_qualified=False,old_failures_preserved=True)
 def save():s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 def interrupt(sig,frame):raise TimeoutError('Frozen900s/12GiBRSSresourceguard')
 signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),begin+900,12*1024**3,OUT/'resource-guard.json'),daemon=True);guard.start()
 try:
  import torch
  from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_block_surface_controls import surface_control
  torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction((8*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
  class Budget(TwoLevelP2):
   def check(self):
    if torch.cuda.max_memory_allocated()>8*1024**3:raise MemoryError('Frozen8GiBGPUallocated')
   def tangent(self,cache,v):
    a=super().tangent(cache,v);self.check();return a
  with np.load(ROOT/p['reference12']['path'],allow_pickle=False) as a:points=a['material_probes_m'].copy();reference=a['positions_m'].copy()
  fields={}
  for job in p['jobs']:
   disk();start=time.monotonic();row=dict(job,completed=False);t=None
   try:
    m=StableP2Block(tuple(job['cells']),young=4500,poisson=.35,density=625,gravity=0,traction=1,order=3);m.load,control=surface_control(m.rest,m.tets,m.faces,'transverse_end',m.elements);t=Budget(m,'cuda:0',4000);t.check()
    if job['tets']==12000:
     rng=np.random.default_rng(2026102201);r=torch.tensor(rng.normal(size=m.rest.shape),dtype=torch.float64,device=t.device);z=torch.tensor(rng.normal(size=m.rest.shape),dtype=torch.float64,device=t.device);c=torch.tensor(rng.normal(size=t.coarse_free.numel()),dtype=torch.float64,device=t.device);constant=float(abs(torch.sparse.mm(t.P,torch.ones((len(t.coarse_rest),1),dtype=t.dtype,device=t.device))-1).max());affine=float(abs(torch.sparse.mm(t.P,t.coarse_rest)-t.rest).max());adj=float(abs((r*t.prolong(c)).sum()-(t.restrict(r)*c).sum()));adjscale=max(1.,abs(float((r*t.prolong(c)).sum())));cache=t.evaluate(torch.zeros_like(t.rest));B=t.make_preconditioner(cache,0.);Br=B(r);Bz=B(z);positive=float((r*Br).sum());sym=float(abs((r*Bz).sum()-(z*Br).sum()));symscale=max(1.,abs(float((r*Bz).sum())));passed=constant<=1e-11 and affine<=1e-12 and adj<=1e-10*adjscale and positive>0 and sym<=1e-10*symscale
     s['controls'].append(dict(name='GPU_exactP_projectedPT_additiveSPD',passed=passed,constant_error=constant,affine_error_m=affine,adjoint_error=adj,adjoint_scale=adjscale,positive_quadratic_form=positive,symmetry_error=sym,symmetry_scale=symscale));save()
     if not passed:raise RuntimeError('CoarseGPUcontrolfailed')
     del r,z,c,cache,B,Br,Bz
    answer=t.solve();q=answer['positions'].cpu().numpy();residual=(answer['cache']['gradient']-t.force).cpu().numpy();diag=m.diagnostics(dict(positions=q,residual=residual,J=answer['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=answer['iterations']));physical=original_physics(diag);queries=m.query(q,points);f=OUT/f'transverse-tets{job["tets"]}.npz';disk();np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=queries)
    row.update(completed=True,physics_passed=physical,diagnostics=diag,history=answer['history'],linear_records=t.linear_records,known_control=control,coarse_symmetry_defect_N_m=t.coarse_symmetry_defect_N_m,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)));fields[job['tets']]=queries
    if job['tets']==12000:
     delta=float(np.linalg.norm(q-reference,axis=1).max());s['parity_allnode_max_q_m']=delta;s['parity_passed']=physical and delta<=1e-8
     if not s['parity_passed']:raise RuntimeError('12kfromRESToriginalfieldparityfailed')
    else:s['physical_fine_passed']=physical
    if not physical:raise RuntimeError('Originalphysicalgatesfailed')
   except Exception as e:row.update(error=repr(e),traceback=traceback.format_exc(),linear_records=getattr(t,'linear_records',[]));raise
   finally:row['elapsed_s']=time.monotonic()-start;s['cases'].append(row);write(OUT/f'tets{job["tets"]}.json',row);save()
   del answer,t,m;torch.cuda.empty_cache()
  allfields={}
  for nx,r in zip([16,20],p['old_fields']):
   with np.load(ROOT/r['path'],allow_pickle=False) as a:allfields[nx]=a['probe_positions_m'].copy()
  with np.load(ROOT/p['reference24']['path'],allow_pickle=False) as a:allfields[24]=a['probe_positions_m'].copy()
  allfields[32]=fields[49152];levels=[16,20,24,32];delta=[float(np.linalg.norm(allfields[b]-allfields[a],axis=-1).max()) for a,b in zip(levels[:-1],levels[1:])];fine=3*delta[-1];s.update(completed=True,all_actual_nx=levels,all_adjacent_max63probe_error_m=delta,conditional_fine_indicator_m=fine,finest_multiplier=3.,spatial_passed=fine<=20e-6,passed=s['physical_fine_passed'] and s['parity_passed'] and fine<=20e-6)
 except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:save();guard.terminate();guard.join(timeout=2)
 return s
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':
  s=run(a.sha);print(json.dumps({k:s.get(k) for k in ['completed','passed','elapsed_s','parity_passed','physical_fine_passed','conditional_fine_indicator_m','error']}));raise SystemExit(0 if s['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
  if sha(OP/'protocol.json')!=a.sha:raise ValueError('Changedprotocol')
  disk();r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=900);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_two_level_static','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
