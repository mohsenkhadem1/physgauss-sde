"""Exact-bulk quadrature, fixed both ends, true gravity: bounded spatial check."""
import argparse,json,os,signal,subprocess,time,traceback,multiprocessing,resource,shutil
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
OP=ROOT/'runs/operator-dev/p2-gravity-refinement-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-gravity-refinement-v01'
CELLS=[(12,6,6),(16,8,8),(20,10,10),(24,12,12)]
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserve frozen protocol')
 prior=ROOT/'runs/operator-dev/p2-supported-gravity-v01/protocol.json';sources=list(json.loads(prior.read_text())['source_sha256'])+['experiments/run_p2_gravity_refinement.py','experiments/campaign_p2_exact_quadrature.py','experiments/run_p2_torch_benchmark.py','experiments/run_static_p2_axial_continuation.py']
 control=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-v01/summary.json'
 if not json.loads(control.read_text())['passed']:raise ValueError('Independent four-support CPU controls required')
 p=dict(schema='C38-bothends-gravity-exactbulk-v01',status='FROZEN_BEFORE_NEW_SOLVES',source_sha256={n:sha(ROOT/n) for n in sources},CPU_control=dict(path=str(control.relative_to(ROOT)),sha256=sha(control)),cells=CELLS,coarse_CPU_GPU_parity_cells=[4,2,2],physics=dict(E_Pa=100000.,nu=.35,density_kg_m3=625.,gravity_m_s2=9.81,support='both_ends',pattern='transverse_end',surface_force_N=[0,0,-.12],mapping='uniform',note='Right-face surface force acts on fixed nodes; deformation in this check is driven by gravity'),quadrature='Positive80pointdegree6exact polynomial bulk rule; constant stableNH material',all_from_REST=True,GPU=2,CPU_threads=2,hard_cap_s=900,GPU_allocated_limit_bytes=8*1024**3,RSS_limit_bytes=12*1024**3,disk_reserve_bytes=30*1024**3,gates=dict(original_free_R_N=1e-10,force_balance_N=1e-8,moment_balance_Nm=1e-8,exact_clamp_m=0,minimum_J=.1,positive_whole_element_Bernstein=True,coarse_allnode_CPU_GPU_parity_m=1e-8,conditional_fine_spatial_indicator_m=20e-6),indicator='Max63fixedquerydelta/(actualhmaxratio-1), assumedorder>=1engineeringindicator NOT rigorousbound',limits=['DifferentphysicalexperimentfromE4500transversecantilever; oldfailurespreserved','No dynamics/materialpopulation/posterior/imagecohort admission','Peakstrain and bulk nonrigid motion reported; no convergedpeakstress claim'])
 write(OP/'protocol.json',p);print(sha(OP/'protocol.json'))
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Protocol/device mismatch')
 p=json.loads((OP/'protocol.json').read_text())
 for n,h in p['source_sha256'].items():
  if sha(ROOT/n)!=h:raise ValueError('Changed source '+n)
 if sha(ROOT/p['CPU_control']['path'])!=p['CPU_control']['sha256']:raise ValueError('Changed independent control')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,meshes=[],protocol_sha256=digest,dynamic_or_dataset_admission=False)
 def save():s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
 def interrupt(*_):raise TimeoutError('Frozen900s/12GiB guard')
 signal.signal(signal.SIGTERM,interrupt)
 from experiments.run_static_p2_axial_continuation import watchdog
 guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+900,12*1024**3,OUT/'resource-guard.json'),daemon=True);guard.start()
 try:
  import torch
  from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
  from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
  torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(p['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
  def check():
   if torch.cuda.max_memory_allocated()>p['GPU_allocated_limit_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['RSS_limit_bytes']:raise MemoryError('Frozen resource guard')
   if shutil.disk_usage(ROOT).free<p['disk_reserve_bytes']:raise OSError('Frozen disk reserve')
  def geometry(cells):return ExactBulkP2Geometry(tuple(cells),young=100000.,gravity=9.81,support='both_ends',mapping='uniform')
  class Budget(StreamingSupportedP2):
   def evaluate(self,u):
    c=super().evaluate(u);check();return c
   def tangent(self,c,v):
    out=super().tangent(c,v);check();return out
  # Independent sparse CPU exact80 check precedes fine GPU fields.
  m=geometry((4,2,2));cpu=StableP2Block((4,2,2),young=100000.,poisson=.35,density=625.,gravity=0.,traction=0.,order=3)
  np.testing.assert_array_equal(cpu.rest,m.rest);cpu.bary=m.bary.copy();cpu.weights=m.weights.copy();cpu.N=m.N.copy();cpu.load=m.load.copy();cpu.mask=m.mask.copy();cpu.free=np.flatnonzero(~cpu.mask.ravel())
  a=cpu.solve(force_tolerance_N=1e-10);t=Budget(m,'cuda:0',4000,8,support_kind='both_ends');b=t.solve();delta=float(np.linalg.norm(b['positions'].cpu().numpy()-a['positions'],axis=1).max())
  s['independent_CPU_GPU_exact80']=dict(allnode_difference_m=delta,passed=delta<=1e-8,CPU_free_R_N=float(np.linalg.norm(a['residual'][~m.mask])),GPU_free_R_N=float(torch.linalg.vector_norm(b['residual'])))
  save()
  if delta>1e-8:raise RuntimeError('Independent coarse exact80 CPU/GPU parity failed')
  del a,b,t,cpu,m;torch.cuda.empty_cache();fields=[]
  points=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
  for cells in p['cells']:
   begin=time.monotonic();m=geometry(cells);t=Budget(m,'cuda:0',4000,8,support_kind='both_ends');a=t.solve();q=a['positions'].cpu().numpy();R=(a['cache']['gradient']-t.force).cpu().numpy();Q=m.query(q,points)
   f=OUT/f'tets{len(m.tets)}.npz';np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,known_load_N=m.load,material_probes_m=points,probe_positions_m=Q)
   diag=m.diagnostics(dict(positions=q,residual=R,J=a['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=a['iterations']));cert=whole_p2_jacobian_bounds(m,q)
   physical=bool(diag['original_free_residual_N']<=1e-10 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and diag['minimum_J']>.1 and cert['all_elements_positive'])
   row=dict(cells=cells,completed=True,physics_passed=physical,hmax_m=m.maximum_reference_tet_edge_m,diagnostics=diag,Bernstein_minimum_lower_bound=cert['minimum_lower_bound'],elapsed_s=time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),history=a['history'],linear_records=t.linear_records);s['meshes'].append(row);fields.append(Q);save()
   if not physical:raise RuntimeError('Original physical gate failed')
   del a,t,m;torch.cuda.empty_cache()
  delta=[float(np.linalg.norm(b-a,axis=-1).max()) for a,b in zip(fields[:-1],fields[1:])];factors=[1/(a['hmax_m']/b['hmax_m']-1) for a,b in zip(s['meshes'][:-1],s['meshes'][1:])];indicators=[a*b for a,b in zip(delta,factors)]
  s.update(completed=True,passed=indicators[-1]<=20e-6,adjacent_max_query_difference_m=delta,conditional_indicators_m=indicators)
 except Exception as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:save();guard.terminate();guard.join(timeout=2)
 return s
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','launch']);a.add_argument('--sha');a=a.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':
  r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error','conditional_indicators_m']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserve execution')
  r=dict(pid=os.getpid(),terminal=False,protocol_sha256=a.sha,hard_cap_s=900);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_gravity_refinement','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
