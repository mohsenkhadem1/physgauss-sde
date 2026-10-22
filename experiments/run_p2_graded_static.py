"""Same transverse experiment; graded straight P2 mesh refinement only."""
import argparse,json,os,signal,subprocess,time,traceback,multiprocessing
import numpy as np
from experiments.run_p2_torch_benchmark import ROOT,sha,write
OP=ROOT/'runs/operator-dev/p2-graded-static-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-graded-static-v01'
SOURCES=['experiments/run_p2_graded_static.py','experiments/campaign_p2_geometry.py','experiments/campaign_p2_coarse_preconditioner.py','experiments/campaign_p2_block_preconditioner.py','experiments/campaign_p2_torch.py','experiments/campaign_block_surface_controls.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_p2_consistent_transient.py','experiments/boundary_block.py','experiments/run_static_p2_axial_continuation.py','tests/test_campaign_p2_geometry.py']
CELLS=[(20,10,10),(24,12,12),(32,16,16)]
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenprotocol')
 from experiments.campaign_p2_geometry import P2Geometry
 meshes=[]
 for cells in CELLS:
  m=P2Geometry(cells,mapping='clamp_graded');meshes.append(dict(cells=cells,hmax_m=m.maximum_reference_tet_edge_m,hmin_m=m.minimum_reference_tet_edge_m,volume_ratio=m.cell_volume_ratio))
 p=dict(schema='P2-graded-static-refinement-v0.1',status='FROZEN_BEFORE_SOLVES',GPU=0,CPU_threads=2,wall_cap_s=900,GPU_allocated_limit_bytes=8*1024**3,RSS_limit_bytes=12*1024**3,meshes=meshes,source_sha256={s:sha(ROOT/s) for s in SOURCES},physics=dict(E_Pa=4500,nu=.35,rho=625,force_pattern='transverse_end',known_force_N=[0,0,-.12],BC='fullleft',gravity=0,physical_domain_m=[.24,.08,.08]),mapping='x=L*(x/L)^2,y/z=halfwidth*sin(pi*coordinate/width);cornersonlythenstraightP2edgeenrichment',gates=dict(original_free_R_N=1e-9,global_force_N=1e-8,actual_moment_Nm=1e-8,exact_clamp_m=0,positive_global_J_certificate=True,conditional_spatial_indicator_m=20e-6),indicator='delta24to32/(actual_hmax24/actual_hmax32-1), assumedp>=1 in this gradedfamily; engineeringindicatorNOTprovenbound',limitations=['Allolduniformmeshfailurespreserved','Gradingmayworsenaspectratios; originalphysics/operatorunchanged','Q4/time/contact/inferencequalificationseparate','Basegeometrybytesarenotpeakworkingmemory;actualguardsauthoritative','Fixed63queriesunchanged;nopointtrimming'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')
def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changedsource '+f)
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(completed=False,passed=False,meshes=[],protocol_sha256=digest)
 def interrupted(*_):raise TimeoutError('Ownedresourceguard/operatorstop; seeexecutionreceipt')
 signal.signal(signal.SIGTERM,interrupted)
 from experiments.run_static_p2_axial_continuation import watchdog
 guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+900,12*1024**3,OUT/'resource-guard.json'),daemon=True);guard.start()
 try:
  import torch,shutil
  from experiments.campaign_p2_geometry import P2Geometry
  from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2
  torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction((8*1024**3)/torch.cuda.get_device_properties(0).total_memory,0);fields=[];points=np.stack(np.meshgrid(np.linspace(0,.24,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),axis=-1).reshape(-1,3)
  for spec in p['meshes']:
   if shutil.disk_usage(ROOT).free<30*1024**3:raise RuntimeError('30GiBdiskreserve')
   begin=time.monotonic();m=P2Geometry(tuple(spec['cells']),mapping='clamp_graded');t=TwoLevelP2(m,'cuda:0');a=t.solve();q=a['positions'].cpu().numpy();residual=(a['cache']['gradient']-t.force).cpu().numpy();diag=m.diagnostics(dict(positions=q,residual=residual,J=a['cache']['minimum_J'].cpu().numpy()[None],tolerance_N=1e-10,iterations=a['iterations']));Q=m.query(q,points);fields.append(Q)
   passed=diag['original_free_residual_N']<=1e-9 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and (diag['global_affine_F_positive_J_sufficient_lower_bound'] or 0)>0
   file=OUT/f'tets{len(m.tets)}.npz';np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=q,constraint_mask=m.mask,applied_force_N=m.load,material_probes_m=points,probe_positions_m=Q)
   row=dict(**spec,completed=True,physics_passed=passed,diagnostics=diag,elapsed_s=time.monotonic()-begin,base_geometry_array_bytes=m.base_geometry_array_bytes,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),history=a['history'],linear_records=t.two_level_records,capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)));state['meshes'].append(row);write(OUT/f'tets{len(m.tets)}.json',row);write(OUT/'summary.json',state)
   if not passed:raise RuntimeError('Originalphysicsgatefailed')
   del a,t,m;torch.cuda.empty_cache()
  delta=[float(np.linalg.norm(b-a,axis=-1).max()) for a,b in zip(fields[:-1],fields[1:])];factors=[1/(a['hmax_m']/b['hmax_m']-1) for a,b in zip(p['meshes'][:-1],p['meshes'][1:])];indicators=[a*b for a,b in zip(delta,factors)];state.update(completed=True,passed=all(r['physics_passed'] for r in state['meshes']) and indicators[-1]<=20e-6,adjacent_maximum_fixedquery_delta_m=delta,actual_hmax_indicator_factors=factors,conditional_fine_indicators_m=indicators)
 except Exception as e:state.update(error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:guard.terminate();guard.join(timeout=2);state['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args()
 if a.action=='prepare':print(prepare())
 elif a.action=='run':
  r=run(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error','conditional_fine_indicators_m']}));raise SystemExit(0 if r['completed'] else 1)
 else:
  if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
  r=dict(pid=os.getpid(),terminal=False,hard_cap_s=900,protocol_sha256=a.sha);write(OP/'execution.json',r)
  with (OP/'worker.log').open('x') as log:
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_graded_static','run','--sha',a.sha],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});r['worker_pid']=child.pid;write(OP/'execution.json',r);start=time.monotonic()
   try:code=child.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode;r['timeout']=True
  r.update(terminal=True,exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',r);print(json.dumps(r));raise SystemExit(code)
