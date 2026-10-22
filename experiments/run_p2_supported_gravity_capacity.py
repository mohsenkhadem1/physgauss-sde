"""Fine pure-gravity capacity assessment ONLY: no ODE/Newton/equilibrium solve.

Construct actual T10 consistent M_ff LU and the unchanged Warp bulk backend;
measure rest and hash-bound completed static states before scheduling trajectories.
"""
from pathlib import Path
import argparse,ast,json,os,sys,time,threading,traceback,subprocess
import numpy as np
from experiments.run_p2_supported_gravity_parity import ROOT,ENV_SITE,GPU,sha,write,record,runtime_pins
from experiments.campaign_c38_static_family import build_geometry,array_record
from experiments.campaign_supported_gravity_dop import WarpBulkP2Force,SupportedBulkDynamics
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
OP=ROOT/'runs/operator-dev/p2-supported-gravity-capacity-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-capacity-v01'
CELLS=[(20,10,10),(24,12,12)]

def source_closure():
 todo=[Path(__file__),ROOT/'tests/test_run_p2_supported_gravity_capacity.py'];seen={}
 while todo:
  p=todo.pop();key=str(p.relative_to(ROOT))
  if key in seen:continue
  seen[key]=sha(p)
  for node in ast.walk(ast.parse(p.read_text())):
   if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
    f=ROOT/(node.module.replace('.','/')+'.py')
    if f.exists():todo.append(f)
 return dict(sorted(seen.items()))

def structured_p2_counts(cells):
 x,y,z=cells;corners=(x+1)*(y+1)*(z+1);edges=x*(y+1)*(z+1)+(x+1)*y*(z+1)+(x+1)*(y+1)*z+x*y*(z+1)+x*(y+1)*z+(x+1)*y*z+x*y*z
 return dict(tetrahedra=6*x*y*z,nodes=corners+edges,fixed_nodes=2*(2*y+1)*(2*z+1),quadrature_and_guard_gradient_bytes=(24+90)*6*x*y*z*10*3*8)

PRIOR=ROOT/'runs/gaussian-campaign-v01/p2-gravity-refinement-v01'
PRIOR_PROTOCOL=ROOT/'runs/operator-dev/p2-gravity-refinement-v01/protocol.json'

def static_records():
 summary=json.loads((PRIOR/'summary.json').read_text());protocol=json.loads(PRIOR_PROTOCOL.read_text())
 if not summary.get('completed') or not summary.get('passed') or protocol['physics']['E_Pa']!=100000. or protocol['physics']['support']!='both_ends':raise ValueError('Exact completed bothends E100k static reference required')
 for f,h in protocol['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed frozen static physical source '+f)
 rows=[]
 for n in [12000,20736]:
  row=next(x for x in summary['meshes'] if x['capture']['path'].endswith('/tets'+str(n)+'.npz'))
  if not row['completed'] or not row['physics_passed'] or sha(ROOT/row['capture']['path'])!=row['capture']['sha256']:raise ValueError('Changed completed static capture')
  rows.append(row['capture'])
 return [record(PRIOR/'summary.json'),record(PRIOR_PROTOCOL)]+rows

def prescribed_static_state(model,capture):
 if sha(ROOT/capture['path'])!=capture['sha256']:raise ValueError('Changed prescribed static field')
 with np.load(ROOT/capture['path'],allow_pickle=False) as a:
  for key,value in [('rest_m',model.rest),('elements',model.elements),('tets',model.tets),('constraint_mask',model.mask)]:np.testing.assert_array_equal(a[key],value)
  u=a['positions_m']-model.rest;delta=a['known_load_N']-model.load
 if np.max(abs(delta[~model.mask]))>1e-12 or np.max(abs(u[model.mask]))!=0:raise ValueError('Only known fixed-node traction may differ; free equations and exactsupports unchanged')
 if np.max(abs(delta.sum(0)-[0,0,-.12]))>1e-12:raise ValueError('Expected historical fixed-end traction difference only')
 return u,dict(maximum_free_load_difference_N=float(abs(delta[~model.mask]).max()),fixed_load_difference_resultant_N=delta.sum(0).tolist(),historical_right_traction_only_at_fixed_nodes=True)

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserve prospective capacity protocol')
 tiny=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-parity-v01/summary.json';execution=ROOT/'runs/operator-dev/p2-supported-gravity-parity-v01/execution.json';s=json.loads(tiny.read_text());e=json.loads(execution.read_text())
 if not s['completed'] or not s['backend_parity_passed'] or not e['terminal'] or e['exit_code']!=0:raise ValueError('Completed tiny implementation parity required')
 p=dict(schema='supported-pure-gravity-capacity-NO-INTEGRATION-v0.1',status='FROZEN_CAPACITY_ONLY_REQUIRES_ROOT_REVIEW',gpu_uuid=GPU,CPU_threads=2,CPU_affinity_cores=sorted(os.sched_getaffinity(0))[:2],wall_cap_s=180,resource_limits=LIMITS,source_sha256=source_closure(),runtime_source_sha256=runtime_pins(),taskset=dict(path='/usr/bin/taskset',sha256=sha('/usr/bin/taskset')),tiny_evidence=[record(tiny),record(execution)],static_prescribed_evidence=static_records(),cells=CELLS,geometry_estimates=[dict(cells=c,**structured_p2_counts(c)) for c in CELLS],physical_case=dict(E_Pa=100000.,poisson=.35,rho_kg_m3=625.,g_m_s2=9.81,support='both_ends',body_force_only=True,known_surface_force_N=0.,bulk='Keast24 exactdegree6 stableNH',Jguards='original80+10; originalstrict.1'),assessment=dict(states=['rest','hash-bound completed exact80 static state at same mesh/material/support/bodyforce; oldfixedendtraction removed from fixedreactions only'],warm_calls=1,timed_force_calls=3,timed_full_mechanics_calls=3,mass='analytic consistent full scalarCSR; SciPy M_ff sparseLU; retainM_cf',no_ODE_Newton_or_equilibrium_calls=True),gates=dict(mass_residual_N=1e-9,minimum_J_strictly_above=.1,exact_fixed_u_v=0,prescribed_static_free_R_N=1e-10),future_trajectory_plan=dict(cells=CELLS,configs=[dict(rtol=1e-9,u_atol_m=1e-13,v_atol_m_s=1e-11,max_step_s=h) for h in [1e-4,5e-5]],interval_s=[0,.1],shared_outputs=25,physical_query_count=63,temporal_allnode_q_m=20e-6,temporal_allnode_v_m_s=1e-4,spatial_query_q_indicator_m=20e-6,spatial_query_v_indicator_m_s=1e-4,indicator='max_query_delta/(actualhmax_coarse/actualhmax_fine-1); conditionalassumedp>=1, notrigorousbound',budget_status='Notfixed/notauthorized; freezeaftercapacitymeasurements beforeanytrajectory'),nonclaims=['No physical response trajectory generated','No spatial/time qualification or dataset admission','Completed static states are prescribed inputs, not dynamic responses','Timing includes explicit host transfers; no high-rate count inferred as known'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed protocol')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 if runtime_pins()!=p['runtime_source_sha256']:raise ValueError('Changed runtime')
 for r in p['tiny_evidence']+p['static_prescribed_evidence']:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed prerequisite/prescribed field')
 if static_records()!=p['static_prescribed_evidence']:raise ValueError('Changed original static source closure')
 return p

def run(digest):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Declared GPU0 UUID only')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,passed=False,protocol=record(OP/'protocol.json'),meshes=[],no_ODE_or_Newton_calls=True);write(OUT/'summary.json',s)
 try:
  import torch,gc
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0' or not Path(wp.__file__).resolve().is_relative_to(ENV_SITE):raise ValueError('Pinned Warp runtime required')
  wp.config.kernel_cache_dir=str(OP/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());torch.cuda.set_per_process_memory_fraction(min(.95,LIMITS['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory));torch.ones(1,device='cuda:0').sum().item()
  lock=threading.Lock();stop=threading.Event();measurements=[]
  def guard(tag,reserve_memory_bytes=0):
   with lock:
    a=collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch);validate_budget(a,reserve_memory_bytes=reserve_memory_bytes,reserve_write_bytes=16*1024**2,limits=p['resource_limits']);measurements.append(a);write(OP/'resource-latest.json',dict(tag=tag,**a));return a
  def monitor():
   while not stop.wait(1):
    try:guard('watchdog')
    except BaseException as error:write(OP/'resource-failure.json',dict(error=repr(error)));os.kill(os.getpid(),__import__('signal').SIGTERM);return
  guard('before_models',reserve_memory_bytes=2*1024**3);thread=threading.Thread(target=monitor,daemon=True);thread.start()
  for cells in p['cells']:
   begin=time.monotonic();m=build_geometry(tuple(cells),100000.,'both_ends',Fz=0.,rule='keast24');geometry_s=time.monotonic()-begin
   estimate=structured_p2_counts(cells)
   if len(m.rest)!=estimate['nodes'] or len(m.elements)!=estimate['tetrahedra'] or m.mask[:,0].sum()!=estimate['fixed_nodes']:raise ValueError('Structured geometry count disagreement')
   guard('before_backend',reserve_memory_bytes=2*estimate['quadrature_and_guard_gradient_bytes']);t=time.monotonic();backend=WarpBulkP2Force(m,device='cuda:0');backend_s=time.monotonic()-t;t=time.monotonic();d=SupportedBulkDynamics(m,backend);mass_LU_s=time.monotonic()-t
   capture=next(r for r in p['static_prescribed_evidence'] if r['path'].endswith('/tets'+str(len(m.elements))+'.npz'));u,load_difference=prescribed_static_state(m,capture);zero=np.zeros_like(u)
   t=time.monotonic();backend.evaluate(zero);warm_s=time.monotonic()-t;state_records=[]
   for name,displacement in [('rest',zero),('completed_static',u)]:
    y=d.pack(displacement,zero);force_times=[];mechanics_times=[];states=[];cache=None
    for _ in range(p['assessment']['timed_force_calls']):
     t=time.monotonic();cache=backend.evaluate(displacement);force_times.append(time.monotonic()-t)
    for _ in range(p['assessment']['timed_full_mechanics_calls']):
     t=time.monotonic();states.append(d.mechanics(y));mechanics_times.append(time.monotonic()-t)
    free_R=float(np.linalg.norm((cache['gradient_N']-d.force)[~m.mask]));rowstate=dict(name=name,timed_force_s=force_times,timed_full_mechanics_s=mechanics_times,free_original_elastic_force_residual_N=free_R,elastic_energy_J=cache['elastic_energy_J'],mass_residual_max_N=max(x['mass_residual_N'] for x in states),minimum_J=min(x['minimum_J'] for x in states),maximum_fixed_u_m=float(abs(displacement[m.mask]).max()))
    rowstate['passed']=rowstate['mass_residual_max_N']<=1e-9 and rowstate['minimum_J']>.1 and rowstate['maximum_fixed_u_m']==0 and (name=='rest' or free_R<=1e-10);state_records.append(rowstate)
   M,L,U=d.mass,d.mass_lu.L,d.mass_lu.U;csr_bytes=lambda a:int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)
   row=dict(cells=cells,nodes=len(m.rest),tetrahedra=len(m.elements),free_nodes=len(d.free_nodes),geometry_s=geometry_s,backend_init_s=backend_s,full_consistent_Mff_LU_s=mass_LU_s,warm_force_including_JIT_s=warm_s,prescribed_states=state_records,historical_fixed_load_difference=load_difference,M_scalar_nnz=int(M.nnz),M_scalar_CSR_bytes=csr_bytes(M),Mff_nnz=int(d.mass_ff.nnz),LU_nnz=int(L.nnz+U.nnz),LU_array_bytes=csr_bytes(L)+csr_bytes(U),LU_fill_ratio=float((L.nnz+U.nnz)/d.mass_ff.nnz),mass_residual_max_N=max(x['mass_residual_max_N'] for x in state_records),minimum_J=min(x['minimum_J'] for x in state_records),maximum_fixed_u_m=float(abs(u[m.mask]).max()),force=array_record(m.load),rest=array_record(m.rest),elements=array_record(m.elements),hmax_m=m.maximum_reference_tet_edge_m,elapsed_s=time.monotonic()-begin)
   row['passed']=all(x['passed'] for x in state_records)
   row['final_resources']=guard('after_mesh');s['meshes'].append(row);write(OUT/(str(len(m.elements))+'tets.json'),row);write(OUT/'summary.json',s)
   if not row['passed']:raise ValueError('Unchanged prescribed-state guards failed')
   del m,backend,d,M,L,U,states,y,u,zero;gc.collect();torch.cuda.empty_cache()
  s.update(completed=True,passed=all(r['passed'] for r in s['meshes']))
 except BaseException as error:s.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  if 'stop' in locals():stop.set()
  if 'thread' in locals():thread.join(timeout=2)
  s['elapsed_s']=time.monotonic()-start
  if 'measurements' in locals() and measurements:s['maximum_observed_resources']={k:max(x[k] for x in measurements) for k in ['gpu_bytes','rss_bytes','disk_bytes']}
  write(OUT/'summary.json',s)
 return s

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('No automatic retry')
 if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 if GPU in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):raise RuntimeError('GPU0 not idle')
 e=dict(terminal=False,controller_pid=os.getpid(),protocol_sha256=digest);write(OP/'execution.json',e);start=time.monotonic()
 def on_child(pid):e['worker_pid']=pid;write(OP/'execution.json',e)
 try:
  with (OP/'worker.log').open('x') as log:e.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_supported_gravity_capacity','run','--sha',digest],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=p['wall_cap_s'],on_child=on_child))
 except BaseException as error:e['controller_error']=repr(error)
 finally:
  s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=e.get('exit_code')==0 and s.get('completed') is True,passed=s.get('passed',False),elapsed_s=time.monotonic()-start);write(OP/'execution.json',e)
 return e

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','launch','run']);a.add_argument('--sha');args=a.parse_args();r=prepare() if args.action=='prepare' else launch(args.sha) if args.action=='launch' else run(args.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if args.action=='prepare' or r.get('completed') else 1)
