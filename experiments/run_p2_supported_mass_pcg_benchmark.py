"""Prescribed-RHS exact consistent-mass GPU PCG/LU parity; no ODE/Newton."""
from pathlib import Path
import argparse,ast,json,os,sys,time,threading,traceback,subprocess
import numpy as np
from experiments.run_p2_supported_gravity_parity import ROOT,ENV_SITE,GPU,sha,write,record,runtime_pins
from experiments.run_p2_supported_gravity_capacity import static_records,prescribed_static_state
from experiments.campaign_c38_static_family import build_geometry
from experiments.campaign_supported_gravity_dop import WarpBulkP2Force,SupportedBulkDynamics
from experiments.campaign_consistent_mass_pcg import ConsistentMassPCG
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
OP=ROOT/'runs/operator-dev/p2-supported-mass-pcg-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-supported-mass-pcg-v01'
CELLS=[(4,2,2),(20,10,10),(24,12,12)]

def source_closure():
 todo=[Path(__file__),ROOT/'tests/test_campaign_consistent_mass_pcg.py'];seen={}
 while todo:
  p=todo.pop();key=str(p.relative_to(ROOT))
  if key in seen:continue
  seen[key]=sha(p)
  for node in ast.walk(ast.parse(p.read_text())):
   if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
    f=ROOT/(node.module.replace('.','/')+'.py')
    if f.exists():todo.append(f)
 return dict(sorted(seen.items()))

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen numerical-control stage')
 prior=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-capacity-v01/summary.json';execution=ROOT/'runs/operator-dev/p2-supported-gravity-capacity-v01/execution.json';s=json.loads(prior.read_text());e=json.loads(execution.read_text())
 if not s['completed'] or not s['passed'] or not e['terminal'] or e['exit_code']!=0:raise ValueError('Completed original capacity control required')
 p=dict(schema='supported-full-consistent-mass-PCG-numerical-control-v0.1',status='FROZEN_BEFORE_GPU_CALLS_REQUIRES_ROOT_REVIEW',gpu_uuid=GPU,CPU_threads=2,CPU_affinity_cores=sorted(os.sched_getaffinity(0))[:2],wall_cap_s=180,resource_limits=LIMITS,source_sha256=source_closure(),runtime_source_sha256=runtime_pins(),taskset=dict(path='/usr/bin/taskset',sha256=sha('/usr/bin/taskset')),capacity_evidence=[record(prior),record(execution)],static_prescribed_evidence=static_records(),cells=CELLS,physical_case=dict(E_Pa=100000.,nu=.35,rho_kg_m3=625.,g_m_s2=9.81,support='both_ends',surface_force_N=0.,same_exact_analytic_full_consistent_mass=True),states=dict(tiny96=['rest'],fine12000_20736=['rest','old_hashbound_exact80_static']),solver=dict(absolute_tolerance_N=1e-12,relative_tolerance=1e-11,effective_tolerance='min(absolute,relative*RHSnorm); onlytightens',maximum_iterations=256,true_residual_every=8,zero_initial_guess_every_call=True,preconditioner='OriginalmassdiagonalONLY; physicalMneverlumped',false_recurrence_convergence='restartfromoriginalresidual; nocap/tolerancechange'),timing=dict(warm_PCG_calls=1,repeated_PCG_calls=3,repeated_LU_calls=3,includes_host_device_transfers=True,initialization_separately_recorded=True),gates=dict(original_mass_R_N=1e-12,relative_acceleration_L2=1e-9,maximum_acceleration_difference_m_s2=1e-6,fixed_Mcf_reaction_difference_N=1e-10,manufactured_kinetic_absolute_J=1e-10,manufactured_work_absolute_J=1e-12,original_prescribed_static_free_R_N=1e-10,sampled_J_minimum=.1),energy_work_probe='NOTtrajectory:manufacturedv=acceleration*.001s;u=acceleration*.5*(.001s)^2; comparefullMkineticandactualFdotu',nonclaims=['No ODE/Newton/physicaltrajectory','No alteredphysicalmass/energy/force/material/support','No dynamic/spatial/data/inferencequalification','Numericalbenchmarkonly; no posterior/observations'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed protocol')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed source '+f)
 if runtime_pins()!=p['runtime_source_sha256']:raise ValueError('Changed runtime')
 for r in p['capacity_evidence']+p['static_prescribed_evidence']:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed control/input')
 if static_records()!=p['static_prescribed_evidence']:raise ValueError('Changed original static source closure')
 return p

def run(digest):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('PhysicalGPU0 only')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();summary=dict(completed=False,passed=False,protocol=record(OP/'protocol.json'),meshes=[],noODE_noNewton=True);write(OUT/'summary.json',summary)
 try:
  import torch,gc
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0' or not Path(wp.__file__).resolve().is_relative_to(ENV_SITE):raise ValueError('Pinned Warp required')
  wp.config.kernel_cache_dir=str(OP/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());torch.cuda.set_per_process_memory_fraction(min(.95,LIMITS['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory));torch.ones(1,device='cuda:0').sum().item()
  stop=threading.Event();lock=threading.Lock();metrics=[]
  def guard(tag,reserve_memory_bytes=0):
   with lock:
    a=collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch);validate_budget(a,reserve_memory_bytes=reserve_memory_bytes,reserve_write_bytes=16*1024**2,limits=p['resource_limits']);metrics.append(a);write(OP/'resource-latest.json',dict(tag=tag,**a));return a
  def monitor():
   while not stop.wait(1):
    try:guard('watchdog')
    except BaseException as error:write(OP/'resource-failure.json',dict(error=repr(error)));os.kill(os.getpid(),__import__('signal').SIGTERM);return
  guard('before_models',reserve_memory_bytes=2*1024**3);thread=threading.Thread(target=monitor,daemon=True);thread.start()
  for cells in p['cells']:
   begin=time.monotonic();m=build_geometry(tuple(cells),100000.,'both_ends',Fz=0.,rule='keast24');backend=WarpBulkP2Force(m,device='cuda:0');t=time.monotonic();d=SupportedBulkDynamics(m,backend);LU_init=time.monotonic()-t;t=time.monotonic();solver=ConsistentMassPCG(d.mass_ff.tocsr(),device='cuda:0',absolute_tolerance_N=p['solver']['absolute_tolerance_N'],relative_tolerance=p['solver']['relative_tolerance'],max_iterations=p['solver']['maximum_iterations'],true_residual_every=p['solver']['true_residual_every']);PCG_init=time.monotonic()-t
   zero=np.zeros_like(m.rest);states=[('rest',zero)]
   if len(m.elements)>96:
    r=next(r for r in p['static_prescribed_evidence'] if r['path'].endswith('/tets'+str(len(m.elements))+'.npz'));u,difference=prescribed_static_state(m,r);states.append(('old_completed_static',u))
   np.testing.assert_array_equal(solver.solve(np.zeros((len(d.free_nodes),3))),0);zero_record=dict(solver.last_record);results=[]
   for name,u in states:
    cache=backend.evaluate(u);net=m.load-cache['gradient_N'];b=net[d.free_nodes];warm=solver.solve(b);warm_record=dict(solver.last_record);PCG_times=[];LU_times=[];records=[]
    for _ in range(p['timing']['repeated_PCG_calls']):
     t=time.monotonic();actual=solver.solve(b);PCG_times.append(time.monotonic()-t);records.append(dict(solver.last_record))
    for _ in range(p['timing']['repeated_LU_calls']):
     t=time.monotonic();oracle=d.mass_lu.solve(b);LU_times.append(time.monotonic()-t)
    error=actual-oracle;relative=float(np.linalg.norm(error)/max(np.linalg.norm(oracle),np.finfo(float).tiny));R=float(np.linalg.norm(d.mass_ff@actual-b));full=np.zeros_like(zero);expected=full.copy();full[d.free_nodes]=actual;expected[d.free_nodes]=oracle;reaction_error=float(np.linalg.norm((d.mass@(full-expected))[d.fixed_nodes]));v=full*.001;vref=expected*.001;kinetic_error=abs(.5*float(np.sum(v*(d.mass@v)))-.5*float(np.sum(vref*(d.mass@vref))));work_error=abs(float(np.sum(m.load*(full-expected)))*.5*.001**2)
    row=dict(state=name,PCG_s=PCG_times,LU_s=LU_times,initial_warm_record=warm_record,PCG_records=records,independent_SciPy_true_R_N=R,relative_acceleration_L2=relative,maximum_acceleration_difference_m_s2=float(abs(error).max()),fixed_Mcf_reaction_difference_N=reaction_error,manufactured_kinetic_difference_J=kinetic_error,manufactured_work_difference_J=work_error,original_elastic_free_R_N=float(np.linalg.norm(net[d.free_nodes])),minimum_J=cache['minimum_J'])
    g=p['gates'];row['passed']=bool(R<=g['original_mass_R_N'] and relative<=g['relative_acceleration_L2'] and row['maximum_acceleration_difference_m_s2']<=g['maximum_acceleration_difference_m_s2'] and reaction_error<=g['fixed_Mcf_reaction_difference_N'] and kinetic_error<=g['manufactured_kinetic_absolute_J'] and work_error<=g['manufactured_work_absolute_J'] and cache['minimum_J']>g['sampled_J_minimum'] and (name=='rest' or row['original_elastic_free_R_N']<=g['original_prescribed_static_free_R_N']));results.append(row)
    if not row['passed']:break
   row=dict(cells=cells,tetrahedra=len(m.elements),nodes=len(m.rest),LU_initialization_s=LU_init,PCG_initialization_s=PCG_init,zero_RHS=zero_record,states=results,elapsed_s=time.monotonic()-begin,passed=len(results)==len(states) and all(r['passed'] for r in results));row['resources']=guard('after_mesh');summary['meshes'].append(row);write(OUT/(str(len(m.elements))+'tets.json'),row);write(OUT/'summary.json',summary)
   if not row['passed']:raise ValueError('Strict physical mass/LU parity failed; preserve all records')
   del solver,d,m,backend,states,net,full,expected;gc.collect();torch.cuda.empty_cache()
  summary.update(completed=True,passed=True)
 except BaseException as error:summary.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  if 'stop' in locals():stop.set()
  if 'thread' in locals():thread.join(timeout=2)
  summary['elapsed_s']=time.monotonic()-start
  if 'metrics' in locals() and metrics:summary['maximum_observed_resources']={k:max(x[k] for x in metrics) for k in ['gpu_bytes','rss_bytes','disk_bytes']}
  write(OUT/'summary.json',summary)
 return summary

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('No automatic retry')
 if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 if GPU in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):raise RuntimeError('GPU0 not idle')
 e=dict(terminal=False,controller_pid=os.getpid(),protocol_sha256=digest);write(OP/'execution.json',e);start=time.monotonic()
 def on_child(pid):e['worker_pid']=pid;write(OP/'execution.json',e)
 try:
  with (OP/'worker.log').open('x') as log:e.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_supported_mass_pcg_benchmark','run','--sha',digest],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=p['wall_cap_s'],on_child=on_child))
 except BaseException as error:e['controller_error']=repr(error)
 finally:
  s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=e.get('exit_code')==0 and s.get('completed') is True,passed=s.get('passed',False),elapsed_s=time.monotonic()-start);write(OP/'execution.json',e)
 return e

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','launch','run']);a.add_argument('--sha');args=a.parse_args();r=prepare() if args.action=='prepare' else launch(args.sha) if args.action=='launch' else run(args.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','elapsed_s','error']}));raise SystemExit(0 if args.action=='prepare' or r.get('completed') else 1)
