"""Prospective tiny96 pure-gravity CPU80/GPU24 trajectory implementation check.

No physical integration at import/prepare. Launch requires a separately reviewed
protocol hash. This is neither spatial admission nor a new material law.
"""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,time,traceback,threading,subprocess
import numpy as np
import scipy
from experiments.campaign_c38_static_family import build_geometry,array_record
from experiments.campaign_static_p2_block import shape_gradients
from experiments.campaign_newton_stable_material import stable_material
from experiments.campaign_supported_gravity_dop import WarpBulkP2Force,SupportedBulkDynamics,AcceptedDOPCheckpoint,streaming_dop853
from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/p2-supported-gravity-parity-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-parity-v01'
ENV_SITE=ROOT/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'
GPU='GPU-5b432ce5-a24a-68ba-150f-682265260a9b'
CONFIGS=[dict(id='accuracy1',rtol=1e-7,displacement_atol_m=1e-11,velocity_atol_m_s=1e-9),dict(id='accuracy2',rtol=1e-9,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11)]
POLICY=dict(every_accepted_steps=32,on_initial=True,on_new_common_output=True,on_final=True,max_uncommitted_accepted_steps_bound=32)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,p)
def record(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=sha(p))

class NumpyBulkP2Force:
 """Vectorized independent NumPy original stable-NH force, no Warp/Torch call."""
 def __init__(self,model):
  self.model=model;self.elements=model.elements.copy();self.grad=np.stack([shape_gradients(b[None],model.linear_gradients) for b in model.bary]);self.check_grad=np.stack([shape_gradients(b[None],model.linear_gradients) for b in model.check_bary]);self.weights=model.weights[:,None]*model.volumes[None]
 def evaluate(self,u):
  m=self.model;u=np.asarray(u)
  if u.shape!=m.rest.shape or u.dtype!=np.float64 or not np.isfinite(u).all():raise ValueError('Finite matching FP64 displacement required')
  local=u[self.elements];F=np.eye(3)+np.einsum('eai,qeaj->qeij',local,self.grad,optimize=True);checks=np.eye(3)+np.einsum('eai,qeaj->qeij',local,self.check_grad,optimize=True);J=np.linalg.det(F);Jc=np.linalg.det(checks);j=float(min(J.min(),Jc.min()))
  if not np.isfinite(J).all() or not np.isfinite(Jc).all() or j<=m.minimum_J:raise ValueError('Original sampled J guard failed; no repair')
  W,P,_=stable_material(F,m.young,m.poisson,tangent=False);local_g=np.einsum('qe,qeij,qeaj->eai',self.weights,P,self.grad,optimize=True);g=np.zeros_like(u);np.add.at(g,self.elements,local_g);energy=float(np.sum(self.weights*W))
  if not np.isfinite(energy) or not np.isfinite(g).all():raise ValueError('Nonfinite original energy/gradient')
  return dict(elastic_energy_J=energy,gradient_N=g,minimum_J=j)

def models():
 gpu=build_geometry((4,2,2),100000.,'both_ends',Fz=0.,rule='keast24')
 cpu=build_geometry((4,2,2),100000.,'both_ends',Fz=0.,rule='exact80')
 # Independently recompute physical body force from actual exact80 quadrature.
 independently_integrated=np.zeros(len(cpu.rest));local=cpu.volumes[:,None]*np.einsum('q,qa->a',cpu.weights,cpu.N);np.add.at(independently_integrated,cpu.elements,local)
 independent_force=np.zeros_like(cpu.rest);independent_force[:,2]=-cpu.density*9.81*independently_integrated
 if np.max(abs(independent_force-cpu.load))>1e-12 or np.any(gpu.public_force_components['surface_N']):raise ValueError('Pure body-only gravity required, no hidden fixed-end traction')
 cpu.load=independent_force
 for name in ['rest','elements','tets','mask']:np.testing.assert_array_equal(getattr(cpu,name),getattr(gpu,name))
 return cpu,gpu

def source_closure():
 todo=[Path(__file__),ROOT/'tests/test_run_p2_supported_gravity_parity.py'];seen={}
 while todo:
  p=todo.pop();key=str(p.relative_to(ROOT))
  if key in seen:continue
  seen[key]=sha(p)
  for node in ast.walk(ast.parse(p.read_text())):
   if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
    f=ROOT/(node.module.replace('.','/')+'.py')
    if f.exists():todo.append(f)
 return dict(sorted(seen.items()))

def runtime_pins():
 # Pin installed algorithm and native-kernel sources actually imported by this stage.
 sr=Path(scipy.__file__).parent
 scipy_files=['integrate/_ivp/rk.py','integrate/_ivp/base.py','integrate/_ivp/common.py','integrate/_ivp/dop853_coefficients.py','sparse/linalg/_dsolve/linsolve.py']
 warp_files=['warp/__init__.py','warp/_src/context.py','warp/_src/types.py','warp/_src/codegen.py','warp/config.py','warp/bin/warp.so']
 return dict(scipy={f:sha(sr/f) for f in scipy_files},warp={f:sha(ENV_SITE/f) for f in warp_files},versions=dict(numpy=np.__version__,scipy=scipy.__version__,warp='1.18.0'))

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen prospective stage already exists')
 cpu,gpu=models();cores=sorted(os.sched_getaffinity(0))[:2]
 p=dict(schema='supported-pure-gravity-tiny96-backend-parity-v0.1',status='FROZEN_BEFORE_TRAJECTORIES_REQUIRES_ROOT_REVIEW',gpu_uuid=GPU,wall_cap_s=600,CPU_threads=2,CPU_affinity_cores=cores,resource_limits=LIMITS,source_sha256=source_closure(),runtime_source_sha256=runtime_pins(),taskset=dict(path='/usr/bin/taskset',sha256=sha('/usr/bin/taskset')),model=dict(cells=[4,2,2],tetrahedra=len(cpu.elements),nodes=len(cpu.rest),young_Pa=100000.,poisson=.35,density_kg_m3=625.,gravity_m_s2=9.81,support='both_ends',initial_displacement_velocity='exact zero',physical_law='unchanged polynomial stable Neo-Hookean; no contact/damping',surface_force_N=0.,constant_body_force_only=True,CPU_quad_points=80,GPU_quad_points=24,GPU_additional_J_guards=90,rest=array_record(cpu.rest),elements=array_record(cpu.elements),fixed_mask=array_record(cpu.mask),CPU_actual_body_load=array_record(cpu.load),GPU_actual_body_load=array_record(gpu.load),independent_load_max_difference_N=float(np.max(abs(cpu.load-gpu.load)))),configs=CONFIGS,case_order=[b+'-'+c['id'] for c in CONFIGS for b in ['numpy_cpu','warp_gpu']],outputs_s=np.linspace(0.,.1,25).tolist(),maximum_step_s=1e-4,checkpoint_policy=POLICY,gates=dict(energy_absolute_J=1e-8,mass_residual_N=1e-9,sampled_J_strictly_above=.1,common_whole_Bernstein_lower_strictly_above=0.,same_accuracy_position_m=1e-8,same_accuracy_velocity_m_s=1e-5,accuracy_refinement_position_m=20e-6,accuracy_refinement_velocity_m_s=1e-4,same_state_gradient_L2_N=1e-10,same_state_energy_J=1e-12,same_state_mass_weighted_acceleration_L2_N=1e-10),nonclaims=['No spatial or continuum dynamic qualification','No continuous-time determinant certificate','No damping/contact/inference/calibration/learned prediction claim','Shared analytic consistent-mass assembly and SciPyDOP853; independent bulk force implementations','Pure body force differs only at fixed-node reactions from historical fixed-end traction controls'],resource_estimate=dict(trajectories=4,minimum_steps_per_case=1000,RHS_calls_approx=50000,tiny_geometry_allocations_MB_less_than=100,expected_wall_s='150–550 estimate; strict600s budget preserves partial outcomes; no automatic retries'),durability='Current accepted y/f/h and all scalar prefix/common25states; commits every32accepted or newcommon or final; slots alternate by commit generation')
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed prospective protocol')
 p=json.loads((OP/'protocol.json').read_text())
 for f,h in p['source_sha256'].items():
  if sha(ROOT/f)!=h:raise ValueError('Changed stage dependency '+f)
 if runtime_pins()!=p['runtime_source_sha256']:raise ValueError('Changed installed runtime source/version')
 return p

def pair(q,v,r,s,*,qgate,vgate):
 qe=float(np.linalg.norm(q-r,axis=-1).max());ve=float(np.linalg.norm(v-s,axis=-1).max());return dict(maximum_allnode_position_difference_m=qe,maximum_allnode_velocity_difference_m_s=ve,passed=qe<=qgate and ve<=vgate)

def run(digest):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Only declared physicalGPU0 permitted')
 OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic();summary=dict(completed=False,backend_parity_passed=False,temporal_pair_passed=False,pid=os.getpid(),protocol=record(OP/'protocol.json'),cases=[],nonclaims=p['nonclaims']);write(OUT/'summary.json',summary)
 try:
  import torch
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0' or not Path(wp.__file__).resolve().is_relative_to(ENV_SITE):raise ValueError('Wrong actual Warp runtime')
  wp.config.kernel_cache_dir=str(OP/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());torch.cuda.set_per_process_memory_fraction(min(.95,LIMITS['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory));torch.ones(1,device='cuda:0').sum().item()
  guard_failure=[];stop=threading.Event();guard_lock=threading.Lock()
  def guard(tag,reserve_write_bytes=0,reserve_memory_bytes=0):
   with guard_lock:
    metrics=collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch);validate_budget(metrics,reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes,limits=p['resource_limits']);write(OP/'resource-latest.json',dict(tag=tag,unix_time_s=time.time(),**metrics));return metrics
  def monitor():
   while not stop.wait(1.):
    try:guard('watchdog')
    except BaseException as e:
     write(OP/'resource-failure.json',dict(error=repr(e),unix_time_s=time.time()));guard_failure.append(repr(e));os.kill(os.getpid(),__import__('signal').SIGTERM);return
  guard('before_model',reserve_memory_bytes=100*1024**2);thread=threading.Thread(target=monitor,daemon=True);thread.start()
  cpu,gpu=models();nb=NumpyBulkP2Force(cpu);wb=WarpBulkP2Force(gpu,device='cuda:0');dynamics={'numpy_cpu':SupportedBulkDynamics(cpu,nb),'warp_gpu':SupportedBulkDynamics(gpu,wb)}
  for k in ['data','indices','indptr']:np.testing.assert_array_equal(getattr(dynamics['numpy_cpu'].mass,k),getattr(dynamics['warp_gpu'].mass,k))
  controls=[]
  for name,u in [('rest',np.zeros_like(cpu.rest)),('finite_bending',np.column_stack([np.zeros(len(cpu.rest)),np.zeros(len(cpu.rest)),-.001*16*(cpu.rest[:,0]/.24)**2*(1-cpu.rest[:,0]/.24)**2]))]:
   a,b=nb.evaluate(u),wb.evaluate(u);row=dict(state=name,gradient_L2_difference_N=float(np.linalg.norm(a['gradient_N']-b['gradient_N'])),energy_difference_J=abs(a['elastic_energy_J']-b['elastic_energy_J']));da,db=[d.mechanics(d.pack(u,np.zeros_like(u))) for d in dynamics.values()];row['mass_weighted_acceleration_L2_difference_N']=float(np.linalg.norm(dynamics['numpy_cpu'].mass@(da['acceleration_m_s2']-db['acceleration_m_s2'])))
   row['passed']=row['mass_weighted_acceleration_L2_difference_N']<=p['gates']['same_state_mass_weighted_acceleration_L2_N'] and row['gradient_L2_difference_N']<=p['gates']['same_state_gradient_L2_N'] and row['energy_difference_J']<=p['gates']['same_state_energy_J'];controls.append(row)
  summary['same_state_controls']=controls;write(OUT/'summary.json',summary)
  if not all(x['passed'] for x in controls):raise ValueError('Independent original bulk force parity failed before trajectories')
  for config in p['configs']:
   for backend,d in dynamics.items():
    case=backend+'-'+config['id'];begin=time.monotonic();meta=dict(case=case,protocol_sha256=digest,source_sha256=p['source_sha256'],checkpoint_policy=POLICY,rest=array_record(d.model.rest),force=array_record(d.force));
    class GuardedCheckpoint(AcceptedDOPCheckpoint):
     def __call__(self,*args):
      guard('before_checkpoint_'+case,reserve_write_bytes=16*1024**2,reserve_memory_bytes=16*1024**2);return super().__call__(*args)
    checkpoint=GuardedCheckpoint(OUT/case,meta)
    d.rhs_calls=0;d.maximum_RHS_mass_residual_N=0.;d.minimum_RHS_J=float('inf')
    initial=d.pack(np.zeros_like(cpu.rest),np.zeros_like(cpu.rest));atol=np.r_[np.full(d.size,config['displacement_atol_m']),np.full(d.size,config['velocity_atol_m_s'])];answer=streaming_dop853(d.rhs,d.telemetry,initial,p['outputs_s'],rtol=config['rtol'],atol=atol,max_step_s=p['maximum_step_s'],energy_tolerance_J=p['gates']['energy_absolute_J'],checkpoint=checkpoint,checkpoint_every_accepted_steps=POLICY['every_accepted_steps']);u,v=map(np.asarray,zip(*[d.unpack(y) for y in answer['common_states']]));q=u+d.model.rest;bounds=[whole_p2_jacobian_bounds(d.model,x)['minimum_lower_bound'] for x in q]
    if min(bounds)<=0:raise ValueError('Common-output whole-element positivity certificate failed')
    reactions=np.array([d.mechanics(y)['reaction_N'] for y in answer['common_states']]);f=OUT/(case+'.npz');M=d.mass
    guard('before_final_capture_'+case,reserve_write_bytes=16*1024**2,reserve_memory_bytes=16*1024**2)
    np.savez(f,times_s=answer['times_s'],positions_m=q,displacements_m=u,velocities_m_s=v,reactions_N=reactions,rest_m=d.model.rest,elements=d.model.elements,tets=d.model.tets,fixed_mask=d.mask,known_body_force_N=d.force,mass_data_kg=M.data,mass_indices=M.indices,mass_indptr=M.indptr,mass_shape=M.shape,common_whole_J_lower=np.array(bounds),accepted_times_s=np.array([r['time_s'] for r in answer['accepted_scalar_ledger']]))
    telemetry=answer.pop('accepted_scalar_ledger');write(OUT/case/'accepted-ledger.json',telemetry);row=dict(id=case,completed=True,passed=True,backend=backend,config=config,elapsed_s=time.monotonic()-begin,accepted_steps=answer['accepted_steps'],nfev=answer['nfev'],rhs_calls=d.rhs_calls,maximum_RHS_mass_residual_N=d.maximum_RHS_mass_residual_N,minimum_RHS_J=d.minimum_RHS_J,minimum_common_whole_J_lower=min(bounds),maximum_energy_error_J=max(abs(r['mechanical_energy_J']-answer['initial_mechanical_energy_J']) for r in telemetry+answer['common_telemetry']),checkpoint_commits=answer['checkpoint_commits'],maximum_observed_uncommitted_accepted_steps=answer['maximum_observed_uncommitted_accepted_steps'],capture=record(f),accepted_scalar_ledger=record(OUT/case/'accepted-ledger.json'));write(OUT/(case+'.json'),row);summary['cases'].append(row);write(OUT/'summary.json',summary);print(json.dumps(row),flush=True)
  pairs=[];refinements=[]
  for config in p['configs']:
   with np.load(OUT/('numpy_cpu-'+config['id']+'.npz')) as a,np.load(OUT/('warp_gpu-'+config['id']+'.npz')) as b:
    np.testing.assert_array_equal(a['times_s'],b['times_s']);pairs.append(dict(id=config['id'],**pair(a['positions_m'],a['velocities_m_s'],b['positions_m'],b['velocities_m_s'],qgate=1e-8,vgate=1e-5)))
  for backend in dynamics:
   with np.load(OUT/(backend+'-accuracy1.npz')) as a,np.load(OUT/(backend+'-accuracy2.npz')) as b:refinements.append(dict(id=backend,**pair(a['positions_m'],a['velocities_m_s'],b['positions_m'],b['velocities_m_s'],qgate=20e-6,vgate=1e-4)))
  summary.update(completed=True,backend_parity_passed=all(x['passed'] for x in pairs),temporal_pair_passed=all(x['passed'] for x in refinements),implementation_pairs=pairs,accuracy_refinements=refinements)
 except BaseException as e:summary.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  if 'stop' in locals():stop.set()
  if 'thread' in locals():thread.join(timeout=2)
  summary['elapsed_s']=time.monotonic()-started;write(OUT/'summary.json',summary)
 return summary

def launch(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('No automatic retry')
 if sha(p['taskset']['path'])!=p['taskset']['sha256']:raise ValueError('Changed taskset')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 if GPU in apps:raise RuntimeError('Declared GPU0 is not idle')
 e=dict(terminal=False,completed=False,controller_pid=os.getpid(),protocol_sha256=digest);write(OP/'execution.json',e);begin=time.monotonic()
 def on_child(pid):e['worker_pid']=pid;write(OP/'execution.json',e)
 try:
  with (OP/'worker.log').open('x') as log:
   e.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_supported_gravity_parity','run','--sha',digest],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=p['wall_cap_s'],on_child=on_child))
 except BaseException as error:e['controller_error']=repr(error)
 finally:
  s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=e.get('exit_code')==0 and s.get('completed') is True,backend_parity_passed=s.get('backend_parity_passed',False),elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e)
 return e

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','launch','run']);a.add_argument('--sha');args=a.parse_args();r=prepare() if args.action=='prepare' else launch(args.sha) if args.action=='launch' else run(args.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','backend_parity_passed','temporal_pair_passed','elapsed_s','error']}));raise SystemExit(0 if args.action=='prepare' or r.get('completed') else 1)
