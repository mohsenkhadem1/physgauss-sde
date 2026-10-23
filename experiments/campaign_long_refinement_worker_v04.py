"""New single fine40h256 worker; unchanged v03 physics/check anatomy.

Infrastructure interruptions preserve accepted checkpoints; continuation requires
an independently reviewed segment recipe. This module never resumes/retries.
"""
import json,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.campaign_long_trajectory_runtime_v03 import sha,record,atomic_json,append_ledger,verify_pins,checkpoint,load_checkpoint,watchdog
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/gaussian-campaign-v01/long-smooth-refinement-v04'
GPUS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b',)
JOB=dict(id='fine40_h256',cells=[40,20,20],steps=256,slot=0)

def restore_signals():
 if hasattr(signal,'pthread_sigmask'):signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})

def install_worker_limits(seconds,step_seconds):
 restore_signals()
 def stop(signum,frame):raise TimeoutError('Worker hard/step wall cap or owned stop signal '+str(signum))
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,min(seconds,step_seconds))

def validate_lease(job,gpu,cores):
 if job!=JOB or gpu!=GPUS[0] or cores!=[0,1]:raise ValueError('Only frozen fine40h256/GPU0/CPU0,1 lease permitted; GPU3 NEVER')
 return True

def worker(p,job,digest):
 from experiments.run_campaign_c38_graded_family_warp import ENV_SITE
 from experiments.campaign_owned_gpu_budget_v02 import validate_cpu,collect_resources,validate_budget
 slot=job['slot'];gpu=GPUS[slot];cores=p['CPU_cores'][slot];validate_lease(job,os.environ.get('CUDA_VISIBLE_DEVICES'),list(os.sched_getaffinity(0)))
 folder=OUT/job['id'];folder.mkdir(parents=True,exist_ok=False);begin=time.monotonic();hard_deadline=begin+24000
 result=dict(completed=False,passed=False,job=job,pid=os.getpid(),protocol_sha256=digest,accepted_steps=0,admission=False)
 def save():result['seconds']=time.monotonic()-begin;atomic_json(folder/'result.json',result)
 install_worker_limits(24000,600)
 # Independent external watchdog covers blocked native kernels, RSS and main alarm.
 import multiprocessing
 atomic_json(folder/'step-deadline.json',dict(deadline_monotonic_s=min(hard_deadline,begin+600)))
 watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),hard_deadline,folder/'step-deadline.json',12*1024**3,folder/'resource-guard.json'),daemon=True);watcher.start();result['watchdog_pid']=watcher.pid
 try:
  import torch
  sys.path.append(str(ENV_SITE));import warp as wp
  wp.config.kernel_cache_dir=str(folder/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2)
  validate_cpu(cores,os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads())
  if torch.cuda.device_count()!=1:raise ValueError('Exactly one declared GPU visible')
  torch.cuda.set_per_process_memory_fraction(min(1.,8*1024**3/torch.cuda.get_device_properties(0).total_memory),0);torch.zeros(1,device='cuda:0').sum().item()
  def resources():
   metrics=collect_resources(gpu_uuid=gpu,pid=os.getpid(),roots=[folder],torch=torch);validate_budget(metrics,reserve_write_bytes=3*1024**3);return metrics
  result['startup_resources']=resources();save()
  from experiments.campaign_long_supported_block import LongSupportedGeometry,StreamingLongSupportedP2,compile_forces
  from experiments.campaign_c38_verified_warp_solver import IndependentlyCheckedWarpCGMixin
  from experiments.campaign_p2_smooth_midpoint import midpoint_step,smooth_bump
  from experiments.campaign_p2_transient import P2State
  from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.campaign_long_trajectory_design_v03 import full_mass_momentum_ledger
  class Solver(IndependentlyCheckedWarpCGMixin,StreamingLongSupportedP2):
   def evaluate(self,u):
    value=super().evaluate(u);resources();return value
   def tangent(self,c,v):
    value=super().tangent(c,v);self.resource_calls=getattr(self,'resource_calls',0)+1
    if self.resource_calls%16==0:resources()
    return value
  initialized=time.monotonic();m=LongSupportedGeometry(tuple(job['cells']),support='both_ends');m80=LongSupportedGeometry(tuple(job['cells']),support='both_ends',rule='exact80')
  op=Solver(m,'cuda:0',4000,8,support_kind='both_ends');op80=Solver(m80,'cuda:0',4000,8,support_kind='both_ends');M=scalar_mass(m)
  result['constructor_seconds']=time.monotonic()-initialized;result['constructor_resources']=resources();source=next(r['capture'] for r in p['inputs'] if r['cells']==job['cells']);verify_pins({str(ROOT/source['path']):source['sha256']})
  with np.load(ROOT/source['path'],allow_pickle=False) as f:
   for name,key in [('rest','rest_m'),('tets','tets'),('elements','elements'),('faces','faces'),('mask','constraint_mask'),('load','known_total_force_N')]:
    if not np.array_equal(getattr(m,name),f[key]):raise ValueError('Exact own gravity-equilibrium geometry/BC/force required')
   u0=f['displacement_m'].copy()
  state=P2State(torch.tensor(u0,device='cuda:0'),torch.zeros_like(op.rest),0.)
  components=compile_forces(m,[0.,0.,-2.]);body=torch.tensor(components['body_N'],device='cuda:0');surface=torch.tensor(components['surface_N'],device='cuda:0');n=job['steps'];h=1/n
  points=np.stack(np.meshgrid(np.linspace(0,.48,7),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3)
  np.savez_compressed(folder/'geometry.npz',rest_m=m.rest,elements=m.elements,tets=m.tets,faces=m.faces,constraint_mask=m.mask,points_m=points)
  result['hmax_m']=m.maximum_reference_tet_edge_m;result['initial_field']=source;result['geometry']=record(folder/'geometry.npz');result['driver_provenance']=p['driver_provenance']
  def clock(k,s):
   path=folder/f'clock-{k:03d}.npz'
   with path.open('xb') as f:np.savez_compressed(f,q=m.rest+s.u.cpu().numpy(),u=s.u.cpu().numpy(),v=s.v.cpu().numpy(),probe_q=m.query(m.rest+s.u.cpu().numpy(),points),probe_v=m.query(s.v.cpu().numpy(),points));f.flush();os.fsync(f.fileno())
   return record(path)
  checkpoint(folder/'checkpoints',u0,np.zeros_like(u0),0,0.,digest,job['id']);result['clocks']=[dict(time_s=0.,capture=clock(0,state))];save()
  accumulated_P=np.zeros(3);accumulated_L=np.zeros(3);total_work=total_defect=total_abs_defect=total_stationarity=0.
  for k in range(1,n+1):
   step_begin=time.monotonic();atomic_json(folder/'step-deadline.json',dict(deadline_monotonic_s=min(hard_deadline,step_begin+600)));signal.setitimer(signal.ITIMER_REAL,min(600,max(.001,hard_deadline-step_begin)))
   F=body+surface*smooth_bump((k-.5)*h,1.);old=state;linear_begin=len(getattr(op,'independent_linear_checks',[]));answer,ledger=midpoint_step(op,old,h,F);torch.cuda.synchronize();solve_seconds=time.monotonic()-step_begin
   mid=(old.u+answer.u)/2;acceleration=(answer.v-old.v)/h;Ma=op.mass(acceleration);c24=[op.evaluate(z) for z in [old.u,mid,answer.u]];c80=[op80.evaluate(z) for z in [old.u,mid,answer.u]]
   r80=Ma+c80[1]['gradient']-F;reaction=r80.masked_fill(~op.mask,0.);balance=F+reaction-Ma;R80=float(torch.linalg.vector_norm(r80[~op.mask]));gd=float(torch.linalg.vector_norm(c24[1]['gradient']-c80[1]['gradient']));ed=max(abs(float(a['energy']-b['energy'])) for a,b in zip(c24,c80));force_defect=float(torch.linalg.vector_norm(balance.sum(0)));moment_defect=float(torch.linalg.vector_norm(torch.linalg.cross(op.rest+mid,balance,dim=1).sum(0)))
   du=answer.u-old.u;dU=float(c80[2]['energy']-c80[0]['energy']);dK=.5*float((answer.v*op.mass(answer.v)).sum()-(old.v*op.mass(old.v)).sum());work=float((F*du).sum());defect=dU-float((c80[1]['gradient']*du).sum());stationarity=float((r80*du).sum());identity=dU+dK-work-defect-stationarity
   qm=m.rest+mid.cpu().numpy();qe=m.rest+answer.u.cpu().numpy();certificates=[whole_p2_jacobian_bounds(m80,q) for q in [qm,qe]];certs=[dict(all_elements_positive=c['all_elements_positive'],minimum_lower_bound=c['minimum_lower_bound']) for c in certificates]
   # Full CPU CSR gives an independent mass/momentum action, including M_cf.
   old_u,old_v,new_u,new_v=[z.cpu().numpy() for z in [old.u,old.v,answer.u,answer.v]];f_np=F.cpu().numpy();reaction_np=reaction.cpu().numpy()
   momentum=full_mass_momentum_ledger(m.rest,old_u,old_v,new_u,new_v,M,f_np,reaction_np,h)
   Pdelta=(M@new_v-M@old_v).sum(0)-h*(f_np+reaction_np).sum(0);Ldelta=np.cross(m.rest+new_u,M@new_v).sum(0)-np.cross(m.rest+old_u,M@old_v).sum(0)-h*np.cross(qm,f_np+reaction_np).sum(0);accumulated_P+=Pdelta;accumulated_L+=Ldelta
   cg=op.independent_linear_checks[linear_begin:];mass_parity=float(np.linalg.norm((M@((new_v-old_v)/h))-Ma.cpu().numpy()))
   passed=all(float(c['minimum_J'])>.1 for c in c24+c80) and R80<=1e-9 and gd<=1e-10 and ed<=1e-11 and force_defect<=1e-8 and moment_defect<=1e-8 and abs(identity)<=1e-10 and abs(ledger['ledger_identity_defect_J'])<=1e-10 and all(c['all_elements_positive'] for c in certs) and all(c['passed'] for c in cg) and mass_parity<=1e-9 and momentum['linear_residual_N_s']<=h*1e-8 and momentum['angular_residual_N_m_s']<=h*1e-8 and np.linalg.norm(accumulated_P)<=1e-8 and np.linalg.norm(accumulated_L)<=1e-8
   row=dict(step=k,time_s=k*h,midpoint_time_s=(k-.5)*h,force_coefficient=smooth_bump((k-.5)*h),passed=bool(passed),ledger24=ledger,original_free_R80_N=R80,gradient24_80_difference_N=gd,energy24_80_difference_J=ed,force_balance_N=force_defect,moment_balance_Nm=moment_defect,work80_J=work,potential_midpoint_defect80_J=defect,stationarity80_J=stationarity,energy_work_identity80_J=identity,whole_P2_J=certs,independent_CG_records=cg,mass_CPU_GPU_difference_N=mass_parity,momentum=momentum,cumulative_linear_residual_N_s=float(np.linalg.norm(accumulated_P)),cumulative_angular_residual_N_m_s=float(np.linalg.norm(accumulated_L)),solve_seconds=solve_seconds,resources=resources())
   if not passed:append_ledger(folder/'failed-candidate.jsonl',row);raise RuntimeError('Original residual/quadrature/J/work/fullM/momentum gate failed')
   # Checks complete before making this candidate an accepted state.
   checkpoint(folder/'checkpoints',new_u,new_v,k,k*h,digest,job['id'],accepted_ledger=row);state=answer
   if k%(n//8)==0:result['clocks'].append(dict(time_s=k*h,capture=clock(k//(n//8),state)))
   total_work+=work;total_defect+=defect;total_abs_defect+=abs(defect);total_stationarity+=stationarity
   row['whole_step_seconds']=time.monotonic()-step_begin;append_ledger(folder/'accepted-ledger.jsonl',row)
   result.update(accepted_steps=k,last_accepted_time_s=k*h,last_step_seconds=row['whole_step_seconds'],last_solve_seconds=solve_seconds,external_work_J=total_work,potential_defect_signed_J=total_defect,potential_defect_absolute_sum_J=total_abs_defect,stationarity_work_J=total_stationarity,maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),two_level_cache_entries=len(op.coarse_factor_cache));save()
   del c24,c80,mid,Ma,r80,reaction,balance,certificates
  result.update(completed=True,passed=True,accepted_ledger=record(folder/'accepted-ledger.jsonl'),geometry=record(folder/'geometry.npz'),final_checkpoint_commit=record(folder/'checkpoints'/'COMMIT.json'));save()
 except BaseException as e:result['failure']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc());save();raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);save();watcher.terminate();watcher.join(timeout=2)
 return result
