"""NEW fixed four-trajectory C3 midpoint pipeline. Source-only until root release.

Every accepted state has original checks before durable state publication.
No training, observations, admission or automatic retry/resume.
"""
import argparse,ast,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments.campaign_long_trajectory_runtime_v02 import GPUS,JOBS,sha,record,atomic_json,append_ledger,validate_lease,verify_pins,checkpoint,load_checkpoint,supervise,watchdog
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/long-smooth-trajectory-v02'
OUT=ROOT/'runs/gaussian-campaign-v01/long-smooth-trajectory-v02'
SOURCE='experiments/run_long_smooth_trajectory_v02.py'
CAP=ROOT/'runs/operator-dev/long-midpoint-capacity-v01'


def sources():
 seen=set();pending=[SOURCE,'tests/test_campaign_long_trajectory_runtime_v02.py','tests/test_run_long_smooth_trajectory_v02.py','tests/test_campaign_long_trajectory_design.py']
 while pending:
  name=pending.pop()
  if name in seen:continue
  seen.add(name)
  for n in ast.walk(ast.parse((ROOT/name).read_text())):
   if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments.'):
    path='experiments/'+n.module.split('.')[1]+'.py'
    if (ROOT/path).exists():pending.append(path)
 return sorted(seen)


def prepare():
 # Only metadata/source hashing. Never opens physical arrays or calls CUDA/FE.
 if OP.exists():raise FileExistsError('Exclusive new execution namespace required')
 cap=json.loads((CAP/'protocol.json').read_text());terminal=json.loads((CAP/'terminal-proof.json').read_text())
 if not terminal['completed'] or not terminal['controller_worker_and_owned_group_absent']:raise ValueError('Capacity terminal prerequisite')
 OP.mkdir(parents=True)
 pins={str(ROOT/n):sha(ROOT/n) for n in sources()}
 for r in cap['static_prerequisites']+[x['capture'] for x in cap['input_fields']]:pins[str(ROOT/r['path'])]=r['sha256']
 for n in ['protocol.json','compact-summary.json','terminal-proof.json','completed-publication-inventory.json']:pins[str(CAP/n)]=sha(CAP/n)
 p=dict(schema='actual-C3-long-trajectory-v02',status='FROZEN_PENDING_ROOT_FULL_SOURCE_REVIEW_NO_PHYSICS',source_pins=pins,
        inputs=cap['input_fields'],jobs=list(JOBS),GPUs=list(GPUS),CPU_cores=[[0,1],[2,3],[4,5]],
        physics=cap['physics'],pulse=dict(duration_s=1.,formula='256*s**4*(1-s)**4 on0<s<1 else0',surface_force_N=[0.,0.,-2.],initial_actual_time_s=0.,initial_velocity='EXACT_ZERO'),
        gates=cap['gates'],clocks_s=(np.arange(9)/8).tolist(),
        temporal=dict(steps=[32,64,128],mesh=[40,20,20],required_observed_order_minimum=1.,indicator_factor=1.,allnode_and63queries=True),
        spatial=dict(meshes=[[32,16,16],[40,20,20]],steps=128,actual_h_conditional_order_minimum=1.,queries=63),
        combined_gates=dict(q_m=20e-6,v_m_s=100e-6,temporal_plus_spatial=True),
        resources=dict(per_trajectory_wall_cap_s=12600,global_wall_cap_s=13500,per_step_cap_s=600,cleanup_per_owned_group_s=3,max_owned_groups=3,absolute_cleanup_cap_s=12,CPU_threads_per_job=2,GPU_bytes=8*1024**3,RSS_bytes=12*1024**3,filesystem_free_reserve_bytes=30*1024**3,projected_remaining_writes_reserve_bytes=3*1024**3,global_projected_concurrent_writes_reserve_bytes=9*1024**3,own_disk_bytes=30*1024**3,max_concurrent=3),
        taskset=cap['taskset'],installed_warp_API_sha256=cap['installed_warp_API_sha256'],
        failure_policy='All four fixed planned. First three fine40 jobs launch concurrently; after h32 terminalPASS+reap slot0 launches coarse32h128. Any physical/resource/technical failure cancels queued launch and drains all owned active peers; retain accepted states and all denominators. No early pair gate, retries, resume or extra trajectories.',
        supersedes_unexecuted_design='runs/operator-dev/long-smooth-trajectory-design-v01/PLAN.md; no files edited',
        work_energy='Report actual nonlinear midpoint potential defect separately; not exact nonlinear energy conservation',
        checkpoint='Every accepted state: immutable generation, data+metadata hashes, atomic pointer last, current+previous retained; nine immutable clock q/v captures. No partial Newton states accepted.',
        admission=False,raw_arrays_LOCAL=True,forecast_hours=dict(GPU_hours=6.649,wall_hours=2.911,method='Measured81.88s fine40/43.69s32 whole-check step multiplications; reuse speedups unmeasured'))
 atomic_json(OP/'protocol.json',p);return sha(OP/'protocol.json')


def validate(digest):
 p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest:raise ValueError('Protocol digest mismatch')
 verify_pins(p['source_pins'])
 if p['jobs']!=list(JOBS) or p['GPUs']!=list(GPUS) or p['CPU_cores']!=[[0,1],[2,3],[4,5]] or p['resources']['global_wall_cap_s']!=13500 or p['resources']['per_trajectory_wall_cap_s']!=12600:raise ValueError('Exact fixed job/resource matrix required')
 if p['pulse']!=dict(duration_s=1.,formula='256*s**4*(1-s)**4 on0<s<1 else0',surface_force_N=[0.,0.,-2.],initial_actual_time_s=0.,initial_velocity='EXACT_ZERO'):raise ValueError('Frozen physical prehistory/pulse required')
 verify_pins({p['taskset']['path']:p['taskset']['sha256']})
 from experiments.run_campaign_c38_graded_family_warp import ENV_SITE
 verify_pins({str(ENV_SITE/n):h for n,h in p['installed_warp_API_sha256'].items()})
 return p


def restore_signals():
 if hasattr(signal,'pthread_sigmask'):signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})


def install_worker_limits(seconds,step_seconds):
 restore_signals()
 def stop(signum,frame):raise TimeoutError('Worker hard/step wall cap or owned stop signal '+str(signum))
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,min(seconds,step_seconds))


def worker(p,job,digest):
 from experiments.run_campaign_c38_graded_family_warp import ENV_SITE
 from experiments.campaign_owned_gpu_budget_v02 import validate_cpu,collect_resources,validate_budget
 slot=job['slot'];gpu=GPUS[slot];cores=p['CPU_cores'][slot];validate_lease(job,os.environ.get('CUDA_VISIBLE_DEVICES'),list(os.sched_getaffinity(0)))
 folder=OUT/job['id'];folder.mkdir(parents=True,exist_ok=False);begin=time.monotonic();hard_deadline=begin+12600
 result=dict(completed=False,passed=False,job=job,pid=os.getpid(),protocol_sha256=digest,accepted_steps=0,admission=False)
 def save():result['seconds']=time.monotonic()-begin;atomic_json(folder/'result.json',result)
 install_worker_limits(12600,600)
 # Independent external watchdog covers blocked native kernels, RSS and main alarm.
 import multiprocessing
 atomic_json(folder/'step-deadline.json',dict(deadline_monotonic_s=min(hard_deadline,begin+600)))
 watcher=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),hard_deadline,folder/'step-deadline.json',12*1024**3,folder/'resource-guard.json'),daemon=True);watcher.start()
 try:
  import torch
  sys.path.append(str(ENV_SITE));import warp as wp
  wp.config.kernel_cache_dir=str(folder/'warp-cache');torch.set_num_threads(2);torch.set_num_interop_threads(2)
  validate_cpu(cores,os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads())
  if torch.cuda.device_count()!=1:raise ValueError('Exactly one declared GPU visible')
  torch.cuda.set_per_process_memory_fraction(min(1.,8*1024**3/torch.cuda.get_device_properties(0).total_memory),0);torch.zeros(1,device='cuda:0').sum().item()
  def resources():
   metrics=collect_resources(gpu_uuid=gpu,pid=os.getpid(),roots=[folder],torch=torch);validate_budget(metrics,reserve_write_bytes=9*1024**3);return metrics
  result['startup_resources']=resources();save()
  from experiments.campaign_long_supported_block import LongSupportedGeometry,StreamingLongSupportedP2,compile_forces
  from experiments.campaign_c38_verified_warp_solver import IndependentlyCheckedWarpCGMixin
  from experiments.campaign_p2_smooth_midpoint import midpoint_step,smooth_bump
  from experiments.campaign_p2_transient import P2State
  from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.campaign_long_trajectory_design import full_mass_momentum_ledger
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
  result['hmax_m']=m.maximum_reference_tet_edge_m;result['initial_field']=source
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


def aggregate(p,digest):
 from experiments.campaign_long_trajectory_design import convergence,pair_difference
 results={j['id']:json.loads((OUT/j['id']/'result.json').read_text()) for j in JOBS}
 if not all(r['completed'] and r['passed'] and r['accepted_steps']==r['job']['steps'] and r['protocol_sha256']==digest for r in results.values()):raise ValueError('All four terminal physical trajectories with matching protocol required')
 for key,r in results.items():
  verify_pins({r[k]['path']:r[k]['sha256'] for k in ['accepted_ledger','geometry','final_checkpoint_commit']})
  rows=[json.loads(line) for line in Path(r['accepted_ledger']['path']).read_text().splitlines()]
  if len(rows)!=r['job']['steps'] or any(not row['passed'] or row['step']!=i+1 or row['time_s']!=(i+1)/r['job']['steps'] for i,row in enumerate(rows)):raise ValueError('Complete unchanged accepted-step ledger required')
  last=load_checkpoint(OUT/key/'checkpoints',digest,key)[2]
  if last['step']!=r['job']['steps'] or last['time_s']!=1.:raise ValueError('Final accepted checkpoint required')
 arrays={}
 for key,r in results.items():
  q=[];v=[];pq=[];pv=[]
  if [c['time_s'] for c in r['clocks']]!=p['clocks_s']:raise ValueError('Nine exact accepted clocks required')
  for c in r['clocks']:
   verify_pins({c['capture']['path']:c['capture']['sha256']})
   with np.load(c['capture']['path'],allow_pickle=False) as f:q.append(f['q']);v.append(f['v']);pq.append(f['probe_q']);pv.append(f['probe_v'])
  arrays[key]=dict(q=np.array(q),v=np.array(v),pq=np.array(pq),pv=np.array(pv))
 a,b,c,d=[arrays[j['id']] for j in JOBS];probe=convergence(a['pq'],b['pq'],c['pq'],a['pv'],b['pv'],c['pv'],d['pq'],c['pq'],d['pv'],c['pv'],results[JOBS[3]['id']]['hmax_m'],results[JOBS[2]['id']]['hmax_m'])
 temporal={}
 for field,gate in [('q',20e-6),('v',100e-6)]:
  d0,d1=pair_difference(a[field],b[field]),pair_difference(b[field],c[field]);temporal[field]=dict(coarse_delta=d0,fine_delta=d1,p_at_least_one=bool(d0>=2*d1),passed=bool(d0>=2*d1 and d1<=gate))
 summary=dict(completed=True,physical_trajectories_passed=True,conditional_convergence_passed=bool(probe['passed'] and all(x['passed'] for x in temporal.values())),protocol_sha256=digest,probe_combined=probe,allnode_temporal=temporal,admission=False,raw_fields_LOCAL=True,results={k:record(OUT/k/'result.json') for k in results})
 atomic_json(OP/'convergence-summary.json',summary);return summary


def controller(digest):
 restore_signals();p=validate(digest)
 if OUT.exists():raise FileExistsError('No retry/overwrite')
 OUT.mkdir();logs=[]
 # Read-only preflight; no allocation. Existing unrelated processes block only their leased UUID.
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
 if any(g in apps for g in GPUS):raise RuntimeError('All leased GPU0..2 must be idle before launch')
 import shutil
 if shutil.disk_usage(OUT).free-9*1024**3<30*1024**3:raise RuntimeError('Three projected3GiB writers plus30GiB filesystem reserve required')
 def spawn(job):
  slot=job['slot'];validate_lease(job,GPUS[slot],p['CPU_cores'][slot]);f=(OP/(job['id']+'.log')).open('x');logs.append(f)
  command=[p['taskset']['path'],'-c',','.join(map(str,p['CPU_cores'][slot])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_long_smooth_trajectory_v02','worker','--sha',digest,'--job',job['id']]
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPUS[slot],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},cwd=ROOT)
  atomic_json(OP/(job['id']+'-launch.json'),dict(pid=child.pid,controller_pid=os.getpid(),job=job,protocol_sha256=digest,command=command));return child
 def terminal(job):
  path=OUT/job['id']/'result.json'
  if not path.exists():return False
  r=json.loads(path.read_text());passed=r['completed'] and r['passed'] and r['accepted_steps']==job['steps'] and r['protocol_sha256']==digest
  # Preserve exact completed metadata outside the live output namespace.
  cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(sha(path)+'.json');cache.parent.mkdir(parents=True,exist_ok=True)
  if cache.exists():
   if cache.read_bytes()!=path.read_bytes():raise ValueError('Exact metadata cache mismatch')
  else:
   with cache.open('xb') as f:f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
  return passed
 try:
  passed=supervise(list(JOBS),spawn,terminal,global_cap_s=13500,per_job_cap_s=12600,receipt_path=OP/'execution.json')
  if passed:aggregate(p,digest)
  return passed
 finally:
  for f in logs:f.close()


def start(digest):
 validate(digest)
 if (OP/'controller-start.json').exists() or OUT.exists():raise FileExistsError('No duplicate start')
 with (OP/'controller.log').open('x') as log:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_long_smooth_trajectory_v02','controller','--sha',digest],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
 atomic_json(OP/'controller-start.json',dict(controller_pid=child.pid,protocol_sha256=digest,durable=True,UI_closure_independent=True));return child.pid

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','controller','worker']);parser.add_argument('--sha');parser.add_argument('--job');args=parser.parse_args()
 if args.action=='prepare':print(prepare())
 elif args.action=='start':print(start(args.sha))
 elif args.action=='controller':raise SystemExit(0 if controller(args.sha) else 1)
 else:
  restore_signals();p=validate(args.sha);job=next(j for j in JOBS if j['id']==args.job);worker(p,job,args.sha)
