"""Fixed96tet Q8 Warp trajectory bridge against completed Torch reference.

Manual public DOP853.step() is the same adaptive algorithm as solve_ivp.
Dense interpolation is requested only when a fixed output is crossed. All
accepted states receive conservation checks and durable uncompressed snapshots.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time,traceback
import numpy as np
from scipy.integrate import DOP853
from experiments.run_p2_contact_dop853 import ROOT,GPU,sha,write,build,SCIPY_ROOT,SCIPY_FILES,CONFIGS,SOURCES as OLD_SOURCES
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
OP=ROOT/'runs/operator-dev/p2-warp-96-bridge-v02';OUT=ROOT/'runs/gaussian-campaign-v01/p2-warp-96-bridge-v02';PRIOR=ROOT/'runs/gaussian-campaign-v01/p2-contact-dop853-v03';OLD_PROTOCOL=ROOT/'runs/operator-dev/p2-contact-dop853-v03/protocol.json'
CONTROL=ROOT/'runs/gaussian-campaign-v01/p2-warp-trajectory-v03/summary.json';CONTROL_EXEC=ROOT/'runs/operator-dev/p2-warp-trajectory-v03/execution.json'
from experiments.benchmark_p2_warp_rhs import ENV_SITE
from experiments.continue_p2_contact_dop853_v03 import manual_dop853
from experiments.campaign_p2_torch_contact import quadratic_face_minimum
from experiments.campaign_owned_gpu_budget import LIMITS,GiB,collect_resources,validate_budget,validate_cpu,run_owned_command
CONFIG=dict(name='warp_bridge',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
SOURCES=OLD_SOURCES+['experiments/continue_p2_contact_dop853_v02.py','tests/test_continue_p2_contact_dop853_v02.py','experiments/continue_p2_contact_dop853_v03.py','tests/test_continue_p2_contact_dop853_v03.py','experiments/campaign_p2_warp_rhs.py','experiments/benchmark_p2_warp_rhs.py','experiments/run_p2_warp_trajectory_tight.py','tests/test_run_p2_warp_trajectory_tight.py','experiments/run_p2_warp_96_bridge_v02.py','tests/test_run_p2_warp_96_bridge_v02.py','experiments/campaign_owned_gpu_budget.py','tests/test_campaign_owned_gpu_budget.py']

def bridge_pair(oldq,oldv,oldE,q,v,E,energytol):
 qe=float(np.linalg.norm(oldq-q,axis=2).max());ve=float(np.linalg.norm(oldv-v,axis=2).max());ee=float(abs(oldE-E).max())
 return dict(maximum_allnode_position_difference_m=qe,maximum_allnode_velocity_difference_m_s=ve,maximum_output_energy_difference_J=ee,passed=qe<=1e-8 and ve<=1e-5 and ee<=energytol)

def imported_records():
 cp=PRIOR/'accuracy3.json';case=json.loads(cp.read_text());old=json.loads(OLD_PROTOCOL.read_text());f=ROOT/case['capture']['path']
 if case.get('completed') is not True or case.get('passed') is not True or case['name']!='accuracy3' or case['rtol']!=1e-10 or sha(f)!=case['capture']['sha256']:raise ValueError('Exactcompletedtightercaseonly')
 for k in ['rtol','displacement_atol_m','velocity_atol_m_s','impulse_atol_Ns']:
  if case[k]!=CONFIG[k]:raise ValueError('Differentnumericsettings')
 for name,h in old['source_sha256'].items():
  if sha(ROOT/name)!=h:raise ValueError('Priorphysicalsourcechanged')
 if old['physics']['kappa_N_m3']!=1e9 or old['physics']['E_Pa']!=300000. or old['physics']['surface_Q']!=8:raise ValueError('Changedphysicalcase')
 if not json.loads(CONTROL.read_text()).get('passed') or not json.loads(CONTROL_EXEC.read_text()).get('terminal') or not json.loads((PRIOR/'summary.json').read_text()).get('time_reference_qualified'):raise ValueError('Qualified reference and completed backend control required')
 return {str(p.relative_to(ROOT)):sha(p) for p in [cp,f,OLD_PROTOCOL,PRIOR/'summary.json',CONTROL,CONTROL_EXEC]},case

def estimate_prefix_bytes(n_states,n_outputs,state_dimension):
 return 8*((n_states+n_outputs+2)*state_dimension+6*(n_states+n_outputs))+128*1024

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenstage')
 superseded=ROOT/'runs/operator-dev/p2-warp-96-bridge-v01/protocol.json'
 if (superseded.parent/'execution.json').exists():raise ValueError('Original unexecutedprotocol expected')
 cores=sorted(os.sched_getaffinity(0))[-2:]
 if len(cores)!=2:raise ValueError('Two permitted CPU cores required')
 records,case=imported_records();old=json.loads(OLD_PROTOCOL.read_text());p=dict(schema='Guarded-Fixed96-Q8-Warp-implementation-bridge-v0.2',status='FROZEN_BEFORE_BRIDGE_CALLS',role='Same96tet/Q8/law/clock/tightaccuracy; isolatebackend only; preservealloldresults',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=1500,resource_limits=LIMITS,CPU_affinity_cores=cores,taskset=dict(path='/usr/bin/taskset',sha256=sha(__import__('pathlib').Path('/usr/bin/taskset'))),superseded_unexecuted_protocol=dict(path=str(superseded.relative_to(ROOT)),sha256=sha(superseded)),source_sha256={f:sha(ROOT/f) for f in SOURCES},scipy_source_sha256={f:sha(SCIPY_ROOT/f) for f in SCIPY_FILES+['integrate/_ivp/base.py','integrate/__init__.py']},imported_reference_sha256=records,old_source_pins=old['source_sha256'],physics=old['physics'],installed_warp_source_sha256=json.loads((ROOT/'runs/operator-dev/p2-warp-rhs-v01/gpu-protocol.json').read_text())['installed_warp_source_sha256'],implementation_gates=dict(allnode_q_m=1e-8,allnode_v_m_s=1e-5),config=CONFIG,maximum_step_s=old['maximum_step_s'],interval_s=old['interval_s'],outputs_s=old['outputs_s'],gates=old['gates'],initialization=old['initialization'],checkpoint='Afterevery512acceptedsteps andfinal:atomicUNCOMPRESSED allacceptedstates/times/energy/P, allcompletedcommonoutputs/stats, y/f/h_next forpossiblefutureone-steprestart. Entire committed prefix retained; hardkill may lose up to511 uncommitted accepted tail states.',orchestration='PublicDOP853.step;checkenergy/Paftereveryacceptedstep;dense_outputonlyoncommonquerycrossing',nonclaims=old['nonclaims'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('scipy_source_sha256',SCIPY_ROOT),('imported_reference_sha256',ROOT),('old_source_pins',ROOT),('installed_warp_source_sha256',ENV_SITE)]:
  if any(sha(base/f)!=v for f,v in p[group].items()):raise ValueError('Changedpinnedinput:'+group)
 OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,backend_qualified=False,pid=os.getpid(),protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),cases=[],nonclaims=p['nonclaims']);config=p['config']
 try:
  import torch,scipy,sys
  sys.path.append(str(ENV_SITE));import warp as wp
  if wp.__version__!='1.18.0':raise ValueError('Wrong pinnedWarp')
  wp.config.kernel_cache_dir=str(OP/'warp-cache')
  from experiments.campaign_p2_warp_rhs import WarpP2Force
  from experiments.campaign_p2_torch_contact import PlaneContactP2
  from experiments.campaign_p2_consistent_transient import scalar_mass
  torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());device='cuda:0';torch.cuda.set_per_process_memory_fraction(min(.95,p['resource_limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory),0);torch.ones(1,device=device).sum().item()
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True);own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())];s.update(own_compute_records=own,versions=dict(torch=torch.__version__,scipy=scipy.__version__,numpy=np.__version__));write(OUT/'summary.json',s)
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('WrongphysicalGPU')
  oldcase=json.loads((PRIOR/'accuracy3.json').read_text());s['cases'].append(dict(**oldcase,imported=True))
  with np.load(ROOT/oldcase['capture']['path'],allow_pickle=False) as z:oldq=z['query_positions_m'].copy();oldpositions=z['positions_m'].copy();oldv=z['velocities_m_s'].copy();oldE=z['output_mechanical_energies_J'].copy();oldtimes=z['times_s'].copy();oldrest=z['rest_m'].copy();oldtets=z['tets'].copy()
  m=build([4,2,2],300000.,9.81);np.testing.assert_array_equal(oldtimes,p['outputs_s']);np.testing.assert_array_equal(oldrest,m.rest);np.testing.assert_array_equal(oldtets,m.tets);obj=PlaneContactP2(m,device,kappa_A=1e9,floor_z=0.,surface_order=8);arrays={k:getattr(obj,k).cpu().numpy().copy() for k in WarpP2Force.REQUIRED};w=WarpP2Force(arrays,young=m.young,poisson=m.poisson,kappa_A=1e9,device=device);M=torch.tensor(scalar_mass(m).toarray(),dtype=torch.float64,device=device);L=torch.linalg.cholesky(M);n=len(m.rest);size=3*n;force=obj.force;torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device=device)*torch.tensor(GRAVITY,dtype=torch.float64,device=device),force,atol=1e-13,rtol=1e-13)
  def guard(tag,reserve_write_bytes=0,reserve_memory_bytes=0):
   metrics=collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch);snapshot=dict(tag=tag,unix_time_s=time.time(),**metrics,reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes,protocol_sha256=digest)
   try:validate_budget(metrics,reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes,limits=p['resource_limits'])
   except Exception:
    write(OP/'resource-failure.json',snapshot);raise
   write(OP/'resource-latest.json',snapshot)
   with (OP/'resource-history.jsonl').open('a') as history:history.write(json.dumps(snapshot)+'\n')
  guard('startup_after_model_allocation')
  u0,v0,_,_=initial_vectors(m.rest,1/1920);y0=np.r_[u0.ravel(),v0.ravel(),np.zeros(3)];uv=lambda a:torch.tensor(a,dtype=torch.float64,device=device);initialP=(M@uv(v0)).sum(0).cpu().numpy();cache=w.evaluate(u0);initialE=float(cache['energy']+.5*(uv(v0)*(M@uv(v0))).sum()-(force*uv(u0)).sum());energytol=max(1e-9,1e-6*float(M.sum())*9.81*.05);probe=np.array([[x,y,z] for x in np.linspace(0,.24,9) for y in [-.04,0,.04] for z in [-.04,0,.04]])+np.array([0,0,.09]);stats=dict(rhs_calls=0,maximum_masssolve_R_N=0.,minimum_RHS_J=1.);Et=[];Pt=[];Jmin=1.;penmax=0.;outputE=[];outputP=[];outputpen=[];checkedoutputs=0;started=time.monotonic()
  def rhs(t,y):
   cache=w.evaluate(y[:size].reshape(n,3));net=force-uv(cache['gradient']);acc=torch.cholesky_solve(net,L);R=float(torch.linalg.vector_norm(M@acc-net));stats['rhs_calls']+=1;stats['maximum_masssolve_R_N']=max(stats['maximum_masssolve_R_N'],R);stats['minimum_RHS_J']=min(stats['minimum_RHS_J'],float(cache['minimum_J']))
   if not np.isfinite(R) or R>1e-8:raise ValueError('Originalmasssolvefailed')
   return np.r_[y[size:2*size],acc.detach().cpu().numpy().ravel(),net.sum(0).detach().cpu().numpy()]
  def balance(y):
   u=uv(y[:size].reshape(n,3));v=uv(y[size:2*size].reshape(n,3));cache=w.evaluate(y[:size].reshape(n,3));E=float(cache['energy']+.5*(v*(M@v)).sum()-(force*u).sum());P=(M@v).sum(0).cpu().numpy()-initialP-y[2*size:];facez=(m.rest+y[:size].reshape(n,3))[arrays['surface_elements'],2];minimum,_=quadratic_face_minimum(facez);return E,float(np.linalg.norm(P)),float(cache['minimum_J']),max(0.,-float(minimum.min()))
  def after(info):
   nonlocal Jmin,penmax,checkedoutputs
   E,P,J,pen=balance(info['accepted_states'][-1]);Et.append(E);Pt.append(P);Jmin=min(Jmin,J);penmax=max(penmax,pen)
   while checkedoutputs<len(info['output_states']):
    E,P,J,pen=balance(info['output_states'][checkedoutputs]);outputE.append(E);outputP.append(P);outputpen.append(pen);checkedoutputs+=1
   count=len(info['accepted_times'])-1;solver=info['solver']
   if count%512==0 or solver.status=='finished':
    estimate=estimate_prefix_bytes(len(info['accepted_times']),len(info['output_times']),len(solver.y));guard('before_prefix_'+str(count),reserve_write_bytes=estimate,reserve_memory_bytes=estimate)
    tmp=OUT/'warp_bridge-prefix.tmp.npz';f=OUT/'warp_bridge-prefix.npz';np.savez(tmp,accepted_times_s=np.asarray(info['accepted_times']),accepted_states=np.asarray(info['accepted_states']).T,accepted_energy_J=np.asarray(Et),accepted_momentum_errors_Ns=np.asarray(Pt),common_times_s=np.asarray(info['output_times']),common_states=np.asarray(info['output_states']).T,common_energy_J=np.asarray(outputE),common_momentum_errors_Ns=np.asarray(outputP),common_penetration_m=np.asarray(outputpen),current_y=solver.y,current_f=solver.f,next_h_abs_s=solver.h_abs,accepted_steps=count,nfev=solver.nfev,protocol_sha256=digest,minimum_accepted_J=Jmin,maximum_accepted_penetration_m=penmax);tmp.replace(f)
    write(OUT/'progress.json',dict(accepted_steps=count,accepted_time_s=solver.t,**stats,maximum_accepted_energy_balance_error_J=float(abs(np.asarray(Et)-initialE).max()),maximum_accepted_momentum_balance_error_Ns=max(Pt),prefix_sha256=sha(f)))
  atol=np.r_[np.full(size,config['displacement_atol_m']),np.full(size,config['velocity_atol_m_s']),np.full(3,config['impulse_atol_Ns'])];answer=manual_dop853(rhs,y0,[T0,END],p['outputs_s'],config,atol,after);y=answer['output_states'];q=y[:size].T.reshape(-1,n,3)+m.rest;v=y[size:2*size].T.reshape(-1,n,3);pred=np.asarray([m.query(x,probe) for x in q]);energyerror=max(float(abs(np.asarray(Et)-initialE).max()),float(abs(np.asarray(outputE)-initialE).max()));perror=max(Pt+outputP);passed=energyerror<=energytol and perror<=1e-8 and stats['maximum_masssolve_R_N']<=1e-8
  finalestimate=answer['accepted_states'].nbytes+estimate_prefix_bytes(len(answer['accepted_times']),len(answer['output_times']),len(y0));guard('before_final_capture',reserve_write_bytes=finalestimate,reserve_memory_bytes=finalestimate)
  f=OUT/'warp_bridge.npz';np.savez(f,times_s=answer['output_times'],positions_m=q,velocities_m_s=v,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,surface_elements=obj.surface_elements.cpu().numpy(),fixed_node_ids=np.array([],int),floor_z_m=0.,query_times_s=answer['output_times'],query_positions_m=pred,material_probes_m=probe,accepted_times_s=answer['accepted_times'],accepted_states=answer['accepted_states'],accepted_mechanical_energies_J=np.asarray(Et),accepted_momentum_balance_errors_Ns=np.asarray(Pt),output_mechanical_energies_J=np.asarray(outputE),output_momentum_balance_errors_Ns=np.asarray(outputP),output_analytic_face_penetrations_m=np.asarray(outputpen),volume_integration_weights_kg=m.density*m.body_integral_weights)
  guard('after_final_capture')
  row=dict(**config,completed=True,passed=passed,elapsed_s=time.monotonic()-started,nfev=answer['nfev'],accepted_steps=answer['accepted_steps'],**stats,minimum_accepted_J=Jmin,maximum_accepted_analytic_face_penetration_m=penmax,maximum_accepted_energy_balance_error_J=float(abs(np.asarray(Et)-initialE).max()),maximum_accepted_momentum_balance_error_Ns=max(Pt),maximum_output_energy_balance_error_J=float(abs(np.asarray(outputE)-initialE).max()),maximum_output_momentum_balance_error_Ns=max(outputP),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)));s['cases'].append(row);write(OUT/'warp_bridge.json',row);pair=bridge_pair(oldpositions,oldv,oldE,q,v,np.asarray(outputE),energytol);s.update(completed=True,backend_qualified=all(c['passed'] for c in s['cases']) and pair['passed'],accuracy_pair=pair,initial_mechanical_energy_J=initialE,energy_tolerance_J=energytol,maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),backend='Warpforce+sameTorchfullMCholesky',warp_version=wp.__version__)
 except Exception as e:s.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 return s

def launch(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 p=json.loads((OP/'protocol.json').read_text())
 if sha(__import__('pathlib').Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(r.split(',')[1].strip()==GPU for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU1notidle')
 e=dict(controller_pid=os.getpid(),terminal=False,completed=False,wall_cap_s=1500,protocol_sha256=digest,CPU_affinity_cores=p['CPU_affinity_cores']);write(OP/'execution.json',e);begin=time.monotonic()
 def child_started(pid):e['worker_pid']=pid;write(OP/'execution.json',e)
 try:
  with (OP/'worker.log').open('x') as log:
   command=[p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_warp_96_bridge_v02','run','--sha',digest]
   cleanup=run_owned_command(command,environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=1500,on_child=child_started);e.update(cleanup)
 except BaseException as error:e.update(controller_error=repr(error),exit_code=None)
 finally:
  s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=e.get('exit_code')==0 and s.get('completed') is True,backend_qualified=s.get('backend_qualified',False),elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e)
 return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','backend_qualified','elapsed_s','error','exit_code']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
