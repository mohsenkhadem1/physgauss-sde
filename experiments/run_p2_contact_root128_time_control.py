"""One fresh loose root128 time control vs completed tight root128; unchanged physics."""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_contact_dop853 import ROOT,sha,write,SCIPY_ROOT
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
from experiments.continue_p2_contact_dop853_v03 import manual_dop853
from experiments.run_p2_warp_surface_q16 import estimate_prefix_bytes
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
from experiments.benchmark_p2_warp_rhs import ENV_SITE
from experiments.run_p2_corrected_contact_reference import pair_metrics
GPU='GPU-5b432ce5-a24a-68ba-150f-682265260a9b'
OP=ROOT/'runs/operator-dev/p2-contact-root128-time-control-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-contact-root128-time-control-v01'
BASE=ROOT/'runs/operator-dev/p2-corrected-contact-order128-trajectory-v01'
CONTROL_CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-order128-trajectory-v01/outer128_accuracy_1e10/trajectory.npz'
EXPECTED='5a2c638041780380c3e7a9dc3afc56036bdb0aec48b1deff4416ff94247a0fee'
CONFIG=dict(name='root128_accuracy_1e9',rtol=1e-9,displacement_atol_m=1e-12,velocity_atol_m_s=1e-10,impulse_atol_Ns=1e-13)

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 prior=json.loads((BASE/'protocol.json').read_text());done=json.loads((BASE/'result-summary.json').read_text())
 tight=done['cases'][0]
 if not done['completed'] or not tight['completed'] or not tight['passed'] or tight['config']['rtol']!=1e-10 or sha(CONTROL_CAPTURE)!=EXPECTED:raise ValueError('Exactcompletedtight128required')
 sources=list(prior['source_sha256'])+['experiments/run_p2_contact_root128_time_control.py','tests/test_run_p2_contact_root128_time_control.py']
 records=[BASE/'protocol.json',BASE/'result-summary.json',BASE/'terminal-verification.json',ROOT/'runs/operator-dev/p2-contact-spatial-design-v01/spatial-design.json']
 p={**prior,'schema':'root128-loose-vs-import-tight-fixed96-time-control-v1','source_sha256':{f:sha(ROOT/f) for f in dict.fromkeys(sources)},'evidence_sha256':{**prior['evidence_sha256'],**{str(f.relative_to(ROOT)):sha(f) for f in records}},'control_capture':dict(path=str(CONTROL_CAPTURE.relative_to(ROOT)),sha256=EXPECTED),'GPU_uuid':GPU,'CPU_affinity_cores':[0,1],'configs':[CONFIG],'case_cap_s':600,'phase_cap_s':600,'corrections':'NO physical/operator correction: one NEW root128 loose accuracy vs exact completed root128 tight at SAME96 mesh. The old root64 time and64->128 surface pair are not a128time qualification.','forecast':dict(completed_tight128_worker_s=tight['elapsed_s'],new_loose_expected_worker_s=[80,250],phase_cap_s=600,estimate_not_bound=True),'pretrajectory_controls':'Four hashbound SAME-GEOMETRY tight128 fields CPU128vsGPU128 E1e-12/g1e-10/J2e-14/fullMR1e-8; independent prior26-state adaptive-plane/bulk proof retained as provenance, not relabelled as new-mesh controls.','execution':'ONE FRESH loose worker; importcompletedtight128 only, no rerun. Hard600s ownedPGTERM/KILL/reap.','nonclaims':['Timepair at fixed96mesh/root128 ONLY; no spatial or continuum or hardfloor/cohort admission','Root64time+surface128velocity sum101.64um/s is retained NOT a128time pass','AuxP shares RK quadrature; consistency not independent integration proof','New finer meshes require separate geometry-appropriate adaptiveCPUplane/GPUcontrols and freshaccuracy pairs','No material/mass/force/energy/contact tolerance changes']}
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('scipy_source_sha256',SCIPY_ROOT),('installed_warp_source_sha256',ENV_SITE),('CPU_evidence_sha256',ROOT),('evidence_sha256',ROOT)]:
  if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changedpin:'+group)
 if sha(CONTROL_CAPTURE)!=EXPECTED or p['configs']!=[CONFIG] or p['phase_cap_s']!=600 or p['case_cap_s']!=600 or p['GPU_uuid']!=GPU or p['CPU_affinity_cores']!=[0,1] or p['physics']['surface']!='root128forceE finitecompliantreferencearea' or p['physics']['cells']!=[4,2,2] or p['maximum_step_s']!=1/30000:raise ValueError('Exactscopeonly')
 return p

def time_pair(a,b,energytol):
 np.testing.assert_array_equal(a['times_s'],b['times_s'])
 r=pair_metrics(a['positions_m'],a['velocities_m_s'],a['output_mechanical_energies_J'],b['positions_m'],b['velocities_m_s'],b['output_mechanical_energies_J'],energytol,a['query_positions_m'],b['query_positions_m'])
 r['accepted_partitions_distinct']=not np.array_equal(a['accepted_times_s'],b['accepted_times_s'])
 r['declared_accuracy_controls_differ']=True
 r['time_reference_qualified']=bool(r['passed'] and r['accepted_partitions_distinct'])
 return r

def run(digest,label):
 p=validate(digest)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('WrongGPU')
 matches=[c for c in p['configs'] if c['name']==label]
 if len(matches)!=1:raise ValueError('Exactdeclaredcaseonly')
 config=matches[0];dest=OUT/label;dest.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,passed=False,pid=os.getpid(),protocol_sha256=digest,config=config)
 try:
  import torch,scipy,sys
  from scipy.linalg import cho_factor,cho_solve
  sys.path.append(str(ENV_SITE));import warp as wp
  from experiments.campaign_p2_warp_reference_order128 import WarpReferenceForce128
  from experiments.run_p2_contact_validation import build
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.campaign_p2_torch_contact import quadratic_face_minimum
  if wp.__version__!='1.18.0':raise ValueError('PinnedWarp1.18required')
  torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());device='cuda:0';torch.cuda.set_per_process_memory_fraction(p['resource_limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.ones(1,device=device).sum().item();wp.config.kernel_cache_dir=str(OP/'warp-cache')
  def guard(tag,reserve_write_bytes=0,reserve_memory_bytes=0):
   r=dict(tag=label+':'+tag,unix_time_s=time.time(),**collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch),reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes);validate_budget(r,reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes,limits=p['resource_limits']);write(OP/(label+'-resource-latest.json'),r)
   with (OP/(label+'-resource-history.jsonl')).open('a') as f:f.write(json.dumps(r)+'\n')
  guard('startup_ownGPU');m=build([4,2,2],300000.,9.81);w=WarpReferenceForce128(m,device=device);n=len(m.rest);size=3*n;Mt=scalar_mass(m).toarray();M=torch.tensor(Mt,dtype=torch.float64,device=device);L=torch.linalg.cholesky(M);force=torch.tensor(m.load,dtype=torch.float64,device=device);torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device=device)*torch.tensor(GRAVITY,dtype=torch.float64,device=device),force,atol=1e-13,rtol=1e-13)
  # Fresh pretrajectory GPU implementation check against CPUcombined backend;
  # the latter is already independently checked against NumPy+adaptiveplane.
  cpu=WarpReferenceForce128(m,device='cpu');Lc=cho_factor(Mt)
  with np.load(CONTROL_CAPTURE,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['rest_m'],m.rest);qs=z['positions_m'];states=[np.ascontiguousarray(qs[i]-m.rest) for i in [2,3,6]];tt=z['accepted_times_s'];j=int(np.argmin(abs(tt-.102)));states.append(np.ascontiguousarray(z['accepted_states'][:size,j].reshape(n,3)))
  controls=[]
  for u in states:
   a=w.evaluate(u);b=cpu.evaluate(u);net=force-torch.tensor(a['gradient'],dtype=torch.float64,device=device);acc=torch.cholesky_solve(net,L);row=dict(E_J=abs(a['energy']-b['energy']),g_L2_N=float(np.linalg.norm(a['gradient']-b['gradient'])),J_difference=abs(a['minimum_J']-b['minimum_J']),mass_R_N=float(torch.linalg.vector_norm(M@acc-net)),acceleration_L2_difference_m_s2=float(np.linalg.norm(acc.cpu().numpy()-cho_solve(Lc,m.load-b['gradient']))));row['passed']=row['E_J']<=1e-12 and row['g_L2_N']<=1e-10 and row['J_difference']<=2e-14 and row['mass_R_N']<=1e-8;controls.append(row)
  write(OP/(label+'-GPU-controls.json'),dict(completed=True,passed=all(r['passed'] for r in controls),controls=controls,protocol_sha256=digest))
  if not all(r['passed'] for r in controls):raise ValueError('CombinedGPUparityfailedBEFOREtrajectory')
  del cpu,states;guard('pretrajectory_parity_passed');s.update(versions=dict(torch=torch.__version__,scipy=scipy.__version__,warp=wp.__version__,numpy=np.__version__),pretrajectory_controls_passed=True);write(dest/'summary.json',s)
  uv=lambda a:torch.tensor(a,dtype=torch.float64,device=device);u0,v0,_,_=initial_vectors(m.rest,1/1920);y0=np.r_[u0.ravel(),v0.ravel(),np.zeros(3)];initialP=(M@uv(v0)).sum(0).cpu().numpy();c=w.evaluate(u0);initialE=float(c['energy']+.5*(uv(v0)*(M@uv(v0))).sum()-(force*uv(u0)).sum());energytol=p['gates']['energy_J'];faces=w.surface.faces.numpy();probe=np.array([[x,y,z] for x in np.linspace(0,.24,9) for y in [-.04,0,.04] for z in [-.04,0,.04]])+np.array([0,0,.09]);stats=dict(rhs_calls=0,maximum_masssolve_R_N=0.,minimum_RHS_J=float('inf'));Et=[];Pt=[];outE=[];outP=[];outpen=[];checked=0;Jmin=float('inf');penmax=0.;clock=time.monotonic()
  def rhs(t,y):
   c=w.evaluate(y[:size].reshape(n,3));net=force-uv(c['gradient']);a=torch.cholesky_solve(net,L);R=float(torch.linalg.vector_norm(M@a-net));stats['rhs_calls']+=1;stats['maximum_masssolve_R_N']=max(stats['maximum_masssolve_R_N'],R);stats['minimum_RHS_J']=min(stats['minimum_RHS_J'],c['minimum_J'])
   if not np.isfinite(R) or R>1e-8:raise ValueError('OriginalfullMsolvefailed')
   return np.r_[y[size:2*size],a.cpu().numpy().ravel(),net.sum(0).cpu().numpy()]
  def balance(y):
   u=y[:size].reshape(n,3);v=uv(y[size:2*size].reshape(n,3));c=w.evaluate(u);E=float(c['energy']+.5*(v*(M@v)).sum()-(force*uv(u)).sum());P=float(np.linalg.norm((M@v).sum(0).cpu().numpy()-initialP-y[2*size:]));minimum,_=quadratic_face_minimum((m.rest+u)[faces,2]);return E,P,c['minimum_J'],max(0.,-float(minimum.min()))
  def after(info):
   nonlocal checked,Jmin,penmax
   E,P,J,pen=balance(info['accepted_states'][-1]);Et.append(E);Pt.append(P);Jmin=min(Jmin,J);penmax=max(penmax,pen)
   while checked<len(info['output_states']):
    E,P,J,pen=balance(info['output_states'][checked]);outE.append(E);outP.append(P);outpen.append(pen);checked+=1
   count=len(info['accepted_times'])-1;solver=info['solver']
   if count%512==0 or solver.status=='finished':
    estimate=estimate_prefix_bytes(len(info['accepted_times']),len(info['output_times']),len(solver.y));guard('prefix_'+str(count),estimate,estimate);f=dest/'accepted-prefix.npz';tmp=dest/'accepted-prefix.tmp.npz';np.savez(tmp,accepted_times_s=np.asarray(info['accepted_times']),accepted_states=np.asarray(info['accepted_states']).T,accepted_energy_J=np.asarray(Et),accepted_momentum_errors_Ns=np.asarray(Pt),common_times_s=np.asarray(info['output_times']),common_states=np.asarray(info['output_states']).T,common_energy_J=np.asarray(outE),common_momentum_errors_Ns=np.asarray(outP),common_penetration_m=np.asarray(outpen),current_y=solver.y,current_f=solver.f,next_h_abs_s=solver.h_abs,accepted_steps=count,nfev=solver.nfev,protocol_sha256=digest,minimum_accepted_J=Jmin,maximum_accepted_penetration_m=penmax);tmp.replace(f);write(dest/'progress.json',dict(accepted_steps=count,accepted_time_s=solver.t,**stats,maximum_accepted_energy_error_J=float(abs(np.asarray(Et)-initialE).max()),maximum_accepted_P_error_Ns=max(Pt),prefix_sha256=sha(f)))
  atol=np.r_[np.full(size,config['displacement_atol_m']),np.full(size,config['velocity_atol_m_s']),np.full(3,config['impulse_atol_Ns'])];answer=manual_dop853(rhs,y0,[T0,END],p['outputs_s'],config,atol,after);y=answer['output_states'];q=y[:size].T.reshape(-1,n,3)+m.rest;v=y[size:2*size].T.reshape(-1,n,3);pred=np.array([m.query(x,probe) for x in q]);Eerror=max(float(abs(np.asarray(Et)-initialE).max()),float(abs(np.asarray(outE)-initialE).max()));Perror=max(Pt+outP);passed=Eerror<=energytol and Perror<=1e-8 and stats['maximum_masssolve_R_N']<=1e-8;guard('final_capture',answer['accepted_states'].nbytes+estimate_prefix_bytes(len(answer['accepted_times']),9,len(y0)),answer['accepted_states'].nbytes);f=dest/'trajectory.npz';np.savez(f,times_s=answer['output_times'],positions_m=q,velocities_m_s=v,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,surface_elements=faces,fixed_node_ids=np.array([],int),floor_z_m=0.,query_positions_m=pred,query_velocities_m_s=np.array([m.query(x,probe) for x in v]),material_probes_m=probe,accepted_times_s=answer['accepted_times'],accepted_states=answer['accepted_states'],accepted_mechanical_energies_J=np.asarray(Et),accepted_momentum_balance_errors_Ns=np.asarray(Pt),output_mechanical_energies_J=np.asarray(outE),output_momentum_balance_errors_Ns=np.asarray(outP),output_analytic_face_penetrations_m=np.asarray(outpen));guard('after_final_capture');s.update(completed=True,passed=passed,integration_elapsed_s=time.monotonic()-clock,accepted_steps=answer['accepted_steps'],nfev=answer['nfev'],**stats,maximum_accepted_energy_error_J=float(abs(np.asarray(Et)-initialE).max()),maximum_output_energy_error_J=float(abs(np.asarray(outE)-initialE).max()),maximum_accepted_momentum_error_Ns=max(Pt),maximum_output_momentum_error_Ns=max(outP),minimum_accepted_J=Jmin,maximum_analytic_face_penetration_m=penmax,initial_energy_J=initialE,energy_tolerance_J=energytol,capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
 except Exception as error:s.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(dest/'summary.json',s)
 return s

def launch(digest):
 p=validate(digest);begin=time.monotonic()
 def cap(*a):raise TimeoutError('Frozen600sphasecap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(600)
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 OUT.mkdir(parents=True,exist_ok=False);e=dict(controller_pid=os.getpid(),terminal=False,completed=False,time_reference_qualified=False,protocol_sha256=digest,workers=[]);write(OP/'execution.json',e);r={}
 try:
  if sha(Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
  if any(GPU in line for line in apps.splitlines()):raise ValueError('GPU0notidle')
  record=dict(case=CONFIG['name'],terminal=False);e['workers'].append(record)
  def started(pid):record['worker_pid']=pid;write(OP/'execution.json',e)
  with (OP/'worker.log').open('x') as log:record.update(run_owned_command([p['taskset']['path'],'--cpu-list','0,1',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_contact_root128_time_control','run','--sha',digest,'--case',CONFIG['name']],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=max(1.,600-(time.monotonic()-begin)),on_child=started))
  record['terminal']=True;f=OUT/CONFIG['name']/'summary.json';r=json.loads(f.read_text()) if f.exists() else {};record['completed']=record.get('exit_code')==0 and r.get('completed') is True
  if not record['completed']:raise RuntimeError('Incompleteworker;preserveandstop')
  if sha(ROOT/r['capture']['path'])!=r['capture']['sha256']:raise ValueError('Changednewcapture')
  arrays=[]
  for f in [CONTROL_CAPTURE,ROOT/r['capture']['path']]:
   with np.load(f,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in ['times_s','positions_m','velocities_m_s','output_mechanical_energies_J','query_positions_m','accepted_times_s']})
  pair=time_pair(*arrays,p['gates']['energy_J']);e.update(completed=True,time_reference_qualified=bool(r['passed'] and pair['time_reference_qualified']),accuracy_pair=pair)
 except BaseException as error:e['error']=repr(error)
 finally:signal.alarm(0);e.update(terminal=True,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e);write(OUT/'summary.json',dict(**e,cases=[r],imported_tight128=p['control_capture'],nonclaims=p['nonclaims']))
 return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');p.add_argument('--case');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha,a.case) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','time_reference_qualified','elapsed_s','error']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
