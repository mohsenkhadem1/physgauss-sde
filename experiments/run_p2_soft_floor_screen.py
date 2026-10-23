"""NEW softer1e6 compliant-floor physical screen; old1e9 failures untouched."""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_contact_dop853 import ROOT,sha,write,SCIPY_ROOT
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
from experiments.continue_p2_contact_dop853_v03 import manual_dop853
from experiments.run_p2_warp_surface_q16 import estimate_prefix_bytes
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,terminate_owned
from experiments.benchmark_p2_warp_rhs import ENV_SITE
from experiments.run_p2_contact_spatial_pairs import spatial_pair
from experiments.compare_p2_contact_spatial_pairs import combine
OP=ROOT/'runs/operator-dev/p2-soft-floor-screen-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-soft-floor-screen-v01'
SOURCE='experiments/run_p2_soft_floor_screen.py';TEST='tests/test_run_p2_soft_floor_screen.py'
CONFIG=dict(name='accuracy_1e10',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)
JOBS=[dict(id='96tets',cells=[4,2,2],GPU_uuid='GPU-5b432ce5-a24a-68ba-150f-682265260a9b',CPU_affinity_cores=[0,1],wall_cap_s=600),dict(id='324tets',cells=[6,3,3],GPU_uuid='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb',CPU_affinity_cores=[2,3],wall_cap_s=900),dict(id='768tets',cells=[8,4,4],GPU_uuid='GPU-571378aa-c44e-ffb8-b346-1e409f9821f9',CPU_affinity_cores=[4,5],wall_cap_s=1800)]

def prepare():
 base=ROOT/'runs/operator-dev/p2-contact-spatial-pairs-v01/768tets/protocol-v02.json';prior=json.loads(base.read_text());cpu=ROOT/'runs/operator-dev/p2-soft-floor-controls-v01';cp=json.loads((cpu/'CPU-protocol.json').read_text());proof=json.loads((cpu/'CPU-result.json').read_text())
 if not proof['completed'] or not proof['passed'] or len(proof['meshes'])!=3:raise ValueError('All12NEWfloorCPUcontrolsrequired')
 sources=list(dict.fromkeys(list(prior['source_sha256'])+list(cp['source_sha256'])+[SOURCE,TEST]));records=[base,cpu/'CPU-protocol.json',cpu/'CPU-result.json',ROOT/'runs/operator-dev/p2-contact-rigid-elastic-diagnosis-v01/protocol.json',ROOT/'runs/operator-dev/p2-contact-rigid-elastic-diagnosis-v01/result.json',ROOT/'runs/operator-dev/p2-contact-spatial-pairs-v01/spatial-comparison.json']
 p={**prior,'schema':'NEWphysical-softerfloor1e6-three-mesh-screen-v01','status':'FROZEN_BEFORE_NEW_BOUNDARY_EXPERIMENT','source_sha256':{f:sha(ROOT/f) for f in sources},'evidence_sha256':{**prior['evidence_sha256'],**{str(f.relative_to(ROOT)):sha(f) for f in records}},'jobs':JOBS,'configs':[CONFIG],'global_wall_cap_s':1900,'per_job_caps_s':[600,900,1800],'physics':{**prior['physics'],'kappa_N_m3':1e6,'cells':'PER_DECLARED_JOB','surface':'root128forceE finitecompliantreferencearea atNEWkappa1e6'},'corrections':'NEW physical kappa=1e6 floor and explicitly declared three-mesh single-accuracy screen; no numerical repair of the old boundary experiment.', 'boundary_change':'Physicalfloorcompliance1000xsofterthanold1e9; NOTnumericalrepair/relabeloldFAIL. SameE/nu/rho/g/gap/initial/end/clock/fullM/Keast24/J141/noDamping/noFriction.','pretrajectory_controls':'Everyfreshjob4THISgeometry positiveCPUstrictadaptiveEg+CPU64/128+GPU128/fullM checks atEXPLICITNEWkappa1e6 beforeDOP.','GPU_uuid':None,'CPU_affinity_cores':None,'phase_id':None,'metadata_directory':str(OP.relative_to(ROOT)),'output':str(OUT.relative_to(ROOT)),'phase_cap_s':None,'case_cap_s':None,'execution':'Oneownedglobal1900s controller launches3disjointGPUs/CPU2workersinparallel. Perjob600/900/1800. Retainallcompleted/failed/timeout outcomes; no retries. AllownedPGTERM3s/KILL/reap onglobalinterrupt/cap.','screen_qualification':'Single tightaccuracy permesh gives only prospective fixed-operator spatialscreen. Futuretimepairs/surfaceorderrequired evenifPASS; CONTACTDATASETCLOSED.','forecast':{'old_boundary_tight_measured_s':[100.8,149.0,400.0],'new_boundary_cost':'Unknown; sameoperatorsize, lessboundary stiffness mayreduceadaptivecount. Caps NOTspeedclaim.'},'nonclaims':['NEWboundaryphysicalexperiment only; old1e9spatialFAIL and768timeFAIL preserved','No timepair qualification fromone tightaccuracy','Fixedroot128spatialscreen, not continuum/surfaceorder/cohort/hardflooradmission','No rigidprojection/damping/force-or-material-retuning/noiserelaxation','Sameoriginal20um/100um-s q/vconditionalgates andcontraction; alloutcomesretained']}
 for stale in ['control_capture','superseded_initial_metadata_protocol','imported96_capture_provenance_only']:p.pop(stale,None)
 if (OP/'protocol.json').exists():raise FileExistsError('Preserve newphysicalprotocol')
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changednewphysicalprotocol')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('scipy_source_sha256',SCIPY_ROOT),('installed_warp_source_sha256',ENV_SITE),('CPU_evidence_sha256',ROOT),('evidence_sha256',ROOT)]:
  if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changedpin:'+group)
 if p['jobs']!=JOBS or p['configs']!=[CONFIG] or p['global_wall_cap_s']!=1900 or p['physics']['kappa_N_m3']!=1e6 or p['physics']['E_Pa']!=300000. or p['physics']['nu']!=.35 or p['physics']['rho_kg_m3']!=625. or p['maximum_step_s']!=1/30000 or p['resource_limits']!=LIMITS:raise ValueError('Exactnewboundaryscreenonly')
 return p
def run(digest,job_id):
 protocol=OP/'protocol.json';p=validate(digest);job=next((j for j in p['jobs'] if j['id']==job_id),None)
 if job is None:raise ValueError('Onlydeclaredscreenjob')
 p={**p,'GPU_uuid':job['GPU_uuid'],'CPU_affinity_cores':job['CPU_affinity_cores'],'physics':{**p['physics'],'cells':job['cells']}};label=CONFIG['name'];meta=OP/job_id;out=OUT/job_id;meta.mkdir(parents=True,exist_ok=True);GPU=job['GPU_uuid']
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('WrongGPU')
 matches=[c for c in p['configs'] if c['name']==label]
 if len(matches)!=1:raise ValueError('Exactdeclaredcaseonly')
 config=matches[0];dest=out/label;dest.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,passed=False,pid=os.getpid(),protocol_sha256=digest,config=config)
 try:
  import torch,scipy,sys
  from scipy.linalg import cho_factor,cho_solve
  sys.path.append(str(ENV_SITE));import warp as wp
  from experiments.campaign_p2_warp_reference_order128 import WarpReferenceForce128
  from experiments.run_p2_contact_validation import build
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.campaign_p2_torch_contact import quadratic_face_minimum
  if wp.__version__!='1.18.0':raise ValueError('PinnedWarp1.18required')
  torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());device='cuda:0';torch.cuda.set_per_process_memory_fraction(p['resource_limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.ones(1,device=device).sum().item();wp.config.kernel_cache_dir=str(meta/'warp-cache')
  def guard(tag,reserve_write_bytes=0,reserve_memory_bytes=0):
   r=dict(tag=label+':'+tag,unix_time_s=time.time(),**collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[meta,out],torch=torch),reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes);validate_budget(r,reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes,limits=p['resource_limits']);write(meta/(label+'-resource-latest.json'),r)
   with (meta/(label+'-resource-history.jsonl')).open('a') as f:f.write(json.dumps(r)+'\n')
  guard('startup_ownGPU');m=build(p['physics']['cells'],300000.,9.81);w=WarpReferenceForce128(m,kappa_A=1e6,device=device);n=len(m.rest);size=3*n;Mt=scalar_mass(m).toarray();M=torch.tensor(Mt,dtype=torch.float64,device=device);L=torch.linalg.cholesky(M);force=torch.tensor(m.load,dtype=torch.float64,device=device);torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device=device)*torch.tensor(GRAVITY,dtype=torch.float64,device=device),force,atol=1e-13,rtol=1e-13)
  # ALL controls are rebuilt from THIS public rest geometry, never96 response fields.
  from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
  from experiments.campaign_soft_floor_controls import assess_geometry_controls,manufactured_states
  cpu=WarpReferenceForce128(m,kappa_A=1e6,device='cpu');cpu64=WarpReferenceForce(m,kappa_A=1e6,device='cpu');Lc=cho_factor(Mt)
  def save_controls(rows):write(meta/(label+'-GPU-controls.json'),dict(completed=False,controls=rows,protocol_sha256=digest))
  controls=assess_geometry_controls(m,cpu,cpu64,w,kappa_A=1e6,after_state=save_controls)
  for row,(_,u) in zip(controls['controls'],manufactured_states(m.rest)):
   a=w.evaluate(u);net=force-torch.tensor(a['gradient'],dtype=torch.float64,device=device);acc=torch.cholesky_solve(net,L);R=float(torch.linalg.vector_norm(M@acc-net));row['GPU_original_full_mass_R_N']=R;row['GPU_CPU_acceleration_L2_difference_m_s2']=float(np.linalg.norm(acc.cpu().numpy()-cho_solve(Lc,m.load-cpu.evaluate(u)['gradient'])));row['passed']=bool(row['passed'] and np.isfinite(R) and R<=1e-8)
  controls['passed']=all(row['passed'] for row in controls['controls']);write(meta/(label+'-GPU-controls.json'),dict(**controls,protocol_sha256=digest))
  if not controls['passed']:raise ValueError('NEWgeometry independentCPU/GPU controls failed BEFOREtrajectory')
  del cpu,cpu64;guard('pretrajectory_parity_passed');s.update(cells=p['physics']['cells'],versions=dict(torch=torch.__version__,scipy=scipy.__version__,warp=wp.__version__,numpy=np.__version__),pretrajectory_controls_passed=True);write(dest/'summary.json',s)
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

def run_global(digest):
 p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(j['GPU_uuid'] in apps for j in JOBS):raise RuntimeError('All3declaredGPUsmustbeidle')
 if not set(range(6)).issubset(os.sched_getaffinity(0)):raise ValueError('All6declaredCPUcoresrequired')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();state=dict(controller_pid=os.getpid(),completed=False,terminal=False,protocol_sha256=digest,records=[],retries=0,contact_dataset_admitted=False);write(OP/'execution.json',state);active={};logs=[];previous={}
 def stopped(signum,*args):raise InterruptedError('Ownedglobalcontroller stop:'+str(signum))
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:previous[sig]=signal.signal(sig,stopped)
 signal.alarm(1900)
 try:
  if sha(Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
  for job in JOBS:
   meta=OP/job['id'];meta.mkdir(parents=True,exist_ok=True);log=(meta/'worker.log').open('x');logs.append(log)
   command=[p['taskset']['path'],'--cpu-list',','.join(map(str,job['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_soft_floor_screen','run','--sha',digest,'--job',job['id']]
   blocked=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
   try:
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':job['GPU_uuid'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});row=dict(job=job,pid=child.pid,terminal=False,launch_offset_s=time.monotonic()-start);state['records'].append(row);active[job['id']]=(child,row,time.monotonic())
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
   write(OP/'execution.json',state)
  while active:
   for name,(child,row,launched) in list(active.items()):
    timeout=time.monotonic()-launched>=row['job']['wall_cap_s']
    if timeout and child.poll() is None:row['timeout']=True;terminate_owned(child)
    if child.poll() is None:continue
    row.update(terminal=True,exit_code=child.wait());file=OUT/name/CONFIG['name']/'summary.json';r=json.loads(file.read_text()) if file.exists() else {};row.update(completed=r.get('completed',False),passed=r.get('passed',False))
    if file.exists():
     data=file.read_bytes();h=sha(file);cache=ROOT/'runs/operator-dev/completed-metadata-cache-v01'/f'{h}.json'
     cache.parent.mkdir(parents=True,exist_ok=True)
     if cache.exists():assert cache.read_bytes()==data
     else:
      with cache.open('xb') as f:f.write(data)
     row['summary']=dict(path=str(file.relative_to(ROOT)),sha256=h,external_cache=str(cache.relative_to(ROOT)))
    del active[name];write(OP/'execution.json',state)
   if active:time.sleep(.2)
  state.update(completed=all(r['exit_code']==0 and r['completed'] for r in state['records']),all_mechanical_checks_passed=all(r['passed'] for r in state['records']))
  if state['completed']:
   fields=[]
   for job in JOBS:
    summary=json.loads((OUT/job['id']/CONFIG['name']/'summary.json').read_text());cap=summary['capture']
    if sha(ROOT/cap['path'])!=cap['sha256']:raise ValueError('Changednewboundarycapture')
    with np.load(ROOT/cap['path'],allow_pickle=False) as z:fields.append({k:z[k].copy() for k in ['times_s','material_probes_m','query_positions_m','query_velocities_m_s']})
   metrics=[spatial_pair(fields[0],fields[1],2.),spatial_pair(fields[1],fields[2],3.)];state.update(spatial_comparisons=metrics,**combine(*metrics))
 except BaseException as error:state['error']=repr(error)
 finally:
  signal.alarm(0)
  for child,row,_ in active.values():row.update(terminal=True,controller_cleanup=True,exit_code=terminate_owned(child))
  for log in logs:log.close()
  for sig,old in previous.items():signal.signal(sig,old)
  state.update(terminal=True,elapsed_s=time.monotonic()-start,contact_dataset_admitted=False,future_time_and_surface_order_required=True);write(OP/'execution.json',state);write(OUT/'summary.json',state)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','run','launch']);p.add_argument('--sha');p.add_argument('--job');a=p.parse_args()
 if a.action=='prepare':print(json.dumps(prepare()))
 elif a.action=='validate':print(json.dumps(dict(validated=bool(validate(a.sha)),GPU_calls=False)))
 else:
  r=run(a.sha,a.job) if a.action=='run' else run_global(a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','elapsed_s','error','finest_conditional_q_gate_met','finest_conditional_v_gate_met']}));raise SystemExit(0 if r.get('completed') else 1)
