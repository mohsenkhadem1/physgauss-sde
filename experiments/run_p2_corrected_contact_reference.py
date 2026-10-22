"""Prospective corrected96P2 DOP853 pair: exactbulk24 + root64surface.

Two fresh isolated workers, unchanged material/fullM/force/initial state/clock.
Old Q3/Duffy results are retained; this explicitly corrects TWO integrations.
"""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_contact_dop853 import ROOT,GPU,sha,write,SCIPY_ROOT,SCIPY_FILES
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
from experiments.continue_p2_contact_dop853_v03 import manual_dop853
from experiments.run_p2_warp_surface_q16 import SOURCES as OLD_SOURCES,estimate_prefix_bytes
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
from experiments.benchmark_p2_warp_rhs import ENV_SITE
OP=ROOT/'runs/operator-dev/p2-corrected-contact-reference-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-reference-v01';CPU=ROOT/'runs/operator-dev/p2-warp-reference-force-v01';CONTROL_CAPTURE=ROOT/'runs/gaussian-campaign-v01/p2-warp-surface-q16-v01/surface_Q16.npz'
CONFIGS=[dict(name='accuracy_1e9',rtol=1e-9,displacement_atol_m=1e-12,velocity_atol_m_s=1e-10,impulse_atol_Ns=1e-13),dict(name='accuracy_1e10',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)]

def pair_metrics(q1,v1,E1,q2,v2,E2,energytol,probe1,probe2):
 r=dict(maximum_allnode_position_difference_m=float(np.linalg.norm(q1-q2,axis=2).max()),maximum_fixedquery_position_difference_m=float(np.linalg.norm(probe1-probe2,axis=2).max()),maximum_allnode_velocity_difference_m_s=float(np.linalg.norm(v1-v2,axis=2).max()),maximum_output_energy_difference_J=float(abs(E1-E2).max()))
 r['passed']=r['maximum_allnode_position_difference_m']<=20e-6 and r['maximum_fixedquery_position_difference_m']<=20e-6 and r['maximum_allnode_velocity_difference_m_s']<=1e-4 and r['maximum_output_energy_difference_J']<=energytol
 return r

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserveprotocol')
 cp=CPU/'CPU-protocol-v02.json';cr=CPU/'CPU-fixedstate-v02-result.json';controls=json.loads(cr.read_text())
 if not controls.get('completed') or not controls.get('passed'):raise ValueError('CombinedCPUproofrequired')
 cpins=json.loads(cp.read_text())['source_sha256'];sources=list(dict.fromkeys(OLD_SOURCES+list(cpins)+['experiments/campaign_p2_warp_reference_force.py','experiments/run_p2_corrected_contact_reference.py','tests/test_run_p2_corrected_contact_reference.py']))
 prior=json.loads((ROOT/'runs/operator-dev/p2-warp-surface-q16-v01/protocol.json').read_text())
 p=dict(schema='correctedbulk24-root64-fixed96-DOPtimepair-v1',status='FROZEN_PENDING_ROOT_REVIEW_BEFORE_NEW_TRAJECTORIES',source_sha256={f:sha(ROOT/f) for f in sources},scipy_source_sha256={f:sha(SCIPY_ROOT/f) for f in SCIPY_FILES+['integrate/_ivp/base.py','integrate/__init__.py']},installed_warp_source_sha256=prior['installed_warp_source_sha256'],CPU_evidence_sha256={str(f.relative_to(ROOT)):sha(f) for f in [cp,cr]},control_capture=dict(path=str(CONTROL_CAPTURE.relative_to(ROOT)),sha256=sha(CONTROL_CAPTURE)),GPU_uuid=GPU,CPU_affinity_cores=[30,31],CPU_threads=2,resource_limits=LIMITS,case_cap_s=1800,phase_cap_s=3600,taskset=dict(path='/usr/bin/taskset',sha256=sha(Path('/usr/bin/taskset'))),configs=CONFIGS,physics=dict(cells=[4,2,2],E_Pa=300000.,nu=.35,rho_kg_m3=625.,gravity_m_s2=GRAVITY.tolist(),initial_bottom_gap_m=.05,kappa_N_m3=1e9,floor_z_m=0.,bulk='Keast24degree6exactconstantstableNH',surface='root64forceE finitecompliantreferencearea',guards=141,full_consistent_mass=True,physical_damping=0.,friction=0.,all_free=True),corrections='EXPLICIT TWO: oldQ3bulk->exactKeast24 AND oldDuffycontact->root64. No oldDuffy kernel launch/subtraction; no material/mass/load retuning.',maximum_step_s=1/30000,interval_s=[T0,END],outputs_s=(T0+np.arange(9)/120).tolist(),initializer='EXACTanalyticuniformfreefall u=.5*g*T0²,v=g*T0 atT0=1/12; noBEstarteffect; positiveinitialgap15.9375mm',gates=dict(energy_J=max(1e-9,1e-6*.96*9.81*.05),momentum_Ns=1e-8,consistent_mass_R_N=1e-8,minimum_J=.1,allnode_and_fixedquery_pair_m=20e-6,allnode_velocity_pair_m_s=1e-4),pretrajectory_controls='Eachfreshworker: same4hashboundsavedfields GPUvsCPUcombinedE1e-12/g1e-10/J2e-14/fullMR1e-8; CPUcombinedalreadyindependentlyNumPy/adaptivevalidated',forecast=dict(root64_inclusive_GPU_ms=.5871839355677366,old_fullRHS_inclusive_ms=.987,conservative_new_fullRHS_estimate_ms=1.8,old_tight_Q16_RHS=552551,estimated_RHS_cost_s=994.5918,compile_energy_check_checkpoint_allowance_s=500,case_cap_s=1800,phase_cap_s=3600,estimate_not_bound=True),checkpoint='Every512accepted/final:atomicUNCOMPRESSEDfullacceptedprefix+times/E/P,allcommonoutputs/E/P/penetration,currenty/currentf/nexth; hardkillcanlose<=511uncommittedtail. Noresumeclaim.',execution='TwoFRESHworkerssequentialGPU1;continueifcompletedconservationnegative,stopnewifworkerincomplete/timeout; no retries/redraw; ownedPGTERM/KILL/reap; phasealarm3600s',nonclaims=['AuxPsharesRKquadrature,consistencynotindependenttimeproof','Fixed96mesh/finitekappa referenceonly; nohardfloor/friction/continuum/contactdatasetadmission','TimepairpassdoesNOTestablishroot64surfaceorderstabilityatnewstates','WholepositiveJguardpointsarenotglobalintervalcertificate','OldQ3/Duffyscope/failures preserved; newcomparisonnotcontact-only'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def validate(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 p=json.loads((OP/'protocol.json').read_text())
 for group,base in [('source_sha256',ROOT),('scipy_source_sha256',SCIPY_ROOT),('installed_warp_source_sha256',ENV_SITE),('CPU_evidence_sha256',ROOT)]:
  if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changedpin:'+group)
 if sha(ROOT/p['control_capture']['path'])!=p['control_capture']['sha256']:raise ValueError('Changedcontrolcapture')
 return p

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
  from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
  from experiments.run_p2_contact_validation import build
  from experiments.campaign_p2_consistent_transient import scalar_mass
  from experiments.campaign_p2_torch_contact import quadratic_face_minimum
  if wp.__version__!='1.18.0':raise ValueError('PinnedWarp1.18required')
  torch.set_num_threads(2);torch.set_num_interop_threads(2);validate_cpu(p['CPU_affinity_cores'],os.sched_getaffinity(0),os.environ,torch.get_num_threads(),torch.get_num_interop_threads());device='cuda:0';torch.cuda.set_per_process_memory_fraction(p['resource_limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.ones(1,device=device).sum().item();wp.config.kernel_cache_dir=str(OP/'warp-cache')
  def guard(tag,reserve_write_bytes=0,reserve_memory_bytes=0):
   r=dict(tag=label+':'+tag,unix_time_s=time.time(),**collect_resources(gpu_uuid=GPU,pid=os.getpid(),roots=[OP,OUT],torch=torch),reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes);validate_budget(r,reserve_write_bytes=reserve_write_bytes,reserve_memory_bytes=reserve_memory_bytes,limits=p['resource_limits']);write(OP/(label+'-resource-latest.json'),r)
   with (OP/(label+'-resource-history.jsonl')).open('a') as f:f.write(json.dumps(r)+'\n')
  guard('startup_ownGPU');m=build([4,2,2],300000.,9.81);w=WarpReferenceForce(m,device=device);n=len(m.rest);size=3*n;Mt=scalar_mass(m).toarray();M=torch.tensor(Mt,dtype=torch.float64,device=device);L=torch.linalg.cholesky(M);force=torch.tensor(m.load,dtype=torch.float64,device=device);torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device=device)*torch.tensor(GRAVITY,dtype=torch.float64,device=device),force,atol=1e-13,rtol=1e-13)
  # Fresh pretrajectory GPU implementation check against CPUcombined backend;
  # the latter is already independently checked against NumPy+adaptiveplane.
  cpu=WarpReferenceForce(m,device='cpu');Lc=cho_factor(Mt)
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
  atol=np.r_[np.full(size,config['displacement_atol_m']),np.full(size,config['velocity_atol_m_s']),np.full(3,config['impulse_atol_Ns'])];answer=manual_dop853(rhs,y0,[T0,END],p['outputs_s'],config,atol,after);y=answer['output_states'];q=y[:size].T.reshape(-1,n,3)+m.rest;v=y[size:2*size].T.reshape(-1,n,3);pred=np.array([m.query(x,probe) for x in q]);Eerror=max(float(abs(np.asarray(Et)-initialE).max()),float(abs(np.asarray(outE)-initialE).max()));Perror=max(Pt+outP);passed=Eerror<=energytol and Perror<=1e-8 and stats['maximum_masssolve_R_N']<=1e-8;guard('final_capture',answer['accepted_states'].nbytes+estimate_prefix_bytes(len(answer['accepted_times']),9,len(y0)),answer['accepted_states'].nbytes);f=dest/'trajectory.npz';np.savez(f,times_s=answer['output_times'],positions_m=q,velocities_m_s=v,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,surface_elements=faces,fixed_node_ids=np.array([],int),floor_z_m=0.,query_positions_m=pred,material_probes_m=probe,accepted_times_s=answer['accepted_times'],accepted_states=answer['accepted_states'],accepted_mechanical_energies_J=np.asarray(Et),accepted_momentum_balance_errors_Ns=np.asarray(Pt),output_mechanical_energies_J=np.asarray(outE),output_momentum_balance_errors_Ns=np.asarray(outP),output_analytic_face_penetrations_m=np.asarray(outpen));guard('after_final_capture');s.update(completed=True,passed=passed,integration_elapsed_s=time.monotonic()-clock,accepted_steps=answer['accepted_steps'],nfev=answer['nfev'],**stats,maximum_accepted_energy_error_J=float(abs(np.asarray(Et)-initialE).max()),maximum_output_energy_error_J=float(abs(np.asarray(outE)-initialE).max()),maximum_accepted_momentum_error_Ns=max(Pt),maximum_output_momentum_error_Ns=max(outP),minimum_accepted_J=Jmin,maximum_analytic_face_penetration_m=penmax,initial_energy_J=initialE,energy_tolerance_J=energytol,capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
 except Exception as error:s.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(dest/'summary.json',s)
 return s

def launch(digest):
 begin=time.monotonic()
 def phasecap(*a):raise TimeoutError('Frozen3600sphasecap')
 signal.signal(signal.SIGALRM,phasecap);signal.alarm(3600);p=validate(digest)
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 OUT.mkdir(parents=True,exist_ok=False);e=dict(controller_pid=os.getpid(),terminal=False,completed=False,time_reference_qualified=False,protocol_sha256=digest,workers=[]);write(OP/'execution.json',e);rows=[]
 try:
  if sha(Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
  for config in p['configs']:
   remaining=p['phase_cap_s']-(time.monotonic()-begin)
   if remaining<=0:raise TimeoutError('Noremainingphasebudget')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
   if any(GPU in line for line in apps.splitlines()):raise ValueError('GPU1notidle')
   label=config['name'];record=dict(case=label,terminal=False);e['workers'].append(record)
   def started(pid):record['worker_pid']=pid;write(OP/'execution.json',e)
   with (OP/(label+'-worker.log')).open('x') as log:record.update(run_owned_command([p['taskset']['path'],'--cpu-list','30,31',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_corrected_contact_reference','run','--sha',digest,'--case',label],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=min(p['case_cap_s'],remaining),on_child=started))
   record['terminal']=True;file=OUT/label/'summary.json';r=json.loads(file.read_text()) if file.exists() else {};record['completed']=record.get('exit_code')==0 and r.get('completed') is True;rows.append(r);write(OP/'execution.json',e)
   if not record['completed']:raise RuntimeError('Incompleteworker;preserveprefixandstopnewlaunches')
  arrays=[]
  for row in rows:
   if sha(ROOT/row['capture']['path'])!=row['capture']['sha256']:raise ValueError('Changedcompletedcapture')
   with np.load(ROOT/row['capture']['path'],allow_pickle=False) as z:arrays.append([z[k].copy() for k in ['positions_m','velocities_m_s','output_mechanical_energies_J','query_positions_m']])
  a,b=arrays;pair=pair_metrics(a[0],a[1],a[2],b[0],b[1],b[2],p['gates']['energy_J'],a[3],b[3]);e.update(completed=True,time_reference_qualified=all(r['passed'] for r in rows) and pair['passed'],accuracy_pair=pair)
 except BaseException as error:e['error']=repr(error)
 finally:signal.alarm(0);e.update(terminal=True,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e);write(OUT/'summary.json',dict(**e,cases=rows,nonclaims=p['nonclaims']))
 return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');p.add_argument('--case');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha,a.case) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','passed','time_reference_qualified','elapsed_s','error']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
