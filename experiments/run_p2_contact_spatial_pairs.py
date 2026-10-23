"""Prospective new-mesh root128 contact time pairs; no old runtime edits.

One independent phase per mesh. Every fresh trajectory rebuilds four controls
from its own public rest geometry. Cross-mesh comparison uses physical material
queries only; a fixed surface operator is not a continuum contact certificate.
"""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
from experiments.run_p2_contact_dop853 import ROOT,sha,write,SCIPY_ROOT
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
from experiments.continue_p2_contact_dop853_v03 import manual_dop853
from experiments.run_p2_warp_surface_q16 import estimate_prefix_bytes
from experiments.campaign_owned_gpu_budget_v02 import LIMITS,collect_resources,validate_budget,validate_cpu,run_owned_command
from experiments.benchmark_p2_warp_rhs import ENV_SITE
from experiments.run_p2_contact_root128_time_control import time_pair
META=ROOT/'runs/operator-dev/p2-contact-spatial-pairs-v01'
SOURCE='experiments/run_p2_contact_spatial_pairs.py'
TEST='tests/test_run_p2_contact_spatial_pairs.py'
PHASES={'324tets':dict(cells=[6,3,3],GPU_uuid='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb',CPU_affinity_cores=[2,3],case_caps_s=[900,1800]),'768tets':dict(cells=[8,4,4],GPU_uuid='GPU-571378aa-c44e-ffb8-b346-1e409f9821f9',CPU_affinity_cores=[4,5],case_caps_s=[1800,3600])}
CONFIGS=[dict(name='accuracy_1e9',rtol=1e-9,displacement_atol_m=1e-12,velocity_atol_m_s=1e-10,impulse_atol_Ns=1e-13),dict(name='accuracy_1e10',rtol=1e-10,displacement_atol_m=1e-13,velocity_atol_m_s=1e-11,impulse_atol_Ns=1e-13)]

def prepare():
 base=ROOT/'runs/operator-dev/p2-contact-root128-time-control-v01';prior=json.loads((base/'protocol.json').read_text());done=json.loads((base/'result-summary.json').read_text());cpu=ROOT/'runs/operator-dev/p2-contact-geometry-controls-v01';proof=json.loads((cpu/'CPU-result.json').read_text())
 if not done['completed'] or not done['time_reference_qualified'] or not proof['completed'] or not proof['passed']:raise ValueError('Completedcoarse128timeANDnewgeometryCPUcontrols required')
 sources=list(prior['source_sha256'])+[SOURCE,TEST]+list(json.loads((cpu/'CPU-protocol.json').read_text())['source_sha256']);records=[base/'protocol.json',base/'result-summary.json',base/'terminal-verification.json',cpu/'CPU-protocol.json',cpu/'CPU-result.json',cpu/'preserved-full-H-failure.json',ROOT/'runs/operator-dev/p2-contact-spatial-design-v01/spatial-design.json'];results=[]
 for name,phase in PHASES.items():
  op=META/name;out=ROOT/'runs/gaussian-campaign-v01/p2-contact-spatial-pairs-v01'/name
  if (op/'protocol-v02.json').exists():raise FileExistsError('Preserveprotocol')
  configs=[dict(c,wall_cap_s=cap) for c,cap in zip(CONFIGS,phase['case_caps_s'])]
  p={**prior,'schema':'root128-newmesh-contact-time-pair-v01','corrections':'ONLY declared mesh geometry and two numerical accuracy settings; unchanged root128/Keast24/fullM/stableNH/bodygravity/plane law/timeclock. No96response-state transplantation.','imported96_capture_provenance_only':True,'superseded_initial_metadata_protocol':dict(path=str((op/'protocol.json').relative_to(ROOT)),sha256=sha(op/'protocol.json')),'phase_id':name,'status':'FROZEN_PENDING_ROOT_REVIEW_NO_GPU_LAUNCH','metadata_directory':str(op.relative_to(ROOT)),'output':str(out.relative_to(ROOT)),'source_sha256':{f:sha(ROOT/f) for f in dict.fromkeys(sources)},'evidence_sha256':{**prior['evidence_sha256'],**{str(f.relative_to(ROOT)):sha(f) for f in records}},'GPU_uuid':phase['GPU_uuid'],'CPU_affinity_cores':phase['CPU_affinity_cores'],'configs':configs,'phase_cap_s':sum(phase['case_caps_s']),'case_cap_s':None,'physics':{**prior['physics'],'cells':phase['cells']},'pretrajectory_controls':'Everyfreshworker THISgeometry analyticfreefall/smoothsmallnoncontact/uniformslightpenetration/smoothpartialcontact: independentNumPy+strictadaptiveEgONLY vsCPU128,CPU64vs128,GPU128vsCPU128E/g/J andFULLconsistentM originalR beforeintegration. No96savedresponsecontrols. No H claim/status2waiver.','forecast':dict(measured96_loose_phase_s=done['elapsed_s'],new_mesh_scaling='324tet3.375xE/2.25xfaces;768tet8xE/4xfaces, scalarM225->637->1377, adaptivecountunknown',per_case_caps_s=phase['case_caps_s'],caps_are_conservative_not_speed_claim=True),'execution':'TwoFRESHisolatedworkers permesh, originalDOP853/clock/law; no retry. Stopnewlaunches onincomplete/control/resourcefailure, retain completednegativeenergy/timeoutcome. OwnedPGTERM/KILL/reap, hardphase sumcaps.','fixed_spatial_queries':'Exact81=(9x3x3)commonrestpoints and9common120Hzclocks; evaluateP2positionsANDphysicalvelocity. NevercompareallnodesACROSSmeshes.','spatial_gate':dict(maximum_query_q_m=20e-6,maximum_query_v_m_s=1e-4,contraction_required=True,conditional_assumed_order_p_at_least=1,indicator_factor_96_to324=2.,indicator_factor_324_to768=3.),'surface_scope':'ROOT128FIXEDOPERATORmeshcomparison only; manufacturedCPU64/128/adaptivechecks are NOTtrajectorysurface-orderproof. No automatic continuum/contactdatasetqualification.','nonclaims':['Knownsingle finitecompliant frictionless drop only; not rigidfloor/friction/dataset','Timeandspatialq/vcriteria are SEPARATE; no combinederrorbound','Surface128 fixed-operator spatialcomparison until separatelyreviewed new-mesh surfaceorder controls','No force/material/mass/energy/gradient/J tolerance changes; no H/control claim','OriginalfullE/g/Hstatus2negative remains explicit; NEW reference onlyEg atoriginaltolerances']}
  write(op/'protocol-v02.json',p);results.append(dict(path=str((op/'protocol-v02.json').relative_to(ROOT)),sha256=sha(op/'protocol-v02.json')))
 return results

def validate(protocol,digest):
 path=ROOT/protocol
 if path.is_absolute() and not path.resolve().is_relative_to(ROOT):raise ValueError('Repositoryprotocol only')
 if sha(path)!=digest:raise ValueError('Changedprotocol')
 p=json.loads(path.read_text());phase=PHASES.get(p.get('phase_id'))
 if p.get('schema')!='root128-newmesh-contact-time-pair-v01' or phase is None:raise ValueError('Exactnewmeshphase')
 for group,base in [('source_sha256',ROOT),('scipy_source_sha256',SCIPY_ROOT),('installed_warp_source_sha256',ENV_SITE),('CPU_evidence_sha256',ROOT),('evidence_sha256',ROOT)]:
  if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changedpin:'+group)
 if p['configs']!=[dict(c,wall_cap_s=cap) for c,cap in zip(CONFIGS,phase['case_caps_s'])] or p['phase_cap_s']!=sum(phase['case_caps_s']) or p['GPU_uuid']!=phase['GPU_uuid'] or p['CPU_affinity_cores']!=phase['CPU_affinity_cores'] or p['physics']['cells']!=phase['cells'] or p['maximum_step_s']!=1/30000 or p['resource_limits']!=LIMITS:raise ValueError('Exactscoperesourcesonly')
 if p['output']!=f"runs/gaussian-campaign-v01/p2-contact-spatial-pairs-v01/{p['phase_id']}" or p['metadata_directory']!=f"runs/operator-dev/p2-contact-spatial-pairs-v01/{p['phase_id']}":raise ValueError('Freshuniquephasepaths')
 if sha(ROOT/p['control_capture']['path'])!=p['control_capture']['sha256']:raise ValueError('Changedimported96tightcontrol')
 return p

def spatial_pair(a,b,factor):
 for k in ['times_s','material_probes_m']:np.testing.assert_array_equal(a[k],b[k])
 q=float(np.linalg.norm(a['query_positions_m']-b['query_positions_m'],axis=-1).max());v=float(np.linalg.norm(a['query_velocities_m_s']-b['query_velocities_m_s'],axis=-1).max())
 return dict(maximum_common81_q_difference_m=q,maximum_common81_v_difference_m_s=v,conditional_p1_factor=factor,conditional_q_indicator_m=factor*q,conditional_v_indicator_m_s=factor*v,q_passed=factor*q<=20e-6,v_passed=factor*v<=1e-4,assumed_p_at_least1_not_rigorous_bound=True,fixed_operator_ONLY=True)
def run(protocol,digest,label):
 p=validate(protocol,digest);OP=ROOT/p['metadata_directory'];OUT=ROOT/p['output'];GPU=p['GPU_uuid']
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
  guard('startup_ownGPU');m=build(p['physics']['cells'],300000.,9.81);w=WarpReferenceForce128(m,device=device);n=len(m.rest);size=3*n;Mt=scalar_mass(m).toarray();M=torch.tensor(Mt,dtype=torch.float64,device=device);L=torch.linalg.cholesky(M);force=torch.tensor(m.load,dtype=torch.float64,device=device);torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device=device)*torch.tensor(GRAVITY,dtype=torch.float64,device=device),force,atol=1e-13,rtol=1e-13)
  # ALL controls are rebuilt from THIS public rest geometry, never96 response fields.
  from experiments.campaign_p2_warp_reference_force import WarpReferenceForce
  from experiments.campaign_contact_geometry_controls import assess_geometry_controls,manufactured_states
  cpu=WarpReferenceForce128(m,device='cpu');cpu64=WarpReferenceForce(m,device='cpu');Lc=cho_factor(Mt)
  def save_controls(rows):write(OP/(label+'-GPU-controls.json'),dict(completed=False,controls=rows,protocol_sha256=digest))
  controls=assess_geometry_controls(m,cpu,cpu64,w,after_state=save_controls)
  for row,(_,u) in zip(controls['controls'],manufactured_states(m.rest)):
   a=w.evaluate(u);net=force-torch.tensor(a['gradient'],dtype=torch.float64,device=device);acc=torch.cholesky_solve(net,L);R=float(torch.linalg.vector_norm(M@acc-net));row['GPU_original_full_mass_R_N']=R;row['GPU_CPU_acceleration_L2_difference_m_s2']=float(np.linalg.norm(acc.cpu().numpy()-cho_solve(Lc,m.load-cpu.evaluate(u)['gradient'])));row['passed']=bool(row['passed'] and np.isfinite(R) and R<=1e-8)
  controls['passed']=all(row['passed'] for row in controls['controls']);write(OP/(label+'-GPU-controls.json'),dict(**controls,protocol_sha256=digest))
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

def launch(protocol,digest):
 p=validate(protocol,digest);OP=ROOT/p['metadata_directory'];OUT=ROOT/p['output'];GPU=p['GPU_uuid'];begin=time.monotonic()
 def cap(*a):raise TimeoutError('Frozenphasecap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(p['phase_cap_s'])
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 OUT.mkdir(parents=True,exist_ok=False);e=dict(controller_pid=os.getpid(),terminal=False,completed=False,time_reference_qualified=False,protocol_sha256=digest,workers=[]);write(OP/'execution.json',e);rows=[]
 try:
  if sha(Path(p['taskset']['path']))!=p['taskset']['sha256']:raise ValueError('Changedtaskset')
  for config in p['configs']:
   remaining=p['phase_cap_s']-(time.monotonic()-begin)
   if remaining<=0:raise TimeoutError('No remaining phasebudget')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
   if any(GPU in line for line in apps.splitlines()):raise ValueError('DeclaredGPU notidle')
   label=config['name'];record=dict(case=label,terminal=False);e['workers'].append(record)
   def started(pid):record['worker_pid']=pid;write(OP/'execution.json',e)
   with (OP/(label+'-worker.log')).open('x') as log:record.update(run_owned_command([p['taskset']['path'],'--cpu-list',','.join(map(str,p['CPU_affinity_cores'])),str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_contact_spatial_pairs','run','--protocol',protocol,'--sha',digest,'--case',label],environment={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},log=log,timeout=min(config['wall_cap_s'],remaining),on_child=started))
   record['terminal']=True;f=OUT/label/'summary.json';r=json.loads(f.read_text()) if f.exists() else {};record['completed']=record.get('exit_code')==0 and r.get('completed') is True;rows.append(r);write(OP/'execution.json',e)
   if not record['completed']:raise RuntimeError('Incompleteworker; preserveall andstopnewlaunches')
  arrays=[]
  for row in rows:
   if sha(ROOT/row['capture']['path'])!=row['capture']['sha256']:raise ValueError('Changedcompletedcapture')
   with np.load(ROOT/row['capture']['path'],allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in ['times_s','positions_m','velocities_m_s','output_mechanical_energies_J','query_positions_m','query_velocities_m_s','accepted_times_s']})
  pair=time_pair(*arrays,p['gates']['energy_J']);pair['maximum_common81_query_velocity_difference_m_s']=float(np.linalg.norm(arrays[0]['query_velocities_m_s']-arrays[1]['query_velocities_m_s'],axis=-1).max());e.update(completed=True,time_reference_qualified=bool(all(r['passed'] for r in rows) and pair['time_reference_qualified']),accuracy_pair=pair)
 except BaseException as error:e['error']=repr(error)
 finally:signal.alarm(0);e.update(terminal=True,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e);write(OUT/'summary.json',dict(**e,cases=rows,mesh_spatial_qualified=False,surface_trajectory_qualified=False,nonclaims=p['nonclaims']))
 return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch','validate']);p.add_argument('--protocol');p.add_argument('--sha');p.add_argument('--case');a=p.parse_args()
 if a.action=='prepare':print(json.dumps(prepare()))
 elif a.action=='validate':p=validate(a.protocol,a.sha);print(json.dumps(dict(validated=True,phase=p['phase_id'],GPU_calls=False)))
 else:
  r=run(a.protocol,a.sha,a.case) if a.action=='run' else launch(a.protocol,a.sha);print(json.dumps({k:r.get(k) for k in ['completed','passed','time_reference_qualified','elapsed_s','error']}));raise SystemExit(0 if r.get('completed') else 1)
