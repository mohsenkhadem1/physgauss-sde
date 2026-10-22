"""Higher-rate fixed-law P2 contact with exact analytic freefall initialization."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time,traceback
import numpy as np
from experiments.run_p2_contact_validation import ROOT,GPU,sha,write,build,SOURCES as BASE_SOURCES
OP=ROOT/'runs/operator-dev/p2-contact-refinement-v02';OUT=ROOT/'runs/gaussian-campaign-v01/p2-contact-refinement-v02'
RATES=[1920,3840,7680,15360];T0=1/12;END=.15;GRAVITY=np.array([0.,0.,-9.81]);SOURCES=BASE_SOURCES+['experiments/run_p2_contact_refinement.py','tests/test_run_p2_contact_refinement.py']

def initial_vectors(rest,h):
 rest=np.asarray(rest,float)
 if h<=0 or h>=T0 or rest.ndim!=2 or rest.shape[1]!=3:raise ValueError('Declaredfinitephysicalstep/geometry required')
 now=np.broadcast_to(.5*T0*T0*GRAVITY,rest.shape).copy();v=np.broadcast_to(T0*GRAVITY,rest.shape).copy();old=np.broadcast_to(.5*(T0-h)**2*GRAVITY,rest.shape).copy();oldv=np.broadcast_to((T0-h)*GRAVITY,rest.shape).copy()
 if np.min(rest[:,2]+now[:,2])<=0:raise ValueError('Analyticinitializationmuststrictlyprecedecontact')
 return now,v,old,oldv

def can_stop(errors):return len(errors)>=2 and errors[-1]<=20e-6 and errors[-1]<errors[-2]

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenstage')
 prior=ROOT/'runs/gaussian-campaign-v01/p2-contact-validation-v01/summary.json';old=json.loads(prior.read_text())
 if len(old['controls'])!=2 or not all(c['passed'] for c in old['controls']):raise ValueError('Completedindependentcontrolsrequired')
 p=dict(schema='Analytic-start-P2-contact-rate-refinement-v0.2',status='FROZEN_BEFORE_NEW_PHYSICS',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=1800,source_sha256={f:sha(ROOT/f) for f in SOURCES},prior_controls=dict(path=str(prior.relative_to(ROOT)),sha256=sha(prior)),rates_hz=RATES,cells=[4,2,2],conditional_fine_cells=[8,4,4],physics=dict(size_m=[.24,.08,.08],young_Pa=300000.,nu=.35,rho_kg_m3=625.,gravity_m_s2=GRAVITY.tolist(),initial_bottom_gap_m=.05,kappa_N_m3=1e9,surface_order=8,friction=0.,damping=0.,fixed_world_plane_z_m=0.,boundary='allfree',law='Finite reference-area P2 surface penalty, stableNH, fullconsistentT10mass'),initialization=dict(time_s=T0,end_s=END,current_u='.5*g*t0^2',current_v='g*t0',previous_u='.5*g*(t0-h)^2',previous_v='g*(t0-h)',minimum_initial_gap_m=.05-.5*9.81*T0*T0,role='Exactstressfreeanalyticcontinuumfreefall; explicitlydifferentnumericalstartupfromoldBE-startstage'),gates=dict(objective_R_N=1e-10,physical_R_N=1e-9,initializer_position_error_m=1e-10,minimum_J=.1,common120Hz_81query_difference_m=20e-6),early_stop='Atleasttwoadjacenterrors;current<=20umANDstrictlycontractsagainstpreviouserror',conditional_fine_forecast='Onlyafterpassingratepair; remainingwalltime >8*finest96runtime+30s; samechosenfineHz. Planningestimateonly',checkpoint_interval_steps=32,nonclaims=['FixedfinitecompliantfloorONLY, no rigidnonpenetration','Analyticfreefallstartupisnumericalchoice, noalteredmaterial/load','Timecriteriondoesnotestablishspatialcriterion','AllnativeDPpositionskept; no kinematicrenderprojection'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 if any(sha(ROOT/f)!=v for f,v in p['source_sha256'].items()) or sha(ROOT/p['prior_controls']['path'])!=p['prior_controls']['sha256']:raise ValueError('Changeddependency')
 OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,time_qualified=False,space_qualified=False,pid=os.getpid(),protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),cases=[],adjacent_checks=[],nonclaims=p['nonclaims']);current=previous=None
 try:
  import torch
  from experiments.campaign_p2_torch_contact import PlaneContactP2
  from experiments.campaign_p2_transient import P2State,step_p2
  torch.set_num_threads(2);device='cuda:0';torch.ones(1,device=device).sum().item()
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True);own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())];s.update(own_compute_records=own,torch_version=torch.__version__);write(OUT/'summary.json',s)
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('WrongphysicalGPU')
  probe=np.array([[x,y,z] for x in np.linspace(0,.24,9) for y in [-.04,0,.04] for z in [-.04,0,.04]])+np.array([0,0,.09]);fields={};errors=[];qualified_rate=None
  def case(cells,hz,control=False):
   nonlocal current,previous
   started=time.monotonic();m=build(cells,300000.,9.81);obj=PlaneContactP2(m,device,kappa_A=1e9,floor_z=0.,surface_order=8);arrays=initial_vectors(m.rest,1/hz);tensor=lambda x:torch.tensor(x,dtype=torch.float64,device=device);current=P2State(tensor(arrays[0]),tensor(arrays[1]),T0);previous=P2State(tensor(arrays[2]),tensor(arrays[3]),T0-1/hz);label=f'tets{len(m.tets)}-hz{hz}';initial=obj.evaluate(current.u)
   if bool(initial['contact_active'].any()):raise ValueError('Analyticinitializerhascontact')
   if control:
    nxt,row=step_p2(obj,current,previous,1/hz,obj.force);expected=.5*(T0+1/hz)**2*GRAVITY;error=float(abs(nxt.u.cpu().numpy()-expected).max());s['initializer_control']=dict(passed=error<=1e-10,maximum_position_error_m=error,original_physical_R_N=row['original_physical_free_R_N'],initial_minimum_gap_m=float(initial['contact_gap'].min()),initial_contact_energy_J=float(initial['contact_energy']))
    if error>1e-10:raise RuntimeError('Analyticinitializercontrolfailed')
    # Start the actualcase againfromexactspecifiedanalyticstate.
   positions=[(obj.rest+current.u).cpu().numpy()];velocities=[current.v.cpu().numpy()];samples=[m.query(positions[0],probe)];ledger=[];steps=hz//15;stride=hz//120
   def checkpoint():
    tmp=OUT/(label+'-checkpoint.tmp.npz');f=OUT/(label+'-checkpoint.npz');np.savez_compressed(tmp,accepted_u_m=current.u.detach().cpu().numpy(),accepted_v_m_s=current.v.detach().cpu().numpy(),previous_u_m=previous.u.detach().cpu().numpy(),previous_v_m_s=previous.v.detach().cpu().numpy(),time_s=current.time_s,step=len(ledger),h_s=1/hz,positions_m=np.asarray(positions),velocities_m_s=np.asarray(velocities),query_positions_m=np.asarray(samples),protocol_sha256=digest);tmp.replace(f);write(OUT/(label+'-checkpoint.json'),dict(last_accepted_step=len(ledger),time_s=current.time_s,checkpoint_sha256=sha(f),ledger=ledger))
   checkpoint()
   for k in range(1,steps+1):
    nxt,row=step_p2(obj,current,previous,1/hz,obj.force);row.update(obj.contact_diagnostics(nxt.u));row['total_mechanical_energy_J']=row['internal_potential_energy_J']+row['kinetic_energy_J']-float((obj.force*nxt.u).sum());previous,current=current,nxt;ledger.append(row);q=(obj.rest+current.u).detach().cpu().numpy();positions.append(q);velocities.append(current.v.detach().cpu().numpy())
    if k%stride==0:samples.append(m.query(q,probe))
    if k%32==0:checkpoint()
   checkpoint();f=OUT/(label+'.npz');np.savez_compressed(f,times_s=T0+np.arange(steps+1)/hz,positions_m=np.asarray(positions),velocities_m_s=np.asarray(velocities),rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,surface_elements=obj.surface_elements.cpu().numpy(),floor_z_m=0.,fixed_node_ids=np.array([],int),query_times_s=T0+np.arange(9)/120,material_probes_m=probe,query_positions_m=np.asarray(samples),volume_integration_weights_kg=m.density*m.body_integral_weights)
   row=dict(cells=cells,hz=hz,tets=len(m.tets),nodes=len(m.rest),completed=True,elapsed_s=time.monotonic()-started,maximum_original_physical_R_N=max(r['original_physical_free_R_N'] for r in ledger),maximum_objective_R_N=max(r['original_objective_free_R_N'] for r in ledger),minimum_J=min(r['minimum_J'] for r in ledger),maximum_analytic_face_penetration_m=max(r['maximum_analytic_P2_face_penetration_m'] for r in ledger),maximum_quadrature_penetration_m=max(r['maximum_quadrature_penetration_m'] for r in ledger),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),ledger=ledger);s['cases'].append(row);write(OUT/(label+'.json'),row);write(OUT/'summary.json',s);return np.asarray(samples),row
  for hz in RATES:
   q,row=case([4,2,2],hz,control=hz==RATES[0]);fields[hz]=q
   if len(fields)>1:
    prevhz=RATES[RATES.index(hz)-1];error=float(np.linalg.norm(q-fields[prevhz],axis=2).max());errors.append(error);s['adjacent_checks'].append(dict(coarse_hz=prevhz,fine_hz=hz,maximum_fixedquery_difference_m=error,tolerance_m=20e-6,contraction_ratio=errors[-1]/errors[-2] if len(errors)>1 else None,passed=error<=20e-6));write(OUT/'summary.json',s)
    if can_stop(errors):qualified_rate=hz;s['time_qualified']=True;break
  if qualified_rate:
   finest=s['cases'][-1];remaining=1800-(time.monotonic()-begin);forecast=8*finest['elapsed_s']+30;s['fine_mesh_forecast']=dict(remaining_s=remaining,forecast_s=forecast,will_run=remaining>forecast)
   if remaining>forecast:
    fine,row=case([8,4,4],qualified_rate);error=float(np.linalg.norm(fine-fields[qualified_rate],axis=2).max());s['spatial_check']=dict(hz=qualified_rate,coarse_tets=96,fine_tets=768,maximum_fixedquery_difference_m=error,tolerance_m=20e-6,passed=error<=20e-6);s['space_qualified']=error<=20e-6
  s.update(completed=True,selected_rate_hz=qualified_rate,maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
 except Exception as e:
  if current is not None:np.savez_compressed(OUT/'last-accepted-state.npz',u_m=current.u.detach().cpu().numpy(),v_m_s=current.v.detach().cpu().numpy(),previous_u_m=previous.u.detach().cpu().numpy() if previous is not None else np.zeros_like(current.u.cpu().numpy()),time_s=current.time_s)
  s.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 return s

def launch(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(r.split(',')[1].strip()==GPU for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU1notidle')
 e=dict(controller_pid=os.getpid(),terminal=False,completed=False,wall_cap_s=1800,protocol_sha256=digest);write(OP/'execution.json',e);begin=time.monotonic()
 with (OP/'worker.log').open('x') as log:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_contact_refinement','run','--sha',digest],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});e['worker_pid']=child.pid;write(OP/'execution.json',e)
  try:code=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;e['timeout']=True
 s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=code==0 and s.get('completed') is True,exit_code=code,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e);return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','time_qualified','space_qualified','elapsed_s','error','exit_code']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
