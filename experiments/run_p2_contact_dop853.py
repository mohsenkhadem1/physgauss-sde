"""Independent adaptive high-order ODE time reference for fixed P2 contact.

u'=v, M v'=Fgravity-grad(Uelastic+Ucontact); full constant consistent mass.
Two declared tolerances; positive-J guarded RHS stages; physical energy and
integrated-force momentum checked only at accepted states and fixed outputs.
"""
from __future__ import annotations
import argparse,importlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from experiments.run_p2_contact_validation import ROOT,GPU,sha,write,build,SOURCES as BASE_SOURCES
from experiments.run_p2_contact_refinement import T0,END,GRAVITY,initial_vectors
OP=ROOT/'runs/operator-dev/p2-contact-dop853-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-contact-dop853-v01'
SCIPY_ROOT=Path(importlib.import_module('scipy').__file__).resolve().parent
SCIPY_FILES=['integrate/_ivp/ivp.py','integrate/_ivp/common.py','integrate/_ivp/rk.py','integrate/_ivp/dop853_coefficients.py']
SOURCES=BASE_SOURCES+['experiments/run_p2_contact_dop853.py','experiments/run_p2_contact_refinement.py','experiments/campaign_p2_consistent_transient.py','experiments/campaign_stable_transient.py','tests/test_run_p2_contact_dop853.py']
CONFIGS=[dict(name='accuracy1',rtol=1e-7,displacement_atol_m=1e-10,velocity_atol_m_s=1e-8,impulse_atol_Ns=1e-10),dict(name='accuracy2',rtol=1e-9,displacement_atol_m=1e-12,velocity_atol_m_s=1e-10,impulse_atol_Ns=1e-12)]

def integrate(rhs,initial,interval,outputs,config,atol,event=None):
 return solve_ivp(rhs,interval,initial,method='DOP853',rtol=config['rtol'],atol=atol,max_step=1/30000,dense_output=True,t_eval=outputs,events=event)

def accepted_partition(result):
 if result.sol is None or not hasattr(result.sol,'ts') or not hasattr(result.sol,'interpolants'):raise ValueError('PinnedSciPyOdeSolutionAPIrequired')
 t=np.asarray(result.sol.ts,float)
 if len(t)!=len(result.sol.interpolants)+1 or np.any(np.diff(t)<=0):raise ValueError('Invalidacceptedpartition')
 return t

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenstage')
 p=dict(schema='AdaptiveDOP853-fullmass-P2-compliant-contact-v0.1',status='FROZEN_BEFORE_GPU_PHYSICS',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=900,source_sha256={f:sha(ROOT/f) for f in SOURCES},scipy_source_sha256={f:sha(SCIPY_ROOT/f) for f in SCIPY_FILES},configs=CONFIGS,cells=[4,2,2],physics=dict(size_m=[.24,.08,.08],E_Pa=300000.,nu=.35,rho_kg_m3=625.,gravity_m_s2=GRAVITY.tolist(),initial_bottom_gap_m=.05,kappa_N_m3=1e9,surface_Q=8,friction=0.,damping=0.,plane_z_m=0.,BC='allfree',mass='ExactfullconsistentT10constantSPDscalarMreused3coords',law='SamefixedstableNH+finitecompliantreferenceareaP2surfacepenalty'),interval_s=[T0,END],initializer='Exactfreefallu=.5gt0²,v=gt0; no numericalhistoryneeded',maximum_step_s=1/30000,outputs_s=(T0+np.arange(9)/120).tolist(),RHS='[v,Cholesky(M)solve(Fgravity-gradU),sum_i(Fgravity-gradU)_i]',gates=dict(masssolve_R_N=1e-8,minimum_J=.1,maximum_accuracy_pair_fixedfield_q_m=20e-6,maximum_accuracy_pair_velocity_m_s=1e-4,energy_balance_J='max(1e-9,1e-6*initialdropenergy)',momentum_balance_Ns=1e-8),accepted_state_checks='Evaluateallresult.sol.ts endpointstates AND9densecommonoutputs; RHSstageenergyNOTusedasacceptedenergy',checkpoints='Constant+1nonterminatingevent observesacceptedstates; checkpoint every64acceptedsteps; neverchangesRHS/stepacceptance',nonclaims=['AuxiliarymomentumusesidenticalRKquadrature;consistencychecknotindependenttimeaccuracy','Semidiscretefixedmesh/fixedQ/fixedkappa referenceonly','No hard-floor, friction, continuum spatial or Q qualificationfromtoleranceagreement','Two adaptive tolerances are numericalchecks, notrigorouserrorbounds'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 if any(sha(ROOT/f)!=v for f,v in p['source_sha256'].items()) or any(sha(SCIPY_ROOT/f)!=v for f,v in p['scipy_source_sha256'].items()):raise ValueError('Changedsource')
 OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,time_reference_qualified=False,pid=os.getpid(),protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),cases=[],nonclaims=p['nonclaims'])
 try:
  import torch,scipy
  from experiments.campaign_p2_torch_contact import PlaneContactP2
  from experiments.campaign_p2_consistent_transient import scalar_mass
  torch.set_num_threads(2);device='cuda:0';torch.ones(1,device=device).sum().item()
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True);own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())];s.update(own_compute_records=own,versions=dict(torch=torch.__version__,scipy=scipy.__version__,numpy=np.__version__));write(OUT/'summary.json',s)
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('WrongphysicalGPU')
  m=build([4,2,2],300000.,9.81);obj=PlaneContactP2(m,device,kappa_A=1e9,floor_z=0.,surface_order=8);M=torch.tensor(scalar_mass(m).toarray(),dtype=torch.float64,device=device);L=torch.linalg.cholesky(M);n=len(m.rest);size=3*n;force=obj.force
  torch.testing.assert_close(M@torch.ones((n,3),dtype=torch.float64,device=device)*torch.tensor(GRAVITY,dtype=torch.float64,device=device),force,atol=1e-13,rtol=1e-13)
  u0,v0,_,_=initial_vectors(m.rest,1/1920);y0=np.r_[u0.ravel(),v0.ravel(),np.zeros(3)];initialP=(M@torch.tensor(v0,dtype=torch.float64,device=device)).sum(0).cpu().numpy();totalmass=float(M.sum());dropenergy=totalmass*9.81*.05;energytol=max(1e-9,1e-6*dropenergy);initial_cache=obj.evaluate(torch.tensor(u0,dtype=torch.float64,device=device));initialE=float(initial_cache['energy']+.5*(torch.tensor(v0,dtype=torch.float64,device=device)*(M@torch.tensor(v0,dtype=torch.float64,device=device))).sum()-(force*torch.tensor(u0,dtype=torch.float64,device=device)).sum());s.update(nodes=n,tets=len(m.tets),total_consistent_mass_kg=totalmass,initial_drop_energy_J=dropenergy,energy_tolerance_J=energytol,initial_mechanical_energy_J=initialE);write(OUT/'summary.json',s)
  probe=np.array([[x,y,z] for x in np.linspace(0,.24,9) for y in [-.04,0,.04] for z in [-.04,0,.04]])+np.array([0,0,.09]);fields={};vels={};energies={}
  for config in CONFIGS:
   started=time.monotonic();label=config['name'];stats=dict(rhs_calls=0,maximum_masssolve_R_N=0.,minimum_RHS_J=1.);event_times=[]
   def rhs(t,y):
    u=torch.tensor(y[:size].reshape(n,3),dtype=torch.float64,device=device);v=y[size:2*size].reshape(n,3);cache=obj.evaluate(u);net=force-cache['gradient'];acc=torch.cholesky_solve(net,L);R=float(torch.linalg.vector_norm(M@acc-net));stats['rhs_calls']+=1;stats['maximum_masssolve_R_N']=max(stats['maximum_masssolve_R_N'],R);stats['minimum_RHS_J']=min(stats['minimum_RHS_J'],float(cache['minimum_J']))
    if not np.isfinite(R) or R>1e-8:raise ValueError('OriginalconsistentmasssolveRfailed')
    return np.r_[v.ravel(),acc.detach().cpu().numpy().ravel(),net.sum(0).detach().cpu().numpy()]
   def checkpoint(t,y):
    # scipy1.18.1callsconstantnonzeroeventsatinitial/acceptedstepstatesonly.
    event_times.append(float(t))
    if len(event_times)%64==0 or t==END:
     tmp=OUT/(label+'-accepted-checkpoint.tmp.npz');f=OUT/(label+'-accepted-checkpoint.npz');np.savez_compressed(tmp,accepted_time_s=t,state=y,protocol_sha256=digest);tmp.replace(f);write(OUT/(label+'-progress.json'),dict(accepted_steps=len(event_times)-1,accepted_time_s=t,**stats))
    return 1.
   atol=np.r_[np.full(size,config['displacement_atol_m']),np.full(size,config['velocity_atol_m_s']),np.full(3,config['impulse_atol_Ns'])];times=np.asarray(p['outputs_s']);answer=integrate(rhs,y0.copy(),[T0,END],times,config,atol,checkpoint)
   if not answer.success:raise RuntimeError('DOP853failed:'+answer.message)
   accepted=accepted_partition(answer)
   np.testing.assert_array_equal(accepted,np.asarray(event_times));np.testing.assert_allclose(accepted[[0,-1]],[T0,END],atol=0,rtol=0)
   ya=answer.sol(accepted);q=answer.y[:size].T.reshape(-1,n,3)+m.rest;v=answer.y[size:2*size].T.reshape(-1,n,3);acceptedE=[];acceptedP=[];acceptedMinJ=1.;acceptedPen=0.;outputE=[];outputP=[];outputPen=[]
   def balance(y):
    u=torch.tensor(y[:size].reshape(n,3),dtype=torch.float64,device=device);v=torch.tensor(y[size:2*size].reshape(n,3),dtype=torch.float64,device=device);cache=obj.evaluate(u);E=float(cache['energy']+.5*(v*(M@v)).sum()-(force*u).sum());P=(M@v).sum(0).cpu().numpy()-initialP-y[2*size:];diagnostic=obj.contact_diagnostics(u,cache);return E,float(np.linalg.norm(P)),float(cache['minimum_J']),diagnostic['maximum_analytic_P2_face_penetration_m']
   for j in range(len(accepted)):
    E,P,J,penetration=balance(ya[:,j]);acceptedE.append(E);acceptedP.append(P);acceptedMinJ=min(acceptedMinJ,J);acceptedPen=max(acceptedPen,penetration)
   for j in range(len(times)):
    E,P,J,penetration=balance(answer.y[:,j]);outputE.append(E);outputP.append(P);outputPen.append(penetration)
   energymat=np.asarray(acceptedE);energyerror=max(float(abs(energymat-initialE).max()),float(abs(np.asarray(outputE)-initialE).max()));perror=float(max(acceptedP+outputP));pred=np.asarray([m.query(x,probe) for x in q]);passed=energyerror<=energytol and perror<=1e-8 and stats['maximum_masssolve_R_N']<=1e-8
   f=OUT/(label+'.npz');np.savez_compressed(f,times_s=times,positions_m=q,velocities_m_s=v,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,surface_elements=obj.surface_elements.cpu().numpy(),fixed_node_ids=np.array([],int),floor_z_m=0.,query_times_s=times,query_positions_m=pred,material_probes_m=probe,accepted_times_s=accepted,accepted_states=ya,accepted_mechanical_energies_J=energymat,accepted_momentum_balance_errors_Ns=np.asarray(acceptedP),output_mechanical_energies_J=np.asarray(outputE),output_momentum_balance_errors_Ns=np.asarray(outputP),output_analytic_face_penetrations_m=np.asarray(outputPen),volume_integration_weights_kg=m.density*m.body_integral_weights)
   row=dict(**config,completed=True,passed=passed,elapsed_s=time.monotonic()-started,nfev=answer.nfev,accepted_steps=len(accepted)-1,**stats,minimum_accepted_J=acceptedMinJ,maximum_accepted_analytic_face_penetration_m=acceptedPen,maximum_output_momentum_balance_error_Ns=max(outputP),maximum_output_energy_balance_error_J=float(abs(np.asarray(outputE)-initialE).max()),maximum_accepted_energy_balance_error_J=float(abs(energymat-initialE).max()),maximum_accepted_momentum_balance_error_Ns=float(max(acceptedP)),maximum_output_analytic_face_penetration_m=max(outputPen),output_energy_J=outputE,capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)));s['cases'].append(row);fields[label]=pred;vels[label]=v;energies[label]=np.asarray(outputE);write(OUT/(label+'.json'),row);write(OUT/'summary.json',s)
    # Conservationfailuresarepreserved; stillcompletebothfixedaccuracychecks.
  qerror=float(np.linalg.norm(fields['accuracy1']-fields['accuracy2'],axis=2).max());verror=float(np.linalg.norm(vels['accuracy1']-vels['accuracy2'],axis=2).max());Eerror=float(abs(energies['accuracy1']-energies['accuracy2']).max());pair=dict(maximum_fixedquery_position_difference_m=qerror,maximum_allnode_velocity_difference_m_s=verror,maximum_output_energy_difference_J=Eerror,passed=qerror<=20e-6 and verror<=1e-4 and Eerror<=energytol);s.update(completed=True,time_reference_qualified=all(r['passed'] for r in s['cases']) and pair['passed'],accuracy_pair=pair,maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
 except Exception as e:s.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 return s

def launch(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(r.split(',')[1].strip()==GPU for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU1notidle')
 e=dict(controller_pid=os.getpid(),terminal=False,completed=False,wall_cap_s=900,protocol_sha256=digest);write(OP/'execution.json',e);begin=time.monotonic()
 with (OP/'worker.log').open('x') as log:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_contact_dop853','run','--sha',digest],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});e['worker_pid']=child.pid;write(OP/'execution.json',e)
  try:code=child.wait(timeout=900)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;e['timeout']=True
 s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=code==0 and s.get('completed') is True,time_reference_qualified=s.get('time_reference_qualified',False),exit_code=code,elapsed_s=time.monotonic()-begin);write(OP/'execution.json',e);return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','time_reference_qualified','elapsed_s','error','exit_code']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
