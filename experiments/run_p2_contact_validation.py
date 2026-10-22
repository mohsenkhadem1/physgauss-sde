"""Bounded float64 P2 compliant-plane canonical and drop refinement screen."""
from __future__ import annotations
import argparse,hashlib,json,math,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/p2-contact-validation-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-contact-validation-v01';GPU='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb'
SOURCES=['experiments/run_p2_contact_validation.py','experiments/campaign_p2_torch_contact.py','experiments/campaign_p2_transient.py','experiments/campaign_p2_torch.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py','experiments/campaign_newton_contact_bdf2.py','tests/test_run_p2_contact_validation.py']
JOBS=[dict(name=f'drop6_hz{hz}',cells=[1,1,1],hz=hz,kappa=1e9,Q=8,E=300000.) for hz in [240,480,960]]+[dict(name='drop96_hz960',cells=[4,2,2],hz=960,kappa=1e9,Q=8,E=300000.),dict(name='drop6_Q16',cells=[1,1,1],hz=960,kappa=1e9,Q=16,E=300000.),dict(name='drop6_kappa10',cells=[1,1,1],hz=960,kappa=1e10,Q=8,E=300000.),dict(name='drop6_E600k',cells=[1,1,1],hz=960,kappa=1e9,Q=8,E=600000.)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def analytic_triangle_mass():
 # Independent exact barycentric polynomial integration, no Duffy quadrature.
 pol=[]
 for i in range(3):
  a=[0,0,0];a[i]=2;b=[0,0,0];b[i]=1;pol.append({tuple(a):2,tuple(b):-1})
 for i,j in [(0,1),(0,2),(1,2)]:
  a=[0,0,0];a[i]=a[j]=1;pol.append({tuple(a):4})
 def integrate(a):return 2*math.prod(math.factorial(n) for n in a)/math.factorial(sum(a)+2)
 c=np.array([sum(v*integrate(a) for a,v in p.items()) for p in pol]);G=np.array([[sum(v*w*integrate(tuple(x+y for x,y in zip(a,b))) for a,v in p.items() for b,w in q.items()) for q in pol] for p in pol])
 return c,G

def exact_allactive_surface(model,q,kappa,floor):
 from experiments.campaign_p2_torch_contact import boundary_trace
 ids,A=boundary_trace(model);c,G=analytic_triangle_mass();z=np.asarray(q)[ids,2];g=np.zeros_like(q)
 gz=kappa*A[:,None]*(z@G-floor*c);np.add.at(g[:,2],ids.ravel(),gz.ravel())
 U=.5*kappa*np.sum(A*(np.einsum('fa,ab,fb->f',z,G,z)-2*floor*(z@c)+floor*floor))
 return U,g

def build(cells,E,gravity,shift=.09):
 from experiments.campaign_newton_stable_material import StableP2Block
 m=StableP2Block(tuple(cells),young=E,poisson=.35,density=625.,gravity=gravity,traction=0.,order=3,boundary='left_face');m.mask[:]=False;m.free=np.arange(m.rest.size);m.rest=m.rest+np.array([0,0,shift]);return m

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozenstage')
 p=dict(schema='Float64-P2-compliant-contact-validation-v0.1',status='FROZEN_BEFORE_GPU_PHYSICS',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=600,source_sha256={f:sha(ROOT/f) for f in SOURCES},jobs=JOBS,physics=dict(size_m=[.24,.08,.08],rho_kg_m3=625.,nu=.35,gravity_m_s2=[0,0,-9.81],initial_bottom_gap_m=.05,duration_s=.15,friction=0.,contact_damping=0.,material_damping=0.,floor_z_m=0.,law='ReferenceareaP2surfacepenalty finitekappaN/m3; no rigidnonpenetration',time='BEstartup+BDF2 physicalD2q=v,MD2v+gradU=F; fullconsistentT10mass'),controls=['Independentexactbarycentricpolynomial allactivecontactforce/energy+CPUstableNHmanufacturedobjectiveGPUsolve','6tetfreefallBDF2exactscalaroracle240Hz.025s'],gates=dict(original_objective_R_N=1e-10,original_physical_R_N=1e-9,minimum_J=.1,manufactured_displacement_error_m=1e-8,manufactured_force_error_N=1e-8,freefall_scalar_position_error_m=1e-10,maximum_common120Hz_fixed81query_difference_m=20e-6),nonclaims=['Finitecompliantfloor, no friction/sticking/CCD/nonpenetration','Screensnotautomaticdataadmission; failedtime/mesh/Q/kappachecksretained','6tet/96tetcomparisonispreliminaryspatialscreen, notcontinuumcertificate','Common120Hzquerychecks do notboundunobservedbetweenoutput extrema'])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 if sha(OP/'protocol.json')!=digest or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Changedprotocol/device')
 p=json.loads((OP/'protocol.json').read_text())
 if any(sha(ROOT/f)!=v for f,v in p['source_sha256'].items()):raise ValueError('Changedsource')
 OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,qualified=False,pid=os.getpid(),protocol=dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=digest),controls=[],cases=[],nonclaims=p['nonclaims']);current=None;previous=None
 try:
  import torch
  from experiments.campaign_p2_torch import TorchP2
  from experiments.campaign_p2_torch_contact import PlaneContactP2
  from experiments.campaign_p2_transient import P2State,step_p2
  from experiments.campaign_newton_contact_bdf2 import scalar_step
  torch.set_num_threads(2);device='cuda:0';torch.ones(1,device=device).sum().item()
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True);own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())];s['own_compute_records']=own;s['torch_version']=torch.__version__;write(OUT/'summary.json',s)
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('WrongphysicalGPU')
  m=build([1,1,1],300000.,0.,shift=0.);obj=PlaneContactP2(m,device,kappa_A=100.,floor_z=1.,surface_order=4);u=np.zeros_like(m.rest);u[:,0]=.002*m.rest[:,0];u[:,2]=.001*m.rest[:,0]-.001
  q=m.rest+u;U,g=exact_allactive_surface(m,q,100.,1.);elastic,ge,_,J=m.elastic(q);expected_force=g+ge
  target=torch.tensor(u,dtype=torch.float64,device=device);cache=obj.evaluate(target);gerr=float(abs(cache['gradient'].cpu().numpy()-expected_force).max());Uerr=abs(float(cache['contact_energy'])-U)
  ans=obj.solve(force=torch.tensor(expected_force,dtype=torch.float64,device=device),target=target,effective_dt=.01);qerr=float(abs(ans['displacement'].cpu().numpy()-u).max());passed=gerr<=1e-8 and Uerr<=1e-12 and qerr<=1e-8
  s['controls'].append(dict(name='independentmanufacturedallactive',passed=passed,maximum_force_error_N=gerr,contact_energy_error_J=Uerr,maximum_displacement_error_m=qerr,original_residual_N=float(torch.linalg.vector_norm(ans['residual']))));write(OUT/'summary.json',s)
  if not passed:raise RuntimeError('Independentcontactforcecontrolfailed')
  m=build([1,1,1],300000.,9.81);obj=TorchP2(m,device);current=P2State(torch.zeros_like(obj.rest),torch.zeros_like(obj.rest));previous=None;z=v=oldz=oldv=0.;error=0.;maxR=0.
  for k in range(6):
   new,row=step_p2(obj,current,previous,1/240,obj.force);args={} if k==0 else dict(oldz=oldz,oldv=oldv);zz,vv=scalar_step(z,v,1/240,mass=.96,k=1.,radius=-10.,**args);oldz,oldv=z,v;z,v=zz,vv;error=max(error,float(abs(new.u[:,2]-z).max()));maxR=max(maxR,row['original_physical_free_R_N']);previous,current=current,new
  s['controls'].append(dict(name='independentfreefallscalarBDF2',passed=error<=1e-10,maximum_scalar_position_error_m=error,maximum_original_physical_R_N=maxR));write(OUT/'summary.json',s)
  if error>1e-10:raise RuntimeError('Freefallcontrolfailed')
  fields={};probe0=np.array([[x,y,z] for x in np.linspace(0,.24,9) for y in [-.04,0,.04] for z in [-.04,0,.04]])+np.array([0,0,.09])
  for job in JOBS:
   started=time.monotonic();m=build(job['cells'],job['E'],9.81);obj=PlaneContactP2(m,device,kappa_A=job['kappa'],floor_z=0.,surface_order=job['Q']);current=P2State(torch.zeros_like(obj.rest),torch.zeros_like(obj.rest));previous=None;positions=[m.rest.copy()];velocities=[np.zeros_like(m.rest)];ledger=[];sample_positions=[m.query(m.rest,probe0)];steps=round(.15*job['hz']);sampleids=np.arange(0,steps+1,job['hz']//120)
   for k in range(1,steps+1):
    new,row=step_p2(obj,current,previous,1/job['hz'],obj.force);diagnostics=obj.contact_diagnostics(new.u);row.update(diagnostics);row['total_mechanical_energy_J']=row['internal_potential_energy_J']+row['kinetic_energy_J']-float((obj.force*new.u).sum());previous,current=current,new;ledger.append(row);q=(obj.rest+current.u).detach().cpu().numpy();positions.append(q);velocities.append(current.v.detach().cpu().numpy())
    if k%(job['hz']//120)==0:sample_positions.append(m.query(q,probe0))
   f=OUT/(job['name']+'.npz');np.savez_compressed(f,times_s=np.arange(steps+1)/job['hz'],positions_m=np.array(positions),velocities_m_s=np.array(velocities),rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,surface_elements=obj.surface_elements.cpu().numpy(),fixed_node_ids=np.array([],int),floor_z_m=0.,query_times_s=sampleids/job['hz'],material_probes_m=probe0,query_positions_m=np.array(sample_positions))
   row=dict(**job,completed=True,elapsed_s=time.monotonic()-started,nodes=len(m.rest),tets=len(m.tets),maximum_original_physical_R_N=max(r['original_physical_free_R_N'] for r in ledger),maximum_original_objective_R_N=max(r['original_objective_free_R_N'] for r in ledger),minimum_J=min(r['minimum_J'] for r in ledger),maximum_analytic_face_penetration_m=max(r['maximum_analytic_P2_face_penetration_m'] for r in ledger),maximum_quadrature_penetration_m=max(r['maximum_quadrature_penetration_m'] for r in ledger),maximum_CG_iterations=max(r['CG_iterations'] for r in ledger),maximum_Newton_iterations=max(r['Newton_iterations'] for r in ledger),capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)),ledger=ledger)
   s['cases'].append(row);fields[job['name']]=np.array(sample_positions);write(OUT/(job['name']+'.json'),row);write(OUT/'summary.json',s)
  pairs=[('drop6_hz240','drop6_hz480','time240to480'),('drop6_hz480','drop6_hz960','time480to960'),('drop6_hz960','drop96_hz960','mesh6to96'),('drop6_hz960','drop6_Q16','surfaceQ8to16'),('drop6_hz960','drop6_kappa10','kappa1e9to1e10')];checks=[]
  for a,b,role in pairs:
   error=float(np.linalg.norm(fields[a]-fields[b],axis=2).max());checks.append(dict(role=role,maximum_common_fixedquery_difference_m=error,tolerance_m=20e-6,passed=error<=20e-6))
  s.update(completed=True,qualified=all(r['passed'] for r in checks),refinement_checks=checks,maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
 except Exception as e:
  if current is not None:
   np.savez_compressed(OUT/'last-accepted-state.npz',displacement_m=current.u.detach().cpu().numpy(),velocities_m_s=current.v.detach().cpu().numpy(),time_s=current.time_s)
  s.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 return s

def launch(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changedprotocol')
 if (OP/'execution.json').exists():raise FileExistsError('Preserveexecution')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if any(r.split(',')[1].strip()==GPU for r in apps.splitlines() if ',' in r):raise RuntimeError('GPU1notidle')
 e=dict(controller_pid=os.getpid(),terminal=False,completed=False,wall_cap_s=600,protocol_sha256=digest);write(OP/'execution.json',e);start=time.monotonic()
 with (OP/'worker.log').open('x') as log:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_p2_contact_validation','run','--sha',digest],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});e['worker_pid']=child.pid;write(OP/'execution.json',e)
  try:code=child.wait(timeout=600)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode;e['timeout']=True
 s=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};e.update(terminal=True,completed=code==0 and s.get('completed') is True,qualified=s.get('qualified',False),exit_code=code,elapsed_s=time.monotonic()-start);write(OP/'execution.json',e);return e
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','launch']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha) if a.action=='run' else launch(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','qualified','elapsed_s','error','exit_code']}));raise SystemExit(0 if a.action=='prepare' or r.get('completed') else 1)
