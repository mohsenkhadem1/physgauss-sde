"""Prospective standalone NewtonVBD API/constitutive/force smoke, not convergence."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
import numpy as np
try:
 from .boundary_block import block_mesh,lumped_masses
except ImportError:
 from boundary_block import block_mesh,lumped_masses
ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'runs/operator-dev/newton-runtime-v01/env';SITE=ENV/'lib/python3.12/site-packages'
PROTOCOL='runs/operator-dev/newton-vbd-probe-v01/protocol.json';OUTPUT='runs/gaussian-campaign-v01/newton-vbd-probe-v01'
SOURCES=['experiments/campaign_newton_vbd_probe.py','experiments/campaign_newton_vbd_kernels.py','experiments/boundary_block.py','tests/test_campaign_newton_vbd_probe.py']
EXTERNAL=['newton/_src/solvers/vbd/particle_vbd_kernels.py','newton/_src/solvers/vbd/solver_vbd.py','newton/_src/sim/builder.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def material(E=5000.,nu=.35):
 if not np.isfinite([E,nu]).all() or E<=0 or not 0<nu<.5:raise ValueError('Finite positive E and0<nu<.5 required')
 mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));return mu,lam

def element_reference(X,T,q,E=5000.,nu=.35):
 X=np.asarray(X,dtype=float);T=np.asarray(T,dtype=int);q=np.asarray(q,dtype=float);mu,lam=material(E,nu);kappa=lam+mu
 Dm=np.swapaxes(X[T[:,1:]]-X[T[:,:1]],-1,-2);inv=np.linalg.inv(Dm);V=np.linalg.det(Dm)/6
 gradients=np.concatenate([-inv.sum(1)[:,None],inv],axis=1);F=np.swapaxes(q[T[:,1:]]-q[T[:,:1]],-1,-2)@inv;J=np.linalg.det(F)
 if np.any(J<=0):raise ValueError('PositiveJ references only')
 cof=J[:,None,None]*np.linalg.inv(F).transpose(0,2,1);P=mu*F+(kappa*(J-1)-mu)[:,None,None]*cof
 local=-V[:,None,None]*np.einsum('eiJ,eaJ->eai',P,gradients)
 cg=np.einsum('eiJ,eaJ->eai',cof,gradients)
 H=V[:,None,None,None]*(mu*np.sum(gradients**2,axis=2)[:,:,None,None]*np.eye(3)[None,None]+kappa*np.einsum('eai,eaj->eaij',cg,cg))
 W=.5*mu*(np.sum(F**2,axis=(1,2))-3)-mu*(J-1)+.5*kappa*(J-1)**2
 return dict(local_force_N=local,self_H_N_m=H,energy_J=float(V@W),J=J,poses=inv)

def block_inputs(fixed=True):
 X,T,F=block_mesh((.24,.08,.08),(4,2,2),(.12,0,0));mass=lumped_masses(X,T,625.)
 ids=np.flatnonzero(abs(X[:,0])<1e-13) if fixed else np.array([],dtype=int)
 weights=np.zeros(len(X));right=F[np.all(abs(X[F,0]-.24)<1e-13,axis=1)]
 for face in right:
  area=.5*np.linalg.norm(np.cross(X[face[1]]-X[face[0]],X[face[2]]-X[face[0]]));weights[face]+=area/3
 weights/=.08*.08;force=np.zeros_like(X);force[:,0]=weights
 return X,T,F,mass,ids,force

def prepare():
 p=ROOT/PROTOCOL
 if p.exists():raise FileExistsError('Preserve frozen probe')
 d=dict(schema='newton-VBD-preliminary-probe-v0.1',status='FROZEN_BEFORE_GPU_CALLS',CPU_threads=2,GPU=0,CUDA_VISIBLE_DEVICES='0',hard_cap_s=600,
 source_sha256={p:sha(ROOT/p) for p in SOURCES},external_source_sha256={p:sha(SITE/p) for p in EXTERNAL},
 requested_versions=dict(newton='1.7.0',warp='1.18.0'),materials=dict(E_Pa=5000.,nu=.35,rho_kg_m3=625.,damping=0.,builder_mu_lambda='StandardLame;kernelinternallykappa=lambda+mu'),
 kernel_cases=['rest','finiteaffinediagonal','finiteshear','properrotation'],kernel_force_absolute_tolerance_N=2e-6,kernel_H_absolute_tolerance_N_m=3e-4,kernel_relative_tolerance=5e-5,
 model=dict(size_m=[.24,.08,.08],cells=[4,2,2],tet_count=96,particle_mass='ExactreferenceP1tetvolumelumping;positiveevenatfixednodes',fixed='clearACTIVEflagretainsphysicalmass',surface_skin_stiffness=0.,particle_radius=0.),
 tests=dict(freefall=dict(steps=12,hz=240,iterations=16,gravity_m_s2=[0,0,-9.81],maximum_velocity_error_m_s=2e-4,maximum_translation_nonuniformity_m=2e-5),
 fixed_one_step=dict(steps=1,hz=240,iterations=16,gravity_m_s2=[0,0,0],right_reference_traction_total_N=1.,implicit_residual='Measureonly;noiteration/time/spatialqualificationfromthissmoke')),
 qualifications=['StableSmithNeoHookeanlawNOTCPUlogJlaw;smallstrainE/numatchonly.','InstalledkernelpervertexHcomparisonisnotafullglobalHapproximation.','PreliminaryAPI/force/constitutivecontrolonly,noconvergence/contact/staticqualification.'])
 write(p,d);return dict(path=PROTOCOL,sha256=sha(p))

def validate(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed probe protocol')
 d=json.loads((ROOT/path).read_text())
 if d['source_sha256']!={p:sha(ROOT/p) for p in SOURCES} or d['external_source_sha256']!={p:sha(SITE/p) for p in EXTERNAL}:raise ValueError('Changed source/dependency')
 if d['GPU']!=0 or d['hard_cap_s']!=600 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Frozen device/budget required')
 return d

def run(path,digest):
 d=validate(path,digest);out=ROOT/OUTPUT
 if out.exists():raise FileExistsError('Fresh probe output required')
 out.mkdir(parents=True);start=time.monotonic();state=dict(completed=False,passed=False,protocol=dict(path=path,sha256=digest),pid=os.getpid(),qualification='PRELIMINARY API/constitutivecontrol; no convergenceclaim')
 signal.signal(signal.SIGALRM,lambda a,b:(_ for _ in ()).throw(TimeoutError('600sprobe budget')));signal.alarm(600)
 try:
  import importlib.metadata
  import warp as wp
  wp.config.kernel_cache_dir=str(out/'warp-cache');wp.init();device=wp.get_device('cuda:0')
  import newton
  from experiments.campaign_newton_vbd_kernels import tet_vertex_probe
  state['runtime']=dict(versions={p:importlib.metadata.version(p) for p in ['newton','warp-lang','numpy','scipy']},device=str(device),device_name=device.name,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],external_source_sha256=d['external_source_sha256'])
  if state['runtime']['versions']['newton']!='1.7.0' or state['runtime']['versions']['warp-lang']!='1.18.0':raise ValueError('Wrong runtime versions')
  X=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1],[.1,.1,.1]],dtype=float);T=np.array([[0,1,2,3],[4,2,1,3]],dtype=np.int32)
  angle=.4;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
  Fs=[np.eye(3),np.diag([1.02,.99,.995]),np.array([[1.,.025,-.01],[0,1,.01],[0,0,1.]]),Q];mu,lam=material();poses=element_reference(X,T,X)['poses'];rows=[]
  for name,A in zip(d['kernel_cases'],Fs):
   q=X@A.T;ref=element_reference(X,T,q);f=wp.zeros((2,4),dtype=wp.vec3,device=device);H=wp.zeros((2,4),dtype=wp.mat33,device=device)
   wp.launch(tet_vertex_probe,dim=8,inputs=[wp.array(q,dtype=wp.vec3,device=device),wp.array(T,dtype=wp.int32,ndim=2,device=device),wp.array(poses,dtype=wp.mat33,device=device),mu,lam],outputs=[f,H],device=device);wp.synchronize_device(device)
   actual=f.numpy();actualH=H.numpy();fe=float(np.max(abs(actual-ref['local_force_N'])));he=float(np.max(abs(actualH-ref['self_H_N_m'])));ftol=2e-6+5e-5*float(np.max(abs(ref['local_force_N'])));htol=3e-4+5e-5*float(np.max(abs(ref['self_H_N_m'])))
   rows.append(dict(case=name,force_max_error_N=fe,H_max_error_N_m=he,force_tolerance_N=ftol,H_tolerance_N_m=htol,passed=fe<=ftol and he<=htol))
  state['kernel_parity']=rows;write(out/'summary.json',state)
  if not all(r['passed'] for r in rows):raise RuntimeError('InstalledVBDkernelparityfailed')
  def build(fixed,gravity):
   X,T,F,mass,ids,force=block_inputs(fixed);builder=newton.ModelBuilder(gravity=tuple(gravity))
   for i,(x,m) in enumerate(zip(X,mass)):builder.add_particle(tuple(x),(0.,0.,0.),float(m),radius=0.,flags=0 if i in ids else int(newton.ParticleFlags.ACTIVE))
   for tet in T:builder.add_tetrahedron(*map(int,tet),k_mu=mu,k_lambda=lam,k_damp=0.)
   builder.color();model=builder.finalize(device=device)
   np.testing.assert_allclose(model.particle_mass.numpy(),mass,rtol=1e-7,atol=1e-12);np.testing.assert_array_equal(model.particle_q.numpy(),X.astype(np.float32));np.testing.assert_array_equal(model.tet_indices.numpy(),T)
   np.testing.assert_array_equal(model.tet_materials.numpy(),np.broadcast_to(np.array([mu,lam,0],np.float32),(len(T),3)))
   flags=model.particle_flags.numpy();np.testing.assert_array_equal(flags[ids],0)
   if not np.all(model.particle_inv_mass.numpy()>0):raise RuntimeError('Physicalmass/invmassunexpectedlylost')
   solver=newton.solvers.SolverVBD(model,iterations=16,particle_enable_self_contact=False,particle_enable_tile_solve=False)
   return model,solver,X,T,F,mass,ids,force
  state['model_controls']=[]
  for fixed,steps,gravity in [(False,12,(0,0,-9.81)),(True,1,(0,0,0))]:
   model,solver,X,T,F,mass,ids,force=build(fixed,gravity);a=model.state();b=model.state();control=model.control();h=1/240;data=[]
   for k in range(steps):
    qold=a.particle_q.numpy().astype(float);vold=a.particle_qd.numpy().astype(float);a.clear_forces()
    if fixed:a.particle_f.assign(wp.array(force,dtype=wp.vec3,device=device))
    solver.step(a,b,control,None,h);wp.synchronize_device(device);q=b.particle_q.numpy().astype(float);v=b.particle_qd.numpy().astype(float)
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise RuntimeError('NonfiniteVBDstate')
    ref=element_reference(X,T,q);internal=np.zeros_like(q);np.add.at(internal,T,ref['local_force_N']);external=force if fixed else mass[:,None]*gravity
    residual=mass[:,None]*(v-vold)/h-internal-external;active=model.particle_flags.numpy()!=0
    if len(ids) and not np.array_equal(q[ids],X[ids].astype(np.float32).astype(float)):raise RuntimeError('Inactivefixedparticlemoved')
    data.append(dict(step=k+1,minimum_J=float(ref['J'].min()),maximum_velocity_error_vs_gravity_m_s=float(np.max(abs(v-(k+1)*h*np.asarray(gravity)))) if not fixed else None,original_implicit_residual_N=float(np.linalg.norm(residual[active]))))
    a,b=b,a
   displacement=q-X.astype(np.float32).astype(float);center=mass@displacement/mass.sum();spread=float(np.sqrt(np.mean((displacement-center)**2)));v_error=float(np.max(abs(v-steps*h*np.asarray(gravity)))) if not fixed else None
   row=dict(kind='fixed_one_step' if fixed else'freefall',steps=steps,physicalmass_sum_kg=float(model.particle_mass.numpy().sum()),fixed_nodes=len(ids),material_first_row=model.tet_materials.numpy()[0].tolist(),kernel_kappa_Pa=float(mu+lam),applied_force_sum_N=force.sum(0).tolist() if fixed else[0,0,0],translation_nonuniformity_coordinate_RMS_m=spread,velocity_max_error_m_s=v_error,ledger=data,passed=True)
   if not fixed:row['passed']=v_error<=2e-4 and spread<=2e-5
   np.savez_compressed(out/(row['kind']+'.npz'),rest_m=X,tets=T,faces=F,positions_m=q,velocities_m_s=v,physical_masses_kg=mass,fixed_node_ids=ids,force_N=force if fixed else np.zeros_like(force));state['model_controls'].append(row);write(out/'summary.json',state)
   if not row['passed']:raise RuntimeError('Freefall controlfailed')
  gpuinfo=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;gpu0=gpuinfo.splitlines()[0].split(',')[1].strip();apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
  if not own or any(r.split(',')[1].strip()!=gpu0 for r in own):raise RuntimeError('WrongGPUcomputeassignment')
  state.update(completed=True,passed=True,own_GPU_compute_records=own)
 except Exception as e:state.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:signal.alarm(0);state['elapsed_s']=time.monotonic()-start;write(out/'summary.json',state)
 return state

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol',default=PROTOCOL);p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r))
 if a.action=='run':raise SystemExit(0 if r['completed'] and r['passed'] else 1)
