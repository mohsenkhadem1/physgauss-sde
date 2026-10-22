"""Stock physical-time BE block/freefall/area-compliant-plane validation.

No position/velocity prediction repair. A reference-area lumped surface penalty
is explicit and mesh independent in units; hard contact is not assumed.
"""
from __future__ import annotations
import argparse,importlib.metadata,json,os,subprocess,time,traceback
import numpy as np
from experiments.campaign_newton_contact_controls_v04 import ROOT,ENV,SITE,GPU,sha,write
from experiments.campaign_newton_surface_contact import surface_areas,penalty,area_contact_solver_class
from experiments.boundary_block import block_mesh,lumped_masses
from experiments.campaign_newton_vbd_probe_v02 import material,actual_input_reference
from experiments.campaign_newton_vbd_static_v02 import assemble,query
STAGE='newton-block-contact-v01';OP=ROOT/'runs/operator-dev'/STAGE;OUT=ROOT/'runs/gaussian-campaign-v01'/STAGE
JOBS=[dict(name='freefall',cells=[4,2,2],E=300000.,hz=480,iterations=128,floor=False,gap=.05,kappa=1e9,duration=.2),dict(name='floor_gravity',cells=[4,2,2],E=300000.,hz=480,iterations=128,floor=True,gap=0.,kappa=1e9,duration=.3)]+[dict(name=n,cells=c,E=E,hz=hz,iterations=it,floor=True,gap=.05,kappa=kap,duration=.3) for n,c,E,hz,it,kap in [('drop300',[4,2,2],300000.,480,128,1e9),('drop300_time',[4,2,2],300000.,960,128,1e9),('drop300_iteration',[4,2,2],300000.,480,512,1e9),('drop600',[4,2,2],600000.,480,128,1e9),('drop300_mesh',[8,4,4],300000.,480,128,1e9),('drop300_penalty',[4,2,2],300000.,480,128,1e10)]]
SOURCES=['experiments/campaign_newton_block_contact.py','experiments/run_newton_block_contact.py','tests/test_campaign_newton_block_contact.py','experiments/campaign_newton_surface_contact.py','experiments/boundary_block.py','experiments/campaign_newton_contact_controls_v04.py','experiments/campaign_newton_vbd_probe_v02.py','experiments/campaign_newton_vbd_static_v02.py','experiments/campaign_newton_vbd_static_kernels.py','experiments/campaign_newton_vbd_kernels.py','experiments/campaign_newton_static_initialization.py']
EXTERNAL=['newton/_src/solvers/vbd/solver_vbd.py','newton/_src/solvers/vbd/particle_vbd_kernels.py','newton/_src/solvers/vbd/rigid_vbd_kernels.py','newton/_src/sim/builder.py']

def geometry(cells,gap):
 X,T,F=block_mesh((.24,.08,.08),tuple(cells),(.12,0,.04+gap));mass=lumped_masses(X,T,625.);A=surface_areas(X,F)
 if abs(mass.sum()-.96)>1e-12 or abs(A.sum()-.0896)>1e-12:raise ValueError('Wrong physical geometry/mass/surface')
 return X,T,F,mass,A

def probes():return np.array([[x,y,z] for x in np.linspace(0,.24,9) for y in [-.04,0,.04] for z in [-.04,0,.04]])

def nonrigid(X,q,mass):
 cx=np.average(X,axis=0,weights=mass);cy=np.average(q,axis=0,weights=mass);U,_,V=np.linalg.svd((X-cx).T@(mass[:,None]*(q-cy)));D=np.eye(3);D[2,2]=np.linalg.det(U@V);R=U@D@V;e=q-((X-cx)@R+cy)
 return float(np.sqrt(np.sum(mass[:,None]*e**2)/(3*mass.sum())))

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen declaration exists')
 controls=ROOT/'runs/gaussian-campaign-v01/newton-contact-controls-v04/summary.json'
 if not json.loads(controls.read_text())['passed']:raise ValueError('Area/equilibrium controls required')
 p=dict(schema='Newton-stock-BE-area-compliant-block-v0.1',status='FROZEN_BEFORE_BLOCK_CALLS',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=600,jobs=JOBS,source_sha256={s:sha(ROOT/s) for s in SOURCES},external_source_sha256={s:sha(SITE/s) for s in EXTERNAL},controls=dict(path=str(controls.relative_to(ROOT)),sha256=sha(controls)),physics=dict(size_m=[.24,.08,.08],density_kg_m3=625.,poisson=.35,gravity_m_s2=[0,0,-9.81],particle_radius_m=0.,plane_height_m=0.,plane_margin_m=0.,friction=0.,contact_damping=0.,material_damping=0.,soft_contact_gap_m=.02,penalty='U=.5*kappa_A*sum(reference_surface_lumped_area_i*max(-z_i,0)^2)',kappa_units='N/m^3',force='stock backward Euler, no momentum/predictor repairs'),gates=dict(minimum_J=.1,maximum_freefall_nonuniform_displacement_m=20e-6,maximum_freefall_COM_BE_error_m=20e-6,maximum_fixed_material_query_difference_m=20e-6,maximum_iteration_query_difference_m=20e-6,maximum_penalty_query_difference_m=20e-6),query_times='common60Hz endpoints through.3s; fixed81materialqueries; allnativeq also saved',nonclaims=['Finitecompliant penalty not rigid nonpenetration','Surfacevertex quadrature is refinement-tested, notexactcontinuouscollision','Physicalvelocity/force/energy are diagnostics, no artificialvelocityrepair','No continuum/dynamicqualification if any mesh/time/iteration/penalty gate fails'])
 write(OP/'protocol.json',p);return dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=sha(OP/'protocol.json'))

def validate(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed protocol')
 p=json.loads((ROOT/path).read_text())
 for key,base in [('source_sha256',ROOT),('external_source_sha256',SITE)]:
  if any(sha(base/f)!=h for f,h in p[key].items()):raise ValueError('Changed dependency')
 if p['jobs']!=JOBS or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or sha(ROOT/p['controls']['path'])!=p['controls']['sha256']:raise ValueError('Changed scope/device/controls')
 return p

def run(path,digest):
 p=validate(path,digest);OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,qualified=False,pid=os.getpid(),protocol=dict(path=path,sha256=digest),cases=[],nonclaims=p['nonclaims'])
 try:
  import warp as wp,newton
  from experiments.campaign_newton_vbd_static_kernels import elastic_local
  versions={k:importlib.metadata.version(k) for k in ['newton','warp-lang','numpy']}
  if versions['newton']!='1.7.0' or versions['warp-lang']!='1.18.0':raise ValueError('Changed runtime')
  wp.config.kernel_cache_dir=str(OUT/'warp-cache');wp.init();device=wp.get_device('cuda:0');allocation=wp.zeros(1,device=device);wp.synchronize_device(device)
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
  s.update(versions=versions,own_compute_records=own);write(OUT/'summary.json',s)
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('Wrong physical GPU')
  fields={}
  for job in JOBS:
   started=time.monotonic();X,T,F,mass,A=geometry(job['cells'],job['gap']);mu,lam=material(job['E']);builder=newton.ModelBuilder(gravity=(0.,0.,-9.81))
   for x,m in zip(X,mass):builder.add_particle(tuple(x),(0.,0.,0.),float(m),radius=0.)
   for tet in T:builder.add_tetrahedron(*map(int,tet),k_mu=mu,k_lambda=lam,k_damp=0.)
   if job['floor']:
    cfg=newton.ModelBuilder.ShapeConfig();cfg.ke=1e10;cfg.kd=0.;cfg.mu=0.;cfg.margin=0.;cfg.has_particle_collision=True;builder.add_ground_plane(height=0.,cfg=cfg)
   builder.color();model=builder.finalize(device=device);model.soft_contact_ke=1e10;model.soft_contact_kd=0.;model.soft_contact_mu=0.
   np.testing.assert_allclose(model.particle_mass.numpy(),mass,rtol=1e-7);np.testing.assert_array_equal(model.particle_radius.numpy(),0.);np.testing.assert_array_equal(model.particle_flags.numpy(),1)
   np.testing.assert_array_equal(model.tet_indices.numpy(),T);np.testing.assert_array_equal(model.tet_materials.numpy(),np.broadcast_to(np.array([mu,lam,0],np.float32),(len(T),3)))
   if job['floor']:
    np.testing.assert_array_equal(model.shape_margin.numpy(),0.);np.testing.assert_array_equal(model.shape_body.numpy(),-1)
   solver_type=area_contact_solver_class(newton.solvers.SolverVBD,wp,A,job['kappa']) if job['floor'] else newton.solvers.SolverVBD
   solver=solver_type(model,iterations=job['iterations'],particle_enable_self_contact=False,particle_enable_tile_solve=False,rigid_compliant_alm=False,rigid_contact_k_start=1e10,rigid_avbd_beta=0.,rigid_soft_enable_dat=False)
   pipeline=newton.CollisionPipeline(model,broad_phase='nxn',soft_contact_gap=.02) if job['floor'] else None;contacts=pipeline.contacts() if pipeline else None
   a=model.state();b=model.state();control=model.control();q0=a.particle_q.numpy().astype(float);positions=[q0.copy()];velocities=[a.particle_qd.numpy().astype(float)];ledger=[]
   poses=model.tet_poses.numpy();mats=model.tet_materials.numpy();native_mass=model.particle_mass.numpy().astype(float);local=wp.zeros((len(T),4),dtype=wp.vec3,device=device);h=1/job['hz'];hh=float(np.float32(h));steps=round(job['duration']*job['hz']);initialP=np.zeros(3);contactimp=np.zeros(3);internalimp=np.zeros(3)
   for k in range(steps):
    a.clear_forces()
    if pipeline:pipeline.collide(a,contacts)
    solver.step(a,b,control,contacts,h);wp.synchronize_device(device);q=b.particle_q.numpy().astype(float);v=b.particle_qd.numpy().astype(float)
    ref=actual_input_reference(q,T,poses,float(mats[0,0]),float(mats[0,1]));J=ref['J']
    if not np.isfinite(q).all() or not np.isfinite(v).all() or J.min()<=.1:
     np.savez_compressed(OUT/(job['name']+'-failed.npz'),positions_m=q,velocities_m_s=v,J=J,step=k+1);raise ValueError('Nonfinite/inverted nativeblock')
    wp.launch(elastic_local,dim=len(T)*4,inputs=[b.particle_q,model.tet_indices,model.tet_poses,model.tet_materials],outputs=[local],device=device);wp.synchronize_device(device);internal=assemble(local.numpy(),T,len(X))
    contactU,normalforce=penalty(q[:,2],A,job['kappa']) if pipeline else (0.,np.zeros(len(X)));fcontact=np.zeros_like(q);fcontact[:,2]=normalforce;contactimp+=hh*fcontact.sum(0);internalimp+=hh*internal.sum(0)
    gravity=native_mass[:,None]*np.array([0,0,-9.81]);R=native_mass[:,None]*(q-solver.inertia.numpy().astype(float))/hh**2-internal-fcontact
    t=(k+1)*hh;momentum=np.sum(native_mass[:,None]*v,axis=0);balance=momentum-initialP-native_mass.sum()*np.array([0,0,-9.81])*t-contactimp-internalimp
    # Native restpose energy evaluated at actual float32 q/poses/materials.
    inv=np.asarray(poses,float);Ft=np.swapaxes(q[T[:,1:]]-q[T[:,:1]],-1,-2)@inv;vol=1/(6*np.linalg.det(inv));W=.5*float(mats[0,0])*(np.sum(Ft**2,axis=(1,2))-3)-float(mats[0,0])*(J-1)+.5*float(mats[0,0]+mats[0,1])*(J-1)**2
    energy=float(vol@W+.5*np.sum(native_mass[:,None]*v*v)+np.sum(native_mass*9.81*q[:,2])+contactU)
    ledger.append(dict(step=k+1,time_s=t,minimum_J=float(J.min()),maximum_stretch_departure=float(abs(np.linalg.svd(Ft,compute_uv=False)-1).max()),penetration_m=float(max(0,-q[:,2].min())) if pipeline else 0.,contact_count=int(contacts.soft_contact_count.numpy()[0]) if contacts else 0,contact_force_N=fcontact.sum(0).tolist(),literal_BE_force_residual_N=float(np.linalg.norm(R)),total_mechanical_energy_J=energy,momentum_impulse_balance_Ns=float(np.linalg.norm(balance)),nonrigid_coordinate_RMS_m=nonrigid(X,q,mass)))
    positions.append(q.copy());velocities.append(v.copy());a,b=b,a
   positions=np.array(positions);velocities=np.array(velocities);sample=np.arange(0,steps+1,job['hz']//60);probe=probes()+np.array([0,0,.04+job['gap']]);pred=np.array([query(X,T,x,probe) for x in positions[sample]])
   file=OUT/(job['name']+'.npz');np.savez_compressed(file,times_s=np.arange(steps+1)*hh,positions_m=positions,velocities_m_s=velocities,rest_m=X,tets=T,faces=F,fixed_node_ids=np.array([],int),floor_z_m=0. if pipeline else np.nan,reference_surface_areas_m2=A,physical_masses_kg=mass,readback_masses_kg=native_mass,readback_inverse_mass=model.particle_inv_mass.numpy(),readback_poses=poses,readback_materials=mats,query_positions_m=pred,query_times_s=sample/job['hz'],material_probes_m=probe)
   row=dict(**job,completed=True,capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)),nodes=len(X),tets=len(T),minimum_J=min(r['minimum_J'] for r in ledger),maximum_penetration_m=max(r['penetration_m'] for r in ledger),maximum_stretch_departure=max(r['maximum_stretch_departure'] for r in ledger),maximum_nonrigid_coordinate_RMS_m=max(r['nonrigid_coordinate_RMS_m'] for r in ledger),maximum_literal_BE_force_residual_N=max(r['literal_BE_force_residual_N'] for r in ledger),maximum_momentum_impulse_balance_Ns=max(r['momentum_impulse_balance_Ns'] for r in ledger),maximum_energy_increase_J=float(np.maximum(np.diff([r['total_mechanical_energy_J'] for r in ledger]),0).max(initial=0)),elapsed_s=time.monotonic()-started,ledger=ledger)
   if not pipeline:
    times=np.arange(steps+1)*hh;expected=-9.81*hh**2*np.arange(steps+1)*(np.arange(steps+1)+1)/2;delta=positions-q0;com=np.einsum('n,tni->ti',native_mass,delta)/native_mass.sum();nonuniform=np.max(np.linalg.norm(delta-com[:,None],axis=2));error=np.max(abs(com[:,2]-expected));row.update(freefall_COM_BE_error_m=float(error),freefall_nonuniform_displacement_m=float(nonuniform),freefall_passed=bool(max(error,nonuniform)<=20e-6))
   s['cases'].append(row);fields[job['name']]=pred;write(OUT/(job['name']+'.json'),row);write(OUT/'summary.json',s)
  checks=[]
  for other,role in [('drop300_time','timestep480vs960'),('drop300_iteration','iterations128vs512'),('drop300_mesh','mesh96vs768'),('drop300_penalty','kappa1e9vs1e10')]:
   diff=float(np.linalg.norm(fields[other]-fields['drop300'],axis=2).max());checks.append(dict(role=role,maximum_fixedquery_difference_m=diff,tolerance_m=20e-6,passed=diff<=20e-6))
  s.update(completed=True,qualified=all(r['passed'] for r in checks) and s['cases'][0]['freefall_passed'],refinement_checks=checks)
 except Exception as e:s.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol');p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r));raise SystemExit(0 if a.action=='prepare' or r['completed'] else 1)
