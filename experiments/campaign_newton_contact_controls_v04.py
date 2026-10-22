"""Canonical stock-BE Newton particle/plane controls; no momentum/history repair.

The independent oracle solves m(z-y)/h²=k max(radius-z,0) exactly.
Contact stiffness is explicitly fixed, friction/damping zero. This validates the
specified penalty model, not hard contact or deformable surface convergence.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'runs/operator-dev/newton-runtime-v01/env'
SITE=ENV/'lib/python3.12/site-packages'
STAGE='newton-contact-controls-v04'
OP=ROOT/'runs/operator-dev'/STAGE
OUT=ROOT/'runs/gaussian-campaign-v01'/STAGE
GPU='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb'
SOURCES=['experiments/campaign_newton_contact_controls_v04.py','experiments/run_newton_contact_controls_v04.py','tests/test_campaign_newton_contact_controls_v04.py','experiments/campaign_newton_surface_contact.py']
EXTERNAL=['newton/_src/solvers/vbd/solver_vbd.py','newton/_src/solvers/vbd/rigid_vbd_kernels.py','newton/_src/solvers/vbd/particle_vbd_kernels.py','newton/_src/sim/builder.py']
JOBS=[dict(name='equilibrium960',hz=960,floor=True,z=.001-.01*9.81/10000,vx=0.,duration=.1),dict(name='area_equivalent960',hz=960,floor=True,z=.001-.01*9.81/10000,vx=0.,duration=.1)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def oracle(z,v,h,mass=.01,k=10000.,radius=.001,gravity=-9.81,floor=True):
 y=z+h*v+h*h*gravity
 q=y if not floor or y>=radius else (mass*y+h*h*k*radius)/(mass+h*h*k)
 return q,(q-z)/h

def metrics(q,v,refq,refv,mass,k,radius,h,floor):
 q=np.asarray(q);v=np.asarray(v)
 force=k*np.maximum(radius-q[:,2],0) if floor else np.zeros(len(q))
 impulse=mass*(v[-1]-v[0])-np.array([0,0,-mass*9.81*h*(len(q)-1)])
 predicted=np.array([0,0,h*force[1:].sum()])
 energy=.5*mass*(v*v).sum(1)+mass*9.81*q[:,2]+(.5*k*np.maximum(radius-q[:,2],0)**2 if floor else 0)
 return dict(maximum_position_oracle_error_m=float(abs(q-refq).max()),maximum_velocity_oracle_error_m_s=float(abs(v-refv).max()),maximum_radius_corrected_penetration_m=float(np.maximum(radius-q[:,2],0).max()) if floor else 0.,contact_impulse_Ns=predicted.tolist(),momentum_contact_impulse_Ns=impulse.tolist(),impulse_balance_error_Ns=float(np.linalg.norm(impulse-predicted)),maximum_energy_increase_J=float(np.maximum(np.diff(energy),0).max(initial=0)),initial_energy_J=float(energy[0]),final_energy_J=float(energy[-1]),maximum_tangential_velocity_error_m_s=float(abs(v[:,0]-v[0,0]).max()))

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen declaration exists')
 p=dict(schema='Newton-stock-BE-canonical-contact-v0.1',status='FROZEN_BEFORE_GPU_CALLS',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=600,jobs=JOBS,source_sha256={x:sha(ROOT/x) for x in SOURCES},external_source_sha256={x:sha(SITE/x) for x in EXTERNAL},physics=dict(mass_kg=.01,radius_m=.001,plane_height_m=0.,plane_margin_m=0.,gravity_m_s2=[0,0,-9.81],contact_stiffness_N_m=10000.,contact_damping_Ns_m=0.,friction=0.,soft_contact_gap_m=.02,iterations=32,rigid_contact_k_start=10000.,rigid_avbd_beta=0.,rigid_compliant_alm=False,rigid_soft_enable_dat=False),gates=dict(position_oracle_error_m=20e-6,velocity_oracle_error_m_s=5e-4,impulse_balance_Ns=2e-5,maximum_energy_increase_J=1e-7,tangential_velocity_error_m_s=5e-5,equilibrium_force_error_N=5e-6),prior_failures_retained=['v01constructor rejects documented negative contactk seed','v02freefall960 fails1um gate11.524um; v03prospective20um project physical-field target, not v02relabel'],nonclaims=['Specified compliant quadratic contact only, not exact nonpenetration','Single particle controls do not qualify block surface/mesh/time accuracy','Stock physical-time BE; no contact-free momentum or accepted-iterate wrappers'])
 write(OP/'protocol.json',p);return dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=sha(OP/'protocol.json'))

def validate(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed protocol')
 p=json.loads((ROOT/path).read_text())
 for key,base in [('source_sha256',ROOT),('external_source_sha256',SITE)]:
  if any(sha(base/f)!=v for f,v in p[key].items()):raise ValueError('Changed pinned source')
 if p['jobs']!=JOBS or p['gpu_uuid']!=GPU or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Scope or GPU changed')
 return p

def run(path,digest):
 p=validate(path,digest);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 result=dict(protocol=dict(path=path,sha256=digest),completed=False,passed=False,pid=os.getpid(),cases=[],nonclaims=p['nonclaims'])
 try:
  import warp as wp,newton
  versions={k:importlib.metadata.version(k) for k in ['newton','warp-lang','numpy']}
  if versions['newton']!='1.7.0' or versions['warp-lang']!='1.18.0':raise ValueError('Wrong isolated SDK')
  wp.config.kernel_cache_dir=str(OUT/'warp-cache');wp.init();device=wp.get_device('cuda:0');probe=wp.zeros(1,device=device);wp.synchronize_device(device)
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout
  own=[x for x in apps.splitlines() if x.split(',')[0].strip()==str(os.getpid())]
  result.update(versions=versions,device=device.name,own_compute_records=own);write(OUT/'summary.json',result)
  if not own or any(x.split(',')[1].strip()!=GPU for x in own):raise ValueError('Wrong own physical GPU')
  for job in JOBS:
   h=1/job['hz'];steps=round(job['duration']*job['hz']);builder=newton.ModelBuilder(gravity=(0.,0.,-9.81))
   builder.add_particle((0.,0.,job['z']),(job['vx'],0.,0.),.01,radius=.001)
   if job['floor']:
    cfg=newton.ModelBuilder.ShapeConfig();cfg.ke=10000.;cfg.kd=0.;cfg.mu=0.;cfg.margin=0.;cfg.has_particle_collision=True
    builder.add_ground_plane(height=0.,cfg=cfg)
   builder.color();model=builder.finalize(device=device);model.soft_contact_ke=10000.;model.soft_contact_kd=0.;model.soft_contact_mu=0.
   np.testing.assert_array_equal(model.particle_radius.numpy(),np.float32(.001));np.testing.assert_allclose(model.particle_mass.numpy(),.01,rtol=1e-7)
   if job['floor']:
    np.testing.assert_array_equal(model.shape_margin.numpy(),0.);np.testing.assert_array_equal(model.shape_material_ke.numpy(),10000.);np.testing.assert_array_equal(model.shape_body.numpy(),-1)
   from experiments.campaign_newton_surface_contact import area_contact_solver_class
   solver_type=area_contact_solver_class(newton.solvers.SolverVBD,wp,np.array([.25]),40000.) if job['name'].startswith('area_') else newton.solvers.SolverVBD
   solver=solver_type(model,iterations=32,particle_enable_self_contact=False,particle_enable_tile_solve=False,rigid_compliant_alm=False,rigid_contact_k_start=10000.,rigid_avbd_beta=0.,rigid_soft_enable_dat=False)
   pipeline=newton.CollisionPipeline(model,broad_phase='nxn',soft_contact_gap=.02) if job['floor'] else None
   contacts=pipeline.contacts() if job['floor'] else None
   a=model.state();b=model.state();control=model.control();q=[a.particle_q.numpy()[0].astype(float)];v=[a.particle_qd.numpy()[0].astype(float)];refq=[q[0].copy()];refv=[v[0].copy()];counts=[];readback=[]
   mass=float(model.particle_mass.numpy()[0]);radius=float(model.particle_radius.numpy()[0]);hh=float(np.float32(h))
   for k in range(steps):
    a.clear_forces()
    if pipeline:pipeline.collide(a,contacts)
    solver.step(a,b,control,contacts,h);wp.synchronize_device(device)
    q.append(b.particle_q.numpy()[0].astype(float));v.append(b.particle_qd.numpy()[0].astype(float))
    z,vz=oracle(refq[-1][2],refv[-1][2],hh,mass=mass,k=10000.,radius=radius,floor=job['floor'])
    refq.append(np.array([refq[-1][0]+hh*refv[-1][0],0,z]));refv.append(np.array([refv[-1][0],0,vz]))
    if contacts:
     count=int(contacts.soft_contact_count.numpy()[0]);counts.append(count)
     if count:
      ks=solver.body_particle_contact_penalty_k.numpy()[:count];np.testing.assert_array_equal(ks,10000.);readback.extend(ks.tolist())
      np.testing.assert_allclose(contacts.soft_contact_normal.numpy()[:count],[ [0,0,1] ]*count,atol=1e-7)
    if not np.isfinite(q[-1]).all() or not np.isfinite(v[-1]).all():raise ValueError('Nonfinite canonical state')
    a,b=b,a
   q=np.array(q);v=np.array(v);m=metrics(q,v,np.array(refq),np.array(refv),mass,10000.,radius,hh,job['floor'])
   equilibrium=float(abs(10000*max(radius-q[-1,2],0)-mass*9.81)) if job['name'].startswith('equilibrium') else None
   g=p['gates'];passed=m['maximum_position_oracle_error_m']<=g['position_oracle_error_m'] and m['maximum_velocity_oracle_error_m_s']<=g['velocity_oracle_error_m_s'] and m['impulse_balance_error_Ns']<=g['impulse_balance_Ns'] and m['maximum_energy_increase_J']<=g['maximum_energy_increase_J'] and m['maximum_tangential_velocity_error_m_s']<=g['tangential_velocity_error_m_s'] and (equilibrium is None or equilibrium<=g['equilibrium_force_error_N'])
   file=OUT/(job['name']+'.npz');np.savez_compressed(file,times_s=np.arange(steps+1)*hh,positions_m=q,velocities_m_s=v,oracle_positions_m=np.array(refq),oracle_velocities_m_s=np.array(refv),contact_counts=np.array(counts),mass_kg=mass,radius_m=radius)
   row=dict(**job,completed=True,passed=bool(passed),**m,equilibrium_force_error_N=equilibrium,maximum_contact_count=max(counts,default=0),actual_contact_penalty_values=sorted(set(readback)),capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)))
   result['cases'].append(row);write(OUT/'summary.json',result)
   if not passed:raise RuntimeError('Canonical case failed: '+job['name'])
  result.update(completed=True,passed=True)
 except Exception as e:
  result.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:result['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',result)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol');p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r));raise SystemExit(0 if a.action=='prepare' or r['completed'] else 1)
