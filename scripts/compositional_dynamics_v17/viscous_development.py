"""Fixed-viscosity exploratory drop/striker; paired timestep diagnostics, no old-result rescue."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,multiprocessing,argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet,SphereBody
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_dynamics_v17.viscous import ViscousMidpoint
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,AttachmentSpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
PLAN=REPORT/'viscous-development-plan.json';RAW=ROOT/'runs/compositional-dynamics-v17/viscous-development'
def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve existing plan/data')
 cases=[]
 for dt in [25e-6,12.5e-6]:
  cases += [dict(name=f'drop-40mm-dt{dt:.8f}',family='drop',height_m=.04,duration_s=.75,dt_s=dt),dict(name=f'finite-striker-dt{dt:.8f}',family='striker',duration_s=.20,sphere_radius_m=.025,sphere_mass_kg=.08,sphere_center_m=[.06,0,.08],sphere_velocity_m_s=[0,0,-1],dt_s=dt)]
 paths=closure([Path(__file__)])+[ROOT/'scripts/compositional_dynamics_v17/test_viscous.py'];inputs=[REPORT/'viscous-development-design.txt']
 write(PLAN,dict(scope='EXPLORATORY development, not qualified truth. No learned-law improvement claim. All declared failures retained.',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs},cases=cases,workers=2,seconds=3600,n=3,dt_s=25e-6,volume_order=3,surface_order=8,sample_interval_s=.001,dimensions_m=[.12,.06,.06],young_Pa=12000,poisson=.35,density_kg_m3=625,gravity_m_s2=9.81,penalty_N_m3=4e7,minimum_available_memory_GiB=24,diagnostic_gates=dict(minimum_J=.1,energy_relative=.01,maximum_penetration_m=.0012),sampling='Actual integer steps only; no temporal interpolation',physical_damping='objective internal KelvinVoigt eta20/zeta40 Pa*s; synthetic prescribed properties',eta_Pa_s=20.,zeta_Pa_s=40.,integrator='explicit midpoint; dissipated energy midpoint integral',restitution='none'))
def verify(p):
 for group in ['source_sha256','input_sha256']:
  if any(sha(ROOT/f)!=h for f,h in p[group].items()):raise ValueError('Pinned source/input changed')
def build(p,c):
 family=c['family'];colliders=[];bodies=[];attachments=tuple(AttachmentSpec(6000,(.5,.5),(.375,.375))for _ in range(2))
 if family=='drop':colliders=[RigidPlane(point=(0,0,-.03-c['height_m']),penalty_N_m3=p['penalty_N_m3'],name='floor')]
 if family=='striker':
  colliders=[RigidSphere(center=c['sphere_center_m'],radius=c['sphere_radius_m'],penalty_N_m3=p['penalty_N_m3'],name='striker')];bodies=[SphereBody('striker',c['sphere_mass_kg'],np.array(c['sphere_center_m']),np.array(c['sphere_velocity_m_s']),np.zeros(3))]
 m=DynamicContactModel(GeometrySpec(dimensions=tuple(p['dimensions_m'])),n=p['n'],order=p['volume_order'],support_mode='bilateral'if family=='striker'else'free',attachments=attachments,bulk_energy=StableNeoHookean(p['young_Pa'],p['poisson']),density=p['density_kg_m3'],gravity=(0,0,-p['gravity_m_s2'])if family=='drop'else(0,0,0),colliders=tuple(colliders),surface_order=p['surface_order']);return m,ViscousMidpoint(m,bodies,eta=p['eta_Pa_s'],zeta=p['zeta_Pa_s']),colliders

def worker(args):
 c,deadline=args;p=json.loads(PLAN.read_text());verify(p);p['dt_s']=c['dt_s'];torch.set_num_threads(1);folder=RAW/c['name'];folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();row=dict(case=c,name=c['name'],plan_sha256=sha(PLAN),completed=False,scope=p['scope']);ledger=[];saved={k:[]for k in ['times_s','u','velocities','surface_frames','sphere_centers_m','sphere_velocities_m_s','sphere_omega_rad_s','tool_wrenches_N_Nm','saved_dense_penetration_m']}
 try:
  m,e,initial_colliders=build(p,c);u=np.zeros_like(m.rest)
  if c['family']=='release':u[:,2]=-c['compression']*m.rest[:,2]
  s=e.initial_state(u=u);E0=e.energy(s);scale=max(abs(E0),m.total_mass*p['gravity_m_s2']*c.get('height_m',0),1e-10);q=face_grid(17);dense=face_grid(65);X=m.geometry.map(q);Xd=m.geometry.map(dense);names=list(s.spheres);stride=round(p['sample_interval_s']/p['dt_s']);steps=round(c['duration_s']/p['dt_s'])
  if not np.isclose(stride*p['dt_s'],p['sample_interval_s']):raise ValueError('Noninteger sample stride')
  for step in range(steps+1):
   if step:
    if time.monotonic()>=deadline:raise TimeoutError('Whole exploratory stage budget')
    s=e.step(s,p['dt_s'])
   cf=sum((r['force_on_rigid_N']for r in s.diagnostics['contacts']),np.zeros(3));ct=sum((r['torque_on_rigid_Nm']for r in s.diagnostics['contacts']),np.zeros(3));kin=e.energy(s)-s.potential
   ledger.append([step*p['dt_s'],e.energy(s),kin,s.diagnostics['elastic_J'],s.diagnostics['contact_J'],s.diagnostics['bilateral_J'],-np.sum(m.load*s.u),s.diagnostics['minimum_J'],s.diagnostics['maximum_sampled_penetration_m'],*cf,*ct,*e.momentum(s),*e.angular_momentum(s),s.diagnostics['viscous_power_W'],e.dissipated_J,e.energy(s)+e.dissipated_J])
   if step%stride==0:
    positions=Xd+m.sample_material(s.u,dense);depth=0.
    for collider in m.colliders:
     gap=collider.signed_distance(torch.as_tensor(positions,dtype=torch.float64));depth=max(depth,float(torch.clamp_min(-gap,0).max()))
    tool=[]
    if c['family']=='striker':
     for matrix,target,origin in zip(m.anchor_matrices,m.anchor_targets,m.tool_centers):
      nodal=matrix@(s.u-target);tool.append(np.r_[nodal.sum(0),np.cross(m.rest+s.u-origin,nodal).sum(0)])
    vals=dict(times_s=step*p['dt_s'],u=s.u.copy(),velocities=s.v.copy(),surface_frames=(X+m.sample_material(s.u,q)).reshape(6,17,17,3),sphere_centers_m=np.array([s.spheres[n]['center']for n in names]).reshape(-1,3),sphere_velocities_m_s=np.array([s.spheres[n]['velocity']for n in names]).reshape(-1,3),sphere_omega_rad_s=np.array([s.spheres[n]['omega']for n in names]).reshape(-1,3),tool_wrenches_N_Nm=np.array(tool).reshape(-1,6),saved_dense_penetration_m=depth)
    for k,v in vals.items():saved[k].append(v)
  row['completed']=True
 except Exception as ex:row['error']=f'{type(ex).__name__}: {ex}'
 if ledger:
  arrays={k:np.asarray(v)for k,v in saved.items()};arrays.update(ledger=np.array(ledger),surface_rest=X.reshape(6,17,17,3),rest_nodes=m.rest,material_coordinates=m.material_coordinates,mass_row_sums=np.asarray(e.M.sum(1)).ravel());np.savez_compressed(folder/'trajectory.npz',**arrays)
  row.update(trajectory_sha256=sha(folder/'trajectory.npz'),saved_frames=len(saved['times_s']),steps_completed=len(ledger)-1,initial_energy_J=E0,energy_scale_J=scale,maximum_relative_energy_balance_error=float(np.max(abs(np.array(ledger)[:,23]-E0))/scale),maximum_relative_mechanical_energy_change=float(np.max(abs(np.array(ledger)[:,1]-E0))/scale),dissipated_energy_J=float(e.dissipated_J),minimum_J=float(np.min(np.array(ledger)[:,7])),maximum_dense_penetration_m=max(saved['saved_dense_penetration_m']),sphere_names=names,radii_m=[e.radii[n]for n in names],masses_kg=[e.bodies[n].mass for n in names],colliders=[dict(type='sphere'if isinstance(co,RigidSphere)else'plane',**asdict(co))for co in initial_colliders],support_specs=[dict(**asdict(a),face_axis=0,face_side=i,face_origin_m=m.tool_centers[i].tolist(),face_dimensions_m=p['dimensions_m'][1:],footprint='Normalized Gaussian times4y(1-y)4z(1-z), nonzero full face interior; not hard rectangle',tool_pose_m_rad=[0]*6)for i,a in enumerate(m.attachments)]if c['family']=='striker'else[],mass='consistentQ2',physical_damping=dict(eta_Pa_s=p['eta_Pa_s'],zeta_Pa_s=p['zeta_Pa_s'],law='objective spatial KelvinVoigt'),integrator=p['integrator'],fixed_tool_work_J=0.,ledger_columns=['time_s','total_energy_J','kinetic_J','elastic_J','contact_J','bilateral_J','gravity_relative_J','minimum_J','sampled_penetration_m','contact_force_x','contact_force_y','contact_force_z','contact_torque_x','contact_torque_y','contact_torque_z','momentum_x','momentum_y','momentum_z','angular_momentum_x','angular_momentum_y','angular_momentum_z','viscous_power_W','dissipated_energy_J','energy_plus_dissipation_J'])
  g=p['diagnostic_gates'];row['diagnostic_gates_pass']=row['completed']and row['minimum_J']>g['minimum_J']and row['maximum_relative_energy_balance_error']<=g['energy_relative']and row['maximum_dense_penetration_m']<=g['maximum_penetration_m'];row['reference_qualified']=False
 row.update(seconds=time.monotonic()-start,path=str(folder.relative_to(ROOT)));write(folder/'complete.json',row);return row

def main():
 p=json.loads(PLAN.read_text());verify(p)
 if RAW.exists():raise ValueError('Preserve existing output')
 available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
 if available<p['minimum_available_memory_GiB']*2**30:raise RuntimeError('Memory headroom unavailable')
 RAW.mkdir(parents=True);start=time.monotonic();deadline=start+p['seconds']-30;report=dict(plan_sha256=sha(PLAN),expected_keys=[c['name']for c in p['cases']],rows=[],terminal=False,memory_available_bytes=available);out=REPORT/'viscous-development-results.json';pool=multiprocessing.get_context('spawn').Pool(p['workers'])
 try:
  it=pool.imap_unordered(worker,[(c,deadline-15)for c in p['cases']])
  for c in p['cases']:report['rows'].append(it.next(timeout=max(.01,deadline-time.monotonic())));write(out,report)
 except Exception as ex:report['error']=f'{type(ex).__name__}: {ex}'
 finally:
  pool.terminate();pool.join();rows=[]
  for c in p['cases']:
   f=RAW/c['name']/'complete.json'
   if f.exists():
    r=json.loads(f.read_text())
    if r['plan_sha256']!=sha(PLAN)or r['case']!=c:raise ValueError('Harvest identity mismatch')
    rows.append(r)
  verify(p);report.update(rows=rows,terminal=True,completed=sum(r['completed']for r in rows),missing_keys=[c['name']for c in p['cases']if c['name']not in{r['name']for r in rows}],seconds=time.monotonic()-start);write(out,report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
