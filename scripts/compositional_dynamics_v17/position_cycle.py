"""Smooth prescribed-position compression with explicit actuator-work ledger."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,multiprocessing,argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,AttachmentSpec
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
PLAN=REPORT/'position-cycle-plan.json';RAW=ROOT/'runs/compositional-dynamics-v17/position-cycle'
def controls(t):
 t=float(t)
 if t<=.1:a=np.clip(t/.1,0,1)
 elif t<=.2:a=1.
 else:a=np.clip((.3-t)/.1,0,1)
 f=a*a*a*(10-15*a+6*a*a);p=np.zeros((2,6));p[:,0]=[.018*f,-.018*f];return p

def build(n=3):
 m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=3,support_mode='bilateral',attachments=tuple(AttachmentSpec(6000,(.5,.5),(.375,.375))for _ in range(2)),bulk_energy=StableNeoHookean(12000,.35),gravity=(0,0,0),density=625);return m,VelocityVerlet(m)
def tool_wrenches(m,u):
 result=[]
 for M,d,center,pose in zip(m.anchor_matrices,m.anchor_targets,m.tool_centers,m.poses):
  f=M@(u-d);result.append(np.r_[f.sum(0),np.cross(m.rest+u-center-pose[:3],f).sum(0)])
 return np.array(result)
def controlled_step(e,s,dt,pose):
 m=e.model;old=m.poses.copy()
 if np.any(old[:,3:]!=0)or np.any(pose[:,3:]!=0):raise ValueError('This work ledger covers translations only')
 f0=tool_wrenches(m,s.u);m.set_controls(pose);new=e.step(s,dt);f1=tool_wrenches(m,new.u);work=-.5*np.sum((f0[:,:3]+f1[:,:3])*(pose[:,:3]-old[:,:3]));return new,float(work),f1

def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve earlier plan/data')
 paths=closure([Path(__file__)])+[ROOT/'scripts/compositional_dynamics_v17/test_position_cycle.py']
 write(PLAN,dict(scope='Separate exploratory position-input family; original sudden-release failures retained. No drop/reference qualification.',source_sha256={str(f.relative_to(ROOT)):sha(f)for f in paths},cases=[dict(name=f'cycle-dt{dt:.8f}',dt_s=dt)for dt in [25e-6,12.5e-6]],seconds=900,workers=2,n=3,duration_s=.3,sample_interval_s=.001,geometry_m=[.12,.06,.06],young_Pa=12000,poisson=.35,density_kg_m3=625,attachment_stiffness_N_per_m=6000,gravity=[0,0,0],ramp_hold_unload_s=[.1,.1,.1],compression_command_fraction=.3,end_tool_displacements_m=[.018,-.018],profile='C2 quintic smoothstep, xtranslation only',integrator='VelocityVerlet, cached old force underoldcontrols; new force undernewcontrols',work='Externalwork onsoftbody = minus trapezoidal force_on_tools dot tooltranslation increment; rotations forbidden',energy_scale_J=.5*12000*.3**2*.12*.06*.06,gates=dict(world_coordinate_RMS_m=.0003,shape_coordinate_RMS_m=.00015,minimum_J=.1,energy_work_balance_relative=.01),minimum_available_memory_GiB=24,physical_damping='none',contact='none'))
def verify(p):
 if any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Position-cycle source changed')
def worker(arg):
 c,deadline=arg;p=json.loads(PLAN.read_text());verify(p);torch.set_num_threads(1);start=time.monotonic();folder=RAW/c['name'];folder.mkdir(parents=True,exist_ok=True);row=dict(name=c['name'],case=c,plan_sha256=sha(PLAN),completed=False,reference_qualified=False);records=[];saved={k:[]for k in ['times_s','u','velocities','surface_frames','tool_poses_m_rad','tool_wrenches_N_Nm','actuator_work_J']}
 try:
  m,e=build(p['n']);m.set_controls(controls(0));s=e.initial_state();E0=e.energy(s);W=0.;q=face_grid(17);X=m.geometry.map(q);dt=c['dt_s'];stride=round(p['sample_interval_s']/dt);steps=round(p['duration_s']/dt)
  if not np.isclose(stride*dt,p['sample_interval_s']):raise ValueError('Exact sampled timestep ratio')
  for i in range(steps+1):
   if i:
    if time.monotonic()>=deadline:raise TimeoutError('Whole position-cycle stage cap')
    s,dW,wrench=controlled_step(e,s,dt,controls(i*dt));W+=dW
   else:wrench=tool_wrenches(m,s.u)
   E=e.energy(s);physical_gravity=-np.sum(m.density*m.nodal_volumes[:,None]*m.gravity*s.u)
   records.append([i*dt,E,.5*np.sum(s.v*(e.M@s.v)),s.diagnostics['elastic_J'],s.diagnostics['bilateral_J'],physical_gravity,s.diagnostics['minimum_J'],W,E-W-E0,*e.momentum(s),*e.angular_momentum(s)])
   if i%stride==0:
    vals=dict(times_s=i*dt,u=s.u.copy(),velocities=s.v.copy(),surface_frames=(X+m.sample_material(s.u,q)).reshape(6,17,17,3),tool_poses_m_rad=m.poses.copy(),tool_wrenches_N_Nm=wrench,actuator_work_J=W)
    for k,v in vals.items():saved[k].append(v)
  row['completed']=True
 except Exception as ex:row['error']=f'{type(ex).__name__}: {ex}'
 if records:
  arrays={k:np.array(v)for k,v in saved.items()};arrays.update(ledger=np.array(records),surface_rest=X.reshape(6,17,17,3),rest_nodes=m.rest,material_coordinates=m.material_coordinates,mass_row_sums=np.asarray(e.M.sum(1)).ravel());np.savez_compressed(folder/'trajectory.npz',**arrays);balance=float(np.max(abs(np.array(records)[:,8]))/p['energy_scale_J']);minJ=float(np.min(np.array(records)[:,6]));row.update(path=str(folder.relative_to(ROOT)),trajectory_sha256=sha(folder/'trajectory.npz'),steps_completed=len(records)-1,saved_frames=len(saved['times_s']),minimum_J=minJ,maximum_relative_energy_work_error=balance,diagnostic_gates_pass=row['completed']and minJ>p['gates']['minimum_J']and balance<=p['gates']['energy_work_balance_relative'],initial_energy_J=E0,energy_scale_J=p['energy_scale_J'],final_actuator_work_J=W,colliders=[],sphere_names=[],support_specs=[dict(**asdict(a),face_axis=0,face_side=i,face_origin_m=m.tool_centers[i].tolist(),face_dimensions_m=[.06,.06],footprint='Normalized taperedGaussian overfullfaceinterior',tool_pose_m_rad=[0]*6)for i,a in enumerate(m.attachments)],support_motion_array='tool_poses_m_rad',ledger_columns=['time_s','mechanical_energy_J','kinetic_J','elastic_J','bilateral_J','gravity_J','minimum_J','actuator_work_J','energy_minus_work_minus_initial_J','momentum_x','momentum_y','momentum_z','angular_momentum_x','angular_momentum_y','angular_momentum_z'],scope=p['scope'])
 row['seconds']=time.monotonic()-start;write(folder/'complete.json',row);return row

def main():
 p=json.loads(PLAN.read_text());verify(p)
 if RAW.exists():raise ValueError('Preserve earlier outputs')
 avail=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 if avail<p['minimum_available_memory_GiB']*2**30:raise RuntimeError('Insufficient memory headroom')
 RAW.mkdir(parents=True);start=time.monotonic();deadline=start+p['seconds']-30;report=dict(plan_sha256=sha(PLAN),rows=[],terminal=False,expected_keys=[c['name']for c in p['cases']]);out=REPORT/'position-cycle-results.json';pool=multiprocessing.get_context('spawn').Pool(p['workers'])
 try:
  it=pool.imap_unordered(worker,[(c,deadline-15)for c in p['cases']])
  for _ in p['cases']:report['rows'].append(it.next(timeout=max(.01,deadline-time.monotonic())));write(out,report)
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
