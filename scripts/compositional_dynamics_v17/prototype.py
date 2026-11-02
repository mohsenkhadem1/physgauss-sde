"""Bounded development-only dynamic prototype; explicit failures and source pins."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,signal,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-dynamics-v17';RAW=ROOT/'runs/compositional-dynamics-v17/prototype';PLAN=REPORT/'prototype-plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve previous prototype')
 paths=closure([Path(__file__)])+[ROOT/'scripts/compositional_dynamics_v17/test_dynamics.py',ROOT/'scripts/compositional_dynamics_v17/PROTOTYPE_PLAN.txt']
 write(PLAN,dict(scope='Development feasibility only; no reserved data/training',rms_definition='coordinate RMS; point-vector RMS divided by sqrt(3)',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},cases=['release','drop'],settings=[dict(n=1,dt=dt)for dt in [.0002,.0001,.00005]]+[dict(n=2,dt=.00005)],surface_order=8,duration=dict(release=.04,drop=.25),dimensions=[.12,.06,.06],young_Pa=12000.,poisson=.35,density=625.,floor_z=-.05,penalty_N_m3=4e7,compression=.2,sample_interval_s=.001,seconds=1200,workers=1,physical_damping='none',restitution_reset='none',gates=dict(world_RMS_over_L=.005,shape_RMS_over_L=.0025,impulse_relative=.05,impulse_absolute_Ns=.0001,apex_over_L=.02,event_time_over_sqrt_Lg=.01,penetration_over_L=.02,minimum_J=.1,energy_relative=.01)))
def verify(p):
 if any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Prototype source changed')
def face_grid(count):
 uv=np.stack(np.meshgrid(np.linspace(0,1,count),np.linspace(0,1,count),indexing='ij'),-1).reshape(-1,2);faces=[]
 for axis in range(3):
  for side in (0.,1.):
   q=np.empty((len(uv),3));q[:,axis]=side;q[:,[j for j in range(3)if j!=axis]]=uv;faces.append(q)
 return np.concatenate(faces)
def run_case(case,n,dt,p,deadline):
 key=f'{case}-n{n}-dt{dt:.8f}';folder=RAW/key;folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();result=dict(key=key,case=case,n=n,dt=dt,plan_sha256=sha(PLAN),completed=False)
 m=DynamicContactModel(GeometrySpec(dimensions=tuple(p['dimensions'])),n=n,order=3,support_mode='free',bulk_energy=StableNeoHookean(p['young_Pa'],p['poisson']),gravity=(0,0,-9.81)if case=='drop'else(0,0,0),density=p['density'],colliders=(RigidPlane(point=(0,0,p['floor_z']),penalty_N_m3=p['penalty_N_m3'],name='floor'),)if case=='drop'else(),surface_order=p['surface_order'])
 engine=VelocityVerlet(m);u=np.zeros_like(m.rest)
 if case=='release':u[:,2]=-p['compression']*m.rest[:,2]
 s=engine.initial_state(u=u);E0=engine.energy(s);scale=max(abs(E0),m.total_mass*9.81*.02 if case=='drop'else 0,1e-10)
 stride=round(p['sample_interval_s']/dt);steps=round(p['duration'][case]/dt)
 if not np.isclose(stride*dt,p['sample_interval_s']):raise ValueError('Exact sample timestep ratio required')
 query=face_grid(17);dense=face_grid(65);rest_surface=m.geometry.map(query).reshape(6,17,17,3);times=[];us=[];vs=[];surfaces=[];ledger=[];sample_depth=[]
 def save_state(step):
  energy=engine.energy(s);force=sum((x['force_on_rigid_N']for x in s.diagnostics['contacts']),np.zeros(3));torque=sum((x['torque_on_rigid_Nm']for x in s.diagnostics['contacts']),np.zeros(3))
  ledger.append([step*dt,energy,.5*np.sum(s.v*(engine.M@s.v)),s.diagnostics['elastic_J'],s.diagnostics['contact_J'],s.diagnostics['bilateral_J'],-np.sum(m.load*s.u),s.diagnostics['minimum_J'],s.diagnostics['maximum_sampled_penetration_m'],*force,*torque,*engine.momentum(s),*engine.angular_momentum(s)])
  if step%stride==0:
   times.append(step*dt);us.append(s.u.copy());vs.append(s.v.copy());surfaces.append((m.geometry.map(query)+m.sample_material(s.u,query)).reshape(6,17,17,3))
   sample_depth.append(max(0.,p['floor_z']-float((m.geometry.map(dense)+m.sample_material(s.u,dense))[:,2].min()))if case=='drop'else 0.)
 try:
  save_state(0)
  for i in range(1,steps+1):
   if time.monotonic()>=deadline:raise TimeoutError('Whole prototype stage cap')
   s=engine.step(s,dt);save_state(i)
  result['completed']=True
 except Exception as e:result.update(error=f'{type(e).__name__}: {e}',failure_time_s=s.time,failure_attempt_time_s=s.time+dt)
 arrays=dict(times_s=np.asarray(times),u=np.asarray(us),velocities=np.asarray(vs),rest_nodes=m.rest,surface_rest=rest_surface,surface_frames=np.asarray(surfaces),material_coordinates=m.material_coordinates,mass_row_sums=np.asarray(engine.M.sum(1)).ravel(),ledger=np.asarray(ledger),saved_dense_penetration_m=np.asarray(sample_depth))
 np.savez_compressed(folder/'trajectory.npz',**arrays);result.update(trajectory_sha256=sha(folder/'trajectory.npz'),saved_frames=len(times),steps_completed=len(ledger)-1,seconds=time.monotonic()-start,initial_energy_J=E0,energy_scale_J=scale,maximum_relative_energy_drift=max(abs(np.asarray(ledger)[:,1]-E0),default=float('inf'))/scale,maximum_dense_penetration_m=max(sample_depth,default=0.),minimum_J=float(np.asarray(ledger)[:,7].min())if ledger else None,ledger_columns=['time_s','total_energy_J','kinetic_J','elastic_J','contact_J','bilateral_J','gravity_relative_J','minimum_J','sampled_penetration_m','contact_force_x','contact_force_y','contact_force_z','contact_torque_x','contact_torque_y','contact_torque_z','momentum_x','momentum_y','momentum_z','angular_momentum_x','angular_momentum_y','angular_momentum_z'],physical_damping='none; conservative potential and velocityVerlet',mass='consistentQ2',colliders=[dict(type='plane',point=[0,0,p['floor_z']],normal=[0,0,1],penalty_N_m3=p['penalty_N_m3'])]if case=='drop'else[])
 write(folder/'complete.json',result);return result

def main():
 p=json.loads(PLAN.read_text());verify(p)
 if RAW.exists():raise ValueError('No overwrite of prototype results')
 RAW.mkdir(parents=True);start=time.monotonic();deadline=start+p['seconds']-20;expected_keys=[f"{case}-n{x['n']}-dt{x['dt']:.8f}"for case in p['cases']for x in p['settings']];report=dict(plan_sha256=sha(PLAN),expected=8,expected_keys=expected_keys,rows=[],complete=False)
 torch.set_num_threads(1)
 for case in p['cases']:
  for settings in p['settings']:
   if time.monotonic()>=deadline:break
   try:row=run_case(case,**settings,p=p,deadline=deadline)
   except Exception as e:row=dict(key=f"{case}-n{settings['n']}-dt{settings['dt']:.8f}",case=case,**settings,completed=False,error=f'{type(e).__name__}: {e}',stage='initialization_or_artifact_write',plan_sha256=sha(PLAN))
   report['rows'].append(row);write(REPORT/'prototype-results.json',report)
 verify(p);report.update(terminal=True,complete=len(report['rows'])==8,completed=sum(x['completed']for x in report['rows']),missing=8-len(report['rows']),missing_keys=[k for k in expected_keys if k not in {r['key']for r in report['rows']}],seconds=time.monotonic()-start);write(REPORT/'prototype-results.json',report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
