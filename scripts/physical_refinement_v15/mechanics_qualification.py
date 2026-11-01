"""Read-only development mesh/wrench audit for the unchanged Q2 fixture."""
from pathlib import Path
import sys,json,gc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
from scripts.position_ood.fixture import controls
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,CASES,ACTIONS,write,sha
GATES=dict(field_coordinate_rms_m=2e-5,field_max_coordinate_m=1e-4,force_absolute_floor_N=1e-4,force_relative=.005,torque_absolute_floor_Nm=1e-5,torque_relative=.005,residual_N=1e-9,balance_N=1e-7)
def wrench(model,u,poses):
 model.set_controls(poses);forces=[];torques=[]
 for i,(M,d) in enumerate(zip(model.anchor_matrices,model.anchor_targets)):
  f=M@(u-d);center=np.array([0. if i==0 else .24,0.,0.])+poses[i,:3];lever=model.rest+d-center
  forces.append(f.sum(0));torques.append(np.cross(lever,f).sum(0))
 return np.asarray(forces),np.asarray(torques)
def main():
 results={};paths=[]
 for name in CASES:
  results[name]={}
  for complete in sorted((BASE/'mechanics-pilot'/name).glob('*/complete.json')):
   meta=json.loads(complete.read_text());spec=meta['spec'];model=SmoothMaterialFixture(spec['n'],**spec['case'],gravity=(0,0,0),field_amplitude_J=0,order=spec['order'],device='cpu');rows=[]
   for j,(action,amp) in enumerate(ACTIONS):
    p=complete.parent/f'action{j}.npz';a=np.load(p);pose=controls(action,amp);f,t=wrench(model,a['nodal_displacement'],pose);np.testing.assert_allclose(f,a['reaction_forces_N'],atol=1e-12,rtol=1e-12)
    rows.append(dict(action=action,poses_m_rad=pose.tolist(),forces_N=f.tolist(),torques_Nm=t.tolist(),torque_origin_world_m=(np.array([[0,0,0],[.24,0,0]])+pose[:,:3]).tolist()));paths.append(p)
   results[name][complete.parent.name]=rows;model.engine=None;del model;gc.collect()
 comparisons=[]
 for name,levels in results.items():
  ns=sorted(int(k.split('-')[0][1:]) for k in levels if k.endswith('q3'))
  for n,m in zip(ns,ns[1:]):
   for j in range(len(ACTIONS)):
    a=np.load(BASE/'mechanics-pilot'/name/f'n{n}-q3/action{j}.npz');b=np.load(BASE/'mechanics-pilot'/name/f'n{m}-q3/action{j}.npz');diff=a['field']-b['field'];force=np.asarray(levels[f'n{n}-q3'][j]['forces_N'])-np.asarray(levels[f'n{m}-q3'][j]['forces_N']);torque=np.asarray(levels[f'n{n}-q3'][j]['torques_Nm'])-np.asarray(levels[f'n{m}-q3'][j]['torques_Nm']);ftol=1e-4+.005*np.linalg.norm(b['reaction_forces_N'],axis=1);ttol=1e-5+.005*np.linalg.norm(levels[f'n{m}-q3'][j]['torques_Nm'],axis=1)
    er=float(np.sqrt(np.mean(diff**2)));em=float(np.abs(diff).max());fp=bool((np.linalg.norm(force,axis=1)<=ftol).all());tp=bool((np.linalg.norm(torque,axis=1)<=ttol).all())
    comparisons.append(dict(name=name,mesh_pair=[n,m],action=ACTIONS[j][0],field_rms_mm=er*1000,field_max_mm=em*1000,force_difference_N=np.linalg.norm(force,axis=1).tolist(),force_tolerance_N=ftol.tolist(),torque_difference_Nm=np.linalg.norm(torque,axis=1).tolist(),torque_tolerance_Nm=ttol.tolist(),field_pass=er<=2e-5 and em<=1e-4,force_pass=fp,torque_pass=tp,passed=bool(er<=2e-5 and em<=1e-4 and fp and tp)))
 # Differential sign/origin check, NOT a forward-model accuracy claim.
 model=SmoothMaterialFixture(2,**CASES['homogeneous'],gravity=(0,0,0),device='cpu');pose=controls(*ACTIONS[2]);u=np.load(BASE/'mechanics-pilot/homogeneous/n2-q3/action2.npz')['nodal_displacement'];f,t=wrench(model,u,pose);epsilon=1e-6;finite=np.zeros((2,3))
 for i in range(2):
  R=Rotation.from_rotvec(pose[i,3:]).as_matrix()
  for k in range(3):
   values=[]
   for sign in (1,-1):
    p=pose.copy();axis=np.eye(3)[k]*epsilon*sign;p[i,3:]=Rotation.from_matrix(Rotation.from_rotvec(axis).as_matrix()@R).as_rotvec();model.set_controls(p);values.append(model.evaluate(u)['potential_J'])
   finite[i,k]=-(values[0]-values[1])/(2*epsilon)
 torque_error=float(np.max(np.abs(finite-t)))
 report=dict(scope='Development only; two-resolution consistency, not independent physical truth',gates=GATES,wrench_convention='Force on each controller; world torque about translated rest-end center, target-surface lever, units N and Nm',wrenches=results,comparisons=comparisons,torque_energy_derivative_max_abs_error_Nm=torque_error,source_sha256=sha(Path(__file__)),artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
 write(REPORT/'mechanics-development-qualification.json',report);print(json.dumps(dict(comparisons=comparisons,torque_derivative_error_Nm=torque_error)),flush=True)
if __name__=='__main__':main()
