"""Bounded integration pilot: no learned inference, no calibrated uncertainty."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch
from .scene import PhysicalHypothesis,Intervention,build
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--young',type=float,required=True);p.add_argument('--device',default='cpu');p.add_argument('--dt',type=float,default=.00005);a=p.parse_args()
    torch.set_num_threads(1);h=PhysicalHypothesis(a.young)
    action=Intervention(a.kind,gravity=(0,0,0),force_N=(1.,0.,0.),floor_z_m=-1.) if a.kind=='force_pulse' else Intervention(a.kind)
    key=f'{a.kind}-{int(a.young)}-dt{a.dt:g}';out=ROOT/'runs/editable-scene-v23'/key;out.mkdir(exist_ok=False)
    scene,s=build(h,action,device=a.device);m=scene.models[0];mass=np.asarray(scene.M[0].sum(1)).ravel();total=mass.sum()
    positions=[];velocities=[];rows=[];initial_energy=scene.energy(s);start=time.monotonic()
    for i in range(round(.24/a.dt)+1):
        if time.monotonic()-start>900:raise TimeoutError('15 minute per-case limit')
        if i:s=scene.step(s,a.dt)
        if i%round(.002/a.dt)==0:
            x=m.rest+s.u[0];positions.append(x.copy());velocities.append(s.v[0].copy())
            X=m.rest-m.rest.mean(0);Y=x-x.mean(0);u,_,vt=np.linalg.svd(X.T@Y);D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt);R=u@D@vt
            body=s.diagnostics['bodies'][0]
            rows.append(dict(time_s=s.time,centroid_m=(mass@x/total).tolist(),deformation_rms_mm=float(np.sqrt(np.mean(np.sum((Y-X@R)**2,axis=1)))*1000),
                min_J=body['minimum_J'],penetration_mm=body['max_penetration_m']*1000,energy_work_drift_J=scene.energy(s)-initial_energy-s.work,momentum_kg_m_s=scene.momentum(s).tolist()))
    np.savez_compressed(out/'states.npz',positions=positions,velocities=velocities,rest=m.rest,times=[r['time_s'] for r in rows])
    report=dict(complete=True,hypothesis=asdict(h),action=asdict(action),dt_s=a.dt,n=1,rows=rows,
        max_deformation_mm=max(r['deformation_rms_mm']for r in rows),max_penetration_mm=max(r['penetration_mm']for r in rows),
        min_J=min(r['min_J']for r in rows),max_energy_work_drift_J=max(abs(r['energy_work_drift_J'])for r in rows),
        elapsed_s=time.monotonic()-start,source_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path(__file__).with_name('scene.py'))},
        scope='Known-material, known-boundary implementation pilot. Three chosen hypotheses are sensitivity samples, not a calibrated posterior. Coarse mesh; contact penalty; no friction or real-data accuracy claim.')
    if a.kind=='force_pulse':report['force_impulse_error_Ns']=float(np.linalg.norm(np.array(rows[-1]['momentum_kg_m_s'])-np.array(action.force_N)*(action.end_s-action.start_s)/2))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(key,{k:report[k] for k in ('max_deformation_mm','max_penetration_mm','min_J','elapsed_s')},flush=True)

if __name__=='__main__':main()
