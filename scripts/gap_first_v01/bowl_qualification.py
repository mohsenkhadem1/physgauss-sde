"""Development qualification, explicitly not a benchmark or tuned test set."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from .bowl_physics import run,build,Bowl,rotation_y

ROOT=Path('reports/gap-first-v01/bowl')

def metrics(r):
    rest=r['rest']; frames=r['frames']
    A=rest-rest.mean(0); nonrigid=[]; total=[]
    for frame in frames:
        B=frame-frame.mean(0); u,s,vt=np.linalg.svd(A.T@B)
        R=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
        nonrigid.append(np.sqrt(np.mean(np.sum((A@R-B)**2,axis=1))))
        total.append(np.sqrt(np.mean(np.sum((frame-rest)**2,axis=1))))
    return dict(min_j=float(r['min_j']),penetration_mm=float(r['max_penetration_m']*1000),
        maximum_nonrigid_rms_mm=float(max(nonrigid)*1000),final_nonrigid_rms_mm=float(nonrigid[-1]*1000),
        final_total_motion_rms_mm=float(total[-1]*1000),final_maximum_speed_m_s=float(np.linalg.norm(r['velocities'][-1],axis=1).max()),
        mass_kg=float(r['mass'].sum()),max_reaction_N=float(np.linalg.norm(r['reactions'][:,0],axis=1).max()))

def run_all():
    ROOT.mkdir(parents=True,exist_ok=True)
    configs=[dict(n=3,dt=1e-4),dict(n=3,dt=5e-5),dict(n=6,dt=5e-5),
             dict(n=3,dt=5e-5,penalty=9600.),dict(n=3,dt=5e-5,subdivisions=4),
             dict(n=3,dt=1e-4,interaction='push',duration=1.1),
             dict(n=3,dt=1e-4,interaction='tilt',duration=1.1)]
    out={'scope':'Development qualification; no held-out outcomes or protocol tuning', 'cases':[]}
    runs=[]
    for i,c in enumerate(configs):
        start=time.monotonic()
        try:
            r=run(**c);runs.append(r)
            np.savez_compressed(f'runs/gap-first-v01/bowl/qualification-{i}.npz',**r)
            row=dict(config=c,elapsed_s=time.monotonic()-start,status='ok',**metrics(r))
        except Exception as e:
            runs.append(None);row=dict(config=c,status='failed',error=repr(e))
        out['cases'].append(row);print(row,flush=True)
        (ROOT/'qualification.json').write_text(json.dumps(out,indent=2))
    for j in (1,2,3,4):
        a,b=runs[1],runs[j]
        if a is None or b is None: continue
        dist=np.linalg.norm(a['rest'][:,None]-b['rest'][None],axis=2)
        ids=dist.argmin(1)
        assert dist.min(1).max()<1e-12
        delta=a['frames']-b['frames'][:,ids]
        out['cases'][j]['relative_to_half_dt_n3_rms_mm']=float(np.sqrt(np.mean(delta**2))*1000)
        out['cases'][j]['relative_to_half_dt_n3_max_point_mm']=float(np.linalg.norm(delta,axis=2).max()*1000)
    s=build();p,v=s.surface_state();test=np.arange(len(p)*3).reshape(-1,3)*1e-8
    scattered=s.distribute_surface_force(test)
    out['surface_force_sum_error_N']=float(np.linalg.norm(scattered.sum(0)-test.sum(0)))
    out['surface_torque_error_Nm']=float(np.linalg.norm(np.cross(s.x,scattered).sum(0)-np.cross(p,test).sum(0)))
    out['surface_area_error_m2']=float(abs(s.surface_patch_area.sum()-s.surface_area.sum()))
    bowl=Bowl(radius=.09,center=(0,0,.09))
    q=np.array([[0,0,-.001],[0,0,.001],[.091,0,.09],[0,0,.2]])
    phi,normal=bowl.query(q)
    assert phi[0]<0 and phi[1]>0 and phi[2]<0 and np.isinf(phi[3]) and normal[0,2]>0 and normal[2,0]<0
    out['normal_sign_checks']='pass: floor inward/up, side inward, open top'
    out['source_sha256']=hashlib.sha256(Path('scripts/gap_first_v01/bowl_physics.py').read_bytes()).hexdigest()
    (ROOT/'qualification.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':run_all()
