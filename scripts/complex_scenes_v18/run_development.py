"""Resumable per-case development rollout with atomic status and failure prefix."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from scripts.complex_scenes_v18.scenarios import build_case,FAMILIES
from scripts.complex_scenes_v18.pair_contact import SurfaceBinding

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def run(case,n,dt,variant=0,label='pilot',surface_samples=5,seconds=1800,contact_device='cpu'):
    torch.set_num_threads(1);out=ROOT/'runs/complex-scenes-v18'/label/f'{case}-v{variant}-n{n}-dt{dt:.8f}-s{surface_samples}'
    if out.exists():raise ValueError('Preserve previous attempt: '+str(out))
    out.mkdir(parents=True);t0=time.monotonic();records=[];saved=[];status=dict(family=case,variant=variant,n=n,dt_s=dt,surface_samples=surface_samples,completed=False,development_only=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((ROOT/'scripts/complex_scenes_v18').glob('*.py'))})
    write(out/'status.json',dict(status,state='running',pid=os.getpid()))
    try:
        scene,s,meta=build_case(case,n,surface_samples,variant,contact_device=contact_device);status.update(meta);status['contact_device']=contact_device
        sample=[SurfaceBinding.build(m,9)for m in scene.models]
        E0=scene.energy(s);P0=scene.momentum(s);stride=round(.002/dt);steps=round(meta['duration_s']/dt)
        if not np.isclose(stride*dt,.002):raise ValueError('Integer sampling stride required')
        for i in range(steps+1):
            if i:
                if time.monotonic()-t0>seconds:raise TimeoutError('Declared development case budget')
                s=scene.step(s,dt)
            E=scene.energy(s);d=s.diagnostics
            minJ=min(b['minimum_J']for b in d['bodies']);depth=max([b['max_penetration_m']for b in d['bodies']]+[(d['pair']or {}).get('max_sampled_penetration_m',0.)])
            records.append([s.time,E,s.work,E-E0-s.work,minJ,depth,*scene.momentum(s)])
            if i%stride==0:
                saved.append(dict(time=s.time,surface=np.stack([b.B@(m.rest+u)for b,m,u in zip(sample,scene.models,s.u)]),rigid=np.array([b['center']for b in s.rigid.values()]).reshape(-1,3)))
            if i%200==0:write(out/'status.json',dict(status,state='running',steps=i,planned_steps=steps,elapsed_s=time.monotonic()-t0,pid=os.getpid()))
        status['completed']=True
    except Exception as e:
        status.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    if records:
        a=np.asarray(records);np.savez_compressed(out/'trajectory.npz',ledger=a,times_s=np.array([r['time']for r in saved]),surface_positions_m=np.array([r['surface']for r in saved]),rigid_centers_m=np.array([r['rigid']for r in saved]),material_queries=sample[0].q,rest_surface=np.stack([b.B@m.rest for b,m in zip(sample,scene.models)]))
        status.update(steps_completed=len(records)-1,saved_frames=len(saved),minimum_J=float(a[:,4].min()),maximum_penetration_m=float(a[:,5].max()),maximum_energy_minus_work_drift_J=float(abs(a[:,3]).max()),trajectory_sha256=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest(),ledger_columns=['time_s','mechanical_energy_J','actuator_work_J','energy_minus_initial_minus_work_J','minimum_J','penetration_m','Px','Py','Pz'])
    status.update(state='complete'if status['completed']else'failed',elapsed_s=time.monotonic()-t0,path=str(out.relative_to(ROOT)))
    write(out/'status.json',status);write(out/'complete.json',status);return status


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=FAMILIES,required=True);ap.add_argument('--n',type=int,default=1);ap.add_argument('--dt',type=float,default=50e-6);ap.add_argument('--variant',type=int,default=0);ap.add_argument('--label',default='pilot');ap.add_argument('--surface-samples',type=int,default=5);ap.add_argument('--seconds',type=int,default=1800);ap.add_argument('--contact-device',default='cpu');a=ap.parse_args()
    r=run(a.case,a.n,a.dt,a.variant,a.label,a.surface_samples,a.seconds,a.contact_device);print(json.dumps(r,indent=2));raise SystemExit(0 if r['completed']else 1)
