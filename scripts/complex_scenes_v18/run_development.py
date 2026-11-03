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


def run(case,n,dt,variant=0,label='pilot',surface_samples=5,seconds=1800,contact_device='cpu',minimum_j=.1,dense_j_samples=0,bulk_quadrature='gauss3',world=None,query=None,save_state=False,split='development'):
    if split not in ('development','held'):raise ValueError('Explicit evaluation split required')
    if split=='held' and (world is None or query is None):raise ValueError('Held runs require explicit world and query')
    torch.set_num_threads(1);out=ROOT/'runs/complex-scenes-v18'/label/f'{case}-v{variant}-n{n}-dt{dt:.8f}-s{surface_samples}'
    if out.exists():raise ValueError('Preserve previous attempt: '+str(out))
    out.mkdir(parents=True);t0=time.monotonic();records=[];saved=[];status=dict(family=case,variant=variant,n=n,dt_s=dt,surface_samples=surface_samples,completed=False,development_only=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((ROOT/'scripts/complex_scenes_v18').glob('*.py'))})
    status.update(evaluation_split=split,development_only=split=='development')
    write(out/'status.json',dict(status,state='running',pid=os.getpid()))
    try:
        if world is None and query is None:
            scene,s,meta=build_case(case,n,surface_samples,variant,contact_device=contact_device,bulk_quadrature=bulk_quadrature)
        else:
            from scripts.complex_scenes_v18.world_queries import build_reference
            if world is None or query is None or query.family!=case or query.variant!=variant:raise ValueError('Explicit matching world and query required')
            scene,s,meta=build_reference(world,query,n=n,surface_samples=surface_samples,contact_device=contact_device,bulk_quadrature=bulk_quadrature)
        status.update(meta);status.update(contact_device=contact_device,bulk_quadrature=bulk_quadrature,evaluation_split=split,development_only=split=='development')
        status.update(rigid_names=list(scene.rigid),rigid_masses_kg={name:body.mass for name,body in scene.rigid.items()})
        save_state=save_state or world is not None
        status['full_nodal_state_saved']=save_state
        if save_state:
            geometry={}
            for index,(m,M)in enumerate(zip(scene.models,scene.M)):
                for key,value in dict(rest=m.rest,elements=m.elements,quadrature=m.quad,mass_data=M.data,mass_indices=M.indices,mass_indptr=M.indptr,mass_shape=M.shape,mass_format=M.format,nodal_mass_kg=np.real_if_close(np.asarray(M.sum(1)).ravel()),gravity_m_s2=m.gravity).items():geometry[f'{key}_{index}']=value
            np.savez_compressed(out/'geometry.npz',**geometry)
            status['geometry_sha256']=hashlib.sha256((out/'geometry.npz').read_bytes()).hexdigest()
        if not 0<minimum_j<1:raise ValueError('Positive finite orientation guard required')
        for m in scene.models:m.minimum_J=minimum_j
        status['minimum_J_guard']=minimum_j
        auditors=[]
        if dense_j_samples:
            from scripts.complex_scenes_v18.dense_jacobian import DenseJacobianAudit
            auditors=[DenseJacobianAudit(m,dense_j_samples)for m in scene.models];status.update(dense_J_samples_per_axis=dense_j_samples,minimum_dense_J=1.,dense_J_check_frame_dt_s=.002)
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
                if auditors:
                    dense=min(a.minimum(u)for a,u in zip(auditors,s.u));status['minimum_dense_J']=min(status['minimum_dense_J'],dense)
                    if dense<=0:
                        error=ValueError('Dense sampled orientation inversion');error.scene_candidate=dict(u=s.u,rigid=s.rigid,time=s.time);raise error
                saved.append(dict(time=s.time,surface=np.stack([b.B@(m.rest+u)for b,m,u in zip(sample,scene.models,s.u)]),rigid=np.array([b['center']for b in s.rigid.values()]).reshape(-1,3)))
                if save_state:saved[-1].update(nodes=np.stack([np.real_if_close(m.rest+u)for m,u in zip(scene.models,s.u)]),velocities=np.stack([np.real_if_close(v)for v in s.v]))
            if i%200==0:write(out/'status.json',dict(status,state='running',steps=i,planned_steps=steps,elapsed_s=time.monotonic()-t0,pid=os.getpid()))
        status['completed']=True
    except Exception as e:
        status.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
        candidate=getattr(e,'scene_candidate',None)
        if candidate is not None:
            np.savez_compressed(out/'failure-candidate.npz',**{f'u_{i}':u for i,u in enumerate(candidate['u'])})
            minima=[]
            with torch.no_grad():
                for m,u in zip(scene.models,candidate['u']):
                    engine=m.engine;ue=engine.tensor(u)[engine.elements]
                    minima.append(min(float(torch.linalg.det(engine.I+torch.einsum('eai,qeaj->qeij',ue,g)).min())for g in (engine.grad,engine.check_grad)))
            status.update(failure_candidate_time_s=candidate['time'],failure_candidate_minimum_J=minima,failure_candidate_sha256=hashlib.sha256((out/'failure-candidate.npz').read_bytes()).hexdigest())
    if records:
        if save_state and saved:
            np.savez_compressed(out/'nodal-state.npz',positions_m=np.stack([r['nodes']for r in saved]),velocities_m_s=np.stack([r['velocities']for r in saved]),times_s=np.array([r['time']for r in saved]))
            status['nodal_state_sha256']=hashlib.sha256((out/'nodal-state.npz').read_bytes()).hexdigest()
        a=np.asarray(records);np.savez_compressed(out/'trajectory.npz',ledger=a,times_s=np.array([r['time']for r in saved]),surface_positions_m=np.array([r['surface']for r in saved]),rigid_centers_m=np.array([r['rigid']for r in saved]),material_queries=sample[0].q,rest_surface=np.stack([b.B@m.rest for b,m in zip(sample,scene.models)]))
        status.update(steps_completed=len(records)-1,saved_frames=len(saved),minimum_J=float(a[:,4].min()),maximum_penetration_m=float(a[:,5].max()),maximum_energy_minus_work_drift_J=float(abs(a[:,3]).max()),trajectory_sha256=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest(),ledger_columns=['time_s','mechanical_energy_J','actuator_work_J','energy_minus_initial_minus_work_J','minimum_J','penetration_m','Px','Py','Pz'])
    status.update(state='complete'if status['completed']else'failed',elapsed_s=time.monotonic()-t0,path=str(out.relative_to(ROOT)))
    write(out/'status.json',status);write(out/'complete.json',status);return status


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=FAMILIES,required=True);ap.add_argument('--n',type=int,default=1);ap.add_argument('--dt',type=float,default=50e-6);ap.add_argument('--variant',type=int,default=0);ap.add_argument('--label',default='pilot');ap.add_argument('--surface-samples',type=int,default=5);ap.add_argument('--seconds',type=int,default=1800);ap.add_argument('--contact-device',default='cpu');ap.add_argument('--minimum-j',type=float,default=.1);ap.add_argument('--dense-j-samples',type=int,default=0);ap.add_argument('--bulk-quadrature',choices=['gauss3','lobatto4','lobatto5'],default='gauss3');a=ap.parse_args()
    r=run(a.case,a.n,a.dt,a.variant,a.label,a.surface_samples,a.seconds,a.contact_device,a.minimum_j,a.dense_j_samples,a.bulk_quadrature);print(json.dumps(r,indent=2));raise SystemExit(0 if r['completed']else 1)
