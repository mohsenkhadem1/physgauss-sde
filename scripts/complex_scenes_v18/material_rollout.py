"""Frozen shared material and optional bulk-scale extension development rollout.

Known scene boundary diagnostic. Generator material callbacks are replaced before
any dynamic step. This is not the eventual public-scene inference adapter.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from scripts.complex_scenes_v18.scenarios import build_case
from scripts.complex_scenes_v18.pair_contact import SurfaceBinding
from scripts.complex_scenes_v18.material_extension import IndependentBulkScale
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK


def run(case,variant,parameters,*,label,duration=None,n=1,dt=50e-6,seconds=1800,query=None,law_type='learned',split='development'):
    if split not in ('development','held'):raise ValueError('Explicit evaluation split required')
    torch.set_num_threads(1)
    out=ROOT/'runs/complex-scenes-v18'/label/f'{case}-v{variant}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    frames=[];times=[];nodes=[];velocities=[];rigid=[];record=dict(completed=False,case=case,variant=variant,parameters=parameters,development_only=split=='development',evaluation_split=split,known_boundary=True,n=n,dt_s=dt,checkpoint_sha256=hashlib.sha256(Path(BULK).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        record['law_type']=law_type
        if law_type=='learned':
            law=IndependentBulkScale().double();law.base.load_state_dict(torch.load(BULK,map_location='cpu',weights_only=True)['state_dict']);law.eval()
            for p in law.parameters():p.requires_grad_(False)
        elif law_type=='analytic':
            from scripts.complex_scenes_v18.analytic_control import AnalyticControl
            law=AnalyticControl();record['checkpoint_sha256']=None
        else:raise ValueError('Unknown constitutive model')
        def provider(i):
            def field(q,i=i):
                scale=q.new_full(q.shape[:-1],parameters.get('scale',0.)+(parameters.get('second_scale',0.)if i else 0.))
                scale=scale+parameters.get('contrast',0.)*torch.tanh(8*(q[...,0]-parameters.get('center',.5)))
                return torch.stack([scale,torch.full_like(scale,parameters.get('shape',0.)),torch.full_like(scale,parameters.get('family_flag',-1.)),torch.full_like(scale,parameters.get('log_bulk',0.))],-1)
            return law.energy,field
        if query is None:
            scene,s,meta=build_case(case,n=n,variant=variant,material_provider=provider)
        else:
            from scripts.complex_scenes_v18.world_queries import build_prediction
            if query.family!=case or query.variant!=variant:raise ValueError('Query identity mismatch')
            scene,s,meta=build_prediction(query,provider,n=n)
            record['public_query']=meta['public_query']
        s=scene.initial(s.u,s.v);sample=[SurfaceBinding.build(m,9)for m in scene.models]
        record.update(rigid_names=list(scene.rigid),rigid_masses_kg={name:body.mass for name,body in scene.rigid.items()})
        if query is not None:
            geometry={}
            for i,(m,M)in enumerate(zip(scene.models,scene.M)):
                geometry[f'rest_{i}']=m.rest;geometry[f'elements_{i}']=m.elements
                geometry[f'nodal_mass_kg_{i}']=np.real_if_close(np.asarray(M.sum(1)).ravel());geometry[f'gravity_m_s2_{i}']=m.gravity
            np.savez_compressed(out/'geometry.npz',**geometry)
            record['geometry_sha256']=hashlib.sha256((out/'geometry.npz').read_bytes()).hexdigest()
        duration=min(duration,meta['duration_s'])if duration else meta['duration_s'];steps=round(duration/dt);stride=round(.002/dt)
        if not np.isclose(stride*dt,.002):raise ValueError('Sample clock')
        E0=scene.energy(s);minJ=1.;drift=0.
        for i in range(steps+1):
            if i:
                if time.monotonic()-start>seconds:raise TimeoutError('Explicit forward budget')
                s=scene.step(s,dt)
            minJ=min(minJ,min(b['minimum_J']for b in s.diagnostics['bodies']))
            if i%stride==0:
                frames.append(np.stack([b.B@(m.rest+u)for b,m,u in zip(sample,scene.models,s.u)]));times.append(s.time);drift=max(drift,abs(scene.energy(s)-E0-s.work))
                rigid.append(np.array([body['center']for body in s.rigid.values()]).reshape(-1,3))
                if query is not None:
                    nodes.append(np.stack([np.real_if_close(m.rest+u)for m,u in zip(scene.models,s.u)]));velocities.append(np.stack([np.real_if_close(v)for v in s.v]))
            if i%200==0:write(out/'status.json',dict(record,pid=os.getpid(),steps=i,planned_steps=steps,elapsed_s=time.monotonic()-start))
        record.update(completed=True,minimum_J=minJ,maximum_sampled_energy_minus_work_drift_J=drift)
    except Exception as e:record['error']=f'{type(e).__name__}: {e}'
    if frames:
        np.savez_compressed(out/'trajectory.npz',surface_positions_m=frames,times_s=times,rigid_centers_m=rigid)
        record['trajectory_sha256']=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest()
    if nodes:
        np.savez_compressed(out/'nodal-state.npz',positions_m=nodes,velocities_m_s=velocities,times_s=times)
        record['nodal_state_sha256']=hashlib.sha256((out/'nodal-state.npz').read_bytes()).hexdigest()
    record['elapsed_s']=time.monotonic()-start;write(out/'complete.json',record);return record
