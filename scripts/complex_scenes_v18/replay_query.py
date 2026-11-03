"""Replay sealed fitted springs under public changed interactions, without targets.

Strictly requires saved fitted graph identities; historical development fits
without a graph cannot silently regenerate nearest-neighbor parameter indexing.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.complex_scenes_v18.run_development import write
from scripts.complex_scenes_v18.world_queries import Query


def replay(fitdir,query,out,seconds=1800):
    fitdir=Path(fitdir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    seal=json.loads((fitdir/'fit-seal.json').read_text());checkpoint=fitdir/'best.pt';graph=fitdir/'graph.npz'
    if not seal['completed']or seal['forecast_targets_read']:raise ValueError('Unsealed model')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=seal['checkpoint_sha256']:raise ValueError('Checkpoint integrity')
    if not graph.exists()or hashlib.sha256(graph.read_bytes()).hexdigest()!=seal.get('graph_sha256'):raise ValueError('Saved graph identity is mandatory')
    method=seal['method'];state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if method=='phystwin':
        from scripts.complex_scenes_v18.differentiable_phystwin import DifferentiablePhysTwinScene as Class
    elif method=='spring-gaus':
        from scripts.complex_scenes_v18.differentiable_spring import DifferentiableSpringScene as Class
    else:raise ValueError('Unsupported sealed method identifier')
    sim=Class(query.family,query.variant,query=query)
    with np.load(graph)as saved:
        for i,model in enumerate(sim.native):
            if method=='phystwin':
                if not np.array_equal(saved[f'edges_{i}'],sim.edges[i].cpu().numpy()):raise ValueError('Query changed rest graph')
            else:
                index=saved[f'knn_index_{i}'];length=saved[f'origin_len_{i}']
                if index.shape!=tuple(model.knn_index.shape)or length.shape!=tuple(model.origin_len.shape):raise ValueError('Query changed rest graph dimensions')
                model.knn_index=torch.as_tensor(index,device='cuda',dtype=model.knn_index.dtype)
                model.origin_len=torch.as_tensor(length,device='cuda',dtype=model.origin_len.dtype)
    if len(state['parameters'])!=len(sim.parameters):raise ValueError('Parameter count changed')
    with torch.no_grad():
        for p,value in zip(sim.parameters,state['parameters']):
            if p.shape!=value.shape:raise ValueError('Parameter shape changed')
            p.copy_(value.to(p))
    native=sim.native_replay()if method=='phystwin'else None
    current=sim.initial();dt=seal['dt_s'];stride=round(.002/dt);frames=[];times=[];nodes=[];velocities=[];rigid=[];start=time.monotonic()
    geometry={}
    for i,(m,M)in enumerate(zip(sim.base.scene.models,sim.base.scene.M)):
        geometry[f'rest_{i}']=m.rest;geometry[f'elements_{i}']=m.elements
        geometry[f'nodal_mass_kg_{i}']=np.real_if_close(np.asarray(M.sum(1)).ravel());geometry[f'gravity_m_s2_{i}']=m.gravity
    np.savez_compressed(out/'geometry.npz',**geometry)
    record=dict(completed=False,development_only=True,method=method,checkpoint_sha256=seal['checkpoint_sha256'],graph_sha256=seal['graph_sha256'],public_query=sim.base.metadata['public_query'],forecast_targets_read=False,fit_seal_sha256=hashlib.sha256((fitdir/'fit-seal.json').read_bytes()).hexdigest())
    record.update(rigid_names=sim.names,rigid_masses_kg={name:body.mass for name,body in sim.base.scene.rigid.items()})
    try:
        for step in range(round(sim.base.metadata['duration_s']/dt)+1):
            if time.monotonic()-start>seconds:raise TimeoutError('Explicit replay budget')
            if step:
                if native is not None:native.step(dt)
                else:current=sim.step(current,step-1,dt,training=False)
            if step%stride==0:
                x=native.sample_surface()if native is not None else sim.positions(current).detach().cpu().numpy()
                if not np.isfinite(x).all()or abs(x).max()>10:raise FloatingPointError('Invalid trajectory')
                frames.append(x);times.append(step*dt)
                if native is not None:
                    nodes.append(np.stack([a.detach().cpu().numpy()for a in native.x]));velocities.append(np.stack([a.detach().cpu().numpy()for a in native.v]))
                    rigid.append(np.array([body['center']for body in native.rigid.values()]).reshape(-1,3))
                else:
                    nodes.append(np.stack([a.detach().cpu().numpy()for a in current[:sim.count]]));velocities.append(np.stack([a.detach().cpu().numpy()for a in current[sim.count:2*sim.count]]))
                    rigid.append(np.array([a.detach().cpu().numpy()for a in current[2*sim.count:2*sim.count+len(sim.names)]]).reshape(-1,3))
        record['completed']=True
    except Exception as e:record['error']=str(e)
    np.savez_compressed(out/'trajectory.npz',surface_positions_m=frames,times_s=times,rigid_centers_m=rigid)
    np.savez_compressed(out/'nodal-state.npz',positions_m=nodes,velocities_m_s=velocities,times_s=times)
    record.update(geometry_sha256=hashlib.sha256((out/'geometry.npz').read_bytes()).hexdigest(),nodal_state_sha256=hashlib.sha256((out/'nodal-state.npz').read_bytes()).hexdigest())
    record.update(saved_frames=len(frames),elapsed_s=time.monotonic()-start,trajectory_sha256=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest())
    write(out/'complete.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fitdir',required=True);p.add_argument('--query-json',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(1);r=replay(args.fitdir,Query(**json.loads(Path(args.query_json).read_text())),args.output);print(json.dumps(r));raise SystemExit(0 if r['completed']else 1)
