"""Replay a sealed UniPhy material latent under a public drop intervention."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import taichi as ti
from scripts.complex_scenes_v18.uniphy_runtime import models,wrapper
from scripts.complex_scenes_v18.uniphy_drop import add_bowl,add_sphere
from scripts.complex_scenes_v18.world_queries import Query,drop_particle_cloud
from scripts.complex_scenes_v18.particle_tracks import binding
from scripts.complex_scenes_v18.pair_contact import material_surface
from scripts.complex_scenes_v18.run_development import write


def replay(fitdir,query,out,seconds=1800):
    fitdir=Path(fitdir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    seal=json.loads((fitdir/'fit-seal.json').read_text());checkpoint=fitdir/'best.pt'
    if not seal['completed']or seal['forecast_targets_read']or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=seal['checkpoint_sha256']:raise ValueError('Invalid sealed latent')
    _,initial=drop_particle_cloud(query);offset=np.array([.3,.3,.3],np.float32);x=(initial+offset).astype(np.float32)
    ti.init(arch=ti.cuda,device_memory_GB=2,debug=False);nets,latent,_=models()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    with torch.no_grad():latent.weight.copy_(saved['latent'].to(latent.weight))
    dt=seal['dt_s'];steps=round(.24/dt);chunk=32;stride=round(.002/dt)
    if not np.isclose(stride*dt,.002):raise ValueError('Observation clock mismatch')
    w=wrapper(x,np.repeat(x[:,None],chunk+1,axis=1),nets,dt=dt,steps=steps,chunk=chunk,density=625.,gravity=(0,0,-9.81),forward_only=True)
    sim=w.simulator;add_bowl(sim,(np.array([.06,0,-.08 if query.variant else -.07])+offset).tolist(),.10 if query.variant else .12)
    if query.variant:add_sphere(sim,(np.array([.09,.035,-.06])+offset).tolist(),.015)
    w.simulator_variables_initialize();w.update_traj_latent_learnable(latent.weight[0].detach())
    q,_=material_surface(9);index,weight=binding(q);frames=[];particles=[];velocities=[];determinants=[];times=[];start=time.monotonic()
    from dataclasses import asdict
    record=dict(completed=False,development_only=True,method='UniPhy native MPM with fixed bowl/sphere hooks',public_query=asdict(query),checkpoint_sha256=seal['checkpoint_sha256'],forecast_targets_read=False)
    try:
        for step in range(steps+1):
            if time.monotonic()-start>seconds:raise TimeoutError('Explicit native replay budget')
            if step:
                with torch.no_grad():sim.advance_F_stress(step-1,0)
                if not sim.cfl_satisfy[None]:raise FloatingPointError('Native CFL failure')
            if step%stride==0:
                local=step%chunk;pos=sim.x.to_numpy()[:len(x),local].astype(float)-offset
                velocity=sim.v.to_numpy()[:len(x),local];F=sim.F.to_numpy()[:len(x),local];J=np.linalg.det(F)
                if not np.isfinite(pos).all()or not np.isfinite(velocity).all()or not np.isfinite(J).all():raise FloatingPointError('Nonfinite native state')
                particles.append(pos);velocities.append(velocity);determinants.append(J);frames.append((pos[index]*weight[...,None]).sum(1)[None]);times.append(step*dt)
            if step%200==0:write(out/'status.json',dict(record,pid=os.getpid(),steps=step,planned_steps=steps,elapsed_s=time.monotonic()-start))
        record['completed']=True
    except Exception as e:record['error']=str(e)
    np.savez_compressed(out/'trajectory.npz',surface_positions_m=frames,times_s=times)
    np.savez_compressed(out/'particle-state.npz',positions_m=particles,velocities_m_s=velocities,detF=determinants,times_s=times,mass_kg=sim.p_mass.to_numpy()[:len(x)],gravity_m_s2=np.array([0,0,-9.81]))
    record.update(saved_frames=len(frames),elapsed_s=time.monotonic()-start,trajectory_sha256=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest(),particle_state_sha256=hashlib.sha256((out/'particle-state.npz').read_bytes()).hexdigest(),minimum_native_detF=float(np.min(determinants))if determinants else None)
    write(out/'complete.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fitdir',required=True);p.add_argument('--query-json',required=True);p.add_argument('--output',required=True);args=p.parse_args();torch.set_num_threads(1)
    r=replay(args.fitdir,Query(**json.loads(Path(args.query_json).read_text())),args.output);print(json.dumps(r));raise SystemExit(0 if r['completed']else 1)
