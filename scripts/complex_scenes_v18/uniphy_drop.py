"""Native UniPhy MPM development drop with declared analytic bowl extension.

Author MPM, pretrained constitutive networks and chunking are unmodified.
The bowl/sphere hooks use the author's analytic collision API and its separating
velocity convention. Mean released latent is initialization, NOT a fitted result.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import taichi as ti
from scripts.complex_scenes_v18.uniphy_runtime import models,wrapper
from scripts.complex_scenes_v18.run_development import ROOT,write


def add_bowl(sim,center,radius):
    @ti.func
    def velocity(I,v):
        x=I.cast(sim.dtype)*sim.dx[None]-ti.Vector(center)
        gap=x[2]-(x[0]*x[0]+x[1]*x[1])/(2*radius)
        if gap<=1e-6:
            n=ti.Vector([-x[0]/radius,-x[1]/radius,1.]).normalized()
            v=v-n*ti.min(n.dot(v),0.)
        return v
    sim.analytic_collision.append(velocity)


def add_sphere(sim,center,radius):
    @ti.func
    def velocity(I,v):
        x=I.cast(sim.dtype)*sim.dx[None]-ti.Vector(center);r=x.norm()
        if r<=radius and r>1e-10:
            n=x/r;v=v-n*ti.min(n.dot(v),0.)
        return v
    sim.analytic_collision.append(velocity)


def run(args):
    out=ROOT/'runs/complex-scenes-v18'/args.label/f'container_drop-v{args.variant}'
    out.mkdir(parents=True,exist_ok=False);t=time.monotonic();frames=[];times=[]
    record=dict(method='Native UniPhy MPM with bowl/sphere analytic geometry hooks',development_only=True,fitted=False,completed=False,variant=args.variant,dt_s=args.dt,initial_latent='Mean of released training latent bank',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        ti.init(arch=ti.cuda,device_memory_GB=2,debug=False);nets,latent,bank=models()
        dx=.02;h=dx/2;offset=np.array([.3,.3,.3],np.float32)
        q=np.stack(np.meshgrid((np.arange(12)+.5)/12,(np.arange(6)+.5)/6,(np.arange(6)+.5)/6,indexing='ij'),-1).reshape(-1,3)
        x=q*np.array([.12,.06,.06])+np.array([0,-.03,-.03]);initial=x.copy();x=(x+offset).astype(np.float32)
        steps=round(args.duration/args.dt);chunk=32
        w=wrapper(x,np.repeat(x[:,None],chunk+1,axis=1),nets,dt=args.dt,steps=steps,chunk=chunk,density=625.,gravity=(0,0,-9.81),forward_only=True)
        sim=w.simulator;center=(np.array([.06,0,-.08 if args.variant else -.07])+offset).tolist()
        add_bowl(sim,center,.10 if args.variant else .12)
        if args.variant:add_sphere(sim,(np.array([.09,.035,-.06])+offset).tolist(),.015)
        w.simulator_variables_initialize();w.update_traj_latent_learnable(latent.weight[0].detach())
        stride=round(.002/args.dt)
        if not np.isclose(stride*args.dt,.002):raise ValueError('Observation clock mismatch')
        frames.append(initial);times.append(0.)
        for i in range(steps):
            if time.monotonic()-t>args.seconds:raise TimeoutError('Explicit native rollout budget')
            with torch.no_grad():sim.advance_F_stress(i,0)
            if not sim.cfl_satisfy[None]:raise RuntimeError('Native CFL failure at '+str(i))
            if (i+1)%stride==0:
                # Native chunk rollover copies the last state to slot zero.
                local=(i+1)%chunk;position=sim.x.to_numpy()[:len(x),local].astype(float)-offset
                if not np.isfinite(position).all():raise FloatingPointError('Nonfinite native positions')
                frames.append(position);times.append((i+1)*args.dt)
            if (i+1)%200==0:
                write(out/'status.json',dict(record,pid=os.getpid(),steps=i+1,planned_steps=steps,elapsed_s=time.monotonic()-t))
        record.update(completed=True,particles=len(x),grid_spacing_m=dx,particle_spacing_m=h,steps=steps,total_mass_kg=625.*h**3*len(x))
    except Exception as e:record.update(error=str(e),traceback=traceback.format_exc())
    if frames:np.savez_compressed(out/'trajectory.npz',positions_m=frames,times_s=times,material_queries=q)
    record.update(elapsed_s=time.monotonic()-t,saved_frames=len(frames));write(out/'complete.json',record);print(json.dumps(record,indent=2));return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',type=int,choices=[0,1],default=0);p.add_argument('--label',default='uniphy-drop-pilot-01');p.add_argument('--dt',type=float,default=50e-6);p.add_argument('--duration',type=float,default=.24);p.add_argument('--seconds',type=int,default=3600);a=p.parse_args();torch.set_num_threads(1);r=run(a);raise SystemExit(0 if r['completed']else 1)
