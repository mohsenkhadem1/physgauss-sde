"""Spring-Gaus author force/step with explicit common scene boundary adapter.

This is NOT the unmodified end-to-end Spring-Gaus method. Additions: Q2-node
geometry, known lumped masses, common compliant supports/curved normal contact,
and sampled two-object coupling. Native internal forces and semi-implicit Euler
are unchanged; native ground is placed out of scene. Fits must use forward
finite differences because external contact crosses the CPU/NumPy boundary.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,types,hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from yacs.config import CfgNode as CN
from scripts.complex_scenes_v18.native_smoke import load_spring
from scripts.complex_scenes_v18.scenarios import build_case,FAMILIES
from scripts.complex_scenes_v18.pair_contact import SurfaceBinding
from scripts.complex_scenes_v18.run_development import ROOT,write


class SpringScene:
    def __init__(self,case,variant=0,n=1,log10_k=-.3):
        self.scene,initial,self.metadata=build_case(case,n=n,variant=variant)
        self.native=[];self.x=[];self.v=[];self.rigid=initial.rigid;self.time=0.
        Class=load_spring()
        for m,M,u,v in zip(self.scene.models,self.scene.M,initial.u,initial.v):
            mass=np.asarray(M.sum(1)).ravel()
            if not (mass>0).all():raise ValueError('Positive lumped mass required')
            cfg=CN(dict(K_NEIGHBORS=12,K_BINDING=8,N_STEP=1,PRETRAINED=None,INIT_VELOCITY=[0,0,0],G=m.gravity.tolist(),FITTING=True,FIX_MASS=True,FIX_K=True,SINGLE_K=True,DAMPING=True,FIX_DAMP=True,SOFT_K=False,SPRING_BC=False,UNLINEAR_FORCE=True,DATA=dict(DT=.000025,BC=[[[0,0,-100],[0,0,1]]],GLOBAL_K=10**log10_k,GLOBAL_M=float(mass.mean()),GLOBAL_DAMP=.00001)))
            native=Class(cfg,torch.tensor(m.rest,dtype=torch.float32,device='cuda'),init_velocity=[0,0,0]).cuda()
            native.global_m.data.copy_(torch.tensor(np.log10(mass),dtype=torch.float32,device='cuda'))
            native.scene_external_force=torch.zeros_like(native.init_xyz)
            original=native.compute_force
            def combined(this,xyz,v,K,damp,original=original):
                return original(xyz,v,K,damp)+this.scene_external_force
            native.compute_force=types.MethodType(combined,native)
            self.native.append(native);self.x.append(torch.tensor(m.rest+u,dtype=torch.float32,device='cuda'));self.v.append(torch.tensor(v,dtype=torch.float32,device='cuda'))
        self.sample=[SurfaceBinding.build(m,9)for m in self.scene.models]

    def external(self):
        forces=[];rf={k:b.mass*self.scene.models[0].gravity.copy()for k,b in self.scene.rigid.items()};depth=0.
        coords=[x.detach().cpu().numpy().astype(float)for x in self.x]
        for i,(m,x)in enumerate(zip(self.scene.models,coords)):
            self.scene.controls(i,self.time);u=x-m.rest
            m.colliders=tuple(replace(c,center=self.rigid[c.name]['center'])if c.name in self.rigid else c for c in m.colliders)
            _,residual,_,records=m.contact_response(m.engine.tensor(u))
            force=-residual.detach().cpu().numpy()
            for A,target in zip(m.anchor_matrices,m.anchor_targets):force-=A@(u-target)
            for r in records:
                depth=max(depth,r['max_sampled_penetration_m'])
                if r['name']in rf:rf[r['name']]+=r['force_on_rigid_N'].detach().cpu().numpy()
            forces.append(force)
        if self.scene.pair is not None:
            _,pair,diag=self.scene.pair.evaluate(coords);forces=[a+b for a,b in zip(forces,pair)];depth=max(depth,diag['max_sampled_penetration_m'])
        return forces,rf,depth

    def step(self,dt):
        # Autograd is used internally by the common contact operator.
        forces,rf,depth=self.external()
        with torch.no_grad():
            for i,(model,f)in enumerate(zip(self.native,forces)):
                model.scene_external_force=torch.tensor(f,dtype=torch.float32,device='cuda')
                K=10**model.global_k.expand_as(model.origin_len)/(model.origin_len+model.eps)
                damp=10**model.damp/(model.origin_len+model.eps)
                self.x[i],self.v[i]=model.step(self.x[i],self.v[i],K,10**model.global_m,torch.sigmoid(model.rebound_k),torch.sigmoid(model.fric_k),damp,dt)
                if not torch.isfinite(self.x[i]).all()or self.x[i].abs().max()>10:raise FloatingPointError('Native spring trajectory diverged')
        for k,b in self.rigid.items():
            b['velocity']+=dt*rf[k]/self.scene.rigid[k].mass;b['center']+=dt*b['velocity']
        self.time+=dt;return depth

    def sample_surface(self):return np.stack([b.B@x.detach().cpu().numpy()for b,x in zip(self.sample,self.x)])


def run(args):
    out=ROOT/'runs/complex-scenes-v18'/args.label/f'{args.case}-v{args.variant}'
    if out.exists():raise ValueError('Preserve prior attempt '+str(out))
    out.mkdir(parents=True);start=time.monotonic();record=dict(method='Spring-Gaus author internal forces/integrator with common scene boundary adapter',development_only=True,completed=False,parameters=dict(log10_k=args.log10_k),case=args.case,variant=args.variant,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    frames=[];times=[];depth=0.
    try:
        sim=SpringScene(args.case,args.variant,log10_k=args.log10_k);duration=min(args.duration,sim.metadata['duration_s'])if args.duration else sim.metadata['duration_s'];steps=round(duration/args.dt);stride=round(.002/args.dt)
        if not np.isclose(stride*args.dt,.002):raise ValueError('Sample clock')
        frames.append(sim.sample_surface());times.append(0.)
        for i in range(1,steps+1):
            if time.monotonic()-start>args.seconds:raise TimeoutError('Explicit rollout budget')
            depth=max(depth,sim.step(args.dt))
            if i%stride==0:frames.append(sim.sample_surface());times.append(sim.time)
            if i%200==0:write(out/'status.json',dict(record,steps=i,planned_steps=steps,elapsed_s=time.monotonic()-start,pid=os.getpid()))
        record.update(completed=True,steps=steps,dt_s=args.dt,maximum_penetration_m=depth)
    except Exception as e:record['error']=f'{type(e).__name__}: {e}'
    if frames:np.savez_compressed(out/'trajectory.npz',surface_positions_m=np.asarray(frames),times_s=np.asarray(times))
    record['elapsed_s']=time.monotonic()-start;write(out/'complete.json',record);print(json.dumps(record,indent=2));return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=FAMILIES,required=True);p.add_argument('--variant',type=int,default=0);p.add_argument('--label',default='spring-adapter-pilot');p.add_argument('--log10-k',type=float,default=-.3);p.add_argument('--dt',type=float,default=25e-6);p.add_argument('--duration',type=float);p.add_argument('--seconds',type=int,default=1800);a=p.parse_args();torch.set_num_threads(1);r=run(a);raise SystemExit(0 if r['completed']else 1)
