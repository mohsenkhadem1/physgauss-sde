"""Author PhysTwin Warp springs/integration with a common boundary adapter.

Not the unmodified end-to-end method: deterministic geometry KNN, known lumped
masses, compliant fixtures and curved/pair contact are explicit adaptations.
Native ground is translated outside the scene; touching it is a failure.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ast,hashlib,json,sys,time,types
import numpy as np
import torch
import warp as wp
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.scenarios import build_case,FAMILIES
from scripts.complex_scenes_v18.pair_contact import SurfaceBinding
from scripts.complex_scenes_v18.spring_scene import SpringScene

AUTHOR=ROOT/'runs/compositional-dynamics-v17/PhysTwin-author'


def kernels():
    # Load exact author kernel definitions without GUI/keyboard dependencies.
    name='v18_author_phystwin_kernels'
    if name in sys.modules:return sys.modules[name]
    path=AUTHOR/'qqtt/model/diff_simulator/spring_mass_warp.py'
    selected=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in ('eval_springs','update_vel_from_force','integrate_ground_collision')]
    if len(selected)!=3:raise ValueError('Pinned native kernel definitions missing')
    wp.init();module=types.ModuleType(name);module.__file__=str(path);module.wp=wp;sys.modules[name]=module
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module


def graph(rest,k=12):
    rest=np.asarray(rest,dtype=float)
    if not 0<k<len(rest):raise ValueError('Neighbor count must be below node count')
    distance=((rest[:,None]-rest[None,:])**2).sum(-1)
    np.fill_diagonal(distance,np.inf)
    nearest=np.argsort(distance,axis=1,kind='stable')[:,:k]
    edges=np.unique(np.sort(np.stack([np.repeat(np.arange(len(rest)),k),nearest.ravel()],1),axis=1),axis=0)
    return edges.astype(np.int32)


class PhysTwinScene:
    external=SpringScene.external
    sample_surface=SpringScene.sample_surface

    def __init__(self,case,variant=0,log10_k=-.3,query=None):
        self.kernels=kernels()
        if query is None:self.scene,state,self.metadata=build_case(case,variant=variant)
        else:
            from scripts.complex_scenes_v18.world_queries import build_boundary_geometry
            if query.family!=case or query.variant!=variant:raise ValueError('Query identity mismatch')
            self.scene,state,self.metadata=build_boundary_geometry(query)
        self.time=0.;self.rigid=state.rigid;self.x=[];self.v=[];self.buffers=[];self.offset=np.array([0,0,1.])
        self.sample=[SurfaceBinding.build(m,9)for m in self.scene.models]
        for m,M,u,v in zip(self.scene.models,self.scene.M,state.u,state.v):
            mass=np.asarray(M.sum(1)).ravel();edges=graph(m.rest)
            def arr(a,dtype=wp.float32):return wp.array(np.asarray(a),dtype=dtype,device='cuda:0')
            b=dict(x=arr(m.rest+u+self.offset,wp.vec3),v=arr(v,wp.vec3),edges=arr(edges,wp.vec2i),mass=arr(mass),mass_numpy=mass,
                   rest=arr(np.linalg.norm(m.rest[edges[:,1]]-m.rest[edges[:,0]],axis=-1)),Y=arr(np.full(len(edges),np.log(10**log10_k))),
                   force=wp.zeros(len(m.rest),dtype=wp.vec3,device='cuda:0'),empty=wp.zeros(0,dtype=wp.vec3,device='cuda:0'),
                   zero=arr([0.]),edges_numpy=edges)
            for name in ('v_tmp','x_next','v_next'):b[name]=wp.zeros(len(m.rest),dtype=wp.vec3,device='cuda:0')
            self.buffers.append(b);self.x.append(torch.tensor(m.rest+u,dtype=torch.float32));self.v.append(torch.tensor(v,dtype=torch.float32))

    def step(self,dt):
        forces,rf,depth=self.external()
        for i,(m,b,force)in enumerate(zip(self.scene.models,self.buffers,forces)):
            # Compensate native hardcoded gravity; requested scene gravity is public.
            force=force+b['mass_numpy'][:,None]*(m.gravity+np.array([0,0,9.8]))
            b['force'].assign(force.astype(np.float32))
            wp.launch(self.kernels.eval_springs,dim=len(b['edges_numpy']),inputs=[b['x'],b['v'],b['empty'],b['empty'],len(m.rest),b['edges'],b['rest'],b['Y'],0.,0.,1e8,b['force']],device='cuda:0')
            wp.launch(self.kernels.update_vel_from_force,dim=len(m.rest),inputs=[b['v'],b['force'],b['mass'],dt,0.,1.,b['v_tmp']],device='cuda:0')
            trial=b['x'].numpy()+dt*b['v_tmp'].numpy()
            if not np.isfinite(trial).all()or trial[:,2].min()<=0 or abs(trial).max()>10:raise FloatingPointError('Divergence or translated native floor reached')
            wp.launch(self.kernels.integrate_ground_collision,dim=len(m.rest),inputs=[b['x'],b['v_tmp'],b['zero'],b['zero'],dt,1.,b['x_next'],b['v_next']],device='cuda:0')
            b['x'],b['x_next']=b['x_next'],b['x'];b['v'],b['v_next']=b['v_next'],b['v']
            self.x[i]=torch.from_numpy(b['x'].numpy()-self.offset);self.v[i]=torch.from_numpy(b['v'].numpy())
        for name,b in self.rigid.items():
            b['velocity']+=dt*rf[name]/self.scene.rigid[name].mass;b['center']+=dt*b['velocity']
        self.time+=dt;return depth


def run(args):
    out=ROOT/'runs/complex-scenes-v18'/args.label/f'{args.case}-v{args.variant}';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();frames=[];times=[];record=dict(completed=False,development_only=True,method='PhysTwin author Warp internal springs and integrator with common boundary adapter',case=args.case,variant=args.variant,log10_k=args.log10_k,author_commit='81c718790a37e5e0102eb77af2c6edd34a9db25f',source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())
    try:
        sim=PhysTwinScene(args.case,args.variant,args.log10_k);duration=min(args.duration,sim.metadata['duration_s'])if args.duration else sim.metadata['duration_s']
        steps=round(duration/args.dt);stride=round(.002/args.dt);depth=0.
        if not np.isclose(stride*args.dt,.002):raise ValueError('Sample clock')
        np.savez_compressed(out/'graph.npz',**{f'edges_{i}':b['edges_numpy']for i,b in enumerate(sim.buffers)})
        for step in range(steps+1):
            if time.monotonic()-start>args.seconds:raise TimeoutError('Declared runtime budget')
            if step:depth=max(depth,sim.step(args.dt))
            if step%stride==0:frames.append(sim.sample_surface());times.append(sim.time)
            if step%200==0:write(out/'status.json',dict(record,pid=os.getpid(),steps=step,planned_steps=steps,elapsed_s=time.monotonic()-start))
        record.update(completed=True,maximum_penetration_m=depth)
    except Exception as e:record['error']=f'{type(e).__name__}: {e}'
    if frames:np.savez_compressed(out/'trajectory.npz',surface_positions_m=frames,times_s=times)
    record['elapsed_s']=time.monotonic()-start;write(out/'complete.json',record);print(json.dumps(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=FAMILIES,required=True);p.add_argument('--variant',type=int,default=0);p.add_argument('--label',default='phystwin-adapter-pilot');p.add_argument('--log10-k',type=float,default=-.3);p.add_argument('--dt',type=float,default=25e-6);p.add_argument('--duration',type=float);p.add_argument('--seconds',type=int,default=1800)
    a=p.parse_args();torch.set_num_threads(1);r=run(a);raise SystemExit(0 if r['completed']else 1)
