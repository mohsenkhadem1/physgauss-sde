"""Float64 batched implementation of the qualified NumPy bowl equations.

Acceleration only: same mesh, material, mass, quadrature, contact and timestep.
Parity must pass before use. Requires explicit GPU assignment by caller.
"""
import numpy as np
import torch
import json,hashlib,time
from pathlib import Path
from .bowl_physics import build,command

class Batch:
    def __init__(self,theta,n=3,penalty=38400.,subdivisions=2,device='cuda:0'):
        self.base=build(n=n,penalty=penalty,subdivisions=subdivisions)
        self.theta=np.asarray(theta,float)
        self.configuration=json.dumps(dict(n=n,penalty=penalty,subdivisions=subdivisions),sort_keys=True)
        self.device=device;self.dtype=torch.float64
        self.tensor=lambda x:torch.as_tensor(x,dtype=self.dtype,device=device)
        self.index=lambda x:torch.as_tensor(x,dtype=torch.int64,device=device)
        s=self.base;self.H=len(theta)
        self.x=self.tensor(s.rest).expand(self.H,-1,-1).clone();self.v=torch.zeros_like(self.x)
        self.tets=self.index(s.tets);self.invD=self.tensor(s.invD);self.volume=self.tensor(s.volume)
        self.mass=self.tensor(s.mass);self.qids=self.index(s.surface_node_ids);self.qw=self.tensor(s.surface_barycentric)
        self.stiffness=self.tensor(penalty*s.surface_patch_stiffness_weights)
        self.damping=self.tensor(np.sqrt(penalty*s.surface_patch_stiffness_weights*s.surface_patch_mass))
        e=self.tensor(np.asarray(theta)[:,0]);self.mu=e/(2*1.3);self.lam=e*.3/(1.3*.4)
        self.friction=self.tensor(np.asarray(theta)[:,1]);self.gravity=self.tensor([0,0,-9.81])
        self.tet_scatter=self.tets.flatten()[None,:,None].expand(self.H,-1,3)
        self.q_scatter=self.qids.flatten()[None,:,None].expand(self.H,-1,3)
        self.minimum=torch.ones(self.H,dtype=self.dtype,device=device)
        self.maximum=torch.zeros_like(self.minimum);self.work=torch.zeros_like(self.minimum)

    def deformation(self):
        return (self.x[:,self.tets[:,1:]]-self.x[:,self.tets[:,0,None]]).transpose(-1,-2)@self.invD

    def internal(self):
        F=self.deformation();J=torch.linalg.det(F);inverse=torch.linalg.inv(F)
        self.minimum=torch.minimum(self.minimum,J.amin(1))
        P=self.mu[:,None,None,None]*F+(self.lam[:,None]*torch.log(J)-self.mu[:,None])[:,:,None,None]*inverse.transpose(-1,-2)
        dv=(self.v[:,self.tets[:,1:]]-self.v[:,self.tets[:,0,None]]).transpose(-1,-2)
        L=dv@self.invD@inverse;rate=.5*(L+L.transpose(-1,-2))
        P=P+4*J[:,:,None,None]*(rate@inverse.transpose(-1,-2))
        columns=-self.volume[None,:,None,None]*(P@self.invD.transpose(-1,-2))
        nodal=torch.cat((-columns.sum(-1)[:,:,None,:],columns.transpose(-1,-2)),dim=2)
        force=torch.zeros_like(self.x).scatter_add_(1,self.tet_scatter,nodal.reshape(self.H,-1,3))
        return force,J

    def contact(self,t,interaction):
        points=(self.x[:,self.qids]*self.qw[None,:,:,None]).sum(2)
        velocity=(self.v[:,self.qids]*self.qw[None,:,:,None]).sum(2)
        states=command(t,interaction);total=torch.zeros_like(points);reactions=[];centers=[]
        for c in self.base.colliders:
            state=states.get(c.name,{})
            center=self.tensor(state.get('center',c.center));cv=self.tensor(state.get('velocity',c.velocity))
            omega=self.tensor(state.get('omega',c.omega));rotation=self.tensor(state.get('rotation',c.rotation))
            arm=points-center;length=torch.linalg.vector_norm(arm,dim=-1)
            normal=arm/torch.clamp(length[:,:,None],min=1e-15)
            if c.kind=='bowl':
                phi=c.radius-length;normal=-normal
                phi=torch.where((arm@rotation)[:,:,2]<=0,phi,torch.inf)
            else:phi=length-c.radius
            depth=torch.clamp(-phi,min=0);active=phi<0
            relative=velocity-cv-torch.linalg.cross(omega.expand_as(arm),arm,dim=-1)
            vn=(relative*normal).sum(-1)
            magnitude=torch.clamp(self.stiffness*depth-self.damping*vn,min=0)*active
            tangent=relative-vn[:,:,None]*normal;speed=torch.linalg.vector_norm(tangent,dim=-1)
            tangential=-self.friction[:,None,None]*magnitude[:,:,None]*tangent/torch.clamp(speed[:,:,None],min=.01)
            cf=magnitude[:,:,None]*normal+tangential;total+=cf
            reactions.append(-cf.sum(1));centers.append(center)
            self.maximum=torch.maximum(self.maximum,depth.amax(1))
            self.power=(tangential*relative).sum((1,2)) if len(reactions)==1 else self.power+(tangential*relative).sum((1,2))
        nodal=self.qw[None,:,:,None]*total[:,:,None,:]
        force=torch.zeros_like(self.x).scatter_add_(1,self.q_scatter,nodal.reshape(self.H,-1,3))
        return force,torch.stack(reactions,1),torch.stack(centers)

    @torch.no_grad()
    def step(self,t,dt,interaction):
        force,J=self.internal();contact,reactions,centers=self.contact(t,interaction)
        self.v+=dt*((force+contact)/self.mass[None,:,None]+self.gravity)
        self.x+=dt*self.v;self.work+=self.power*dt
        return reactions,centers

    @torch.no_grad()
    def run(self,duration=1.3,dt=5e-5,interaction='heldout_push',progress_path=None,checkpoint_path=None,deadline_unix=None):
        for collider in self.base.colliders:
            for key,value in command(0,interaction).get(collider.name,{}).items():setattr(collider,key,np.asarray(value).copy())
        steps=int(np.ceil(duration/dt));dt=duration/steps
        save_times=np.linspace(0,duration,int(round(duration*50))+1)
        frames=[self.x.cpu().numpy().copy()];velocities=[self.v.cpu().numpy().copy()]
        centers=[np.array([c.center for c in self.base.colliders])]
        reactions=[np.zeros((self.H,2,3))];times=[0.];next_save=1
        reaction_averages=[np.zeros((self.H,2,3))]
        impulse=torch.zeros((self.H,2,3),dtype=self.dtype,device=self.device)
        start_step=0
        signature=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('bowl_physics.py').read_bytes()).hexdigest()
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            with np.load(checkpoint_path) as saved:
                if str(saved['signature'])!=signature or str(saved['configuration'])!=self.configuration or int(saved['steps'])!=steps or str(saved['interaction'])!=interaction or float(saved['dt'])!=dt or not np.array_equal(saved['theta'],self.theta):raise RuntimeError('Checkpoint differs from frozen engine/parameters')
                self.x=self.tensor(saved['x']);self.v=self.tensor(saved['v']);self.minimum=self.tensor(saved['minimum']);self.maximum=self.tensor(saved['maximum']);self.work=self.tensor(saved['work'])
                impulse=self.tensor(saved['impulse'])
                frames=list(saved['frames']);velocities=list(saved['velocities']);centers=list(saved['centers']);reactions=list(saved['reactions']);reaction_averages=list(saved['reaction_averages']);times=list(saved['times'])
                start_step=int(saved['next_step']);next_save=len(times)
        def save_checkpoint(next_step):
            if checkpoint_path is None:return
            path=Path(checkpoint_path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,signature=signature,configuration=self.configuration,steps=steps,interaction=interaction,dt=dt,theta=self.theta,impulse=impulse.cpu().numpy(),
                x=self.x.cpu().numpy(),v=self.v.cpu().numpy(),minimum=self.minimum.cpu().numpy(),maximum=self.maximum.cpu().numpy(),work=self.work.cpu().numpy(),
                frames=np.asarray(frames),velocities=np.asarray(velocities),centers=np.asarray(centers),reactions=np.asarray(reactions),reaction_averages=np.asarray(reaction_averages),times=np.asarray(times),next_step=next_step)
            temporary.replace(path)
        for k in range(start_step,steps):
            if deadline_unix is not None and k%100==0 and time.time()>=deadline_unix:
                save_checkpoint(k);raise RuntimeError('Campaign deadline reached; saved resumable checkpoint')
            reaction,center=self.step(k*dt,dt,interaction)
            self.time=(k+1)*dt
            impulse+=reaction*dt
            if next_save<len(save_times) and (k+1)*dt>=save_times[next_save]-1e-12:
                if progress_path is not None and next_save%5==0:
                    Path(progress_path).write_text(json.dumps(dict(time_s=self.time,min_j=self.minimum.cpu().tolist(),max_penetration_m=self.maximum.cpu().tolist())))
                if not torch.isfinite(self.x).all() or self.minimum.min()<=.02:raise FloatingPointError(f'Batched simulation invalid at {self.time:.7g}s; minJ={self.minimum.cpu().tolist()}')
                frames.append(self.x.cpu().numpy().copy());velocities.append(self.v.cpu().numpy().copy())
                centers.append(center.cpu().numpy());reactions.append(reaction.cpu().numpy())
                reaction_averages.append((impulse/((k+1)*dt-times[-1])).cpu().numpy());impulse.zero_()
                times.append((k+1)*dt);next_save+=1
                if next_save%10==0 or k+1==steps:save_checkpoint(k+1)
                if deadline_unix is not None and time.time()>=deadline_unix:
                    save_checkpoint(k+1);raise RuntimeError('Campaign deadline reached; saved resumable checkpoint')
        self.minimum=torch.minimum(self.minimum,torch.linalg.det(self.deformation()).amin(1))
        if not torch.isfinite(self.x).all() or self.minimum.min()<=.02:raise FloatingPointError('Invalid final batched state')
        return dict(frames=np.asarray(frames).swapaxes(0,1),velocities=np.asarray(velocities).swapaxes(0,1),
            centers=np.asarray(centers),times=np.asarray(times),reactions=np.asarray(reactions).swapaxes(0,1),
            reaction_averages=np.asarray(reaction_averages).swapaxes(0,1),
            min_j=self.minimum.cpu().numpy(),max_penetration_m=self.maximum.cpu().numpy(),friction_work_j=self.work.cpu().numpy(),dt=dt)
