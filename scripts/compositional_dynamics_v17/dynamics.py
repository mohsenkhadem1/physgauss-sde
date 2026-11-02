"""Consistent-Q2 inertia and velocity Verlet around existing energy gradients.

No learned position correction, restitution reset, friction or artificial drag.
Normal contact remains finite-penalty conservative contact. Its timestep and
surface/volume resolution must be qualified for every declared dynamic regime.
"""
from dataclasses import dataclass,replace
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from scripts.phystwin.tensor_fixture import basis
from scripts.compositional_neural_mechanics.contact_mechanics import ContactQ2Mechanics,RigidSphere,RigidPlane


def consistent_mass(model):
    N,_=basis(model.quad);weights=model.engine.weights.detach().cpu().numpy()
    local=model.density*np.einsum('qe,qa,qb->eab',weights,N,N)
    ids=model.elements;rows=np.broadcast_to(ids[:,:,None],local.shape);cols=np.broadcast_to(ids[:,None,:],local.shape)
    M=sparse.coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(len(model.rest),)*2).tocsc()
    if not np.isfinite(M.data).all()or np.max(abs((M-M.T).data),initial=0)>1e-13:raise ValueError('Invalid symmetric consistent mass')
    if not np.allclose(np.asarray(M.sum(1)).ravel(),model.density*model.nodal_volumes,rtol=1e-11,atol=1e-14):raise ValueError('Inertia/gravity mass mismatch')
    return M


class DynamicContactModel(ContactQ2Mechanics):
    def __init__(self,*args,support_mode='bilateral',**kwargs):
        if support_mode not in ('free','bilateral'):raise ValueError('Explicit support mode required')
        if support_mode=='free'and kwargs.get('interface_energy')is not None:raise ValueError('Free object has no bilateral interface energy')
        super().__init__(*args,**kwargs);self.support_mode=support_mode
        if support_mode=='free':
            zero=sparse.csr_matrix((len(self.rest),)*2)
            self.anchor_matrices=[zero.copy(),zero.copy()];self.fixture_scalar=zero
            self.spring_matrix=sparse.csr_matrix((self.rest.size,)*2);self.set_controls(np.zeros((2,6)))
        self.last_contact_records=[]
    def contact_response(self,*args,**kwargs):
        result=super().contact_response(*args,**kwargs)
        self.last_contact_records=[{k:(v.detach().cpu().numpy()if v.ndim else float(v.detach()))if torch.is_tensor(v)else v for k,v in record.items()}for record in result[3]]
        return result


@dataclass
class SphereBody:
    name:str
    mass:float
    center:np.ndarray
    velocity:np.ndarray
    omega:np.ndarray


@dataclass
class State:
    time:float
    u:np.ndarray
    v:np.ndarray
    spheres:dict
    force:np.ndarray
    body_forces:dict
    body_torques:dict
    potential:float
    diagnostics:dict


class VelocityVerlet:
    def __init__(self,model,bodies=()):
        self.model=model;self.M=consistent_mass(model);self.factor=splu(self.M)
        self.bodies={b.name:b for b in bodies}
        if len(self.bodies)!=len(bodies):raise ValueError('Distinct moving-body identities required')
        self.radii={c.name:c.radius for c in model.colliders if isinstance(c,RigidSphere)}
        for name,b in self.bodies.items():
            if name not in self.radii or not np.isfinite(b.mass)or b.mass<=0:raise ValueError('Finite positive mass and matching sphere collider required')
            for a in (b.center,b.velocity,b.omega):
                if np.asarray(a).shape!=(3,)or not np.isfinite(a).all():raise ValueError('Finite sphere state required')
        self.inertias={name:.4*b.mass*self.radii[name]**2 for name,b in self.bodies.items()}
    def acceleration(self,force):
        a=self.factor.solve(force)
        if not np.isfinite(a).all()or np.linalg.norm(self.M@a-force)>1e-12+1e-10*np.linalg.norm(force):raise FloatingPointError('Unresolved consistent-mass acceleration')
        return a
    def evaluate(self,u,spheres):
        self.model.colliders=tuple(replace(c,center=spheres[c.name]['center'])if c.name in spheres else c for c in self.model.colliders)
        for name,s in spheres.items():
            for c in self.model.colliders:
                if isinstance(c,RigidPlane):
                    n=np.asarray(c.normal,float);n/=np.linalg.norm(n)
                    if (s['center']-np.asarray(c.point))@n<self.radii[name]:raise ValueError('Moving sphere reaches plane: rigid-rigid contact outside this prototype scope')
        out=self.model.evaluate(u);force=-out['residual'];records=self.model.last_contact_records
        bf={name:self.bodies[name].mass*self.model.gravity.copy()for name in spheres};bt={name:np.zeros(3)for name in spheres}
        for r in records:
            if r['name']in spheres:bf[r['name']]+=r['force_on_rigid_N'];bt[r['name']]+=r['torque_on_rigid_Nm']
        potential=out['potential_J']-sum(self.bodies[name].mass*self.model.gravity@s['center']for name,s in spheres.items())
        diag=dict(minimum_J=float(out['J'].min()),elastic_J=out['elastic_J'],contact_J=out['contact_J'],bilateral_J=out['spring_J'],maximum_sampled_penetration_m=out['max_sampled_penetration_m'],contacts=records)
        if not np.isfinite(force).all()or not np.isfinite(potential):raise FloatingPointError('Nonfinite dynamic force/energy')
        return force,bf,bt,float(potential),diag
    def initial_state(self,u=None,v=None):
        shape=self.model.rest.shape;u=np.zeros(shape)if u is None else np.array(u,float,copy=True);v=np.zeros(shape)if v is None else np.array(v,float,copy=True)
        if u.shape!=shape or v.shape!=shape or not np.isfinite(u).all()or not np.isfinite(v).all():raise ValueError('Finite matched soft state required')
        spheres={name:dict(center=np.array(b.center,float,copy=True),velocity=np.array(b.velocity,float,copy=True),omega=np.array(b.omega,float,copy=True))for name,b in self.bodies.items()}
        return State(0.,u,v,spheres,*self.evaluate(u,spheres))
    def momentum(self,s):
        return (self.M@s.v).sum(0)+sum((self.bodies[n].mass*b['velocity']for n,b in s.spheres.items()),np.zeros(3))
    def angular_momentum(self,s):
        L=np.cross(self.model.rest+s.u,self.M@s.v).sum(0)
        return L+sum((np.cross(b['center'],self.bodies[n].mass*b['velocity'])+self.inertias[n]*b['omega']for n,b in s.spheres.items()),np.zeros(3))
    def energy(self,s):
        return s.potential+.5*float(np.sum(s.v*(self.M@s.v)))+sum(.5*self.bodies[n].mass*(b['velocity']@b['velocity'])+.5*self.inertias[n]*(b['omega']@b['omega'])for n,b in s.spheres.items())
    def total_force_torque(self,s):
        f=s.force.sum(0);t=np.cross(self.model.rest+s.u,s.force).sum(0)
        for name,b in s.spheres.items():f+=s.body_forces[name];t+=np.cross(b['center'],s.body_forces[name])+s.body_torques[name]
        return f,t
    def step(self,s,dt):
        if not np.isfinite(dt)or dt<=0:raise ValueError('Finite positive timestep required')
        vh=s.v+.5*dt*self.acceleration(s.force);u=s.u+dt*vh;spheres={}
        for n,b in s.spheres.items():
            vel=b['velocity']+.5*dt*s.body_forces[n]/self.bodies[n].mass;omega=b['omega']+.5*dt*s.body_torques[n]/self.inertias[n]
            spheres[n]=dict(center=b['center']+dt*vel,velocity=vel,omega=omega)
        force,bf,bt,U,diag=self.evaluate(u,spheres);v=vh+.5*dt*self.acceleration(force)
        for n,b in spheres.items():b['velocity']+=.5*dt*bf[n]/self.bodies[n].mass;b['omega']+=.5*dt*bt[n]/self.inertias[n]
        new=State(s.time+dt,u,v,spheres,force,bf,bt,U,diag)
        if not np.isfinite(v).all():raise FloatingPointError('Nonfinite dynamic velocity')
        f0,t0=self.total_force_torque(s);f1,t1=self.total_force_torque(new)
        diag['momentum_step_balance_Ns']=float(np.linalg.norm(self.momentum(new)-self.momentum(s)-.5*dt*(f0+f1)))
        diag['angular_step_balance_Nms']=float(np.linalg.norm(self.angular_momentum(new)-self.angular_momentum(s)-.5*dt*(t0+t1)))
        return new
