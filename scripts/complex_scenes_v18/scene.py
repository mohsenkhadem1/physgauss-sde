"""Coupled conservative mechanics with independent soft-object identities.

External actuator power is computed from the same attachment energy. Smooth
attachment release accounts for the explicit change of attachment energy.
"""
from dataclasses import dataclass,replace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scripts.compositional_dynamics_v17.dynamics import consistent_mass
from scripts.compositional_neural_mechanics.contact_mechanics import RigidSphere


@dataclass
class SceneState:
    time: float
    u: list
    v: list
    rigid: dict
    forces: list
    rigid_forces: dict
    potential: float
    power: float
    diagnostics: dict
    work: float=0.


class CoupledScene:
    def __init__(self,models,*,pair_contact=None,rigid_bodies=(),motions=None):
        if not models:raise ValueError('At least one soft object')
        self.models=models;self.M=[consistent_mass(m)for m in models];self.factor=[splu(M)for M in self.M]
        self.pair=pair_contact;self.rigid={b.name:b for b in rigid_bodies}
        self.motions=motions or [None]*len(models)
        self.anchor_base=[[a.copy()for a in m.anchor_matrices]for m in models]
        if len(self.rigid)!=len(rigid_bodies)or len(self.motions)!=len(models):raise ValueError('Scene identity/shape mismatch')
        for b in rigid_bodies:
            if b.mass<=0 or not np.isfinite(b.mass):raise ValueError('Finite positive rigid mass required')
            if not any(isinstance(c,RigidSphere)and c.name==b.name for m in models for c in m.colliders):raise ValueError('Missing rigid collision geometry')
        if any(np.any(m.gravity!=models[0].gravity)for m in models):raise ValueError('Common world gravity required')

    def controls(self,i,t):
        m=self.models[i];motion=self.motions[i]
        if motion is None:return None
        poses,velocity,gates,rates=motion(t)
        if poses.shape!=(2,6)or velocity.shape!=(2,3)or np.any(poses[:,3:]):raise ValueError('Explicit translation-only actuator path required')
        if np.min(gates)<0 or not np.isfinite(np.r_[poses.ravel(),velocity.ravel(),gates,rates]).all():raise ValueError('Invalid actuator state')
        m.anchor_matrices=[g*A for g,A in zip(gates,self.anchor_base[i])]
        m.fixture_scalar=sum(m.anchor_matrices);m.spring_matrix=sparse.kron(m.fixture_scalar,sparse.eye(3),format='csr')
        m.set_controls(poses)
        return velocity,rates

    def evaluate(self,u,rigid,t):
        forces=[];potential=0.;power=0.;details=[]
        rf={name:b.mass*self.models[0].gravity.copy()for name,b in self.rigid.items()}
        for i,(m,ui)in enumerate(zip(self.models,u)):
            ctrl=self.controls(i,t)
            m.colliders=tuple(replace(c,center=rigid[c.name]['center'])if c.name in rigid else c for c in m.colliders)
            out=m.evaluate(ui);forces.append(-out['residual']);potential+=out['potential_J']
            if ctrl is not None:
                velocity,rates=ctrl
                for j,(A,base,target)in enumerate(zip(m.anchor_matrices,self.anchor_base[i],m.anchor_targets)):
                    delta=ui-target;reaction=A@delta
                    power-=float(reaction.sum(0)@velocity[j]);power+=.5*rates[j]*float(np.sum(delta*(base@delta)))
            contacts=m.last_contact_records
            for c in contacts:
                if c['name']in rf:rf[c['name']]+=c['force_on_rigid_N']
            details.append(dict(minimum_J=float(out['J'].min()),elastic_J=out['elastic_J'],attachment_J=out['spring_J'],contact_J=out['contact_J'],max_penetration_m=out['max_sampled_penetration_m'],contacts=contacts))
        pairdiag=None
        if self.pair is not None:
            U,F,pairdiag=self.pair.evaluate([m.rest+ui for m,ui in zip(self.models,u)])
            potential+=U;forces=[a+b for a,b in zip(forces,F)]
        potential-=sum(self.rigid[n].mass*self.models[0].gravity@s['center']for n,s in rigid.items())
        if not np.isfinite(potential)or any(not np.isfinite(f).all()for f in forces):raise FloatingPointError('Nonfinite scene response')
        return forces,rf,float(potential),float(power),dict(bodies=details,pair=pairdiag)

    def initial(self,u=None,v=None):
        u=[np.zeros_like(m.rest)for m in self.models]if u is None else [np.array(a,float,copy=True)for a in u]
        v=[np.zeros_like(m.rest)for m in self.models]if v is None else [np.array(a,float,copy=True)for a in v]
        rigid={n:dict(center=np.array(b.center,float,copy=True),velocity=np.array(b.velocity,float,copy=True))for n,b in self.rigid.items()}
        if len(u)!=len(self.models)or len(v)!=len(u)or any(a.shape!=m.rest.shape or b.shape!=m.rest.shape for a,b,m in zip(u,v,self.models)):raise ValueError('State object/coordinate mismatch')
        return SceneState(0.,u,v,rigid,*self.evaluate(u,rigid,0.))

    def energy(self,s):
        return s.potential+.5*sum(float(np.sum(v*(M@v)))for v,M in zip(s.v,self.M))+.5*sum(self.rigid[n].mass*float(b['velocity']@b['velocity'])for n,b in s.rigid.items())

    def momentum(self,s):
        return sum(((M@v).sum(0)for M,v in zip(self.M,s.v)),np.zeros(3))+sum((self.rigid[n].mass*b['velocity']for n,b in s.rigid.items()),np.zeros(3))

    def step(self,s,dt):
        if not np.isfinite(dt)or dt<=0:raise ValueError('Positive timestep required')
        vh=[v+.5*dt*factor.solve(f)for v,factor,f in zip(s.v,self.factor,s.forces)]
        u=[a+dt*b for a,b in zip(s.u,vh)];rigid={}
        for n,b in s.rigid.items():
            v=b['velocity']+.5*dt*s.rigid_forces[n]/self.rigid[n].mass
            rigid[n]=dict(center=b['center']+dt*v,velocity=v)
        try:force,rf,U,power,diag=self.evaluate(u,rigid,s.time+dt)
        except Exception as error:
            error.scene_candidate=dict(u=u,rigid=rigid,time=s.time+dt)
            raise
        v=[a+.5*dt*factor.solve(f)for a,factor,f in zip(vh,self.factor,force)]
        for n,b in rigid.items():b['velocity']+=.5*dt*rf[n]/self.rigid[n].mass
        return SceneState(s.time+dt,u,v,rigid,force,rf,U,power,diag,s.work+.5*dt*(s.power+power))
