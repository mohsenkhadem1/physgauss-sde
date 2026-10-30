"""Rigid-object comparator using precisely the v13 nodal contact interface.

No deformation, no force/material labels, no observation of forecast geometry.
The mass and inertia come from the same initial tetrahedral lumped masses as FEM.
Only free rigid objects are supported; fixed-node fixtures are rejected explicitly.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .physics import FEMSystem


class RigidSystem(FEMSystem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.fixed.any():raise ValueError('Rigid baseline requires free object; fixed-node fixture not implemented')
        self.total_mass=float(self.mass.sum())
        self.center=np.average(self.rest,axis=0,weights=self.mass)
        self.local=self.rest-self.center
        self.rotation=np.eye(3)
        self.body_inertia=sum(m*(np.dot(r,r)*np.eye(3)-np.outer(r,r)) for m,r in zip(self.mass,self.local))
        self.last_fixed_reaction=np.zeros(3)

    def internal_forces(self):
        return np.zeros_like(self.x),np.linalg.det(self.deformation())

    def elastic_energy(self,x=None):return 0.

    def step(self,dt,tool_states=None):
        if dt<=0:raise ValueError('Positive timestep required')
        for c in self.colliders:
            if tool_states and c.name in tool_states:
                if np.isfinite(c.mass):raise ValueError('Cannot prescribe a dynamic body')
                for key,value in tool_states[c.name].items():
                    if key not in ('center','velocity','rotation','omega'):raise ValueError(key)
                    setattr(c,key,np.asarray(value,float).copy())
        # Extract momentum from public nodal state, allowing an explicitly known
        # initial velocity. Internal shape/velocity components are projected out.
        center=np.average(self.x,axis=0,weights=self.mass)
        arm=self.x-center
        velocity=np.average(self.v,axis=0,weights=self.mass)
        angular=np.cross(arm,self.mass[:,None]*(self.v-velocity)).sum(0)
        force,reactions,torques,depth,power,count=self.contact_forces()
        rf,rt,rd,rp,rc=self.rigid_contact_forces()
        reactions=[a+b for a,b in zip(reactions,rf)];torques=[a+b for a,b in zip(torques,rt)]
        depth=max(depth,rd);power+=rp;count+=rc
        velocity+=dt*(force.sum(0)/self.total_mass+self.gravity)
        angular+=dt*np.cross(arm,force).sum(0)
        inertia=self.rotation.T@self.body_inertia@self.rotation
        omega=np.linalg.solve(inertia,angular)
        self.rotation=self.rotation@Rotation.from_rotvec(omega*dt).as_matrix().T
        self.center=center+dt*velocity
        self.x=self.local@self.rotation+self.center
        # Recompute omega at updated orientation so world angular momentum is
        # retained despite a changing inertia tensor.
        inertia=self.rotation.T@self.body_inertia@self.rotation
        omega=np.linalg.solve(inertia,angular)
        self.v=velocity+np.cross(omega,self.x-self.center)
        for c,reaction,torque in zip(self.colliders,reactions,torques):
            if np.isfinite(c.mass):
                c.velocity+=dt*(reaction/c.mass+self.gravity);c.omega+=dt*torque/c.inertia;c.center+=dt*c.velocity
        J=np.linalg.det(self.deformation())
        if not np.isfinite(self.x).all() or not np.isfinite(self.v).all():raise FloatingPointError('Nonfinite rigid state')
        self.min_j=min(self.min_j,float(J.min()));self.max_penetration=max(self.max_penetration,depth)
        self.friction_work+=power*dt;self.contact_samples+=count;self.time+=dt
        self.last_reactions={c.name:r.copy() for c,r in zip(self.colliders,reactions)}
        self.last_torques={c.name:r.copy() for c,r in zip(self.colliders,torques)}
        return dict(min_j=float(J.min()),max_penetration=depth,friction_power=power,contacts=count)

    def metadata(self):
        return dict(label='Rigid object + shared v13 contact',deformation=False,
                    mass='same SI lumped tetrahedral mass and resulting inertia as FEM',
                    contact='inherited identical nodal contact and dynamic collider coupling',
                    calibration='friction from calibration positions only; no elasticity parameter',
                    limits='free rigid object; fixed-node constraints unsupported; first-order orientation integration')
