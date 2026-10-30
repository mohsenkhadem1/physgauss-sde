"""Development candidate: consistent P1 inertia with unchanged contact laws.

Scalar tetrahedral mass rho*V/20*(I+11^T), applied independently to x/y/z.
Factor the fixed free-node block once. Static Dirichlet nodes have zero
acceleration; their reactions include off-diagonal inertial coupling.
self.mass remains the exact row sum for inherited contact damping/gravity inputs.
No frozen physics code is modified. Smaller explicit stable steps may be needed.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from .physics import FEMSystem
from .rigid import RigidSystem
from .clipped_surface_contact import ClippedSurfaceFEMSystem


def assemble_mass(rest,tets,density):
    rest=np.asarray(rest,float);tets=np.asarray(tets,int)
    D=np.transpose(rest[tets[:,1:]]-rest[tets[:,0,None]],(0,2,1));volume=np.linalg.det(D)/6
    if density<=0 or np.any(volume<=0):raise ValueError('Positive density and oriented tetrahedra required')
    local=(density*volume/20)[:,None,None]*(np.eye(4)+np.ones((4,4)))
    rows=np.broadcast_to(tets[:,:,None],local.shape);columns=np.broadcast_to(tets[:,None,:],local.shape)
    return coo_matrix((local.ravel(),(rows.ravel(),columns.ravel())),shape=(len(rest),len(rest))).tocsc()


class ConsistentInertia:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mass_matrix=assemble_mass(self.rest,self.tets,self.material.density)
        self.mass_row_sum=np.asarray(self.mass_matrix.sum(axis=1)).ravel()
        if not np.allclose(self.mass_row_sum,self.mass,rtol=1e-12,atol=1e-15):raise ValueError('Consistent mass row sums differ from shared contact masses')
        self.free_nodes=np.flatnonzero(~self.fixed)
        self._mass_factor=splu(self.mass_matrix[self.free_nodes][:,self.free_nodes].tocsc()) if len(self.free_nodes) else None
        self.last_constraint_reactions=np.zeros_like(self.x)

    def mass_acceleration(self,force):
        force=np.asarray(force,float)
        if force.shape!=self.x.shape:raise ValueError('Force must be [N,3]')
        acc=np.zeros_like(force)
        if self._mass_factor is not None:acc[self.free_nodes]=self._mass_factor.solve(force[self.free_nodes])
        return acc

    def momentum(self):
        p=(self.mass_matrix@self.v).sum(0)
        for c in self.colliders:
            if np.isfinite(c.mass):p+=c.mass*c.velocity
        return p

    def angular_momentum(self):
        p=np.cross(self.x,self.mass_matrix@self.v).sum(0)
        for c in self.colliders:
            if np.isfinite(c.mass):p+=np.cross(c.center,c.mass*c.velocity)+c.inertia*c.omega
        return p

    def kinetic_energy(self):
        energy=.5*float(np.sum(self.v*(self.mass_matrix@self.v)))
        for c in self.colliders:
            if np.isfinite(c.mass):energy+=.5*c.mass*float(c.velocity@c.velocity)+.5*c.inertia*float(c.omega@c.omega)
        return energy

    def mass_metadata(self):
        return dict(type='Consistent scalar P1 tetrahedral mass',formula='rho*V/20*(I+ones)',
                    solver='one sparse LU factorization of static free-node M_ff, reused for three Cartesian RHS',
                    constraints='static fixed nodes; zero velocity/acceleration; inertially coupled reactions',
                    contact='unchanged inherited forces and damping; self.mass remains original row sum',
                    timestep='explicit symplectic Euler; stability must be requalified')

    def metadata(self):
        inherited=getattr(super(),'metadata',None)
        data=inherited() if inherited is not None else {}
        data.update(mass='consistent P1 tetrahedral inertia; unchanged row sums only for inherited contact damping',mass_discretization=self.mass_metadata())
        if isinstance(self,RigidSystem):data['rigid_inertia']='exact continuum volume integral of uniform tetrahedral body'
        return data

    def _prescribe(self,tool_states):
        for c in self.colliders:
            if tool_states and c.name in tool_states:
                if np.isfinite(c.mass):raise ValueError('Cannot prescribe a dynamic body')
                for key,value in tool_states[c.name].items():
                    if key not in ('center','velocity','rotation','omega'):raise ValueError(key)
                    setattr(c,key,np.asarray(value,float).copy())

    def _all_contact(self):
        cf,reactions,torques,depth,power,count=self.contact_forces()
        rf,rt,rd,rp,rc=self.rigid_contact_forces()
        return cf,[a+b for a,b in zip(reactions,rf)],[a+b for a,b in zip(torques,rt)],max(depth,rd),power+rp,count+rc

    def _finish_step(self,dt,reactions,torques,depth,power,count):
        for c,reaction,torque in zip(self.colliders,reactions,torques):
            if np.isfinite(c.mass):
                c.velocity+=dt*(reaction/c.mass+self.gravity);c.omega+=dt*torque/c.inertia;c.center+=dt*c.velocity
        J=np.linalg.det(self.deformation())
        if not np.isfinite(self.x).all() or not np.isfinite(self.v).all() or J.min()<=.02:raise FloatingPointError('Step inverted/degenerate consistent-mass element; reduce dt')
        self.min_j=min(self.min_j,float(J.min()));self.max_penetration=max(self.max_penetration,depth)
        self.friction_work+=power*dt;self.contact_samples+=count;self.time+=dt
        # This integrator updates diagnostics itself; it does not invoke the
        # surface fallback step wrapper, preventing double accumulation.
        self.friction_active_samples+=self._last_friction_active+self._last_rigid_friction_active
        self.friction_saturated_samples+=self._last_friction_saturated+self._last_rigid_friction_saturated
        self.last_reactions={c.name:r.copy() for c,r in zip(self.colliders,reactions)}
        self.last_torques={c.name:r.copy() for c,r in zip(self.colliders,torques)}
        return dict(min_j=float(J.min()),max_penetration=depth,friction_power=power,contacts=count)


class ConsistentMassMixin(ConsistentInertia):
    def step(self,dt,tool_states=None):
        if dt<=0:raise ValueError('Positive timestep required')
        self._prescribe(tool_states)
        internal,_=self.internal_forces();cf,reactions,torques,depth,power,count=self._all_contact()
        total=internal+cf+self.mass[:,None]*self.gravity
        acc=self.mass_acceleration(total)
        self.last_constraint_reactions=self.mass_matrix@acc-total
        self.last_fixed_reaction=self.last_constraint_reactions[self.fixed].sum(0)
        self.v+=dt*acc;self.v[self.fixed]=0.
        self.x+=dt*self.v;self.x[self.fixed]=self.fixed_targets[self.fixed]
        return self._finish_step(dt,reactions,torques,depth,power,count)


class ConsistentRigidMassMixin(ConsistentInertia):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Integrating the quadratic moment with P1 mass gives exact continuum
        # inertia of a uniform tetrahedral body, including a uniform block.
        second=self.local.T@(self.mass_matrix@self.local)
        self.body_inertia=np.trace(second)*np.eye(3)-second

    def step(self,dt,tool_states=None):
        if dt<=0:raise ValueError('Positive timestep required')
        self._prescribe(tool_states)
        center=np.average(self.x,axis=0,weights=self.mass)
        arm=self.x-center;velocity=(self.mass_matrix@self.v).sum(0)/self.total_mass
        angular=np.cross(arm,self.mass_matrix@(self.v-velocity)).sum(0)
        force,reactions,torques,depth,power,count=self._all_contact()
        velocity+=dt*(force.sum(0)/self.total_mass+self.gravity)
        angular+=dt*np.cross(arm,force).sum(0)
        inertia=self.rotation.T@self.body_inertia@self.rotation
        omega=np.linalg.solve(inertia,angular)
        self.rotation=self.rotation@Rotation.from_rotvec(omega*dt).as_matrix().T
        self.center=center+dt*velocity;self.x=self.local@self.rotation+self.center
        inertia=self.rotation.T@self.body_inertia@self.rotation
        omega=np.linalg.solve(inertia,angular);self.v=velocity+np.cross(omega,self.x-self.center)
        return self._finish_step(dt,reactions,torques,depth,power,count)


class ConsistentFEMSystem(ConsistentMassMixin,FEMSystem):pass
class ConsistentClippedFEMSystem(ConsistentMassMixin,ClippedSurfaceFEMSystem):pass
class ConsistentRigidSystem(ConsistentRigidMassMixin,RigidSystem):pass


def with_consistent_mass(base):
    if issubclass(base,ConsistentInertia):return base
    mixin=ConsistentRigidMassMixin if issubclass(base,RigidSystem) else ConsistentMassMixin
    return type('Consistent'+base.__name__,(mixin,base),{})
