"""Small 3-D tetrahedral nonlinear FEM with explicitly coupled penalty contact.

SI units. Compressible Neo-Hookean elasticity, objective Kelvin-Voigt damping,
lumped mass, symplectic Euler. Contact is node sampled, not CCD/IPC; self-contact
and multiple deformable bodies are not implemented. Positive-J checks reject
invalid states rather than silently clamping constitutive response.
"""
from dataclasses import dataclass, field
import itertools
import numpy as np


@dataclass
class Material:
    young: float = 12000.
    poisson: float = .3
    density: float = 1000.
    viscosity: float = 1.

    def __post_init__(self):
        if self.young <= 0 or not -1 < self.poisson < .49 or self.density <= 0 or self.viscosity < 0:
            raise ValueError('Invalid material parameters')

    @property
    def mu(self): return self.young / (2 * (1 + self.poisson))
    @property
    def lam(self): return self.young * self.poisson / ((1+self.poisson)*(1-2*self.poisson))


def make_block(size=(.06,.04,.04), cells=(3,2,2), center=(0.,0.,.04)):
    """Conforming six-tetrahedra subdivision per Cartesian cell."""
    size=np.asarray(size,float); cells=np.asarray(cells,int)
    if np.any(cells < 1) or np.any(size <= 0): raise ValueError('Positive mesh dimensions required')
    axes=[np.linspace(-s/2,s/2,n+1) for s,n in zip(size,cells)]
    x=np.array(list(itertools.product(*axes)))+np.asarray(center)
    shape=tuple(cells+1); tets=[]
    for base in itertools.product(*[range(n) for n in cells]):
        for perm in itertools.permutations(range(3)):
            p=np.array(base); vertices=[np.ravel_multi_index(tuple(p),shape)]
            for axis in perm:
                p=p.copy();p[axis]+=1;vertices.append(np.ravel_multi_index(tuple(p),shape))
            tets.append(vertices)
    tets=np.asarray(tets,int)
    D=np.transpose(x[tets[:,1:]]-x[tets[:,0,None]],(0,2,1))
    negative=np.linalg.det(D)<0
    tets[negative,1],tets[negative,2]=tets[negative,2].copy(),tets[negative,1].copy()
    return x,tets


@dataclass
class Collider:
    """SDF shape, positive on allowed side. Dynamic coupling supports spheres.

    plane: normal points into allowed space; center lies on surface.
    sphere/box: solid obstacle. bowl: spherical cavity, open above center z;
    its lower hemispherical wall has no rim thickness.
    Box half_extents and rotation define an oriented cuboid.
    """
    kind: str = 'plane'
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.,0.,1.]))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    half_extents: np.ndarray = field(default_factory=lambda: np.array([.02,.03,.03]))
    radius: float = .03
    mass: float = np.inf
    friction: float = .4
    stiffness: float = 3000.  # N/m per fixed 0.0004 m^2 surface patch
    damping_ratio: float = .5
    friction_speed: float = .01
    name: str = 'collider'

    def __post_init__(self):
        for name in ('center','velocity','omega','normal','rotation','half_extents'):
            setattr(self,name,np.array(getattr(self,name),float,copy=True))
        if self.kind not in ('plane','sphere','box','bowl'): raise ValueError(self.kind)
        if np.isfinite(self.mass) and self.kind != 'sphere': raise ValueError('Only movable spheres currently supported')
        if self.mass<=0 or self.radius<=0 or self.stiffness<=0 or self.friction<0 or self.friction_speed<=0:
            raise ValueError('Invalid collider parameters')
        self.normal/=np.linalg.norm(self.normal)

    @property
    def inertia(self): return .4*self.mass*self.radius**2

    def query(self,x):
        r=x-self.center
        if self.kind=='plane':
            return r@self.normal,np.broadcast_to(self.normal,x.shape).copy()
        if self.kind in ('sphere','bowl'):
            length=np.linalg.norm(r,axis=1)
            normal=r/np.maximum(length[:,None],1e-15)
            normal[length<1e-15]=[1.,0.,0.]
            if self.kind=='sphere': return length-self.radius,normal
            phi=self.radius-length;normal=-normal
            phi=np.where(r[:,2]<=0,phi,np.inf)
            return phi,normal
        local=r@self.rotation
        q=np.abs(local)-self.half_extents
        outside=np.maximum(q,0);length=np.linalg.norm(outside,axis=1)
        phi=length+np.minimum(q.max(axis=1),0)
        normal=outside*np.sign(local)/np.maximum(length[:,None],1e-15)
        inside=length<=1e-15
        axes=np.argmax(q[inside],axis=1);normal[inside]=0
        signs=np.sign(local[inside,axes]);signs[signs==0]=1
        normal[np.flatnonzero(inside),axes]=signs
        return phi,normal@self.rotation.T


class FEMSystem:
    def __init__(self, rest, tets, material=None, colliders=None, gravity=(0.,0.,-9.81), fixed=None):
        self.rest=np.asarray(rest,float).copy(); self.tets=np.asarray(tets,int).copy()
        self.material=material or Material();self.colliders=list(colliders or [])
        self.x=self.rest.copy();self.v=np.zeros_like(self.x);self.gravity=np.asarray(gravity,float)
        D=np.transpose(self.rest[self.tets[:,1:]]-self.rest[self.tets[:,0,None]],(0,2,1))
        self.volume=np.linalg.det(D)/6
        if np.any(self.volume<=0): raise ValueError('Rest tetrahedra must have positive orientation')
        # Lump rest boundary-triangle area onto vertices. Interior nodes do not
        # collide. A fixed physical reference patch makes contact compliance
        # comparable across mesh resolutions; it is never fitted per mesh.
        faces=np.concatenate([self.tets[:,ids] for ids in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
        unique,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
        self.surface_triangles=unique[counts==1]
        tri=self.rest[self.surface_triangles]
        area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        self.surface_area=np.zeros(len(self.rest))
        np.add.at(self.surface_area,self.surface_triangles.ravel(),np.repeat(area/3,3))
        self.contact_area_weights=self.surface_area/.0004
        self.boundary_nodes=self.surface_area>0
        self.invD=np.linalg.inv(D)
        self.mass=np.zeros(len(self.x))
        np.add.at(self.mass,self.tets.ravel(),np.repeat(self.volume*self.material.density/4,4))
        if np.any(self.mass<=0): raise ValueError('Unused mesh vertices')
        self.fixed=np.zeros(len(self.x),bool) if fixed is None else np.asarray(fixed,bool)
        self.fixed_targets=self.rest.copy();self.time=0.
        self.max_penetration=0.;self.min_j=1.;self.friction_work=0.;self.contact_samples=0
        self.friction_active_samples=0;self.friction_saturated_samples=0
        self.last_reactions={};self.last_torques={};self.last_fixed_reaction=np.zeros(3)

    def deformation(self,x=None):
        x=self.x if x is None else np.asarray(x)
        Ds=np.transpose(x[self.tets[:,1:]]-x[self.tets[:,0,None]],(0,2,1))
        return Ds@self.invD

    def elastic_energy(self,x=None):
        F=self.deformation(x);J=np.linalg.det(F)
        if np.any(J<=0): return np.inf
        logJ=np.log(J);m=self.material
        W=.5*m.mu*((F*F).sum((1,2))-3)-m.mu*logJ+.5*m.lam*logJ**2
        return float(W@self.volume)

    def internal_forces(self):
        F=self.deformation();J=np.linalg.det(F)
        if np.any(J<=.02) or not np.isfinite(F).all(): raise FloatingPointError('Invalid FEM state: J <= .02 or nonfinite')
        invF=np.linalg.inv(F);m=self.material
        P=m.mu*F+(m.lam*np.log(J)-m.mu)[:,None,None]*np.transpose(invF,(0,2,1))
        if m.viscosity:
            dv=np.transpose(self.v[self.tets[:,1:]]-self.v[self.tets[:,0,None]],(0,2,1))
            L=dv@self.invD@invF;rate=.5*(L+np.transpose(L,(0,2,1)))
            P+=2*m.viscosity*J[:,None,None]*(rate@np.transpose(invF,(0,2,1)))
        columns=-self.volume[:,None,None]*(P@np.transpose(self.invD,(0,2,1)))
        force=np.zeros_like(self.x)
        nodal=np.concatenate((-columns.sum(axis=2)[:,None,:],np.transpose(columns,(0,2,1))),axis=1)
        np.add.at(force,self.tets.ravel(),nodal.reshape(-1,3))
        return force,J

    def contact_forces(self):
        force=np.zeros_like(self.x);reactions=[];torques=[];max_depth=0.;power=0.;count=0;friction_count=0;saturated_count=0
        for c in self.colliders:
            phi,normal=c.query(self.x);active=(phi<0)&self.boundary_nodes;depth=np.where(self.boundary_nodes,np.maximum(-phi,0),0)
            depth=np.where(np.isfinite(depth),depth,0)
            arm=self.x-c.center
            relative=self.v-c.velocity-np.cross(c.omega,arm)
            vn=np.sum(relative*normal,axis=1)
            reduced=self.mass if not np.isfinite(c.mass) else self.mass*c.mass/(self.mass+c.mass)
            stiffness=c.stiffness*self.contact_area_weights
            damping=2*c.damping_ratio*np.sqrt(stiffness*reduced)
            magnitude=np.maximum(stiffness*depth-damping*vn,0)*active
            tangent=relative-vn[:,None]*normal;speed=np.linalg.norm(tangent,axis=1)
            tangential=-c.friction*magnitude[:,None]*tangent/np.maximum(speed[:,None],c.friction_speed)
            friction_active=active&(magnitude>0)&(c.friction>0)
            friction_count+=int(friction_active.sum());saturated_count+=int((friction_active&(speed>=c.friction_speed)).sum())
            cf=magnitude[:,None]*normal+tangential
            force+=cf;reactions.append(-cf.sum(axis=0));torques.append(np.cross(arm,-cf).sum(axis=0))
            max_depth=max(max_depth,float(depth.max()));power+=float(np.sum(tangential*relative));count+=int(active.sum())
        self._last_friction_active=friction_count;self._last_friction_saturated=saturated_count
        return force,reactions,torques,max_depth,power,count

    def rigid_contact_forces(self):
        """Sphere/static or sphere/sphere contact with a shared contact point.

        Midpoint of penetrated surfaces gives exact equal/opposite torque balance.
        Rigid contact is a point model using unscaled collider stiffness (N/m).
        """
        forces=np.zeros((len(self.colliders),3));torques=np.zeros_like(forces)
        maximum=0.;power=0.;count=0;friction_count=0;saturated_count=0
        for i,a in enumerate(self.colliders):
            if not np.isfinite(a.mass): continue
            for j,b in enumerate(self.colliders):
                if i==j or (np.isfinite(b.mass) and j<i): continue
                phi,normal=b.query(a.center[None]);depth=a.radius-phi[0]
                if depth<=0 or not np.isfinite(depth): continue
                n=normal[0];point=a.center-(a.radius-.5*depth)*n
                ra=point-a.center;rb=point-b.center
                relative=a.velocity+np.cross(a.omega,ra)-b.velocity-np.cross(b.omega,rb)
                vn=float(relative@n);tangent=relative-vn*n
                reduced=a.mass if not np.isfinite(b.mass) else a.mass*b.mass/(a.mass+b.mass)
                stiffness=min(a.stiffness,b.stiffness)
                damping=2*min(a.damping_ratio,b.damping_ratio)*np.sqrt(stiffness*reduced)
                magnitude=max(stiffness*depth-damping*vn,0.)
                tangential=-min(a.friction,b.friction)*magnitude*tangent/max(np.linalg.norm(tangent),a.friction_speed,b.friction_speed)
                if magnitude>0 and min(a.friction,b.friction)>0:
                    friction_count+=1
                    saturated_count+=int(np.linalg.norm(tangent)>=max(a.friction_speed,b.friction_speed))
                force=magnitude*n+tangential
                forces[i]+=force;forces[j]-=force
                torques[i]+=np.cross(ra,force);torques[j]+=np.cross(rb,-force)
                maximum=max(maximum,float(depth));power+=float(tangential@relative);count+=1
        self._last_rigid_friction_active=friction_count;self._last_rigid_friction_saturated=saturated_count
        return forces,torques,maximum,power,count

    def momentum(self):
        p=(self.mass[:,None]*self.v).sum(0)
        for c in self.colliders:
            if np.isfinite(c.mass): p+=c.mass*c.velocity
        return p

    def angular_momentum(self):
        p=np.cross(self.x,self.mass[:,None]*self.v).sum(0)
        for c in self.colliders:
            if np.isfinite(c.mass): p+=np.cross(c.center,c.mass*c.velocity)+c.inertia*c.omega
        return p

    def step(self,dt,tool_states=None):
        """Set prescribed collider poses/velocities at step start, then integrate.

        tool_states is {collider_name: {center,velocity,rotation,omega}}. Never
        supply measured forces; prescribed velocities must match pose derivative.
        """
        if dt<=0: raise ValueError('Positive timestep required')
        for c in self.colliders:
            if tool_states and c.name in tool_states:
                if np.isfinite(c.mass): raise ValueError('Cannot prescribe a dynamic body')
                for key,value in tool_states[c.name].items():
                    if key not in ('center','velocity','rotation','omega'): raise ValueError(key)
                    setattr(c,key,np.asarray(value,float).copy())
        internal,J=self.internal_forces()
        cf,reactions,torques,depth,power,count=self.contact_forces()
        rf,rt,rd,rp,rc=self.rigid_contact_forces()
        reactions=[a+b for a,b in zip(reactions,rf)];torques=[a+b for a,b in zip(torques,rt)]
        depth=max(depth,rd);power+=rp;count+=rc
        total=internal+cf+self.mass[:,None]*self.gravity
        self.last_fixed_reaction=-total[self.fixed].sum(0)
        self.v+=dt*total/self.mass[:,None];self.v[self.fixed]=0
        self.x+=dt*self.v;self.x[self.fixed]=self.fixed_targets[self.fixed]
        for c,reaction,torque in zip(self.colliders,reactions,torques):
            if np.isfinite(c.mass):
                c.velocity+=dt*(reaction/c.mass+self.gravity);c.omega+=dt*torque/c.inertia;c.center+=dt*c.velocity
        nextJ=np.linalg.det(self.deformation())
        if not np.isfinite(self.x).all() or nextJ.min()<=.02: raise FloatingPointError('Step inverted/degenerate FEM element; reduce dt')
        self.min_j=min(self.min_j,float(nextJ.min()));self.max_penetration=max(self.max_penetration,depth)
        self.friction_work+=power*dt;self.contact_samples+=count;self.time+=dt
        self.friction_active_samples+=self._last_friction_active+self._last_rigid_friction_active
        self.friction_saturated_samples+=self._last_friction_saturated+self._last_rigid_friction_saturated
        self.last_reactions={c.name:r.copy() for c,r in zip(self.colliders,reactions)}
        self.last_torques={c.name:r.copy() for c,r in zip(self.colliders,torques)}
        return dict(min_j=float(nextJ.min()),max_penetration=depth,friction_power=power,contacts=count)


def simulate(system,duration,dt=1e-4,fps=50,trajectory=None):
    """Return saved positions, rigid poses and diagnostics; no post-hoc repairs."""
    steps=int(np.ceil(duration/dt));dt=duration/steps
    save_times=np.linspace(0,duration,int(round(duration*fps))+1)
    frames=[system.x.copy()];velocities=[system.v.copy()];centers=[[c.center.copy() for c in system.colliders]]
    times=[0.];reactions=[np.zeros((len(system.colliders),3))];next_save=1
    for k in range(steps):
        states=None if trajectory is None else trajectory(k*dt)
        system.step(dt,states)
        if next_save<len(save_times) and (k+1)*dt>=save_times[next_save]-1e-12:
            frames.append(system.x.copy());velocities.append(system.v.copy());centers.append([c.center.copy() for c in system.colliders])
            reactions.append([system.last_reactions[c.name] for c in system.colliders]);times.append((k+1)*dt);next_save+=1
    return dict(frames=np.asarray(frames),velocities=np.asarray(velocities),centers=np.asarray(centers),times=np.asarray(times),reactions=np.asarray(reactions),
                min_j=system.min_j,max_penetration_m=system.max_penetration,friction_work_j=system.friction_work,contact_samples=system.contact_samples,dt=dt,
                friction_active_samples=system.friction_active_samples,friction_saturated_samples=system.friction_saturated_samples,
                friction_saturated_fraction=system.friction_saturated_samples/max(system.friction_active_samples,1))
