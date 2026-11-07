"""Isolated concave-bowl stress experiment; no changes to frozen v13 mechanics.

Neo-Hookean P1 tetrahedral dynamics and regularized Coulomb penalty contact.
Surface integration uses subdivided actual boundary triangles, not box samples.
The bowl is an open lower hemisphere, without rim thickness, CCD or self-contact.
"""
from dataclasses import dataclass
import time
import numpy as np
from scripts.coupled_contact_v13.physics import Collider, FEMSystem, Material, make_block, simulate
from scripts.coupled_contact_v13.surface_contact import SurfaceContactMixin


@dataclass
class Bowl(Collider):
    kind: str = 'bowl'

    def query(self, x):
        r = np.asarray(x)-self.center
        local = r @ self.rotation
        length = np.linalg.norm(r, axis=1)
        normal = -r/np.maximum(length[:, None], 1e-15)
        normal[length < 1e-15] = [0, 0, 1]
        return np.where(local[:, 2] <= 0, self.radius-length, np.inf), normal


def shape_mesh(n=3, height=.048, geometry_cells=3):
    """Spherified-cube ellipsoid, with mild asymmetric taper; positive volume."""
    q, tets = make_block((2, 2, 2), (n, n, n), (0, 0, 0))
    if n != geometry_cells:
        if n % geometry_cells: raise ValueError('Refinement must preserve base tetrahedral geometry')
        base,_=shape_mesh(geometry_cells,height,geometry_cells)
        coord=(q+1)*geometry_cells/2
        ij=np.minimum(np.floor(coord).astype(int),geometry_cells-1)
        local=coord-ij;order=np.argsort(-local,axis=1)
        mapped=[]
        for index,u,perm in zip(ij,local,order):
            vertices=[index.copy()]
            for axis in perm:
                index=index.copy();index[axis]+=1;vertices.append(index)
            ids=np.ravel_multi_index(np.asarray(vertices).T,(geometry_cells+1,)*3)
            a=u[perm];weights=np.array([1-a[0],a[0]-a[1],a[1]-a[2],a[2]])
            mapped.append(weights@base[ids])
        return np.asarray(mapped),tets
    q2 = q*q
    s = np.empty_like(q)
    for k in range(3):
        a, b = [j for j in range(3) if j != k]
        s[:, k] = q[:, k]*np.sqrt(1-q2[:, a]/2-q2[:, b]/2+q2[:, a]*q2[:, b]/3)
    s *= [.044, .029, .027]
    s[:, 1] *= 1+.22*s[:, 0]/.044
    s[:, 2] += .006*(s[:, 0]/.044)**2
    s += [.012, 0, height]
    return s, tets


def triangle_quadrature(rest, faces, subdivisions=2):
    ids=[]; weights=[]; areas=[]
    for face in faces:
        v=rest[face]
        area=np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0]))/2
        def b(i,j): return np.array([1-(i+j)/subdivisions, i/subdivisions, j/subdivisions])
        for i in range(subdivisions):
            for j in range(subdivisions-i):
                weights.append((b(i,j)+b(i+1,j)+b(i,j+1))/3)
                ids.append(face);areas.append(area/subdivisions**2)
                if i+j < subdivisions-1:
                    weights.append((b(i+1,j)+b(i+1,j+1)+b(i,j+1))/3)
                    ids.append(face);areas.append(area/subdivisions**2)
    return np.asarray(ids),np.asarray(weights),np.asarray(areas)


class TriangleContact(SurfaceContactMixin):
    subdivisions=2

    def __init__(self,*args,**kwargs):
        # Skip box-specific SurfaceContactMixin initialization, retain force law.
        super(SurfaceContactMixin,self).__init__(*args,**kwargs)
        self.surface_node_ids,self.surface_barycentric,self.surface_patch_area=triangle_quadrature(self.rest,self.surface_triangles,self.subdivisions)
        self.surface_points_rest=np.einsum('qi,qic->qc',self.surface_barycentric,self.rest[self.surface_node_ids])
        self.surface_patch_mass=self.mass.sum()*self.surface_patch_area/self.surface_patch_area.sum()
        self.surface_patch_stiffness_weights=self.surface_patch_area/.0004


class BowlFEM(TriangleContact,FEMSystem):
    pass


def rotation_y(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def build(young=2500.,friction=.35,n=3,penalty=4800.,subdivisions=2,system_class=BowlFEM):
    rest,tets=shape_mesh(n)
    cls=type('Configured'+system_class.__name__,(system_class,),{'subdivisions':subdivisions})
    bowl=Bowl(center=(0,0,.09),radius=.09,friction=friction,stiffness=penalty,name='bowl')
    pusher=Collider(kind='sphere',center=(-.052,0,.15),radius=.018,friction=friction,stiffness=penalty,name='pusher')
    return cls(rest,tets,Material(young=young,poisson=.3,density=1000,viscosity=2.),[bowl,pusher])


def smooth(t,start,end):
    z=np.clip((t-start)/(end-start),0,1)
    return z*z*(3-2*z), (6*z*(1-z)/(end-start) if start<t<end else 0.)


def command(t,interaction='settle'):
    states={}
    if interaction in ('tilt','calibration_tilt'):
        u,v=smooth(t,.55,.9)
        amplitude=.2 if interaction=='calibration_tilt' else .45
        states['bowl']=dict(rotation=rotation_y(amplitude*u),omega=np.array([0,amplitude*v,0.]))
    elif interaction in ('push','calibration_push','heldout_push'):
        lower,lower_v=smooth(t,.55,.75)
        u,v=smooth(t,.75,1.05)
        # Enter through the open rim before translating; never through the wall.
        travel={'push':.035,'calibration_push':.025,'heldout_push':.045}[interaction]
        center=np.array([-.052+travel*u,0,.15-.105*lower]);velocity=np.array([travel*v,0,-.105*lower_v])
        if interaction=='heldout_push':center=center[[1,0,2]];velocity=velocity[[1,0,2]]
        states['pusher']=dict(center=center,velocity=velocity)
    return states


def run(young=2500.,friction=.35,n=3,dt=1e-4,duration=.55,interaction='settle',penalty=4800.,subdivisions=2,system_class=BowlFEM,deadline_unix=None):
    system=build(young,friction,n,penalty,subdivisions,system_class)
    for collider in system.colliders:
        for key,value in command(0,interaction).get(collider.name,{}).items():setattr(collider,key,np.asarray(value).copy())
    original_step=system.step;impulses=[]
    def recorded_step(step_dt,states=None):
        if deadline_unix is not None and len(impulses)%100==0 and time.time()>=deadline_unix:raise RuntimeError('Campaign deadline reached during CPU trajectory')
        result=original_step(step_dt,states)
        impulses.append(np.array([system.last_reactions[c.name] for c in system.colliders])*step_dt)
        return result
    system.step=recorded_step
    result=simulate(system,duration,dt,50,lambda t:command(t,interaction))
    cumulative=np.concatenate([np.zeros((1,len(system.colliders),3)),np.cumsum(np.asarray(impulses),axis=0)])
    indices=np.rint(result['times']/result['dt']).astype(int)
    averages=np.zeros_like(result['reactions'])
    averages[1:]=(cumulative[indices[1:]]-cumulative[indices[:-1]])/np.diff(result['times'])[:,None,None]
    result['reaction_averages']=averages
    result.update(rest=system.rest,tets=system.tets,faces=system.surface_triangles,mass=system.mass,young=young,friction=friction,penalty=penalty,interaction=interaction)
    return result
