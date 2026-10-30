"""Post-audit extension: material surface quadrature instead of vertex contact.

Initial object must be an axis-aligned tetrahedral box. Midpoint patches are fixed
in material coordinates, use actual P1 tetrahedral interpolation, and move/deform
with the object. No CCD, self-contact, contact complementarity or exact stiction.
The frozen vertex-contact implementation remains untouched.
"""
import numpy as np
from .physics import FEMSystem


def box_surface_quadrature(rest,spacing=.003):
    if spacing<=0:raise ValueError('Positive spacing required')
    lo=np.min(rest,axis=0);hi=np.max(rest,axis=0);size=hi-lo
    if np.any(size<=0):raise ValueError('Nondegenerate box required')
    points=[];areas=[]
    for axis in range(3):
        tang=[k for k in range(3) if k!=axis]
        counts=np.ceil(size[tang]/spacing).astype(int)
        widths=size[tang]/counts
        grids=[lo[k]+(np.arange(n)+.5)*w for k,n,w in zip(tang,counts,widths)]
        uv=np.array(np.meshgrid(*grids,indexing='ij')).reshape(2,-1).T
        for side in (lo[axis],hi[axis]):
            p=np.empty((len(uv),3));p[:,axis]=side;p[:,tang]=uv
            points.append(p);areas.append(np.full(len(uv),np.prod(widths)))
    return np.concatenate(points),np.concatenate(areas)


def tetrahedral_embedding(rest,tets,points):
    tet=rest[tets];D=np.transpose(tet[:,1:]-tet[:,0,None],(0,2,1));inverse=np.linalg.inv(D)
    ids=[];weights=[]
    for p in points:
        b=np.einsum('tij,tj->ti',inverse,p-tet[:,0]);b=np.column_stack((1-b.sum(1),b))
        candidates=np.flatnonzero(np.all(b>=-1e-9,axis=1)&np.all(b<=1+1e-9,axis=1))
        if not len(candidates):raise ValueError('Surface point outside actual tetrahedral mesh; box required')
        k=candidates[0];w=np.maximum(b[k],0);w/=w.sum();ids.append(tets[k]);weights.append(w)
    ids=np.asarray(ids,int);weights=np.asarray(weights)
    if not np.allclose(np.einsum('qi,qic->qc',weights,rest[ids]),points,rtol=0,atol=1e-10):raise ValueError('Invalid surface embedding')
    return ids,weights


class SurfaceContactMixin:
    surface_spacing=.003

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.surface_points_rest,self.surface_patch_area=box_surface_quadrature(self.rest,self.surface_spacing)
        self.surface_node_ids,self.surface_barycentric=tetrahedral_embedding(self.rest,self.tets,self.surface_points_rest)
        expected=2*sum(np.prod(np.ptp(self.rest,axis=0)[[i,j]]) for i,j in [(0,1),(0,2),(1,2)])
        if not np.isclose(self.surface_area.sum(),expected,rtol=1e-8):raise ValueError('Rest boundary must be complete box surface')
        self.surface_patch_mass=self.mass.sum()*self.surface_patch_area/self.surface_patch_area.sum()
        self.surface_patch_stiffness_weights=self.surface_patch_area/.0004

    def surface_state(self):
        return (np.einsum('qi,qic->qc',self.surface_barycentric,self.x[self.surface_node_ids]),
                np.einsum('qi,qic->qc',self.surface_barycentric,self.v[self.surface_node_ids]))

    def distribute_surface_force(self,force):
        contributions=self.surface_barycentric[:,:,None]*force[:,None,:]
        return np.column_stack([np.bincount(self.surface_node_ids.ravel(),weights=contributions[:,:,k].ravel(),minlength=len(self.x)) for k in range(3)])

    def contact_forces(self):
        points,velocity=self.surface_state();force=np.zeros_like(points)
        reactions=[];torques=[];max_depth=0.;power=0.;count=0;friction_count=0;saturated_count=0
        for c in self.colliders:
            phi,normal=c.query(points);active=phi<0;depth=np.maximum(-phi,0);depth=np.where(np.isfinite(depth),depth,0)
            arm=points-c.center;relative=velocity-c.velocity-np.cross(c.omega,arm)
            vn=np.sum(relative*normal,axis=1)
            reduced=self.surface_patch_mass if not np.isfinite(c.mass) else self.surface_patch_mass*c.mass/(self.surface_patch_mass+c.mass)
            stiffness=c.stiffness*self.surface_patch_stiffness_weights
            damping=2*c.damping_ratio*np.sqrt(stiffness*reduced)
            magnitude=np.maximum(stiffness*depth-damping*vn,0)*active
            tangent=relative-vn[:,None]*normal;speed=np.linalg.norm(tangent,axis=1)
            tangential=-c.friction*magnitude[:,None]*tangent/np.maximum(speed[:,None],c.friction_speed)
            cf=magnitude[:,None]*normal+tangential;force+=cf
            reactions.append(-cf.sum(0));torques.append(np.cross(arm,-cf).sum(0))
            max_depth=max(max_depth,float(depth.max()));power+=float(np.sum(tangential*relative));count+=int(active.sum())
            fa=active&(magnitude>0)&(c.friction>0);friction_count+=int(fa.sum());saturated_count+=int((fa&(speed>=c.friction_speed)).sum())
        self._last_friction_active=friction_count;self._last_friction_saturated=saturated_count
        return self.distribute_surface_force(force),reactions,torques,max_depth,power,count

    def step(self,dt,tool_states=None):
        # Frozen rigid comparator does not accumulate friction diagnostics, while
        # FEM already does. Add them only when the parent left them unchanged.
        before=self.friction_active_samples
        result=super().step(dt,tool_states)
        if self.friction_active_samples==before:
            self.friction_active_samples+=self._last_friction_active+self._last_rigid_friction_active
            self.friction_saturated_samples+=self._last_friction_saturated+self._last_rigid_friction_saturated
        return result

    def surface_contact_metadata(self):
        return dict(type='Material-box surface midpoint quadrature',spacing_m=self.surface_spacing,points=len(self.surface_points_rest),area_m2=float(self.surface_patch_area.sum()),reference_patch_area_m2=.0004,damping_mass='total mass times patch area divided by total rest surface area',limitations=['Box rest shape','Fixed material quadrature, not continuous collision detection','No self-contact','Coulomb-capped viscous friction, no exact stiction'])


class SurfaceFEMSystem(SurfaceContactMixin,FEMSystem):
    pass


def with_surface_contact(base):
    """Wrap a FEM, native-spring or rigid class with exactly the same contact."""
    if issubclass(base,SurfaceContactMixin):return base
    return type('Surface'+base.__name__,(SurfaceContactMixin,base),{})
