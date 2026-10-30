"""Conservative acceleration of identical material surface midpoint quadrature.

No adaptive change in quadrature or force law: a rest-patch radius multiplied by
a global deformation-gradient norm bounds every fine sample's distance from its
patch center. A 1-Lipschitz signed distance then rejects provably distant patches.
The open bowl query is not globally Lipschitz and is evaluated exhaustively.
"""
import numpy as np
from .surface_contact import SurfaceFEMSystem,with_surface_contact,box_surface_quadrature,tetrahedral_embedding


class SparseSurfaceContactMixin:
    surface_spacing=.0015
    broadphase_spacing=.003

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p,_=box_surface_quadrature(self.rest,self.broadphase_spacing)
        self.broadphase_ids,self.broadphase_weights=tetrahedral_embedding(self.rest,self.tets,p)
        lo=self.rest.min(0);size=np.ptp(self.rest,axis=0)
        groups=[];radii=[];fine_offset=0;coarse_offset=0
        for axis in range(3):
            tang=[k for k in range(3) if k!=axis]
            coarse=np.ceil(size[tang]/self.broadphase_spacing).astype(int)
            fine=np.ceil(size[tang]/self.surface_spacing).astype(int)
            widths=size[tang]/coarse;nf=int(np.prod(fine));nc=int(np.prod(coarse))
            for _ in range(2):
                uv=self.surface_points_rest[fine_offset:fine_offset+nf,tang]
                ij=np.floor((uv-lo[tang])/widths).astype(int);ij=np.minimum(ij,coarse-1)
                groups.append(coarse_offset+ij[:,0]*coarse[1]+ij[:,1])
                radii.extend([.5*np.linalg.norm(widths)]*nc)
                coarse_offset+=nc;fine_offset+=nf
        self.fine_to_broadphase=np.concatenate(groups);self.broadphase_radius=np.asarray(radii)
        if len(self.fine_to_broadphase)!=len(self.surface_points_rest):raise ValueError('Surface grouping mismatch')
        center=np.einsum('qi,qic->qc',self.broadphase_weights,self.rest[self.broadphase_ids])
        if np.any(np.linalg.norm(self.surface_points_rest-center[self.fine_to_broadphase],axis=1)>self.broadphase_radius[self.fine_to_broadphase]+1e-12):raise ValueError('Invalid conservative rest bound')

    def candidate_indices(self,c,centers,stretch_bound):
        if c.kind=='bowl':return np.arange(len(self.surface_points_rest))
        phi,_=c.query(centers)
        near=phi<=stretch_bound*self.broadphase_radius+1e-12
        return np.flatnonzero(near[self.fine_to_broadphase])

    def contact_forces(self):
        center=np.einsum('qi,qic->qc',self.broadphase_weights,self.x[self.broadphase_ids])
        # Frobenius bounds the spectral norm on every affine tetrahedron and
        # therefore the material-path Lipschitz constant across their union.
        stretch=float(np.linalg.norm(self.deformation(),axis=(1,2)).max())
        force=np.zeros_like(self.x);reactions=[];torques=[];maximum=0.;power=0.;count=0;fc=0;sc=0
        for c in self.colliders:
            ids=self.candidate_indices(c,center,stretch)
            points=np.einsum('qi,qic->qc',self.surface_barycentric[ids],self.x[self.surface_node_ids[ids]])
            phi,normal=c.query(points);active=phi<0
            ids=ids[active];points=points[active];normal=normal[active];depth=-phi[active]
            if not len(ids):reactions.append(np.zeros(3));torques.append(np.zeros(3));continue
            velocity=np.einsum('qi,qic->qc',self.surface_barycentric[ids],self.v[self.surface_node_ids[ids]])
            arm=points-c.center;relative=velocity-c.velocity-np.cross(c.omega,arm);vn=np.sum(relative*normal,axis=1)
            mass=self.surface_patch_mass[ids];reduced=mass if not np.isfinite(c.mass) else mass*c.mass/(mass+c.mass)
            stiffness=c.stiffness*self.surface_patch_stiffness_weights[ids];damping=2*c.damping_ratio*np.sqrt(stiffness*reduced)
            magnitude=np.maximum(stiffness*depth-damping*vn,0)
            tangent=relative-vn[:,None]*normal;speed=np.linalg.norm(tangent,axis=1)
            tangential=-c.friction*magnitude[:,None]*tangent/np.maximum(speed[:,None],c.friction_speed)
            cf=magnitude[:,None]*normal+tangential
            contributions=self.surface_barycentric[ids,:,None]*cf[:,None,:];nodeids=self.surface_node_ids[ids].ravel()
            force+=np.column_stack([np.bincount(nodeids,weights=contributions[:,:,k].ravel(),minlength=len(self.x)) for k in range(3)])
            reactions.append(-cf.sum(0));torques.append(np.cross(arm,-cf).sum(0))
            maximum=max(maximum,float(depth.max()));power+=float(np.sum(tangential*relative));count+=len(ids)
            fa=(magnitude>0)&(c.friction>0);fc+=int(fa.sum());sc+=int((fa&(speed>=c.friction_speed)).sum())
        self._last_friction_active=fc;self._last_friction_saturated=sc
        return force,reactions,torques,maximum,power,count


class SparseSurfaceFEMSystem(SparseSurfaceContactMixin,SurfaceFEMSystem):
    pass


def with_sparse_surface_contact(base):
    if issubclass(base,SparseSurfaceContactMixin):return base
    return type('SparseSurface'+base.__name__,(SparseSurfaceContactMixin,with_surface_contact(base)),{})
