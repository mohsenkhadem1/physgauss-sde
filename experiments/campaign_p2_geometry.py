"""Straight-reference T10 geometry for matrix-free mechanics, without CPU H.

Optional axis grading changes only the mesh inside the same rectangular solid.
Only corner coordinates are mapped; quadratic edge nodes are rebuilt as true
physical midpoints. It never creates an unaccounted curved reference element.
"""
import numpy as np
from experiments.boundary_block import block_mesh
from experiments.campaign_static_p2_block import enrich,duffy,shapes,EDGES,QuadraticBlock
from experiments.campaign_block_surface_controls import SIZE,support_groups,surface_control


class P2Geometry:
    query=QuadraticBlock.query
    diagnostics=QuadraticBlock.diagnostics

    def __init__(self,cells,*,young=4500.,poisson=.35,density=625.,gravity=0.,pattern='transverse_end',support='left_face',mapping='uniform',order=3):
        if len(cells)!=3 or any(type(n) is not int or n<1 for n in cells):raise ValueError('Three positive integer cell counts required')
        if mapping not in ['uniform','clamp_graded'] or not np.isfinite([young,poisson,density,gravity]).all() or young<=0 or not -1<poisson<.5 or density<=0:raise ValueError('Declared mesh and physical material required')
        self.young=float(young);self.poisson=float(poisson);self.density=float(density);self.minimum_J=.1;self.mapping=mapping;self.order=order
        X,self.tets,self.faces=block_mesh(SIZE,tuple(cells),(SIZE[0]/2,0,0))
        if mapping=='clamp_graded':
            X=X.copy();X[:,0]=SIZE[0]*(X[:,0]/SIZE[0])**2
            for axis in [1,2]:X[:,axis]=SIZE[axis]/2*np.sin(np.pi*X[:,axis]/SIZE[axis])
        self.rest,self.elements=enrich(X,self.tets);self.reference_local=self.rest
        self.rotation=np.eye(3);self.translation=np.zeros(3)
        corners=X[self.tets];Dm=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2);det=np.linalg.det(Dm)
        if not np.isfinite(det).all() or np.any(det<=0):raise ValueError('Positive affine reference tetrahedra required')
        inverse=np.linalg.inv(Dm);self.linear_gradients=np.concatenate([-inverse.sum(1)[:,None],inverse],1);self.volumes=det/6
        edge_lengths=np.stack([np.linalg.norm(corners[:,i]-corners[:,j],axis=-1) for i,j in EDGES])
        self.maximum_reference_tet_edge_m=float(edge_lengths.max());self.minimum_reference_tet_edge_m=float(edge_lengths.min());self.cell_volume_ratio=float(self.volumes.max()/self.volumes.min())
        if not np.isclose(self.volumes.sum(),np.prod(SIZE),rtol=1e-13,atol=1e-16):raise ValueError('Same physical solid volume required')
        self.bary,self.weights=duffy(order);self.N=shapes(self.bary);self.check_bary=np.concatenate([np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES])])
        groups,self.support_active=support_groups(self.rest,support);self.support_masks=groups;self.mask=np.repeat(groups.any(0)[:,None],3,1);self.free=np.flatnonzero(~self.mask.ravel())
        self.load,self.force_metadata=surface_control(self.rest,self.tets,self.faces,pattern,self.elements)
        local=self.volumes[:,None]*np.einsum('q,qa->a',self.weights,self.N);self.body_integral_weights=np.zeros(len(self.rest));np.add.at(self.body_integral_weights,self.elements,local)
        self.load[:,2]-=gravity*self.density*self.body_integral_weights
        self.base_geometry_array_bytes=sum(a.nbytes for a in [self.rest,self.tets,self.faces,self.elements,self.linear_gradients,self.volumes,self.mask,self.load,self.body_integral_weights])
        # The inherited diagnostic has a legacy estimate field. Do not confuse
        # geometry storage with the solver's actual working-memory peak.
        self.estimated_peak_memory_bytes=None
