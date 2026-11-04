"""Restore initial-shape-aware tetrahedral filtering before learning more latents."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scripts.external_benchmark.fem import ExternalFEM
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend


class GeometryAwareGaussianBoundaryBackend(WideGaussianTableBoundaryBackend):
    def __init__(self,payload,device='cpu'):
        super().__init__(payload,device)
        self.model=ExternalFEM(payload,device=device,field=False,contact=True,nonconvex=True,materials=1)
        self.base_weights=self.model.weights.clone()
        tet=self.model.elements.cpu().numpy();edges=np.concatenate([tet[:,[0,j]]for j in (1,2,3)])
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(self.model.n,self.model.n))
        components,_=connected_components(graph,directed=False)
        if components!=1 or bool((self.model.mass<=0).any()):raise ValueError('Filtered mesh disconnected or has unused nodes; increase geometry resolution before inference')
        self.geometry_diagnostic={'nodes':self.model.n,'elements':len(tet),'connected_components':int(components),'volume_m3':float(self.model.vol.sum())*self.model.scale**3,'filter':self.model.mesh_filter}
