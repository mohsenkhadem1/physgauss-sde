"""NEW graded8x4x4 rest correction; original fine operator remains unchanged.

This is a preconditioner-only alternative, never a physical stiffness repair.
The old uniform4x2x2 coarse class is immutable. Scope is the canonical block's
explicit full-left clamp. GPU use requires a separately frozen driver/protocol.
"""
import numpy as np
import torch
from scipy import sparse
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2,compatible_prolongation
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_torch import LinearFailure


class SparseGradedCoarseGeometry(P2Geometry):
    elastic=StableP2Block.elastic

    def __init__(self,*,young,poisson,density):
        super().__init__((8,4,4),young=young,poisson=poisson,density=density,gravity=0.,mapping='clamp_graded',order=3)
        dofs=(3*self.elements[...,None]+np.arange(3)).reshape(len(self.elements),30)
        self.rows=np.repeat(dofs,30,1).ravel().astype(np.int32)
        self.cols=np.tile(dofs,(1,30)).ravel().astype(np.int32)
        self.load[:]=0.  # Only a rest tangent/mass PRECONDITIONER is assembled.


class GradedCoarseTwoLevelP2(TwoLevelP2):
    def __init__(self,model,device='cpu',maximum_cg_iterations=4000):
        expected=np.repeat((abs(model.rest[:,0])<1e-13)[:,None],3,1)
        if not np.array_equal(model.mask,expected):raise ValueError('Explicit full-left clamp required for graded coarse correction')
        BlockJacobiP2.__init__(self,model,device,maximum_cg_iterations)
        coarse=SparseGradedCoarseGeometry(young=model.young,poisson=model.poisson,density=model.density)
        P=compatible_prolongation(coarse,model.rest);self.prolongation_csr=P;coo=P.tocoo()
        indices=torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.int64,device=self.device)
        self.P=torch.sparse_coo_tensor(indices,torch.tensor(coo.data,dtype=self.dtype,device=self.device),P.shape,check_invariants=True).coalesce();self.PT=self.P.transpose(0,1).coalesce()
        self.coarse_rest=torch.tensor(coarse.rest,dtype=self.dtype,device=self.device);self.coarse_free=torch.tensor(coarse.free,dtype=torch.int64,device=self.device);self.coarse_mask=torch.tensor(coarse.mask,dtype=torch.bool,device=self.device)
        K=coarse.elastic(coarse.rest,True)[2][coarse.free][:,coarse.free].toarray();M=sparse.kron(scalar_mass(coarse),sparse.eye(3),format='csr')[coarse.free][:,coarse.free].toarray()
        self.coarse_symmetry_defect_N_m=float(abs(K-K.T).max())
        if self.coarse_symmetry_defect_N_m>256*np.finfo(float).eps*max(1.,float(abs(K).max())):raise LinearFailure('Graded coarse rest tangent exceeds roundoff symmetry guard')
        # Symmetrization is only preconditioner assembly roundoff, not a change
        # to the physical fine Hessian/operator or a stiffness regularization.
        self.Kc=torch.tensor(.5*(K+K.T),dtype=self.dtype,device=self.device);self.Mc=torch.tensor(.5*(M+M.T),dtype=self.dtype,device=self.device)
        self.coarse_factor_cache={};self.two_level_records=[];self.coarse_cells=(8,4,4);self.coarse_mapping='clamp_graded';self.coarse_first_x_m=float(np.unique(coarse.rest[:coarse.tets.max()+1,0])[1])

    def cg(self,*args,**kwargs):
        n=len(self.linear_records);m=len(self.two_level_records)
        try:return super().cg(*args,**kwargs)
        finally:
            # Correct only descriptive records; arithmetic is the frozen CG.
            for row in self.linear_records[n:]+self.two_level_records[m:]:
                row['preconditioner']='additive_exactnodeblock_plus_graded8x4x4_restP2'
                row['coarse_free_dofs']=int(self.coarse_free.numel())
                row['coarse_mapping']='clamp_graded'


class StreamingGradedCoarseP2(StreamingTwoLevelP2,GradedCoarseTwoLevelP2):
    """Cooperative constructor keeps the streamed geometry-proxy memory bound."""
    pass
