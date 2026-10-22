"""Declared fixed-face/patch coarse corrections for compatible P2 mechanics.

The physical operator remains the supplied fine model's original FP64 energy,
full Hessian and consistent mass. Coarse supports affect only preconditioning;
no fine constraint is inferred, softened or moved. The frozen full-left-only
backend remains unchanged. This module does not admit a simulation dataset.
"""
import numpy as np
import torch
from scipy import sparse
from experiments.campaign_block_surface_controls import SUPPORTS, support_groups
from experiments.campaign_p2_coarse_preconditioner import TwoLevelP2, compatible_prolongation
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_p2_torch import LinearFailure


class SupportedTwoLevelP2(TwoLevelP2):
    def __init__(self, model, device='cpu', maximum_cg_iterations=4000, *, support_kind=None):
        # The streaming subclass supplies its explicit declaration before the
        # cooperative constructor reaches this class. Standalone use requires
        # the keyword; masks are never used to guess a support declaration.
        kind = support_kind if support_kind is not None else getattr(self, 'declared_support_kind', None)
        if kind not in SUPPORTS:
            raise ValueError('Explicit declared fixed support kind required')
        groups, active = support_groups(model.rest, kind)
        expected = np.repeat(groups.any(0)[:, None], 3, 1)
        if not np.array_equal(model.mask, expected):
            raise ValueError('Fine mask must exactly match the declared fixed support')
        self.declared_support_kind = kind
        # Initialize the original operator directly. TwoLevelP2.__init__ is
        # intentionally scoped to full-left and is not changed by this branch.
        BlockJacobiP2.__init__(self, model, device, maximum_cg_iterations)
        coarse = StableP2Block((4, 2, 2), young=model.young, poisson=model.poisson,
                               density=model.density, gravity=0., traction=0., order=3)
        coarse_groups, _ = support_groups(coarse.rest, kind)
        coarse.mask = np.repeat(coarse_groups.any(0)[:, None], 3, 1)
        coarse.free = np.flatnonzero(~coarse.mask.ravel())
        P = compatible_prolongation(coarse, model.rest)
        self.prolongation_csr = P
        coo = P.tocoo()
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64, device=self.device)
        self.P = torch.sparse_coo_tensor(indices, torch.tensor(coo.data, dtype=self.dtype, device=self.device),
                                        P.shape, check_invariants=True).coalesce()
        self.PT = self.P.transpose(0, 1).coalesce()
        self.coarse_rest = torch.tensor(coarse.rest, dtype=self.dtype, device=self.device)
        self.coarse_free = torch.tensor(coarse.free, dtype=torch.int64, device=self.device)
        self.coarse_mask = torch.tensor(coarse.mask, dtype=torch.bool, device=self.device)
        K = coarse.elastic(coarse.rest, True)[2][coarse.free][:, coarse.free].toarray()
        M = sparse.kron(scalar_mass(coarse), sparse.eye(3), format='csr')[coarse.free][:, coarse.free].toarray()
        self.coarse_symmetry_defect_N_m = float(np.max(abs(K-K.T)))
        if self.coarse_symmetry_defect_N_m > 256*np.finfo(float).eps*max(1., float(np.max(abs(K)))):
            raise LinearFailure('Coarse original tangent exceeds roundoff symmetry guard')
        self.Kc = torch.tensor(.5*(K+K.T), dtype=self.dtype, device=self.device)
        self.Mc = torch.tensor(.5*(M+M.T), dtype=self.dtype, device=self.device)
        self.coarse_factor_cache = {}
        self.two_level_records = []


class StreamingSupportedP2(StreamingTwoLevelP2, SupportedTwoLevelP2):
    """The same streamed exact operator with explicitly declared fixed supports.

    Cooperative MRO routes the geometry proxy to SupportedTwoLevelP2's new
    checked constructor. Streaming's quadrature/memory behavior is unchanged.
    """
    def __init__(self, model, device='cpu', maximum_cg_iterations=4000,
                 quadrature_chunk_size=8, *, support_kind):
        self.declared_support_kind = support_kind
        super().__init__(model, device, maximum_cg_iterations, quadrature_chunk_size)
