"""Additive exact-node block plus fixed coarse compatible P2 correction.

Original FP64 energy/HVP/mass/residual/line-search/J gates remain unchanged.
The coarse rest tangent is a PRECONDITIONER only. No GPU calls at import and no
runtime monkeypatching. Current explicit scope: canonical block/full left clamp.
"""
import numpy as np
import torch
from scipy import sparse
from experiments.campaign_p2_block_preconditioner import BlockJacobiP2
from experiments.campaign_p2_torch import LinearFailure
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_static_p2_block import shapes
from experiments.diagnose_nonlinear_physical_probes import interpolate


def compatible_prolongation(coarse,points,chunk_size=4096):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite fine material coordinates required')
    rows=[];cols=[];values=[];nvertex=coarse.tets.max()+1
    for start in range(0,len(points),chunk_size):
        p=points[start:start+chunk_size];_,ids,bary=interpolate(coarse.rest[:nvertex],coarse.tets,coarse.rest[None,:nvertex],p)
        rows.append(np.repeat(np.arange(start,start+len(p)),10));cols.append(coarse.elements[ids].ravel());values.append(shapes(bary).ravel())
    P=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(points),len(coarse.rest))).tocsr();P.eliminate_zeros();return P


class TwoLevelP2(BlockJacobiP2):
    def __init__(self,model,device='cpu',maximum_cg_iterations=4000):
        expected=np.repeat((abs(model.rest[:,0])<1e-13)[:,None],3,1)
        if not np.array_equal(model.mask,expected):raise ValueError('Declared full-left fixed clamp required by this coarse model')
        super().__init__(model,device,maximum_cg_iterations)
        coarse=StableP2Block((4,2,2),young=model.young,poisson=model.poisson,density=model.density,gravity=0.,traction=0.,order=3)
        P=compatible_prolongation(coarse,model.rest);self.prolongation_csr=P;coo=P.tocoo();indices=torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.int64,device=self.device)
        self.P=torch.sparse_coo_tensor(indices,torch.tensor(coo.data,dtype=self.dtype,device=self.device),P.shape,check_invariants=True).coalesce();self.PT=self.P.transpose(0,1).coalesce()
        self.coarse_rest=torch.tensor(coarse.rest,dtype=self.dtype,device=self.device);self.coarse_free=torch.tensor(coarse.free,dtype=torch.int64,device=self.device);self.coarse_mask=torch.tensor(coarse.mask,dtype=torch.bool,device=self.device)
        K=coarse.elastic(coarse.rest,True)[2][coarse.free][:,coarse.free].toarray();M=sparse.kron(scalar_mass(coarse),sparse.eye(3),format='csr')[coarse.free][:,coarse.free].toarray()
        self.coarse_symmetry_defect_N_m=float(np.max(abs(K-K.T)))
        if self.coarse_symmetry_defect_N_m>256*np.finfo(float).eps*max(1.,float(np.max(abs(K)))):raise LinearFailure('Coarse rest tangent exceeds roundoff symmetry guard')
        # Symmetrization affects only preconditioner roundoff, never physical H.
        self.Kc=torch.tensor(.5*(K+K.T),dtype=self.dtype,device=self.device);self.Mc=torch.tensor(.5*(M+M.T),dtype=self.dtype,device=self.device);self.coarse_factor_cache={};self.two_level_records=[]

    def prolong(self,x):
        c=torch.zeros_like(self.coarse_rest);c.reshape(-1)[self.coarse_free]=x
        return torch.sparse.mm(self.P,c).masked_fill(self.mask,0.)

    def restrict(self,r):
        c=torch.sparse.mm(self.PT,r.masked_fill(self.mask,0.))
        return c.reshape(-1)[self.coarse_free]

    def coarse_factor(self,mass_factor):
        key=float(mass_factor)
        if not np.isfinite(key) or key<0:raise ValueError('Finite nonnegative original mass factor required')
        if key not in self.coarse_factor_cache:
            L,info=torch.linalg.cholesky_ex(self.Kc+key*self.Mc)
            if int(info)!=0 or not bool(torch.isfinite(L).all()):raise LinearFailure('Coarse original rest-operator factor failed; no regularization')
            self.coarse_factor_cache[key]=L
        return self.coarse_factor_cache[key]

    def make_preconditioner(self,cache,mass_factor):
        blocks=self.nodal_blocks(cache,mass_factor);fixed=self.mask[:,:,None]|self.mask[:,None,:];blocks=blocks.masked_fill(fixed,0.)+torch.diag_embed(self.mask.to(self.dtype));Lb,info=torch.linalg.cholesky_ex(blocks)
        if bool((info!=0).any()) or not bool(torch.isfinite(Lb).all()):raise LinearFailure('Original node block failed; no repair')
        Lc=self.coarse_factor(mass_factor)
        def apply(r):
            rr=r.masked_fill(self.mask,0.);local=torch.cholesky_solve(rr[...,None],Lb).squeeze(-1);coarse=torch.cholesky_solve(self.restrict(rr)[:,None],Lc).squeeze(-1)
            return (local+self.prolong(coarse)).masked_fill(self.mask,0.)
        return apply

    def cg(self,cache,rhs,mass_factor,rtol=1e-8,atol=1e-12,maximum_iterations=None):
        cap=self.maximum_cg_iterations if maximum_iterations is None else maximum_iterations
        if type(cap) is not int or cap<=0:raise ValueError('Positive explicit CG cap required')
        def op(v):
            vv=v.masked_fill(self.mask,0.);a=self.tangent(cache,vv)
            if mass_factor:a=a+mass_factor*self.mass(vv)
            return a.masked_fill(self.mask,0.)
        B=self.make_preconditioner(cache,mass_factor);b=rhs.masked_fill(self.mask,0.);x=torch.zeros_like(b);r=b.clone();z=B(r);p=z.clone();rz=(r*z).sum();tol=max(atol,rtol*float(torch.linalg.vector_norm(r)));restarts=0
        def accepted(k,true):
            row=dict(iterations=k,original_true_residual_N=float(torch.linalg.vector_norm(true)),requested_tolerance_N=tol,residual_restarts=restarts,maximum_iterations=cap,preconditioner='additive_exactnodeblock_plus_coarse4x2x2_restP2',regularization=0.,coarse_free_dofs=int(self.coarse_free.numel()));self.linear_records.append(row);self.two_level_records.append(row.copy());return x,k
        if float(torch.linalg.vector_norm(r))<=tol:return accepted(0,r)
        for k in range(1,cap+1):
            Ap=op(p);curvature=(p*Ap).sum()
            if not bool(torch.isfinite(curvature)) or float(curvature)<=0:raise LinearFailure('Original FP64 tangent has nonpositive search curvature')
            alpha=rz/curvature;x=x+alpha*p;r=r-alpha*Ap
            if float(torch.linalg.vector_norm(r))<=tol:
                actual=b-op(x)
                if float(torch.linalg.vector_norm(actual))<=max(2*tol,1e-11):return accepted(k,actual)
                r=actual;z=B(r);p=z.clone();rz=(r*z).sum();restarts+=1;continue
            z=B(r);new=(r*z).sum();p=z+(new/rz)*p;rz=new
        self.linear_records.append(dict(iterations=cap,original_true_residual_N=float(torch.linalg.vector_norm(b-op(x))),maximum_iterations=cap,failed=True,regularization=0.))
        raise LinearFailure('Original two-level PCG residual exceeds explicit cap')
