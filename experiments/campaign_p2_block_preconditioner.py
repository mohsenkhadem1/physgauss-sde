"""Exact nodal3x3 block-Jacobi preconditioner, original full P2 operator.

Numerical solver only: unchanged stable-NH energy, mass, Hessian, line search,
residual/J/constraint gates. No regularization/eigenvalue projection. The larger
CG cap is explicit. Frozen TorchP2 is inherited, never patched.
"""
import torch
from experiments.campaign_p2_torch import TorchP2,LinearFailure


class BlockJacobiP2(TorchP2):
    def __init__(self,model,device='cpu',maximum_cg_iterations=4000):
        if type(maximum_cg_iterations) is not int or maximum_cg_iterations<=0:raise ValueError('Positive explicit CG cap required')
        super().__init__(model,device);self.maximum_cg_iterations=maximum_cg_iterations;self.linear_records=[]

    def nodal_blocks(self,cache,mass_factor=0.):
        """Exact original diagonal node blocks. Same-node d²detF vanishes."""
        mu=self.young/(2*(1+self.poisson));lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson));kappa=lam+mu
        cg=torch.einsum('qeij,qeaj->qeai',cache['cofactor'],self.grad)
        local=kappa*torch.einsum('qe,qeai,qeaj->eaij',self.weights,cg,cg)
        isotropic=(self.weights[...,None]*mu*(self.grad*self.grad).sum(-1)).sum(0)
        I=torch.eye(3,dtype=self.dtype,device=self.device);local=local+isotropic[...,None,None]*I
        blocks=torch.zeros((len(self.rest),3,3),dtype=self.dtype,device=self.device).index_add_(0,self.elements.reshape(-1),local.reshape(-1,3,3))
        if mass_factor:blocks=blocks+mass_factor*self.mass_diagonal[:,0,None,None]*I
        return blocks

    def cg(self,cache,rhs,mass_factor,rtol=1e-8,atol=1e-12,maximum_iterations=None):
        cap=self.maximum_cg_iterations if maximum_iterations is None else maximum_iterations
        if type(cap) is not int or cap<=0:raise ValueError('Positive CG cap required')
        def op(v):
            vv=v.masked_fill(self.mask,0.);out=self.tangent(cache,vv)
            if mass_factor:out=out+mass_factor*self.mass(vv)
            return out.masked_fill(self.mask,0.)
        blocks=self.nodal_blocks(cache,mass_factor);fixed=self.mask[:,:,None]|self.mask[:,None,:]
        blocks=blocks.masked_fill(fixed,0.)+torch.diag_embed(self.mask.to(self.dtype))
        L,info=torch.linalg.cholesky_ex(blocks)
        if bool((info!=0).any()) or not bool(torch.isfinite(L).all()):raise LinearFailure('Original nodal block Cholesky failed; no repair')
        def precondition(r):return torch.cholesky_solve(r[...,None],L).squeeze(-1).masked_fill(self.mask,0.)
        b=rhs.masked_fill(self.mask,0.);x=torch.zeros_like(b);r=b.clone();z=precondition(r);p=z.clone();rz=(r*z).sum();tolerance=max(atol,rtol*float(torch.linalg.vector_norm(r)));restarts=0
        def accepted(k,true):
            self.linear_records.append(dict(iterations=k,original_true_residual_N=float(torch.linalg.vector_norm(true)),requested_tolerance_N=tolerance,residual_restarts=restarts,maximum_iterations=cap,preconditioner='exact_nodal3x3_block_Jacobi',regularization=0.));return x,k
        if float(torch.linalg.vector_norm(r))<=tolerance:return accepted(0,r)
        for k in range(1,cap+1):
            Ap=op(p);curvature=(p*Ap).sum()
            if not bool(torch.isfinite(curvature)) or float(curvature)<=0:raise LinearFailure('Original tangent has nonpositive search curvature')
            alpha=rz/curvature;x=x+alpha*p;r=r-alpha*Ap
            if float(torch.linalg.vector_norm(r))<=tolerance:
                actual=b-op(x)
                if float(torch.linalg.vector_norm(actual))<=max(2*tolerance,1e-11):return accepted(k,actual)
                r=actual;z=precondition(r);p=z.clone();rz=(r*z).sum();restarts+=1;continue
            z=precondition(r);new=(r*z).sum();p=z+(new/rz)*p;rz=new
        self.linear_records.append(dict(iterations=cap,original_true_residual_N=float(torch.linalg.vector_norm(b-op(x))),maximum_iterations=cap,failed=True,regularization=0.))
        raise LinearFailure('Original block-PCG residual exceeds explicit iteration cap')
