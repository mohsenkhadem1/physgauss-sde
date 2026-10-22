"""Float64 matrix-free compatible T10 elasticity and exact consistent mass.

Same stable Neo-Hookean W/P as the pinned Newton comparison. Full determinant
cross terms are retained in the tangent action. No constitutive clipping,
row-sum mass lumping, hidden damping or contact is introduced.
"""
from __future__ import annotations
import numpy as np
import torch
from experiments.campaign_static_p2_block import EDGES,shape_gradients,shapes


def consistent_t10():
    B=np.zeros((10,10),float)
    B[:4,:4]=1.;np.fill_diagonal(B[:4,:4],6.)
    for i in range(4):
        for j,e in enumerate(EDGES):B[i,4+j]=B[4+j,i]=-4. if i in e else -6.
    for i,a in enumerate(EDGES):
        for j,b in enumerate(EDGES):B[4+i,4+j]=32. if i==j else 16. if set(a)&set(b) else 8.
    return B/420


def material_torch(F,young,poisson):
    I=torch.eye(3,device=F.device,dtype=F.dtype);G=F-I
    tr=G.diagonal(dim1=-2,dim2=-1).sum(-1)
    second=.5*(tr*tr-(G*G.transpose(-1,-2)).sum((-1,-2)))
    detG=(G[..., :,0]*torch.linalg.cross(G[..., :,1],G[..., :,2],dim=-1)).sum(-1)
    jm=tr+second+detG;cof=torch.stack([torch.linalg.cross(F[..., :,1],F[..., :,2],dim=-1),torch.linalg.cross(F[..., :,2],F[..., :,0],dim=-1),torch.linalg.cross(F[..., :,0],F[..., :,1],dim=-1)],-1)
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson));kappa=lam+mu;s=kappa*jm-mu
    W=mu*(.5*(G*G).sum((-1,-2))-second-detG)+.5*kappa*jm*jm
    P=mu*G+s[...,None,None]*(cof-I)+kappa*jm[...,None,None]*I
    return W,P,cof,s,1+jm


class LinearFailure(RuntimeError):pass
class NonlinearFailure(RuntimeError):pass


class TorchP2:
    def __init__(self,model,device='cpu'):
        self.device=torch.device(device);self.dtype=torch.float64
        tensor=lambda a,dtype=self.dtype:torch.as_tensor(np.asarray(a).copy(),dtype=dtype,device=self.device)
        self.rest=tensor(model.rest);self.elements=tensor(model.elements,torch.int64);self.mask=tensor(model.mask,torch.bool)
        self.young=model.young;self.poisson=model.poisson;self.density=model.density
        self.grad=tensor(np.stack([shape_gradients(b[None],model.linear_gradients) for b in model.bary]))
        self.check_grad=tensor(np.stack([shape_gradients(b[None],model.linear_gradients) for b in model.check_bary]))
        self.weights=tensor(model.weights[:,None]*model.volumes[None,:]);self.volumes=tensor(model.volumes)
        self.mass_local=tensor(model.density*model.volumes[:,None,None]*consistent_t10()[None])
        self.mass_diagonal=self.scatter(self.mass_local.diagonal(dim1=-2,dim2=-1)[...,None].expand(-1,-1,3))
        self.force=tensor(model.load);self.minimum_J=model.minimum_J;self.energy_scale=model.young*float(np.sum(model.volumes))
        self.identity=torch.eye(3,device=self.device,dtype=self.dtype)

    def scatter(self,values):
        out=torch.zeros_like(self.rest)
        return out.index_add_(0,self.elements.reshape(-1),values.reshape(-1,3))

    def mass(self,v):return self.scatter(torch.einsum('eab,ebi->eai',self.mass_local,v[self.elements]))

    def evaluate(self,u):
        if u.shape!=self.rest.shape or u.dtype!=self.dtype:raise ValueError('Float64 nodal displacement required')
        local=u[self.elements];F=self.identity+torch.einsum('eai,qeaj->qeij',local,self.grad)
        W,P,cof,s,J=material_torch(F,self.young,self.poisson)
        Fcheck=self.identity+torch.einsum('eai,qeaj->qeij',local,self.check_grad)
        Jcheck=torch.linalg.det(Fcheck)
        jmin=torch.minimum(J.min(),Jcheck.min())
        if not bool(torch.isfinite(J).all()) or not bool(torch.isfinite(Jcheck).all()) or float(jmin)<=self.minimum_J:raise ValueError('Positive Jacobian guard failed; no repair')
        energy=(self.weights*W).sum();gradient=self.scatter(torch.einsum('qe,qeij,qeaj->eai',self.weights,P,self.grad))
        return dict(energy=energy,gradient=gradient,F=F,cofactor=cof,s=s,minimum_J=jmin)

    def tangent(self,cache,v):
        dF=torch.einsum('eai,qeaj->qeij',v[self.elements],self.grad);F=cache['F'];cof=cache['cofactor']
        dc=torch.stack([torch.linalg.cross(dF[..., :,1],F[..., :,2],dim=-1)+torch.linalg.cross(F[..., :,1],dF[..., :,2],dim=-1),torch.linalg.cross(dF[..., :,2],F[..., :,0],dim=-1)+torch.linalg.cross(F[..., :,2],dF[..., :,0],dim=-1),torch.linalg.cross(dF[..., :,0],F[..., :,1],dim=-1)+torch.linalg.cross(F[..., :,0],dF[..., :,1],dim=-1)],-1)
        mu=self.young/(2*(1+self.poisson));lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson));kappa=lam+mu
        dP=mu*dF+kappa*(cof*dF).sum((-1,-2))[...,None,None]*cof+cache['s'][...,None,None]*dc
        return self.scatter(torch.einsum('qe,qeij,qeaj->eai',self.weights,dP,self.grad))

    def diagonal(self,cache):
        mu=self.young/(2*(1+self.poisson));lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson));kappa=lam+mu
        cg=torch.einsum('qeij,qeaj->qeai',cache['cofactor'],self.grad)
        local=(self.weights[...,None,None]*(mu*(self.grad*self.grad).sum(-1)[...,None]+kappa*cg*cg)).sum(0)
        return self.scatter(local)

    def cg(self,cache,rhs,mass_factor,rtol=1e-8,atol=1e-12,maximum_iterations=1000):
        def op(v):
            vv=v.masked_fill(self.mask,0.);out=self.tangent(cache,vv)
            if mass_factor:out=out+mass_factor*self.mass(vv)
            return out.masked_fill(self.mask,0.)
        diagonal=self.diagonal(cache)+mass_factor*self.mass_diagonal
        if not bool((diagonal[~self.mask]>0).all()):raise LinearFailure('Nonpositive original diagonal')
        safe=diagonal.masked_fill(self.mask,1.)
        x=torch.zeros_like(rhs);r=rhs.masked_fill(self.mask,0.);z=r/safe;p=z.clone();rz=(r*z).sum()
        tolerance=max(atol,rtol*float(torch.linalg.vector_norm(r)))
        if float(torch.linalg.vector_norm(r))<=tolerance:return x,0
        for k in range(1,maximum_iterations+1):
            Ap=op(p);curvature=(p*Ap).sum()
            if not bool(torch.isfinite(curvature)) or float(curvature)<=0:raise LinearFailure('Original tangent has nonpositive search curvature')
            alpha=rz/curvature;x=x+alpha*p;r=r-alpha*Ap
            if float(torch.linalg.vector_norm(r))<=tolerance:
                actual=rhs.masked_fill(self.mask,0.)-op(x)
                if float(torch.linalg.vector_norm(actual))<=max(2*tolerance,1e-11):return x,k
                # Finite precision restart retains the original operator.
                r=actual
            z=r/safe;new=(r*z).sum();p=z+(new/rz)*p;rz=new
        raise LinearFailure('Original PCG residual exceeds iteration cap')

    def solve(self,force=None,initial=None,target=None,effective_dt=None,force_tolerance=1e-10,maximum_iterations=25):
        force=self.force if force is None else force
        u=torch.zeros_like(self.rest) if initial is None else initial.clone()
        if not bool(torch.isfinite(u).all()) or bool((u[self.mask]!=0).any()):raise ValueError('Finite exactly constrained initial displacement required')
        factor=0. if effective_dt is None else 1/effective_dt**2
        if factor and target is None:raise ValueError('Dynamic inertial target required')
        history=[]
        def objective(x):
            cache=self.evaluate(x);res=cache['gradient']-force;value=cache['energy']-(force*x).sum()
            if factor:
                d=x-target;md=self.mass(d);res=res+factor*md;value=value+.5*factor*(d*md).sum()
            return cache,res.masked_fill(self.mask,0.),value
        for it in range(maximum_iterations+1):
            cache,r,potential=objective(u);norm=float(torch.linalg.vector_norm(r));row=dict(iteration=it,residual_N=norm,potential_J=float(potential),minimum_J=float(cache['minimum_J']));history.append(row)
            if norm<=force_tolerance:return dict(displacement=u,positions=self.rest+u,cache=cache,residual=r,history=history,iterations=it)
            if it==maximum_iterations:raise NonlinearFailure('Original nonlinear residual exceeds cap: '+str(history[-1]))
            direction,cg_steps=self.cg(cache,-r,factor,rtol=min(.01,max(1e-8,.1*norm**.5)));row['cg_iterations']=cg_steps
            slope=float((r*direction).sum())
            if not np.isfinite(slope) or slope>=0:raise NonlinearFailure('Original Newton direction is not descent')
            for k in range(30):
                alpha=2.**(-k);trial=u+alpha*direction
                try:cc,rr,vv=objective(trial)
                except ValueError:continue
                energy_roundoff=128*np.finfo(float).eps*max(abs(float(potential)),self.energy_scale)
                if float(vv)<=float(potential)+1e-4*alpha*slope or(abs(float(vv-potential))<=energy_roundoff and float(torch.linalg.vector_norm(rr))<norm):
                    u=trial;row['alpha']=alpha;break
            else:raise NonlinearFailure('Positive-J original-objective line search failed')
        raise AssertionError('Unreachable')
