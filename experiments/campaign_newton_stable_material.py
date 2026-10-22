"""Exact installed Newton1.7 VBD stable-NH continuum law, independent NumPy.

Authored mu/lambda are Lame parameters. Kernel kappa=lambda+mu. Rest-zero
energy differs from logJ NH at finite strain. Full Hessian retains d²J;
VBD's omission is exact only after same-P1-vertex diagonal contraction.
"""
import itertools
import numpy as np
from scipy.optimize import brentq
from scipy import sparse
from experiments.campaign_static_uniaxial_bar import lame
from experiments.campaign_static_supported_block import StaticSupportedBlock,SIZE
from experiments.campaign_static_p2_block import QuadraticBlock,shape_gradients
EPS=np.zeros((3,3,3))
for p in itertools.permutations(range(3)):
    EPS[p]=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
I=np.eye(3);IDENTITY4=np.einsum('ik,JL->iJkL',I,I)


def stable_material(F,young,poisson,tangent=True):
    F=np.asarray(F,dtype=float)
    if F.shape[-2:]!=(3,3) or not np.isfinite(F).all():raise ValueError('Finite3x3deformation required')
    mu,lam=lame(young,poisson);kappa=lam+mu;G=F-I;tr=np.trace(G,axis1=-2,axis2=-1)
    second=.5*(tr*tr-np.einsum('...ij,...ji->...',G,G));detG=np.linalg.det(G);jminus=tr+second+detG
    cof=np.stack([np.cross(F[..., :,1],F[..., :,2]),np.cross(F[..., :,2],F[..., :,0]),np.cross(F[..., :,0],F[..., :,1])],-1)
    W=mu*(.5*np.sum(G*G,axis=(-2,-1))-second-detG)+.5*kappa*jminus*jminus
    s=kappa*jminus-mu;P=mu*G+s[...,None,None]*(cof-I)+kappa*jminus[...,None,None]*I
    A=None
    if tangent:
        dcof=np.einsum('ikm,JLn,...mn->...iJkL',EPS,EPS,F)
        A=mu*IDENTITY4+kappa*np.einsum('...iJ,...kL->...iJkL',cof,cof)+s[...,None,None,None,None]*dcof
    return W,P,A


def uniaxial_oracle(force=1.,young=5000.,poisson=.35,size=SIZE):
    mu,lam=lame(young,poisson);kappa=lam+mu;L,W,H=size
    def lateral_square(s):return ((kappa+mu)*s-mu)/(kappa*s*s)
    def axial(s):return mu*(s-lateral_square(s)/s)
    s=brentq(lambda s:W*H*axial(s)-force,1.,2.,xtol=5e-15,rtol=1e-14) if force else 1.
    y=lateral_square(s);F=np.diag([s,np.sqrt(y),np.sqrt(y)]);energy,stress,_=stable_material(F,young,poisson,False)
    return {'stretch':s,'lateral_stretch':float(np.sqrt(y)),'J':s*y,'Piola_axial_Pa':axial(s),'energy_J':float(L*W*H*energy),'lateral_stress_Pa':float(stress[1,1])}


def p1_tet_response(rest,positions,young,poisson):
    X=np.asarray(rest);x=np.asarray(positions);Dm=(X[1:]-X[0]).T;V=np.linalg.det(Dm)/6
    if V<=0:raise ValueError('Positive rest tetrahedron required')
    inv=np.linalg.inv(Dm);g=np.concatenate([-inv.sum(0)[None],inv],0);F=x.T@g
    W,P,A=stable_material(F,young,poisson)
    grad=V*np.einsum('iJ,aJ->ai',P,g);H=V*np.einsum('aJ,iJkL,bL->aibk',g,A,g).reshape(12,12)
    return float(V*W),-grad,H,F


class StableP1Block(StaticSupportedBlock):
    solve=QuadraticBlock.solve
    def elastic(self,positions,tangent=False):
        F=I+np.einsum('eai,eaJ->eiJ',(np.asarray(positions)-self.rest)[self.tets],self.gradients);J=np.linalg.det(F)
        if not np.isfinite(J).all() or np.any(J<=self.minimum_J):raise ValueError('PositiveJreferenceguard, no repair')
        W,P,A=stable_material(F,self.young,self.poisson,tangent);ge=np.einsum('e,eiJ,eaJ->eai',self.volumes,P,self.gradients);g=np.zeros_like(self.rest);np.add.at(g,self.tets,ge);H=None
        if tangent:
            blocks=np.einsum('e,eaJ,eiJkL,ebL->eaibk',self.volumes,self.gradients,A,self.gradients,optimize=True)
            H=sparse.coo_matrix((blocks.ravel(),(self.rows,self.cols)),shape=(self.rest.size,self.rest.size)).tocsr()
        return float(self.volumes@W),g,H,J


class StableP2Block(QuadraticBlock):
    def elastic(self,positions,tangent=False):
        x=np.asarray(positions);u=x-self.rest
        if x.shape!=self.rest.shape or not np.isfinite(x).all():raise ValueError('FiniteP2positions required')
        energy=0.;gradient=np.zeros_like(x);Hvalues=np.zeros((len(self.elements),30,30)) if tangent else None;allJ=[]
        for start in range(0,len(self.elements),256):
            sl=slice(start,start+256);T=self.elements[sl];ue=u[T];g0=self.linear_gradients[sl];V=self.volumes[sl];ge=np.zeros_like(ue);He=np.zeros((len(T),10,3,10,3)) if tangent else None
            for b in self.check_bary:
                g=shape_gradients(b[None],g0);J=np.linalg.det(I+np.einsum('eai,eaJ->eiJ',ue,g))
                if not np.isfinite(J).all() or np.any(J<=self.minimum_J):raise ValueError('P2nodalJacobian guard; no repair')
                allJ.append(J)
            for b,w in zip(self.bary,self.weights):
                g=shape_gradients(b[None],g0);F=I+np.einsum('eai,eaJ->eiJ',ue,g);J=np.linalg.det(F)
                if not np.isfinite(J).all() or np.any(J<=self.minimum_J):raise ValueError('P2quadratureJacobian guard; no repair')
                allJ.append(J);W,P,A=stable_material(F,self.young,self.poisson,tangent);v=V*w;energy+=float(v@W);ge+=np.einsum('e,eiJ,eaJ->eai',v,P,g)
                if tangent:He+=np.einsum('e,eaJ,eiJkL,ebL->eaibk',v,g,A,g,optimize=True)
            np.add.at(gradient,T,ge)
            if tangent:Hvalues[sl]=He.reshape(-1,30,30)
        H=None if not tangent else sparse.coo_matrix((Hvalues.ravel(),(self.rows,self.cols)),shape=(x.size,x.size)).tocsr()
        return energy,gradient,H,np.concatenate(allJ)
