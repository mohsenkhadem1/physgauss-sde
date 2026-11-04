"""Objective one-mode initial natural strain, isolated from existing object banks.

For observed/material-reference deformation F and a fixed material metric A,
W_ref(F,A) = det(A)^(-1) W_NH(F A). A maps a local natural configuration into the
chosen observed reference; F A is the elastic deformation from that natural state.
The prefactor converts natural-volume energy density to observed-reference volume.
Mass and all boundary/body loads stay unchanged (natural density may change with
det(A) under this fixed observed-reference mass convention). Thus this is an initial-strain
hypothesis, NOT residual-force cancellation or an identified global rest mesh.
"""
import numpy as np
import torch
from scripts.external_benchmark.fem import ExternalFEM


def natural_neo_hookean(F,A,mu,lam,*,tangent=False):
    """Energy density, physical first Piola, and dP_iJ/dF_kL, batch [E,...]."""
    if F.shape[-2:]!=(3,3) or A.shape!=F.shape:
        raise ValueError('Matching [E,3,3] physical gradients and natural metrics required')
    Ja=torch.linalg.det(A);elastic=F@A;J=torch.linalg.det(elastic)
    if not bool(torch.isfinite(Ja).all() and torch.isfinite(J).all()) or bool((Ja<=0).any()) or bool((J<=.03).any()):
        raise ValueError('Positive natural metric and nonsingular elastic deformation required')
    logJ=J.log();invT=torch.linalg.inv(elastic).transpose(-1,-2)
    energy=(.5*mu*(elastic.square().sum((-1,-2))-3)-mu*logJ+.5*lam*logJ.square())/Ja
    Pe=mu[:,None,None]*elastic+(lam*logJ-mu)[:,None,None]*invT
    P=(Pe@A.transpose(-1,-2))/Ja[:,None,None]
    C=None
    if tangent:
        I=torch.eye(3,dtype=F.dtype,device=F.device)
        Ce=mu[:,None,None,None,None]*torch.einsum('ik,jl->ijkl',I,I)[None]
        Ce=Ce+lam[:,None,None,None,None]*torch.einsum('eij,ekl->eijkl',invT,invT)
        Ce=Ce+(mu-lam*logJ)[:,None,None,None,None]*torch.einsum('eil,ekj->eijkl',invT,invT)
        C=torch.einsum('eimkn,ejm,eln->eijkl',Ce,A,A)/Ja[:,None,None,None,None]
    return energy,P,C


class InitialPrestrainFEM(ExternalFEM):
    """ExternalFEM with only its bulk constitutive response replaced.

    Appends one unconstrained latent theta[i_prestrain]; actual amplitude is
    max_log_stretch*tanh(latent). A_e = I + (exp(s*h_e)-1) a a^T,
    with h_e in [0,1] computed from INITIAL material height along unit axis a.
    The axis is supplied material-frame metadata, never recomputed from x(t).
    The mode is deliberately restricted; initial prestrain is not identified
    uniquely from video. It may be incompatible across elements, representing a
    natural metric/residual strain rather than a globally integrable rest mesh.
    """
    def __init__(self,data,device='cpu',*,material_axis=(0.,0.,1.),max_log_stretch=.3,**kwargs):
        if not np.isfinite(max_log_stretch) or max_log_stretch<=0:
            raise ValueError('Positive finite prestrain bound required')
        super().__init__(data,device=device,**kwargs)
        axis=self.t(material_axis)
        if axis.shape!=(3,) or not bool(torch.isfinite(axis).all()) or float(axis.norm())<=0:
            raise ValueError('Finite nonzero material axis required')
        self.material_axis=axis/axis.norm();self.max_log_stretch=float(max_log_stretch)
        heights=self.rest@self.material_axis
        extent=heights.max()-heights.min()
        if float(extent)<=1e-12:raise ValueError('Nonzero extent along material prestrain axis required')
        self.material_height=((heights-heights.min())/extent)[self.elements].mean(-1)
        self.i_prestrain=self.i_field+1+int(self.contact)
        self.parameter_count=self.i_prestrain+1

    def natural_metric(self,theta):
        if theta.ndim!=1 or len(theta)!=self.parameter_count:
            raise ValueError('Append one prestrain latent to the original parameter vector')
        latent=theta[self.i_prestrain]
        if not bool(torch.isfinite(latent)):raise ValueError('Finite initial-prestrain latent required')
        s=self.max_log_stretch*torch.tanh(latent)
        stretch=torch.exp(s*self.material_height)
        direction=self.material_axis[:,None]*self.material_axis[None,:]
        return self.eye[None]+(stretch-1)[:,None,None]*direction[None]

    def bulk_response(self,x,theta,A,*,tangent=False):
        material=torch.exp(self.feature@theta[self.i_material:self.i_field])
        mu=material/2.7;lam=material*.35/(1.35*.3)
        F=torch.einsum('eai,eaj->eij',x[self.elements],self.grad)
        W,P,C=natural_neo_hookean(F,A,mu,lam,tangent=tangent)
        local=torch.einsum('e,eij,eaj->eai',self.vol,P,self.grad)
        residual=torch.zeros_like(x).index_add(0,self.elements.reshape(-1),local.reshape(-1,3))
        H=None
        if tangent:
            localH=torch.einsum('e,eijkl,eaj,ebl->eaibk',self.vol,C,self.grad,self.grad)
            H=torch.zeros((self.ndof,self.ndof),dtype=self.dtype,device=self.device).index_put((self.rows,self.cols),localH.reshape(-1),accumulate=True)
        return (self.vol*W).sum(),residual,H

    def evaluate(self,x,theta,frame,tangent=False):
        A=self.natural_metric(theta);base_theta=theta[:self.i_prestrain]
        # Preserve every external term exactly, including gravity, floor, grip,
        # optional fields/compliance, inertial terms and the inherited weak gauge.
        energy,residual,H=super().evaluate(x,base_theta,frame,tangent)
        identity=self.eye[None].expand(len(self.elements),-1,-1)
        oldE,oldR,oldH=self.bulk_response(x,base_theta,identity,tangent=tangent)
        newE,newR,newH=self.bulk_response(x,base_theta,A,tangent=tangent)
        energy=energy-oldE+newE;residual=residual-oldR+newR
        if tangent:H=H-oldH+newH
        return energy,residual,H

    def min_elastic_jacobian(self,x,theta):
        F=torch.einsum('eai,eaj->eij',x[self.elements],self.grad)
        return float(torch.linalg.det(F@self.natural_metric(theta)).detach().min())
