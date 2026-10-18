"""Exact constant-input and acceleration-noise integrals for damped modes.

Each mode is balanced before a small augmented matrix exponential. Pairwise
Kronecker sums integrate every off-diagonal spectral covariance as well as the
diagonal. Their spectra are stable; no anti-damped Van Loan block is needed.
Used by the corrected PAC transition density; constant drift/noise is frozen over each interval.
"""
import torch


def integrals(omega_squared,damping,duration):
    """Return drive [B,2M,M] and covariance kernel [B,M,M,2,2]."""
    b,m=omega_squared.shape
    if damping.shape!=(b,) or duration<=0:raise ValueError('Invalid modal dimensions or interval')
    if bool((omega_squared<0).any()) or bool((damping<0).any()):raise ValueError('Modes must be stable')
    scale=(omega_squared+damping[:,None].square()+duration**-2).sqrt()
    a=omega_squared.new_zeros(b,m,2,2)
    a[...,0,1]=scale;a[...,1,0]=-omega_squared/scale;a[...,1,1]=-damping[:,None]
    eye=torch.eye(2,dtype=a.dtype,device=a.device)
    # Row-major vec(X): dX/dt = A_i X + X A_j^T.
    generator=(a[:,:,None,:,None,:,None]*eye[None,None,None,None,:,None,:]
        +eye[None,None,None,:,None,:,None]*a[:,None,:,None,:,None,:]).reshape(b,m,m,4,4)
    augmented=a.new_zeros(b,m,m,5,5);augmented[...,:4,:4]=generator
    augmented[...,3,4]=1.
    kernel=torch.matrix_exp(augmented*duration)[...,:4,4].reshape(b,m,m,2,2)
    physical=torch.stack((scale.reciprocal(),torch.ones_like(scale)),-1)
    kernel=kernel*physical[:,:,None,:,None]*physical[:,None,:,None,:]
    kernel=(kernel+kernel.permute(0,2,1,4,3))/2
    affine=a.new_zeros(b,m,3,3);affine[...,:2,:2]=a;affine[...,1,2]=1.
    response=torch.matrix_exp(affine*duration)[...,:2,2]*physical
    drive=a.new_zeros(b,2*m,m)
    drive[:,:m]=torch.diag_embed(response[...,0]);drive[:,m:]=torch.diag_embed(response[...,1])
    return drive,kernel


def covariance(spectral,kernel):
    """Propagate a possibly time-varying full modal acceleration spectrum."""
    b,t,m,_=spectral.shape
    blocks=spectral[:,:,None,None]*kernel.permute(0,3,4,1,2)[:,None]
    result=blocks.permute(0,1,2,4,3,5).reshape(b,t,2*m,2*m)
    return (result+result.transpose(-1,-2))/2
