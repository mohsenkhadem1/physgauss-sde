"""Single-noise observation contract for simulation-based inference."""
import torch


def noisy_context(clean,mean,scale,sigma,generator=None):
    if sigma<0 or bool((scale<=0).any()):raise ValueError('Nonnegative noise and positive scales required')
    noise=torch.randn(clean.shape,dtype=clean.dtype,device=clean.device,generator=generator)*sigma
    return (clean+noise-mean)/scale
