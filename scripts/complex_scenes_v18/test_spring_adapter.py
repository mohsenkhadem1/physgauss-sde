"""Run with the baseline environment on an explicitly selected CUDA device."""
import types
import numpy as np
import torch
from scripts.complex_scenes_v18.spring_scene import SpringScene


def test_native_step_unchanged_and_external_force_units():
    sim=SpringScene('clamp_release');m=sim.native[0];x=sim.x[0];v=sim.v[0]
    K=10**m.global_k.expand_as(m.origin_len)/(m.origin_len+m.eps)
    damp=10**m.damp/(m.origin_len+m.eps);mass=10**m.global_m;dt=25e-6
    args=(K,mass,torch.sigmoid(m.rebound_k),torch.sigmoid(m.fric_k),damp,dt)
    wrapper=m.compute_force
    with torch.no_grad():
        xa,va=m.step(x.clone(),v.clone(),*args)
        m.compute_force=types.MethodType(type(m).compute_force,m)
        xb,vb=m.step(x.clone(),v.clone(),*args)
        assert torch.equal(xa,xb)and torch.equal(va,vb)
        m.compute_force=wrapper;m.scene_external_force[:]=torch.tensor([.01,-.02,.03],device='cuda')
        xc,vc=m.step(x.clone(),v.clone(),*args)
        torch.testing.assert_close(vc-vb,dt*m.scene_external_force/mass[:,None],atol=1e-9,rtol=1e-5)
        torch.testing.assert_close(xc-xb,dt*(vc-vb),atol=8e-9,rtol=1e-4)
        assert np.isclose(float(mass.sum()),.27,rtol=1e-6)
