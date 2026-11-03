import numpy as np
import torch
import taichi as ti
from types import SimpleNamespace
from scripts.complex_scenes_v18.particle_tracks import binding
from scripts.complex_scenes_v18.uniphy_track_loss import TrackedLoss


def test_native_observation_loss_and_particle_adjoint_match_torch():
    ti.init(arch=ti.cpu,cpu_max_num_threads=1)
    rng=np.random.default_rng(12);indices,weights=binding(rng.uniform(0,1,(7,3)),(3,3,3));observed=rng.normal(size=(4,7,3)).astype(np.float32);values=rng.normal(size=(27,7,3)).astype(np.float32);steps=np.array([0,2,4,6])
    sim=SimpleNamespace(x=ti.Vector.field(3,ti.f32,shape=values.shape[:2],needs_grad=True),loss=ti.field(ti.f32,shape=(),needs_grad=True));sim.x.from_numpy(values)
    loss=TrackedLoss(sim,indices,weights,observed,steps);loss.compute();sim.loss.grad[None]=1.;loss.compute.grad()
    x=torch.tensor(values,requires_grad=True);i=torch.tensor(indices.astype(np.int64));w=torch.tensor(weights)
    pred=torch.stack([(x[i,int(t)]*w[...,None]).sum(1)for t in steps[1:]])
    expected=(pred-torch.tensor(observed[1:])).square().sum(-1).mean()*1e6;expected.backward()
    np.testing.assert_allclose(float(sim.loss[None]),float(expected.detach()),rtol=2e-6)
    np.testing.assert_allclose(sim.x.grad.to_numpy(),x.grad.numpy(),rtol=2e-5,atol=.1)
    ti.reset()
