import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from scripts.complex_scenes_v18.differentiable_spring import DifferentiableSpringScene


def test_native_adapter_rollout_parity_and_checkpoint_gradient():
    torch.set_num_threads(1);sim=DifferentiableSpringScene('clamp_release',per_edge=False);dt=25e-6
    state=sim.initial()
    for i in range(20):state=sim.step(state,i,dt,training=False);sim.base.step(dt)
    np.testing.assert_allclose(state[0].detach().cpu(),sim.base.x[0].detach().cpu(),atol=1e-8,rtol=1e-6)
    np.testing.assert_allclose(state[1].detach().cpu(),sim.base.v[0].detach().cpu(),atol=1e-7,rtol=1e-5)
    def objective(use_checkpoint):
        state=list(sim.initial());state[0]=state[0]+.01*(state[0]-state[0].mean(0));state=tuple(state)
        def segment(*s):
            for i in range(5):s=sim.step(s,i,dt,training=True)
            return s
        for _ in range(2):state=checkpoint(segment,*state,use_reentrant=False)if use_checkpoint else segment(*state)
        loss=(sim.positions(state)-sim.positions(sim.initial())).square().sum()
        return loss,torch.autograd.grad(loss,sim.native[0].global_k)[0]
    a,ga=objective(False);b,gb=objective(True)
    torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb);assert torch.isfinite(ga)and ga.abs()>0
