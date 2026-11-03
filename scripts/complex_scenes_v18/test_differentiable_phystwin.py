import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from scripts.complex_scenes_v18.differentiable_phystwin import DifferentiablePhysTwinScene


def test_native_replay_and_checkpoint_gradient_agree():
    torch.set_num_threads(1);sim=DifferentiablePhysTwinScene('clamp_release');dt=25e-6
    def segment(*state):
        for i in range(20):state=sim.step(state,i,dt)
        return state
    state=segment(*sim.initial());loss=sim.positions(state).square().sum();g=torch.autograd.grad(loss,sim.parameters)
    other=checkpoint(segment,*sim.initial(),use_reentrant=False);g2=torch.autograd.grad(sim.positions(other).square().sum(),sim.parameters)
    for a,b in zip(g,g2):torch.testing.assert_close(a,b,rtol=2e-4,atol=1e-8)
    native=sim.native_replay()
    for _ in range(20):native.step(dt)
    np.testing.assert_allclose(native.sample_surface(),sim.positions(state).detach().cpu().numpy(),rtol=0,atol=3e-6)
