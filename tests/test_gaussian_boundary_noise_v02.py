import torch
from scripts.gaussian_boundary_flow_v02.noise import noisy_context


def test_exactly_one_noise_draw_and_normalization():
    clean=torch.zeros(100000,2);mean=torch.zeros(2);scale=torch.tensor([2.,.5])
    x=noisy_context(clean,mean,scale,.01,torch.Generator().manual_seed(19))*scale
    assert torch.allclose(x.std(0),torch.full((2,),.01),rtol=.01)
    assert torch.count_nonzero(clean)==0
    assert torch.equal(noisy_context(clean,mean,scale,0),clean)
