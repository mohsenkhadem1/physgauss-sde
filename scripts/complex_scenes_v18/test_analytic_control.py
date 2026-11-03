import torch
from scripts.complex_scenes_v18.analytic_control import AnalyticControl
from scripts.compositional_neural_mechanics.mechanics import AnalyticNeoHookean


def test_control_matches_reference_family_without_known_parameters():
    F=torch.tensor([[[1.1,.1,0],[0,.9,0],[0,0,1.05]]],dtype=torch.float64,requires_grad=True)
    z=torch.zeros((1,4),dtype=torch.float64);z[:,0]=.2
    expected=AnalyticNeoHookean(.3)(F,torch.log(torch.tensor([[10000.]],dtype=torch.float64))+.2)
    actual=AnalyticControl().energy(F,z)
    torch.testing.assert_close(actual,expected)
    a=torch.autograd.grad(actual.sum(),F,retain_graph=True)[0];b=torch.autograd.grad(expected.sum(),F)[0]
    torch.testing.assert_close(a,b)
