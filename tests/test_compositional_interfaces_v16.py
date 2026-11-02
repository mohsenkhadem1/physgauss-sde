import numpy as np
import torch
from scripts.compositional_neural_mechanics.interfaces import AnalyticInterfaceLaw,SharedInterfaceLaw,interface_derivatives


def test_reference_rotation_and_convex_tangent():
    torch.manual_seed(911)
    for law in (AnalyticInterfaceLaw(),SharedInterfaceLaw()):
        z=torch.tensor([[np.log(700.),.6,0.],[np.log(1500.),.8,1.]],dtype=torch.float64)
        zero=torch.zeros(2,3,dtype=torch.float64)
        W,t,H=interface_derivatives(law,zero,z,True)
        torch.testing.assert_close(W,torch.zeros_like(W),atol=1e-15,rtol=0)
        torch.testing.assert_close(t,zero,atol=1e-15,rtol=0)
        torch.testing.assert_close(H,z[:,0].exp()[:,None,None]*torch.eye(3,dtype=torch.float64),atol=1e-9,rtol=1e-12)
        x=torch.tensor([[.001,-.0003,.0005],[-.002,.001,.0004]],dtype=torch.float64)
        Q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        W,t,H=interface_derivatives(law,x,z,True)
        Wr,tr,Hr=interface_derivatives(law,x@Q,z,True)
        torch.testing.assert_close(Wr,W);torch.testing.assert_close(tr,t@Q)
        torch.testing.assert_close(H,H.transpose(-1,-2))
        assert torch.linalg.eigvalsh(H).min()>0


def test_work_conjugacy_and_tangent_directional_derivative():
    law=SharedInterfaceLaw();z=torch.tensor([np.log(900.),.7,.4],dtype=torch.float64)
    x=torch.tensor([.001,.0004,-.0002],dtype=torch.float64);v=torch.tensor([.2,-.3,.4],dtype=torch.float64)
    W,t,H=interface_derivatives(law,x,z,True);eps=1e-7
    plus=interface_derivatives(law,x+eps*v,z);minus=interface_derivatives(law,x-eps*v,z)
    torch.testing.assert_close((plus[0]-minus[0])/(2*eps),t@v,rtol=1e-7,atol=1e-8)
    torch.testing.assert_close((plus[1]-minus[1])/(2*eps),H@v,rtol=1e-7,atol=1e-6)
