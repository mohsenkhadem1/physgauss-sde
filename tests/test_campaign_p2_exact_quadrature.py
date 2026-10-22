from math import factorial
import numpy as np
import torch
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule,ExactBulkP2Geometry
from experiments.campaign_p2_torch import TorchP2
from experiments.campaign_static_p2_block import shapes


def test_all_tetrahedron_monomials_through_degree_six_are_exact_positive_rule():
    b,w=exact_degree6_rule();assert len(w)==80 and np.all(w>0)
    for i in range(7):
        for j in range(7-i):
            for k in range(7-i-j):
                expected=6*factorial(i)*factorial(j)*factorial(k)/factorial(i+j+k+3)
                value=w@(b[:,1]**i*b[:,2]**j*b[:,3]**k)
                assert abs(value-expected)<2e-15


def test_bulk_energy_force_and_tangent_match_independent_higher_tensor_rule():
    m=ExactBulkP2Geometry((1,1,1),pattern='axial_pull');t=TorchP2(m)
    X=m.rest;u=np.column_stack([.1*X[:,0]+.3*X[:,1]*X[:,2],-.02*X[:,1]+.1*X[:,0]*X[:,2],.03*X[:,2]+.2*X[:,0]*X[:,1]]);v=np.random.default_rng(98).normal(size=X.shape)
    # Independent 125-point construction, with another barycentric ordering.
    from numpy.polynomial.legendre import leggauss
    x,w=leggauss(5);x=(x+1)/2;w=w/2;bary=[];weights=[]
    for i,a in enumerate(x):
        for j,b in enumerate(x):
            for k,c in enumerate(x):
                bary.append([a,(1-a)*b,(1-a)*(1-b)*c,(1-a)*(1-b)*(1-c)])
                weights.append(6*w[i]*w[j]*w[k]*(1-a)**2*(1-b))
    m.bary=np.array(bary);m.weights=np.array(weights);m.N=shapes(m.bary);higher=TorchP2(m)
    a=t.evaluate(torch.tensor(u));b=higher.evaluate(torch.tensor(u))
    torch.testing.assert_close(a['energy'],b['energy'],rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(a['gradient'],b['gradient'],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(t.tangent(a,torch.tensor(v)),higher.tangent(b,torch.tensor(v)),rtol=1e-12,atol=1e-10)
