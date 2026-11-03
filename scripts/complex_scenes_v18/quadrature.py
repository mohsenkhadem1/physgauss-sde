"""Optional endpoint-inclusive volume quadrature for severe Q2 deformation.

Gauss-Lobatto order >=4 integrates the affine-Q2 consistent mass exactly and
samples constitutive energy at element faces. It changes numerical quadrature,
not the constitutive law; convergence must still be measured separately.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre
from scripts.phystwin.tensor_fixture import basis
from scripts.compositional_neural_mechanics.mechanics import EnergyEngine


def install_lobatto(model,order):
    if order not in (4,5):raise ValueError('Declared Lobatto orders are four and five')
    poly=Legendre.basis(order-1);q=np.r_[-1.,poly.deriv().roots(),1.];w=2/(order*(order-1)*poly(q)**2)
    model.quad=np.stack(np.meshgrid(q,q,q,indexing='ij'),-1).reshape(-1,3)
    model.weights=np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),-1),axis=-1).ravel()/8
    model.order=order;model.engine=EnergyEngine(model,model.device)
    weights=model.engine.weights.detach().cpu().numpy();model.volumes=weights.sum(0);N,_=basis(model.quad)
    integral=np.einsum('qe,qa->ea',weights,N);model.nodal_volumes=np.zeros(len(model.rest));np.add.at(model.nodal_volumes,model.elements,integral)
    model.total_mass=float(model.volumes.sum()*model.density);model.volume_weights=model.nodal_volumes/model.volumes.sum();model.set_controls(model.poses)
