import numpy as np
import torch
import pytest
from experiments.campaign_p2_torch_contact import triangle_rule,boundary_trace,quadratic_face_minimum,PlaneContactP2
from experiments.campaign_p2_torch import TorchP2
from experiments.campaign_newton_stable_material import StableP2Block

def model():return StableP2Block(cells=(1,1,1),young=5000.,poisson=.35,density=625.,gravity=0.,traction=0.,order=3,boundary='left_face')

def test_positive_triangle_quadrature_area_and_quadratic_trace():
 b,N,w=triangle_rule(4)
 assert np.all(w>0)
 np.testing.assert_allclose(w.sum(),1.)
 np.testing.assert_allclose(N.sum(1),1.,atol=1e-15)
 np.testing.assert_allclose(w@N,[0,0,0,1/3,1/3,1/3],atol=2e-16)
 np.testing.assert_allclose(w@b,[1/3]*3,atol=2e-16)
 m=model();six,area=boundary_trace(m)
 np.testing.assert_allclose(area.sum(),.0896)
 for i,j in [(3,0),(4,0),(5,1)]:assert np.all(six[:,i]>=8)

@pytest.mark.parametrize("floor",[1.,.00371])
def test_contact_energy_gradient_HVP_and_diagonal_consistency(floor):
 m=model();obj=PlaneContactP2(m,kappa_A=1e5,floor_z=floor,surface_order=4);base=TorchP2(m)
 torch.manual_seed(7);u=torch.randn_like(obj.rest)*1e-4;v=torch.randn_like(u);v=v/torch.linalg.vector_norm(v);cache=obj.evaluate(u);b=base.evaluate(u)
 eps=1e-7;plus=obj.evaluate(u+eps*v);minus=obj.evaluate(u-eps*v)
 np.testing.assert_allclose(float((plus['energy']-minus['energy'])/(2*eps)),float((cache['gradient']*v).sum()),rtol=2e-8,atol=1e-7)
 np.testing.assert_allclose(((plus['gradient']-minus['gradient'])/(2*eps)).numpy(),obj.tangent(cache,v).numpy(),rtol=2e-7,atol=2e-5)
 contactHv=obj.tangent(cache,v)-base.tangent(b,v)
 assert float((v*contactHv).sum())>=0
 np.testing.assert_allclose(contactHv[:,:2],0.,atol=1e-12)
 diag=obj.diagonal(cache)-base.diagonal(b)
 for node in [0,8,len(u)-1]:
  e=torch.zeros_like(u);e[node,2]=1.
  explicit=(obj.tangent(cache,e)-base.tangent(b,e))[node,2]
  np.testing.assert_allclose(float(diag[node,2]),float(explicit),atol=1e-10)
 np.testing.assert_array_equal(obj.mass(v),base.mass(v))
 np.testing.assert_array_equal(obj.mask,base.mask)

def test_uniform_flat_reference_law_and_zero_separation():
 obj=PlaneContactP2(model(),kappa_A=100.,floor_z=1.,surface_order=4)
 u=torch.zeros_like(obj.rest);cache=obj.evaluate(u);d=obj.contact_diagnostics(u,cache)
 # Closed rest block centered atz0: integralz=0, so resultant=kappa*area*floor.
 np.testing.assert_allclose(d['contact_normal_resultant_N'],100*.0896,atol=1e-13)
 np.testing.assert_allclose(-float(cache['gradient'][:,2].sum()),100*.0896,atol=1e-13)
 empty=PlaneContactP2(model(),kappa_A=100.,floor_z=-1.,surface_order=4);c=empty.evaluate(u)
 assert float(c['contact_energy'])==0 and not bool(c['contact_active'].any())
 np.testing.assert_array_equal(c['gradient'],TorchP2(model()).evaluate(u)['gradient'])

def test_exact_face_minimum_catches_undershoot_between_all_six_nodes():
 # (u-.25)^2+(v-.25)^2-.04 positive atALL6nodes; negativeinside.
 b=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
 values=(b[:,1]-.25)**2+(b[:,2]-.25)**2-.04
 assert np.all(values>0)
 minimum,where=quadratic_face_minimum(values[None])
 np.testing.assert_allclose(minimum,[-.04],atol=1e-16)
 np.testing.assert_allclose(where,[[.5,.25,.25]],atol=1e-16)
 # A convex edge minimum and a linear/degeneratecase.
 edge=(b[:,1]-.25)**2+b[:,2]+.1
 minimum,where=quadratic_face_minimum(np.stack([edge,2*b[:,1]+b[:,2]+.3]))
 np.testing.assert_allclose(minimum,[.1,.3],atol=1e-15)
 np.testing.assert_allclose(where[0],[.75,.25,0],atol=1e-15)


def test_crossing_affine_face_against_independent_exact_clipped_integral():
 # Unit reference triangle (area.5), gap=u-a; exact clipped-domain integral.
 # Kinks make finite Gauss quadrature approximate: retain/report that fact.
 a=.37;exactU=.5*(a**3/3-a**4/12);exactF=a*a/2-a**3/6
 errors=[]
 for order in [4,32]:
  b,N,w=triangle_rule(order);gap=b[:,1]-a
  U=.25*np.sum(w*np.minimum(gap,0)**2);F=.5*np.sum(w*np.maximum(-gap,0))
  errors.append([abs(U-exactU),abs(F-exactF)])
 assert errors[1][0]<1e-6 and errors[1][1]<1e-5
 assert np.all(np.array(errors[1])<np.array(errors[0]))
