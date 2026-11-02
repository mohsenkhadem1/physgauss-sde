import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.campaign_newton_stable_material import stable_material
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_dynamics_v17.stable_polynomial import StableNeoHookeanPolynomial

def response(F):
 x=torch.tensor(F,dtype=torch.float64,requires_grad=True);W=StableNeoHookeanPolynomial()(x,None);P=torch.autograd.grad(W,x,create_graph=True)[0];H=torch.stack([torch.autograd.grad(P.reshape(-1)[i],x,retain_graph=True)[0]for i in range(9)]).reshape(3,3,3,3)
 return float(W.detach()),P.detach().numpy(),H.detach().numpy()

def test_energy_stress_and_full_tangent_against_independent_analytic_law():
 R=Rotation.from_rotvec([.2,-.4,.7]).as_matrix();Fs=[np.eye(3),np.eye(3)+1e-8*np.arange(9).reshape(3,3),np.array([[.65,.2,0],[0,1.4,.15],[0,0,1.2]]),R,R@np.diag([.7,1.3,.9])]
 for F in Fs:
  W,P,H=response(F);a,b,c=stable_material(F,12000.,.35,True)
  assert np.isfinite(H).all();np.testing.assert_allclose(W,a,atol=1e-11,rtol=1e-12);np.testing.assert_allclose(P,b,atol=1e-10,rtol=1e-12);np.testing.assert_allclose(H,c,atol=1e-9,rtol=1e-12)
  old=StableNeoHookean()(torch.tensor(F,dtype=torch.float64),None);np.testing.assert_allclose(W,float(old),atol=1e-11,rtol=1e-12)

def test_tangent_directional_difference_at_rest_and_large_strain():
 D=np.array([[.1,.2,-.1],[.05,-.15,.1],[.3,.1,.05]])
 for F in [np.eye(3),np.diag([.6,1.5,1.1])]:
  _,P,H=response(F);h=1e-6;fd=(response(F+h*D)[1]-response(F-h*D)[1])/(2*h);np.testing.assert_allclose(np.einsum('iJkL,kL->iJ',H,D),fd,atol=2e-5,rtol=1e-7)
