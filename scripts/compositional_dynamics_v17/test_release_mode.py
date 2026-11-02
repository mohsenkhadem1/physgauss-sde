import numpy as np
import torch
from scripts.compositional_dynamics_v17.release_mode_diagnostic import reference_tangent_energy,fit_projection
from experiments.campaign_newton_stable_material import stable_material

def test_reference_linear_hessian_equals_exact_stable_nh_tangent():
 F=torch.eye(3,dtype=torch.float64,requires_grad=True);W=reference_tangent_energy(F,None);P=torch.autograd.grad(W,F,create_graph=True)[0];H=torch.stack([torch.autograd.grad(P.reshape(-1)[i],F,retain_graph=True)[0]for i in range(9)]).reshape(3,3,3,3).detach().numpy();_,_,A=stable_material(np.eye(3),12000.,.35,True)
 assert np.isfinite(H).all();np.testing.assert_allclose(H,A,atol=1e-10)

def test_modal_fit_rejects_nyquist_and_records_short_window():
 t=np.arange(41)*.001;out=fit_projection(t,np.cos(2*np.pi*450*t),450);assert not out['resolved']
 out=fit_projection(t,np.cos(2*np.pi*10*t),10);assert out['resolved']and out['fewer_than_two_reference_cycles']
