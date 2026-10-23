from types import SimpleNamespace
import numpy as np
import pytest
from experiments import campaign_p2_contact_adaptive_energy_gradient as c

def test_constant_negative_plane_exact_energy_and_T6_shape_integrals():
 z=np.full(6,-1e-5,dtype=np.float64);A=.004;K=1e9;r=c.integrate_face_energy_gradient(z,A,kappa_A=K)
 assert abs(r['energy_J']-.5*K*A*1e-10)<1e-12
 expected=K*A*(-1e-5)*np.array([0.,0.,0.,1/3,1/3,1/3]);np.testing.assert_allclose(r['gradient_N'],expected,atol=1e-10,rtol=1e-10)
 assert not r['Hessian_computed_or_qualified']

def test_status2_never_waived_even_if_error_estimate_zero(monkeypatch):
 def fail(*args,**kwargs):return np.zeros(7),0.,SimpleNamespace(success=False,status=2,intervals=[],neval=1)
 monkeypatch.setattr(c,'quad_vec',fail)
 with pytest.raises(RuntimeError,match="'status': 2"):c.integrate_face_energy_gradient(np.full(6,-1e-5),.004)
