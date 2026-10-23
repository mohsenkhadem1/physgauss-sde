import numpy as np
import pytest
from experiments.campaign_soft_floor_controls import independent_adaptive_response
from experiments.run_p2_contact_validation import build
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule
from experiments.campaign_p2_contact_adaptive_energy_gradient import integrate_face_energy_gradient

def test_explicit_new_floor_force_energy_scale_without_old_global_change():
 z=np.full(6,-1e-5);a=integrate_face_energy_gradient(z,.004,kappa_A=1e6);b=integrate_face_energy_gradient(z,.004,kappa_A=1e9)
 np.testing.assert_allclose(a['energy_J']*1000,b['energy_J'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(a['gradient_N']*1000,b['gradient_N'],atol=1e-10,rtol=1e-12)
 from experiments import campaign_contact_geometry_controls as old
 assert old.integrate_face.__module__=='experiments.campaign_p2_contact_adaptive_energy_gradient'

def test_new_floor_rest_and_fullmass_inputs_are_not_modified():
 m=build((1,1,1),300000.,9.81);X=m.rest.copy();force=m.load.copy();b,w=positive_keast24();old,_=exact_degree6_rule();a=dict(bary=b,weights=w[:,None]*m.volumes[None],guard_bary=np.concatenate([m.check_bary,m.bary,old,b]));u=np.zeros_like(X);u[:,2]=-X[:,2].min()-1e-5
 r=independent_adaptive_response(m,u,a,kappa_A=1e6);assert r['contact_energy']>0 and r['all_adaptive_faces_completed'];np.testing.assert_array_equal(m.rest,X);np.testing.assert_array_equal(m.load,force)
 with pytest.raises(ValueError):independent_adaptive_response(m,u,a,kappa_A=0.)
