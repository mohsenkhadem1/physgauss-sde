import inspect
import numpy as np
import pytest
from experiments import campaign_c38_static_family as m


def test_public_force_units_resultants_and_signed_consistent_body_integrals():
    model=m.build_geometry((4,2,2),100000.,'both_ends',0.);c=m.compile_known_forces(model.rest,model.tets,model.faces,model.elements,2.)
    np.testing.assert_allclose(c['surface_N'].sum(0),[0,0,2],atol=1e-12)
    np.testing.assert_allclose(c['body_N'].sum(0),[0,0,-625*.24*.08*.08*9.81],atol=1e-12)
    assert c['metadata']['physical_mass_kg']==pytest.approx(.96,abs=1e-14)
    assert (c['body_integral_weights_m3']<0).any() # legitimate P2 coefficients
    assert abs(c['body_integral_weights_m3'].sum()-.24*.08*.08)<1e-15
    assert np.linalg.norm(c['body_N'][model.mask].ravel())>0 # never drop fixture loads
    np.testing.assert_allclose(np.cross(model.rest-[.12,0,0],c['total_N']).sum(0),0,atol=1e-12)


def test_force_sign_linearity_and_invariance_to_E_and_BC():
    a=m.build_geometry((4,2,2),75000.,'both_ends',2.);b=m.build_geometry((4,2,2),150000.,'separated_bottom_patches',2.)
    np.testing.assert_array_equal(a.load,b.load)
    assert not {'young','E','support','BC','theta','hypothesis'}&set(inspect.signature(m.compile_known_forces).parameters)
    zero=m.compile_known_forces(a.rest,a.tets,a.faces,a.elements,0.);neg=m.compile_known_forces(a.rest,a.tets,a.faces,a.elements,-2.)
    np.testing.assert_allclose(a.load+neg['total_N'],2*zero['total_N'],atol=1e-15)
    assert not a.public_force_components['surface_N'][a.mask].any()
    assert not b.public_force_components['surface_N'][b.mask].any()


def test_true_P2_trace_and_patch_conformity_reject_missing_midnode():
    model=m.build_geometry((4,2,2),100000.,'separated_bottom_patches',0.)
    assert model.mask_trace['passed'] and model.mask_trace['all6_P2_shape_coefficients_fixed']
    six=m.boundary_faces6(model.elements,model.faces);mask=model.mask.copy();ids=six[np.all(abs(model.rest[six[:,:3],2]+.04)<1e-13,1),3:].ravel();fixed=ids[mask[ids,0]][0];mask[fixed]=False
    with pytest.raises(ValueError,match='ALL P2'):m.mask_trace_certificate(model.rest,model.elements,model.faces,mask,'separated_bottom_patches')
    with pytest.raises(ValueError):m.build_geometry((6,2,2),100000.,'separated_bottom_patches',0.)


def test_keast_keeps_original_80_J_guards_and_consistent_force():
    a=m.build_geometry((4,2,2),100000.,'both_ends',-2.,rule='keast24');b=m.build_geometry((4,2,2),100000.,'both_ends',-2.,rule='exact80')
    assert len(a.bary)==24 and len(b.bary)==80 and len(a.check_bary)>=90
    np.testing.assert_array_equal(a.check_bary[-80:],b.bary)
    for key in ('rest','tets','elements','faces','load','mask'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert a.consistent_body_weight_max_difference_m3*625*9.81<1e-12
