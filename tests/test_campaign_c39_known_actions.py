import numpy as np
import pytest
from experiments.campaign_c39_known_actions import compile_known_action,ACTIONS
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
from experiments.campaign_c38_static_family import compile_known_forces


@pytest.fixture(scope='module')
def geometry():return SupportedBoundaryGradedGeometry((4,2,2),75000.,'both_ends',0.,rule='keast24')


def test_lateral_force_has_correct_reference_moment_and_unchanged_body(geometry):
    m=geometry;a=compile_known_action(m.rest,m.tets,m.faces,m.elements,'middle_side_2N')
    np.testing.assert_allclose(a['surface_N'].sum(0),[0.,2.,0.],rtol=0,atol=1e-12)
    np.testing.assert_allclose(a['metadata']['rest_surface_moment_Nm'],[-.08,0.,0.],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(a['body_N'],compile_known_forces(m.rest,m.tets,m.faces,m.elements,0.)['body_N'])
    assert a['metadata']['no_material_or_support_input']


def test_old_vertical_cases_are_exactly_reproduced_and_input_unchanged(geometry):
    m=geometry;before=m.rest.copy()
    for key,F in [('gravity',0.),('middle_down_2N',-2.),('canonical_up_1N',1.)]:
        a=compile_known_action(m.rest,m.tets,m.faces,m.elements,key)
        old=compile_known_forces(m.rest,m.tets,m.faces,m.elements,F)
        np.testing.assert_array_equal(a['total_N'],old['total_N'])
    np.testing.assert_array_equal(m.rest,before)


def test_support_and_material_never_enter_force_compiler(geometry):
    m=geometry;n=SupportedBoundaryGradedGeometry((4,2,2),150000.,'separated_bottom_patches',0.,rule='keast24')
    np.testing.assert_array_equal(m.rest,n.rest)
    for key in ACTIONS:
        a=compile_known_action(m.rest,m.tets,m.faces,m.elements,key)
        b=compile_known_action(n.rest,n.tets,n.faces,n.elements,key)
        np.testing.assert_array_equal(a['total_N'],b['total_N'])
    with pytest.raises(ValueError):compile_known_action(m.rest,m.tets,m.faces,m.elements,'unknown')


def test_left_patch_exact_load_moment_and_spatial_scope(geometry):
    m=geometry;a=compile_known_action(m.rest,m.tets,m.faces,m.elements,'left_down_2N')
    np.testing.assert_allclose(a['surface_N'].sum(0),[0.,0.,-2.],rtol=0,atol=1e-12)
    np.testing.assert_allclose(a['metadata']['rest_surface_moment_Nm'],[0.,-.18,0.],rtol=0,atol=1e-12)
    assert np.all(a['surface_N'][m.rest[:,0]>.06+1e-13]==0)
