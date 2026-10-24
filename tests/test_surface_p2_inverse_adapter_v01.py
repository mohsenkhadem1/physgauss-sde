"""Algebraic adapter controls only; no physics or posterior admission."""
import numpy as np
from experiments import campaign_surface_p2_static_v01 as p2
from experiments.campaign_surface_force_study_v01 import observation_points
from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference
from experiments.campaign_surface_p2_study_design_v01 import load_catalog


def test_medium_P2_modal_identity_with_P2_queries_and_all_modes():
    f=p2.SparseStaticForward(2);points=np.concatenate([observation_points(),[[0,.04,.04],[.36,.04,.04]]])
    # Tiny forces exercise the adapter independently of physical test outcomes.
    loads=np.stack(list(p2.qualification_loads(2).values()))*1e-4
    model=ModalStaticReference(f.Kunit_free,f.W_free,loads.reshape(2,-1).T[f.free],p2.point_query_matrix(f,points))
    theta=np.array([[2000.,5.],[4000.,np.sqrt(150.)],[8000.,30.]])
    expected=p2.field_queries(2,f.staticforward(theta,loads),points).transpose(0,2,3,1).reshape(3,3*len(points),2)
    np.testing.assert_allclose(model(theta),expected,rtol=2e-9,atol=2e-17)
    assert model.modes==1125 and model.diagnostics['all_modes_retained']


def test_complete_catalog_is_P2_specific_and_targets_disjoint():
    from experiments.campaign_surface_p2_study_design_v01 import design
    d=design();assert not(set(d['target_loads']) & set(d['normal_actions']+d['axial_actions']))
    for n in (1,2,4):
        catalog=load_catalog(n);X=p2.geometry(n)[0]
        assert len(catalog)==10
        central=p2.qualification_loads(n)
        np.testing.assert_array_equal(catalog['normal_x180'],central['normal'])
        np.testing.assert_array_equal(catalog['paired_axial_x180'],central['paired_axial'])
        for name,force in catalog.items():
            center=int(name.rsplit('x',1)[-1])/1000
            expected=[0,0,-.1] if name.startswith('normal') else [.48,0,0]
            moment=[0,.1*(center-.18),0] if name.startswith('normal') else [0,0,0]
            np.testing.assert_allclose(force.sum(0),expected,atol=2e-13,rtol=0)
            np.testing.assert_allclose(np.cross(X-[.18,.04,.04],force).sum(0),moment,atol=2e-14,rtol=0)
