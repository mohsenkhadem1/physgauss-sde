import numpy as np
import pytest
from experiments.campaign_static_supported_block import StaticSupportedBlock,SIZE,KINDS,support_nodes


def test_physical_support_patches_nested_and_exact_gravity_traction():
    coarse=StaticSupportedBlock()
    expected={'left_face':[9,25],'right_face':[9,25],'both_ends':[18,50],'bottom_face':[15,45],'separated_bottom_patches':[12,30]}
    for kind in KINDS:
        counts=[]
        for cells in [(4,2,2),(8,4,4)]:
            m=StaticSupportedBlock(cells,boundary=kind,traction=1.)
            assert np.isclose(m.node_mass.sum(),.96)
            np.testing.assert_allclose(m.load.sum(0),[1.,0.,-9.4176],rtol=0,atol=2e-14)
            counts.append(int(m.mask[:,0].sum()))
            lookup={tuple(np.round(x,13)):i for i,x in enumerate(m.reference_local)}
            common=[lookup[tuple(np.round(x,13))] for x in coarse.reference_local]
            np.testing.assert_array_equal(m.mask[common,0],support_nodes(coarse.reference_local,kind))
        assert counts==expected[kind]
    with pytest.raises(ValueError,match='free fall'):StaticSupportedBlock(boundary='free')


def test_energy_gradient_known_body_load_and_exact_constraints_reactions():
    m=StaticSupportedBlock(young=300000.,gravity=.3)
    rng=np.random.default_rng(8);u=rng.normal(size=m.rest.shape)*1e-5;u[m.mask]=0
    direction=rng.normal(size=u.shape);direction[m.mask]=0;eps=1e-7
    state=m.evaluate(u,True)
    finite=(m.evaluate(u+eps*direction)['potential_J']-m.evaluate(u-eps*direction)['potential_J'])/(2*eps)
    assert finite==pytest.approx(float(np.sum(state['residual']*direction)),rel=2e-7,abs=2e-8)
    answer=m.solve();d=m.diagnostics(answer)
    assert d['maximum_constraint_error_m']==0
    assert d['force_balance_defect_N']<1e-8 and d['actual_moment_balance_defect_Nm']<1e-9
    assert d['original_free_residual_N']<=d['residual_tolerance_N']


def test_bulk_norm_affine_invariant_to_mesh_and_passive_frame():
    models=[StaticSupportedBlock(c,gravity=0.) for c in [(4,2,2),(8,4,4)]]
    values=[]
    for m in models:
        assert m.bulk_nonrigid_RMS(m.rest+np.array([.2,.1,-.3]))<1e-15
        values.append(m.bulk_nonrigid_RMS(m.rest@np.diag([1.04,.99,1.01])))
    np.testing.assert_allclose(values,values[0],rtol=1e-13,atol=1e-16)
    angle=.7;Q=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    a=StaticSupportedBlock(young=5000.,gravity=0.,traction=.1);b=StaticSupportedBlock(young=5000.,gravity=0.,traction=.1,rotation=Q,translation=[.1,-.2,.3])
    qa=a.solve()['positions'];qb=b.solve()['positions']
    np.testing.assert_allclose(qb,qa@Q.T+np.array([.1,-.2,.3]),rtol=0,atol=2e-12)
