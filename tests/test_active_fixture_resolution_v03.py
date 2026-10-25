import numpy as np
from scipy.sparse.linalg import norm
from experiments import campaign_active_fixture_resolution_v03 as m


def test_overlap_geometry_operator_loads_bitidentical_and_no_original_mutation():
    f=m.make_forward(2,m.loads.LAYOUTS[0]['id']);old=m.loads.make_forward(2,m.loads.LAYOUTS[0]['id'])
    for name in ('X','elements','faces','fixed','free','linear_gradients','volumes','rows','cols'):np.testing.assert_array_equal(getattr(f,name),getattr(old,name))
    for name in ('Kunit','W','Kunit_free','W_free'):assert norm(getattr(f,name)-getattr(old,name))==0
    loads,info=m.load_catalog(2);original,expected=m.loads.load_catalog(2)
    assert info==expected
    for name in loads:np.testing.assert_array_equal(loads[name],original[name])
    assert m.core._topology(2)['X'].shape==(425,3)


def test_new_declared_topology_positive_volume_exact_clamps_and_nested_nodes():
    previous=None
    for n in (3,6,12):
        t=m._topology(n);v=t['corner_X'][t['corner_tets']];vol=np.linalg.det((v[:,1:]-v[:,:1]).transpose(0,2,1))/6
        assert len(t['elements'])==24*n**3 and len(t['X'])==(8*n+1)*(2*n+1)**2
        assert np.all(vol>0);np.testing.assert_allclose(vol.sum(),.36*.08*.08,atol=2e-17,rtol=0)
        assert np.all((t['X'][t['fixed'],0]==0)|(t['X'][t['fixed'],0]==.36))
        if previous is not None:
            # Exact rational lattice index nesting; coordinates agree to ulp.
            fine=np.rint(t['X']/[.36,.08,.08]*[8*n,2*n,2*n]).astype(int)
            coarse=np.rint(previous/[.36,.08,.08]*[8*n,2*n,2*n]).astype(int)
            assert set(map(tuple,coarse))<=set(map(tuple,fine))
        previous=t['X']


def test_affine_quadratic_queries_and_common_finest12_metric_have_exact_physical_mean():
    pts=np.array([[0,0,0],[.36,.08,.08],[.135,.055,.08],[.09,.02,.08],[.18,.04,.04],[.123,.031,.017]])
    polynomial=lambda x:np.column_stack([.002*x[:,0]+.01*x[:,0]*x[:,1],.03*x[:,1]**2,-.01*x[:,2]+.04*x[:,0]*x[:,2]])
    for n in (3,6,12):
        X=m.geometry(n)[0];np.testing.assert_allclose(m.field_queries(n,polynomial(X),pts),polynomial(pts),atol=8e-19,rtol=0)
    constant=np.array([1e-5,-2e-5,3e-5]);fields=[(3,np.zeros_like(m.geometry(3)[0])),(6,np.tile(constant,(len(m.geometry(6)[0]),1)))]
    d=m.common_rms(fields,finest=12);np.testing.assert_allclose(d[1,0],np.linalg.norm(constant)/np.sqrt(3),atol=2e-19,rtol=0)
    volume=sum(w.sum() for _,w in m.bulk_query_blocks(12));np.testing.assert_allclose(volume,.36*.08*.08,atol=2e-17,rtol=0)
