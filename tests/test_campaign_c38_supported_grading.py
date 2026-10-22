import numpy as np
import pytest
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry as Geometry, graded_axes
from experiments.campaign_c38_static_family import compile_known_forces, mask_trace_certificate
from experiments.campaign_static_p2_block import EDGES, shape_gradients
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_p2_coarse_preconditioner import compatible_prolongation
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_block_surface_controls import support_groups


def test_fixed_axis_map_conforms_to_both_support_and_load_edges():
    x,y,z = graded_axes((12,6,6))
    assert all(a in x for a in (0.,.06,.18,.24))
    for a in (x,y,z): assert np.all(np.diff(a)>0)
    np.testing.assert_allclose(x+x[::-1],.24,atol=3e-17)
    np.testing.assert_allclose(y+y[::-1],0.,atol=2e-17)
    assert y[1]-y[0] < y[3]-y[2]
    with pytest.raises(ValueError): graded_axes((6,4,4))


@pytest.mark.parametrize('support',['both_ends','separated_bottom_patches'])
def test_rebuilt_affine_tets_full_six_node_support_and_volume(support):
    m=Geometry((8,4,4),100000.,support,-2.)
    assert m.mask_trace['passed'] and (m.volumes>0).all()
    assert m.volumes.sum()==pytest.approx(.24*.08*.08,abs=1e-17)
    for k,(i,j) in enumerate(EDGES):
        np.testing.assert_array_equal(m.rest[m.elements[:,4+k]],(m.rest[m.tets[:,i]]+m.rest[m.tets[:,j]])/2)
    for b in ([.1,.2,.3,.4],[1.,0,0,0]):
        g=shape_gradients(np.array(b)[None],m.linear_gradients)
        np.testing.assert_allclose(g.sum(1),0,atol=2e-13)
        np.testing.assert_allclose(np.einsum('eai,eaJ->eiJ',m.rest[m.elements],g),np.broadcast_to(np.eye(3),(len(m.elements),3,3)),atol=3e-14,rtol=0)
    broken=m.mask.copy();mid=m.elements[:,4:].ravel();node=mid[broken[mid,0]][0];broken[node]=False
    with pytest.raises(ValueError):mask_trace_certificate(m.rest,m.elements,m.faces,broken,support)


def test_geometry_rule_and_public_load_independent_of_hypotheses():
    a=Geometry((8,4,4),75000.,'both_ends',2.)
    b=Geometry((8,4,4),150000.,'separated_bottom_patches',2.,rule='exact80')
    for name in ('rest','tets','faces','elements','load','volumes','body_integral_weights'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    np.testing.assert_array_equal(a.check_bary[-80:],b.bary)
    public=compile_known_forces(a.rest,a.tets,a.faces,a.elements,2.)
    np.testing.assert_array_equal(public['total_N'],a.load)
    np.testing.assert_allclose(a.load.sum(0),[0,0,2-9.4176],atol=1e-12)
    np.testing.assert_allclose(np.cross(a.rest-[.12,0,0],a.load).sum(0),0,atol=1e-12)
    assert np.any(a.load[a.mask]) # retain constrained body force


@pytest.mark.parametrize('support',['both_ends','separated_bottom_patches'])
def test_coarse_P_PT_preserves_affine_fields_support_trace_and_adjoint(support):
    fine=Geometry((8,4,4),100000.,support,0.)
    coarse=StableP2Block((4,2,2),young=100000.,gravity=0.,traction=0.)
    groups,_=support_groups(coarse.rest,support);coarse.mask=np.repeat(groups.any(0)[:,None],3,1)
    P=compatible_prolongation(coarse,fine.rest)
    np.testing.assert_allclose(P@np.ones(len(coarse.rest)),1.,atol=2e-15)
    np.testing.assert_allclose(P@coarse.rest,fine.rest,atol=6e-17)
    rng=np.random.default_rng(20261022);u=rng.normal(size=coarse.rest.shape);u[coarse.mask]=0
    f=rng.normal(size=fine.rest.shape);q=P@u
    np.testing.assert_allclose(q[fine.mask],0.,atol=3e-14)
    assert np.sum(q*f)==pytest.approx(np.sum(u*(P.T@f)),abs=2e-12)


def test_full_consistent_mass_matches_positive_quadrature_not_row_lumping():
    m=Geometry((8,4,4),100000.,'both_ends',0.)
    M=scalar_mass(m);rng=np.random.default_rng(21);v=rng.normal(size=(len(m.rest),3))
    # Independent degree6 positive quadrature integrates squared P2 fields.
    qv=np.einsum('qa,eai->eqi',m.N,v[m.elements]);energy=m.density*np.einsum('e,q,eqi,eqi->',m.volumes,m.weights,qv,qv)
    assert np.sum(v*(M@v))==pytest.approx(energy,rel=2e-14)
    assert energy>0 and np.max(abs((M-M.T).data),initial=0) < 64*np.finfo(float).eps*max(abs(M.data))
    local_gram=np.einsum('q,qa,qb->ab',m.weights,m.N,m.N)
    assert np.linalg.eigvalsh(local_gram).min()>0 # positive local mass + connected positive-volume mesh
    np.testing.assert_array_equal(np.unique(m.elements),np.arange(len(m.rest)))
    assert float(np.ones(len(m.rest))@(M@np.ones(len(m.rest))))==pytest.approx(.96,abs=2e-15)
    assert np.max(abs((M@np.ones(len(m.rest)))-m.density*m.body_integral_weights))<3e-17
