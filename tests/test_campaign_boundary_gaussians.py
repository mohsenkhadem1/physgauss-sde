"""Boundary quadrature/constitutive/role tests; no solver, image or recipe."""
import numpy as np
import pytest
from experiments import campaign_boundary_gaussians as b

X=np.array([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
FACES=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],dtype=np.int64)


def patches(role):
    return [b.GaussianPatch([.2,.3,.1],np.diag([.2,.3,.4])**2,role),
        b.GaussianPatch([1.,.2,.4],np.array([[.3,.03,0],[.03,.2,.01],[0,.01,.25]])**2+np.eye(3)*.01,role)]


def response(x=None,v=None,y=None,vy=None,K=None,D=None):
    q=b.SurfaceQuadrature.from_mesh(X,FACES)
    return b.compliant_support_response(q,patches('candidate_boundary_hypothesis'),
        positions_world_m=X+.2 if x is None else x,velocities_world_m_s=np.ones_like(X)*.1 if v is None else v,
        target_positions_world_m=np.broadcast_to(X,(2,4,3)) if y is None else y,
        target_velocities_world_m_s=np.zeros((2,4,3)) if vy is None else vy,
        stiffness_world_N_m=np.array([np.diag([2.,3.,4.]),np.diag([1.,2.,3.])]) if K is None else K,
        damping_world_Ns_m=np.array([np.diag([.2,.3,.4]),np.diag([.1,.2,.3])]) if D is None else D)


def test_lumped_area_integrates_affine_triangle_field_exactly():
    points=np.array([[0.,0,0],[2.,0,0],[0,1.,0],[.2,.2,.2]])
    q=b.SurfaceQuadrature.from_mesh(points,np.array([[0,1,2]]))
    np.testing.assert_allclose(q.node_area_m2,[1/3,1/3,1/3,0],rtol=0,atol=1e-16)
    linear=points@np.array([1.2,-.7,.3])+2.
    assert q.node_area_m2@linear==pytest.approx(linear[:3].mean(),abs=1e-15)
    assert q.node_area_m2.sum()==pytest.approx(1.)


def test_graph_api_normalization_active_and_no_interior_force():
    points=np.vstack([X,[.3,.2,.1]])
    centers=np.array([[.2,.1,.3],[.4,.3,.2]])
    weights,area=b.normalized_surface_weights(points,FACES,centers,np.broadcast_to(np.eye(3)*.01,(2,3,3)),np.array([True,False]))
    assert weights.shape==(2,5) and area.shape==(5,)
    np.testing.assert_allclose(weights.sum(1),[1,0],rtol=0,atol=2e-16)
    assert weights[:,-1].sum()==0 and area[-1]==0
    assert np.isfinite(weights).all()


def test_material_gaussian_rigid_coordinate_change_and_translation():
    angle=.63;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    center=np.array([[.1,.2,.3]]);cov=np.array([[[.2,.04,0],[.04,.3,.01],[0,.01,.1]]])
    w,_=b.normalized_surface_weights(X,FACES,center,cov,np.array([True]))
    t=np.array([.5,-.3,1.]);other,_=b.normalized_surface_weights(X@Q.T+t,FACES,center@Q.T+t,Q@cov@Q.T,np.array([True]))
    np.testing.assert_allclose(w,other,rtol=1e-14,atol=2e-16)


def test_compliant_force_is_negative_energy_gradient_with_affine_targets():
    rng=np.random.default_rng(123);x=X+.03*rng.normal(size=X.shape);direction=rng.normal(size=X.shape)
    A=np.array([[1.,.05,0],[0,1.,.02],[0,0,1.]])
    y=np.stack([X@A.T+[.01,0,.02],X@A.T+[-.01,.01,0]])
    result=response(x=x,y=y);eps=1e-6
    derivative=(response(x=x+eps*direction,y=y)['elastic_energy_J']-response(x=x-eps*direction,y=y)['elastic_energy_J'])/(2*eps)
    expected=-np.sum(result['elastic_forces_each_world_N'].sum(0)*direction)
    assert derivative==pytest.approx(expected,rel=2e-10,abs=2e-11)
    assert result['kind']=='explicit_compliant_fixture_hypothesis_NOT_hard_constraint'


def test_compliant_damping_nonnegative_and_moving_target_power_balance():
    rng=np.random.default_rng(7);x=X+rng.normal(size=X.shape)*.03;v=rng.normal(size=X.shape)
    y=np.broadcast_to(X,(2,4,3)).copy()+.01;vy=rng.normal(size=(2,4,3))*.1
    result=response(x=x,v=v,y=y,vy=vy)
    assert result['dissipation_rate_W']>=0
    energy_rate=-np.sum(result['elastic_forces_each_world_N']*(v[None]-vy))
    force_power=np.sum(result['forces_world_N']*v)
    assert force_power+energy_rate+result['dissipation_rate_W']==pytest.approx(result['target_actuator_power_W'],abs=2e-15)
    # Zero damping has exact0 dissipation, including finite rigid motion.
    assert response(D=np.zeros((2,3,3)))['dissipation_rate_W']==0.


def test_world_observer_rotation_preserves_energy_dissipation_and_rotates_force():
    a=.48;Q=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    rng=np.random.default_rng(12);x=X+.1*rng.normal(size=X.shape);v=rng.normal(size=X.shape)
    y=np.broadcast_to(X,(2,4,3)).copy();vy=np.zeros_like(y);K=np.array([np.diag([1.,2.,3.]),np.diag([2.,3.,4.])]);D=K*.2
    first=response(x,v,y,vy,K,D);other=response(x@Q.T+[1,2,3],v@Q.T,y@Q.T+[1,2,3],vy@Q.T,Q@K@Q.T,Q@D@Q.T)
    assert first['elastic_energy_J']==pytest.approx(other['elastic_energy_J'],abs=2e-15)
    assert first['dissipation_rate_W']==pytest.approx(other['dissipation_rate_W'],abs=2e-15)
    np.testing.assert_allclose(first['forces_world_N']@Q.T,other['forces_world_N'],rtol=1e-14,atol=2e-15)


@pytest.mark.parametrize('which',['K','D'])
def test_negative_stiffness_or_damping_rejected_without_repair(which):
    bad=np.array([np.diag([-1.,1.,2.]),np.eye(3)])
    with pytest.raises(ValueError,match='PSD'):response(**{which:bad})


def test_load_resultant_normalization_and_torque_difference_are_explicit():
    q=b.SurfaceQuadrature.from_mesh(X,FACES);p=patches('known_hardware');force=np.array([[.1,-.2,.3],[-.1,.2,-.3]])
    r=b.distributed_load_response(q,p,total_forces_world_N=force,positions_world_m=X+[.2,.1,0],handle_origins_world_m=np.array([X[0],X[1]]))
    np.testing.assert_allclose(r['integrated_total_forces_world_N'],force,rtol=0,atol=1e-16)
    np.testing.assert_allclose(r['forces_world_N'].sum(0),[0,0,0],rtol=0,atol=1e-16)
    assert np.linalg.norm(r['moment_difference_from_rigid_origin_Nm'])>.01
    np.testing.assert_allclose(r['moments_about_world_origin_Nm'],np.cross((X+[.2,.1,0])[None],r['forces_each_world_N']).sum(1),rtol=0,atol=1e-16)
    assert 'NOT_rigid_handle_reference' in r['kind']


def test_public_feature_projection_excludes_latent_support_and_force_kernels():
    q=b.SurfaceQuadrature.from_mesh(X,FACES)
    features=b.feature_groups(q,support_patches=patches('candidate_boundary_hypothesis'),load_patches=patches('known_hardware'))
    public=b.public_features(features)
    assert set(public['feature_groups'])=={'public_known_load_gaussians'}
    changed=patches('candidate_boundary_hypothesis');changed[0]=b.GaussianPatch([99,98,97],np.eye(3),'candidate_boundary_hypothesis',False)
    second=b.public_features(b.feature_groups(q,support_patches=changed,load_patches=patches('known_hardware')))
    for key,value in public['feature_groups']['public_known_load_gaussians'].items():np.testing.assert_array_equal(value,second['feature_groups']['public_known_load_gaussians'][key])
    private=b.feature_groups(q,support_patches=changed,load_patches=patches('candidate_force_hypothesis'))
    assert not b.public_features(private)['feature_groups']
    assert public['schema']['support_capacity']==public['schema']['handle_capacity']==2


def test_exact_reference_decode_preserves_order_masks_targets_without_soft_substitution():
    masks=np.zeros((2,8,3),dtype=bool);masks[0,0]=True;targets=np.zeros((2,8,3));patches=[[3,2],[5,4]]
    result=b.exact_reference_boundary(hard_node_axis_masks=masks,hard_targets_world_m=targets,ordered_handle_patch_nodes=patches,handle_origin_node_ids=np.array([3,5]))
    assert result['Gaussian_substitution_performed'] is False and result['fixture_kind']=='exact_hard_node_axis_constraints'
    np.testing.assert_array_equal(result['ordered_handle_patch_nodes'][0],[3,2]);np.testing.assert_array_equal(result['hard_node_axis_masks'],masks)
    result['hard_node_axis_masks'][:]=False;assert masks.any()
    masks[0,3]=True
    with pytest.raises(ValueError,match='overlap'):b.exact_reference_boundary(hard_node_axis_masks=masks,hard_targets_world_m=targets,ordered_handle_patch_nodes=patches,handle_origin_node_ids=np.array([3,5]))


def test_world_contact_needs_a_real_law_and_inactive_load_cannot_carry_force():
    with pytest.raises(NotImplementedError,match='contact'):b.world_contact_response()
    q=b.SurfaceQuadrature.from_mesh(X,FACES);p=patches('known_hardware');p[0]=b.GaussianPatch([0,0,0],np.eye(3),'known_hardware',False)
    with pytest.raises(ValueError,match='Inactive'):b.distributed_load_response(q,p,total_forces_world_N=np.ones((2,3)),positions_world_m=X,handle_origins_world_m=X[:2])


def test_rank_one_semidefinite_zero_modes_are_accepted_without_matrix_repair():
    direction=np.array([1.,2.,3.]);K=np.outer(direction,direction)
    np.testing.assert_array_equal(b.psd(K,'rank-one stiffness'),K)
    null=np.array([2.,-1.,0.]);x=X+null
    r=response(x=x,v=np.broadcast_to(null,X.shape),K=np.stack([K,K]),D=np.stack([K,K]))
    assert r['elastic_energy_J']==pytest.approx(0.,abs=1e-13)
    assert r['dissipation_rate_W']==pytest.approx(0.,abs=1e-13)
    np.testing.assert_allclose(r['forces_world_N'],0.,rtol=0,atol=1e-13)
