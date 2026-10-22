"""Manufactured rigid/affine deformation and independent-object statistics."""
import numpy as np
import pytest
from experiments import audit_point_rotation_deformation as audit


def mesh():
    rest=np.array([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-2.]])
    return rest,np.array([[0,1,2,3],[0,2,1,4]])


def rotation(a):return np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])


def test_proper_rigid_motion_removed_for_both_weightings_and_strain_objective():
    r,tets=mesh();_,vol,w=audit.reference_geometry(r,tets);q=np.stack([r@rotation(a).T+[.3,-.2,.4] for a in [0.,.3,1.2]])
    for weights in [np.full(len(r),1/len(r)),w]:
        fit=audit.rigid_fit(r,q,weights)
        assert np.max(fit['nonrigid_coordinate_MSE_m2'])<1e-29
        np.testing.assert_allclose(fit['proper_rotation_determinant'],1.,atol=2e-15)
        np.testing.assert_allclose(fit['rotation_world_from_rest'],np.stack([rotation(a) for a in [0.,.3,1.2]]),atol=1e-15)
        np.testing.assert_allclose(fit['translation_m'],np.tile([.3,-.2,.4],(3,1)),atol=1e-15)
    strain=audit.tet_strain(r,tets,q)
    np.testing.assert_allclose(strain['minimum_J'],1.,atol=2e-15)
    assert np.max(strain['principal_stretch_MSE_deviation'])<1e-30


def test_affine_principal_stretches_green_strain_and_rotation_invariance():
    r,tets=mesh();A=np.diag([1.02,.99,1.005]);q=(r@A.T)[None];rotated=(q@rotation(.8).T)+[.2,.1,-.3]
    a=audit.tet_strain(r,tets,q);b=audit.tet_strain(r,tets,rotated)
    expected=np.array([1.02,.99,1.005]);E=.5*(expected**2-1)
    assert a['principal_stretch_MSE_deviation'][0]==pytest.approx(np.mean((expected-1)**2),abs=1e-17)
    assert a['green_lagrange_frobenius_MSE'][0]==pytest.approx(np.sum(E**2),abs=1e-17)
    assert a['minimum_J'][0]==pytest.approx(np.prod(expected),abs=1e-15)
    assert a['minimum_principal_stretch'][0]==pytest.approx(.99)
    assert a['maximum_principal_stretch'][0]==pytest.approx(1.02)
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=2e-15)


def test_volume_lumped_weights_use_reference_tet_volumes_not_uniform_nodes():
    r,tets=mesh();_,volume,w=audit.reference_geometry(r,tets)
    np.testing.assert_allclose(volume,[1/3,2/3]);np.testing.assert_allclose(w,[.25,.25,.25,1/12,1/6])
    q=np.repeat(r[None],2,axis=0)+[.3,.4,0.]
    for weights in [w,np.full(5,.2)]:
        fit=audit.rigid_fit(r,q,weights)
        np.testing.assert_allclose(fit['total_coordinate_MSE_m2'],.25/3,atol=1e-16)
        assert np.max(fit['nonrigid_coordinate_MSE_m2'])<1e-29


def test_reflections_not_mistaken_for_proper_rigid_motion_and_inversions_fail():
    r,tets=mesh();q=r[None]*[-1,1,1];fit=audit.rigid_fit(r,q,np.full(5,.2))
    assert fit['nonrigid_coordinate_MSE_m2'][0]>.01
    assert fit['proper_rotation_determinant'][0]==pytest.approx(1.)
    with pytest.raises(ValueError,match='Nonpositive'):audit.tet_strain(r,tets,q)


def test_ratio_denominator_and_frame_count_weighted_AB_combination():
    r,tets=mesh();q=np.repeat(r[None],3,axis=0);fit=audit.rigid_fit(r,q,np.full(5,.2));s=audit.tet_strain(r,tets,q)
    empty=audit.window_metrics(fit,s,[0,1,2]);assert empty['total_coordinate_RMS_m']==0 and empty['nonrigid_to_total_RMS_ratio'] is None
    a=dict(empty,total_coordinate_RMS_m=2.,nonrigid_coordinate_RMS_m=.2)
    b=dict(empty,frames=6,total_coordinate_RMS_m=1.,nonrigid_coordinate_RMS_m=.1)
    c=audit.combine_windows(a,b)
    assert c['frames']==9 and c['total_coordinate_RMS_m']==pytest.approx(np.sqrt(2.))
    assert c['nonrigid_to_total_RMS_ratio']==pytest.approx(.1) and c['nonrigid_to_total_squared_motion_fraction']==pytest.approx(.01)


def test_population_all32_paired_objects_and_constant_bootstrap():
    r,tets=mesh();q=np.repeat(r[None],2,axis=0)+[.3,.4,0.];fit=audit.rigid_fit(r,q,np.full(5,.2));s=audit.tet_strain(r,tets,q)
    v=audit.window_metrics(fit,s,[0,1]);v.update(nonrigid_coordinate_RMS_m=.0001,nonrigid_to_total_RMS_ratio=.001,nonrigid_to_total_squared_motion_fraction=1e-6)
    windows={name:dict(v) for name in ['A_full','A_common_future','B_full_reset','AB_full','AB_common']}
    objects=[dict(object_index=i,windows={k:windows for k in ['uniform_nodal','rest_volume_lumped_mass']}) for i in range(16,48)]
    result=audit.population_summary(objects)['uniform_nodal']['AB_common']
    np.testing.assert_allclose(result['nonrigid_coordinate_RMS_m']['object_bootstrap_mean_95_interval'],[.0001,.0001])
    assert result['objects_nonrigid_XYZ_coordinate_RMS_above_0p2mm_scalar_reference']==0
    with pytest.raises(ValueError,match='All32'):audit.population_summary(objects[:-1])
