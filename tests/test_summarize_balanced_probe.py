"""Analytic deformation/rigid-fit/wrench checks, without any campaign reads."""
from pathlib import Path
import numpy as np
import pytest
from experiments import summarize_balanced_probe as summary


def tetrahedron():
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])


def test_proper_uniform_rigid_fit_removes_finite_motion_but_not_reflection():
    rest=tetrahedron()
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    q=np.stack((rest,rest@R.T+[1.,2.,3.]))
    result=summary.rigid_residual(rest,q)
    assert result['uniform_node_coordinate_RMS_total_motion_m']>1
    assert result['uniform_node_coordinate_RMS_after_best_fit_rigid_m']<2e-15
    assert result['maximum_fitted_rotation_angle_deg']==pytest.approx(90.)
    reflected=rest.copy();reflected[:,0]*=-1
    assert summary.rigid_residual(rest,reflected[None])['uniform_node_coordinate_RMS_after_best_fit_rigid_m']>.1


def test_shared_gaussian_field_affine_strain_and_rigid_objectivity():
    rest=tetrahedron();tets=np.array([[0,1,2,3]])
    F=np.diag([1.2,.8,1.1])
    result=summary.strain_summary(rest,tets,(rest@F.T)[None])
    assert result['minimum_J']==pytest.approx(1.056)
    assert result['minimum_principal_stretch']==pytest.approx(.8)
    assert result['maximum_principal_stretch']==pytest.approx(1.2)
    assert result['reference_volume_weighted_principal_strain_RMS']==pytest.approx(np.sqrt((.2**2+.2**2+.1**2)/3))
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rigid=summary.strain_summary(rest,tets,(rest@R.T+[3.,4.,5.])[None])
    assert rigid['minimum_J']==pytest.approx(1.)
    assert rigid['maximum_absolute_principal_stretch_minus_one']<1e-14


def test_actual_endpoint_wrench_can_change_when_reference_pair_is_central():
    origins=np.array([[-1.,0.,0.],[1.,0.,0.]])
    force=np.broadcast_to([[-1.,0.,0.],[1.,0.,0.]],(360,2,3)).copy()
    end=origins.copy();end[1,1]=.1
    times=np.arange(361)/120
    rows=[]
    for i in range(360):
        rows.append(dict(held_handle_forces_world_N=force[i].tolist(),
            handle_centers_start_m=origins.tolist(),handle_centers_end_m=end.tolist(),
            actual_interval_s=times[i:i+2].tolist(),
            moment_about_world_origin_start_end_Nm=np.stack((np.cross(origins,force[i]),np.cross(end,force[i]))).tolist()))
    result=summary.wrench_summary(rows,force,origins)
    assert result['maximum_applied_handle_resultant_N']==0
    assert result['maximum_reference_origin_handle_moment_Nm']==0
    assert result['maximum_actual_endpoint_handle_moment_Nm']==pytest.approx(.1)
    np.testing.assert_allclose(result['end_endpoint_angular_impulse_quadrature_Nm_s'],[0.,0.,-.3],atol=1e-15)
    rows[-1]['moment_about_world_origin_start_end_Nm'][1][1][2]=0
    with pytest.raises(ValueError,match='Logged endpoint moments'):
        summary.wrench_summary(rows,force,origins)


def test_distance_summary_removes_rigid_translation_and_reports_extension():
    rest=tetrahedron();origins=rest[:2]
    q=np.stack((rest,rest*1.2+3.))
    result=summary.distance_summary(rest,np.array([[0,1,2,3]]),q,origins,q[:,:2])
    assert result['maximum_absolute_relative_handle_separation_change']==pytest.approx(.2)
    assert result['maximum_edge_length_ratio']==pytest.approx(1.2)
    assert result['uniform_edge_relative_length_change_RMS']==pytest.approx(.2/np.sqrt(2))


def test_live_controller_gate_never_opens_capture_arrays(monkeypatch):
    design=dict(object_index=15,role='DEVELOPMENT_EXCITATION_DIAGNOSTIC',
        inference_or_reserved_comparison=False,source_sha256={})
    prepared=dict(design=summary.DESIGN,cells=[{},{}],source_sha256={})
    monkeypatch.setattr(summary,'checked',lambda r: Path(r['path']))
    monkeypatch.setattr(summary,'resolve',lambda p: Path(p))
    def read(path):
        if str(path)==summary.DESIGN['path']:return design
        if str(path).endswith('prepared.json'):return prepared
        return dict(terminal=False)
    monkeypatch.setattr(summary,'read',read)
    monkeypatch.setattr(np,'load',lambda *a,**kw: pytest.fail('No arrays may be read while controller is live'))
    with pytest.raises(ValueError,match='Controller not terminal'):
        summary.run('controller.json','hash','not-created')
