import copy,hashlib,json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments import campaign_c38_study as s

def test_semantic_factor_and_object_role_barriers():
    rows=[]
    for i,role in enumerate(s.ROLES):
        f=s.factors([5000+i*100,625,.35,0],np.arange(8)==i,np.array([0.,.1,.2]),np.array([[i+1.,0],[0,i+.1]]))
        rows.append(dict(episode_id='e'+str(i),object_id='o'+str(i),role=role,**f))
    assert s.validate_catalog(rows)['joint_factor_OOD']
    bad=copy.deepcopy(rows);bad[1]['force_history_sha256']=bad[0]['force_history_sha256']
    with pytest.raises(ValueError,match='reused'):s.validate_catalog(bad)
    bad=copy.deepcopy(rows);bad[1]['object_id']=bad[0]['object_id']
    with pytest.raises(ValueError,match='Object'):s.validate_catalog(bad)
    with pytest.raises(ValueError,match='counts'):s.validate_catalog(rows,expected_counts={r:2 for r in s.ROLES})

def test_future_values_cannot_affect_prefix_or_baselines():
    times=np.arange(5)*.1;q=np.zeros((5,2,3));q[:,0,0]=times
    t,a=s.causal_prefix(times,q,.2);q[3:]=np.nan;tt,b=s.causal_prefix(times,q,.2)
    np.testing.assert_array_equal(t,tt);np.testing.assert_array_equal(a,b)
    target=np.array([.3,.4]);p=s.constant_velocity(t,a,target)
    np.testing.assert_allclose(p[:,0,0],target)
    np.testing.assert_allclose(s.persistence(t,a,target),np.repeat(a[-1,None],2,axis=0))
    with pytest.raises(ValueError,match='Strictly future'):s.forecast_metrics(p,p,[.2,.3],.2)

def test_rigid_twist_retains_shape_with_positive_quadrature_measure():
    X=np.array([[0.,0,0],[.2,0,0],[0,.1,0],[0,0,.08]])
    times=np.array([0.,.1]);target=np.array([.2,.3]);velocity=np.array([.03,-.02,.01]);omega=np.array([0,0,1.2])
    center=X.mean(0)
    def at(t):return (X-center)@Rotation.from_rotvec(omega*t).as_matrix().T+center+velocity*t
    past=np.stack([at(t) for t in times]);expected=np.stack([at(t) for t in target])
    np.testing.assert_allclose(s.rigid_extrapolation(X,times,past,target,np.ones(4)),expected,atol=2e-16)
    # Signed P2 integration coefficients cannot masquerade as material masses.
    with pytest.raises(ValueError,match='positive'):s.rigid_extrapolation(X,times,past,target,np.array([-1,2,2,2]))

def test_admission_is_required_and_hash_bound(tmp_path):
    (tmp_path/'source.py').write_text('# manufactured source\n');(tmp_path/'report.json').write_text('{"passed":true}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    gate=dict(qualified_for_prediction=True,spatial_qualified=False,temporal_qualified=True,nonlinear_solve_qualified=True,constitutive_matched=True,domain_sha256='domain',evidence=[dict(path='report.json',sha256=sha(tmp_path/'report.json'))],source_sha256={'source.py':sha(tmp_path/'source.py')})
    p=tmp_path/'admission.json';p.write_text(json.dumps(gate));manifest=dict(domain_sha256='domain',forward_admission=dict(path='admission.json',sha256=sha(p)))
    with pytest.raises(ValueError,match='spatial'):s.verify_admission(manifest,root=tmp_path)
    gate['spatial_qualified']=True;p.write_text(json.dumps(gate));manifest['forward_admission']['sha256']=sha(p)
    assert s.verify_admission(manifest,root=tmp_path)['qualified_for_prediction']
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='primary'):s.verify_admission(manifest,root=tmp_path)

def test_object_bootstrap_and_common_forecast_scores():
    p=np.zeros((3,4,3));q=np.ones_like(p)*.001;r=s.forecast_metrics(p,q,[.2,.3,.4],.1)
    assert np.isclose(r['coordinate_RMSE_m'],.001) and not r['observations_after_cutoff_used']
    ci=s.paired_object_interval([1,2,3,4],[2,3,4,5])
    assert ci['mean_difference']==-1 and ci['percentile_95percent_interval']==[-1,-1]


def test_actual_rgbd_payload_excludes_future_images_and_physical_labels():
    t=np.array([0.,.1,.2]);depth=np.ones((3,1,2,2));rgb=np.zeros((*depth.shape,3));visible=np.ones_like(depth,dtype=bool)
    args=dict(rest_nodes=np.zeros((10,3)),elements=np.arange(10)[None],surface_triangles=np.array([[0,1,2]]),observation_times=t,rgb=rgb,depth_m=depth,visible=visible,camera_poses=np.tile(np.eye(4),(3,1,1,1)),intrinsics=np.tile(np.eye(3),(3,1,1,1)),cutoff=.1,force_times=np.array([0.,.1,.2]),force_coefficients=np.ones((2,1)),load_basis=np.zeros((1,10,3)))
    first=s.rgbd_public_prefix(**args);args['depth_m'][2]=np.nan;args['rgb'][2]=np.nan;args['camera_poses'][2]=np.nan
    second=s.rgbd_public_prefix(**args)
    for key in first:
        if isinstance(first[key],np.ndarray):np.testing.assert_array_equal(first[key],second[key])
    assert len(first['observation_times'])==2 and not {'material','support','positions','future_positions'}&set(first)
    with pytest.raises(TypeError):s.rgbd_public_prefix(**args,material=np.ones(4))
