import numpy as np
from scripts.physical_refinement_v15.sensor import distribution,measure,SensorSettings

def image():return dict(rgb=np.ones((192,256,3))*.5,depth=np.ones((192,256))*.3,camera={'known':True})

def test_force_block_and_visual_indices_no_hidden_mean():
    a=image();v=distribution(a);f=distribution(a,force=np.array([1.,2.,3.,-1.,-2.,-3.]),visual_covariance=v['covariance'],wrench_origin_world_m=[.24,0,0])
    assert f['mean'].shape==(70,);np.testing.assert_array_equal(f['visual_indices'],np.arange(64));np.testing.assert_array_equal(f['force_indices'],np.arange(64,67))
    assert not f['covariance'][:64,64:].any();np.testing.assert_array_equal(f['covariance'][:64,:64],v['covariance'])
    m=measure(a,np.random.default_rng(2),force=[1,2,3],visual_covariance=v['covariance'])
    assert 'mean'not in m and 'values'in m and 'material'not in m
    m2=measure(a,np.random.default_rng(2),force=[1,2,3],visual_covariance=v['covariance']);np.testing.assert_array_equal(m['values'],m2['values'])

def test_common_covariance_reused_across_candidate_images():
    a=image();C=distribution(a)['covariance'];b=image();b['depth']+=.01
    d=distribution(b,visual_covariance=C);np.testing.assert_array_equal(C,d['covariance']);assert not np.array_equal(d['mean'],distribution(a)['mean'])

def test_wrench_units_covariance_and_strict_shapes():
    import pytest
    a=image();settings=SensorSettings(force_sd_N=.02,torque_sd_Nm=.003)
    d=distribution(a,settings,force=np.arange(6),wrench_origin_world_m=[.24,.006,0])
    np.testing.assert_allclose(np.diag(d['covariance'])[-6:],np.square([.02]*3+[.003]*3))
    np.testing.assert_array_equal(d['torque_indices'],[67,68,69]);np.testing.assert_array_equal(d['reaction_indices'],np.arange(64,70))
    assert d['channel_units'][-6:]==['N']*3+['N m']*3 and d['wrench_origin_world_m']==[.24,.006,0]
    for invalid in [np.zeros((2,3)),[1,2],[1,2,3,4]]:
        with pytest.raises(ValueError):distribution(a,force=invalid)
    with pytest.raises(ValueError):distribution(a,force=np.zeros(6))
    with pytest.raises(ValueError):distribution(a,SensorSettings(torque_sd_Nm=0))
