import numpy as np
from scripts.phystwin.v07_study import Experiment,indices,select

def test_stopping_accounts_for_cost_and_no_repeat():
 assert select([.02,.01],.03,[1,.25],.1)==-1
 assert select([.002,.009],.03,[1,.25],.02)==1
 assert select([np.inf,.001],.03,[1,1],.001)==1

def test_view_is_initial_load_from_new_camera():
 ix=indices();assert len(ix)==279
 np.testing.assert_array_equal(ix[27:36],ix[:9])
 ex=Experiment.__new__(Experiment)
 np.testing.assert_array_equal(ex.observed([6]),np.r_[np.arange(9),np.arange(27,36)])
 np.testing.assert_array_equal(ex.observed([6,1]),ex.observed([1,6]))

def test_shared_field_model_matches_dense_moments():
 from scipy.interpolate import RegularGridInterpolator
 from scipy.sparse.linalg import spsolve
 from scripts.phystwin.sequential_reference import BOUNDS
 rng=np.random.default_rng(4);axes=[np.linspace(*b,4) for b in BOUNDS]
 interp=[RegularGridInterpolator(axes,rng.normal(0,.001,(4,4,4,279)),method='cubic',solver=spsolve) for _ in range(3)]
 ex=Experiment(interp,5,np.zeros(279),np.eye(279)*1e-7,'cpu');m=ex.model([1,6]);y=np.zeros((1,len(ex.observed([1,6]))));w=m.weights(y)
 np.testing.assert_allclose(m.target_mean(w).numpy(),(w@m.M).numpy(),atol=1e-14)
 assert np.all(np.isfinite(ex.predict([1,6],np.zeros(36))['sd']))

def test_real_capture_rejects_fit_test_overlap():
 import pytest
 from scripts.phystwin.v07_pilot_data import validate
 d=dict(times_s=np.arange(4.),tracked_points_m=np.zeros((4,2,3)),visibility=np.ones((4,2),bool),forces_N=np.zeros((4,3)),force_application_m=np.zeros((4,3)),camera_to_world=np.broadcast_to(np.eye(4),(4,4,4)))
 m=dict(measured_geometry=True,nominal_mass_kg=1.,force_sigma_N=.01,episodes=[dict(start_frame=0,end_frame=2,role='fit',object_id='block',action='initial'),dict(start_frame=2,end_frame=4,role='held_out',object_id='block',action='oblique')])
 assert validate(d,m)['episodes']==2
 m['episodes'][1]['start_frame']=1
 with pytest.raises(AssertionError,match='overlap'):validate(d,m)

def test_planning_does_not_read_future_observations():
 from scipy.interpolate import RegularGridInterpolator
 from scipy.sparse.linalg import spsolve
 from scripts.phystwin.sequential_reference import BOUNDS
 rng=np.random.default_rng(8);axes=[np.linspace(*b,4) for b in BOUNDS]
 interp=[RegularGridInterpolator(axes,rng.normal(0,.001,(4,4,4,279)),method='cubic',solver=spsolve) for _ in range(3)]
 ex=Experiment(interp,5,np.zeros(279),np.eye(279)*1e-7,'cpu');a=np.zeros(36);b=a.copy();b[9:]=1000
 ra,ca=ex.choose([],a,samples=4);rb,cb=ex.choose([],b,samples=4)
 np.testing.assert_array_equal(ra,rb);assert ca==cb
