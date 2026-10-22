import numpy as np
from experiments.video_p2_corrected_contact import tessellation,surface_points,positive_volume_samples,rigid_removed,nearest_schedule
from experiments.campaign_static_p2_block import enrich

def test_true_quadratic_surface_and_recorded_schedule():
 N,tri=tessellation(6);q=np.array([[0,0,0],[1,0,0],[0,1,0],[.5,0,.2],[0,.5,0],[.5,.5,0.]])
 p,t=surface_points(q,np.array([[0,1,2,3,4,5]]),N,tri)
 assert len(t)==36 and len(p)==28 and p[:,2].max()>=.2
 np.testing.assert_allclose(N.sum(1),1,atol=1e-15)
 times=np.array([0.,.02,.07,.1]);requested,ids=nearest_schedule(times,7)
 np.testing.assert_array_equal(times[ids],np.array([times[np.argmin(abs(times-x))] for x in requested]))
 assert ids[0]==0 and ids[-1]==3

def test_exact_volume_positive_rigid_fit():
 X=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]);rest,T=enrich(X,np.array([[0,1,2,3]]));N,w=positive_volume_samples(rest,T)
 a=.4;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
 q=rest@R.T+[.3,-.2,.4];aligned,rms=rigid_removed(rest,q,T,N,w)
 assert np.all(w>0) and abs(w.sum()-1)<1e-15
 np.testing.assert_allclose(aligned,rest,atol=2e-15);assert rms<1e-15
