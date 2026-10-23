import numpy as np
from experiments.diagnose_p2_contact_rigid_elastic import rigid_fit,contribution

def test_positive_weight_rigid_fit_and_instantaneous_spin_exact():
 rng=np.random.default_rng(120);X=rng.normal(size=(30,3));w=rng.uniform(.1,1.,30);ang=.7;R=np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1.]])
 q=X@R.T+[.2,.3,-.4];qc=w@q/w.sum();omega=np.array([.2,-.4,.1]);V=np.array([.1,.2,.3]);v=V+np.cross(omega,q-qc);r=rigid_fit(X,q,v,w,X,q)
 np.testing.assert_allclose(r['rigid_query_positions_m'],q,atol=2e-15);np.testing.assert_allclose(r['rigid_query_velocities_m_s'],v,atol=1e-15);np.testing.assert_allclose(r['angular_velocity_rad_s'],omega,atol=1e-15)
 assert r['bulk_nonrigid_coordinate_RMS_m']<1e-15 and r['bulk_nonrigid_coordinate_velocity_RMS_m_s']<1e-15

def test_decomposition_cross_term_identity_allows_anticorrelation():
 total=np.ones((2,3,3));rigid=3*total;r=contribution(total,rigid)
 assert r['rigid_L2_norm_fraction']==3 and r['nonrigid_L2_norm_fraction']==2
 assert abs(r['rigid_L2_norm_fraction']**2+r['nonrigid_L2_norm_fraction']**2+r['cross_term_over_total_squared']-1)<1e-14
