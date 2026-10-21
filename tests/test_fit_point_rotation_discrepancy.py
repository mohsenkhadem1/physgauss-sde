import numpy as np
from experiments.fit_point_rotation_discrepancy import joint_factors


def test_shared_complete_factors_match_independent_joint_sample_covariance():
    rng = np.random.default_rng(1); t = rng.normal(size=(16,7)); r = rng.normal(size=(16,5))
    f = joint_factors(t,r,np.ones_like(r,dtype=bool))
    mean = np.r_[f['target_mean_m'],f['range_mean_m']]
    b = np.r_[f['target_factor_m'],f['range_factor_m']]
    np.testing.assert_allclose(mean,np.c_[t,r].mean(0),atol=1e-15)
    np.testing.assert_allclose(b@b.T,np.cov(np.c_[t,r],rowvar=False),atol=5e-16)


def test_unobserved_ranges_never_enter_mean_or_factor_and_psd_is_retained():
    t=np.arange(16.,dtype=float).reshape(4,4); r=np.array([[1.,9.,0.],[3.,7.,0.],[5.,8.,0.],[7.,6.,0.]])
    mask=np.array([[1,0,0],[1,1,0],[0,0,0],[0,0,0]],dtype=bool)
    f=joint_factors(t,r,mask); changed=r.copy();changed[~mask]=np.nan
    g=joint_factors(t,changed,mask)
    for key in f: np.testing.assert_array_equal(f[key],g[key])
    np.testing.assert_array_equal(f['paired_hit_count'],[2,1,0])
    np.testing.assert_array_equal(f['range_mean_m'],[2.,7.,0.])
    assert not f['range_factor_m'][1:].any()
    b=np.r_[f['target_factor_m'],f['range_factor_m']]
    assert np.linalg.eigvalsh(b@b.T)[0] > -1e-12
