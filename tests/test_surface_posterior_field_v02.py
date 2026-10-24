import numpy as np
from scipy import sparse
from experiments.campaign_surface_posterior_field_v02 import GroupedPosteriorMeanField


def test_grouped_full_field_matches_all_direct_weighted_hypotheses():
    rng=np.random.default_rng(5512);R=rng.normal(size=(8,8));A=R.T@R+np.eye(8)
    B=np.zeros((8,8));B[:3,:3]=np.array([[2.,.3,.1],[.3,1.,.2],[.1,.2,1.]])
    f=rng.normal(size=(8,3));r=np.array([.001,.002,.004,.01])
    a=rng.uniform(1/8000,1/2000,size=(4,7));conditional=rng.dirichlet(np.ones(7),size=4)
    rho=np.array([.2,.3,.5,0.]);abar=np.sum(conditional*a,axis=1)
    grouped=dict(log_ratio=np.log(r),probability=rho,amplitude_mean=abar)
    expected=sum(rho[i]*conditional[i,j]*np.linalg.solve(A/a[i,j]+B*r[i]/a[i,j],f)
                 for i in range(4) for j in range(7)).T
    model=GroupedPosteriorMeanField(sparse.csr_matrix(A),sparse.csr_matrix(B),f)
    actual=model.mean_grouped(grouped,chunk=2)
    np.testing.assert_allclose(actual,expected,atol=1e-17,rtol=1e-12)
    theta=np.column_stack((1/a.ravel(),np.repeat(r,7)/a.ravel()))
    dense=model.mean(theta,(rho[:,None]*conditional).ravel())
    np.testing.assert_allclose(actual,dense,atol=1e-17,rtol=1e-12)
