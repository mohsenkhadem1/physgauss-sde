import numpy as np
from scripts.compositional_neural_mechanics.reference_posterior_surrogate import design,log_likelihood,summarize_samples

def test_polynomial_response_recovery_independent_points():
    rng=np.random.default_rng(81);x=rng.uniform(-8,8,(100,2));c=rng.normal(size=(15,5));fit=np.linalg.lstsq(design(x,4),design(x,4)@c,rcond=None)[0];test=rng.uniform(-8,8,(37,2))
    np.testing.assert_allclose(design(test,4)@fit,design(test,4)@c,atol=2e-13)

def test_original_likelihood_and_scale():
    rng=np.random.default_rng(2);v=rng.normal(size=(5,180));y=rng.normal(size=24);s=np.linspace(-1,1,5)
    expected=np.array([-.5*(v[i,:156]@v[i,:156]+np.linalg.norm(np.exp(s[i])*v[i,156:]-y)**2)for i in range(5)])
    np.testing.assert_allclose(log_likelihood(v,s,y),expected,atol=1e-12)

def test_importance_weights_not_unweighted_gaussian_intervals():
    x=np.array([[0,0,0],[1,2,3],[2,4,6]],float);logw=np.log([.1,.8,.1]);r=summarize_samples(x,logw,np.array([0,9,11]))
    np.testing.assert_allclose(r['mean'],[1,2,3]);np.testing.assert_allclose(r['covariance'],.2*np.outer([1,2,3],[1,2,3]));assert abs(r['tail8_mass']-.9)<1e-14;assert r['tail_sample_counts']=={'8':2,'10':1};assert r['interval90'][0][0]==0;assert abs(r['log_evidence']+np.log(3))<1e-14

def test_validation_identity_rejects_duplicate_and_changed_hash(tmp_path,monkeypatch):
    import pytest
    import scripts.compositional_neural_mechanics.reference_posterior_surrogate as m
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'OUT',tmp_path/'out');folder=m.OUT/'validation';folder.mkdir(parents=True)
    p=folder/'000.npz';np.savez_compressed(p,response=np.zeros(180));plan={'validation_ratios':[[1.,2.]]*64};rows=[dict(index=i,ratio=[1.,2.],completed=False)for i in range(64)];rows[0].update(completed=True,path='out/validation/000.npz',sha256=m.sha(p));m.validation_identity(rows,plan)
    with pytest.raises(ValueError,match='unique'):m.validation_identity(rows[:-1]+[rows[0]],plan)
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact'):m.validation_identity(rows,plan)
