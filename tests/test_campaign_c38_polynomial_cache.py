import numpy as np
import pytest
from experiments.campaign_c38_polynomial_likelihood_v02 import GaussianPolynomialLikelihood,PolynomialPosterior
from experiments.campaign_c38_polynomial_cache import CachedGaussianPolynomialLikelihood

def objects(cls,**kwargs):
 c=np.array([[.1],[.002],[.0001]]);y=np.array([.1015]);return {k:cls(c*f,y,np.array([[.0002**2]]),**kwargs) for k,f in [('a',1.),('b',1.01)]}

def test_exact_cached_vs_uncached_all_functionals_and_numerical_keys():
 for method in ['adaptive','legendre']:
  a=PolynomialPosterior(objects(GaussianPolynomialLikelihood),method=method,order=64);models=objects(CachedGaussianPolynomialLikelihood);b=PolynomialPosterior(models,method=method,order=64)
  assert a.Z==b.Z and a.log_normalizer==b.log_normalizer
  np.testing.assert_array_equal(a.category_probabilities,b.category_probabilities)
  for E in np.geomspace(75000.,150000.,11):assert a.cdf(E)==b.cdf(E)
  target={'a':[0.,.003,.0002],'b':[.0001,.0025,.0001]}
  for v in [.0015,.0018,.0021]:assert a.coordinate_cdf(target,v)==b.coordinate_cdf(target,v)
  tm={k:np.array(v)[:,None] for k,v in target.items()}
  am=a.target_moments(tm);bm=b.target_moments(tm)
  np.testing.assert_array_equal(am['mean_m'],bm['mean_m']);np.testing.assert_array_equal(am['variance_m2'],bm['variance_m2'])
  assert sum(m.cache_hits for m in models.values())>0
  m=models['a'];miss=m.cache_misses;m.integrate_x(m.lo,m.hi,reference_D=b.reference_D,method=method,order=65);assert m.cache_misses>miss

def test_capacity_recomputes_without_dropping_and_model_mutation_rejects():
 models=objects(CachedGaussianPolynomialLikelihood,max_cache_entries=1);p=PolynomialPosterior(models);m=models['a'];before=p.cdf(95000.);assert p.cdf(95000.)==before and m.cache_capacity_bypasses>0
 m.reference=1.
 with pytest.raises(FloatingPointError,match='changed'):p.cdf(95000.)
