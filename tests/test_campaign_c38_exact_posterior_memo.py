import numpy as np
import pytest
from experiments.campaign_c38_polynomial_likelihood_v02 import PolynomialPosterior,rational
from experiments.campaign_c38_polynomial_cache import CachedGaussianPolynomialLikelihood
from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood,MemoPolynomialPosterior,_topology


def test_same_exact_residual_bounds_and_capacity_no_dropped_boxes():
 rng=np.random.default_rng(54);c=rng.normal(size=(3,54));y=rng.normal(size=54);C=np.eye(54)
 old=CachedGaussianPolynomialLikelihood(c,y,C);new=BoundsMemoLikelihood(c,y,C,max_bounds_entries=2)
 for a,b in [(.51,.53),(.6,.7),(.5,1.),(.61,.999),(.51,.53)]:
  expected=old.D_bounds(rational(a),rational(b));assert new.D_bounds(rational(a),rational(b))==expected;assert new.D_bounds(rational(a),rational(b))==expected
 assert new.bounds_capacity_bypasses>0 and len(new._bounds_cache)<=2
 new.coefficients=((rational(0),rational(0),rational(0)),)
 with pytest.raises(FloatingPointError,match='stale'):new.D_bounds(rational(.5),rational(1))


def test_coordinate_CDF_all_methods_bitidentical_and_shared_topology():
 _topology.cache_clear();c=np.array([[-.4],[0.],[1.]]);old={k:CachedGaussianPolynomialLikelihood(c+delta,[0.],[[.02**2]])for k,delta in [('a',0.),('b',np.array([[.015],[0.],[0.]]))]};new={k:BoundsMemoLikelihood(c+delta,[0.],[[.02**2]])for k,delta in [('a',0.),('b',np.array([[.015],[0.],[0.]]))]};targets={'a':np.array([0.,.001,.002]),'b':np.array([.0001,.0011,.0018])}
 for method,order in [('adaptive',64),('legendre',64),('legendre',128)]:
  a=PolynomialPosterior(old,method=method,order=order);b=MemoPolynomialPosterior(new,method=method,order=order,max_CDF_entries=2)
  assert a.log_normalizer==b.log_normalizer
  for value in [.0014,.0016,.0018,.0020,.0014]:
   expected=a.coordinate_cdf(targets,value);assert b.coordinate_cdf(targets,value)==expected;assert b.coordinate_cdf(targets,value)==expected
  assert b.CDF_hits>0 and b.CDF_capacity_bypasses>0
 assert _topology.cache_info().hits>0


def test_analytic_logE_CDF_and_copy_mutation_guards():
 m=BoundsMemoLikelihood(np.zeros((3,1)),[0.],[[1.]]);p=MemoPolynomialPosterior({'a':m});T={'a':np.array([0.,1.,0.])}
 for value in [.55,.75,.95]:assert p.coordinate_cdf(T,value)['cdf']==pytest.approx(np.log(2*value)/np.log(2),abs=1e-14)
 U={'a':np.array([0.,0.,1.])};r=p.coordinate_cdf(U,.7);expected=r['cdf'];assert r['root_boxes'];r['cdf']=-1;r['root_boxes'][0]['lo']='tampered'
 again=p.coordinate_cdf(U,.7);assert again['cdf']==expected and again['root_boxes'][0]['lo']!='tampered'
 U['a'][0]=.2;assert p.coordinate_cdf(U,.7)['cdf']!=expected
 p.prior[0]=.9
 with pytest.raises(FloatingPointError,match='stale'):p.coordinate_cdf(T,.75)


def test_original_unresolved_root_guards_remain_failures():
 A=float(2**340)
 with pytest.raises(FloatingPointError,match='Level-root'):BoundsMemoLikelihood(np.array([[-.75*A],[A],[0.]]),[0.],[[1.]])
 A=float(2**200);m=BoundsMemoLikelihood(np.array([[-.75*A],[A],[0.]]),[0.],[[1.]],log_drops=(.5,2.,8.,32.));p=MemoPolynomialPosterior({'a':m,'b':m})
 for _ in range(2):
  with pytest.raises(FloatingPointError,match='Target-root posterior mass'):p.coordinate_cdf({'a':[.75**2,-1.5,1.],'b':[.75**2,-1.5,1.]},1e-120)
 with pytest.raises(ValueError,match='SAME acquired'):MemoPolynomialPosterior({'a':BoundsMemoLikelihood(np.zeros((3,1)),[0.],[[1.]]),'b':BoundsMemoLikelihood(np.zeros((3,1)),[1.],[[1.]])})
