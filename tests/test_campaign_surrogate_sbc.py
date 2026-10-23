"""Independent analytic controls only; no actual campaign models/observations/draws."""
import copy,time
import numpy as np
import pytest
from experiments import campaign_surrogate_sbc as s


def coefficients():
 result={}
 for i,c in enumerate(s.CATEGORIES):
  a=np.zeros((3,27));a[1]=np.linspace(.0001,.001,27)*(1+.2*i);b=a.copy();b[2]=np.linspace(-.0001,.0001,27);t=np.zeros((3,189));t[1]=np.linspace(1e-12,.001,189);result[c]=dict(prefix=a,SIDE=b,target=t)
 return result


def flat_posts():
 a=np.zeros((3,27));C=s.NOISE**2*np.eye(27)
 laws={c:s.BoundsMemoLikelihood(a,np.zeros(27),C)for c in s.CATEGORIES}
 return {m:s.posterior(laws,m)for m in s.METHODS}


def test_uniform_logmaterial_monotone_nonmonotone_tiny_and_atoms_independent_analytic():
 for post in flat_posts().values():
  F=s.material_cdf_bounds(post,75000.*2**.37);assert F['cdf']==pytest.approx(.37,abs=1e-12)
  for t in (.5,.625,.75,1.):
   target={c:np.array([0.,1.,0.])for c in s.CATEGORIES};r=s.coordinate_limits(post,target,t);expected=np.log(t/.5)/np.log(2);assert r['right']['cdf']==pytest.approx(expected,abs=1e-12)and not r['atom'];assert r['left']['cdf']==pytest.approx(expected,abs=1e-12)
  target={c:np.array([.75**2,-1.5,1.])for c in s.CATEGORIES};t=.01;a=.75-np.sqrt(t);b=.75+np.sqrt(t);expected=np.log(b/a)/np.log(2);r=s.coordinate_limits(post,target,t);assert r['right']['cdf']==pytest.approx(expected,abs=1e-12)
  tiny={c:np.array([0.,1e-30,0.])for c in s.CATEGORIES};r=s.coordinate_limits(post,tiny,.75e-30);assert not r['atom']and r['right']['cdf']==pytest.approx(np.log(1.5)/np.log(2),abs=1e-12)
  atom={c:np.array([float(i),0.,0.])for i,c in enumerate(s.CATEGORIES)};r=s.coordinate_limits(post,atom,0.);assert r['atom'];assert r['left']['cdf']==0.;assert r['right']['cdf']==pytest.approx(.5)
  pit=s.randomized_pit(r['left'],r['right'],.4);assert pit['cdf']==pytest.approx(.2)
  common={c:np.zeros(3)for c in s.CATEGORIES};r=s.coordinate_limits(post,common,0.);assert r['left']['cdf']==0 and r['right']['cdf']==pytest.approx(1.);assert s.randomized_pit(r['left'],r['right'],.4)['cdf']==pytest.approx(.4)


def test_analytic_prior_moments_and_no_quantile_roots(monkeypatch):
 from experiments import campaign_resolved_prefix_inference as frozen
 monkeypatch.setattr(frozen,'coordinate_quantile',lambda*a,**k:(_ for _ in()).throw(AssertionError('No coordinate root')))
 monkeypatch.setattr(frozen,'material_quantile',lambda*a,**k:(_ for _ in()).throw(AssertionError('No material root')))
 coeff=coefficients()
 for c in s.CATEGORIES:
  coeff[c]['prefix'][:]=0;coeff[c]['SIDE'][:]=0;coeff[c]['target'][:]=0;coeff[c]['target'][1,:]=1e-4
 obs={k:np.zeros(27)for k in s.PHASES};r=s.infer(coeff,obs,'prefix');assert r['completed']and r['numerical_controls_passed']
 mean=.5/np.log(2)*1e-4;var=(.375/np.log(2)-(.5/np.log(2))**2)*1e-8
 for row in r['methods'].values():assert np.allclose(row['moments']['mean_m'],mean,rtol=1e-12);assert np.allclose(row['moments']['variance_m2'],var,rtol=1e-12)


def test_probability_endpoint_projection_and_randomized_coverage_uncertainty():
 def row(v,lo=None,hi=None):return dict(cdf=v,lower=v if lo is None else lo,upper=v if hi is None else hi)
 assert s.probability_bounds(row(1.0000000000000002,.99999999999,1.))['cdf']==1.
 assert s.randomized_pit(row(0.),row(1.),0.)['randomized_equal_tail95_status']=='outside'
 assert s.randomized_pit(row(0.),row(1.),1.)['randomized_equal_tail95_status']=='outside'
 assert s.randomized_pit(row(.025,.0249999,.0250001),row(.025,.0249999,.0250001),.5)['randomized_equal_tail95_status']=='uncertain'
 for bad in [row(.5,.6,.7),row(.5,-.1,.6),row(float('nan'),0.,1.)]:
  with pytest.raises(FloatingPointError):s.probability_bounds(bad)
 with pytest.raises(FloatingPointError):s.randomized_pit(row(.6),row(.5),.5)


def test_deterministic_manufactured_two_phase_laws_directCDF_and_failure_denominators():
 coeff=coefficients();E=100000.;c=s.CATEGORIES[0];obs={k:s.evaluate_polynomial(coeff[c][k],E)+np.linspace(-1e-4,1e-4,27)for k in s.PHASES};truth=dict(young_Pa=E,support_category=c,target_displacement_m=s.evaluate_polynomial(coeff[c]['target'],E).reshape(63,3))
 for phase in s.PHASES:
  pred,posts=s.compile_inference(coeff,obs,phase);assert pred['completed']and pred['numerical_controls_passed'];scored=s.score_laws(coeff,obs,phase,truth,np.full(28,.3),posts=posts);assert scored['numerical_controls_passed'];assert all(len(v['scores'])==28 for v in scored['methods'].values())
 rows=[dict(object_index=i,admitted=i!=63,evaluation=copy.deepcopy(scored))for i in range(64)];out=s.aggregate(rows)
 assert out['objects']==64 and out['numerically_admitted']==63 and all(v['failed']==1 and v['denominator_including_failures']==64 for v in out['quantities'].values())
 with pytest.raises(ValueError):s.aggregate(rows[:-1])
