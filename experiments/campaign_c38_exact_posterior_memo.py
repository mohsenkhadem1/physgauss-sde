"""NEW exact memoization prototype; no tolerance, physics or likelihood changes.

Old source stays immutable. Bounds reuse the SAME54-residual rational enclosure,
not a tighter approximate/alternative polynomial bound. Coordinate-CDF results
are per-posterior; exact target-root topology alone is shared across methods.
Every root-box mass guard and every integration/normalizer error is retained.
"""
from functools import lru_cache
import numpy as np
import sympy as sp
import mpmath as mp
from experiments.campaign_c38_polynomial_likelihood_v02 import PolynomialPosterior,roots,rational,X
from experiments.campaign_c38_polynomial_cache import CachedGaussianPolynomialLikelihood


class BoundsMemoLikelihood(CachedGaussianPolynomialLikelihood):
 def __init__(self,*args,max_bounds_entries=10000,**kwargs):
  if type(max_bounds_entries)is not int or max_bounds_entries<1:raise ValueError('Positive bounds-cache capacity required')
  self._bounds_cache={};self._bounds_context=None;self.max_bounds_entries=max_bounds_entries;self.bounds_hits=0;self.bounds_misses=0;self.bounds_capacity_bypasses=0
  super().__init__(*args,**kwargs);self._bounds_context=(self.coefficients,self.D.as_expr())
 def D_bounds(self,lo,hi):
  context=(self.coefficients,self.D.as_expr())
  if self._bounds_context is not None and context!=self._bounds_context:raise FloatingPointError('Likelihood changed; reject stale exact bounds cache')
  key=(sp.Rational(lo),sp.Rational(hi))
  if key in self._bounds_cache:self.bounds_hits+=1;return self._bounds_cache[key]
  value=super().D_bounds(*key);self.bounds_misses+=1
  if len(self._bounds_cache)<self.max_bounds_entries:self._bounds_cache[key]=value
  else:self.bounds_capacity_bypasses+=1
  return value
 def cache_stats(self):
  return dict(**super().cache_stats(),bounds_entries=len(self._bounds_cache),bounds_hits=self.bounds_hits,bounds_misses=self.bounds_misses,bounds_capacity_bypasses=self.bounds_capacity_bypasses)


@lru_cache(maxsize=20000)
def _topology(coefficients,threshold,lo,hi,digits):
 """Only exact target inequality geometry: independent of data/prior/numerics."""
 poly=sp.Poly(sum(v*X**i for i,v in enumerate(coefficients))-threshold,X,domain=sp.QQ)
 events=tuple(roots(poly,lo,hi,digits));points=sorted(set([lo,hi]+[(a+b)/2 for a,b,k in events]));active=tuple((a,b)for a,b in zip(points[:-1],points[1:])if poly.eval((a+b)/2)<=0)
 return events,active


def _clone(r):
 # All other fields are immutable scalars/strings; avoid copying whole models.
 return dict(r,root_boxes=[dict(x)for x in r['root_boxes']])


class MemoPolynomialPosterior(PolynomialPosterior):
 def __init__(self,*args,max_CDF_entries=10000,**kwargs):
  if type(max_CDF_entries)is not int or max_CDF_entries<1:raise ValueError('Positive CDF-cache capacity required')
  super().__init__(*args,**kwargs);self.max_CDF_entries=max_CDF_entries;self._CDF_cache={};self.CDF_hits=0;self.CDF_misses=0;self.CDF_capacity_bypasses=0;self._memo_context=self._context()
 def _context(self):
  models=tuple((c,m.coefficients,m.D.as_expr(),tuple(m.points),m.lo,m.hi,m.reference,m.log_ratio,m.log_noise_normalizer,tuple(m.bounds),m.noise.tobytes(),m.observed.tobytes(),m.root_digits,m.root_log_energy_tolerance)for c,m in self.models.items())
  return (models,tuple(self.categories),self.prior.tobytes(),self.Z,self.Z_error_estimate,self.reference_D,tuple(self.numerics.items()))
 def coordinate_cdf(self,target_coefficients,threshold,*,maximum_root_mass=1e-8):
  if self._context()!=self._memo_context:raise FloatingPointError('Posterior changed; reject stale exact coordinate-CDF cache')
  if set(target_coefficients)!=set(self.models) or not np.isfinite(threshold):raise ValueError('Separate target polynomial percategory and finite threshold required')
  if not np.isfinite(maximum_root_mass) or not 0<maximum_root_mass<=1e-6:raise ValueError('Explicit strict target-root mass tolerance required')
  cs={}
  for category in self.categories:
   c=np.asarray(target_coefficients[category],dtype=float)
   if c.shape!=(3,) or not np.isfinite(c).all():raise ValueError('One scalar target coordinate quadratic required')
   cs[category]=tuple(rational(v)for v in c)
  key=(tuple(cs[c]for c in self.categories),rational(threshold),float(maximum_root_mass))
  if key in self._CDF_cache:self.CDF_hits+=1;return _clone(self._CDF_cache[key])
  self.CDF_misses+=1;result=0.;errors=0.;uncertain_mass=0.;root_boxes=[]
  for pi,(category,m)in zip(self.prior,self.models.items()):
   events,active=_topology(cs[category],rational(threshold),m.lo,m.hi,m.root_digits)
   for a,b,multiplicity in events:
    if a<b:
     lower,_=m.D_bounds(a,b)
     with mp.workdps(90):
      q=lambda v:mp.mpf(int(sp.numer(v)))/mp.mpf(int(sp.denom(v)))
      log_bound=mp.log(mp.mpf(float(pi)))-mp.log(mp.mpf(self.Z-self.Z_error_estimate))+mp.log(mp.log1p(q((b-a)/a))/mp.mpf(m.log_ratio))-q(lower-self.reference_D)
      bound=min(1.,float(np.nextafter(float(mp.exp(min(log_bound,mp.mpf(0)))),np.inf)))
     uncertain_mass+=bound;root_boxes.append(dict(lo=str(a),hi=str(b),mass_upper=bound,log_mass_upper=float(log_bound),multiplicity=multiplicity))
   for a,b in active:
    v,e=m.integrate_x(a,b,reference_D=self.reference_D,**self.numerics);result+=pi*float(v)/self.Z;errors+=pi*e/self.Z
  if uncertain_mass>maximum_root_mass:raise FloatingPointError('Target-root posterior mass unresolved; no midpoint CDF admission')
  normalizer_error=self.Z_error_estimate/(self.Z-self.Z_error_estimate);total_error=errors+uncertain_mass+normalizer_error
  row=dict(cdf=float(result),cdf_lower=max(0.,float(result-total_error)),cdf_upper=min(1.,float(result+total_error)),numerical_error_estimate=float(errors),normalizer_error_estimate=normalizer_error,uncertain_target_root_mass_upper=uncertain_mass,root_boxes=root_boxes,observation_noise_added=False,root_scope='Conservative root-box mass vs computed normalizer lower estimate; quadrature error is estimated, not an interval certificate')
  if len(self._CDF_cache)<self.max_CDF_entries:self._CDF_cache[key]=_clone(row)
  else:self.CDF_capacity_bypasses+=1
  return row
 def memo_stats(self):return dict(CDF_entries=len(self._CDF_cache),CDF_hits=self.CDF_hits,CDF_misses=self.CDF_misses,CDF_capacity_bypasses=self.CDF_capacity_bypasses,topology_cache=dict(zip(['hits','misses','maxsize','currsize'],_topology.cache_info())),approximation=False)
