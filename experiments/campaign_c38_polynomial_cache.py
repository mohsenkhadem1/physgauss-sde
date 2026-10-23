"""Exact memoization of completed numeric mass segments; no interpolation.

Every value retains its exact rational endpoints, reference shift and full
numerical-method key. New partial endpoint segments are evaluated normally.
Target moment modifiers bypass this scalar cache. Frozen v02 source is unchanged.
"""
import numpy as np
import sympy as sp
from experiments.campaign_c38_polynomial_likelihood_v02 import GaussianPolynomialLikelihood

class CachedGaussianPolynomialLikelihood(GaussianPolynomialLikelihood):
 def __init__(self,*args,max_cache_entries=10000,**kwargs):
  if type(max_cache_entries)is not int or max_cache_entries<1:raise ValueError('Positive explicit cache capacity')
  super().__init__(*args,**kwargs);self.max_cache_entries=max_cache_entries;self._mass_cache={};self.cache_hits=0;self.cache_misses=0;self.cache_capacity_bypasses=0;self._model_context=self._context()
 def _context(self):
  return (self.coefficients,self.D.as_expr(),tuple(self.points),self.lo,self.hi,self.reference,self.log_ratio,self.log_noise_normalizer,tuple(self.bounds),self.noise.tobytes(),self.observed.tobytes())
 def integrate_x(self,lo,hi,*,reference_D,method='adaptive',order=64,rtol=1e-10,atol=1e-12,modifier=None):
  if self._context()!=self._model_context:raise FloatingPointError('Model changed; refuse stale numerical cache')
  if modifier is not None:return super().integrate_x(lo,hi,reference_D=reference_D,method=method,order=order,rtol=rtol,atol=atol,modifier=modifier)
  lo,hi=sp.Rational(lo),sp.Rational(hi)
  if not self.lo<=lo<=hi<=self.hi or method not in ('adaptive','legendre') or type(order)is not int or order<2 or not np.isfinite([rtol,atol]).all() or min(rtol,atol)<=0:raise ValueError('Explicit in-domain interval and numerical method required')
  if lo==hi:return 0.,0.
  points=sorted(set([lo,hi]+[x for x in self.points if lo<x<hi]));total=None;error=0.
  for left,right in zip(points[:-1],points[1:]):
   key=(left,right,sp.Rational(reference_D),method,order,float(rtol),float(atol))
   if key in self._mass_cache:value,err=self._mass_cache[key];self.cache_hits+=1
   else:
    value,err=super().integrate_x(left,right,reference_D=reference_D,method=method,order=order,rtol=rtol,atol=atol);self.cache_misses+=1
    if len(self._mass_cache)<self.max_cache_entries:self._mass_cache[key]=(value,err)
    else:self.cache_capacity_bypasses+=1
   total=value if total is None else total+value;error+=err
  return total,error
 def cache_stats(self):return dict(entries=len(self._mass_cache),hits=self.cache_hits,misses=self.cache_misses,capacity_bypasses=self.cache_capacity_bypasses,approximation=False)
