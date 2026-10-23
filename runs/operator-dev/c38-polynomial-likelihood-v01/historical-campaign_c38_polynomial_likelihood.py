"""Specialised manufactured-stage Gaussian/quadratic likelihood, not admitted physics.

Binary64 whitened residual coefficients define an exact rational polynomial.
Sturm isolation covers stationary and likelihood-level roots of THAT polynomial;
it does not certify floating whitening or a physical/interpolated forward model.
All partitions are integrated in uniform log E, never uniformly in inverse E.
Targets are separate, noiseless-polynomial arguments; no sensor noise is added.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import quad_vec
from numpy.polynomial.legendre import leggauss
X=sp.Symbol('x')

def rational(value):
 value=float(value)
 if not np.isfinite(value):raise ValueError('Finite stored coefficient required')
 a,b=value.as_integer_ratio();return sp.Rational(a,b)

def roots(poly,lo,hi,digits):
 p=sp.Poly(poly,X,domain=sp.QQ)
 if p.is_zero or p.degree()==0:return []
 return [(a,b,int(m)) for (a,b),m in p.intervals(eps=sp.Rational(1,10**digits),inf=lo,sup=hi)]

def _interval_mul(a,b):
 t=[x*y for x in a for y in b];return min(t),max(t)

def _residual_bounds(c,lo,hi):
 b=_interval_mul((c[2],c[2]),(lo,hi));b=(b[0]+c[1],b[1]+c[1]);b=_interval_mul(b,(lo,hi));return b[0]+c[0],b[1]+c[0]

def _square_bounds(a,b):
 return (0 if a<=0<=b else min(a*a,b*b)),max(a*a,b*b)

class GaussianPolynomialLikelihood:
 """Quadratic predicted acquired observations in x=E_reference/E only.

 Coefficients shape[3,D], observed[D], fixed SPD noise[D,D]. Subtraction is
 performed before whitening. No labels, targets, unacquired arrays, or roles.
 """
 def __init__(self,predicted_observation_coefficients,observed,noise_covariance,*,young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,log_drops=(1.,4.,16.,64.)):
  c=np.asarray(predicted_observation_coefficients,dtype=float);y=np.asarray(observed,dtype=float);C=np.asarray(noise_covariance,dtype=float);bounds=np.asarray(young_bounds,dtype=float)
  if c.ndim!=2 or c.shape[0]!=3 or y.shape!=(c.shape[1],) or not len(y) or C.shape!=(len(y),len(y)) or not all(np.isfinite(a).all() for a in [c,y,C]) or not np.allclose(C,C.T,rtol=1e-13,atol=0):raise ValueError('Finite quadratic observations and fixed symmetric covariance required')
  if bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1] or not np.isfinite(young_reference_Pa) or young_reference_Pa<=0 or type(root_digits)is not int or not 20<=root_digits<=100:raise ValueError('Finite positive bounds/reference and explicit root precision required')
  drops=np.asarray(log_drops,dtype=float)
  if drops.ndim!=1 or not len(drops) or not np.isfinite(drops).all() or np.any(drops<=0) or np.any(np.diff(drops)<=0):raise ValueError('Fixed positive increasing level drops required')
  L=np.linalg.cholesky(C);r=c.copy();r[0]-=y;self.whitened=np.linalg.solve(L,r.T).T;self.noise=C.copy();self.bounds=bounds.copy();self.reference=float(young_reference_Pa);self.root_digits=root_digits;self.log_ratio=float(np.log(bounds[1]/bounds[0]));self.log_noise_normalizer=float(-.5*len(y)*np.log(2*np.pi)-np.log(np.diag(L)).sum())
  self.coefficients=tuple(tuple(rational(self.whitened[k,d]) for k in range(3)) for d in range(len(y)));self.lo=rational(self.reference)/rational(bounds[1]);self.hi=rational(self.reference)/rational(bounds[0]);self.D=sp.Poly(sum((a+b*X+c*X**2)**2/2 for a,b,c in self.coefficients),X,domain=sp.QQ);self.stationary=roots(self.D.diff(),self.lo,self.hi,root_digits)
  candidates=[(self.lo,self.lo),(self.hi,self.hi)]+[(a,b) for a,b,m in self.stationary];boxes=[]
  for a,b in candidates:
   sq=[_square_bounds(*_residual_bounds(c,a,b)) for c in self.coefficients];boxes.append((sum(x[0] for x in sq)/2,sum(x[1] for x in sq)/2))
  self.minimum_D_lower=min(x[0] for x in boxes);self.minimum_D_upper=min(x[1] for x in boxes)
  if float(self.minimum_D_upper-self.minimum_D_lower)>1e-12:raise FloatingPointError('Stationary isolation insufficient for declared log-likelihood enclosure; no automatic budget change')
  self.reference_D=min(self.D.eval((a+b)/2) for a,b in candidates)
  self.events=[dict(kind='stationary',lo=str(a),hi=str(b),multiplicity=m) for a,b,m in self.stationary];points=[self.lo,self.hi]+[(a+b)/2 for a,b,m in self.stationary]
  for drop in drops:
   for a,b,m in roots(self.D.as_expr()-self.reference_D-rational(drop),self.lo,self.hi,root_digits):
    self.events.append(dict(kind='level',drop=float(drop),lo=str(a),hi=str(b),multiplicity=m));points.append((a+b)/2)
  self.points=sorted(set(points))
  if len(self.points)<2:raise FloatingPointError('Unresolved partition')

 def local(self,center_x):
  x=sp.Rational(center_x)
  r0=np.array([float(a+b*x+c*x*x) for a,b,c in self.coefficients]);r1=np.array([float(b+2*c*x) for a,b,c in self.coefficients]);r2=np.array([float(c) for a,b,c in self.coefficients]);return r0,r1,r2

 def local_D_change(self,center_x,delta_x):
  a,b,c=self.local(center_x);d=float(delta_x)
  return float(d*(a@b)+d*d*(a@c+.5*(b@b))+d**3*(b@c)+.5*d**4*(c@c))

 def log_likelihood(self,young_Pa):
  E=float(young_Pa)
  if not np.isfinite(E) or not self.bounds[0]<=E<=self.bounds[1]:raise ValueError('No modulus extrapolation')
  x=rational(self.reference)/rational(E);r=np.array([float(a+b*x+c*x*x) for a,b,c in self.coefficients]);return self.log_noise_normalizer-.5*float(r@r)

 def integrate_x(self,lo,hi,*,reference_D,method='adaptive',order=64,rtol=1e-10,atol=1e-12,modifier=None):
  """Integrate positive scaled likelihood over uniform logE, all segments.

  Local rescaling keeps tiny isolated modes visible to each quadrature. Error
  estimates are numerical (not interval certificates). No tail truncation.
  """
  lo,hi=sp.Rational(lo),sp.Rational(hi)
  if not self.lo<=lo<=hi<=self.hi or method not in ('adaptive','legendre') or type(order)is not int or order<2 or not np.isfinite([rtol,atol]).all() or min(rtol,atol)<=0:raise ValueError('Explicit in-domain interval and numerical method required')
  if lo==hi:return 0.,0.
  points=sorted(set([lo,hi]+[x for x in self.points if lo<x<hi]));total=None;error=0.;nodes,weights=leggauss(order)
  for left,right in zip(points[:-1],points[1:]):
   center=(left+right)/2;r0,r1,r2=self.local(center);Doffset=float(self.D.eval(center)-reference_D)
   with mp.workdps(90):
    f=lambda q:mp.mpf(int(sp.numer(q)))/mp.mpf(int(sp.denom(q)))
    tl=float(-mp.log1p(f((right-center)/center))/mp.mpf(self.log_ratio));tr=float(-mp.log1p(f((left-center)/center))/mp.mpf(self.log_ratio))
   width=tr-tl
   if not np.isfinite(width) or width<=0:raise FloatingPointError('Nonrepresentable positive partition width; no dropped interval')
   def integrand(s):
    dt=tl+width*float(s);dx=float(center)*np.expm1(-self.log_ratio*dt);change=dx*(r0@r1)+dx*dx*(r0@r2+.5*(r1@r1))+dx**3*(r1@r2)+.5*dx**4*(r2@r2);weight=np.exp(-(Doffset+change))
    if not np.isfinite(weight):raise FloatingPointError('Unresolved likelihood scaling')
    return weight if modifier is None else weight*np.asarray(modifier(center,dx),dtype=float)
   if method=='adaptive':
    value,err,info=quad_vec(integrand,0.,1.,epsrel=rtol,epsabs=atol,norm='max',limit=256,full_output=True)
    if not info.success:raise FloatingPointError('Partition integration failed; no status waiver')
   else:value=np.tensordot(weights/2,np.asarray([integrand((s+1)/2) for s in nodes]),axes=([0],[0]));err=0.
   total=width*value if total is None else total+width*value;error+=width*float(err)
  if not np.isfinite(total).all():raise FloatingPointError('Nonfinite partition integral')
  return total,error

class PolynomialPosterior:
 """Numerical working-model posterior; no physical admission or calibration."""
 def __init__(self,models,*,category_prior=None,method='adaptive',order=64,rtol=1e-10,atol=1e-12):
  if not isinstance(models,dict) or not models:raise ValueError('Explicit category likelihoods required')
  self.models=dict(models);self.categories=tuple(models);self.prior=np.asarray([1/len(models)]*len(models) if category_prior is None else category_prior,dtype=float);first=next(iter(models.values()))
  if self.prior.shape!=(len(models),) or not np.isfinite(self.prior).all() or np.any(self.prior<=0) or not np.isclose(self.prior.sum(),1,rtol=0,atol=1e-15):raise ValueError('Explicit positive normalized category prior required')
  if any(not np.array_equal(m.bounds,first.bounds) or not np.array_equal(m.noise,first.noise) or m.reference!=first.reference for m in models.values()):raise ValueError('Common known material prior and physical sensor covariance required')
  self.bounds=first.bounds.copy();self.reference_D=min(m.reference_D for m in models.values());self.numerics=dict(method=method,order=order,rtol=rtol,atol=atol);self.errors=[];mass=[]
  for pi,m in zip(self.prior,self.models.values()):
   v,e=m.integrate_x(m.lo,m.hi,reference_D=self.reference_D,**self.numerics);mass.append(pi*float(v));self.errors.append(pi*e)
  self.mass=np.asarray(mass);self.Z=float(self.mass.sum());self.Z_error_estimate=float(sum(self.errors))
  if not np.isfinite(self.Z) or self.Z<=self.Z_error_estimate or self.Z<=0 or self.Z_error_estimate/self.Z>1e-8:raise FloatingPointError('Unresolved normalized posterior mass')
  self.category_probabilities=self.mass/self.Z;self.log_normalizer=float(np.log(self.Z)-float(self.reference_D)+first.log_noise_normalizer)

 def category_interval_mass(self,category,lower_E,upper_E):
  m=self.models[category]
  if not np.isfinite([lower_E,upper_E]).all() or not self.bounds[0]<=lower_E<=upper_E<=self.bounds[1]:raise ValueError('In-domain material interval required')
  v,e=m.integrate_x(rational(m.reference)/rational(upper_E),rational(m.reference)/rational(lower_E),reference_D=self.reference_D,**self.numerics);pi=self.prior[self.categories.index(category)];return float(pi*v/self.Z),float(pi*e/self.Z)

 def cdf(self,E):
  if not np.isfinite(E):raise ValueError('Finite CDF query')
  if E<=self.bounds[0]:return 0.
  if E>=self.bounds[1]:return 1.
  return sum(self.category_interval_mass(c,self.bounds[0],E)[0] for c in self.categories)

 def coordinate_cdf(self,target_coefficients,threshold):
  """Quadratic inequality split; no indicator integrand or finite-atom CDF."""
  if set(target_coefficients)!=set(self.models) or not np.isfinite(threshold):raise ValueError('Separate target polynomial percategory and finite threshold required')
  result=0.;errors=0.
  for pi,(category,m) in zip(self.prior,self.models.items()):
   c=np.asarray(target_coefficients[category],dtype=float)
   if c.shape!=(3,) or not np.isfinite(c).all():raise ValueError('One scalar target coordinate quadratic required')
   poly=sp.Poly(sum(rational(v)*X**i for i,v in enumerate(c))-rational(threshold),X,domain=sp.QQ);events=roots(poly,m.lo,m.hi,m.root_digits);points=sorted(set([m.lo,m.hi]+[(a+b)/2 for a,b,k in events]))
   for a,b in zip(points[:-1],points[1:]):
    if poly.eval((a+b)/2)<=0:
     v,e=m.integrate_x(a,b,reference_D=self.reference_D,**self.numerics);result+=pi*float(v)/self.Z;errors+=pi*e/self.Z
  return dict(cdf=float(result),numerical_error_estimate=float(errors),observation_noise_added=False,root_scope='Exact binary-coefficient quadratic inequality; partition roots isolated, midpoint root error remains enclosed by isolation intervals')

 def target_moments(self,target_coefficients,*,scale_m=200e-6):
  if set(target_coefficients)!=set(self.models) or not np.isfinite(scale_m) or scale_m<=0:raise ValueError('Separate targets and declared positive scale required')
  cs={k:np.asarray(v,dtype=float) for k,v in target_coefficients.items()};first=next(iter(cs.values()))
  if first.ndim!=2 or first.shape[0]!=3 or any(v.shape!=first.shape or not np.isfinite(v).all() for v in cs.values()):raise ValueError('Matching target coefficient arrays[3,D] required')
  m0=next(iter(self.models.values()));x0=(m0.lo+m0.hi)/2;anchor=[sum(rational(first[i,d])*x0**i for i in range(3)) for d in range(first.shape[1])];summed=np.zeros(2*len(anchor));error=0.
  for pi,(category,m) in zip(self.prior,self.models.items()):
   c=cs[category];qq=[tuple(rational(c[i,d]) for i in range(3)) for d in range(c.shape[1])]
   cached={}
   def modifier(center,dx):
    if center not in cached:
     cached[center]=(np.array([float(c0+c1*center+c2*center**2-ref) for (c0,c1,c2),ref in zip(qq,anchor)]),np.array([float(c1+2*c2*center) for c0,c1,c2 in qq]),np.array([float(c2) for c0,c1,c2 in qq]))
    a,b,cc=cached[center];delta=(a+dx*(b+dx*cc))/scale_m;return np.r_[delta,delta*delta]
   value,err=m.integrate_x(m.lo,m.hi,reference_D=self.reference_D,modifier=modifier,**self.numerics);summed+=pi*value;error+=pi*err
  d=len(anchor);mean=summed[:d]/self.Z;variance=scale_m**2*(summed[d:]/self.Z-mean*mean)
  if np.any(variance<0) or not np.isfinite(variance).all():raise FloatingPointError('Unresolved negative centred target variance; no clipping')
  return dict(mean_m=np.array([float(a) for a in anchor])+scale_m*mean,variance_m2=variance,numerical_scaled_integral_error_estimate=error,observation_noise_added=False,scope='Noiseless canonical target under numeric likelihood only; no physical calibration')
