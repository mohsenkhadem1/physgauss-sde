"""Prospective matching-prior surrogate SBC primitives. No I/O or random draws.

Exact NL stored polynomial is the generative model; this is not FEM calibration.
Observations are displacements, avoiding reference-position roundtrip arithmetic.
"""
import numpy as np
from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood, MemoPolynomialPosterior
from experiments.campaign_c38_controlled_observations import POINT_IDS
from experiments.campaign_resolved_prefix_inference import material_cdf_bounds
CATEGORIES=('both_ends','separated_bottom_patches')
METHODS=('adaptive','GL64','GL128')
PHASES=('prefix','SIDE')
NOISE=200e-6
GATES=dict(logZ=1e-7,probability=1e-6,mean_m=1e-7,variance_absolute_m2=1e-14,variance_relative=1e-3)


def evaluate_polynomial(c,E):
 c=np.asarray(c,float)
 if c.shape[0]!=3 or not np.isfinite(c).all() or not 75000<=E<=150000:raise ValueError('Finite declared quadratic/material')
 x=75000./float(E);return (c[2]*x+c[1])*x+c[0]


def validate_coefficients(coeff):
 if set(coeff)!=set(CATEGORIES):raise ValueError('Exactly two frozen support hypotheses')
 for c in CATEGORIES:
  if set(coeff[c])!={'prefix','SIDE','target'}:raise ValueError('Only gravity/fixed SIDE/canonical coefficients')
  for key,n in [('prefix',27),('SIDE',27),('target',189)]:
   a=np.asarray(coeff[c][key])
   if a.shape!=(3,n)or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Exact finite float64 polynomial shape')


def compile_laws(coeff,observed,phase):
 validate_coefficients(coeff)
 if phase not in PHASES or set(observed)!=set(PHASES):raise ValueError('Exact acquired displacement observations')
 for y in observed.values():
  if np.asarray(y).shape!=(27,)or np.asarray(y).dtype!=np.float64 or not np.isfinite(y).all():raise ValueError('Finite float64 acquired27vector')
 keys=('prefix',)if phase=='prefix'else PHASES;y=np.concatenate([observed[k]for k in keys]);C=NOISE**2*np.eye(len(y));laws={}
 for c in CATEGORIES:
  a=np.concatenate([coeff[c][k]for k in keys],axis=1)
  laws[c]=BoundsMemoLikelihood(a,y,C,young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
 return laws


def posterior(laws,method):
 if method not in METHODS:raise ValueError('Frozen independent quadratures')
 return MemoPolynomialPosterior(laws,category_prior=[.5,.5],method='adaptive'if method=='adaptive'else'legendre',order=64 if method=='GL64'else 128)


def compile_inference(coeff,observed,phase):
 """Compile sealed laws/moments only. No coordinate OR material quantile roots."""
 posts={};result=dict(completed=False,numerical_controls_passed=False,phase=phase,methods={},failures=[])
 try:laws=compile_laws(coeff,observed,phase)
 except Exception as e:result['failures'].append(dict(stage='COMPILE',error=repr(e)));return result,posts
 for method in METHODS:
  try:
   p=posterior(laws,method);posts[method]=p;mom=p.target_moments({c:coeff[c]['target']for c in CATEGORIES},scale_m=NOISE)
   result['methods'][method]=dict(completed=True,log_normalizer=p.log_normalizer,scaled_normalizer=p.Z,normalizer_error_estimate=p.Z_error_estimate,category_probabilities=p.category_probabilities.copy(),material_CDF=np.array([p.cdf(E)for E in np.geomspace(75000.,150000.,17)]),moments=mom)
  except Exception as e:result['methods'][method]=dict(completed=False,error=repr(e));result['failures'].append(dict(method=method,error=repr(e)))
 if not all(r['completed']for r in result['methods'].values()):return result,posts
 a=result['methods']['adaptive'];comparisons={}
 for method in METHODS[1:]:
  b=result['methods'][method];va=a['moments']['variance_m2'];vb=b['moments']['variance_m2'];checks=dict(logZ=abs(a['log_normalizer']-b['log_normalizer'])<=GATES['logZ'],category=bool(np.max(abs(a['category_probabilities']-b['category_probabilities']))<=GATES['probability']),material=bool(np.max(abs(a['material_CDF']-b['material_CDF']))<=GATES['probability']),mean=bool(np.max(abs(a['moments']['mean_m']-b['moments']['mean_m']))<=GATES['mean_m']),variance=bool(np.all(abs(va-vb)<=np.maximum(GATES['variance_absolute_m2'],GATES['variance_relative']*va))))
  comparisons[method]=dict(passed=all(checks.values()),checks=checks)
 result.update(completed=True,comparisons=comparisons,numerical_controls_passed=all(c['passed']for c in comparisons.values()));return result,posts



def infer(coeff,observed,phase):return compile_inference(coeff,observed,phase)[0]


def probability_bounds(row):
 """Central endpoint projection only, original numerical bounds unchanged."""
 v,lo,hi=map(float,(row['cdf'],row['lower'],row['upper']))
 if not np.isfinite([v,lo,hi]).all()or not 0<=lo<=hi<=1:raise FloatingPointError('Invalid probability enclosure')
 if v<0 and lo==0:v=0.
 if v>1 and hi==1:v=1.
 if not lo<=v<=hi:raise FloatingPointError('Central estimate inconsistent with supplied bounds')
 return dict(cdf=v,lower=lo,upper=hi)


def coordinate_limits(post,targets,t):
 """Support-aware directed strict/non-strict CDF arithmetic, no quantile roots."""
 from experiments.campaign_resolved_quantiles_v04 import polynomial_support_cdf_bounds
 right=probability_bounds(polynomial_support_cdf_bounds(post,targets,float(t)))
 left=probability_bounds(polynomial_support_cdf_bounds(post,targets,float(t),left_limit=True))
 atom=any(a[1]==0 and a[2]==0 and a[0]==t for a in targets.values())
 if not atom and abs(left['cdf']-right['cdf'])>GATES['probability']:raise FloatingPointError('Continuous left/right numerical CDF disagreement')
 return dict(left=left,right=right,atom=atom)


def randomized_pit(left,right,V):
 L=probability_bounds(left);R=probability_bounds(right)
 if not np.isfinite(V)or not 0<=V<=1 or L['cdf']>R['cdf']or L['lower']>R['upper']:raise FloatingPointError('Ordered strict/non-strict probabilities and independent V')
 p={k:(1-V)*L[k]+V*R[k]for k in ('cdf','lower','upper')};p=probability_bounds(p)
 inside=p['lower']>=.025 and p['upper']<=.975;outside=p['upper']<.025 or p['lower']>.975
 return dict(**p,numerical_width=p['upper']-p['lower'],numerically_qualified=p['upper']-p['lower']<=2*GATES['probability'],randomized_equal_tail95_status='inside'if inside else'outside'if outside else'uncertain',atom=R['cdf']>L['cdf'])


def score_laws(coeff,observed,phase,truth,V,*,posts):
 """Only called after complete saved-law barrier; no quantile root finding."""
 if set(truth)!={'young_Pa','support_category','target_displacement_m'}or truth['support_category']not in CATEGORIES:raise ValueError('Selective generator labels only')
 E=float(truth['young_Pa']);u=np.asarray(truth['target_displacement_m'])
 if u.shape!=(63,3)or not np.array_equal(u,evaluate_polynomial(coeff[truth['support_category']]['target'],E).reshape(63,3)):raise ValueError('Exact unchanged generative target arithmetic')
 V=np.asarray(V)
 if V.shape!=(28,)or not np.isfinite(V).all()or np.any(V<0)or np.any(V>1):raise ValueError('Independent frozen28 PIT uniforms')
 rows={}
 for method in METHODS:
  try:
   p=posts[method];F=probability_bounds(material_cdf_bounds(p,E));scores=[randomized_pit(F,F,float(V[0]))]
   for i,point in enumerate(POINT_IDS):
    for axis in range(3):
     target={c:coeff[c]['target'][:,int(point)*3+axis]for c in CATEGORIES};a=coordinate_limits(p,target,float(u[point,axis]));# Under a continuous logE prior a nonconstant polynomial has zero
     # equality mass. Use its directed strict CDF for BOTH limits; the
     # independent non-strict reduction above must agree within its bounds.
     # Do not interpret reduction roundoff as an atom or randomize it.
     right=a['right']if a['atom']else a['left'];scores.append(randomized_pit(a['left'],right,float(V[1+i*3+axis])))
   rows[method]=dict(completed=True,scores=scores,support_true_probability=float(p.category_probabilities[CATEGORIES.index(truth['support_category'])]))
  except Exception as e:rows[method]=dict(completed=False,error=repr(e))
 qualified=all(r['completed']for r in rows.values());difference=None
 if qualified:
  a=rows['adaptive']['scores'];difference=max(abs(x['cdf']-y['cdf'])for m in METHODS[1:]for x,y in zip(a,rows[m]['scores']))
  qualified=difference<=GATES['probability']and all(x['numerically_qualified']for r in rows.values()for x in r['scores'])
 return dict(methods=rows,numerical_controls_passed=qualified,maximum_method_PIT_difference=difference)


def aggregate(rows):
 """All64 objects retained; each PIT quantity has its OWN denominator64."""
 if len(rows)!=64 or len({r['object_index']for r in rows})!=64 or {r['object_index']for r in rows}!=set(range(64)):raise ValueError('Exact all64 independent SBC objects')
 out=dict(objects=64,numerically_admitted=sum(r['admitted']for r in rows),quantities={})
 from scipy.stats import beta
 for j in range(28):
  inside=outside=uncertain=failed=0;bins=np.zeros(10,dtype=np.int64)
  for r in rows:
   if not r['admitted']:failed+=1;continue
   score=r['evaluation']['methods']['adaptive']['scores'][j];status=score['randomized_equal_tail95_status'];bins[min(9,int(10*score['cdf']))]+=1
   inside+=status=='inside';outside+=status=='outside';uncertain+=status=='uncertain'
  # Definite coverage counts failures/uncertainty as failures; upper possible
  # count is separate. Exact CI concerns independent objects for THIS quantity.
  lo=0. if inside==0 else float(beta.ppf(.025,inside,65-inside));hi=1. if inside==64 else float(beta.ppf(.975,inside+1,64-inside))
  out['quantities'][str(j)]=dict(denominator_including_failures=64,inside=inside,outside=outside,uncertain=uncertain,failed=failed,PIT_histogram_counts=bins.tolist(),definite_coverage_binomial_lower=lo,definite_coverage_binomial_upper=hi)
 return out
