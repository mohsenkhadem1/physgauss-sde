"""NEW v02 central-CDF-domain guard repair; original resolved v01 is untouched.

Likelihood, prior, material and moment loops are copied unchanged. Exact memo
classes preserve arithmetic; only coordinate quantiles use the independent
relative-position AND numerical-CDF qualification. No actual-data execution
is authorized by this source. No files, noise, truth or policy inputs.
"""
from __future__ import annotations
import numpy as np
from experiments.campaign_c38_controlled_observations import validate_public_prefix,POINT_IDS
from experiments.campaign_c38_table_validation import POINTS
from experiments import campaign_c39_side_observations as typed
from experiments.campaign_c38_polynomial_likelihood_v02 import rational
from experiments.campaign_c38_exact_posterior_memo import MemoPolynomialPosterior as PolynomialPosterior
from experiments.campaign_c38_exact_posterior_memo import BoundsMemoLikelihood as CachedGaussianPolynomialLikelihood
CATEGORIES=('both_ends','separated_bottom_patches')
PROBABILITIES=(.025,.5,.975)
METHODS=('adaptive','GL64','GL128')
GATES=dict(logZ=1e-7,category_CDF=1e-6,logE_quantile=1e-5,target_mean_m=1e-7,target_interval_m=1e-7,target_variance_absolute_m2=1e-14,target_variance_relative=1e-3)
ROOT_POSITION_TOL_M=1e-8
ROOT_RELATIVE_POSITION_TOL=1e-8
ROOT_PROBABILITY_TOL=1e-6
from experiments.campaign_resolved_quantiles_v04 import polynomial_coordinate_quantile
ROOT_MATERIAL_UNIT_TOL=1e-7
ROOT_ITERATIONS=80


PROBE_KEYS = typed.PUBLIC_KEYS
SIDE='middle_side_2N'


def validate_acquired_probe(public,probe,*,design):
 """Strictly ONE sanitized acquiredSIDE record; oldDOWN posterior is not input."""
 typed.validate_public_probe(public,probe,design=design)
 if probe['action_id']!=SIDE:raise ValueError('This new compiler is SIDE-only; oldDOWN results are reused by hash')
 return probe


def compile_public_models(candidate,public,probe,*,design):
 """Fresh54 coordinates/prior; validate roles BEFORE hypothesis calls."""
 validate_public_prefix(public,design=design)
 validate_acquired_probe(public,probe,design=design)
 expected=np.array([75000.,np.sqrt(75000.*150000.),150000.])
 if tuple(candidate.categories)!=CATEGORIES or not np.array_equal(candidate.young_nodes,expected) or candidate.initial_candidate is None:raise ValueError('Exact same typedpublic candidate material/support scope required')
 observed=np.concatenate([(np.asarray(public['observed_positions_m'])[0]-np.asarray(public['reference_points_m'])).reshape(-1),(probe['observed_positions_m']-probe['reference_points_m']).reshape(-1)])
 C=np.zeros((54,54));C[:27,:27]=public['sensor_covariance_m2'];C[27:,27:]=probe['sensor_covariance_m2'];models={};targets={}
 for category in CATEGORIES:
  coef=np.concatenate([np.asarray(candidate.observation_coefficients(category,'gravity',public['qualified_point_ids']),dtype=float),np.asarray(candidate.observation_coefficients(category,SIDE,probe['qualified_point_ids']),dtype=float)],axis=1)
  target=np.asarray(candidate.observation_coefficients(category,'canonical_up_1N',np.arange(63,dtype=np.int64)),dtype=float)
  if coef.shape!=(3,54) or target.shape!=(3,189) or not np.isfinite(coef).all() or not np.isfinite(target).all():raise ValueError('Finite candidate displacement coefficients for exact acquired/target point scopes required')
  models[category]=CachedGaussianPolynomialLikelihood(coef,observed,C,young_bounds=(75000.,150000.),young_reference_Pa=75000.,root_digits=50,root_log_energy_tolerance=1e-12,log_drops=(1.,4.,16.,64.))
  targets[category]=target.copy()
 return models,targets


def material_cdf_bounds(post,E):
 if E<=post.bounds[0]:return dict(cdf=0.,lower=0.,upper=0.)
 if E>=post.bounds[1]:return dict(cdf=1.,lower=1.,upper=1.)
 values=[post.category_interval_mass(c,post.bounds[0],E) for c in post.categories];v=sum(x[0] for x in values);err=sum(x[1] for x in values)+post.Z_error_estimate/(post.Z-post.Z_error_estimate)
 return dict(cdf=float(v),lower=max(0.,float(v-err)),upper=min(1.,float(v+err)))


def _global_target_bounds(targets,post):
 values=[]
 for category,m in post.models.items():
  c=[rational(v) for v in targets[category]];x=[m.lo,m.hi]
  if c[2]!=0:
   critical=-c[1]/(2*c[2])
   if m.lo<critical<m.hi:x.append(critical)
  values += [c[0]+c[1]*v+c[2]*v*v for v in x]
 return min(values),max(values)


def coordinate_cdf_bounds(post,targets,value,*,left_limit=False):
 lo,hi=_global_target_bounds(targets,post)
 # A common constant target is exactly an atom, independently of posterior.
 if lo==hi:
  equal=float(lo);v=float(value>equal if left_limit else value>=equal)
  return dict(cdf=v,lower=v,upper=v,constant_target=True)
 row=post.coordinate_cdf(targets,float(value),maximum_root_mass=1e-8);v=row['cdf'];lower=row['cdf_lower'];upper=row['cdf_upper']
 if left_limit:
  atom=sum(post.category_probabilities[i] for i,c in enumerate(post.categories) if targets[c][1]==0 and targets[c][2]==0 and targets[c][0]==value)
  err=post.Z_error_estimate/(post.Z-post.Z_error_estimate);v-=atom;lower-=atom+err;upper-=atom-err
 return dict(cdf=float(v),lower=max(0.,float(lower)),upper=min(1.,float(upper)),constant_target=False)


def bounded_quantile(cdf,lower,upper,probability,*,tolerance,max_iterations=ROOT_ITERATIONS):
 """Return a bracket supported by numerical CDF bounds; fail if unresolved.

 An ambiguous middle CDF triggers two fixed quarter queries, not tighter
 quadrature or a changed likelihood. No finite-atom percentiles are used.
 """
 if not 0<probability<1 or not np.isfinite([lower,upper,tolerance]).all() or lower>=upper or tolerance<=0:raise ValueError('Explicit interior probability, bracket and tolerance required')
 L=cdf(lower);H=cdf(upper)
 if L['upper']>=probability or H['lower']<probability:raise FloatingPointError('Initial quantile CDF bounds do not bracket probability')
 calls=2
 for iteration in range(max_iterations):
  if upper-lower<=tolerance:return dict(value=(lower+upper)/2,lower=lower,upper=upper,cdf_at_lower=L,cdf_at_upper=H,iterations=iteration,CDF_calls=calls,kind='continuous_numerical_CDF_bracket')
  mid=lower+(upper-lower)/2;M=cdf(mid);calls+=1
  if M['upper']<probability:lower=mid;L=M
  elif M['lower']>=probability:upper=mid;H=M
  else:
   # Preserve only sign-qualified endpoints; no point-estimate sign substitute.
   q1=lower+(mid-lower)/2;q3=mid+(upper-mid)/2;A=cdf(q1);B=cdf(q3);calls+=2;changed=False
   if A['upper']<probability:lower=q1;L=A;changed=True
   if B['lower']>=probability:upper=q3;H=B;changed=True
   if not changed:raise FloatingPointError('Quantile unresolved at frozen CDF uncertainty; no tolerance escalation')
 raise FloatingPointError('Frozen quantile iteration cap exhausted')


def material_quantile(post,probability):
 a,b=post.bounds;logratio=np.log(b/a)
 f=lambda unit:material_cdf_bounds(post,float(a*np.exp(logratio*unit)))
 row=bounded_quantile(f,0.,1.,probability,tolerance=ROOT_MATERIAL_UNIT_TOL);return dict(young_Pa=float(a*np.exp(logratio*row['value'])),unit_bracket=row)


def coordinate_quantile(post,targets,probability):
 """Same pushforward law; require relative-position and CDF-qualified roots."""
 return polynomial_coordinate_quantile(post,targets,probability)


def _compare(a,b,*,intervals):
 variance_a=np.asarray(a['canonical_coordinate_variance_m2']);variance_b=np.asarray(b['canonical_coordinate_variance_m2']);varerr=abs(variance_a-variance_b);vartol=np.maximum(GATES['target_variance_absolute_m2'],GATES['target_variance_relative']*variance_a)
 r=dict(logZ_difference=abs(a['log_normalizer']-b['log_normalizer']),maximum_category_probability_difference=float(max(abs(np.asarray(a['category_probabilities'])-b['category_probabilities']))),maximum_material_CDF_difference=float(max(abs(np.asarray(a['material_CDF'])-b['material_CDF']))),maximum_logE_quantile_difference=float(max(abs(np.log(np.asarray(a['material_equal_tail_Pa'])/b['material_equal_tail_Pa'])))),maximum_canonical_mean_coordinate_difference_m=float(abs(np.asarray(a['canonical_mean_displacement_m'])-b['canonical_mean_displacement_m']).max()),maximum_canonical_variance_difference_m2=float(varerr.max()),all_coordinate_variances_within_gate=bool(np.all(varerr<=vartol)),mean_coordinate_variance_difference_m2=float(abs(variance_a.mean()-variance_b.mean())))
 if intervals:r['maximum_canonical_interval_endpoint_difference_m']=float(abs(np.asarray(a['canonical_displacement_equal_tail_m'])-b['canonical_displacement_equal_tail_m']).max())
 r['passed']=r['logZ_difference']<=GATES['logZ'] and r['maximum_category_probability_difference']<=GATES['category_CDF'] and r['maximum_material_CDF_difference']<=GATES['category_CDF'] and r['maximum_logE_quantile_difference']<=GATES['logE_quantile'] and r['maximum_canonical_mean_coordinate_difference_m']<=GATES['target_mean_m'] and r['all_coordinate_variances_within_gate'] and (not intervals or r['maximum_canonical_interval_endpoint_difference_m']<=GATES['target_interval_m'])
 return r


def infer_public_probe(candidate,public,probe,*,design):
 """Zero-I/O function. Return all method outputs/failures; no future/truth API.

 ``completed`` requires every method/function to finish. Admission additionally
 requires ``numerical_controls_passed`` plus the caller's independent forward
 qualification. Object ID is returned for binding only, never used numerically.
 """
 result=dict(completed=False,numerical_controls_passed=False,information_role='C39_PUBLIC_PREFIX_PLUS_ONE_TYPED_SIDE_POSTERIOR_NOT_CALIBRATION',methods={},failures=[],physical_calibration_established=False,is_RGBD_perception=False,is_temporal_forecast=False,noisy_future_measurement_variance_added=False)
 try:
  validate_acquired_probe(public,probe,design=design)
  result.update(object_id=public['object_id'],acquisition_index=probe['acquisition_index'],acquired_probe_action_id=probe['action_id'],acquired_probe_force_world_N=probe['force_vector_N'].copy(),acquired_coordinate_count=54,fresh_prior=True,initial_posterior_weights_reused=False)
  models,targets=compile_public_models(candidate,public,probe,design=design)
 except Exception as error:result['failures'].append(dict(stage='PUBLIC_INPUT_OR_CANDIDATE_VALIDATION',error=repr(error)));return result
 result.update(object_id=public['object_id'],acquisition_index=probe['acquisition_index'],acquired_probe_action_id=probe['action_id'],acquired_probe_force_world_N=probe['force_vector_N'].copy(),acquired_coordinate_count=54,fresh_prior=True,initial_posterior_weights_reused=False,categories=CATEGORIES,material_CDF_grid_Pa=np.geomspace(75000.,150000.,17),canonical_point_ids=np.arange(63,dtype=np.int64),canonical_interval_point_ids=POINT_IDS.copy(),equal_tail_probabilities=PROBABILITIES,gates=GATES.copy(),quantile_resolution=dict(target_maximum_absolute_bracket_width_m=ROOT_POSITION_TOL_M,target_relative_global_polynomial_range=ROOT_RELATIVE_POSITION_TOL,target_bracket_rule='min(10nm,1e-8*global_polynomial_range)',target_returned_CDF_probability_tolerance=ROOT_PROBABILITY_TOL,material_unit_bracket_width=ROOT_MATERIAL_UNIT_TOL,scope='Calculated quantile refinement only; unchanged posterior/material/moments/CDF uncertainty; fail unresolved; no interval widening or calibration claim'));posts={}
 for name in METHODS:
  row=dict(completed=False,stage='NORMALIZER');result['methods'][name]=row
  try:
   post=PolynomialPosterior(models,method='adaptive' if name=='adaptive' else 'legendre',order=64 if name=='GL64' else 128);posts[name]=post;row.update(log_normalizer=post.log_normalizer,normalizer_error_estimate=post.Z_error_estimate,scaled_normalizer=post.Z,normalizer_relative_error_estimate=post.Z_error_estimate/post.Z,category_probabilities=post.category_probabilities.copy());row['stage']='MATERIAL'
   row['material_CDF']=[];row['material_quantile_details']=[]
   for E in result['material_CDF_grid_Pa']:row['material_CDF'].append(post.cdf(E))
   for p in PROBABILITIES:row['material_quantile_details'].append(material_quantile(post,p))
   row['material_equal_tail_Pa']=[r['young_Pa'] for r in row['material_quantile_details']];row['stage']='CANONICAL_MOMENTS'
   moments=post.target_moments(targets,scale_m=200e-6);mean=moments['mean_m'].reshape(63,3);variance=moments['variance_m2'].reshape(63,3);row.update(canonical_mean_displacement_m=mean,canonical_mean_positions_m=POINTS+mean,canonical_coordinate_variance_m2=variance,canonical_mean_coordinate_variance_m2=float(variance.mean()),canonical_moment_numerics={k:v for k,v in moments.items() if k not in ['mean_m','variance_m2','variance_center_m']})
   numerical_var=200e-6**2*moments['second_pass_scaled_integral_error_estimate']/(post.Z-post.Z_error_estimate)+variance*post.Z_error_estimate/(post.Z-post.Z_error_estimate)
   row['canonical_variance_integration_resolution_estimate_m2']=numerical_var
   row['canonical_variance_resolution_scope']='Estimated integration/normalizer error around returned binary64 mean; mean-rounding squared offset and physical forward error are separate; not a calibrated variance floor.'
   if name!='GL64':
    row['stage']='CANONICAL_QUANTILES';details=[];row['canonical_quantile_details']=details
    for point in POINT_IDS:
     for axis in range(3):
      coordinate=int(point)*3+axis;coef={c:v[:,coordinate] for c,v in targets.items()};entry=dict(point_id=int(point),axis=axis,quantiles=[]);details.append(entry)
      for p in PROBABILITIES:entry['quantiles'].append(coordinate_quantile(post,coef,p))
    intervals=np.asarray([[q['value_m'] for q in r['quantiles']] for r in details]).reshape(9,3,3);row.update(canonical_quantile_details=details,canonical_displacement_equal_tail_m=intervals,canonical_position_equal_tail_m=intervals+POINTS[POINT_IDS,:,None])
    lower=np.asarray([[q['lower_m'] for q in r['quantiles']] for r in details]).reshape(9,3,3);upper=np.asarray([[q['upper_m'] for q in r['quantiles']] for r in details]).reshape(9,3,3);row.update(canonical_displacement_quantile_lower_m=lower,canonical_displacement_quantile_upper_m=upper,canonical_position_quantile_lower_m=lower+POINTS[POINT_IDS,:,None],canonical_position_quantile_upper_m=upper+POINTS[POINT_IDS,:,None],canonical_position_interval95_numerical_enclosure_m=np.stack([lower[:,:,0],upper[:,:,2]],-1)+POINTS[POINT_IDS,:,None])
   row.update(completed=True,stage='COMPLETED')
  except Exception as error:row['error']=repr(error);result['failures'].append(dict(method=name,stage=row['stage'],error=repr(error)))
 if not all(r['completed'] for r in result['methods'].values()):return result
 result['comparisons']={'adaptive_vs_GL64':_compare(result['methods']['adaptive'],result['methods']['GL64'],intervals=False),'adaptive_vs_GL128':_compare(result['methods']['adaptive'],result['methods']['GL128'],intervals=True)};checks=[]
 try:
  for source in ['adaptive','GL128']:
   for entry in result['methods'][source]['canonical_quantile_details']:
    point,axis=entry['point_id'],entry['axis'];coef={c:v[:,point*3+axis] for c,v in targets.items()}
    for p,q in zip(PROBABILITIES,entry['quantiles']):
     target=q['value_m'];cdf=coordinate_cdf_bounds(posts['GL64'],coef,target);left=coordinate_cdf_bounds(posts['GL64'],coef,target,left_limit=True);err=max(abs(cdf['cdf']-q['CDF_at']['cdf']),abs(left['cdf']-q['CDF_left']['cdf']));checks.append(dict(source_method=source,point_id=point,axis=axis,probability=p,endpoint_displacement_m=target,CDF_difference=err,passed=err<=GATES['category_CDF']))
  result.update(completed=True,GL64_endpoint_CDF_checks=checks,numerical_controls_passed=all(c['passed'] for c in result['comparisons'].values()) and all(c['passed'] for c in checks),cache_stats={c:m.cache_stats() for c,m in models.items()})
 except Exception as error:result['failures'].append(dict(stage='GL64_ENDPOINT_CDF_CHECK',error=repr(error)))
 return result
