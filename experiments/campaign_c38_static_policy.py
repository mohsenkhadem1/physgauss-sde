"""Public-only, static reset-probe design under an explicitly numerical posterior.

No files, private labels, counterfactual measurements, durations or motion histories.
The two actions each cost ONE acquired 27-coordinate vector after a common-rest
reset. Fantasy common random numbers reduce comparison noise; they do not assert
correlated noise between real acquisitions. Canonical targets are noiseless fields.
"""
from __future__ import annotations
import hashlib
import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from experiments.campaign_c38_public_prefix_inference_v02 import compile_public_models,GATES
from experiments.campaign_c38_polynomial_likelihood_v02 import PolynomialPosterior,rational

ORDERS=(32,64,128)
FANTASY_COUNTS=(64,128,256)
ACTIONS=(-2.,2.)
FANTASY_ROOT_SEED=2026102302
RANDOM_ROOT_SEED=2026102303
DECISION_RULE=dict(paired_MCSE_multiplier=2.,cloud_advantage_discrepancy_multiplier=2.,
 tie_action_index=0,fallback_action_index=0,require_same_action_all_orders_and_budgets=True,
 scope='A stability/tie rule, not a confidence guarantee or an adaptive MC budget')


def _local_polynomial(coeff,center,dx):
 """Evaluate stored quadratic through a rational local center, including constants."""
 c=np.asarray(coeff,dtype=float)
 a=np.array([float(rational(c[0,d])+rational(c[1,d])*center+rational(c[2,d])*center**2) for d in range(c.shape[1])])
 b=np.array([float(rational(c[1,d])+2*rational(c[2,d])*center) for d in range(c.shape[1])])
 return a[None,:]+dx[:,None]*(b[None,:]+dx[:,None]*c[2][None,:])


def positive_cloud(post,targets,action_coefficients,*,order):
 """GL nodes on EVERY exact root partition, with the uniform-logE measure.

 All nodes and categories are retained, including underflowed numerical weights.
 Log weights are saved; no finite cloud is claimed to certify arbitrary tails.
 """
 if order not in ORDERS:raise ValueError('Only prospectively fixed cloud orders allowed')
 if set(targets)!=set(post.categories) or len(action_coefficients)!=2:raise ValueError('Separate targets and exactly two known action predictions required')
 nodes,gw=leggauss(order);logmass=[];categories=[];material=[];tvalues=[];avalues=[[],[]];partitions=0
 for index,(category,m) in enumerate(post.models.items()):
  for left,right in zip(m.points[:-1],m.points[1:]):
   center=(left+right)/2
   with mp.workdps(90):
    f=lambda q:mp.mpf(int(q.p))/mp.mpf(int(q.q))
    tl=float(-mp.log1p(f((right-center)/center))/mp.mpf(m.log_ratio));tr=float(-mp.log1p(f((left-center)/center))/mp.mpf(m.log_ratio))
   width=tr-tl
   if not np.isfinite(width) or width<=0:raise FloatingPointError('Unrepresentable positive root interval; no dropped partition')
   dt=tl+width*(nodes+1)/2;dx=float(center)*np.expm1(-m.log_ratio*dt);x=float(center)+dx
   # Do not replace a sub-ULP local mode by repeated material evaluations.
   if np.any(x<=0) or not np.isfinite(x).all():raise FloatingPointError('Invalid cloud material coordinates')
   r0,r1,r2=m.local(center);off=float(m.D.eval(center)-post.reference_D)
   change=dx*(r0@r1)+dx**2*(r0@r2+.5*(r1@r1))+dx**3*(r1@r2)+.5*dx**4*(r2@r2);d=off+change
   for j in np.flatnonzero(abs(off)+abs(change)>1e8*np.maximum(1.,abs(d))):d[j]=float(m.D.eval(center+rational(dx[j]))-post.reference_D)
   logmass.extend(np.log(post.prior[index])+np.log(width)+np.log(gw/2)-d)
   categories.extend([index]*order);material.extend(m.reference/x);tvalues.append(_local_polynomial(targets[category],center,dx))
   for a in range(2):avalues[a].append(_local_polynomial(action_coefficients[a][category],center,dx))
   partitions+=1
 logmass=np.asarray(logmass);logZ=float(logsumexp(logmass));logw=logmass-logZ;w=np.exp(logw);T=np.concatenate(tvalues);A=np.asarray([np.concatenate(x) for x in avalues]);ids=np.asarray(categories,dtype=np.int64)
 if not np.isfinite(logmass).all() or not np.isfinite(T).all() or not np.isfinite(A).all() or abs(w.sum()-1)>1e-12:raise FloatingPointError('Invalid positive cloud')
 mean=w@(T-T[0])+T[0];var=w@((T-mean)**2);prob=np.bincount(ids,weights=w,minlength=len(post.categories));first=next(iter(post.models.values()))
 return dict(order=order,weights=w,log_weights=logw,category_index=ids,young_Pa=np.asarray(material),targets_m=T,action_predictions_m=A,root_partition_count=partitions,
  log_normalizer=logZ-float(post.reference_D)+first.log_noise_normalizer,category_probabilities=prob,target_mean_m=mean,target_variance_m2=var,underflowed_weight_count=int(np.sum(w==0)),all_nodes_retained=True)


def check_cloud(cloud,post,reference_moments):
 mean=np.asarray(reference_moments['mean_m']);var=np.asarray(reference_moments['variance_m2']);dv=abs(cloud['target_variance_m2']-var);tol=np.maximum(GATES['target_variance_absolute_m2'],GATES['target_variance_relative']*var)
 r=dict(logZ_difference=abs(cloud['log_normalizer']-post.log_normalizer),maximum_category_probability_difference=float(abs(cloud['category_probabilities']-post.category_probabilities).max()),maximum_target_mean_difference_m=float(abs(cloud['target_mean_m']-mean).max()),maximum_target_variance_difference_m2=float(dv.max()),all_target_variances_within_gate=bool(np.all(dv<=tol)),node_count=len(cloud['weights']),root_partition_count=cloud['root_partition_count'],underflowed_weight_count=cloud['underflowed_weight_count'],all_nodes_retained=True)
 r['passed']=r['logZ_difference']<=GATES['logZ'] and r['maximum_category_probability_difference']<=GATES['category_CDF'] and r['maximum_target_mean_difference_m']<=GATES['target_mean_m'] and r['all_target_variances_within_gate']
 return r


def _seed(root,object_id):
 if not isinstance(object_id,str) or len(object_id)!=32 or any(c not in '0123456789abcdef' for c in object_id):raise ValueError('Validated opaque public ID required')
 words=np.frombuffer(hashlib.sha256(object_id.encode('ascii')).digest(),dtype='<u4')
 return np.random.SeedSequence([root]+[int(x) for x in words])


def common_fantasies(cloud,noise_covariance,object_id):
 """Exactly one 256-stream; every budget/order uses its same prefixes."""
 if cloud['order']!=128:raise ValueError('Fantasy outcomes must come from the highest-order cloud')
 L=np.linalg.cholesky(np.asarray(noise_covariance,dtype=float));A=cloud['action_predictions_m'];rng=np.random.default_rng(_seed(FANTASY_ROOT_SEED,object_id));anc=rng.choice(len(cloud['weights']),size=FANTASY_COUNTS[-1],p=cloud['weights']);noise=rng.standard_normal((FANTASY_COUNTS[-1],A.shape[2]))@L.T
 return dict(ancestors=anc,noise_m=noise,observations_m=A[:,anc,:]+noise[None,:,:],root_seed=FANTASY_ROOT_SEED,source_cloud_order=128)


def fantasy_utilities(weights,log_weights,predictions,observations,noise_covariance,targets):
 """Exact finite-cloud Bayes update; KL and positive two-pass target risk.

 Callers may pass any target embedding whose Euclidean norm equals target-field
 distance. The public wrapper uses all six polynomial row-span directions,
 without rank truncation, for bounded memory. Divide trace by field size there.
 """
 w=np.asarray(weights,dtype=float);lw=np.asarray(log_weights,dtype=float);P=np.asarray(predictions,dtype=float);Y=np.asarray(observations,dtype=float);T=np.asarray(targets,dtype=float);C=np.asarray(noise_covariance,dtype=float)
 if w.ndim!=1 or lw.shape!=w.shape or P.ndim!=2 or Y.ndim!=2 or P.shape[0]!=len(w) or P.shape[1]!=Y.shape[1] or T.ndim!=2 or len(T)!=len(w) or C.shape!=(P.shape[1],P.shape[1]) or not all(np.isfinite(a).all() for a in (w,lw,P,Y,T,C)) or np.any(w<0) or abs(w.sum()-1)>1e-12:raise ValueError('Finite normalized cloud, supplied fantasy vectors, fixed SPD noise and target embedding required')
 if not np.allclose(np.exp(lw),w,rtol=2e-14,atol=0):raise ValueError('Weights and retained log weights disagree')
 L=np.linalg.cholesky(C);anchor=P[0];p=np.linalg.solve(L,(P-anchor).T).T;y=np.linalg.solve(L,(Y-anchor).T).T;information=[];risk=[]
 for first in range(0,len(Y),8):
  ll=-.5*np.sum((y[first:first+8,None,:]-p[None,:,:])**2,axis=2);le=logsumexp(ll+lw[None,:],axis=1);W=np.exp(ll+lw[None,:]-le[:,None]);kl=np.sum(W*(ll-le[:,None]),axis=1)
  if np.any(kl < -1e-11):raise FloatingPointError('Negative finite-cloud KL beyond roundoff')
  mean=W@(T-T[0])+T[0];variance=np.zeros(len(W))
  for start in range(0,len(w),512):
   delta=T[None,start:start+512,:]-mean[:,None,:];variance+=np.sum(W[:,start:start+512]*np.sum(delta*delta,axis=2),axis=1)
  information.extend(kl);risk.extend(variance)
 return dict(information_nats=np.asarray(information),target_variance_trace=np.asarray(risk))


def _target_embedding(targets,cloud):
 """No rank truncation: complete at-most-six polynomial row span."""
 C=np.concatenate([targets[c] for c in targets],axis=0);Q,_=np.linalg.qr(C.T,mode='reduced');T=cloud['targets_m'];Z=(T-T[0])@Q
 residual=(T-T[0])-Z@Q.T
 if np.max(abs(residual),initial=0)>1e-13*max(1.,np.max(abs(T),initial=0)):raise FloatingPointError('Target embedding reconstruction failed')
 return Z,T.shape[1]


def summarize_decision(rows,*,criterion):
 """Sign positive favors up, using paired fantasies, never unpaired SEs."""
 key='information_nats' if criterion=='information' else 'target_mean_coordinate_variance_m2';summaries={};advantages={};raw_choice=[];stable=True
 for count in FANTASY_COUNTS:
  byorder={};advs=[]
  for order in ORDERS:
   pair=np.asarray(rows[order][key])[:,:count];d=pair[1]-pair[0] if criterion=='information' else pair[0]-pair[1];mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(count));a=int(mean>0);raw_choice.append(a);advs.append(mean)
   byorder[str(order)]=dict(action_scores=pair.mean(axis=1),paired_up_advantage=mean,paired_advantage_MCSE=se,raw_argbest_action_index=a,MC_count=count)
   stable &= abs(mean)>DECISION_RULE['paired_MCSE_multiplier']*se and mean!=0
  discrepancy=max(advs)-min(advs);stable &= abs(advs[-1])>DECISION_RULE['cloud_advantage_discrepancy_multiplier']*discrepancy
  scores=np.array([byorder[str(o)]['action_scores'] for o in ORDERS]);score_discrepancy=np.max(scores,axis=0)-np.min(scores,axis=0)
  summaries[str(count)]=dict(cloud_orders=byorder,maximum_cloud_advantage_discrepancy=discrepancy,per_action_cloud_score_discrepancy=score_discrepancy);advantages[count]=advs[-1]
 stable &= len(set(raw_choice))==1;selected=raw_choice[-1] if stable else 0
 return dict(selected_action_index=int(selected),selected_force_Fz_N=ACTIONS[selected],stable=bool(stable),fallback_used=not bool(stable),fallback_reason=None if stable else 'Tie, paired-MC uncertainty, cloud-order discrepancy, or budget/order action instability; fixed-down, no extra fantasies',budgets=summaries,final_advantage_change_128_to_256=advantages[256]-advantages[128],scope='Numerical/MC choice stability under stipulated model, not evidence of real action advantage')


def design_static_policy(candidate,public,*,design):
 """Zero-I/O public-only design; external caller seals scores BEFORE acquisition."""
 result=dict(completed=False,numerical_cloud_controls_passed=False,choices_frozen=False,information_role='PUBLIC_PREFIX_STATIC_POLICY_NO_ACTUAL_PROBE_OUTCOMES',failures=[],physical_calibration_established=False,is_RGBD_perception=False,is_temporal_forecast=False,decision_rule=DECISION_RULE.copy(),cloud_orders=ORDERS,fantasy_counts=FANTASY_COUNTS,canonical_target_force_Fz_N=1.,per_policy_measurement_budget_coordinates=27)
 try:
  models,targets=compile_public_models(candidate,public,design=design);result['object_id']=public['object_id'];post=PolynomialPosterior(models);reference=post.target_moments(targets);action=[]
  for force in ACTIONS:
   coefficients={c:np.asarray(candidate.observation_coefficients(c,force,public['qualified_point_ids']),dtype=float) for c in post.categories}
   if any(v.shape!=(3,27) or not np.isfinite(v).all() for v in coefficients.values()):raise ValueError('Finite hypothetical action coefficients required')
   action.append(coefficients)
  clouds={o:positive_cloud(post,targets,action,order=o) for o in ORDERS};checks={str(o):check_cloud(clouds[o],post,reference) for o in ORDERS};result['cloud_checks']=checks
  if not all(r['passed'] for r in checks.values()):raise FloatingPointError('Positive-cloud numerical controls failed BEFORE fantasy design')
  result['numerical_cloud_controls_passed']=True;fantasy=common_fantasies(clouds[128],public['sensor_covariance_m2'],public['object_id']);rows={}
  for order,cloud in clouds.items():
   T,D=_target_embedding(targets,cloud);r=[]
   for a in range(2):
    u=fantasy_utilities(cloud['weights'],cloud['log_weights'],cloud['action_predictions_m'][a],fantasy['observations_m'][a],public['sensor_covariance_m2'],T);r.append(u)
   rows[order]=dict(information_nats=np.asarray([u['information_nats'] for u in r]),target_mean_coordinate_variance_m2=np.asarray([u['target_variance_trace']/D for u in r]))
  random=int(np.random.default_rng(_seed(RANDOM_ROOT_SEED,public['object_id'])).integers(2));result.update(completed=True,information=summarize_decision(rows,criterion='information'),target_risk=summarize_decision(rows,criterion='risk'),fixed_down=dict(action_index=0,force_Fz_N=-2.),fixed_up=dict(action_index=1,force_Fz_N=2.),random=dict(action_index=random,force_Fz_N=ACTIONS[random],root_seed=RANDOM_ROOT_SEED),fantasy_stream=dict(source_cloud_order=128,root_seed=FANTASY_ROOT_SEED,opaque_ID_seed='SHA256 ASCII ID -> little-endian uint32 words; SeedSequence(root,*words)',same_ancestors_and_noise_both_actions=True,same_fantasy_outcomes_all_cloud_orders=True,budget_prefixes=True,maximum_count=256,ancestor_indices=fantasy['ancestors'],pseudo_noise_sha256=hashlib.sha256(np.ascontiguousarray(fantasy['noise_m']).tobytes()).hexdigest()),prior_canonical_mean_coordinate_variance_m2=float(np.mean(reference['variance_m2'])),fantasy_statistics=rows,scope='Ideal static tracked-point hypotheses only. No observations outside acquired prefix. Canonical target excludes sensor noise. Caller must freeze choices before actual probe read; no data admission or population calibration.')
 except Exception as error:result['failures'].append(repr(error))
 return result
