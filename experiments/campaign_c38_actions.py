"""Reset-study action controls with an explicit finite Gaussian working model.

Selection has no actual observation/truth argument. Each reset probe observes
one object from a common rest state; every policy subsequently predicts the
SAME canonical force history from rest. Views observe a common current state.
This utility does not establish physical posterior convergence or model fit.
Correlated GP discrepancy must instead use the existing joint-latent update;
independent Gaussian sensor factors here must not be reused as that update.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.campaign_point_active_views import TargetVariance
from experiments.campaign_c38_study import array_digest


def _weights(weights,n):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(n,) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Finite nonnegative base weights required')
    return w/w.sum()


def _model(predictions,covariance,weights):
    p=np.asarray(predictions,dtype=np.float64);c=np.asarray(covariance,dtype=np.float64)
    if p.ndim!=3 or min(p.shape)<1 or not np.isfinite(p).all() or c.shape!=(p.shape[2],p.shape[2]) or not np.isfinite(c).all() or not np.allclose(c,c.T,atol=1e-15,rtol=1e-12):raise ValueError('Finite [actions,particles,sensors] predictions and shared symmetric covariance required')
    try:L=np.linalg.cholesky(c)
    except np.linalg.LinAlgError as e:raise ValueError('Positive definite sensor covariance required') from e
    return p,c,L,_weights(weights,p.shape[1])


def validate_reset_actions(actions,covariance):
    """Equal acquisition budgets and common targets, not equal force directions.

    Energy/force cap is separately fixed by the forward-domain protocol. This
    metadata check cannot certify whether a physical reset really occurred.
    """
    if not isinstance(actions,list) or not actions:raise ValueError('Explicit action records required')
    keys={'id','kind','duration_s','frame_count','sensor_dimension','reset_to_common_rest','common_state_sha256','target_history_sha256','force_domain_sha256'}
    ids=[];budgets=[];states=[];targets=[];domains=[]
    for a in actions:
        if set(a)!=keys or a['kind'] not in ('view','reset_probe') or type(a['frame_count']) is not int or a['frame_count']<1 or type(a['sensor_dimension']) is not int or a['sensor_dimension']!=len(covariance) or not isinstance(a['id'],str) or not a['id'] or not np.isfinite(a['duration_s']) or a['duration_s']<=0:raise ValueError('Exact acquisition metadata and finite positive matched budgets required')
        if type(a['reset_to_common_rest']) is not bool or (a['kind']=='reset_probe' and not a['reset_to_common_rest']):raise ValueError('Probe must reset to common rest before acquisition')
        for key in ('common_state_sha256','target_history_sha256','force_domain_sha256'):
            if not isinstance(a[key],str) or len(a[key])!=64 or any(x not in '0123456789abcdef' for x in a[key]):raise ValueError('Common state, target history and force domain must be hash-bound')
        ids.append(a['id']);budgets.append((a['duration_s'],a['frame_count'],a['sensor_dimension']));states.append(a['common_state_sha256']);targets.append(a['target_history_sha256']);domains.append(a['force_domain_sha256'])
    if len(set(ids))!=len(ids) or len(set(budgets))!=1 or len(set(states))!=1 or len(set(targets))!=1 or len(set(domains))!=1:raise ValueError('Unique actions, identical budgets, initial state, canonical target and force domain required')


def _posterior(prediction,observations,L,w):
    # Subtract observations before whitening; never build particle Gram N*N.
    delta=np.asarray(observations,dtype=float)[:,None]-prediction[None]
    residual=np.linalg.solve(L,delta.reshape(-1,delta.shape[-1]).T).T.reshape(delta.shape)
    logw=np.full(len(w),-np.inf);positive=w>0;logw[positive]=np.log(w[positive])
    value=logw[None]-.5*np.square(residual).sum(-1)
    value-=value.max(1,keepdims=True);post=np.exp(value);post/=post.sum(1,keepdims=True)
    return post


def selection_fingerprint(targets,predictions,covariance,weights,actions):
    arrays={'targets':array_digest(targets,np.float64),'predictions':array_digest(predictions,np.float64),'covariance':array_digest(covariance,np.float64),'base_weights':array_digest(weights,np.float64)}
    arrays['actions']=hashlib.sha256(json.dumps(actions,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return arrays


def select_reset_actions(targets,predictions,covariance,weights,actions,*,object_index,fixed_index,fantasies=64,fantasy_seed=2026122201,random_seed=2026122202):
    """Compare IG and prediction-risk using common fantasy ancestors/noise.

    `targets` are common canonical *displacements*, not policy-specific probe
    trajectories. Fixed index must have been frozen using calibration objects.
    This finite-particle criterion is a synthetic control until independently
    validated continuous posterior and qualified forward data are admitted.
    """
    p,c,L,w=_model(predictions,covariance,weights);validate_reset_actions(actions,c)
    if len(actions)!=len(p) or type(fantasies) is not int or fantasies<2 or type(object_index) is not int or object_index<0 or type(fixed_index) is not int or not 0<=fixed_index<len(p):raise ValueError('Exact object, fixed policy index and fantasy budget required')
    moments=TargetVariance(targets,w)
    if moments.count!=p.shape[1]:raise ValueError('Shared hypothesis ordering for actions and canonical targets required')
    rng=np.random.default_rng(fantasy_seed);ancestors=rng.choice(len(w),size=fantasies,p=w);noise=rng.standard_normal((fantasies,p.shape[2]))@L.T
    risks=[];information=[];effective=[];positive=w>0;logprior=np.log(w[positive])
    for prediction in p:
        posterior=_posterior(prediction,prediction[ancestors]+noise,L,w)
        risks.append(moments.risks(posterior))
        logpost=np.log(np.maximum(posterior[:,positive],np.finfo(float).tiny))
        information.append(np.sum(posterior[:,positive]*(logpost-logprior),axis=1))
        effective.append(1/np.square(posterior).sum(1))
    risks=np.asarray(risks);information=np.asarray(information)
    risk=risks.mean(1);ig=information.mean(1)
    random=int(np.random.default_rng(np.random.SeedSequence([random_seed,object_index])).integers(len(p)))
    return {'schema':'c38-reset-choice-v01','completed':True,'claim':'Working-model design control; not validated physical uncertainty or intelligent probing',
        'object_index':object_index,'actions':actions,'policies':{'fixed':fixed_index,'random':random,'information_gain':int(np.argmax(ig)),'predictive_risk':int(np.argmin(risk))},
        'expected_target_variance_m2':risk.tolist(),'expected_information_gain_nats':ig.tolist(),
        'target_variance_mc_se_m2':(risks.std(1,ddof=1)/np.sqrt(fantasies)).tolist(),'information_gain_mc_se_nats':(information.std(1,ddof=1)/np.sqrt(fantasies)).tolist(),
        'risk_advantage_mc_se_m2':((risks-risks[np.argmin(risk)]).std(1,ddof=1)/np.sqrt(fantasies)).tolist(),
        'information_advantage_mc_se_nats':((information[np.argmax(ig)]-information).std(1,ddof=1)/np.sqrt(fantasies)).tolist(),
        'expected_particle_ESS':np.mean(effective,axis=1).tolist(),'prior_target_variance_m2':float(moments.risks(w[None])[0]),
        'fantasies':fantasies,'fantasy_seed':fantasy_seed,'random_seed':random_seed,'common_random_numbers':True,
        'canonical_target_common_to_all_policies':True,'physical_reset_assumption':True,
        'input_sha256':selection_fingerprint(targets,p,c,w,actions)}


def freeze_choices(path,selection,source_sha256):
    """Exclusive durable choice record; caller supplies source closure hashes."""
    if selection.get('schema')!='c38-reset-choice-v01' or selection.get('completed') is not True or not source_sha256:raise ValueError('Complete outcome-free selection and source pins required')
    for source,digest in source_sha256.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('Changed selection source')
    record={**selection,'source_sha256':source_sha256,'frozen_before_actual_action_observation':True}
    data=json.dumps(record,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n';path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()


def update_from_frozen(path,digest,policy,predictions,covariance,weights,observation):
    """ONE new action observation, applied to the unchanged base posterior.

    Policies choosing the same action must receive the same supplied observation.
    Reusing this function sequentially on already updated weights is rejected by
    the frozen base-weight hash, preventing repeated early-data likelihoods.
    """
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Choice record hash mismatch')
    record=json.loads(raw)
    if record.get('frozen_before_actual_action_observation') is not True or policy not in record['policies']:raise ValueError('Frozen declared policy required')
    for source,expected in record['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise ValueError('Changed selection source')
    p,c,L,w=_model(predictions,covariance,weights)
    for name,value in (('predictions',p),('covariance',c),('base_weights',w)):
        if array_digest(value,np.float64)!=record['input_sha256'][name]:raise ValueError('Changed action model or base posterior')
    y=np.asarray(observation,dtype=float)
    if y.shape!=(p.shape[2],) or not np.isfinite(y).all():raise ValueError('One finite acquired action observation required')
    index=record['policies'][policy];post=_posterior(p[index],y[None],L,w)[0]
    return post,{'action_index':index,'effective_sample_size':float(1/np.square(post).sum()),'choice_sha256':digest,'all_particles_retained':True,'resampled':False}
