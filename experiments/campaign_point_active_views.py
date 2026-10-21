"""Censored-sensor view design on a fixed empirical point/material posterior.

Selection consumes predicted future targets and sensor predictions only.
Actual observations enter the separate update function after choice freezes.
No geometry, response files, truth, fitting, resampling or simulator access.
"""
from __future__ import annotations
import numpy as np
try:
    from censored_range_observation import CensoredRanges,CensoredRangeModel
except ImportError:
    from experiments.censored_range_observation import CensoredRanges,CensoredRangeModel

METHOD_PROTOCOL_SHA256='ecbc02153336b36ddef33f0c6c01370b012f3ddbeb1d76bd5c71e15145dc3bff'
FANTASIES=32
FANTASY_SEED=2026116000
PARTICLES=2048
TARGET_DIMENSION=30*455*3


def _weights(weights,count):
    w=np.full(count,1/count) if weights is None else np.asarray(weights,dtype=np.float64)
    if w.shape!=(count,) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('Valid particle weights required')
    return w/w.sum()


class TargetVariance:
    """Centered weighted first/second moments; never builds particle Gram NxN.

    Each risk is average coordinate variance (m² for meter targets). Subtract
    one particle and the base mean before moments to avoid common rest offsets.
    Optional torch device uses explicitfloat64 chunked matrix products; it does
    not change particles, features, sensorlikelihood or the score definition.
    """
    def __init__(self,targets,weights=None,*,device='cpu',feature_chunk_size=4096):
        x=np.asarray(targets,dtype=np.float64)
        if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('Finite nonempty [particles,targetcoordinates] required')
        if type(feature_chunk_size) is not int or feature_chunk_size<1:raise ValueError('Positive integer feature chunk size')
        self.weights=_weights(weights,len(x));self.count=len(x);self.features=x.shape[1];self.chunk=feature_chunk_size
        self.centered=x-x[:1];self.centered-=self.weights@self.centered
        self.norms=np.zeros(self.count)
        for first in range(0,self.features,self.chunk):
            block=self.centered[:,first:first+self.chunk]
            self.norms+=np.einsum('ij,ij->i',block,block)/self.features
        self.device=str(device);self._tensor=None
        if self.device!='cpu':
            import torch
            requested=torch.device(self.device)
            if requested.type!='cuda':raise ValueError('Optional acceleration accepts an explicitCUDAdevice only')
            self._tensor=torch.as_tensor(self.centered,dtype=torch.float64,device=requested)

    def risks(self,posterior_weights):
        w=np.asarray(posterior_weights,dtype=np.float64)
        if w.ndim!=2 or w.shape[1]!=self.count or not len(w) or not np.isfinite(w).all() or (w<0).any() or np.any(w.sum(1)<=0):raise ValueError('Finite positive-mass [queries,particles] weights required')
        w=w/w.sum(1,keepdims=True);second=w@self.norms;mean_square=np.zeros(len(w))
        if self._tensor is None:
            for first in range(0,self.features,self.chunk):
                mean=w@self.centered[:,first:first+self.chunk]
                mean_square+=np.einsum('ij,ij->i',mean,mean)/self.features
        else:
            import torch
            with torch.no_grad():
                tw=torch.as_tensor(w,dtype=torch.float64,device=self._tensor.device)
                total=torch.zeros(len(w),dtype=torch.float64,device=self._tensor.device)
                for first in range(0,self.features,self.chunk):
                    mean=tw@self._tensor[:,first:first+self.chunk]
                    total+=torch.square(mean).sum(1)/self.features
                mean_square=total.cpu().numpy()
        variance=second-mean_square
        error_bound=64*np.finfo(float).eps*(np.abs(second)+np.abs(mean_square)+np.finfo(float).tiny)
        if np.any(variance < -error_bound):raise FloatingPointError('Target variance negative beyond binary64 moment roundoff')
        return np.maximum(variance,0.)


def score_censored_views(target_features,candidates,model,*,weights=None,fantasies=32,seed=FANTASY_SEED,device='cpu',feature_chunk_size=4096):
    """Generic synthetic-testable scorer; noactual observation argument exists."""
    if not isinstance(model,CensoredRangeModel) or not candidates:raise ValueError('Proper censoredrange model and predicted candidates required')
    if type(fantasies) is not int or fantasies<2 or type(seed) is not int:raise ValueError('Atleast2fantasies and integerseed required')
    moments=TargetVariance(target_features,weights,device=device,feature_chunk_size=feature_chunk_size);w=moments.weights
    for p in candidates:
        if not isinstance(p,CensoredRanges):raise TypeError('Predicted CensoredRanges required')
        model._check_predictions(p)
        if len(p.visible)!=moments.count:raise ValueError('Shared particleorder/count required')
    posterior_rows=[];ess=[]
    for prediction in candidates:
        observations,_=model.fantasies(prediction,w,count=fantasies,seed=seed)
        rows=[];action_ess=[]
        for f in range(fantasies):
            observation=CensoredRanges(observations.visible[f],observations.residual_m[f])
            posterior,diagnostic=model.condition(prediction,observation,w)
            rows.append(posterior);action_ess.append(diagnostic['effective_sample_size'])
        posterior_rows.extend(rows);ess.append(action_ess)
    risks=moments.risks(np.asarray(posterior_rows)).reshape(len(candidates),fantasies)
    expected=risks.mean(1);selected=int(np.argmin(expected));paired=risks-risks[selected]
    prior=float(moments.risks(w[None])[0])
    return {'selected_index':selected,'prior_target_variance_m2':prior,'expected_target_variance_m2':expected.tolist(),
        'expected_variance_reduction_m2':(prior-expected).tolist(),
        'target_variance_mc_se_m2':(risks.std(1,ddof=1)/np.sqrt(fantasies)).tolist(),
        'selected_advantage_mc_se_m2':(paired.std(1,ddof=1)/np.sqrt(fantasies)).tolist(),
        'expected_effective_sample_size':np.mean(ess,1).tolist(),'fantasies':fantasies,'seed':seed,
        'particle_count':moments.count,'target_coordinate_count':moments.features,'common_fantasy_random_numbers':True,
        'selection_inputs':'Predictedtargets,candidatecensoredranges and frozen proper sensormodel only; noactualmeasurements/truth',
        'risk_method':'Chunked centered first/secondmoments; no particle-by-particle Gram/kernel',
        'risk_device':moments.device,'tie_break':'Firstminimum (lowestindex) on exact floating-point scoreties',
        'claim':'MonteCarlo model-based design criterion, not evidence of true expected predictiongain or intelligent physicalprobing'}


def _check_frozen_sensor(model):
    if (not isinstance(model,CensoredRangeModel) or model.dimension!=36
        or not np.allclose(model.covariance,np.eye(36)*(.0002**2),rtol=1e-14,atol=0)
        or model.visibility_error!=.005 or model.background_std!=.05 or model.outlier_probability!=0):
        raise ValueError('Frozen36ray sensor (0.2mm,visibility.005,background.05m,outlier0) required')


def select_point_active_view(target_features,candidates,model,*,device='cpu'):
    """Exact declared4-view/2048particle/32fantasy futureA method."""
    _check_frozen_sensor(model)
    x=np.asarray(target_features)
    if x.shape!=(PARTICLES,TARGET_DIMENSION) or len(candidates)!=4:raise ValueError('Declared2048particles,30*455*3targets and4views required')
    result=score_censored_views(x,candidates,model,fantasies=FANTASIES,seed=FANTASY_SEED,device=device)
    result.update(method_protocol_sha256=METHOD_PROTOCOL_SHA256,equal_initial_empirical_weights=True,
        views={'frame_index':24,'offsets_azimuth_elevation_deg':[[-60,10],[-20,10],[20,10],[60,10]],'equalcost':True},
        target='FutureAframes31..60,all455nodesXYZ',virtual_views_only=True,travelplanning_or_mechanical_backaction=False)
    return result


def update_after_view(prediction,actual_observation,model):
    """Separate selected-view observation update, retaining all2048 particles.

    Pass exactly one newly acquired view after actionchoice freezes. This
    function does not reapply earlyAobservations or resample/replace particles.
    """
    _check_frozen_sensor(model)
    if prediction.visible.shape!=(PARTICLES,36):raise ValueError('Declared2048particle/36ray view required')
    posterior,diagnostic=model.condition(prediction,actual_observation,np.full(PARTICLES,1/PARTICLES))
    posterior=_weights(posterior,PARTICLES)
    diagnostic['effective_sample_size']=float(1/np.square(posterior).sum())
    return posterior,{**diagnostic,'particle_count':PARTICLES,'all_particle_entries_retained':True,'resampled_or_replaced':False,'information_role':'One suppliedactualview observation AFTERchoicefreeze'}
