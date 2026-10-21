"""Consistent mixed visibility/range likelihood for actual reset probes.

Covariance and visibility/outlier probabilities must be frozen using reserved
TRAIN controls. This module has no BC labels, recipe paths, or calibration-data
access. Unknown support classes affect predictions, never private noise lookup.
Missing range values are omitted variables, not +50mm Gaussian observations.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CensoredRanges:
    visible: np.ndarray
    residual_m: np.ndarray

    def __post_init__(self):
        v=np.asarray(self.visible,dtype=bool);r=np.asarray(self.residual_m,dtype=float)
        if v.shape!=r.shape or v.ndim not in (1,2):raise ValueError('Equal [D] or [K,D] arrays required')
        if not np.isfinite(r[v]).all():raise ValueError('Visible ranges must be finite')
        # Absent coordinates have a fixed harmless placeholder, NEVER scored.
        object.__setattr__(self,'visible',v.copy())
        object.__setattr__(self,'residual_m',np.where(v,r,0.))


def ranges_from_depth(depth, rest_depth_m):
    depth=np.asarray(depth,dtype=float);rest=np.asarray(rest_depth_m,dtype=float)
    visible=np.isfinite(depth)&(depth>0)
    return CensoredRanges(visible,depth-rest)


def measure_censored_depth(depth_image,query):
    depth=np.asarray(depth_image)
    if depth.ndim!=2:raise ValueError('Expected depth image')
    return ranges_from_depth(depth[query['ys'],query['xs']],query['rest_depth_m'])


def predict_censored_ranges(positions,reference,query,frame_ids=None):
    """[K,T,N,3] direct trajectories -> [K,frames*rays] raw ray observations."""
    try:
        from validate_isaac_dataset import ray_depth
    except ImportError:
        from experiments.validate_isaac_dataset import ray_depth
    q=np.asarray(positions)
    if q.ndim!=4 or q.shape[2:]!=reference['rest_points_m'].shape:raise ValueError('Expected [K,T,N,3]')
    ids=np.arange(1,q.shape[1]) if frame_ids is None else np.asarray(frame_ids,dtype=int)
    depths=np.stack([np.concatenate([ray_depth(p[i],reference['surface_faces'],query['pose'],query['K'],query['xs'],query['ys']) for i in ids]) for p in q])
    return ranges_from_depth(depths,np.tile(query['rest_depth_m'],len(ids)))


def concatenate_observations(observations):
    return CensoredRanges(np.concatenate([o.visible for o in observations],axis=-1),
                          np.concatenate([o.residual_m for o in observations],axis=-1))


def _weights(weights,count):
    w=np.ones(count)/count if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(count,) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid particle weights')
    return w/w.sum()


def _logsumexp(a):
    maximum=np.max(a)
    if not np.isfinite(maximum):return maximum
    return maximum+np.log(np.exp(a-maximum).sum())


class CensoredRangeModel:
    """Proper generative working model with one global conservative covariance.

    For predicted hit mask h and observed v, P(v_i=h_i)=1-epsilon.
    Given observed hits, A={v & h} follows N(mu_A,Sigma_A), while B={v & ~h}
    independently follows broad N(0,background_std²). With optional probability
    eta, visibility instead follows independent fair Bernoulli bits and ALL
    observed ranges follow the broad background independently. This broad
    branch is shared across particles and protects against a whole experiment
    outside the surrogate/visibility model. This is a proper
    mixture, not an unnormalized mismatch penalty. A single eta draw per reset
    sequence prevents pretending each correlated frame is a separate outlier.
    """
    def __init__(self,covariance_m2,visibility_error_probability,
                 background_std_m=.05,outlier_probability=.02,cache_size=128):
        covariance=np.asarray(covariance_m2,dtype=float)
        if covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1] or not len(covariance):raise ValueError('Square covariance required')
        if not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T,rtol=1e-10,atol=1e-20):raise ValueError('Finite symmetric covariance required')
        self.cholesky=np.linalg.cholesky(covariance)
        if not 0<visibility_error_probability<.5:raise ValueError('Visibility error must be strictly between0and.5')
        if background_std_m<=0 or not 0<=outlier_probability<1:raise ValueError('Invalid background mixture')
        self.covariance=covariance.copy();self.visibility_error=float(visibility_error_probability)
        self.background_std=float(background_std_m);self.outlier_probability=float(outlier_probability)
        self.dimension=len(covariance);self.cache_size=int(cache_size);self._cache=OrderedDict()
        # The global ray/frame/common covariance is normally isotropic noise
        # plus a small shared factor. Woodbury keeps changing visibility masks
        # cheap, while preserving the exact dense Gaussian likelihood.
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        self._isotropic_variance=float(eigenvalues[0])
        selected=eigenvalues-self._isotropic_variance>max(self._isotropic_variance*1e-10,1e-30)
        factor=eigenvectors[:,selected]*np.sqrt(eigenvalues[selected]-self._isotropic_variance)
        self._shared_factor=factor if factor.shape[1]<self.dimension/2 else None

    def _check_predictions(self,predictions):
        if predictions.visible.ndim!=2 or predictions.visible.shape[1]!=self.dimension:raise ValueError('Expected [particles,D] predictions matching covariance')

    def _subfactor(self,mask):
        key=np.packbits(mask).tobytes()
        if key in self._cache:
            self._cache.move_to_end(key);return self._cache[key]
        if self._shared_factor is not None:
            factor=self._shared_factor[mask];variance=self._isotropic_variance
            small=np.eye(factor.shape[1])+factor.T@factor/variance
            chol=np.linalg.cholesky(small)
            logdet=int(mask.sum())*np.log(variance)+2*np.log(np.diag(chol)).sum()
            result=('woodbury',chol,logdet,factor,variance)
        else:
            matrix=self.covariance[np.ix_(mask,mask)];chol=np.linalg.cholesky(matrix)
            result=('dense',chol,2*np.log(np.diag(chol)).sum())
        if self.cache_size>0:
            self._cache[key]=result
            if len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        return result

    def log_prob(self,predictions,observation):
        self._check_predictions(predictions)
        if observation.visible.shape!=(self.dimension,):raise ValueError('One [D] observation required')
        v=observation.visible;y=observation.residual_m;h=predictions.visible;mu=predictions.residual_m
        matches=(h==v).sum(-1)
        visibility=matches*np.log1p(-self.visibility_error)+(self.dimension-matches)*np.log(self.visibility_error)
        # Group identical predicted visibility patterns; one covariance factor
        # handles all their differing means. This retains each submatrix logdet.
        patterns,inverse=np.unique(h,axis=0,return_inverse=True)
        hybrid=np.zeros(len(h))
        log2pi=np.log(2*np.pi);bgvar=self.background_std**2
        background=-.5*(np.square(y[v]).sum()/bgvar+v.sum()*(log2pi+np.log(bgvar)))
        for group,pattern in enumerate(patterns):
            indices=np.flatnonzero(inverse==group);a=v&pattern;b=v&~pattern
            value=-.5*(np.square(y[b]).sum()/bgvar+b.sum()*(log2pi+np.log(bgvar)))
            if a.any():
                decomposition=self._subfactor(a);kind,chol,logdet=decomposition[:3]
                difference=y[a][None]-mu[indices][:,a]
                if kind=='woodbury':
                    factor,variance=decomposition[3:]
                    quadratic=np.square(difference).sum(-1)/variance
                    if factor.shape[1]:
                        projection=np.linalg.solve(chol,factor.T@difference.T)
                        quadratic=quadratic-np.square(projection).sum(0)/(variance*variance)
                    quadratic=np.maximum(quadratic,0.)
                else:
                    whitened=np.linalg.solve(chol,difference.T)
                    quadratic=np.square(whitened).sum(0)
                hybrid[indices]=value-.5*(quadratic+a.sum()*log2pi+logdet)
            else:hybrid[indices]=value
        eta=self.outlier_probability
        joint=visibility+hybrid
        return joint if eta==0 else np.logaddexp(np.log1p(-eta)+joint,np.log(eta)-self.dimension*np.log(2)+background)

    def condition(self,predictions,observation,weights=None):
        w=_weights(weights,len(predictions.visible));prior_log=np.full_like(w,-np.inf);positive=w>0;prior_log[positive]=np.log(w[positive])
        likelihood=self.log_prob(predictions,observation);joint=prior_log+likelihood;normalizer=_logsumexp(joint)
        if not np.isfinite(normalizer):raise FloatingPointError('No finite mixed-observation evidence')
        posterior=np.exp(joint-normalizer)
        return posterior,{'effective_sample_size':float(1/np.square(posterior).sum()),
                          'log_evidence':float(normalizer),'observed_hit_count':int(observation.visible.sum()),
                          'log_likelihood_min':float(likelihood.min()),'log_likelihood_max':float(likelihood.max())}

    def fantasies(self,predictions,weights=None,count=256,seed=0):
        self._check_predictions(predictions)
        if count<1:raise ValueError('Positive fantasy count')
        rng=np.random.default_rng(seed);w=_weights(weights,len(predictions.visible))
        indices=rng.choice(len(w),count,p=w);h=predictions.visible[indices]
        visible=h^(rng.random(h.shape)<self.visibility_error)
        noise=rng.standard_normal(h.shape)@self.cholesky.T
        foreground=predictions.residual_m[indices]+noise
        background=rng.normal(0,self.background_std,h.shape)
        residual=np.where(h,foreground,background)
        is_outlier=rng.random(count)<self.outlier_probability
        residual[is_outlier]=background[is_outlier]
        visible[is_outlier]=rng.random((is_outlier.sum(),self.dimension))<.5
        return CensoredRanges(visible,residual),{'generating_particle_indices':indices,'outlier_branch':is_outlier}


def centered_target_gram(target_features,weights=None):
    """Particle Gram independent of common target translations/origins."""
    targets=np.asarray(target_features,dtype=float)
    if targets.ndim<2 or not np.isfinite(targets).all():raise ValueError('Finite particle targets required')
    targets=targets.reshape(len(targets),-1)
    if not targets.shape[1]:raise ValueError('Nonempty target dimensions required')
    w=_weights(weights,len(targets))
    # Subtract a reference first so identical huge constants become EXACT zero.
    centered=targets-targets[:1]
    centered=centered-w@centered
    return centered@centered.T/centered.shape[1]


def gram_target_risk(gram,weights=None):
    gram=np.asarray(gram,dtype=float)
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or not np.isfinite(gram).all():raise ValueError('Finite square Gram required')
    w=_weights(weights,len(gram))
    return float(max(0.,w@np.diag(gram)-w@gram@w))


def score_censored_actions(target_features,candidates,model,weights=None,fantasies=256,seed=0):
    """Monte Carlo target-risk design using SAME model for fantasies/conditioning.

    Target features are predicted trajectories, never actual target observations.
    Centered particle Gram avoids trajectory means in each fantasy. Identical
    RNG seed shares ancestors, visibility uniforms, standardized range noise and
    outlier draws across actions; paired score uncertainty is reported.
    """
    targets=np.asarray(target_features,dtype=float)
    if targets.ndim<2 or not np.isfinite(targets).all() or not candidates:raise ValueError('Finite particle targets/candidates required')
    if fantasies<2:raise ValueError('At least two fantasies required for score uncertainty')
    w=_weights(weights,len(targets));gram=centered_target_gram(targets,w);norms=np.diag(gram)
    prior=gram_target_risk(gram,w);scores=[];all_risks=[]
    for action,prediction in enumerate(candidates):
        if len(prediction.visible)!=len(targets):raise ValueError('Particle counts differ')
        observations,_=model.fantasies(prediction,w,fantasies,seed)
        posteriors=[];information=[];ess=[]
        for sample in range(fantasies):
            observation=CensoredRanges(observations.visible[sample],observations.residual_m[sample])
            posterior,diagnostic=model.condition(prediction,observation,w)
            posteriors.append(posterior)
            positive=posterior>0
            information.append(float(np.sum(posterior[positive]*np.log(posterior[positive]/w[positive]))))
            ess.append(diagnostic['effective_sample_size'])
        posterior=np.asarray(posteriors)
        risks=np.maximum(0.,posterior@norms-np.einsum('fi,ij,fj->f',posterior,gram,posterior,optimize=True))
        all_risks.append(risks)
        scores.append({'action_index':action,'expected_target_risk':float(np.mean(risks)),
                       'target_risk_mc_se':float(np.std(risks,ddof=1)/np.sqrt(fantasies)),
                       'expected_information_nats':float(np.mean(information)),
                       'expected_effective_sample_size':float(np.mean(ess))})
    chosen=int(np.argmin([s['expected_target_risk'] for s in scores]))
    risks=np.asarray(all_risks);paired=risks-risks[chosen]
    return {'selected_index':chosen,'prior_target_risk':prior,'actions':scores,'fantasies':fantasies,'seed':seed,
            'expected_risk_reduction':(prior-risks.mean(-1)).tolist(),
            'selected_advantage_standard_error':(paired.std(-1,ddof=1)/np.sqrt(fantasies)).tolist(),
            'common_random_numbers':True,
            'observation_model':'Bernoulli visibility + subcovariance Gaussian valid ranges + whole-experiment broad background mixture'}
