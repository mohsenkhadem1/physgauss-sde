"""Continuous bounded-parameter SMC with analytically marginalized load likelihoods.

The prior is uniform in the declared theta coordinates (log coordinates where
specified), not uniform in transformed physical moduli. This numerical particle
approximation is not an exact continuous posterior. Independent seed/count
comparisons are required; post-resampling ESS is not a convergence certificate.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class SMCConfig:
    particles: int = 1024
    conditional_ess_fraction: float = .8
    moves: int = 4
    initial_scale: float = .6
    minimum_scale: float = .05
    maximum_scale: float = 2.
    target_acceptance: float = .3
    adaptation_rate: float = .5
    minimum_unit_sd: float = .01
    maximum_stages: int = 150
    batch_size: int = 128

    def validate(self):
        if self.particles < 2 or self.moves < 1 or self.batch_size < 1 or self.maximum_stages < 1:
            raise ValueError('Positive counts and at least two particles required')
        if not 0 < self.conditional_ess_fraction < 1 or not 0 < self.target_acceptance < 1:
            raise ValueError('ESS and acceptance fractions must lie in (0,1)')
        if not 0 < self.minimum_scale <= self.initial_scale <= self.maximum_scale:
            raise ValueError('Invalid proposal scale bounds')
        if not 0 < self.minimum_unit_sd <= .5 or not 0 <= self.adaptation_rate <= 1:
            raise ValueError('Invalid bounded adaptation parameters')
        return self


def reflect_unit(x):
    """Fold any real value into [0,1]; diagonal Gaussian proposals stay symmetric."""
    return 1-np.abs(np.mod(np.asarray(x), 2)-1)


def systematic_resample(weights, rng):
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid resampling weights')
    c=np.cumsum(w/w.sum());c[-1]=1.
    return np.searchsorted(c,(rng.random()+np.arange(len(w)))/len(w),side='right')


def conditional_ess(log_likelihood, weights, delta):
    """N*(E_w incremental_weight)^2/E_w incremental_weight^2."""
    ll=np.asarray(log_likelihood);w=np.asarray(weights)
    if delta == 0:return float(len(w))
    active=(w>0)&np.isfinite(ll)
    if not active.any():return 0.
    lw=np.log(w[active]);a=delta*ll[active]
    return float(len(w)*np.exp(2*logsumexp(lw+a)-logsumexp(lw+2*a)))


def next_temperature(beta,ll,weights,fraction):
    target=len(weights)*fraction
    if conditional_ess(ll,weights,1-beta)>=target:return 1.
    low,high=0.,1-beta
    for _ in range(60):
        mid=(low+high)/2
        if conditional_ess(ll,weights,mid)>=target:low=mid
        else:high=mid
    result=beta+low
    if result<=beta:raise RuntimeError('Temperature stalled; inspect likelihood dynamic range')
    return result


@dataclass
class SMCResult:
    theta: np.ndarray
    unit_particles: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    temperature: float
    log_evidence: float
    stages: list
    evaluation_count: int
    invalid_evaluation_count: int
    seed: int
    target_id: str
    bounds: np.ndarray
    proposal_scale: float
    rng_state: dict
    config: SMCConfig

    @property
    def complete(self):return self.temperature==1.

    def moments(self):
        mean=self.weights@self.theta
        centered=self.theta-mean
        covariance=np.einsum('n,ni,nj->ij',self.weights,centered,centered)
        return mean,covariance

    def force_posterior(self,conditional_callback,batch_size=None):
        """Attach exact p(F|theta,y) moments to continuous posterior particles.

        callback(theta_batch) returns (means[batch,force], covariances[batch,f,f]).
        It must use the SAME data/prior/likelihood as the SMC likelihood callback.
        """
        from .inference import Posterior
        if not self.complete:raise ValueError('Incomplete annealing is not the posterior')
        means=[];covariances=[];batch_size=batch_size or self.config.batch_size
        for i in range(0,len(self.theta),batch_size):
            m,c=conditional_callback(self.theta[i:i+batch_size])
            means.append(np.asarray(m));covariances.append(np.asarray(c))
        m=np.concatenate(means);c=np.concatenate(covariances)
        if len(m)!=len(self.weights) or c.shape!=(len(m),m.shape[1],m.shape[1]):
            raise ValueError('Invalid conditional force dimensions')
        if not np.isfinite(m).all() or not np.isfinite(c).all():
            raise ValueError('Invalid force conditional; no replacement allowed')
        return Posterior(self.weights.copy(),m,c,self.log_evidence)

    def report(self):
        mean,cov=self.moments()
        return dict(complete=self.complete,temperature=self.temperature,
            log_evidence=self.log_evidence,parameter_mean=mean.tolist(),
            parameter_covariance=cov.tolist(),particles=len(self.weights),
            unique_particles=int(len(np.unique(self.unit_particles,axis=0))),
            weight_ess=float(1/(self.weights@self.weights)),stages=self.stages,
            evaluation_count=self.evaluation_count,invalid_evaluation_count=self.invalid_evaluation_count,
            seed=self.seed,target_id=self.target_id,bounds=self.bounds.tolist(),config=asdict(self.config),
            scope='Continuous uniform-coordinate prior; finite-particle SMC approximation. Weight ESS after resampling is not convergence.',
            invalid_policy='Nonfinite likelihoods have zero mass or rejected MH proposals; never redrawn. Callback exceptions propagate.')

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        meta=self.report();meta.update(proposal_scale=self.proposal_scale,rng_state=self.rng_state)
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as f:
            np.savez(f,theta=self.theta,unit_particles=self.unit_particles,weights=self.weights,
                     log_likelihood=self.log_likelihood,metadata=np.array(json.dumps(meta)))
        temporary.replace(path)
        report=path.with_suffix(path.suffix+'.json');tmp=report.with_name(report.name+'.tmp')
        tmp.write_text(json.dumps(meta,indent=2));tmp.replace(report)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=json.loads(str(data['metadata']))
            return cls(data['theta'].copy(),data['unit_particles'].copy(),data['weights'].copy(),
                data['log_likelihood'].copy(),m['temperature'],m['log_evidence'],m['stages'],
                m['evaluation_count'],m['invalid_evaluation_count'],m['seed'],m['target_id'],
                np.asarray(m['bounds']),m['proposal_scale'],m['rng_state'],SMCConfig(**m['config']))


def run_smc(log_likelihood, *, bounds=None, config=SMCConfig(), seed=0,
            target_id='', checkpoint=None, resume=False, stop_after_stages=None):
    """Run independent-RNG annealed SMC; callback accepts physical theta batches.

    Return log likelihood INCLUDING observation density normalizers for meaningful
    evidence. Return -inf for each inadmissible physical state. Exceptions are
    not swallowed or retried. Every MH step uses one proposal per particle.
    Checkpoints are written only at completed stage boundaries. Set target_id to
    a frozen protocol/data fingerprint; resume verifies it, bounds/config/seed.
    stop_after_stages is an absolute stage count for checkpoint testing/budgets.
    """
    config.validate()
    if bounds is None:
        from .bank import BOUNDS
        bounds=BOUNDS
    bounds=np.asarray(bounds,dtype=np.float64)
    if bounds.ndim!=2 or bounds.shape[1]!=2 or not np.isfinite(bounds).all() or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError('Expected finite [dimension,lower/upper] bounds')
    n=config.particles;dimension=len(bounds);width=bounds[:,1]-bounds[:,0]
    rng=np.random.default_rng(seed);evaluated=0;invalid=0

    def evaluate(unit):
        nonlocal evaluated,invalid
        outputs=[]
        for start in range(0,len(unit),config.batch_size):
            theta=bounds[:,0]+unit[start:start+config.batch_size]*width
            ll=np.asarray(log_likelihood(theta),dtype=np.float64)
            if ll.shape!=(len(theta),):raise ValueError('Callback must return one log likelihood per particle')
            bad=~np.isfinite(ll);evaluated+=len(theta);invalid+=int(bad.sum())
            outputs.append(np.where(bad,-np.inf,ll))
        return np.concatenate(outputs)

    def result():
        return SMCResult(bounds[:,0]+unit*width,unit.copy(),weights.copy(),ll.copy(),
            beta,logz,list(history),evaluated,invalid,seed,target_id,bounds.copy(),scale,
            rng.bit_generator.state,config)

    if resume:
        if checkpoint is None or not target_id:raise ValueError('Resume requires checkpoint and target fingerprint')
        saved=SMCResult.load(checkpoint)
        if saved.config!=config or saved.seed!=seed or saved.target_id!=target_id or not np.array_equal(saved.bounds,bounds):
            raise ValueError('Checkpoint protocol/bounds/seed mismatch')
        unit=saved.unit_particles.copy();weights=saved.weights.copy();ll=saved.log_likelihood.copy()
        beta=saved.temperature;logz=saved.log_evidence;history=list(saved.stages)
        evaluated=saved.evaluation_count;invalid=saved.invalid_evaluation_count;scale=saved.proposal_scale
        rng.bit_generator.state=saved.rng_state
    else:
        unit=rng.random((n,dimension));ll=evaluate(unit);weights=np.full(n,1/n)
        beta=0.;logz=0.;history=[];scale=config.initial_scale
        valid=np.isfinite(ll)
        if not valid.any():raise RuntimeError('All initial prior particles have zero likelihood; no replacement attempted')
        if not valid.all():
            # Explicit support-conditioning importance step. No rejection sampling
            # redraws: normal SMC resampling retains valid particles and its mass.
            logz=float(np.log(valid.mean()));weights=valid.astype(float)/valid.sum()
            indices=systematic_resample(weights,rng);unit=unit[indices];ll=ll[indices];weights.fill(1/n)
            history.append(dict(kind='support_conditioning',temperature=0.,prior_valid_fraction=float(valid.mean()),
                                log_evidence_increment=logz,invalid_initial=int((~valid).sum())))
        if checkpoint is not None:result().save(checkpoint)
    while beta<1:
        stage_count=sum(s.get('kind')=='tempering' for s in history)
        if stop_after_stages is not None and stage_count>=stop_after_stages:break
        if stage_count>=config.maximum_stages:
            if checkpoint is not None:result().save(checkpoint)
            raise RuntimeError('Maximum temperature stages reached; incomplete checkpoint retained')
        next_beta=next_temperature(beta,ll,weights,config.conditional_ess_fraction)
        delta=next_beta-beta;increment=delta*ll
        lw=np.log(weights)+increment;normalizer=float(logsumexp(lw));newweights=np.exp(lw-normalizer)
        cess=conditional_ess(ll,weights,delta);ess=float(1/(newweights@newweights))
        logz+=normalizer
        indices=systematic_resample(newweights,rng);unit=unit[indices];ll=ll[indices];weights.fill(1/n)
        # Independent coordinate increments are necessary for reflection symmetry.
        # Adapt only between stages; freeze widths/scale throughout these moves.
        unit_sd=np.maximum(np.std(unit,axis=0,ddof=1),config.minimum_unit_sd)
        proposal_sd=scale*unit_sd
        accepted=0;bad_before=invalid;eval_before=evaluated
        for _ in range(config.moves):
            proposal=reflect_unit(unit+rng.normal(size=unit.shape)*proposal_sd)
            proposed_ll=evaluate(proposal)
            logratio=next_beta*(proposed_ll-ll)
            take=np.log(rng.random(n))<np.minimum(0.,logratio)
            unit[take]=proposal[take];ll[take]=proposed_ll[take];accepted+=int(take.sum())
        rate=accepted/(n*config.moves)
        next_scale=float(np.clip(scale*np.exp(config.adaptation_rate*(rate-config.target_acceptance)),
                                 config.minimum_scale,config.maximum_scale))
        history.append(dict(kind='tempering',stage=stage_count+1,previous_temperature=beta,
            temperature=next_beta,conditional_ess=cess,pre_resample_weight_ess=ess,
            log_evidence_increment=normalizer,acceptance_rate=rate,accepted=accepted,
            proposal_count=evaluated-eval_before,invalid_proposals=invalid-bad_before,
            proposal_scale=scale,next_proposal_scale=next_scale,unit_proposal_sd=proposal_sd.tolist(),
            unique_particles=int(len(np.unique(unit,axis=0)))))
        scale=next_scale;beta=next_beta
        if checkpoint is not None:result().save(checkpoint)
    return result()


class MarginalGaussianLikelihood:
    """Exact Gaussian amplitude elimination conditional on continuous theta.

    mapping(theta) -> H[batch,observation,force] OR dict(H=H, offset=[batch,obs]).
    Observation noise may be scalar variance, diagonal or a shared full matrix.
    The callback owns physics and may batch torch FEM; this layer is NumPy.
    """
    def __init__(self,mapping,observations,load_covariance,noise_covariance):
        self.mapping=mapping;self.y=np.asarray(observations,dtype=np.float64)
        self.K=np.asarray(load_covariance,dtype=np.float64)
        R=np.asarray(noise_covariance,dtype=np.float64)
        if self.y.ndim!=1 or self.K.ndim!=2 or self.K.shape[0]!=self.K.shape[1]:
            raise ValueError('Invalid observation/load dimensions')
        self.R=np.eye(len(self.y))*R if R.ndim==0 else np.diag(R) if R.ndim==1 else R
        if self.R.shape!=(len(self.y),len(self.y)):raise ValueError('Invalid noise dimensions')
        np.linalg.cholesky(self.K);np.linalg.cholesky(self.R)

    def evaluate(self,theta,conditionals=False):
        mapped=self.mapping(theta)
        if isinstance(mapped,dict):H=np.asarray(mapped['H']);offset=mapped.get('offset',0.)
        else:H=np.asarray(mapped);offset=0.
        if H.shape!=(len(theta),len(self.y),len(self.K)):raise ValueError('Invalid forward mapping shape')
        residual=self.y-np.broadcast_to(offset,(len(theta),len(self.y)))
        KH=self.K@H.transpose(0,2,1);C=H@KH+self.R
        L=np.linalg.cholesky(C);white=np.linalg.solve(L,residual[...,None])
        loglike=-.5*((white[...,0]**2).sum(1)+2*np.log(L.diagonal(axis1=-2,axis2=-1)).sum(1)+len(self.y)*np.log(2*np.pi))
        if not conditionals:return loglike
        alpha=np.linalg.solve(L.transpose(0,2,1),white)
        mean=(KH@alpha)[...,0];V=np.linalg.solve(L,KH.transpose(0,2,1))
        cov=self.K-V.transpose(0,2,1)@V
        return mean,.5*(cov+cov.transpose(0,2,1))

    def __call__(self,theta):return self.evaluate(theta)

    def conditional(self,theta):return self.evaluate(theta,True)
