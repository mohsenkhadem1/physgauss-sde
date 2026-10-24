"""Prospective force generalization and probing on the unchanged finite reference.

No I/O, posterior sampling, truth queries, training or automatic production run.
All force clocks remain exactly 60 implicit-Euler intervals of length 1/120s.
"""
from __future__ import annotations
import numpy as np
import torch
from experiments import campaign_compliant_continuous_reference as base
from experiments import campaign_compliant_patch_inverse as original
from experiments.campaign_boundary_gaussians import normalized_surface_weights
from scipy.special import logsumexp


class ArbitraryForceForward(base.ExactVectorForward):
    """Exact finite model with caller-supplied, known held force arrays."""
    def __call__(self, theta, force='A'):
        theta=np.asarray(theta,dtype=float)
        if (theta.ndim!=2 or theta.shape[1]!=4 or not len(theta)
                or not np.isfinite(theta).all() or (theta[:,[0,2,3]]<=0).any()):
            raise ValueError('Finite physical theta[B,4] required')
        if isinstance(force,str):
            return super().__call__(theta,force)
        force=np.asarray(force,dtype=float)
        if force.shape==(60,5,3):force=np.broadcast_to(force,(len(theta),60,5,3))
        if force.shape!=(len(theta),60,5,3) or not np.isfinite(force).all():
            raise ValueError('Finite force[60,5,3] or force[B,60,5,3] required')
        # Same algebra as frozen vector model; independent original.matrices
        # assembly is reserved for the separate numerical equality control.
        center=np.column_stack([theta[:,1],np.full(len(theta),.02),np.zeros(len(theta))])
        logw=np.log(self.areas)[None]-.5*np.sum((self.X[None]-center[:,None])**2,2)/theta[:,2,None]**2
        weights=np.exp(logw-logsumexp(logw,axis=1)[:,None])
        diagonal=torch.tensor(np.repeat(weights,3,axis=1))
        mass=self.mass;K=torch.tensor(theta[:,0,None,None])*self.Kunit+torch.diag_embed(diagonal*torch.tensor(theta[:,3,None]));C=torch.diag_embed(.05*diagonal);h=original.H
        chol=torch.linalg.cholesky(K+torch.diag(mass/h**2)[None]+C/h)
        u=torch.zeros((len(theta),15),dtype=torch.float64);v=u.clone();states=[u.clone()]
        held=torch.tensor(force.reshape(len(theta),60,15))
        for step in range(60):
            rhs=held[:,step]+mass*(u/h**2+v/h)+torch.einsum('bij,bj->bi',C,u)/h
            new=torch.cholesky_solve(rhs[:,:,None],chol)[:,:,0]
            v=(new-u)/h;u=new;states.append(u.clone())
        return torch.stack(states,1).reshape(len(theta),61,5,3).numpy()


def independent_numpy_forward(theta,force):
    """Independent first-order 30x30 solve, small control batches only.

    Matrix assembly is the original unvectorized fixture implementation.
    This intentionally differs from the production displacement elimination.
    """
    theta=np.asarray(theta,float);force=np.asarray(force,float)
    if theta.shape!=(4,) or force.shape!=(60,5,3) or not np.isfinite(force).all():
        raise ValueError('One physical theta and one finite force array')
    matrices=original.matrices(theta[None]);M=np.diag(matrices['mass'].numpy())
    K=(matrices['elastic_K']+matrices['fixture_K'])[0].numpy();C=matrices['C'][0].numpy();h=original.H
    L=np.block([[np.eye(15),-h*np.eye(15)],[h*K,M+h*C]])
    u=np.zeros(15);v=u.copy();saved=[u.copy()]
    for F in force.reshape(60,15):
        uv=np.linalg.solve(L,np.concatenate([u,M@v+h*F]))
        u=uv[:15];v=uv[15:];saved.append(u.copy())
    return np.array(saved).reshape(61,5,3)


def force_programs():
    """Predeclared held-out continuations, not selected from observed errors.

    Intervals0..23 are bit-identical to prefix A; interventions start interval24
    (time .2s). Their future control is known to the forecaster, not a label leak.
    """
    X,_,faces=original.tiny_geometry()
    centers=(np.array([.005,.005,.005]),np.array([.09,.075,.075]))
    weights=[normalized_surface_weights(X,faces,c[None],np.eye(3)[None]*.03**2,
                                        np.array([True]))[0][0] for c in centers]
    t=(np.arange(60)+.5)*original.H
    phase=np.clip((t-.2)/.25,0,1);pulse=.02*np.sin(np.pi*phase)**2
    pulse[(t<=.2)|(t>=.45)]=0
    early_phase=np.clip((t-.2)/.15,0,1);early=.02*np.sin(np.pi*early_phase)**2
    early[(t<=.2)|(t>=.35)]=0
    direction=np.array([.6,1.,.3]);direction/=np.linalg.norm(direction)
    specs={'changed_location':(weights[0],direction,pulse),
           'reversed_force':(weights[1],-direction,pulse),
           'early_transverse':(weights[1],np.array([0.,0.,1.]),early)}
    result={}
    for name,(weight,direction,amplitude) in specs.items():
        force=amplitude[:,None,None]*weight[None,:,None]*direction[None,None,:]
        force[:24]=original.known_force('A')[:24]
        result[name]=force
    return result


def design():
    return dict(schema='compliant-held-force-challenge-v0.1',
        scope='Manufactured fixed15DOF linear implicit-Euler reference; no continuum, nonlinear, camera or physical calibration claim',
        prefix_intervals=list(range(24)),target_frames=list(range(25,61)),
        target_nodes=list(range(5)),target_components=['x','y','z'],
        force_programs=list(force_programs()),
        force_known_before_forecast=True,target_used_for_policy='Posterior predictive covariance only; no realized truth or observations',
        candidate_pokes='Reset experiment from zero state, one .02N smooth pulse; same duration, peak force and scalar impulse across candidates',
        poke_observation_frames=[12,24],poke_observation_nodes=[1,4],noise_std_m=original.SIGMA,
        policies=['target_variance','fixed_first','seeded_random','parameter_information'],
        policy_random_seed=2026102401,
        update='Exact Gaussian likelihood importance reweighting, not Gaussian posterior approximation',
        ESS_gate='Raw ESS>=400 AND ESS/number_draws>=.05; report per-chain mass and split-half stability; no ESS establishes calibrated coverage',
        frozen_after_results=False)


def poke_programs():
    """Equal actuation budget: same pulse scalar at two locations, three axes."""
    X,_,faces=original.tiny_geometry();amplitude=np.linalg.norm(original.known_force('A').sum(1),axis=1)
    result={}
    for location,c in enumerate(([.005,.005,.005],[.09,.075,.075])):
        w=normalized_surface_weights(X,faces,np.array([c]),np.eye(3)[None]*.03**2,np.array([True]))[0][0]
        for axis in range(3):
            result[f'location{location}_axis{axis}']=amplitude[:,None,None]*w[None,:,None]*np.eye(3)[axis]
    return result


def _inputs(target,pokes,sigma):
    target=np.asarray(target,float);pokes=np.asarray(pokes,float)
    if (target.ndim!=2 or pokes.ndim!=3 or target.shape[0]!=pokes.shape[1]
            or len(target)<2 or not np.isfinite(target).all() or not np.isfinite(pokes).all()
            or not np.isfinite(sigma) or sigma<=0):raise ValueError('Finite target[N,D], pokes[K,N,P], positive sigma')
    return target,pokes


def gaussian_target_reduction(target,pokes,*,sigma=original.SIGMA):
    """Expected trace reduction Cov(T,Y)(Cov(Y)+noise)^-1Cov(Y,T).

    This is a Gaussian second-moment acquisition approximation. Subsequent
    posterior updates must use the exact likelihood, not this approximation.
    """
    target,pokes=_inputs(target,pokes,sigma);t=target-target.mean(0);scores=[]
    for observations in pokes:
        y=observations-observations.mean(0);cov=y.T@y/len(y)+sigma**2*np.eye(y.shape[1]);cross=t.T@y/len(y)
        scores.append(float(np.sum(cross*np.linalg.solve(cov,cross.T).T)/target.shape[1]))
    return np.asarray(scores)


def parameter_information(unit,pokes,*,sigma=original.SIGMA):
    """Strong parameter-covariance information baseline, without truth parameters.

    Uses posterior covariance whitening: sum of reductions in standardized
    intrinsic coordinates. Not a claim to evaluate a local material Fisher field.
    """
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or not np.isfinite(unit).all():raise ValueError('Finite intrinsic parameter draws')
    centered=unit-unit.mean(0);cov=centered.T@centered/len(unit)
    L=np.linalg.cholesky(cov)  # no jitter or dropped directions
    standardized=np.linalg.solve(L,centered.T).T
    return gaussian_target_reduction(standardized,pokes,sigma=sigma)


def choose_actions(target,pokes,unit,*,sigma=original.SIGMA,seed=2026102401):
    # This function deliberately has no truth or realized observation argument.
    target,pokes=_inputs(target,pokes,sigma)
    scores=gaussian_target_reduction(target,pokes,sigma=sigma)
    parameter_scores=parameter_information(unit,pokes,sigma=sigma)
    return dict(target_variance=int(np.argmax(scores)),fixed_first=0,
                seeded_random=int(np.random.default_rng(seed).integers(len(pokes))),
                parameter_information=int(np.argmax(parameter_scores))),dict(target_variance=scores,parameter_information=parameter_scores)


def exact_reweight(predicted,observation,*,sigma=original.SIGMA):
    x=np.asarray(predicted,float);y=np.asarray(observation,float)
    if (x.ndim!=2 or y.shape!=(x.shape[1],) or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(sigma) or sigma<=0):
        raise ValueError('Finite predictions and observation, positive noise')
    logw=-.5*np.sum(((x-y)/sigma)**2,axis=1);logw-=logw.max();weights=np.exp(logw);weights/=weights.sum()
    ess=float(1/np.square(weights).sum())
    return weights,dict(raw_importance_ESS=ess,ESS_fraction=ess/len(x),maximum_weight=float(weights.max()),
                       importance_gate_passed=bool(ess>=400 and ess/len(x)>=.05),
                       autocorrelation_adjusted_ESS_not_established=True)


def weighted_prediction(target,weights):
    target=np.asarray(target,float);weights=np.asarray(weights,float)
    if (target.ndim!=2 or weights.shape!=(len(target),) or not np.isfinite(target).all()
            or not np.isfinite(weights).all() or (weights<0).any()
            or not np.isclose(weights.sum(),1,atol=1e-13,rtol=0)):
        raise ValueError('Finite predictive matrix and normalized nonnegative weights')
    anchor=target[0];delta=target-anchor;mu=weights@delta;centered=delta-mu
    return anchor+mu,weights@(centered**2)
