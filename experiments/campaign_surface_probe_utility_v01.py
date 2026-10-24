"""Prospective static probe utility, with an explicit Gaussian approximation.

The score is reduction in the best *linear* predictor's target MSE under the
current posterior joint covariance and known independent measurement noise.
It is not exact Bayesian information gain or a calibration certificate. Action
budgets/families, observations and target queries must be fixed by the study.
No truth, simulator parameters or post-probe outcomes enter action selection.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def covariance_utility(weights,action_predictions,target_predictions,*,noise_SD_m):
    w=np.asarray(weights,float);a=np.asarray(action_predictions,float);t=np.asarray(target_predictions,float)
    if w.ndim!=1 or len(w)<2 or a.shape[0]!=len(w) or t.shape[0]!=len(w) or not np.isfinite(noise_SD_m) or noise_SD_m<=0:
        raise ValueError('Common weighted parameter draws and positive observation noise')
    a=a.reshape(len(w),-1);t=t.reshape(len(w),-1)
    if not a.shape[1] or not t.shape[1]:raise ValueError('Nonempty observation and target channels')
    if not all(np.isfinite(x).all() for x in (w,a,t)) or np.any(w<0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-12):raise ValueError('Finite normalized nonnegative posterior weights')
    ac=a-w@a;tc=t-w@t
    aa=ac.T@(w[:,None]*ac);ta=tc.T@(w[:,None]*ac);tt=float(np.sum(w[:,None]*tc*tc)/t.shape[1])
    noisy=aa+noise_SD_m**2*np.eye(a.shape[1]);factor=cho_factor(noisy,lower=True)
    reduction=float(np.sum(ta*cho_solve(factor,ta.T).T)/t.shape[1])
    if reduction < -1e-20 or reduction>tt+max(1e-20,tt*1e-10):raise FloatingPointError('Invalid linear-predictor variance reduction')
    _,logdet=np.linalg.slogdet(np.eye(a.shape[1])+aa/noise_SD_m**2)
    return dict(expected_linear_target_MSE_reduction_m2=reduction,prior_target_MSE_m2=tt,
        expected_linear_remaining_target_MSE_m2=tt-reduction,Gaussian_observation_information_nats=float(.5*logdet),
        approximation='Posterior covariance best-linear-predictor utility; not exact posterior risk or information gain')


def select_probe(weights,candidate_predictions,target_predictions,*,noise_SD_m):
    scores=[covariance_utility(weights,a,target_predictions,noise_SD_m=noise_SD_m) for a in candidate_predictions]
    if not scores:raise ValueError('Nonempty prospective action catalog')
    selected=int(np.argmax([x['expected_linear_target_MSE_reduction_m2'] for x in scores]))
    return selected,scores
