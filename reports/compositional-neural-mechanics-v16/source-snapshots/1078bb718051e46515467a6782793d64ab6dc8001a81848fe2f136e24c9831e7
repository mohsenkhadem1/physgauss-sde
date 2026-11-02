"""Three-dimensional bounded posterior via two-dimensional mechanics + scale.
All priors and densities expressed with respect to ds dr_left dr_right (Jacobian1).
"""
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.integrate import cumulative_simpson
from scipy.optimize import brentq

LOGK_MEAN=np.log(500.);LOGK_BOUNDS=np.log([20.,10000.])

def scale_bounds(r):
    return max(-1.,LOGK_BOUNDS[0]-r[0],LOGK_BOUNDS[0]-r[1]),min(1.,LOGK_BOUNDS[1]-r[0],LOGK_BOUNDS[1]-r[1])


def integrate(nodes,log_quadrature_over_proposal,position_loglikelihood,wrench_abc,wrench_logconstant,track,points):
    if points<3 or points%2!=1 or track not in ('V','VF'):raise ValueError('Odd uniform Simpson grid and explicit track required')
    r=np.asarray(nodes);bounds=np.array([scale_bounds(v)for v in r]);valid=bounds[:,1]>bounds[:,0]
    if not valid.all():raise ValueError('Outside-prior quadrature nodes must be excluded explicitly before integration')
    s=bounds[:,0,None]+(bounds[:,1]-bounds[:,0])[:,None]*np.linspace(0,1,points)
    x=s[:,:,None]+np.stack([np.zeros(len(r)),r[:,0],r[:,1]],1)[:,None,:]
    prior_normalizer=(ndtr(1)-ndtr(-1))*(ndtr(LOGK_BOUNDS[1]-LOGK_MEAN)-ndtr(LOGK_BOUNDS[0]-LOGK_MEAN))**2
    logp=-.5*np.sum((x-np.array([0.,LOGK_MEAN,LOGK_MEAN]))**2,axis=-1)-1.5*np.log(2*np.pi)-np.log(prior_normalizer)
    logp+=np.asarray(position_loglikelihood)[:,None]
    if track=='VF':
        a,b,c=np.asarray(wrench_abc).T;e=np.exp(s)
        logp+=-.5*(a[:,None]*e**2-2*b[:,None]*e+c[:,None])+wrench_logconstant
    coeff=np.ones(points);coeff[1:-1:2]=4;coeff[2:-1:2]=2
    logweight=np.asarray(log_quadrature_over_proposal)[:,None]+logp+np.log(coeff[None,:]*(bounds[:,1]-bounds[:,0])[:,None]/(points-1)/3)
    evidence=logsumexp(logweight);w=np.exp(logweight-evidence)
    mean=np.einsum('ij,ijk->k',w,x);difference=x-mean
    cov=np.einsum('ij,ijk,ijl->kl',w,difference,difference);sd=np.sqrt(np.diag(cov))
    # Integrate a continuous conditional CDF, rather than treating grid nodes as atoms.
    normalized_density=np.exp(np.asarray(log_quadrature_over_proposal)[:,None]+logp-evidence)
    cdf=cumulative_simpson(normalized_density,x=s,axis=1,initial=0)
    if np.min(np.diff(cdf,axis=1)) < -1e-12:raise ValueError('Conditional Simpson CDF nonmonotone: scale grid inadequate')
    intervals=[]
    shifts=np.stack([np.zeros(len(r)),r[:,0],r[:,1]],1)
    for j in range(3):
        def probability(value):return sum(np.interp(value-shifts[k,j],s[k],cdf[k],left=0.,right=cdf[k,-1]) for k in range(len(r)))
        intervals.append([brentq(lambda t:probability(t)-q,float(x[:,:,j].min())-1e-12,float(x[:,:,j].max())+1e-12,xtol=1e-12) for q in (.05,.95)])
    return dict(log_evidence=float(evidence),mean=mean.tolist(),covariance=cov.tolist(),sd=sd.tolist(),interval90=intervals,
                conditional_cdf_total=float(cdf[:,-1].sum()),quadrature_effective_nodes=float(1/np.sum(w.sum(1)**2)))


def compare(a,b):
    sd=np.asarray(b['sd']);mean=float(np.max(np.abs(np.array(a['mean'])-b['mean'])/sd));ends=float(np.max(np.abs(np.array(a['interval90'])-b['interval90'])/sd[:,None]));cov=float(np.linalg.norm(np.array(a['covariance'])-b['covariance'])/np.linalg.norm(b['covariance']))
    return dict(mean_change_in_sd=mean,interval_endpoint_change_in_sd=ends,covariance_relative_change=cov,log_evidence_change=abs(a['log_evidence']-b['log_evidence']),passed=bool(mean<.02 and ends<.05 and cov<.02))
