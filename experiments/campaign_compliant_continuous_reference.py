"""Continuous posterior development for the IMMUTABLE manufactured linear inverse.

Exact-algebra vectorization of the old finite model; no physical approximation,
new observations, changed prior or current cohort data. Adaptive 2D cubature is
an explicitly conditional slice. SMC weights alone never establish convergence.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.special import logsumexp
from experiments import campaign_compliant_patch_inverse as original
from experiments.campaign_tempered_reference import tempered_smc
from experiments.campaign_boundary_gaussians import SurfaceQuadrature
from experiments.campaign_linear_fem import assemble_tetrahedra
from physgauss.chain_diagnostics import diagnose_quantities


@dataclass(frozen=True)
class State:
    coordinates:np.ndarray
    def take(self,ids):return State(self.coordinates[np.atleast_1d(ids)].copy())
    def with_coordinates(self,x):return State(np.asarray(x))


class BoxPrior:
    def __init__(self,dimension):self.lower=np.zeros(dimension);self.upper=np.ones(dimension)
    def sample(self,n,rng):return State(rng.uniform(size=(n,len(self.lower))))
    def log_prob(self,state):
        x=state.coordinates
        return np.where(np.isfinite(x).all(1)&(x>=0).all(1)&(x<=1).all(1),0.,-np.inf)


def physical(unit,dimensions):
    unit=np.asarray(unit,dtype=float)
    if unit.ndim!=2 or unit.shape[1]!=len(dimensions) or not np.isfinite(unit).all() or ((unit<0)|(unit>1)).any():
        raise ValueError('Unit intrinsic coordinates inside the unchanged prior required')
    x=np.tile(original.TRUTH,(len(unit),1));base=np.log(x[:,[0,2,3]]);x[:,[0,2,3]]=base
    x[:,dimensions]=original.LOWER[list(dimensions)]+unit*(original.UPPER-original.LOWER)[list(dimensions)]
    x[:,[0,2,3]]=np.exp(x[:,[0,2,3]])
    return x


class ExactVectorForward:
    """Same 15x15 physical L and held force, vectorized Gaussian weights.

    No eigentruncation, stiffness repair, surrogate, altered integration or
    likelihood. Tight independent equality tests against original.forward gate use.
    """
    def __init__(self):
        self.X,tets,faces=original.tiny_geometry()
        self.Kunit,self.mass,_,_=assemble_tetrahedra(self.X,tets,1.,.35,625.)
        self.areas=SurfaceQuadrature.from_mesh(self.X,faces).node_area_m2
        self.forces={name:torch.tensor(original.known_force(name).reshape(60,15)) for name in ('A','B')}
    def __call__(self,theta,program='A'):
        theta=np.asarray(theta,dtype=float)
        if theta.ndim!=2 or theta.shape[1]!=4 or not len(theta) or not np.isfinite(theta).all() or (theta[:,[0,2,3]]<=0).any():
            raise ValueError('Finite physical theta required')
        center=np.column_stack([theta[:,1],np.full(len(theta),.02),np.zeros(len(theta))])
        logw=np.log(self.areas)[None]-.5*np.sum((self.X[None]-center[:,None])**2,2)/theta[:,2,None]**2
        weights=np.exp(logw-logsumexp(logw,axis=1)[:,None]);diagonal=torch.tensor(np.repeat(weights,3,axis=1))
        K=torch.tensor(theta[:,0,None,None])*self.Kunit+torch.diag_embed(diagonal*torch.tensor(theta[:,3,None]))
        C=.05*diagonal;h=original.H;m=self.mass
        chol=torch.linalg.cholesky(K+torch.diag_embed(m[None]/h**2+C/h))
        u=torch.zeros((len(theta),15),dtype=torch.float64);v=u.clone();saved=[u.clone()]
        for F in self.forces[program]:
            new=torch.cholesky_solve((F[None]+m*(u/h**2+v/h)+C*u/h)[:,:,None],chol)[:,:,0]
            v=(new-u)/h;u=new;saved.append(u.clone())
        return torch.stack(saved,1).reshape(len(theta),61,5,3).numpy()
    def log_likelihood(self,unit,dimensions,observation):
        pred=original.selected_observations(self(physical(unit,dimensions),'A'))
        return -.5*np.sum(((pred-observation)/original.SIGMA)**2,axis=1)-len(observation)*np.log(original.SIGMA*np.sqrt(2*np.pi))


def smc(forward,dimensions,observation,particles,seed):
    return tempered_smc(BoxPrior(len(dimensions)),lambda s:forward.log_likelihood(s.coordinates,dimensions,observation),
        particles=particles,seed=seed,ess_fraction=.8,rejuvenation_steps=8,proposal_scale=.08,
        independence_probability=.25,max_stages=64,periodic_indices=())


def empirical_cdf(samples,knots):return np.mean(np.asarray(samples)[:,None]<=np.asarray(knots)[None],axis=0)


def covariance_proposal(samples):
    mean=samples.mean(0);cov=np.cov(samples,rowvar=False)
    L=np.linalg.cholesky(cov) # no jitter: degenerate pilot explicitly fails
    return mean,cov,L,np.linalg.inv(cov)


def continuous_mcmc(loglik,pilot,*,seed,warmup,draws,chains=8,progress=None):
    """Fixed-kernel MH: fitted independent normal + local reflected DIAGONAL RW.

    Pilot only sets proposal; exact MH ratio corrects its density. ALL chains
    start independently from the PRIOR, not only a selected SMC basin. No tuning
    or warmup extension; warmup excluded and every retained draw unthinned.
    """
    rng=np.random.default_rng(seed);dimension=pilot.shape[1]
    mean,cov,L,inverse=covariance_proposal(pilot)
    x=rng.uniform(size=(chains,dimension));ll=loglik(x);saved=[];savedll=[];counts=np.zeros((2,3),int)
    def logq(z):return -.5*np.einsum('bi,ij,bj->b',z-mean,inverse,z-mean)
    for step in range(warmup+draws):
        kind=rng.choice(3,size=chains,p=[.5,.45,.05])
        candidate=x+rng.normal(size=x.shape)*.06
        folded=np.mod(candidate,2.);candidate=np.minimum(folded,2.-folded)
        independent=mean+rng.normal(size=x.shape)@L.T
        candidate[kind==0]=independent[kind==0];candidate[kind==2]=rng.uniform(size=(int((kind==2).sum()),dimension))
        valid=((candidate>=0)&(candidate<=1)).all(1)
        newll=np.full(chains,-np.inf);newll[valid]=loglik(candidate[valid]) if valid.any() else []
        correction=np.where(kind==0,logq(x)-logq(candidate),0.)
        accept=np.log(rng.random(chains))<np.minimum(0.,newll-ll+correction)
        x[accept]=candidate[accept];ll[accept]=newll[accept]
        phase=int(step>=warmup)
        for k in range(3):counts[phase,k]+=int(np.sum(accept&(kind==k)))
        if phase:saved.append(x.copy());savedll.append(ll.copy())
        if progress and step%100==0:progress(step,warmup+draws)
    samples=np.stack(saved,axis=1);likelihood=np.stack(savedll,axis=1)
    report=diagnose_quantities(torch.tensor(np.concatenate([samples,likelihood[:,:,None]],axis=2)),
        ['unit_'+str(i) for i in range(dimension)]+['log_likelihood'])
    autocorrelation={}
    for j in range(dimension):
        centered=samples[:,:,j]-samples[:,:,j].mean(1,keepdims=True);den=np.sum(centered**2)
        autocorrelation['unit_'+str(j)]={str(lag):float(np.sum(centered[:,:-lag]*centered[:,lag:])/den) for lag in (1,10,50)}
    return samples,dict(diagnostics=report,autocorrelation=autocorrelation,accepted_moves_by_phase_kernel=counts.tolist(),
        warmup=warmup,draws_per_chain=draws,chains=chains,all_initial_states_independent_prior=True,
        fixed_kernel_after_pilot=True,no_thinning=True,proposal_covariance=cov.tolist())
