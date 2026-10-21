"""Shared lowrankGaussian discrepancy for physical targets and censored rays.

T = T_theta + mT + BT*z, h = h_theta + mO + BO*z + sensornoise,
z~N(0,I). The SAME latentz persists across base and extra observations.
Visibility depends on the physical prediction, not on z: a deliberate working
approximation to nonlinear rendering. Observed predictedmisses use independent
broadbackground, never fictitious range residuals. No outlier mixture is fitted.
No cohort, simulator, fitting, posterior resampling or actionrisk aggregation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
try:
    from censored_range_observation import CensoredRanges
except ImportError:
    from experiments.censored_range_observation import CensoredRanges


def _readonly(x,dtype=np.float64):
    y=np.array(x,dtype=dtype,copy=True,order='C');y.flags.writeable=False;return y


def _frozen_ranges(x):
    y=CensoredRanges(x.visible,x.residual_m);y.visible.flags.writeable=False;y.residual_m.flags.writeable=False;return y


@dataclass(frozen=True)
class LatentConditional:
    log_likelihood: np.ndarray
    latent_mean: np.ndarray
    covariance_groups: np.ndarray
    group_index: np.ndarray
    observation_rows: tuple
    predictions: CensoredRanges
    observation: CensoredRanges
    model_fingerprint: str

    def covariance_for_particles(self):
        """Explicit small-problem expansion; mainAPI retains groupedcovariance."""
        return self.covariance_groups[self.group_index]

    def take(self,indices):
        """Select suppliedphysicalparticleindices, withduplicates, withoutweights.

        This is not resampling or a newlikelihood update: caller choosesindices.
        Shared latentconditionals stay aligned after empiricalposterior indexing.
        """
        i=np.asarray(indices)
        if i.ndim!=1 or not len(i) or i.dtype.kind not in 'iu' or np.any(i<0) or np.any(i>=len(self.log_likelihood)):
            raise ValueError('Validexplicitparticleindices required')
        return LatentConditional(_readonly(self.log_likelihood[i]),_readonly(self.latent_mean[i]),self.covariance_groups,
            _readonly(self.group_index[i],np.int64),self.observation_rows,
            _frozen_ranges(CensoredRanges(self.predictions.visible[i],self.predictions.residual_m[i])),self.observation,self.model_fingerprint)


@dataclass(frozen=True)
class ConditionalTargetMoments:
    mean: np.ndarray
    variance_by_group: np.ndarray
    group_index: np.ndarray
    boundary_projection_applied: bool=False

    def variance_for_particles(self):
        return self.variance_by_group[self.group_index]


class JointGaussianDiscrepancy:
    """Rank<=16 properworkingmodel, visibility.005/background.05m fixed.

    isotropic_std_m defaults.2mm and may include a separatelyfitted independent
    observationnugget; it must never fall below the physical.2mm sensornoise.
    Targetmeans/loadings are unprojected: caller enforces candidateBC rows.

    Observationloadings cover one global rowpool (e.g72base+4*36views).
    Explicit distinct rowindices identify every observed/query channel. Target
    covariance stays lowrank; only conditional marginaldiagonals are exported.
    """
    sensor_std_m=.0002
    visibility_error=.005
    background_std_m=.05
    outlier_probability=0.

    def __init__(self,observation_mean_offset_m,observation_loading_m,target_mean_offset_m,target_loading_m,*,isotropic_std_m=.0002):
        if not np.isfinite(isotropic_std_m) or isotropic_std_m<.0002:raise ValueError('Isotropicnoise must be at least physical.2mm sensorstd')
        self.isotropic_std_m=float(isotropic_std_m)
        mo,bo,mt,bt=[np.asarray(x,dtype=np.float64) for x in (observation_mean_offset_m,observation_loading_m,target_mean_offset_m,target_loading_m)]
        if (mo.ndim!=1 or mt.ndim!=1 or not len(mo) or not len(mt) or bo.ndim!=2 or bt.ndim!=2
            or bo.shape[0]!=len(mo) or bt.shape[0]!=len(mt) or bo.shape[1]!=bt.shape[1] or bo.shape[1]>16
            or any(not np.isfinite(x).all() for x in [mo,bo,mt,bt])):
            raise ValueError('Finiteoffsetvectors andcompatiblelowrankloadings r<=16 required')
        self.observation_mean_offset_m=_readonly(mo);self.observation_loading_m=_readonly(bo)
        self.target_mean_offset_m=_readonly(mt);self.target_loading_m=_readonly(bt);self.rank=bo.shape[1]
        h=hashlib.sha256()
        for x in [self.observation_mean_offset_m,self.observation_loading_m,self.target_mean_offset_m,self.target_loading_m]:
            h.update(str(x.shape).encode());h.update(x.tobytes())
        h.update(json.dumps([self.isotropic_std_m,self.sensor_std_m,self.visibility_error,self.background_std_m,0.]).encode());self.fingerprint=h.hexdigest();self._sealed=True

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('Frozenjointmodel; constructanewinstance fornewfactors/settings')
        object.__setattr__(self,name,value)

    def _rows(self,rows,dimension):
        r=np.asarray(rows)
        if (r.shape!=(dimension,) or r.dtype.kind not in 'iu' or len(np.unique(r))!=len(r)
            or not len(r) or np.any(r<0) or np.any(r>=len(self.observation_mean_offset_m))):
            raise ValueError('Explicitdistinctvalidglobal observationrowindices required')
        return r.astype(np.int64)

    def _predictions(self,predictions,rows):
        if not isinstance(predictions,CensoredRanges) or predictions.visible.ndim!=2 or not len(predictions.visible):raise ValueError('Predicted[particles,channels] censoredranges required')
        return self._rows(rows,predictions.visible.shape[1])

    def _posterior(self,p):
        if not isinstance(p,LatentConditional) or p.model_fingerprint!=self.fingerprint:raise ValueError('Conditionalstate from SAMEjointmodel required')
        n=len(p.log_likelihood)
        if p.latent_mean.shape!=(n,self.rank) or p.group_index.shape!=(n,):raise ValueError('Invalidlatentconditional dimensions')
        return n

    def condition(self,predictions,observation,rows):
        """MarginalLL and exact latentGaussian conditional perphysicalparticle."""
        rows=self._predictions(predictions,rows)
        if not isinstance(observation,CensoredRanges) or observation.visible.shape!=(len(rows),):raise ValueError('Oneobservation matchingexplicitrows required')
        h=predictions.visible;v=observation.visible;y=observation.residual_m;n=len(h);variance=self.isotropic_std_m**2
        matches=(h==v).sum(1);ll=matches*np.log1p(-self.visibility_error)+(len(rows)-matches)*np.log(self.visibility_error)
        patterns,group=np.unique(h&v[None],axis=0,return_inverse=True)
        means=np.zeros((n,self.rank));covariances=[];log2pi=np.log(2*np.pi)
        bo=self.observation_loading_m[rows];mo=self.observation_mean_offset_m[rows]
        for g,a in enumerate(patterns):
            indices=np.flatnonzero(group==g);background=v&~a
            ll[indices]+=-.5*(np.square(y[background]).sum()/self.background_std_m**2+background.sum()*np.log(2*np.pi*self.background_std_m**2))
            B=bo[a];precision=np.eye(self.rank)+B.T@B/variance
            if self.rank:
                L=np.linalg.cholesky(precision);C=np.linalg.solve(L.T,np.linalg.solve(L,np.eye(self.rank)));C=(C+C.T)/2
            else:L=np.empty((0,0));C=np.empty((0,0))
            covariances.append(C)
            if a.any():
                residual=y[a][None]-predictions.residual_m[indices][:,a]-mo[a]
                first=np.square(residual).sum(1)/variance
                if self.rank:
                    rhs=B.T@residual.T/variance;projected=np.linalg.solve(L,rhs)
                    second=np.square(projected).sum(0);quad=first-second
                    error=64*np.finfo(float).eps*(np.abs(first)+np.abs(second)+np.finfo(float).tiny)
                    if np.any(quad < -error):raise FloatingPointError('NegativeWoodbury quadratic beyondroundoff')
                    quad=np.maximum(quad,0.);means[indices]=np.linalg.solve(L.T,projected).T
                    logdet=a.sum()*np.log(variance)+2*np.log(np.diag(L)).sum()
                else:quad=first;logdet=a.sum()*np.log(variance)
                ll[indices]+=-.5*(quad+a.sum()*log2pi+logdet)
        if not np.isfinite(ll).all() or not np.isfinite(means).all():raise FloatingPointError('NonfinitejointGaussian conditional')
        return LatentConditional(_readonly(ll),_readonly(means),_readonly(np.array(covariances)),_readonly(group,np.int64),
            tuple(map(int,rows)),_frozen_ranges(predictions),_frozen_ranges(observation),self.fingerprint)

    def marginal_log_likelihood(self,predictions,observation,rows):
        return self.condition(predictions,observation,rows).log_likelihood

    def condition_extra_view(self,base,prediction,observation,rows):
        """Return jointconditional and LLjoint−LLbase; never doublecountsharedz."""
        n=self._posterior(base);rows=self._predictions(prediction,rows)
        if len(prediction.visible)!=n or set(rows)&set(base.observation_rows):raise ValueError('Sameparticles and NEWdistinct observationrows required')
        if not isinstance(observation,CensoredRanges) or observation.visible.shape!=(len(rows),):raise ValueError('Oneextra observation required')
        joint_prediction=CensoredRanges(np.concatenate([base.predictions.visible,prediction.visible],1),np.concatenate([base.predictions.residual_m,prediction.residual_m],1))
        joint_observation=CensoredRanges(np.concatenate([base.observation.visible,observation.visible]),np.concatenate([base.observation.residual_m,observation.residual_m]))
        posterior=self.condition(joint_prediction,joint_observation,tuple(base.observation_rows)+tuple(map(int,rows)))
        return posterior,posterior.log_likelihood-base.log_likelihood

    def conditional_target_moments(self,target_prediction,posterior,*,feature_chunk_size=4096):
        """Conditionalmean and grouped marginalvariances; no dense targetcov."""
        n=self._posterior(posterior);target=np.asarray(target_prediction,dtype=np.float64)
        if target.shape!=(n,len(self.target_mean_offset_m)) or not np.isfinite(target).all() or type(feature_chunk_size) is not int or feature_chunk_size<1:raise ValueError('Finite matchingtargetpredictions andpositivechunks required')
        mean=target+self.target_mean_offset_m+posterior.latent_mean@self.target_loading_m.T
        variances=np.empty((len(posterior.covariance_groups),target.shape[1]))
        L=np.linalg.cholesky(posterior.covariance_groups) if self.rank else None
        for first in range(0,target.shape[1],feature_chunk_size):
            B=self.target_loading_m[first:first+feature_chunk_size]
            variances[:,first:first+len(B)]=np.sum(np.einsum('dr,grs->gds',B,L,optimize=True)**2,axis=-1) if self.rank else 0.
        if np.any(variances<0) or not np.isfinite(variances).all():raise FloatingPointError('Invalidtargetvariance')
        return ConditionalTargetMoments(mean,variances,posterior.group_index)

    def fantasize_view(self,base,prediction,rows,weights,*,count=32,seed=0):
        """Conditionalposterior predictive view from SAME latent sharedwithbase.

        Commonseed givescommon ancestor/z/visibility/noise draws acrossequalD
        candidateviews. Physicalweights must already be conditionedonbaseLL.
        """
        n=self._posterior(base);rows=self._predictions(prediction,rows);w=np.asarray(weights,dtype=np.float64)
        if (len(prediction.visible)!=n or set(rows)&set(base.observation_rows) or w.shape!=(n,)
            or not np.isfinite(w).all() or np.any(w<0) or not np.isfinite(w.sum()) or w.sum()<=0
            or type(count) is not int or count<1 or type(seed) is not int):raise ValueError('Validbaseposteriorweights,newrows,fantasycount/seed required')
        w=w/w.sum();rng=np.random.default_rng(seed);ancestors=rng.choice(n,count,p=w)
        if self.rank:
            C=base.covariance_groups[base.group_index[ancestors]];L=np.linalg.cholesky(C)
            z=base.latent_mean[ancestors]+np.einsum('fij,fj->fi',L,rng.normal(size=(count,self.rank)))
        else:z=np.empty((count,0))
        h=prediction.visible[ancestors];visible=h^(rng.random(h.shape)<self.visibility_error)
        foreground=prediction.residual_m[ancestors]+self.observation_mean_offset_m[rows]+z@self.observation_loading_m[rows].T+rng.normal(0,self.isotropic_std_m,h.shape)
        background=rng.normal(0,self.background_std_m,h.shape);values=np.where(h,foreground,background)
        return CensoredRanges(visible,values),dict(generating_particle_indices=ancestors,conditional_latent_z=z,
            sensor_model='ConditionalsharedGaussianz +independentisotropic foregroundnoise +Bernoulli.005visibility +independent.05m background; eta0',
            isotropic_std_m=self.isotropic_std_m,physical_sensor_std_m=self.sensor_std_m,
            base_rows=base.observation_rows,view_rows=tuple(map(int,rows)))
