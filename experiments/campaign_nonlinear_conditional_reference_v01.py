"""Nonlinear posterior quadrature with a linear Gaussian integration proposal.

All response evaluations use full_predict; amplitude factorization is used
ONLY to choose integration coordinates. No old calibration/qualification
claims transfer to this engine or to its separately admitted response atlas.
"""
import warnings
import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import logsumexp
from scipy.optimize import brentq
from experiments.campaign_surface_conditional_amplitude_v01 import gauss_axis,BOUNDS,LOG_WIDTH
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02,AmplitudePWLPlan
from experiments.campaign_surface_inverse_reference_v01 import compare_grids
from experiments.campaign_nonlinear_tilted_pwl_v01 import NonlinearTiltedPWL,intrinsic_coordinates


class NonlinearConditionalReference(ConditionalAmplitudeReferenceV02):
    def __init__(self,linear_ratio_predict,full_predict,observations,target_labels,*,retain_hypotheses=False,residual_predict=None,evaluation_chunk=1024):
        super().__init__(lambda r,labels:linear_ratio_predict(np.log(r),labels),observations,target_labels,
                         log_ratio_predict=linear_ratio_predict,retain_hypotheses=retain_hypotheses)
        if type(evaluation_chunk)is not int or evaluation_chunk<1:raise ValueError("Positive evaluation chunk")
        self.full_predict=full_predict;self.residual_predict=residual_predict;self.evaluation_chunk=evaluation_chunk

    def _response(self,a,z,labels):
        unit=intrinsic_coordinates(a,z)
        if self.residual_predict is None:return np.asarray(self.full_predict(unit,labels),float)
        unique,inverse=np.unique(z,return_inverse=True)
        g=np.asarray(self._predict(unique,labels),float)[inverse]
        return a[:,None,None]*g+np.asarray(self.residual_predict(unit,labels),float)

    def _full_loglik_az(self,a,z):
        values=self._response(a,z,self.observed)
        if values.shape!=(len(a),len(self.observed),self.channels)or not np.isfinite(values).all():raise ValueError('Full finite observed responses')
        residual=(values[:,self.repeated,:].reshape(len(a),-1)-self.y)/self.sigma
        return self.log_normalizer-.5*np.sum(residual**2,axis=1)

    def full_loglik(self,unit):
        u=np.asarray(unit,float);a=np.exp(-np.log(BOUNDS[0,0])-u[:,0]*LOG_WIDTH[0])
        z=np.log(BOUNDS[1,0])+u[:,1]*LOG_WIDTH[1]+np.log(a)
        return self._full_loglik_az(a,z)

    def _state(self,outer,inner):
        if self._state_cache is not None and self._state_cache['key']==(outer,inner):return self._state_cache
        t,w=gauss_axis(outer);breaks=self._breaks()
        z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
        r,mu,SD,peak=self._statistics(z);lo=np.maximum(1/BOUNDS[0,1],r/BOUNDS[1,1]);hi=np.minimum(1/BOUNDS[0,0],r/BOUNDS[1,0])
        primitive=NonlinearTiltedPWL(mu,SD,lo,hi,log_ratio=z,linear_peak=peak,full_loglik=self.full_loglik,full_loglik_az=self._full_loglik_az,panels=inner,chunk=self.evaluation_chunk)
        lm=logsumexp(primitive.log_weights,axis=1);rowlog=lm+peak+np.log(zw)-np.log(LOG_WIDTH.prod());logZ=float(logsumexp(rowlog))
        rho=np.exp(rowlog-logZ);rho/=rho.sum()
        conditional=np.exp(primitive.log_weights-lm[:,None]);conditional/=conditional.sum(axis=1)[:,None]
        if not np.isfinite(logZ)or not np.isfinite(rho).all()or not np.isfinite(conditional).all():raise FloatingPointError('Invalid nonlinear posterior mass')
        self._state_cache=dict(key=(outer,inner),z=z,r=r,primitive=primitive,rho=rho,conditional=conditional,log_evidence=logZ)
        return self._state_cache

    def integrate(self,*,outer,inner,moment_chunk=64,keep=False,intrinsic=True):
        if type(moment_chunk)is not int or moment_chunk<1:raise ValueError('Positive chunk')
        s=self._state(outer,inner);a=s['primitive'].amplitude
        unit=intrinsic_coordinates(a,s['z'][:,None]).reshape(-1,2);flat_a=a.ravel();flat_z=np.repeat(s['z'],a.shape[1]);p=(s['rho'][:,None]*s['conditional']).ravel()
        anchor=unit[0];ud=unit-anchor;um=anchor+p@ud;uc=ud-p@ud;cov=uc.T@(p[:,None]*uc)
        mean=None;best=(-np.inf,None,None)
        for start in range(0,len(p),moment_chunk):
            u=unit[start:start+moment_chunk];values=self._response(flat_a[start:start+len(u)],flat_z[start:start+len(u)],self.targets).reshape(len(u),-1)
            if values.shape[1]!=len(self.targets)*self.channels or not np.isfinite(values).all():raise ValueError('Finite full target responses')
            contribution=p[start:start+len(u)]@values;mean=contribution if mean is None else mean+contribution
            ll=self._full_loglik_az(flat_a[start:start+len(u)],flat_z[start:start+len(u)]);j=int(np.argmax(ll))
            if ll[j]>best[0]:best=(float(ll[j]),u[j].copy(),values[j].copy())
        variance=np.zeros_like(mean)
        for start in range(0,len(p),moment_chunk):
            u=unit[start:start+moment_chunk];values=self._response(flat_a[start:start+len(u)],flat_z[start:start+len(u)],self.targets).reshape(len(u),-1)
            if not np.isfinite(values).all():raise FloatingPointError('Nonfinite full target response')
            variance+=p[start:start+len(u)]@((values-mean)**2)
        result=dict(n=outer,outer=outer,inner=inner,weighted_hypotheses=len(p),log_evidence=s['log_evidence'],unit_mean=um,unit_covariance=cov,predictive_mean=mean,predictive_SD=np.sqrt(variance),plugin_unit=best[1],plugin_prediction=best[2],plugin_log_likelihood=best[0])
        if intrinsic:
            cdf=np.array([[self.cdf(d,q,outer=outer,inner=inner,log_evidence=s['log_evidence'])for q in np.linspace(0,1,33)]for d in (0,1)])
            if np.any(np.diff(cdf,axis=1)<-1e-12):raise FloatingPointError('Nonmonotone positive full-density CDF')
            qs=np.array([[brentq(lambda q:self.cdf(d,q,outer=outer,inner=inner,log_evidence=s['log_evidence'])-v,0,1,xtol=2e-8,rtol=1e-12)for v in(.05,.5,.95)]for d in (0,1)])
            result.update(unit_CDF=cdf,unit_quantiles=qs)
        if keep and self.retain_hypotheses:
            ll=np.concatenate([self._full_loglik_az(flat_a[i:i+self.evaluation_chunk],flat_z[i:i+self.evaluation_chunk]) for i in range(0,len(p),self.evaluation_chunk)])
            result.update(unit_grid=unit,quadrature_probability=p,log_likelihood=ll)
        return result

    def qualify(self,*,plan=AmplitudePWLPlan()):
        """Separate inner and outer refinements; preserve all coarse failures.

        Qualify only the latest successive pair on BOTH axes under the same
        intrinsic/prediction gates as the tensor reference. Independent adaptive original-a evidence and six quantile-CDF checks are
        also mandatory; physical/catalog admission remains a separate requirement.
        """
        requested=list(dict.fromkeys([(plan.outer[-1],n)for n in plan.inner]+[(n,plan.inner[-1])for n in plan.outer]));results={};levels=[]
        for outer,inner in requested:
            try:
                row=self.integrate(outer=outer,inner=inner,moment_chunk=plan.moment_chunk,keep=(outer,inner)==(plan.outer[-1],plan.inner[-1]));results[outer,inner]=row
                levels.append(dict(outer=outer,inner=inner,status='computed',log_evidence=row['log_evidence'],weighted_hypotheses=row['weighted_hypotheses']))
            except(FloatingPointError,ValueError,np.linalg.LinAlgError)as error:levels.append(dict(outer=outer,inner=inner,status='unresolved',reason=str(error)))
        comparisons={}
        for axis,orders in(('inner',plan.inner),('outer',plan.outer)):
            rows=[]
            for c,f in zip(orders,orders[1:]):
                ck=(plan.outer[-1],c)if axis=='inner'else(c,plan.inner[-1]);fk=(plan.outer[-1],f)if axis=='inner'else(f,plan.inner[-1])
                if ck not in results or fk not in results:check=dict(passed=False,reason='Missing unresolved level')
                else:
                    check=compare_grids(results[ck],results[fk]);difference=float(np.max(abs(results[ck]['unit_mean']-results[fk]['unit_mean'])))
                    check.update(maximum_unit_mean_difference=difference,passed=check['passed']and difference<=.001)
                rows.append(dict(coarse=c,fine=f,**check))
            comparisons[axis]=rows
        refined=all(rows[-1]['passed']for rows in comparisons.values());final=results.get((plan.outer[-1],plan.inner[-1]))
        adaptive=dict(passed=False,reason='Missing final intrinsic result')
        if final is not None:
            try:
                from experiments.campaign_nonlinear_adaptive_oracle_v01 import nonlinear_adaptive_integral
                independent={};relative_errors={}
                with warnings.catch_warnings():
                    warnings.simplefilter('error',IntegrationWarning)
                    for eps in(1e-6,1e-8):
                        full=nonlinear_adaptive_integral(self.full_loglik,self._observed_ratio,self.y,sigma=self.sigma,epsrel=eps)
                        CDF=np.empty((2,3));relative=max(full['estimated_outer_relative_error'],full['maximum_estimated_inner_relative_error'])
                        for dim in(0,1):
                            for j,q in enumerate((.05,.5,.95)):
                                threshold=np.exp(np.log(BOUNDS[dim,0])+final['unit_quantiles'][dim,j]*LOG_WIDTH[dim])
                                partial=nonlinear_adaptive_integral(self.full_loglik,self._observed_ratio,self.y,sigma=self.sigma,CDF=('E'if dim==0 else'K',float(threshold)),epsrel=eps)
                                CDF[dim,j]=np.exp(partial['log_integral']-full['log_integral'])
                                relative=max(relative,partial['estimated_outer_relative_error'],partial['maximum_estimated_inner_relative_error'])
                        if not np.isfinite(full['log_integral'])or not np.isfinite(CDF).all()or not np.isfinite(relative)or np.any((CDF<0)|(CDF>1+1e-8)):
                            raise FloatingPointError('Nonfinite/out-of-domain independent adaptive result')
                        independent[eps]=(full['log_integral'],CDF);relative_errors[str(eps)]=float(relative)
                difference=abs(independent[1e-8][0]-final['log_evidence']);cdf_error=float(np.max(abs(independent[1e-8][1]-[.05,.5,.95])))
                log_refinement=abs(independent[1e-6][0]-independent[1e-8][0]);cdf_refinement=float(np.max(abs(independent[1e-6][1]-independent[1e-8][1])))
                adaptive=dict(passed=bool(difference<=.01 and cdf_error<=.005 and log_refinement<=.01 and cdf_refinement<=.005 and max(relative_errors.values())<=1e-5),
                    log_evidence_difference=float(difference),maximum_CDF_error=cdf_error,CDF_checks_per_epsrel=6,epsrel=[1e-6,1e-8],
                    adaptive_refinement_log_evidence_difference=float(log_refinement),adaptive_refinement_maximum_CDF_difference=cdf_refinement,
                    maximum_QUADPACK_relative_error_estimates=relative_errors,scope='Independent original-a adaptive quadrature and two-tolerance agreement; estimates are not rigorous error bounds or proof against missed peaks')
            except(FloatingPointError,ValueError,RuntimeError,IntegrationWarning)as error:adaptive=dict(passed=False,reason=type(error).__name__+': '+str(error))
        qualified=refined and adaptive['passed']
        return dict(qualified=bool(qualified),local_final=final,diagnostics=dict(schema='nonlinear-conditional-reference-v01',qualified=bool(qualified),levels=levels,successive=comparisons,independent_adaptive=adaptive,
            prior='da dz/(a ΔlogE ΔlogK), original independent loguniform E/K prior',
            all_modes_retained=True,no_parameter_snapping=True,atlas_interpolation_requires_separate_admission=True,no_quadrature_ESS=True,
            conditional_on_admitted_nonlinear_atlas=True,independent_adaptive_per_posterior_required=True,physical_admission_separate=True,
            gates=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01)))
