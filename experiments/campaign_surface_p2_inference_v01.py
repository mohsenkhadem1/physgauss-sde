"""Source-only two-dimensional P2 posterior integration engine.

No cohort, truth parameters, observations, execution, or file I/O are embedded.
Physical catalog admission belongs to the caller. Numerical qualification here
is conditional on the supplied finite linear model, not physical calibration.
"""
from dataclasses import dataclass
import numpy as np
from experiments.campaign_surface_inverse_reference_v01 import (
    posterior_grid, posterior_gauss, compare_grids, gauss_partial_CDF)

LOG_BOUNDS = np.log([[2000., 8000.], [5., 30.]])
NOISE_SD_M = 20e-6


@dataclass(frozen=True)
class IntegrationPlan:
    simpson: tuple = (129, 257, 513, 1025)
    gauss: tuple = (128, 256, 512)
    chunk: int = 1024

    def __post_init__(self):
        if len(self.simpson)<2 or len(self.gauss)<2 or self.chunk<1:
            raise ValueError('At least two fixed levels per integration family')
        if any(isinstance(n,bool) or not isinstance(n,int) or n<5 or n%2==0 for n in self.simpson):
            raise ValueError('Odd Simpson levels >=5')
        if any(isinstance(n,bool) or not isinstance(n,int) or n<4 for n in self.gauss):
            raise ValueError('Gauss levels >=4')
        if any(a>=b for levels in (self.simpson,self.gauss) for a,b in zip(levels,levels[1:])):
            raise ValueError('Strictly increasing levels')


def physical_parameters(unit):
    u=np.asarray(unit,float)
    if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all() or np.any((u<0)|(u>1)):
        raise ValueError('Finite unit-square coordinates')
    return np.exp(LOG_BOUNDS[:,0]+u*np.diff(LOG_BOUNDS,axis=1).ravel())


class LabelledModal:
    """Select output coefficients before matrix multiplication.

    Modal coefficients are query-major, force-minor. This adapter returns
    [batch, selected-force, query]. It also retains an exact Schur particular
    term if represented by an additional zero generalized eigenvalue.
    """
    def __init__(self, modal, labels):
        self.modal=modal;self.labels=tuple(labels)
        q,r=modal.output_shape
        if r!=len(self.labels) or len(set(self.labels))!=r:
            raise ValueError('Unique label per force column')
        self.channels=q
        m=len(modal.eigenvalues)
        if np.asarray(modal.coefficients).shape!=(q*r,m):
            raise ValueError('Full coefficient/eigenvalue shape mismatch')
        self.coefficients=np.asarray(modal.coefficients).reshape(q,r,m)
        self._selected={}

    @classmethod
    def from_forward(cls, forward, *, loads=None, points=None, condensed=True, query_matrix=None):
        # Frozen v01 defaults remain .24N/face. A separately qualified v02
        # catalog must be supplied explicitly; no silent force replacement.
        from experiments.campaign_surface_p2_study_design_v01 import load_catalog, observation_points
        from experiments.campaign_surface_p2_static_v01 import point_query_matrix
        from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference
        from experiments.campaign_surface_condensed_reference_v01 import CondensedStaticReference
        loads=load_catalog(forward.n) if loads is None else loads
        points=observation_points() if points is None else points
        f=np.column_stack([np.asarray(v,float).reshape(-1)[forward.free] for v in loads.values()])
        # Frozen query adapter covers n1/2/4. Extended geometry must inject
        # its versioned query function, independently of the load catalog.
        if query_matrix is None and forward.n not in (1,2,4):
            raise ValueError('Extended P2 geometry requires explicit versioned query_matrix callable')
        P=(point_query_matrix if query_matrix is None else query_matrix)(forward,points)
        constructor=CondensedStaticReference if condensed else ModalStaticReference
        return cls(constructor(forward.Kunit_free,forward.W_free,f,P),tuple(loads))

    def select(self, labels):
        labels=tuple(labels)
        if not labels:raise ValueError('Nonempty selected output labels')
        if labels not in self._selected:
            indices=[self.labels.index(label) for label in labels]
            self._selected[labels]=self.coefficients[:,indices,:].transpose(1,0,2).reshape(len(labels)*self.channels,-1)
        coefficients=self._selected[labels]
        def predict(unit):
            theta=physical_parameters(unit)
            denominator=theta[:,0,None]+theta[:,1,None]*self.modal.eigenvalues
            if not np.isfinite(denominator).all() or np.any(denominator<=0):
                raise FloatingPointError('Nonpositive/nonfinite modal stiffness')
            return ((1/denominator)@coefficients.T).reshape(len(theta),len(labels),self.channels)
        return predict

    def loglikelihood(self, observations, *, sigma=NOISE_SD_M):
        """List of (label, independent observed displacement) pairs.

        Repeated labels retain independent data/noise terms. Reset-probe joint
        inference uses prefix plus probe observations, never importance weights.
        """
        if sigma!=NOISE_SD_M:raise ValueError('This protocol fixes known independent 20um Gaussian noise')
        observations=list(observations)
        if not observations:raise ValueError('At least one observation')
        labels=tuple(dict.fromkeys(label for label,_ in observations))
        predict=self.select(labels);indices=[labels.index(label) for label,_ in observations]
        y=np.asarray([value for _,value in observations],float)
        if y.shape!=(len(observations),self.channels) or not np.isfinite(y).all():
            raise ValueError('Finite observation per channel')
        normalizer=len(observations)*self.channels*np.log(sigma*np.sqrt(2*np.pi))
        def loglik(unit):
            residual=(predict(unit)[:,indices,:]-y)/sigma
            return -.5*np.sum(residual**2,axis=(1,2))-normalizer
        return loglik


def _moment_comparison(coarse,fine):
    errors=dict(log_evidence_difference=float(abs(coarse['log_evidence']-fine['log_evidence'])),
        maximum_unit_mean_difference=float(np.max(abs(coarse['unit_mean']-fine['unit_mean']))))
    passed=errors['log_evidence_difference']<=.01 and errors['maximum_unit_mean_difference']<=.001
    tolerance=np.maximum(2e-6,.01*np.asarray(fine['predictive_SD']))
    for key in ('predictive_mean','predictive_SD'):
        difference=abs(np.asarray(coarse[key])-fine[key])
        errors['maximum_'+key+'_difference_m']=float(np.max(difference))
        passed=passed and bool(np.all(difference<=tolerance))
    return dict(**errors,passed=bool(passed))


def _scalar_level(result):
    # Unit means are local parameter information; public callers redact them.
    return dict(n=result['n'],log_evidence=float(result['log_evidence']),
        unit_mean=np.asarray(result['unit_mean']).tolist(),
        maximum_predictive_SD_m=float(np.max(result['predictive_SD'])),
        minimum_predictive_SD_m=float(np.min(result['predictive_SD'])))


def qualify_integrals(loglik,predict,*,plan=IntegrationPlan()):
    """Execute every frozen integration level; fail closed if final checks fail.

    Earlier failures remain in diagnostics, even if later levels resolve them.
    Results include local raw final Simpson probabilities for the caller only.
    Local diagnostics also contain inferred unit means/CDF thresholds; redact
    these when publishing scalar numerical-error summaries.
    No action, posterior reliability, or physical admission is implied by this
    conditional numerical gate. The six partial-CDF checks use independently
    integrated rectangles at final Simpson q05/50/95 thresholds.
    """
    results={}; records={}; comparisons={}
    for family,levels,integrator in (('Simpson',plan.simpson,posterior_grid),('Gauss',plan.gauss,posterior_gauss)):
        results[family]={};records[family]=[]
        for n in levels:
            try:
                kwargs=dict(n=n,predict=predict,chunk=plan.chunk)
                if family=='Simpson':kwargs['keep']=n==levels[-1]
                result=integrator(loglik,**kwargs)
                results[family][n]=result
                records[family].append(dict(status='computed',**_scalar_level(result)))
            except (FloatingPointError,ValueError,np.linalg.LinAlgError) as error:
                records[family].append(dict(n=n,status='unresolved',reason=str(error)))
        pair_records=[]
        for a,b in zip(levels,levels[1:]):
            if a not in results[family] or b not in results[family]:
                pair_records.append(dict(coarse=a,fine=b,passed=False,reason='Missing unresolved level'))
            else:
                c,f=results[family][a],results[family][b]
                check=_moment_comparison(c,f)
                if family=='Simpson':
                    intrinsic=compare_grids(c,f)
                    check['intrinsic']=intrinsic;check['passed']=check['passed'] and intrinsic['passed']
                pair_records.append(dict(coarse=a,fine=b,**check))
        comparisons[family]=pair_records
    finalS=results['Simpson'].get(plan.simpson[-1]);finalG=results['Gauss'].get(plan.gauss[-1])
    cross=dict(passed=False,reason='Missing final integration level')
    partial=dict(passed=False,reason='Missing final integration level')
    if finalS is not None and finalG is not None:
        cross=_moment_comparison(finalS,finalG)
        try:
            cdf=gauss_partial_CDF(loglik,finalS['unit_quantiles'],n=plan.gauss[-1],
                log_evidence=finalG['log_evidence'],chunk=plan.chunk)
            error=float(np.max(abs(cdf-np.array([.05,.5,.95]))))
            partial=dict(passed=error<=.005,maximum_error=error,values=cdf.tolist(),
                thresholds=np.asarray(finalS['unit_quantiles']).tolist(),checks=6)
        except (FloatingPointError,ValueError) as error:
            partial=dict(passed=False,reason=str(error))
    qualified=all(comparisons[f][-1]['passed'] for f in comparisons) and cross['passed'] and partial['passed']
    diagnostics=dict(schema='surface-p2-inference-v01',qualified=bool(qualified),levels=records,
        successive=comparisons,cross_family=cross,independent_partial_CDF=partial,
        gates=dict(logZ=.01,unit_mean=.001,Simpson_CDF=.005,Simpson_quantile=.002,
            prediction_absolute_m=2e-6,prediction_relative_SD=.01,partial_CDF=.005),
        scope='Conditional finite linear model numerical integration only; physics/catalog admission separate; no action on unqualified posterior',
        no_deterministic_quadrature_ESS=True)
    if finalS is not None:
        diagnostics['unit_boundary_strip_width']=1/32
        diagnostics['unit_boundary_strip_probability']=(finalS['unit_CDF'][:,1]+1-finalS['unit_CDF'][:,-2]).tolist()
    return dict(qualified=bool(qualified),diagnostics=diagnostics,local_final_simpson=finalS,
        predictive_mean=None if finalS is None else finalS['predictive_mean'],
        predictive_SD=None if finalS is None else finalS['predictive_SD'])


def infer(model, observations, target_labels, *, plan=IntegrationPlan()):
    return qualify_integrals(model.loglikelihood(observations),model.select(target_labels),plan=plan)


def cache_storage_estimate(*,channels=24,loads=10,plan=IntegrationPlan(),modes=2976):
    """Optional-cache budget ONLY: no allocation/cache execution.

    A single finest Simpson table serves all nested dyadic levels; Gauss
    tables are independent. This raw-table estimate excludes baseline RSS,
    modal construction workspace, integration grids and BLAS temporaries.
    Caller must enforce total RSS<=4GiB, not treat this estimate as admission.
    """
    finest=plan.simpson[-1]
    if any((finest-1)%(n-1) for n in plan.simpson):raise ValueError('Non-nested Simpson levels')
    tables=(finest**2+sum(n*n for n in plan.gauss))*channels*loads*8
    coefficients=channels*loads*modes*8
    return dict(raw_response_tables_bytes=tables,modal_coefficients_bytes=coefficients,
        total_counted_bytes=tables+coefficients,RSS_limit_bytes=4*1024**3,
        real_cache_execution_authorized=False)
