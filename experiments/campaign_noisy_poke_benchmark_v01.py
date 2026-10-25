"""Pure fixed-grid/noise stress helpers; no draws or I/O on import.

Four noise replicates are paired trials, not32independent objects. Predictive
mean±2SD bands are moment bands with NO nominal Gaussian coverage assertion.
Material intervals invert the full category-mixture continuous intrinsic CDF.
"""
import hashlib,json
import numpy as np
from scipy.optimize import brentq
from experiments import campaign_fixture_balanced_held_v01 as physics
from experiments.campaign_surface_conditional_amplitude_v01 import BOUNDS,LOG_WIDTH
PREFIX_SEED=2026103302;POKE_SEED=2026103303;REPLICATES=4
SOURCE='experiments/campaign_noisy_poke_benchmark_v01.py';TEST='tests/test_noisy_poke_benchmark_v01.py'

def objects():
    return [dict(index=2*k+j,layout=spec['id'],E_Pa=E,K_N_m=float(np.sqrt(150.)))for k,spec in enumerate(physics.LAYOUTS)for j,E in enumerate((2500.,6500.))]
def trials():return [dict(index=8*r+i,replicate=r,object_index=i)for r in range(4)for i in range(8)]
def noise_streams():
    return dict(prefix=np.random.default_rng(PREFIX_SEED).normal(0,physics.SIGMA_M,(4,8,4)),poke=np.random.default_rng(POKE_SEED).normal(0,physics.SIGMA_M,(4,8,8)))
def fingerprint(value):return hashlib.sha256(np.ascontiguousarray(value,dtype=np.float64).tobytes()).hexdigest()
def fixed_action(development_rows):
    if len(development_rows)!=36 or{r['index']for r in development_rows}!=set(range(36))or any(r.get('ranking',{}).get('qualified')is not True for r in development_rows):raise ValueError('All36 completed qualified DEVELOPMENT rankings')
    if any([a.get('action')for a in r['ranking']['actions']]!=list(range(8))for r in development_rows):raise ValueError('Exact ordered candidate action IDs0..7')
    risks=np.array([[a['risk']for a in r['ranking']['actions']]for r in sorted(development_rows,key=lambda r:r['index'])])
    if risks.shape!=(36,8)or not np.isfinite(risks).all()or np.any(risks<0):raise ValueError('All8 finite predicted development risks')
    means=risks.mean(axis=0);return dict(action=int(np.argmin(means)),mean_predicted_risk=means.tolist(),selection='Minimum mean predicted held risk over all completed balanced36 DEVELOPMENT rows; deterministic lowerindex tie; frozen before new noisy data')
def prefix_only(value):
    a=np.asarray(value,float)
    if a.shape!=(4,)or not np.isfinite(a).all():raise ValueError('Exactly4 finite observed prefix channels')
    return a.copy()

def material_intervals(references,probability,*,outer=512,inner=512):
    """True continuous category-mixture central90% intervals, not quantile averages."""
    p=np.asarray(probability,float)
    if len(references)!=4 or p.shape!=(4,)or not np.isfinite(p).all()or np.any(p<0)or not np.isfinite(p.sum())or p.sum()<=0:raise ValueError('All4 positive category probabilities')
    p=p/p.sum();cdf_knots=np.empty((2,33));quantiles=np.empty((2,2))
    for d in range(2):
        def cdf(q):
            values=[]
            for ref in references:
                state=ref._state(outer,inner);values.append(ref.cdf(d,float(q),outer=outer,inner=inner,log_evidence=state['log_evidence']))
            out=float(p@values)
            if not np.isfinite(out)or not 0<=out<=1+32*np.finfo(float).eps:raise FloatingPointError('Unclipped continuous mixture CDF')
            return out
        cdf_knots[d]=[cdf(q)for q in np.linspace(0,1,33)]
        if np.any(np.diff(cdf_knots[d])<-1e-12):raise FloatingPointError('Nonmonotone continuous mixture CDF')
        quantiles[d]=[brentq(lambda q:cdf(q)-alpha,0,1,xtol=2e-8,rtol=1e-12)for alpha in(.05,.95)]
    return dict(unit_quantiles=quantiles,physical_quantiles=np.exp(np.log(BOUNDS[:,0,None])+LOG_WIDTH[:,None]*quantiles),unit_CDF=cdf_knots,method='Full category-mixture continuous intrinsic CDF; central90% interval; fixed-case inclusion is NOT nominal calibration/SBC')

def prediction_metrics(mean,SD,truth):
    mean,SD,truth=[np.asarray(v,float)for v in(mean,SD,truth)]
    if any(v.shape!=(48,)or not np.isfinite(v).all()for v in(mean,SD,truth))or np.any(SD<0):raise ValueError('48 finite clean latent targets and posterior SD')
    mse=float(np.mean((mean-truth)**2));return dict(MSE_m2=mse,RMSE_m=float(np.sqrt(mse)),posterior_variance_m2=float(np.mean(SD**2)),posterior_RMS_SD_m=float(np.sqrt(np.mean(SD**2))),moment_band_2SD_inclusion_fraction=float(np.mean((truth>=mean-2*SD)&(truth<=mean+2*SD))),moment_band_2SD_mean_width_m=float(np.mean(4*SD)),moment_bands_have_no_nominal_coverage_level=True)
def material_metrics(interval,unit_mean,truth):
    q=np.asarray(interval['physical_quantiles'],float);u=np.asarray(unit_mean,float);t=np.asarray(truth,float)
    if q.shape!=(2,2)or u.shape!=(2,)or t.shape!=(2,)or not all(np.isfinite(v).all()for v in(q,u,t))or np.any(q[:,0]>q[:,1]):raise ValueError('Finite E/K interval/mean/truth')
    tu=(np.log(t)-np.log(BOUNDS[:,0]))/LOG_WIDTH
    return dict(central90_inclusion=((t>=q[:,0])&(t<=q[:,1])).tolist(),central90_width=(q[:,1]-q[:,0]).tolist(),unit_squared_error=((u-tu)**2).tolist(),interpretation='Fixed8interior design descriptive interval inclusion; not SBC/population90%calibration')
def compare_policies(before,after,adaptive,fixed):
    if len(after)!=8 or any(type(a)is not int or not 0<=a<8 for a in(adaptive,fixed)):raise ValueError('All8 reset counterfactual outcomes and frozen choices')
    out={}
    for name,indices in(('adaptive',[adaptive]),('fixed',[fixed]),('uniform_random',list(range(8)))):
        keys=('MSE_m2','RMSE_m','posterior_variance_m2','posterior_RMS_SD_m','moment_band_2SD_inclusion_fraction','moment_band_2SD_mean_width_m');r={k:float(np.mean([after[i][k]for i in indices]))for k in keys};r['RMSE_m']=float(np.sqrt(r['MSE_m2']));r['posterior_RMS_SD_m']=float(np.sqrt(r['posterior_variance_m2']));r.update(realized_MSE_improvement_m2=before['MSE_m2']-r['MSE_m2'],variance_down_error_up=bool(r['posterior_variance_m2']<before['posterior_variance_m2']and r['MSE_m2']>before['MSE_m2']),moment_bands_have_no_nominal_coverage_level=True);out[name]=r
    out['paired_adaptive_minus_fixed_MSE_m2']=out['adaptive']['MSE_m2']-out['fixed']['MSE_m2'];out['paired_adaptive_minus_random_MSE_m2']=out['adaptive']['MSE_m2']-out['uniform_random']['MSE_m2'];out['uniform_random_interpretation']='Exact mean8action losses for this frozenprefix+8independentpoke-noise realization; not noise-integrated or population expectation';return out

def aggregate(rows):
    if len(rows)!=32 or{r['index']for r in rows}!=set(range(32)):raise ValueError('All32 prescribed trials including unresolved')
    ordered=sorted(rows,key=lambda r:r['index'])
    if any(any(row.get(k)!=expected[k]for k in('index','object_index','replicate'))for row,expected in zip(ordered,trials())):raise ValueError('Every exact replicate/object trial identity, including unresolved')
    complete=all(r.get('qualified')is True for r in ordered);out=dict(planned_objects=8,planned_noise_replicates=4,planned_paired_trials=32,qualified_trials=sum(r.get('qualified')is True for r in rows),all32_retained=True,comparison_blocked=not complete,population_or_nominal_calibration_claim=False)
    if not complete:return out
    values=[]
    for i in range(8):
        group=[r for r in ordered if r['object_index']==i]
        if len(group)!=4 or{r['replicate']for r in group}!=set(range(4)):raise ValueError('Four fixed noise replicates per object')
        values.append({name:float(np.mean([r['policies'][name]for r in group]))for name in('paired_adaptive_minus_fixed_MSE_m2','paired_adaptive_minus_random_MSE_m2')})
    out.update(equal_object_mean_adaptive_minus_fixed_MSE_m2=float(np.mean([r['paired_adaptive_minus_fixed_MSE_m2']for r in values])),equal_object_mean_adaptive_minus_random_MSE_m2=float(np.mean([r['paired_adaptive_minus_random_MSE_m2']for r in values])),object_noise_averages=values,interpretation='All8equallyweighted fixedobjects; four withinobjectnoise repeats. Descriptive gridconditional/noise stress only, no population CI or adaptive-advantage generalization.')
    return out

def compare_material(before,after,adaptive,fixed):
    if len(after)!=8:raise ValueError('All8 material posterior metrics')
    result={}
    for name,ids in(('adaptive',[adaptive]),('fixed',[fixed]),('uniform_random',list(range(8)))):
        inclusion=np.mean([after[i]['central90_inclusion']for i in ids],axis=0);width=np.mean([after[i]['central90_width']for i in ids],axis=0);error=np.mean([after[i]['unit_squared_error']for i in ids],axis=0)
        result[name]=dict(central90_inclusion=inclusion.tolist(),central90_width=width.tolist(),unit_squared_error=error.tolist(),narrower_but_larger_parameter_error=((width<before['central90_width'])&(error>before['unit_squared_error'])).tolist(),fixedcase_descriptive_not_nominal_calibration=True)
    return result
