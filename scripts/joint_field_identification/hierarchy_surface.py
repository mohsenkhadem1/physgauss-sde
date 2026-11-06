"""Surface-track correction of the finite hierarchy; historical references unchanged.

Gaussian force amplitudes are integrated conditional on spatial hypothesis and
one of five equiprobable load scales. Tau is a finite prior, not quadrature for
an asserted continuous hyperprior. Updates retain the full exact mixture.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.special import ndtr
from . import surface as bank
from .inference import Posterior,fit_posterior

ROOT=bank.ROOT
BASE=ROOT/'runs/joint-field-hierarchy-surface-v05'
REPORT=ROOT/'reports/joint-field-hierarchy-surface-v05'
SCALES=np.geomspace(.5,2.,5)
SCHEDULES={'probes':[10,12,13,14],'views':[0,2,4,9]}
MODELS=['hierarchical','fixed-nominal','variance-matched','oracle-scale']


def logsumexp(x):
    x=np.asarray(x);m=x.max();return float(m+np.log(np.exp(x-m).sum()))


def fit_scale_mixture(H,y,K,noise,scales=SCALES):
    """Theta-major, scale-minor flattened exact joint mixture; return conditionals."""
    conditional=[fit_posterior(H,y,K*s*s,noise) for s in scales]
    logmass=np.array([p.log_evidence for p in conditional])-np.log(len(scales))
    evidence=logsumexp(logmass);scale_weights=np.exp(logmass-evidence)
    weights=np.stack([p.weights*w for p,w in zip(conditional,scale_weights)],axis=1).reshape(-1)
    means=np.stack([p.means for p in conditional],axis=1).reshape(-1,H.shape[-1])
    cov=np.stack([p.covariances for p in conditional],axis=1).reshape(-1,H.shape[-1],H.shape[-1])
    return Posterior(weights,means,cov,evidence),conditional


def aggregate_spatial_moments(p,nspatial,nscale):
    """Exact first two conditional moments within theta, for linear predictions.

    NOT a Gaussian approximation used for subsequent updates or mixture intervals.
    Zero-mass theta rows have harmless zero moments because their weight is zero.
    """
    if nscale==1:return p
    w=p.weights.reshape(nspatial,nscale);mass=w.sum(1)
    normalized=np.divide(w,mass[:,None],out=np.zeros_like(w),where=mass[:,None]>0)
    mu=p.means.reshape(nspatial,nscale,-1)
    C=p.covariances.reshape(nspatial,nscale,mu.shape[-1],mu.shape[-1])
    mean=np.einsum('hs,hsf->hf',normalized,mu)
    centered=mu-mean[:,None]
    covariance=np.einsum('hs,hsfg->hfg',normalized,C)+np.einsum('hs,hsf,hsg->hfg',normalized,centered,centered)
    return Posterior(mass,mean,covariance,p.log_evidence)


def gaussian_mixture_quantiles(weights,means,variances,probabilities=(.05,.95)):
    """Exact marginal mixture CDF inversion to floating-point bisection accuracy."""
    active=weights>0;w=weights[active];mu=means[active];sd=np.sqrt(np.maximum(variances[active],0))
    lo0=(mu-12*sd).min(0);hi0=(mu+12*sd).max(0);out=[]
    for q in probabilities:
        lo,hi=lo0.copy(),hi0.copy()
        for _ in range(55):
            mid=(lo+hi)/2;z=np.divide(mid-mu,sd,out=np.zeros_like(mu),where=sd>0)
            cdf=w@np.where(sd>0,ndtr(z),(mid>=mu).astype(float))
            lo=np.where(cdf<q,mid,lo);hi=np.where(cdf>=q,mid,hi)
        out.append((lo+hi)/2)
    return np.stack(out)


def discrete_quantiles(weights,values,probabilities=(.05,.95)):
    """Finite weighted atoms; preserve ties and actual endpoint mass."""
    values=np.asarray(values)
    if values.ndim==1:values=values[:,None]
    order=np.argsort(values,axis=0);v=np.take_along_axis(values,order,axis=0)
    cumulative=np.cumsum(weights[order],axis=0);out=[]
    for q in probabilities:
        index=np.argmax(cumulative>=q,axis=0)
        # Guard a final cumulative sum just below one through rounding.
        index=np.where(cumulative[-1]<q,len(weights)-1,index)
        out.append(v[index,np.arange(values.shape[1])])
    return np.stack(out)


def update(p,m,action,value,nscale):
    a=m['actions'][action]
    h=np.repeat(a['h'],nscale,axis=0) if nscale>1 else a['h']
    offset=np.repeat(a['offset'],nscale) if nscale>1 else a['offset']
    return p.update_scalar(h,value,bank.NOISE**2,offset)


def force_covariance_parts(p):
    mean=p.weights@p.means;centered=p.means-mean
    conditional=np.einsum('h,hfg->fg',p.weights,p.covariances)
    between=np.einsum('h,hf,hg->fg',p.weights,centered,centered)
    return dict(conditional_force_covariance_trace_N2=float(np.trace(conditional)),
                between_component_force_covariance_trace_N2=float(np.trace(between)),
                total_force_covariance_trace_N2=float(np.trace(conditional+between)))


def metrics(p,data,m,truth,case,nscale,model_scale=None):
    n=len(data['theta']);agg=aggregate_spatial_moments(p,n,nscale)
    pred=agg.target(m['T'],m['offset']);err=pred.mean-truth['target'][case];split=m['split']
    material=agg.weights@data['material'];force=p.weights@p.means
    result=dict(rmse_mm=float(np.sqrt(np.mean(err**2))*1000),
        hidden_rmse_mm=float(np.sqrt(np.mean(err[:split]**2))*1000),
        changed_load_rmse_mm=float(np.sqrt(np.mean(err[split:]**2))*1000),
        predictive_sd_mm=float(np.sqrt(pred.risk)*1000),
        material_relative_rmse=float(np.sqrt(np.mean((material-truth['material'][case])**2)/np.mean(truth['material'][case]**2))),
        force_rmse_N=float(np.sqrt(np.mean((force-truth['force'][case])**2))),
        support_center_error_mm=float(abs(agg.weights@data['theta'][:,2]-truth['theta'][case,2])*1000),
        load_center_error_mm=float(abs(agg.weights@data['theta'][:,4]-truth['theta'][case,4])*1000),
        component_ess=float(1/np.sum(p.weights**2)),spatial_hypothesis_ess=float(1/np.sum(agg.weights**2)),
        **force_covariance_parts(p))
    # Twelve fixed hidden coordinates retain full theta/tau force mixture.
    select_hidden=np.linspace(0,split-1,12,dtype=int)
    Ts=m['T'][:,select_hidden]
    T=np.repeat(Ts,nscale,axis=0) if nscale>1 else Ts
    mu=np.einsum('hdf,hf->hd',T,p.means)
    var=np.einsum('hdf,hfg,hdg->hd',T,p.covariances,T,optimize=True)
    loH,hiH=gaussian_mixture_quantiles(p.weights,mu,var)
    # Reset responses and material are deterministic within theta: no Gaussian
    # moment-matching is used to compute these finite-mixture intervals.
    select_changed=np.linspace(split,len(err)-1,12,dtype=int)
    loC,hiC=discrete_quantiles(agg.weights,m['offset'][:,select_changed])
    loE,hiE=discrete_quantiles(agg.weights,data['material'])
    loF,hiF=gaussian_mixture_quantiles(p.weights,p.means,np.diagonal(p.covariances,axis1=1,axis2=2))
    htrue=truth['target'][case,select_hidden];ctrue=truth['target'][case,select_changed]
    coverH=(htrue>=loH)&(htrue<=hiH);coverC=(ctrue>=loC)&(ctrue<=hiC)
    result.update(coverage90=float(np.r_[coverH,coverC].mean()),hidden_coverage90=float(coverH.mean()),
        changed_coverage90=float(coverC.mean()),interval_width_mm=float(np.r_[hiH-loH,hiC-loC].mean()*1000),
        hidden_interval_width_mm=float((hiH-loH).mean()*1000),changed_interval_width_mm=float((hiC-loC).mean()*1000),
        material_coverage90=float(((truth['material'][case]>=loE)&(truth['material'][case]<=hiE)).mean()),
        material_interval_width_Pa=float((hiE-loE).mean()),
        material_interval_relative_width=float((hiE-loE).mean()/truth['material'][case].mean()),
        force_coverage90=float(((truth['force'][case]>=loF)&(truth['force'][case]<=hiF)).mean()),
        force_interval_width_N=float((hiF-loF).mean()))
    # Continuous-valued summaries of the exact finite theta/tau posterior.
    logE=np.repeat(data['theta'][:,0],nscale)
    tau=np.tile(SCALES,n) if nscale>1 else np.full(n,float(model_scale))
    logtau=np.log(tau);Ecenter=logE-p.weights@logE;Tcenter=logtau-p.weights@logtau
    Evar=float(p.weights@Ecenter**2);Tvar=float(p.weights@Tcenter**2)
    correlation=None if Tvar<1e-20 or Evar<1e-20 else float(p.weights@(Ecenter*Tcenter)/np.sqrt(Evar*Tvar))
    tauw=np.bincount(np.tile(np.arange(nscale),n),weights=p.weights,minlength=nscale) if nscale>1 else np.ones(1)
    tauvalues=SCALES if nscale>1 else np.array([model_scale])
    loTau,hiTau=discrete_quantiles(tauw,tauvalues)
    result.update(logE_logtau_correlation=correlation,posterior_tau_mean=float(p.weights@tau),
        posterior_tau_weights=tauw.tolist(),posterior_tau_values=tauvalues.tolist(),
        tau_interval90=[float(loTau[0]),float(hiTau[0])],
        tau_covered90=bool(loTau[0]<=truth['tau'][case]<=hiTau[0]),
        tau_endpoint_mass=float(tauw[0]+tauw[-1]) if nscale>1 else None,
        material_scale_endpoint_mass=float(agg.weights[np.isclose(data['theta'][:,0],bank.BOUNDS[0,0])|np.isclose(data['theta'][:,0],bank.BOUNDS[0,1])].sum()),
        support_stiffness_endpoint_mass=float(agg.weights[np.isclose(data['theta'][:,3],bank.BOUNDS[3,0])|np.isclose(data['theta'][:,3],bank.BOUNDS[3,1])].sum()))
    # Spatial traction-profile error keeps load-location/force dependence.
    x=np.linspace(-.06,.06,41);profile=np.exp(-.5*((x[None]-data['theta'][:,4,None])/.018)**2)
    profile/=np.trapezoid(profile,x,axis=1)[:,None]
    field=np.einsum('h,hx,htd->txd',agg.weights,profile,agg.means.reshape(n,8,2),optimize=True)
    trueprofile=np.exp(-.5*((x-truth['theta'][case,4])/.018)**2);trueprofile/=np.trapezoid(trueprofile,x)
    truefield=truth['force'][case].reshape(8,2)[:,None,:]*trueprofile[None,:,None]
    result['force_profile_rmse_N_per_m']=float(np.sqrt(np.mean((field-truefield)**2)))
    return result


def generate(data,m,shape,protocol_hash):
    seed=151001+(shape=='step');rng=np.random.default_rng(seed);n=64
    index=rng.integers(len(data['theta']),size=n);scale_index=rng.integers(5,size=n);tau=SCALES[scale_index]
    force=(rng.standard_normal((n,16))@np.linalg.cholesky(bank.load_cov()).T)*tau[:,None]
    y=np.einsum('hof,hf->ho',m['H'][index],force)+rng.normal(scale=bank.NOISE,size=(n,24))
    target=np.einsum('hdf,hf->hd',m['T'][index],force)+m['offset'][index]
    measurements=np.stack([np.einsum('hf,hf->h',a['h'][index],force)+a['offset'][index] for a in m['actions']],axis=1)
    measurements+=rng.normal(scale=bank.NOISE,size=measurements.shape)
    return dict(index=index,scale_index=scale_index,tau=tau,force=force,y=y,target=target,
                measurements=measurements,theta=data['theta'][index],material=data['material'][index],
                seed=np.array(seed),protocol_sha256=np.array(protocol_hash))


def self_tests():
    rng=np.random.default_rng(861);H=rng.normal(size=(3,4,2));y=rng.normal(size=4);K=np.array([[.8,.1],[.1,.4]])
    p,cond=fit_scale_mixture(H,y,K,.1);scales=5
    assert np.isclose(p.weights.sum(),1)
    for i,s in enumerate(SCALES):
        direct=fit_posterior(H,y,K*s*s,.1)
        np.testing.assert_allclose(p.means.reshape(3,5,2)[:,i],direct.means,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(p.covariances.reshape(3,5,2,2)[:,i],direct.covariances,rtol=1e-13,atol=1e-13)
        w=p.weights.reshape(3,5)[:,i];np.testing.assert_allclose(w/w.sum(),direct.weights,rtol=1e-13)
    # Oracle scale=1 is exactly the fixed nominal estimator, and scale evidence
    # includes the changing determinant rather than uniform posterior weights.
    nominal=fit_posterior(H,y,K,.1);np.testing.assert_allclose(cond[2].weights,nominal.weights,atol=1e-14)
    T=rng.normal(size=(3,7,2));offset=rng.normal(size=(3,7));agg=aggregate_spatial_moments(p,3,5)
    full=p.target(np.repeat(T,5,axis=0),np.repeat(offset,5,axis=0));reduced=agg.target(T,offset)
    np.testing.assert_allclose(full.mean,reduced.mean,atol=1e-13)
    np.testing.assert_allclose(full.variance,reduced.variance,atol=1e-13)
    h=rng.normal(size=(3,2));d=rng.normal(size=3)
    updated=p.update_scalar(np.repeat(h,5,axis=0),.4,.2,np.repeat(d,5))
    conditional_updated=[q.update_scalar(h,.4,.2,d) for q in cond]
    for i in range(5):
        np.testing.assert_allclose(updated.means.reshape(3,5,2)[:,i],conditional_updated[i].means,atol=1e-13)
    probe=p.update_scalar(np.zeros_like(p.means),.2,.1,np.repeat(d,5))
    np.testing.assert_allclose(probe.means,p.means,atol=0);np.testing.assert_allclose(probe.covariances,p.covariances,atol=0)
    # Broader baseline matches exactly the mixture prior's second moment.
    multiplier=float(np.sqrt(np.mean(SCALES**2)))
    np.testing.assert_allclose(np.mean([K*s*s for s in SCALES],axis=0),K*multiplier**2,atol=1e-15)
    q=discrete_quantiles(np.array([.1,.8,.1]),np.array([1.,2.,3.]));np.testing.assert_allclose(q[:,0],[1.,3.])
    qg=gaussian_mixture_quantiles(np.ones(1),np.zeros((1,1)),np.ones((1,1)))
    np.testing.assert_allclose(qg[:,0],[-1.64485362695147,1.64485362695147],atol=1e-12)
    # Correction changes sensors and their projection, never the physical prior
    # or numerical inference. The finite numerical core remains AST-identical.
    import ast
    from . import bank as historical_bank
    np.testing.assert_array_equal(bank.BOUNDS,historical_bank.BOUNDS)
    np.testing.assert_array_equal(bank.grid(5),historical_bank.grid(5))
    np.testing.assert_allclose(np.abs(bank.points()[:,1]),.03,rtol=0,atol=0)
    for action in bank.action_specs():
        e=np.array(action['projection']);camera=np.array(action['camera_direction'])
        assert abs(e@camera)<1e-14 and abs(np.linalg.norm(e)-1)<1e-14
        normal=np.array([0.,np.sign(bank.points()[action['point'],1]),0.])
        assert normal@camera>0,'Camera must face the declared surface side'
    names=['logsumexp','fit_scale_mixture','aggregate_spatial_moments','gaussian_mixture_quantiles',
           'discrete_quantiles','update','force_covariance_parts','metrics']
    def functions(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
    original=functions(Path(__file__).with_name('hierarchy.py'));corrected=functions(Path(__file__))
    assert all(original[name]==corrected[name] for name in names)
    return dict(passed=True,checks=['exact_force_conditionals_at_each_scale','oracle_fixed_scale_equivalence',
        'theta_tau_evidence_normalization','aggregated_linear_moments_equal_full_mixture',
        'camera_conditional_updates_equal_explicit_components','reset_probe_preserves_force_conditionals',
        'variance_matched_prior_second_moment','finite_atom_quantiles','Gaussian_quantiles',
        'same_physical_parameter_prior_as_v02','all_landmarks_on_y_boundary',
        'camera_projection_perpendicular_to_view_and_faces_surface','numerical_core_AST_identical_to_v02'],
        variance_matched_std_multiplier=multiplier,discrete_mean_tau_squared=float(np.mean(SCALES**2)))


def freeze():
    REPORT.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
    tests=self_tests();previous=json.loads((ROOT/'reports/joint-field-identification-v01/protocol.json').read_text())
    sources={**previous['source_sha256'],str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             str(Path(bank.__file__).relative_to(ROOT)):hashlib.sha256(Path(bank.__file__).read_bytes()).hexdigest()}
    surface_banks={shape:hashlib.sha256((ROOT/f'runs/joint-field-surface-v04/{shape}.npz').read_bytes()).hexdigest() for shape in ['block','step']}
    for shape in ['block','step']:
        data=dict(np.load(ROOT/f'runs/joint-field-surface-v04/{shape}.npz'))
        np.testing.assert_array_equal(data['theta'],bank.grid(5))
        np.testing.assert_array_equal(data['points'],bank.points())
    for path,digest in sources.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    plan=dict(version='joint-field-hierarchy-surface-v05',case_counts={'block':64,'step':64},seeds={'block':151001,'step':151002},
              spatial_prior='Uniform existing3125spatialdictionary, independently sampled from discrete load scale.',
              tau_values=SCALES.tolist(),tau_weights=(np.ones(5)/5).tolist(),
              force_std_base_N=bank.FORCE_STD,force_correlation_length=.30,
              variance_matched_std_multiplier=tests['variance_matched_std_multiplier'],
              variance_matched_base_std_N=bank.FORCE_STD*tests['variance_matched_std_multiplier'],
              schedules=SCHEDULES,models=MODELS,observation_noise_m=bank.NOISE,
              known_probe_force_N=bank.PROBE_FORCE,initial_observations=24,
              prediction_target='Identical v01 never-acquired192original+192changedload coordinates; equal coordinate weights.',
              intervals='90%equal-tail marginal mixture intervals:12hidden+12changedpredictioncoordinates,20materialqueries,16forcecoefficients; no simultaneous or real-world claim.',
              source_sha256=sources,bank_sha256=surface_banks,
              sensor_points_m=bank.points().tolist(),camera_action_specs=bank.action_specs(),
              correction='Tracks on y=+/-0.03m boundary, not old interior+/-0.025m. Back observations and front side-probes use valid oblique image-plane projection (x+y)/sqrt2. Physical priors, force scales, schedules and inference unchanged.',
              bootstrap='2048pairedcase resamples within each geometry, seed63001; no shape-population inference.',
              scope='Fresh surface-track finite-prior hierarchy only; no raw RGB inversion or continuous-tau identification claim. Known boundary correspondences and orthographic camera projections; visibility audited separately. Fourfixed calibrated or camera measurements; no action reselection. Signedtractions, quasistatic no gravity/contact.')
    path=REPORT/'protocol.json';text=json.dumps(plan,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:raise RuntimeError('Hierarchy protocol already frozen with different source/config')
    path.write_text(text);(REPORT/'tests.json').write_text(json.dumps(tests,indent=2)+'\n')
    print('FROZEN',hashlib.sha256(text.encode()).hexdigest(),flush=True)


def verify_protocol():
    path=REPORT/'protocol.json';protocol=json.loads(path.read_text())
    for name,digest in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+name)
    return protocol,hashlib.sha256(path.read_bytes()).hexdigest()


def run(shape):
    protocol,protocol_hash=verify_protocol();source=ROOT/f'runs/joint-field-surface-v04/{shape}.npz'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['bank_sha256'][shape]
    data=dict(np.load(source));m=bank.maps(data);n=len(data['theta'])
    raw=BASE/shape;raw.mkdir(exist_ok=True);out=REPORT/shape;out.mkdir(exist_ok=True)
    truthpath=raw/'truth.npz'
    if truthpath.exists():
        truth=dict(np.load(truthpath));assert str(truth['protocol_sha256'])==protocol_hash
    else:
        truth=generate(data,m,shape,protocol_hash);np.savez(truthpath,**truth)
    broader=protocol['variance_matched_std_multiplier']
    for case in range(64):
        dest=out/f'case-{case:03d}.json'
        if dest.exists():
            assert json.loads(dest.read_text())['protocol_sha256']==protocol_hash;continue
        began=time.time();p,conditionals=fit_scale_mixture(m['H'],truth['y'][case],bank.load_cov(),bank.NOISE**2)
        states={'hierarchical':(p,5,None),
                'fixed-nominal':(conditionals[2],1,1.),
                'variance-matched':(fit_posterior(m['H'],truth['y'][case],bank.load_cov(broader),bank.NOISE**2),1,broader),
                'oracle-scale':(conditionals[int(truth['scale_index'][case])],1,float(truth['tau'][case]))}
        rows=[]
        for model,(original,nscale,scale) in states.items():
            rows.append(dict(model=model,condition='initial',measurements=0,
                             **metrics(original,data,m,truth,case,nscale,scale)))
            for name,actions in SCHEDULES.items():
                post=original
                for a in actions:post=update(post,m,a,truth['measurements'][case,a],nscale)
                rows.append(dict(model=model,condition=name,measurements=4,actions=actions,
                                 **metrics(post,data,m,truth,case,nscale,scale)))
        doc=dict(shape=shape,case=case,truth_tau=float(truth['tau'][case]),
                 protocol_sha256=protocol_hash,rows=rows,seconds=time.time()-began)
        dest.write_text(json.dumps(doc,indent=2)+'\n');print(shape,case,round(doc['seconds'],2),flush=True)
    verify_protocol();print('COMPLETE',shape,flush=True)


def summarize():
    protocol,protocol_hash=verify_protocol();docs=[]
    for shape in ['block','step']:
        for case in range(64):
            d=json.loads((REPORT/shape/f'case-{case:03d}.json').read_text());assert d['protocol_sha256']==protocol_hash;docs.append(d)
    keys=['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','force_profile_rmse_N_per_m',
          'coverage90','hidden_coverage90','changed_coverage90','interval_width_mm','material_coverage90',
          'material_interval_relative_width','force_coverage90','force_interval_width_N','material_scale_endpoint_mass',
          'support_stiffness_endpoint_mass','support_center_error_mm','load_center_error_mm']
    means=[];paired=[];rng=np.random.default_rng(63001)
    boot=np.concatenate([rng.integers(0,64,size=(2048,64)),64+rng.integers(0,64,size=(2048,64))],axis=1)
    def get(model,condition):return [next(r for r in d['rows'] if r['model']==model and r['condition']==condition) for d in docs]
    for condition in ['initial','views','probes']:
        for model in MODELS:
            rows=get(model,condition)
            entry=dict(model=model,condition=condition,cases=128,**{k:float(np.mean([r[k] for r in rows])) for k in keys})
            if model=='hierarchical':
                entry.update(mean_logE_logtau_correlation=float(np.mean([r['logE_logtau_correlation'] for r in rows])),
                             mean_tau_endpoint_mass=float(np.mean([r['tau_endpoint_mass'] for r in rows])),
                             tau_coverage90=float(np.mean([r['tau_covered90'] for r in rows])))
            means.append(entry)
        for control in MODELS[1:]:
            ours=get('hierarchical',condition);other=get(control,condition)
            for metric in ['rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90']:
                difference=np.array([a[metric]-b[metric] for a,b in zip(ours,other)])
                low,high=np.quantile(difference[boot].mean(1),[.025,.975])
                paired.append(dict(condition=condition,control=control,metric=metric,
                                   mean_hierarchy_minus_control=float(difference.mean()),paired_bootstrap95=[float(low),float(high)]))
    summary=dict(protocol_sha256=protocol_hash,cases=128,means=means,paired_comparisons=paired,
                 scope=protocol['scope'],variance_matched_std_multiplier=protocol['variance_matched_std_multiplier'])
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Surface-track hierarchical force-scale reference: fresh128case finite-prior cohort','',
           'Five equiprobable log-spaced scale values; exact finite mixture, not a continuous hyperprior.',
           'Same original observations and four fixed calibrated-probe or camera measurements for every method.',
           'No inference tuning or action reselection on this cohort. Oracle receives only true load-prior scale.',
           'Gaussian broader control matches exact mixture prior variance: std multiplier %.6f.'%protocol['variance_matched_std_multiplier'],'',
           'condition | model | prediction mm | changed-load mm | material relative RMSE | force N | pred90 coverage | material90 | force90']
    for r in means:lines.append(f"{r['condition']} | {r['model']} | {r['rmse_mm']:.5f} | {r['changed_load_rmse_mm']:.5f} | {r['material_relative_rmse']:.4f} | {r['force_rmse_N']:.5f} | {r['coverage90']:.3f} | {r['material_coverage90']:.3f} | {r['force_coverage90']:.3f}")
    lines+=['','Intervals are coordinate marginals; discrete fields/scale have atoms and may be conservative.',
            'Endpoint mass must be compared with the finite prior endpoint mass0.4, not treated automatically as failure.',
            'Full widths, paired case-bootstrap intervals, covariance decomposition and joint stiffness/scale diagnostics are in JSON.',
            'No claim of recovered unique physical truth, continuous-tau calibration, real-video inference or computational speed superiority.',
            'Historical interior-point v01/v02 results are retained as mathematical sensor references, not visible RGB landmarks.',
            'Force-profile metric is a normalized longitudinal load-basis proxy, not the complete stepped-side physical traction profile.']
    (REPORT/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--shape',choices=['block','step']);ap.add_argument('--summarize',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(json.dumps(self_tests(),indent=2))
    elif args.freeze:freeze()
    elif args.shape:run(args.shape)
    elif args.summarize:summarize()
    else:ap.error('Choose --self-test, --freeze, --shape or --summarize')
