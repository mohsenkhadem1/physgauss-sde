"""Frozen temporal-prior stress follow-up; exact finite covariance mixture."""
from pathlib import Path
import argparse,hashlib,json,time
from datetime import datetime,timezone
import numpy as np
from . import surface as bank
from .inference import Posterior,fit_posterior
from .hierarchy_surface import (SCALES,SCHEDULES,logsumexp,aggregate_spatial_moments,
 gaussian_mixture_quantiles,discrete_quantiles,update,force_covariance_parts)
ROOT=bank.ROOT
BASE=ROOT/'runs/joint-field-temporal-v06'
REPORT=ROOT/'reports/joint-field-temporal-v06'
LENGTHS=np.array([.10,.30,.60])
HYPERS=np.array([(length,scale) for length in LENGTHS for scale in SCALES])
MODELS=['scale-correlation','amplitude-only','fixed-nominal','oracle']

def covariance(length,scale=1.):
 d=np.abs(bank.TIMES[:,None]-bank.TIMES[None,:]);r=np.sqrt(3)*d/length
 return bank.FORCE_STD**2*scale**2*np.kron((1+r)*np.exp(-r),np.eye(2))

def combine(conditional):
 mass=np.array([p.log_evidence for p in conditional])-np.log(len(conditional));evidence=logsumexp(mass)
 weights=np.stack([p.weights*np.exp(z-evidence) for p,z in zip(conditional,mass)],axis=1).reshape(-1)
 means=np.stack([p.means for p in conditional],axis=1).reshape(-1,conditional[0].means.shape[-1])
 cov=np.stack([p.covariances for p in conditional],axis=1).reshape(len(weights),means.shape[-1],means.shape[-1])
 return Posterior(weights,means,cov,evidence)

def fit(H,y,noise):
 conditional=[fit_posterior(H,y,covariance(length,scale),noise) for length,scale in HYPERS]
 return combine(conditional),conditional

def metrics(p,data,m,truth,case,nscale,model_scale=None,hyper_scales=None):
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
    tau=np.tile(np.asarray(hyper_scales),n) if nscale>1 else np.full(n,float(model_scale))
    logtau=np.log(tau);Ecenter=logE-p.weights@logE;Tcenter=logtau-p.weights@logtau
    Evar=float(p.weights@Ecenter**2);Tvar=float(p.weights@Tcenter**2)
    correlation=None if Tvar<1e-20 or Evar<1e-20 else float(p.weights@(Ecenter*Tcenter)/np.sqrt(Evar*Tvar))
    tauw=np.bincount(np.tile(np.arange(nscale),n),weights=p.weights,minlength=nscale) if nscale>1 else np.ones(1)
    tauvalues=np.asarray(hyper_scales) if nscale>1 else np.array([model_scale])
    unique,inverse=np.unique(tauvalues,return_inverse=True)
    tauw=np.bincount(inverse,weights=tauw,minlength=len(unique));tauvalues=unique
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


def self_test():
 rng=np.random.default_rng(260601);H=rng.normal(size=(3,4,16));y=rng.normal(size=4)*.03;noise=.0004
 p,cs=fit(H,y,noise);HH=np.repeat(H,15,axis=0);KK=np.tile(np.stack([covariance(l,s) for l,s in HYPERS]),(3,1,1))
 direct=fit_posterior(HH,y,KK,noise)
 errors={key:float(np.max(np.abs(getattr(p,key)-getattr(direct,key)))) for key in ['weights','means','covariances']}
 errors['evidence']=abs(p.log_evidence-direct.log_evidence)
 assert max(errors.values())<1e-10
 h=rng.normal(size=(3,16));hfull=np.repeat(h,15,axis=0);value=.04
 seq=p.update_scalar(hfull,value,noise)
 batch=fit_posterior(np.concatenate([HH,hfull[:,None]],axis=1),np.r_[y,value],KK,noise)
 for key in ['weights','means','covariances']:assert np.allclose(getattr(seq,key),getattr(batch,key),rtol=1e-9,atol=1e-10)
 probe=p.update_scalar(np.zeros_like(hfull),.02,noise,np.repeat(np.array([.01,.02,.03]),15));assert np.array_equal(probe.means,p.means);assert np.array_equal(probe.covariances,p.covariances)
 assert np.allclose(covariance(.30),bank.load_cov(),rtol=0,atol=0)
 for i,scale in enumerate(SCALES):assert np.array_equal(cs[i].means,fit_posterior(H,y,covariance(.10,scale),noise).means)
 assert np.array_equal(cs[7].means,fit_posterior(H,y,bank.load_cov(),noise).means)
 agg=aggregate_spatial_moments(p,3,15);T=rng.normal(size=(3,5,16));a=agg.target(T);b=p.target(np.repeat(T,15,axis=0));assert np.allclose(a.mean,b.mean);assert np.allclose(a.variance,b.variance)
 # Duplicate scale atoms sum correctly across correlation hypotheses.
 w=p.weights.reshape(3,15).sum(0);unique,inv=np.unique(HYPERS[:,1],return_inverse=True);m=np.bincount(inv,weights=w);assert abs(m.sum()-1)<1e-12
 REPORT.mkdir(parents=True,exist_ok=True)
 doc=dict(passed=True,independent_explicit45component_identity_errors=errors,checks=['sequential_vs_augmented_batch','probe_preserves_force_conditionals','nominal_covariance_exact_identity','oracle_conditional_equivalence','spatial_total_covariance_matches_fullmixture','duplicate_scale_atom_aggregation'])
 (REPORT/'tests.json').write_text(json.dumps(doc,indent=2)+'\n');return doc

def source_hashes():
 names=['temporal_hierarchy.py','temporal_hierarchy_report.py','hierarchy_surface.py','surface.py','bank.py','mechanics.py','inference.py']
 return {str((Path(__file__).parent/n).relative_to(ROOT)):hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names}

def freeze():
 self_test();BASE.mkdir(parents=True,exist_ok=True)
 plan=dict(version='joint-field-temporal-v06',posthoc_followup=True,motivation='v04 temporal-prior failure; fresh cases but not untouched primary test',frozen_utc=datetime.now(timezone.utc).isoformat(),deadline_utc='2026-11-07T07:37:29+00:00',cases_per_shape=32,seeds={'block':261001,'step':261002},true_correlation=.10,
 scales=SCALES.tolist(),correlation_prior=LENGTHS.tolist(),correlation_time_units='Normalized bank.TIMES spanning0..1',oracle_scope='Known true amplitude scale and correlation length only; theta and force amplitudes remain inferred',hyper_prior='Independent uniform discrete scales and lengths',
 schedules=SCHEDULES,models=MODELS,source_sha256=source_hashes(),bank_sha256={s:hashlib.sha256((ROOT/f'runs/joint-field-surface-v04/{s}.npz').read_bytes()).hexdigest() for s in ['block','step']},
 scope='Fresh64case conditional temporal-prior stress test, not matched-prior calibration: true correlation fixed0.10. Known geometry/surface projections. No action reselection, no changes to v04/v05. All cases retained; exact finite mixture, no continuous hyperprior.',
 noise_std_m=bank.NOISE,force_std_N=bank.FORCE_STD,probe_force_N=bank.PROBE_FORCE,
 metrics='Same full target and marginal interval coordinates as v05. Force profile is normalized longitudinal basis proxy, not complete surface traction.',bootstrap='2000 paired case resamples stratified by shape; seed260603')
 path=REPORT/'protocol.json';assert not path.exists(),'Protocol already frozen';path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(hashlib.sha256(path.read_bytes()).hexdigest())

def verify():
 path=REPORT/'protocol.json';p=json.loads(path.read_text());assert p['source_sha256']==source_hashes();return p,hashlib.sha256(path.read_bytes()).hexdigest()

def generate(data,m,shape,sha,protocol):
 rng=np.random.default_rng(protocol['seeds'][shape]);n=32;index=rng.integers(len(data['theta']),size=n);scale_index=rng.integers(5,size=n);tau=SCALES[scale_index]
 force=(rng.standard_normal((n,16))@np.linalg.cholesky(covariance(.10)).T)*tau[:,None]
 y=np.einsum('hof,hf->ho',m['H'][index],force)+rng.normal(scale=bank.NOISE,size=(n,24))
 target=np.einsum('hdf,hf->hd',m['T'][index],force)+m['offset'][index]
 measurements=np.stack([np.einsum('hf,hf->h',a['h'][index],force)+a['offset'][index] for a in m['actions']],axis=1)
 measurements+=rng.normal(scale=bank.NOISE,size=measurements.shape)
 return dict(index=index,scale_index=scale_index,tau=tau,force=force,y=y,target=target,measurements=measurements,theta=data['theta'][index],material=data['material'][index],seed=np.array(protocol['seeds'][shape]),protocol_sha256=np.array(sha))

def run(shape):
 protocol,sha=verify();source=ROOT/f'runs/joint-field-surface-v04/{shape}.npz';assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['bank_sha256'][shape]
 data=dict(np.load(source));m=bank.maps(data);raw=BASE/shape;raw.mkdir(parents=True,exist_ok=True);out=REPORT/shape;out.mkdir(exist_ok=True)
 tp=raw/'truth.npz'
 if tp.exists():truth=dict(np.load(tp));assert str(truth['protocol_sha256'])==sha
 else:truth=generate(data,m,shape,sha,protocol);np.savez(tp,**truth)
 for case in range(32):
  if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol['deadline_utc']):raise RuntimeError('Campaign deadline reached; retain partial cohort')
  dest=out/f'case-{case:03d}.json'
  if dest.exists():assert json.loads(dest.read_text())['protocol_sha256']==sha;continue
  start=time.time();p,cs=fit(m['H'],truth['y'][case],bank.NOISE**2)
  states={'scale-correlation':(p,15,None,HYPERS[:,1]),'amplitude-only':(combine(cs[5:10]),5,None,SCALES),'fixed-nominal':(cs[7],1,1.,None),'oracle':(cs[int(truth['scale_index'][case])],1,float(truth['tau'][case]),None)}
  rows=[]
  for model,(original,nh,scale,hs) in states.items():
   for condition in ['initial','views','probes']:
    post=original
    if condition!='initial':
     for action in SCHEDULES[condition]:post=update(post,m,action,truth['measurements'][case,action],nh)
    r=metrics(post,data,m,truth,case,nh,scale,hs)
    if model=='scale-correlation':
     hw=post.weights.reshape(-1,15).sum(0).reshape(3,5).sum(1);r.update(correlation_posterior_weights=hw.tolist(),true_length_posterior_mass=float(hw[0]),correlation_posterior_mean=float(hw@LENGTHS))
    rows.append(dict(model=model,condition=condition,**r))
  dest.write_text(json.dumps(dict(shape=shape,case=case,protocol_sha256=sha,truth_tau=float(truth['tau'][case]),truth_correlation=.10,rows=rows,seconds=time.time()-start),indent=2)+'\n');print(shape,case,round(time.time()-start,2),flush=True)
 verify();print('COMPLETE',shape,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--shape',choices=['block','step']);a=ap.parse_args()
 if a.freeze:freeze()
 elif a.self_test:print(self_test())
 elif a.shape:run(a.shape)
 else:ap.error('Choose action')
