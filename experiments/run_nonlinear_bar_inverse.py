"""Bounded independent nonlinear static E/K inverse proof, not population UQ."""
from __future__ import annotations
import argparse,hashlib,json,time,signal,itertools
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import cubature
from experiments import campaign_nonlinear_bar_inverse as problem
from experiments.campaign_tempered_reference import tempered_smc
from experiments.campaign_compliant_continuous_reference import BoxPrior,continuous_mcmc
from experiments.campaign_static_uniaxial_consistent import ConsistentPrismaticBar
from physgauss.chain_diagnostics import diagnose_quantities

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(2026121301,2026121302,2026121303)
COUNTS=(8192,32768)
NOISE_SEED=2026121300
CAP=300
PROBABILITIES=(.001,.05,.25,.5,.75,.95,.999)


def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items() if not k.startswith('_')}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n')

def prepare(path):
 path=Path(path)
 if path.exists():raise ValueError('Fresh inverse design required BEFOREobservation')
 physics=ROOT/'runs/operator-dev/static-uniaxial-consistent-v02/protocol.json';d=json.loads(physics.read_text())
 sources={**d['source_sha256'],**{p:sha(ROOT/p) for p in ['experiments/campaign_nonlinear_bar_inverse.py','experiments/run_nonlinear_bar_inverse.py',
  'tests/test_campaign_nonlinear_bar_inverse.py','experiments/campaign_tempered_reference.py','experiments/campaign_compliant_continuous_reference.py','src/physgauss/chain_diagnostics.py']}}
 design=dict(schema='nonlinear-homogeneous-bar-inverse-v0.1',status='FROZEN_BEFORE_OBSERVATION_GENERATION',CPU_threads=2,wall_cap_s=CAP,
  prior=dict(intrinsic='NormalizedUNIFORMlogE/logK',physical_bounds=[[4500.,9000.],[100.,400.]]),truth_E_K=problem.TRUTH,
  known_support='Fullyconsistentfiniteleftaxialspring, knownlocation/direction/law; no unknownGaussianpatch',
  observation=dict(known_forces_N=[1.,3.],raw_channels=['left1','right1','left3','right3'],independent_Gaussian_sigma_m=problem.SIGMA,seed=NOISE_SEED,
   no_extra_difference_channels=True,truth_from_independent_nested_brent=True),
  held_target=dict(force_N=2.,type='FORCEINTERPOLATION; not a futureforecast',channels=['left2','right2'],truth_scored_only_after_posterior_arrays_frozen=True),
  forward_verification=dict(pretruth_plus_prior_points=9,prior_points='4corners+4fixeduniformseed2026121299',forces=[1.,3.],meshes=[48,384],preFEMcells=36,
   posterior_points='Cartesian5/50/95%quantiles E/K, afterposteriorfreeze',post_forces=[1.,2.,3.],post_meshes=[48,384],post_plus_heldtruth_cells=56,
   maximum_full_position_error_m=2e-9,maximum_centerline_prediction_error_m=2e-9,maximum_free_force_residual_N=1e-8,
   vectorroot_original_force_residual_N=1e-12,vector_vs_brent_prior_and_zero_parity_tests=4),
  adaptive=dict(algorithm='IndependentadaptiveGK21productcubature',tolerances=[[1e-6,1e-13],[1e-8,1e-15]],max_subdivisions=512,
   quantile_probabilities=PROBABILITIES,quantile_bisections=18,CDF_knots='0+65equallyspacedbetweenreference.001/.999quantiles+1; secondtolerancereusesfirstgrid',
   maximum_CDF_tolerance_difference=2e-4,maximum_unit_quantile_difference=1e-4,maximum_held_mean_std_difference_m=1e-7),
  SMC=dict(counts=COUNTS,seeds=SEEDS,ess_fraction=.8,rejuvenation_steps=12,proposal_scale=.03,independence_probability=.25,max_stages=64,
   all6and15pairsrequired=True,pre_resampling_ESS_fraction_min=.8,temperature_reaches1=True,resampling_or_priorfailures_fatal=True,final_uniformESS_NOT_convergence=True),
  MCMC=dict(chains=8,all_initial_states_independent_prior=True,seed=2026121310,warmup=1000,draws=4000,no_thinning=True,
   pilot='LastfixedSMC32768seed2026121303, neverbestseedselected',exact_MH_correction=True,no_adaptation_or_warmup_extension=True),
  numeric_gates=dict(maximum_CDF_difference=.05,maximum_unit_quantile_difference=.01,
   held_mean_and_std_max_error_m='max(2e-6,.03*reference_latent_std) eachchannel',
   rank_folded_split_Rhat_less_than=1.01,bulk_tail_ESS_at_least=400,no_frozen_chain=True,
   monitored='Bothintrinsicparameters+loglikelihood and bothphysicalparameters+heldlatentdisplacements',
   SMC_vs_reference_and_MCMC_and_all15pairs=True),
  no_scientific_retuning_no_population_calibration=True,all_attempts_preserved_no_retry=True,
  benchmark_protocol=rec(physics),benchmark_result=rec(ROOT/'runs/gaussian-campaign-v01/static-uniaxial-consistent-v02/summary.json'),
  source_sha256=sources,scope='New nonlinear homogeneousSTATIC two-parameter inverse with knownsupportlaw; not original4D rescue/Gaussianinference/Isaac/populationcoverage')
 path.parent.mkdir(parents=True,exist_ok=True);write(path,design);return rec(path)


def fem_verify(points,forces,output,label):
 rows=[]
 for i,(E,K) in enumerate(points):
  for force in forces:
   expected=problem.independent_predict(np.array([[E,K]]),(force,))[0]
   for cells in ((2,2,2),(4,4,4)):
    model=ConsistentPrismaticBar(cells,young=float(E),spring=float(K),force=float(force));answer=model.solve();u=answer['displacement_local_m'];oracle_u,_=model.oracle_field()
    indices=[int(np.flatnonzero(np.linalg.norm(model.reference_local-np.array([x,0.,0.]),axis=1)<1e-13)[0]) for x in (0.,problem.SIZE[0])]
    error=float(np.max(abs(u[indices,0]-expected)));full=float(np.max(abs(u-oracle_u)));res=float(np.linalg.norm(answer['residual'].ravel()[model.free]))
    rows.append(dict(point=i,E_Pa=float(E),K_N_m=float(K),force_N=float(force),tetrahedra=len(model.tets),centerline_error_m=error,full_position_error_m=full,free_residual_N=res,
     minimum_J=float(answer['J'].min()),passed=error<=2e-9 and full<=2e-9 and res<=1e-8))
    if not rows[-1]['passed']:
     write(output/(label+'.json'),dict(completed=False,rows=rows));raise RuntimeError('IndependentFEM/oracleforwardverificationfailed')
 result=dict(completed=True,passed=True,cells=len(rows),maximum_centerline_error_m=max(r['centerline_error_m'] for r in rows),
  maximum_full_position_error_m=max(r['full_position_error_m'] for r in rows),maximum_free_residual_N=max(r['free_residual_N'] for r in rows),rows=rows)
 write(output/(label+'.json'),result);return result


def adaptive(obs,rtol,atol,knots=None):
 normalizer=-4*np.log(problem.SIGMA*np.sqrt(2*np.pi));reports=[]
 def density(x):return np.exp(problem.log_likelihood(x,obs)-normalizer)
 def integrate(upper,function=density):
  value=cubature(function,np.zeros(2),np.asarray(upper),rule='gk21',rtol=rtol,atol=atol,max_subdivisions=512)
  reports.append(dict(status=value.status,error_estimate=float(np.max(value.error)),subdivisions=value.subdivisions))
  if value.status!='converged':raise RuntimeError('Independentadaptiveintegraltolerancefailed')
  return value.estimate
 Z=float(integrate(np.ones(2)));quantiles=[]
 if not Z>0:raise RuntimeError('Independentadaptiveposteriorhaszeromass')
 for j in range(2):
  q=[]
  for probability in PROBABILITIES:
   lo,hi=0.,1.
   for _ in range(18):
    mid=(lo+hi)/2;upper=np.ones(2);upper[j]=mid
    if integrate(upper)/Z<probability:lo=mid
    else:hi=mid
   q.append((lo+hi)/2)
  quantiles.append(q)
 if knots is None:knots=[np.r_[0.,np.linspace(q[0],q[-1],65),1.] for q in quantiles]
 cdfs=[]
 for j,grid in enumerate(knots):
  cdf=[]
  for knot in grid:
   upper=np.ones(2);upper[j]=knot
   cdf.append(0. if knot==0 else 1. if knot==1 else float(integrate(upper)/Z))
  cdfs.append(cdf)
 def moments(x):
  y=problem.predict(problem.physical(x),(2.,))/.02;w=density(x)
  return np.column_stack([w[:,None]*y,w[:,None]*y**2])
 m=integrate(np.ones(2),moments)/Z;mean=m[:2]*.02;var=(m[2:]-m[:2]**2)*.02**2
 if var.min()<-1e-15:raise RuntimeError('Adaptiveheldvarianceinvalid')
 return dict(rtol=rtol,atol=atol,Z_scaled=Z,log_evidence=float(np.log(Z)+normalizer),integral_checks=reports,
  unit_knots=knots,unit_CDF=cdfs,all_unit_quantiles=quantiles,unit_quantiles=np.array(quantiles)[:,[1,3,5]],
  physical_quantiles=problem.physical(np.array(quantiles).T)[[1,3,5]].T,
  held_mean_m=mean,held_std_m=np.sqrt(np.maximum(var,0)))


def sample_summary(unit,reference):
 theta=problem.physical(unit);y=problem.predict(theta,(2.,))
 return dict(unit_CDF=[np.mean(unit[:,j,None]<=np.asarray(reference['unit_knots'][j])[None],axis=0) for j in range(2)],
  unit_quantiles=np.quantile(unit,[.05,.5,.95],axis=0).T,physical_mean=theta.mean(0),physical_std=theta.std(0),
  physical_quantiles=np.quantile(theta,[.05,.5,.95],axis=0).T,held_mean_m=y.mean(0),held_std_m=y.std(0),held_latent_quantiles=np.quantile(y,[.05,.5,.95],axis=0).T)


def compare(a,b,strict=False):
 cdf=float(np.max(abs(np.array(a['unit_CDF'])-np.array(b['unit_CDF']))));q=float(np.max(abs(np.array(a['unit_quantiles'])-np.array(b['unit_quantiles']))))
 dm=abs(np.array(a['held_mean_m'])-np.array(b['held_mean_m']));ds=abs(np.array(a['held_std_m'])-np.array(b['held_std_m']))
 tolerance=np.full(2,1e-7) if strict else np.maximum(2e-6,.03*np.array(b['held_std_m']))
 return dict(maximum_CDF_difference=cdf,maximum_unit_quantile_difference=q,held_mean_abs_difference_m=dm,held_std_abs_difference_m=ds,tolerance_m=tolerance,
  passed=cdf<=(2e-4 if strict else .05) and q<=(1e-4 if strict else .01) and bool(np.all(dm<=tolerance)&np.all(ds<=tolerance)))


def run(path,digest,output):
 p=ROOT/path
 if sha(p)!=digest:raise ValueError('Newinverseprotocolchanged')
 protocol=json.loads(p.read_text())
 for source,h in protocol['source_sha256'].items():
  if sha(ROOT/source)!=h:raise ValueError('NewinversefrozenSourcechanged:'+source)
 for name in ('benchmark_protocol','benchmark_result'):
  r=protocol[name]
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Benchmarkevidencechanged')
 if protocol['wall_cap_s']!=CAP or protocol['observation']['seed']!=NOISE_SEED:raise ValueError('Fixeddesignrequired')
 out=Path(output)
 if out.exists():raise ValueError('Freshnewinverseoutputrequired')
 out.mkdir(parents=True);write(out/'protocol.json',protocol);torch.set_num_threads(2);start=time.monotonic()
 state=dict(schema='nonlinear-bar-inverse-result-v0.1',completed=False,passed=False,protocol=dict(path=path,sha256=digest),stage='preFEM',SMC_runs=[],
  force_interpolation_not_forecast=True,known_support_location_law=True,old4D_or_Gaussian_validation=False,population_calibration_claim=False,raw_arrays_LOCAL=True)
 def save():state['elapsed_s']=time.monotonic()-start;write(out/'summary.json',state)
 def alarm(_s,_f):raise TimeoutError('Declared300snonlinearinversebudgetreached')
 old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,CAP)
 try:
  prior=np.vstack([np.array(list(itertools.product((0.,1.),repeat=2))),np.random.default_rng(2026121299).uniform(size=(4,2))])
  points=np.vstack([problem.TRUTH,problem.physical(prior)])
  # Both independent models verified before acquiring NEW noisy observations.
  state['pre_FEM']=fem_verify(points,(1.,3.),out,'pre-FEM');save()
  truth=problem.independent_predict(problem.TRUTH[None])[0];obs=truth+np.random.default_rng(NOISE_SEED).normal(0,problem.SIGMA,4)
  np.savez_compressed(out/'observation.npz',observed=obs,exact_acquired=truth,sigma=problem.SIGMA,forces_N=problem.FORCES)
  state['observation']=rec(out/'observation.npz');state['stage']='adaptive';save();refs=[]
  for rtol,atol in ((1e-6,1e-13),(1e-8,1e-15)):
   ref=adaptive(obs,rtol,atol,None if not refs else refs[0]['unit_knots']);refs.append(ref);state['adaptive']=refs;save()
  state['adaptive_agreement']=compare(refs[0],refs[1],True);save()
  if not state['adaptive_agreement']['passed']:raise RuntimeError('Adaptiveindependentreferenceagreementgatefailed')
  ref=refs[-1];state['stage']='SMC';files=[];pilot=None
  for count in COUNTS:
   for seed in SEEDS:
    fit=tempered_smc(BoxPrior(2),lambda s:problem.log_likelihood(s.coordinates,obs),particles=count,seed=seed,
     ess_fraction=.8,rejuvenation_steps=12,proposal_scale=.03,independence_probability=.25,max_stages=64,periodic_indices=())
    diagnostics=fit.diagnostics();reached=abs(diagnostics['stages'][-1]['beta']-1)<1e-14
    essok=all(s['pre_resampling_ess']>=.8*count-1e-6 for s in diagnostics['stages'])
    if not reached or not essok:raise RuntimeError('SMCtemperature/ESS/resamplinggatefailed')
    raw=out/f'smc-{count}-{seed}.npz';np.savez_compressed(raw,unit=fit.state.coordinates,log_likelihood=fit.log_likelihood);files.append(rec(raw))
    summary=sample_summary(fit.state.coordinates,ref);state['SMC_runs'].append(dict(particles=count,seed=seed,diagnostics=diagnostics,summary=summary,vs_reference=compare(summary,ref),temperature_and_ESS_passed=True));pilot=fit.state.coordinates;save()
  state['stage']='MCMC';chain,diagnostics=continuous_mcmc(lambda x:problem.log_likelihood(x,obs),pilot,seed=2026121310,warmup=1000,draws=4000,
   progress=lambda step,total:write(out/'progress.json',dict(stage='MCMC',step=step,planned=total,elapsed_s=time.monotonic()-start)))
  raw=out/'MCMC.npz';np.savez_compressed(raw,unit=chain);files.append(rec(raw));unit=chain.reshape(-1,2);theta=problem.physical(unit);held=problem.predict(theta,(2.,))
  physical=diagnose_quantities(torch.tensor(np.column_stack([theta,held]).reshape(8,4000,4)),['E_Pa','K_N_m','held_left_m','held_right_m'])
  summary=sample_summary(unit,ref);state['MCMC']=dict(**diagnostics,physical_and_held_diagnostics=physical,summary=summary,vs_reference=compare(summary,ref));save()
  runs=state['SMC_runs'];state['SMC_vs_MCMC']=[dict(particles=r['particles'],seed=r['seed'],**compare(r['summary'],summary)) for r in runs]
  state['pairwise']=[dict(first=[a['particles'],a['seed']],second=[b['particles'],b['seed']],**compare(a['summary'],b['summary'])) for i,a in enumerate(runs) for b in runs[i+1:]]
  # Freeze the entire fixed posterior before scoring the held FORCE, or
  # looking at its truth displacement. No source/data/seed/algorithm selection.
  write(out/'posterior-frozen.json',dict(completed=True,posterior_arrays=files,observation=state['observation'],source_sha256=protocol['source_sha256'],
   all_6_SMC_and_8chains_completed=True,held2N_truth_scored=False))
  state['posterior_frozen_before_held_truth']=True;state['stage']='postFEM'
  q=np.asarray(ref['physical_quantiles']);post=np.array(list(itertools.product(q[0],q[1])))
  state['post_FEM']=fem_verify(post,(1.,2.,3.),out,'post-FEM')
  state['held_truth_FEM']=fem_verify(problem.TRUTH[None],(2.,),out,'held-truth-FEM')
  truthheld=problem.independent_predict(problem.TRUTH[None],(2.,))[0]
  intervals=np.array(summary['held_latent_quantiles']);state['held_force_interpolation']=dict(force_N=2.,truth_m=truthheld,posterior_mean_m=summary['held_mean_m'],
   absolute_mean_error_m=abs(summary['held_mean_m']-truthheld),latent_quantiles_05_50_95_m=intervals,latent_90_interval_width_m=intervals[:,2]-intervals[:,0],
   latent_std_m=summary['held_std_m'],noisy_observation_std_m=np.sqrt(summary['held_std_m']**2+problem.SIGMA**2),population_coverage_claim=False)
  linear=np.array([f/problem.TRUTH[1]+problem.SIZE[0]*f/(problem.TRUTH[0]*problem.SIZE[1]*problem.SIZE[2]) for f in (1.,2.,3.)])
  nonlinear=problem.independent_predict(problem.TRUTH[None],(1.,2.,3.))[0][1::2]
  state['nonlinear_vs_smallstrain_truth_right_difference_m']=nonlinear-linear
  passed=all(r['vs_reference']['passed'] for r in runs) and all(r['passed'] for r in state['pairwise']+state['SMC_vs_MCMC'])
  passed=passed and diagnostics['diagnostics']['all_monitored_quantities_pass'] and physical['all_monitored_quantities_pass'] and state['MCMC']['vs_reference']['passed']
  state.update(completed=True,passed=bool(passed),stage='TERMINAL',parameter_identifiable_design='Leftspringtranslation informsK; rightminusleftmaterialextension informsE. Raw4 independent channels, not duplicated differences.')
 except Exception as exc:state.update(stage='PRESERVED_FAILURE',error_type=type(exc).__name__,error=str(exc),failure_history=getattr(exc,'history',[]))
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);save()
 return {k:state[k] for k in ('completed','passed','stage','elapsed_s')}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True);p=sub.add_parser('prepare');p.add_argument('--output',required=True);p=sub.add_parser('run')
 for n in ('protocol','protocol-sha256','output'):p.add_argument('--'+n,required=True)
 a=parser.parse_args();result=prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(result))
 if a.mode=='run':raise SystemExit(0 if result['completed'] else 1)
