"""Numerical centered-moment repair only; original observations/design intact."""
from __future__ import annotations
import argparse,json,time,signal,copy,itertools
from pathlib import Path
import numpy as np
import torch
from experiments import run_nonlinear_bar_inverse as old
from experiments.campaign_centered_cubature import integrate_centered

ROOT=old.ROOT


def prepare(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh numerical-repair protocol required')
 original=ROOT/'runs/operator-dev/nonlinear-bar-inverse-v01/protocol.json';design=json.loads(original.read_text())
 result=ROOT/'runs/gaussian-campaign-v01/nonlinear-bar-inverse-v01/summary.json';failed=json.loads(result.read_text())
 if failed['stage']!='PRESERVED_FAILURE' or len(failed['SMC_runs']) or len(failed['adaptive'])!=2:raise ValueError('Exact completedtwo-reference/zero-sampler parent required')
 sources={**design['source_sha256'],**{s:old.sha(ROOT/s) for s in ['experiments/campaign_centered_cubature.py','tests/test_campaign_centered_cubature.py','experiments/continue_nonlinear_bar_inverse_v02.py']}}
 value=dict(schema='nonlinear-bar-centered-numerical-repair-v0.2',status='FROZEN_BEFORE_CENTERED_QUADRATURE_OR_SAMPLING',
  original_design=old.rec(original),original_failed_result=old.rec(result),observation=failed['observation'],
  pre_FEM=old.rec(ROOT/'runs/gaussian-campaign-v01/nonlinear-bar-inverse-v01/pre-FEM.json'),source_sha256=sources,
  fixed_center_m=failed['adaptive'][-1]['held_mean_m'],fixed_scale_m=old.problem.SIGMA,
  repair='ONEvectorcubature[Z,w*deltaL,w*deltaR,w*deltaL²,w*deltaR²],delta=(held-v01tightmean)/unchanged200um; normalizedcenteredfirst/second, componenterrorpropagation',
  relative_normalizer_agreement='<=5*(rtol+atol/Z) against retained scalarZ',
  v01_mean_parity_maximum_m=1e-8,estimated_mean_std_error_maximum_m=1e-8,
  original_CDF_quantile_references_reused=True,original_two_tolerance_100nm_std_gate_unchanged=True,
  component_errors_heuristic_NOT_rigorous_bounds=True,no_material_negative_variance_clamp=True,
  same_prior_truth_sigma_four_observation_bytes_forces_and_sampling_plan=True,no_noise_regeneration=True,
  CPU_threads=2,wall_cap_s=300,tests=dict(passed=3,elapsed_s=.40,scope='AnalytictruncatedGaussianvariance,250mmconstantshiftinvariance, unsafeZ/negativevariancefail,nonzerocenteredmean correction'),
  all_v01_negative_outcomes_preserved=True,scope=design['scope'])
 p.parent.mkdir(parents=True,exist_ok=True);old.write(p,value);return old.rec(p)


def run(path,digest,output):
 p=ROOT/path
 if old.sha(p)!=digest:raise ValueError('Numericalrepairprotocolchanged')
 plan=json.loads(p.read_text())
 for source,h in plan['source_sha256'].items():
  if old.sha(ROOT/source)!=h:raise ValueError('Frozenrepairsourcechanged:'+source)
 for name in ('original_design','original_failed_result','observation','pre_FEM'):
  r=plan[name]
  if old.sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Preservedparentinputchanged')
 if plan['wall_cap_s']!=300 or plan['fixed_scale_m']!=old.problem.SIGMA:raise ValueError('Unchangedbudget/noisescalerequired')
 parent=json.loads((ROOT/plan['original_failed_result']['path']).read_text());design=json.loads((ROOT/plan['original_design']['path']).read_text())
 out=Path(output)
 if out.exists():raise ValueError('Freshrepairoutputrequired')
 out.mkdir(parents=True);old.write(out/'protocol.json',plan);torch.set_num_threads(2);start=time.monotonic()
 with np.load(ROOT/plan['observation']['path'],allow_pickle=False) as raw:obs=raw['observed'].copy()
 state=dict(schema='nonlinear-bar-inverse-centered-result-v0.2',protocol=dict(path=path,sha256=digest),old4D_or_Gaussian_validation=False,completed=False,passed=False,stage='centered_reference',SMC_runs=[],observation=plan['observation'],pre_FEM=parent['pre_FEM'],
  original_failed_adaptive_agreement=parent['adaptive_agreement'],no_noise_regeneration=True,old_v01_preserved=True,
  force_interpolation_not_forecast=True,known_support_location_law=True,population_calibration_claim=False,raw_arrays_LOCAL=True)
 def save():state['elapsed_s']=time.monotonic()-start;old.write(out/'summary.json',state)
 def alarm(_s,_f):raise TimeoutError('Declared300scenteredcontinuationcapreached')
 handler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  refs=copy.deepcopy(parent['adaptive']);center=np.asarray(plan['fixed_center_m']);sigma=old.problem.SIGMA
  constant=-4*np.log(sigma*np.sqrt(2*np.pi))
  for ref in refs:
   repaired=integrate_centered(lambda x:np.exp(old.problem.log_likelihood(x,obs)-constant),
    lambda x:old.problem.predict(old.problem.physical(x),(2.,)),center,sigma,rtol=ref['rtol'],atol=ref['atol'])
   zerror=abs(repaired['Z_scaled']-ref['Z_scaled'])/ref['Z_scaled'];ztol=5*(ref['rtol']+ref['atol']/ref['Z_scaled'])
   parity=abs(repaired['held_mean_m']-np.asarray(ref['held_mean_m']))
   resolved=np.max(repaired['estimated_mean_error_m'])<=1e-8 and np.max(repaired['estimated_std_error_m'])<=1e-8
   repaired.update(relative_Z_difference_from_v01=zerror,relative_Z_tolerance=ztol,v01_mean_parity_abs_m=parity,
    numerical_repair_checks_passed=bool(zerror<=ztol and np.max(parity)<=1e-8 and resolved))
   if not repaired['numerical_repair_checks_passed']:raise RuntimeError('Centerednormalizer/error/meanparitygatefailed')
   ref.update(v01_raw_moment_mean_m=ref['held_mean_m'],v01_raw_moment_std_m=ref['held_std_m'],held_mean_m=repaired['held_mean_m'],held_std_m=repaired['held_std_m'],centered_repair=repaired)
   state['adaptive']=refs;save()
  state['adaptive_agreement']=old.compare(refs[0],refs[1],True);save()
  if not state['adaptive_agreement']['passed']:raise RuntimeError('Unchangedtwo-toleranceCDF/quantile/100nmheldgatefailed')
  ref=refs[-1];files=[];pilot=None;state['stage']='SMC'
  for count in old.COUNTS:
   for seed in old.SEEDS:
    fit=old.tempered_smc(old.BoxPrior(2),lambda s:old.problem.log_likelihood(s.coordinates,obs),particles=count,seed=seed,
     ess_fraction=.8,rejuvenation_steps=12,proposal_scale=.03,independence_probability=.25,max_stages=64,periodic_indices=())
    diagnostics=fit.diagnostics();essok=all(s['pre_resampling_ess']>=.8*count-1e-6 for s in diagnostics['stages'])
    if abs(diagnostics['stages'][-1]['beta']-1)>1e-14 or not essok:raise RuntimeError('OriginalSMCtemperature/ESSgatefailed')
    raw=out/f'smc-{count}-{seed}.npz';np.savez_compressed(raw,unit=fit.state.coordinates,log_likelihood=fit.log_likelihood);files.append(old.rec(raw))
    summary=old.sample_summary(fit.state.coordinates,ref);state['SMC_runs'].append(dict(particles=count,seed=seed,diagnostics=diagnostics,summary=summary,vs_reference=old.compare(summary,ref),temperature_and_ESS_passed=True));pilot=fit.state.coordinates;save()
  state['stage']='MCMC';chain,diagnostics=old.continuous_mcmc(lambda x:old.problem.log_likelihood(x,obs),pilot,seed=2026121310,warmup=1000,draws=4000,
   progress=lambda step,total:old.write(out/'progress.json',dict(stage='MCMC',step=step,planned=total,elapsed_s=time.monotonic()-start)))
  raw=out/'MCMC.npz';np.savez_compressed(raw,unit=chain);files.append(old.rec(raw));unit=chain.reshape(-1,2);theta=old.problem.physical(unit);held=old.problem.predict(theta,(2.,))
  physical=old.diagnose_quantities(torch.tensor(np.column_stack([theta,held]).reshape(8,4000,4)),['E_Pa','K_N_m','held_left_m','held_right_m'])
  summary=old.sample_summary(unit,ref);state['MCMC']=dict(**diagnostics,physical_and_held_diagnostics=physical,summary=summary,vs_reference=old.compare(summary,ref));save()
  runs=state['SMC_runs'];state['SMC_vs_MCMC']=[dict(particles=r['particles'],seed=r['seed'],**old.compare(r['summary'],summary)) for r in runs]
  state['pairwise']=[dict(first=[a['particles'],a['seed']],second=[b['particles'],b['seed']],**old.compare(a['summary'],b['summary'])) for i,a in enumerate(runs) for b in runs[i+1:]]
  old.write(out/'posterior-frozen.json',dict(completed=True,posterior_arrays=files,observation=plan['observation'],source_sha256=plan['source_sha256'],all_6_SMC_and_8chains_completed=True,held2N_truth_scored=False))
  state['posterior_frozen_before_held_truth']=True;state['stage']='postFEM';q=np.asarray(ref['physical_quantiles']);post=np.array(list(itertools.product(q[0],q[1])))
  state['post_FEM']=old.fem_verify(post,(1.,2.,3.),out,'post-FEM');state['held_truth_FEM']=old.fem_verify(old.problem.TRUTH[None],(2.,),out,'held-truth-FEM')
  truthheld=old.problem.independent_predict(old.problem.TRUTH[None],(2.,))[0];intervals=np.array(summary['held_latent_quantiles'])
  state['held_force_interpolation']=dict(force_N=2.,truth_m=truthheld,posterior_mean_m=summary['held_mean_m'],absolute_mean_error_m=abs(summary['held_mean_m']-truthheld),latent_quantiles_05_50_95_m=intervals,
   latent_90_interval_width_m=intervals[:,2]-intervals[:,0],latent_std_m=summary['held_std_m'],noisy_observation_std_m=np.sqrt(summary['held_std_m']**2+sigma**2),population_coverage_claim=False)
  linear=np.array([f/old.problem.TRUTH[1]+old.problem.SIZE[0]*f/(old.problem.TRUTH[0]*old.problem.SIZE[1]*old.problem.SIZE[2]) for f in (1.,2.,3.)])
  state['nonlinear_vs_smallstrain_truth_right_difference_m']=old.problem.independent_predict(old.problem.TRUTH[None],(1.,2.,3.))[0][1::2]-linear
  passed=all(r['vs_reference']['passed'] for r in runs) and all(r['passed'] for r in state['pairwise']+state['SMC_vs_MCMC'])
  passed=passed and diagnostics['diagnostics']['all_monitored_quantities_pass'] and physical['all_monitored_quantities_pass'] and state['MCMC']['vs_reference']['passed']
  state.update(completed=True,passed=bool(passed),stage='TERMINAL',parameter_identifiable_design='Lefttranslation identifiesK; extraextension identifiesE; fourrawindependentmeasurementscountedonce')
 except Exception as exc:state.update(stage='PRESERVED_FAILURE',error_type=type(exc).__name__,error=str(exc),failure_history=getattr(exc,'history',[]))
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler);save()
 return {k:state[k] for k in ('completed','passed','stage','elapsed_s')}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True);p=sub.add_parser('prepare');p.add_argument('--output',required=True);p=sub.add_parser('run')
 for n in ('protocol','protocol-sha256','output'):p.add_argument('--'+n,required=True)
 a=parser.parse_args();result=prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(result))
 if a.mode=='run':raise SystemExit(0 if result['completed'] else 1)
