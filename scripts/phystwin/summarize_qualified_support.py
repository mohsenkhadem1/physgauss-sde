"""Publish v05 results, including failed scientific claims and numerical checks."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.qualified_reference import BASE,ROOT,LAYOUTS,protocol,write_json
from scripts.phystwin.qualified_policy import verify_seal
OUT=ROOT/'reports/phystwin-qualified-support-v05'


def deformation_audit(spec):
    rows=[];groups={}
    for split in ['development','calibration','test']:
        selected=[]
        for i in range(len(spec[split])):
            with np.load(BASE/'fields'/f'{split}-{i}-qualified.npz') as z:v=z['nonrigid'].copy()
            j=1+int(np.argmax(v[1:3,0]));selected.append([v[j,0],v[j,0]/v[j,1]])
            rows.append(dict(split=split,object=i,best_available_probe=j,
                nonrigid_coordinate_rms_m=v[:,0].tolist(),total_coordinate_rms_m=v[:,1].tolist(),
                nonrigid_rms_fraction=(v[:,0]/v[:,1]).tolist()))
        selected=np.array(selected);groups[split]=dict(minimum_nonrigid_rms_mm=float(1000*selected[:,0].min()),
            median_nonrigid_rms_mm=float(1000*np.median(selected[:,0])),minimum_nonrigid_rms_fraction=float(selected[:,1].min()),
            median_nonrigid_rms_fraction=float(np.median(selected[:,1])))
    with np.load(BASE/'analysis/test-predictions.npz') as z:choices=z['adaptive'].copy()
    test=[r for r in rows if r['split']=='test'];chosen=np.array([[test[i//spec['noise_repeats']][k][a] for k in ['nonrigid_coordinate_rms_m','nonrigid_rms_fraction']] for i,a in enumerate(choices)])
    return dict(scope='Post-evaluation physical deformation diagnostic. Nonrigid coordinate RMS after best rigid alignment divided by total displacement RMS; this is an amplitude ratio, not a variance fraction. Available-probe selection maximizes nonrigid RMS, not the ratio. Not used to change test actions.',
        best_available_probe=groups,chosen_test_adaptive_probe=dict(minimum_nonrigid_rms_mm=float(1000*chosen[:,0].min()),
            median_nonrigid_rms_mm=float(1000*np.median(chosen[:,0])),minimum_nonrigid_rms_fraction=float(chosen[:,1].min()),
            median_nonrigid_rms_fraction=float(np.median(chosen[:,1]))),objects=rows)


def physical_intervals(spec,test,saved,bootstrap_indices):
    repeats=spec['noise_repeats'];n=len(spec['test']);rows=np.arange(n*repeats)
    truth=np.repeat(np.log([[r['E'],r['K']] for r in spec['test']]),repeats,axis=0)
    q=np.stack([saved[f'{i}_log_parameter_quantiles'] for i in range(3)])
    raw=dict(coverage=((q[:,:,:,0]<=truth)&(truth<=q[:,:,:,2])).astype(float),
        log_median_absolute_error=abs(q[:,:,:,1]-truth),log_interval_width=q[:,:,:,2]-q[:,:,:,0])
    choices=dict(initial=np.zeros(len(rows),int),fixed=np.full(len(rows),test['fixed_action']),
        adaptive=saved['adaptive'],fixture_side_heuristic=saved['heuristic'],random=None)
    objects={name:{metric:(values[1:].mean(0) if name=='random' else values[choice,rows]).reshape(n,repeats,2).mean(1)
        for metric,values in raw.items()} for name,choice in choices.items()}
    result={}
    for name,metrics in objects.items():
        result[name]={}
        for j,parameter in enumerate(['E','K']):
            result[name][parameter]={}
            for metric,values in metrics.items():
                v=values[:,j];delta=v-objects['initial'][metric][:,j]
                result[name][parameter][metric]=dict(mean=float(v.mean()),bootstrap95=np.quantile(v[bootstrap_indices].mean(1),[.025,.975]).tolist(),
                    change_from_initial=float(delta.mean()),paired_change_bootstrap95=np.quantile(delta[bootstrap_indices].mean(1),[.025,.975]).tolist())
    return result


def main():
    spec=protocol();analysis=BASE/'analysis';seal=json.loads((analysis/'seal.json').read_text());verify_seal(seal)
    test=json.loads((analysis/'test-summary.json').read_text());mse=json.loads((analysis/'test-object-mse.json').read_text());deformation=json.loads((analysis/'test-deformation-object-mse.json').read_text())
    oracle=json.loads((analysis/'oracle-audit.json').read_text())
    groups=[np.array([i for i,r in enumerate(spec['test']) if r['layout']==l]) for l in LAYOUTS];rng=np.random.default_rng(8801)
    ix=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
    with np.load(analysis/'test-predictions.npz') as saved:physical_ci=physical_intervals(spec,test,saved,ix)
    comparisons={}
    for baseline in ['fixed','random','fixture_side_heuristic']:
        row={}
        for label,values in [('coordinate_mse',mse),('aligned_mse',{k:v['aligned_mse'] for k,v in deformation.items()}),('strain_mse',{k:v['strain_mse'] for k,v in deformation.items()})]:
            a=np.array(values['adaptive']);b=np.array(values[baseline]);benefit=np.sqrt(b[ix].mean(1))-np.sqrt(a[ix].mean(1))
            row[label]=dict(rmse_improvement=float(np.sqrt(b.mean())-np.sqrt(a.mean())),paired_stratified_bootstrap95=np.quantile(benefit,[.025,.975]).tolist())
        comparisons[baseline]=row
    qualification=[]
    for split in ['development','calibration','test']:
        for i in range(len(spec[split])):
            q=json.loads((BASE/'qualification'/f'{split}-{i}.json').read_text());assert q['qualified'],f'{split}-{i} failed qualification'
            prior=[json.loads((BASE/p).read_text()) for p in q.get('prior_p2_attempts',[])]
            qualification.append(dict(split=split,object=i,prior_p2_audits=prior,**q))
    before=test['physical']['initial'];after=test['physical']['adaptive']
    flags={p:dict(narrower=after[f'{p}_mean_log_interval_width']<before[f'{p}_mean_log_interval_width'],median_error_worse=after[f'{p}_log_median_MAE']>before[f'{p}_log_median_MAE'],coverage_lower=after[f'{p}_90_coverage']<before[f'{p}_90_coverage']) for p in ['E','K']}
    for p in ['E','K']:
        flags[p].update(coverage_drop_supported=physical_ci['adaptive'][p]['coverage']['paired_change_bootstrap95'][1]<0,
            median_error_increase_supported=physical_ci['adaptive'][p]['log_median_absolute_error']['paired_change_bootstrap95'][0]>0)
    success=dict(all_reference_objects_qualified=True,posterior_refinement_passed=bool(test['quadrature'][-1]['passed']),
        calibrated_adaptive_object_coverage_at_least90=test['policies']['adaptive']['calibrated_object_all_repeats_coverage']>=.9,
        beats_fixed=comparisons['fixed']['coordinate_mse']['paired_stratified_bootstrap95'][0]>0,
        beats_fixture_heuristic=comparisons['fixture_side_heuristic']['coordinate_mse']['paired_stratified_bootstrap95'][0]>0,
        utility_stable=test['utility_stable']==test['utility_trials'],posterior_grid_action_agreement=test['grid_action_agreement']==test['utility_trials'])
    OUT.mkdir(exist_ok=True,parents=True)
    for f in analysis.glob('*.json'):shutil.copyfile(f,OUT/f.name)
    for f in BASE.glob('*.json'):shutil.copyfile(f,OUT/f.name)
    write_json(OUT/'comparison.json',comparisons);write_json(OUT/'qualification.json',qualification);write_json(OUT/'decision.json',success);write_json(OUT/'confidence-after-probing.json',flags)
    write_json(OUT/'physical-interval-bootstrap.json',physical_ci)
    motion=deformation_audit(spec);write_json(OUT/'deformation-audit.json',motion)
    policies=['initial','fixed','random','adaptive','fixture_side_heuristic'];labels=['Initial','Fixed','Random','Adaptive','Fixture heuristic']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    panels=[('rmse_m',1000,'Held-load coordinate RMSE (mm)'),('aligned_rmse_m',1000,'Shape RMSE after rigid alignment (mm)'),('observed_grid_strain_rmse',1,'Strain RMSE on the observation grid'),('calibrated_object_all_repeats_coverage',100,'Calibrated whole-object coverage (%)'),('calibrated_mean_width_m',1000,'Mean calibrated band width (mm)')]
    colors=['#aaa','#c58040','#b1a36a','#246da8','#42916a']
    for ax,(key,scale,title) in zip(axes.ravel(),panels):
        ax.bar(labels,[test['policies'][k][key]*scale for k in policies],color=colors);ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    ax=axes.ravel()[-1];x=np.arange(5)
    ax.bar(x-.18,[100*test['physical'][p]['E_90_coverage'] for p in policies],.36,label='Material E');ax.bar(x+.18,[100*test['physical'][p]['K_90_coverage'] for p in policies],.36,label='Support K');ax.axhline(90,color='grey',ls='--');ax.set_xticks(x,labels,rotation=25);ax.set_ylim(0,105);ax.set_title('Raw nominal 90% physical interval coverage');ax.legend()
    fig.suptitle('24 held-out objects, four fixture families; four noise repeats per object')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    labels_for_checks=dict(all_reference_objects_qualified='reference qualification',posterior_refinement_passed='posterior convergence',
        calibrated_adaptive_object_coverage_at_least90='at least 90% adaptive whole-object coverage',beats_fixed='advantage over the fixed action',
        beats_fixture_heuristic='advantage over the fixture heuristic',utility_stable='utility sampling stability',posterior_grid_action_agreement='posterior-grid action agreement')
    failed=[labels_for_checks[k] for k,v in success.items() if not v]
    verdict='The preregistered active-probing success criteria were met.' if not failed else 'The preregistered active-probing success criteria were not all met. Failed checks: '+', '.join(failed)+'.'
    lines=['# Qualified nonlinear reference, uncertainty and probing — v05','',verdict,'',
        'All 56 reference objects passed their recorded spatial and quadrature checks. This is finite numerical qualification under specified sampled-field tolerances, not an absolute continuum-error proof or physical validation against real measurements.','',
        '## Held-out results','', '| Policy | Coordinate RMSE (mm) | Aligned shape RMSE (mm) | Whole-object band coverage | Band width (mm) | E interval coverage | K interval coverage |','|---|---:|---:|---:|---:|---:|---:|']
    for label,p in zip(labels,policies):
        q=test['policies'][p];h=test['physical'][p]
        lines.append(f"| {label} | {1000*q['rmse_m']:.4f} | {1000*q['aligned_rmse_m']:.4f} | {100*q['calibrated_object_all_repeats_coverage']:.1f}% | {1000*q['calibrated_mean_width_m']:.3f} | {100*h['E_90_coverage']:.1f}% | {100*h['K_90_coverage']:.1f}% |")
    lines+=['','## Does adaptive probing beat the strong baselines?','', '| Baseline | Adaptive coordinate RMSE improvement (mm) | Paired object-bootstrap 95% interval (mm) |','|---|---:|---|']
    for name,c in comparisons.items():
        r=c['coordinate_mse'];ci=r['paired_stratified_bootstrap95'];lines.append(f"| {name} | {1000*r['rmse_improvement']:.4f} | [{1000*ci[0]:.4f}, {1000*ci[1]:.4f}] |")
    lines+=['','Positive intervals support an advantage over that baseline. Intervals containing zero do not. Bootstrap resampling is stratified by fixture family and clustered by object; four noise repeats are not counted as four independent objects. Aligned-shape and coarse-grid strain comparisons are retained in [comparison.json](comparison.json).','',
        '## Numerical and inference checks','',
        'The original v04 gates remain: displacement RMS 0.02 mm, maximum coordinate difference 0.1 mm, strain relative L2 5%, support traction relative L2 2%, energy 1%, and equilibrium tolerances. Every object requires two consecutive spatial passes and a finest-mesh q3/q4 check. Exact same-face full-support translations use the analytic zero-strain solution instead of relative errors of numerical zero. [qualification.json](qualification.json) retains the per-object evidence and identifies the accepted element space. If the development-validated Q2 fallback is enabled, its method and cross-space validation are included in the accompanying addenda; prior failed P2 attempts remain recorded.','',
        'The forward surrogate is selected only on development data using a 30 μm RMS accuracy gate; [model-selection.json](model-selection.json) records rejected candidates. Posterior refinement checks predictive means, standard deviations and physical marginal quantiles. The categorical fixture is latent. Known geometry and known forces are supplied. No true fixture label is given to a policy.','',
        'Joint numerical discrepancy is fitted on development residuals, then conditioned on both initial and probe observations. Twenty new calibration objects determine the fixed action and a common object-wise prediction-band multiplier; [seal.json](seal.json) hashes the model and data before held-out evaluation. Raw physical parameter intervals are not disguised as calibrated prediction bands. [confidence-after-probing.json](confidence-after-probing.json) flags intervals that narrow while errors or coverage worsen. [Physical interval uncertainty](physical-interval-bootstrap.json) includes stratified object-bootstrap intervals for coverage, error and width, including paired changes after probing. Noise repeats are averaged within each object before resampling.','',
        f"Utility choices stable under sampling refinement: {test['utility_stable']}/{test['utility_trials']}. Choices agreeing under posterior-grid refinement: {test['grid_action_agreement']}/{test['utility_trials']}. All scientific pass/fail decisions are recorded in [decision.json](decision.json).",'',
        '## Confidence after probing','',
        '| Parameter | Change in raw 90% coverage | Paired object-bootstrap 95% interval | Change in absolute log-median error | Paired object-bootstrap 95% interval |','|---|---:|---|---:|---|']
    for parameter in ['E','K']:
        c=physical_ci['adaptive'][parameter]['coverage'];e=physical_ci['adaptive'][parameter]['log_median_absolute_error']
        cc=np.array(c['paired_change_bootstrap95'])*100;ec=e['paired_change_bootstrap95']
        lines.append(f"| {parameter} | {100*c['change_from_initial']:.2f} percentage points | [{cc[0]:.2f}, {cc[1]:.2f}] | {e['change_from_initial']:.4f} | [{ec[0]:.4f}, {ec[1]:.4f}] |")
    lines+=['','Positive coverage changes and negative error changes are favorable. Narrower intervals alone are not evidence of better inference. These are raw physical intervals; prediction-band calibration does not calibrate them.','',
        '## Privileged action diagnostic','',
        '| Choice | Expected held-force RMSE (mm) | Agreement with oracle action |','|---|---:|---:|']
    for name in ['fixed','random','fixture_side_heuristic','adaptive','oracle']:
        r=oracle['scores'][name];agreement=f"{100*r['oracle_choice_agreement']:.1f}%" if 'oracle_choice_agreement' in r else '—'
        lines.append(f"| {name} | {1000*r['expected_rmse_m']:.4f} | {agreement} |")
    lines+=['',f"This diagnostic gives the action evaluator the true object's responses, then integrates future probe noise with {oracle['samples']} samples. It keeps the fitted predictor fixed. It is not an eligible policy, not used for tuning, and its expected errors are not directly comparable to the realized-outcome table above. Oracle action choices agree between 128 and 512 samples in {oracle['choice_refinement_agreement']}/{oracle['trials']} cases; disagreement limits interpretation of this diagnostic. Gaps include uncertainty about the hidden object, not just policy computation error. [Full diagnostic](oracle-audit.json).",'',
        '## Are these objects actually deforming?','',
        '| Split | Minimum best-probe nonrigid RMS (mm) | Median (mm) | Minimum nonrigid / total RMS | Median ratio |','|---|---:|---:|---:|---:|']
    for split,r in motion['best_available_probe'].items():
        lines.append(f"| {split} | {r['minimum_nonrigid_rms_mm']:.3f} | {r['median_nonrigid_rms_mm']:.3f} | {100*r['minimum_nonrigid_rms_fraction']:.1f}% | {100*r['median_nonrigid_rms_fraction']:.1f}% |")
    r=motion['chosen_test_adaptive_probe']
    lines+=['',f"This diagnostic removes best-fit rigid translation and rotation. The table uses each object's available probe with the largest nonrigid RMS. For probes actually chosen by the adaptive policy on test observations, minimum nonrigid RMS is {r['minimum_nonrigid_rms_mm']:.3f} mm and minimum nonrigid/total RMS is {100*r['minimum_nonrigid_rms_fraction']:.1f}%. These percentages are RMS amplitude ratios, not fractions of motion variance. [Per-object and per-load evidence](deformation-audit.json).",'',
        '## Limits','',
        'This is a static synthetic study with four known fixture families and continuous material/support stiffness variation. It does not establish general spatial boundary inference, new topology generalization, gravity/contact performance, or learned PhysTwin uncertainty. Coordinate error includes rigid movement; aligned shape error removes it. Strain is computed on the 81-point observation grid, not interpreted as peak boundary stress.','',
        'Pooled prediction calibration has nominal 90% coverage. The balanced fixture sampling is stratified, so an unconditional exchangeability guarantee is not assumed. Test coverage is measured on only 24 independent objects; material coverage averages four noisy observations per object. Small differences should not be overinterpreted.','',
        'GPU multigrid accelerates the unchanged nonlinear equations; the original linear and nonlinear residuals are checked. Tangent temporaries and quadrature gradients are streamed. CPU/direct-solver parity evidence is retained in [solver-validation.json](solver-validation.json). Raw fields remain local.','', '![Held-out comparison](comparison.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(decision=success,comparison=comparisons),indent=2))

if __name__=='__main__':main()
