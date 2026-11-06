"""Populate paper exclusively from complete, audited surface and hierarchy cohorts."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'papers/cvpr2027-joint-identification';SEC=PAPER/'sections'
SUR=ROOT/'reports/joint-field-surface-v04';HIER=ROOT/'reports/joint-field-hierarchy-surface-v05'

def read(p):return json.loads(p.read_text())
def fmt(x):return f'{x:.4f}'
def pct(x):return f'{100*x:.1f}'
def table(caption,header,rows,cols,label=''):
    return '\n'.join([r'\begin{table*}[!t]\centering\small\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+'}',r'\begin{tabular}{'+cols+r'}\toprule',header+r'\\\midrule',*[' & '.join(map(str,r))+r'\\' for r in rows],r'\bottomrule\end{tabular}',r'\end{table*}'])

def main(draft=False):
    s=read(SUR/'summary.json');h=read(HIER/'summary.json');c=read(SUR/'continuous-summary.json');audit=read(SUR/'evidence-audit.json')
    ha=read(HIER/'completion-audit.json');hf=read(HIER/'finalizer.json')
    assert ha['passed'] and hf['completed'],'Hierarchy completion audit required'
    assert s['complete'] and h['cases']==128 and audit['integrity_pass'],'Completed finite cohorts and valid integrity audit required'
    if not draft:assert c['complete'] and audit['complete'],'Full final continuous audits required'
    temporal_root=ROOT/'reports/joint-field-temporal-v06';tm=read(temporal_root/'summary.json');ta=read(temporal_root/'completion-audit.json');ti=read(temporal_root/'independent-final-audit.json')
    assert tm['complete'] and ta['passed'] and ti['passed'] and tm['cases']==64,'Completed temporal follow-up required'
    tg=lambda model,condition='views':next(rr for rr in tm['means'] if rr['model']==model and rr['condition']==condition)
    tf,tamp,tor=tg('scale-correlation'),tg('amplitude-only'),tg('oracle')
    g=lambda policy,step=4,pop='matched':next(r for r in s['summary'] if r['population']==pop and r['policy']==policy and r['step']==step)
    hg=lambda model,condition:next(r for r in h['means'] if r['model']==model and r['condition']==condition)
    i,a,f,r=g('active',0),g('active'),g('fixed'),g('random');fp=g('fixed-probe');v=g('fixed-view')
    comparison=next(x for x in s['comparisons'] if x['population']=='matched' and x['second']=='fixed' and x['metric']=='rmse_mm')
    view_comparison=next(x for x in s['comparisons'] if x['population']=='matched' and x['second']=='fixed-view' and x['metric']=='rmse_mm')
    hi,hv,hp=hg('hierarchical','initial'),hg('hierarchical','views'),hg('hierarchical','probes');nom=hg('fixed-nominal','probes');broad=hg('variance-matched','probes')
    (SEC/'results-macros.tex').write_text('\n'.join([r'\newcommand{\InitialError}{'+fmt(i['rmse_mm'])+'}',r'\newcommand{\ActiveError}{'+fmt(a['rmse_mm'])+'}',r'\newcommand{\FixedError}{'+fmt(f['rmse_mm'])+'}',r'\newcommand{\FixedViewError}{'+fmt(v['rmse_mm'])+'}',r'\newcommand{\CampaignStatus}{'+(r'\textbf{Working draft: continuous evaluation is running.}' if draft else '')+'}'])+'\n')
    dep=read(SUR/'dependence/summary.json');tr=read(SUR/'traction/summary.json');fine_root=ROOT/'reports/joint-field-fine-v03';fine=read(fine_root/'summary.json');fa=read(fine_root/'aggregation-audit.json')
    assert dep['complete'] and tr['complete'] and fine['complete'] and fa['passed'],'Full dependence/traction/fine audit required'
    dg=lambda model:next(rr for rr in dep['summary'] if rr['shape']=='pooled' and rr['stage']=='active' and rr['model']==model)
    df,di=dg('full-joint'),dg('independent-force-moments')
    continuous_text=(rf"The continuous study completes {c['main_completed']} fits on fresh continuous-prior cases plus {c['diagnostic_completed']} independent seed/particle diagnostics. Of {c['comparison_count']} numerical sensitivity comparisons, {c['flagged_comparisons']} exceed predeclared descriptive thresholds; the full results remain reported. Continuous inference evaluates the same recorded acquisitions, not a newly optimized policy." if c['complete'] else 'Continuous-cohort evaluation is running; no partial-cohort conclusions are reported.')
    if draft and c['main_completed']==192 and not c['complete']:
        ac=next(rr for rr in c['summary'] if rr['population']=='continuous-coarse' and rr['model']=='continuous-smc' and rr['stage']=='active')
        fc=next(rr for rr in c['summary'] if rr['population']=='continuous-coarse' and rr['model']=='continuous-smc' and rr['stage']=='fixed')
        continuous_text=rf"The complete 64-case continuous-prior cohort gives post-adaptive RMSE {fmt(ac['rmse_mm'])}\,mm versus {fmt(fc['rmse_mm'])}\,mm for fixed mixed measurements. Prediction/force 90\% coverages are {pct(ac['coverage90'])}\%/{pct(ac['force_coverage90'])}\%. All 192 primary fits are complete; independent seed/particle diagnostics and their terminal audit remain underway. No final numerical-sensitivity or convergence conclusion is reported in this draft. Actions were selected by the finite planner, and continuous and finite priors differ."
    if c['complete']:
        ca=read(SUR/'continuous-independent-final-audit.json')
        assert ca['complete'] and ca['integrity_pass']
        assert ca['summary_sha256']==hashlib.sha256((SUR/'continuous-summary.json').read_bytes()).hexdigest()
        cg=lambda stage:next(rr for rr in c['summary'] if rr['population']=='continuous-coarse' and rr['model']=='continuous-smc' and rr['stage']==stage)
        ac,fc=cg('active'),cg('fixed')
        cp=next(rr for rr in ca['paired_comparisons'] if rr['comparison']=='continuous-active-minus-fixed' and rr['metric']=='rmse_mm')
        continuous_text=rf"On 64 continuous-prior cases, continuous inference gives post-adaptive RMSE {fmt(ac['rmse_mm'])}\,mm versus {fmt(fc['rmse_mm'])}\,mm for fixed mixed measurements. The posthoc paired 95\% interval for their difference is [{fmt(cp['descriptive_paired95'][0])},{fmt(cp['descriptive_paired95'][1])}]\,mm. Prediction/force 90\% coverages are {pct(ac['coverage90'])}\%/{pct(ac['force_coverage90'])}\%. All 192 main fits and 96 seed/particle diagnostics completed; {c['flagged_comparisons']} of {c['comparison_count']} descriptive sensitivity comparisons exceed the declared thresholds. These checks do not certify convergence. Actions were selected by the finite planner, and the continuous and finite priors differ."
    # Text reflects complete means; no fixed improvement assumptions beyond reported signed differences.
    text=rf'''\paragraph{{Prediction after additional evidence.}}
Across 128 matched surface-track cases, four adaptive scalar measurements change combined coordinate RMSE from {fmt(i['rmse_mm'])} to {fmt(a['rmse_mm'])}\,mm. The development-selected fixed mixed schedule gives {fmt(f['rmse_mm'])}\,mm, camera-only gives {fmt(v['rmse_mm'])}\,mm, and random selection gives {fmt(r['rmse_mm'])}\,mm. The paired adaptive-minus-fixed difference is {fmt(comparison['mean_difference'])}\,mm, with 95\% case-bootstrap interval [{fmt(comparison['ci95'][0])},{fmt(comparison['ci95'][1])}]. Against fixed camera-only, the paired difference is {fmt(view_comparison['mean_difference'])}\,mm [{fmt(view_comparison['ci95'][0])},{fmt(view_comparison['ci95'][1])}]. The adaptive policy selects {s['action_counts']['matched']['view']} camera and {s['action_counts']['matched']['probe']} probe measurements across 512 decisions. These are complete-cohort comparisons under the declared equal-weight target, not selection of favorable objects.

\paragraph{{Different measurements serve different predictions.}}
Adaptive selection gives hidden-motion/new-load errors {fmt(a['hidden_rmse_mm'])}/{fmt(a['changed_load_rmse_mm'])}\,mm. The probe-only policy gives {fmt(fp['hidden_rmse_mm'])}/{fmt(fp['changed_load_rmse_mm'])}\,mm; camera-only gives {fmt(v['hidden_rmse_mm'])}/{fmt(v['changed_load_rmse_mm'])}\,mm. A policy can therefore be useful for changed interactions without being best for reconstructing the original load sequence. The adaptive policy is worse than fixed mixed for new-load and material error; its advantage concerns the combined target. The shared objective makes this tradeoff explicit (Fig.~\ref{{fig:acquisition}}).

\paragraph{{Unknown force magnitude and calibrated evidence.}}
The independent 128-case scale-mixture study tests whether apparent material certainty is supplied by the force prior. Under the hierarchical model, four calibrated probes change new-load error from {fmt(hi['changed_load_rmse_mm'])} to {fmt(hp['changed_load_rmse_mm'])}\,mm and material relative error from {hi['material_relative_rmse']:.3f} to {hp['material_relative_rmse']:.3f}. Four views instead give hidden-motion error {fmt(hv['hidden_rmse_mm'])}\,mm, compared with {fmt(hi['hidden_rmse_mm'])}\,mm initially. Mean log-stiffness/log-force-scale posterior correlation is {hi['mean_logE_logtau_correlation']:.3f} initially and {hp['mean_logE_logtau_correlation']:.3f} after probes. These branches start from the same initial evidence.

After probes, hierarchical force intervals have {pct(hp['force_coverage90'])}\% coverage and {hp['force_interval_width_N']:.3f}\,N mean width; fixed nominal intervals have {pct(nom['force_coverage90'])}\% coverage and {nom['force_interval_width_N']:.3f}\,N width. The variance-matched broader Gaussian gives {pct(broad['force_coverage90'])}\% coverage and {broad['force_interval_width_N']:.3f}\,N width. The hierarchy is thus compared with an explicit wider-prior control, rather than interpreting width alone as calibrated identification.

\paragraph{{Dependence, continuous inference and mismatch.}}
After adaptive evidence, retaining joint dependence gives mean prediction-interval width {fmt(df['interval_width_mm'])}\,mm versus {fmt(di['interval_width_mm'])}\,mm for the moment-independence control, with coverages {pct(df['coverage90'])}\%/{pct(di['coverage90'])}\%. Hidden-motion errors are nearly equal ({fmt(df['hidden_rmse_mm'])}/{fmt(di['hidden_rmse_mm'])}\,mm); the benefit is interval efficiency, not a large mean-prediction gain. The supplement retains all point and representation controls. {continuous_text}

For the finite-dictionary estimator, matched adaptive prediction intervals cover {pct(a['coverage90'])}\% of evaluated coordinates at nominal 90\%. Coverage becomes {pct(g('active',pop='continuous-coarse')['coverage90'])}\% for continuous parameters, {pct(g('active',pop='offgrid-fine')['coverage90'])}\% for finer-mesh truth, and {pct(g('active',pop='force-prior')['coverage90'])}\% for the shorter-correlation force prior. These tests distinguish conditional numerical exactness from robustness to a different generator.
'''
    (SEC/'results-text.tex').write_text(text)
    shutil.copy2(SUR/'acquisition-table.tex',SEC/'acquisition-table.tex')
    tables=[]
    names={'matched':'Matched finite','continuous-coarse':'Continuous coordinates','offgrid-fine':'Finer-mesh truth','force-prior':'Force-correlation mismatch'}
    for pop,label in names.items():
      rows=[]
      for policy,step in [('active',0)]+[(p,4) for p in ['active','fixed','fixed-view','fixed-probe','random']]+[(p,0) for p in ['marginal-map','joint-map','uniform-material','fixed-interaction','independent-moments']]:
        rr=[r for r in s['summary'] if r['population']==pop and r['policy']==policy and r['step']==step]
        if not rr:continue
        r0=rr[0];rows.append(['Initial' if policy=='active' and step==0 else policy.replace('-',' '),r0['n'],step,fmt(r0['hidden_rmse_mm']),fmt(r0['changed_load_rmse_mm']),f"{r0['material_relative_rmse']:.3f}",f"{r0['force_rmse_N']:.3f}",pct(r0['coverage90']) if 'coverage90' in r0 else '--',pct(r0['material_coverage90']) if 'material_coverage90' in r0 else '--',pct(r0['force_coverage90']) if 'force_coverage90' in r0 else '--'])
      tables.append(table(label+r'. All errors are means of case-level metrics. Coverage is marginal at nominal 90\%. Point/representation controls use initial evidence; acquisition rows use four additional scalars. Dashes indicate intervals not evaluated for that control.','Policy & Cases & Added & Hidden mm & New-load mm & Material rel. & Force N & Pred. cov. & Mat. cov. & Force cov.',rows,'lrrrrrrrrr'))
    rows=[]
    for rr in h['means']:
        rows.append([rr['condition'],rr['model'],fmt(rr['hidden_rmse_mm']),fmt(rr['changed_load_rmse_mm']),f"{rr['material_relative_rmse']:.3f}",f"{rr['force_rmse_N']:.3f}",pct(rr['coverage90']),pct(rr['force_coverage90']),f"{rr['force_interval_width_N']:.3f}"])
    tables.append(table(r'Fresh hierarchical force-scale cohort (128 cases). View/probe branches are independent. Oracle receives true prior scale only. Width is mean force marginal 90\% interval width in N.','Evidence & Model & Hidden mm & New-load mm & Material rel. & Force N & Pred. cov. & Force cov. & Width N',rows,'llrrrrrrr'))
    rows=[]
    for rr in (c['summary'] if c['main_completed']==192 else []):
        if rr['population']!='continuous-coarse':continue
        rows.append([rr['stage'],rr['model'],rr['n'],fmt(rr['rmse_mm']),fmt(rr['changed_load_rmse_mm']),f"{rr['material_relative_rmse']:.3f}",pct(rr['coverage90']),pct(rr['force_coverage90'])])
    if rows:tables.append(table('Continuous-prior same-mesh cohort. 1024-particle SMC and finite dictionary receive identical evidence. Their priors differ; neither model selects new actions in this comparison.','Evidence & Model & Cases & Total mm & New-load mm & Material rel. & Pred. cov. & Force cov.',rows,'llrrrrrr'))
    rows=[]
    for rr in dep['summary']:
        if rr['shape']!='pooled':continue
        rows.append([rr['stage'],rr['model'].replace('-',' '),fmt(rr['hidden_rmse_mm']),pct(rr['hidden_coverage90']),fmt(rr['interval_width_mm'])])
    tables.append(table('Joint dependence versus independent force moments on all 128 matched cases. Changed-load predictions and material marginals are identical by construction. Width averages 24 evaluated prediction coordinates. The control preserves exact marginal first/second moments but Gaussianizes the force marginal.','Evidence & Model & Hidden mm & Hidden cov. & Width mm',rows,'llrrr'))
    rows=[[rr['stage'],rr['estimator'],f"{rr['traction_rmse_Pa']:.3f}",f"{rr['top_rmse_Pa']:.3f}",f"{rr['side_rmse_Pa']:.3f}"] for rr in tr['summary']]
    tables.append(table('Posthoc actual surface-traction diagnostic, all 128 matched cases. Area/time-weighted vector RMSE in Pa, averaged over cases; eight-time trapezoidal weighting. The independent control multiplies separate footprint/amplitude posterior means. It does not change fitted marginals.','Evidence & Estimator & Combined Pa & Top Pa & Side Pa',rows,'llrrr'))
    joint={rr['stage']:rr for rr in tr['summary'] if rr['estimator']=='joint'}
    text+=rf'''
\paragraph{{Actual spatial force distributions.}}
A separately declared posthoc diagnostic integrates traction errors over the loaded surfaces and all loading times. Joint posterior-mean traction RMSE changes from {joint['initial']['traction_rmse_Pa']:.2f} to {joint['active']['traction_rmse_Pa']:.2f}\,Pa after adaptive evidence; fixed mixed measurements give {joint['fixed']['traction_rmse_Pa']:.2f}\,Pa. Joint and product-marginal mean fields are nearly identical on this endpoint; the force-field improvement comes from additional evidence. This field metric complements the original one-dimensional load-basis proxy.
'''
    text+=rf'''
\paragraph{{A targeted temporal-prior repair.}}
On 64 fresh conditional stress cases, mixing correlation length as well as amplitude reduces hidden-motion error after four fixed views from {fmt(tamp['hidden_rmse_mm'])} to {fmt(tf['hidden_rmse_mm'])}\,mm. Force 90\% coverage rises from {pct(tamp['force_coverage90'])}\% to {pct(tf['force_coverage90'])}\%, while width rises from {tamp['force_interval_width_N']:.3f} to {tf['force_interval_width_N']:.3f}\,N. The known-covariance oracle gives {pct(tor['force_coverage90'])}\% coverage. Material relative error worsens from {tamp['material_relative_rmse']:.3f} to {tf['material_relative_rmse']:.3f}; improved force interpolation does not establish better material recovery. This posthoc follow-up retains the original mismatch failures.
'''
    (SEC/'results-text.tex').write_text(text)
    temporal_rows=[]
    for rr in tm['means']:
        temporal_rows.append([rr['condition'],rr['model'],fmt(rr['hidden_rmse_mm']),fmt(rr['changed_load_rmse_mm']),f"{rr['material_relative_rmse']:.3f}",pct(rr['coverage90']),pct(rr['force_coverage90']),f"{rr['force_interval_width_N']:.3f}"])
    ttext=table('Posthoc temporal-prior follow-up, all 64 fresh cases. True correlation length 0.10; conditional stress coverage, not prior-averaged calibration. Force interval width is in N. Fixed views and probes branch independently.','Evidence & Model & Hidden mm & Reset mm & Material rel. & Pred. cov. & Force cov. & Width N',temporal_rows,'llrrrrrr')
    ttext+=r'''
\begin{figure*}[t]
\centering\includegraphics[width=.98\textwidth]{figures/temporal-comparison.pdf}
\caption{\textbf{Temporal uncertainty partially repairs a recognized failure.} Fresh 64-case conditional stress test after four fixed views. Error, coverage and width are shown together, with stratified case-bootstrap 95\% intervals. The covariance oracle is given true amplitude scale and correlation length, not true material or force history.}
\end{figure*}
'''
    (SEC/'temporal-results.tex').write_text(ttext)
    shutil.copy2(ROOT/'output/joint-field-temporal-v06/comparison.pdf',PAPER/'figures/temporal-comparison.pdf')
    fr=lambda model,policy='active',step=4:next(rr for rr in fine['summary'] if rr['model']==model and rr['policy']==policy and rr['step']==step)
    co,fi=fr('coarse8'),fr('fine24');co0,fi0=fr('coarse8',step=0),fr('fine24',step=0)
    text+=rf'''
\paragraph{{Forward-model discretization.}}
In the separate 32-case interior-sensor reference, refining the inverse model changes post-adaptive prediction coverage from {pct(co['coverage90'])}\% to {pct(fi['coverage90'])}\%. Overall RMSE changes modestly, {fmt(co['rmse_mm'])} to {fmt(fi['rmse_mm'])}\,mm; its paired interval includes zero improvement. Numerical accuracy of the forward model therefore matters for physical uncertainty as well as response error.
'''
    (SEC/'results-text.tex').write_text(text)
    fine_text=rf'''On 32 fresh high-resolution truths, initial coordinate RMSE is {fmt(co0['rmse_mm'])}\,mm for the coarse inverse model and {fmt(fi0['rmse_mm'])}\,mm for the finer model. After their respective four adaptive measurements, errors are {fmt(co['rmse_mm'])}/{fmt(fi['rmse_mm'])}\,mm and nominal 90\% coverages are {pct(co['coverage90'])}\%/{pct(fi['coverage90'])}\%. Active actions may differ across inverse models; fixed schedules and potential observation noise remain paired. This comparison tests numerical model mismatch with interior sensors, not raw-video performance.
'''
    fine_rows=[]
    for policy,step in [('active',0),('active',4),('fixed',4),('fixed-view',4),('fixed-probe',4),('random',4)]:
      for model in ['coarse8','fine24']:
        rr=fr(model,policy,step);fine_rows.append([model,'initial' if step==0 else policy,fmt(rr['rmse_mm']),fmt(rr['hidden_rmse_mm']),fmt(rr['changed_load_rmse_mm']),f"{rr['material_relative_rmse']:.3f}",pct(rr['coverage90'])])
    fine_text+=table('Paired coarse/fine inverse reference on 32 independently generated 32-mesh truths. Both inverse models use the same 3125 parameter atoms. Sensors are interior material coordinates.','Inverse & Evidence & Total mm & Hidden mm & New-load mm & Material rel. & Pred. cov.',fine_rows,'llrrrrr')
    (SEC/'fine-results.tex').write_text(fine_text)
    (SEC/'supplement-tables.tex').write_text('\n\n'.join(tables))
    for name in ['ambiguity','method','acquisition','acquisition-prediction','coverage','coverage-prediction','changed-load-comparison','material-fields','force-fields','spatial-force-fields']:
        shutil.copy2(ROOT/f'output/joint-field-surface-v04/{name}.pdf',PAPER/f'figures/{name}.pdf')
    for source,dest in [('summary','hierarchy-summary'),('prior-models','hierarchy-prior-models'),('case0-joint-posterior','hierarchy-joint-posterior')]:
        shutil.copy2(ROOT/f'output/joint-field-hierarchy-surface-v05/{source}.pdf',PAPER/f'figures/{dest}.pdf')
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SUR/'summary.json',HIER/'summary.json',HIER/'completion-audit.json',HIER/'finalizer.json',HIER/'condition-comparisons.json',SUR/'continuous-summary.json',SUR/'evidence-audit.json',SUR/'dependence/summary.json',SUR/'traction/summary.json',fine_root/'summary.json',fine_root/'aggregation-audit.json',temporal_root/'summary.json',temporal_root/'completion-audit.json',temporal_root/'protocol.json',temporal_root/'independent-final-audit.json']}
    if c['complete']:
        p=SUR/'continuous-independent-final-audit.json';provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    phroot=SUR/'continuous/posthoc4096'
    if (phroot/'manifest.json').exists():
        phstatus=read(phroot/'status.json')
        if not draft:assert phstatus['status']=='complete' and not phstatus['failures'],'Posthoc numerical audit must finish'
        if phstatus['status']=='complete' and not phstatus['failures']:
            phd=read(phroot/'diagnosis.json')
            large=[rr for rr in phd['records'] if rr['particles']==4096]
            pair=next(rr for rr in phd['comparisons'] if '-n4096-' in rr['first'] and '-n4096-' in rr['second'])
            original=[rr for rr in phd['comparisons'] if '-n4096-' not in rr['first'] and '-n4096-' not in rr['second']]
            cross=[rr for rr in phd['comparisons'] if ('-n4096-' in rr['first']) != ('-n4096-' in rr['second'])]
            assert read(phroot/'independent-final-audit.json')['integrity_pass']
            outcome=('agree within' if not pair['flag_cdf'] and not pair['flag_prediction'] else 'still exceed')
            phtext=rf"\paragraph{{Targeted posthoc numerical check.}} Sensitivity flags for the block's matched case 1 after fixed measurements motivated two fresh-seed, 4096-particle repeats. The original 1024/2048 runs differ by up to {max(rr['max_cdf'] for rr in original):.4f} in parameter marginal CDF and {max(rr['prediction_mean_rms_mm'] for rr in original):.4f}\,mm in target-mean RMS. Strong support/load and support/stiffness correlations are consistent with inefficient diagonal random-walk exploration; these summaries do not establish a separate missed mode. With the same four Metropolis moves per stage, the two larger runs differ by {pair['max_cdf']:.4f} in CDF and {pair['prediction_mean_rms_mm']:.4f}\,mm in target-mean RMS, and {outcome} the original descriptive thresholds. Their final Metropolis acceptance rates are {100*large[0]['final_acceptance']:.1f}\%/{100*large[1]['final_acceptance']:.1f}\%; resampled equal weights are not a convergence diagnostic. The maximum CDF difference between a 4096 run and a lower-count run is still {max(rr['max_cdf'] for rr in cross):.4f}. All original flags and fits remain reported. This selected-case follow-up neither replaces the primary cohort nor certifies global convergence."
            (SEC/'continuous-posthoc.tex').write_text(phtext+'\n')
            for name in ('manifest.json','status.json','diagnosis.json','independent-final-audit.json'):
                p=phroot/name;provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        else:(SEC/'continuous-posthoc.tex').write_text(r'\paragraph{Numerical follow-up pending.} Two additional 4096-particle checks of the flagged case are running; no follow-up conclusions are included in this draft.'+'\n')
    (PAPER/'results-provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Draft results inserted; final numerical audit still required' if draft else 'Final complete results inserted')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--draft',action='store_true',help='Mark draft; omit incomplete continuous conclusions');args=ap.parse_args();main(draft=args.draft)
