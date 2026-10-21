"""Publish saved VAL metrics/plots; never reads TEST or simulator episode files."""
import argparse,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=('25d-rgb','25d-rgbd','25d-prior','chart-rgb','chart-rgbd','chart-prior')
LABELS=('25D RGB','25D RGBD','25D prior','Chart RGB','Chart RGBD','Chart prior')
COLORS=('#c76e32','#29749c','#9a9a9a','#e4a262','#67a4bc','#bdbdbd')


def is_old(row):return row['key'].startswith('isaac-boundary-orbit-v0.1/')


def subset(rows,which):
    return [r for r in rows if which=='all' or is_old(r)==(which=='old')]


def group_bootstrap(rows,metric,baseline,groups,seed=9211,count=2000):
    """Stratified independent-layout resampling; old variants stay together."""
    rng=np.random.default_rng(seed);strata=[]
    for old in (True,False):
        chosen=[r for r in rows if is_old(r)==old]
        table={}
        for row in chosen:table.setdefault(groups[row['key']],[]).append(row)
        strata.append(list(table.values()))
    gains=[]
    for _ in range(count):
        selected=[]
        for stratum in strata:
            selected.extend(r for i in rng.integers(len(stratum),size=len(stratum)) for r in stratum[i])
        gains.append(100*(1-np.mean([r[metric] for r in selected])/np.mean([r[baseline] for r in selected])))
    return np.quantile(gains,[.025,.975]).tolist()


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def build(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Reports are immutable; choose a new output')
    output.mkdir(parents=True);(output/'source.py').write_bytes(Path(__file__).read_bytes())
    catalog=json.loads(Path('runs/gaussian-final-passive-catalog.json').read_text())
    groups={r['key']:r['group_id'] for r in catalog['cases']}
    full={m:json.loads(Path(f'runs/gaussian-final-val-{m}-full-v01/results.json').read_text()) for m in MODELS}
    prefix={m:json.loads(Path(f'runs/gaussian-final-val-{m}-prefix-v01/results.json').read_text()) for m in MODELS}
    params={m:json.loads(Path(f'runs/gaussian-final-parameters-{m}-v01/results.json').read_text()) for m in MODELS}
    windows=[('Withheld 1–2 s gap: ten input views',full,'gap_coordinate_rmse_m','static_gap_coordinate_rmse_m'),
             ('Future 2–3 s: inputs only ≤1 s',prefix,'late_2to3_coordinate_rmse_m','static_late_2to3_coordinate_rmse_m')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    summaries=[]
    for ax,(title,data,metric,baseline) in zip(axes,windows):
        for index,(model,color) in enumerate(zip(MODELS,COLORS)):
            rows=data[model]['episodes'];gain=100*(1-np.mean([r[metric] for r in rows])/np.mean([r[baseline] for r in rows]))
            ci=group_bootstrap(rows,metric,baseline,groups)
            ax.plot(ci,[index,index],color=color,lw=2);ax.plot(gain,index,'o',color=color,ms=7)
            summaries.append({'model':model,'task':title,'gain_percent':gain,'layout_bootstrap95':ci})
        ax.axvline(0,color='#555',lw=1,ls='--');ax.set_title(title);ax.set_xlabel('Improvement over static (%)');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(6),LABELS);axes[0].invert_yaxis();fig.tight_layout()
    fig.savefig(output/'predictive-gain.png',dpi=180);fig.savefig(output/'predictive-gain.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,which,title in zip(axes,['fresh','old'],['Fresh 32 layouts','Original 12 episodes / 4 layouts']):
        coverage=[100*np.mean([r['material']['youngs_modulus_Pa']['covered90'] for r in subset(params[m]['episodes'],which)]) for m in MODELS]
        ax.barh(range(6),coverage,color=COLORS);ax.axvline(90,color='#a22',ls='--',lw=1,label='Nominal 90%')
        ax.set_title(title);ax.set_xlim(0,100);ax.set_xlabel('Empirical Young’s modulus 90% coverage (%)');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(6),LABELS);axes[0].invert_yaxis();fig.tight_layout()
    fig.savefig(output/'material-coverage.png',dpi=180);fig.savefig(output/'material-coverage.pdf');plt.close(fig)
    sections=[]
    for title,data,metric,baseline in windows:
        first=data[MODELS[0]]['episodes']
        rows=[['Static']+[f'{1000*np.mean([r[baseline] for r in subset(first,w)]):.3f}' for w in ('all','fresh','old')]]
        rows += [[label]+[f'{1000*np.mean([r[metric] for r in subset(data[m]["episodes"],w)]):.3f}' for w in ('all','fresh','old')] for m,label in zip(MODELS,LABELS)]
        sections.append('<h2>'+title+'</h2>'+table(['Model','All 44 (mm)','Fresh 32 (mm)','Original 12 (mm)'],rows))
    for which,title in [('fresh','Physical credibility: fresh 32'),('old','Physical credibility: original 12 / 4 layouts')]:
        rows=[]
        for model,label in zip(MODELS,LABELS):
            rs=subset(params[model]['episodes'],which);material=[r['material']['youngs_modulus_Pa'] for r in rs]
            rows.append([label,f'{100*np.mean([r["covered90"] for r in material]):.1f}%',
                f'{np.mean([r["absolute_error"] for r in material])/1000:.2f}',f'{np.mean([r["width90"] for r in material])/1000:.2f}',
                f'{100*np.mean([r["boundary_type"]["correct"] for r in rs]):.1f}%',f'{100*np.mean([r["force_type"]["correct"] for r in rs]):.1f}%',
                f'{np.mean([r["force_type"]["brier"] for r in rs]):.3f}'])
        sections.append('<h2>'+title+'</h2>'+table(['Model','E 90% coverage','E MAE(kPa)','E interval width(kPa)','BC accuracy','Force accuracy','Force Brier↓'],rows))
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaussian deformation: validation report</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#202b35}h1{line-height:1.2}h2{margin-top:2em}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 10px;text-align:right;border-bottom:1px solid #dde2e6}th:first-child,td:first-child{text-align:left}th{background:#eef2f4}.finding{border-left:5px solid #ba6737;background:#faf3eb;padding:16px 22px}.note{color:#53616b;font-size:14px}a{color:#23698a}code{font-size:13px}</style>
<h1>Gaussian deformation prediction: validation evidence</h1><p>152 TRAIN episodes; 44 VAL episodes. No predictive TEST evaluation is included.</p>
<div class="finding"><strong>Held-out gap prediction improved. Physical uncertainty remains unreliable.</strong><p>The 25D RGB/RGBD baselines improve the1–2 s withheld gap. Only the depth-assisted chart model improves the2–3 s future-window mean on this validation set, by 2.93%; RGB-only future prediction remains worse than static. Surface/force constraints make hypotheses valid but do not establish parameter recovery or calibration.</p></div>
<p>All models use the same frozen neural Green teacher and known cuboid reference. Runtime is an observation encoder plus sampled physical hypotheses and a neural decoder: no FEM or per-video parameter optimization. RGBD explicitly uses measured depth and is a separate track.</p>
<h2>Paired predictive gain</h2><img src="predictive-gain.png" alt="Model gains with layout bootstrap intervals"><p class="note">Exploratory 95% bootstrap intervals resample independent layouts within fresh/original strata; original material variants stay together. These VAL cases also selected checkpoints, so this is not confirmatory evidence. The original 12 episodes comprise only 4 layouts × 3 material variants.</p>
'''+''.join(sections[:2])+'''
<h2>Material uncertainty did not calibrate</h2><img src="material-coverage.png" alt="Young modulus coverage separately for fresh and original validation"><p>Pointwise coordinate coverage near90% cannot validate the inferred physical parameters. Force probabilities are often worse than the uniform4-class Brier score 0.75. Mean support/load estimates remain wrong by several centimetres and force estimates by approximately the force scale. Original endpoint materials and fresh continuous material draws are reported separately.</p>
'''+''.join(sections[2:])+'''
<h2>Scope and reproducibility</h2><ul><li>Chart training uses 50% exact ≤1 s prefixes and 50% random frame dropout. The preserved25D baseline uses random dropout only; the forecast difference cannot be attributed to the chart alone.</li><li>The chart is specific to this known cuboid and declared force families. It does not demonstrate arbitrary COLMAP geometry or generalized 3D boundary fields.</li><li>The chart density is 16D intrinsic coefficients plus discrete categories, including nuisance padding. Its noninvertible physical mapping has no 25D Lebesgue density claim.</li><li>Intervals and categorical probabilities remain uncalibrated development estimates. No test-based tuning is included.</li></ul>
<p><a href="../../runs/gaussian-final-validation-summary-v01.json">Machine-readable validation summary</a> · <a href="../../runs/gaussian-final-passive-seal.json">Sealed checkpoint identities</a> · <a href="predictive-gain.pdf">Predictive plot PDF</a> · <a href="material-coverage.pdf">Coverage plot PDF</a></p></html>'''
    (output/'index.html').write_text(body)
    (output/'bootstrap-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return {'report':str(output/'index.html'),'no_test_data_read':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();print(json.dumps(build(args.output),indent=2))
