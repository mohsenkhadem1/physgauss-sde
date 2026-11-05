"""Summarize already scored predictive intervals without reading raw targets."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('reports/gaussian-boundary-flow-v02')
CHANNELS = ('flow_raw','mixture_raw','flow_pose','mixture_pose')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(paths, expected):
    intervals = defaultdict(dict); draws = {}; provenance = {}
    for path, digest in paths.items():
        if not path.endswith('.json') or 'scores' not in Path(path).name: continue
        if sha(path) != digest: raise ValueError('Score changed: '+path)
        data = json.loads(Path(path).read_text())
        if 'posterior_intervals' not in data: continue
        provenance[path] = digest
        for name, value in data['posterior_intervals'].items():
            if name in CHANNELS and value.get('available'):
                if data['case'] in intervals[name]: raise ValueError('Duplicate case/channel')
                intervals[name][data['case']] = value
        for name, value in (data.get('forecast_samples') or {}).items():
            entry = dict(requested=value['requested'], valid=value['valid'], failed=len(value['failures']))
            key = (data['case'],name)
            if key in draws and draws[key] != entry: raise ValueError('Inconsistent draw counts')
            draws[key] = entry
    methods = {}
    for name, cases in intervals.items():
        methods[name] = dict(expected_cases=expected, scored_cases=len(cases),
                            mean_coordinate_coverage=float(np.mean([v['coordinate_coverage'] for v in cases.values()])),
                            mean_width_mm=float(np.mean([v['average_width_mm'] for v in cases.values()])),
                            per_case=cases)
    families = {}
    for name in ('flow','mixture'):
        rows = [v for (_,family),v in draws.items() if family == name]
        families[name] = dict(cases_with_draw_metadata=len(rows), **{key:sum(v[key] for v in rows) for key in ('requested','valid','failed')})
    return dict(methods=methods, draws=families, source_scores_sha256=provenance)


def main():
    soft = json.loads((ROOT/'softvt-validation/verified-summary.json').read_text())
    phys = json.loads((ROOT/'phystwin-all-eight-comparison.json').read_text())
    isaac = json.loads((ROOT/'isaac-recorded-evaluation/summary.json').read_text())
    datasets = {'SoftVT':collect(soft['source_sha256'],12), 'PhysTwin':collect(phys['score_sources_sha256'],8)}
    datasets['Isaac'] = dict(methods={})
    for name in ('flow','mdn','25d','chart'):
        row = isaac['methods'][name]; m = row['episode_macro']
        datasets['Isaac']['methods'][name] = dict(expected_cases=76, scored_cases=row['complete'],
            mean_coordinate_coverage=m['coordinate90_coverage'], mean_width_mm=1000*m['coordinate90_width_m'],
            active_parameter_coverage=m['active_parameter90_coverage'],
            active_parameter_normalized_width=m['active_parameter90_normalized_width'],
            category_accuracy=row['category_accuracy'])
    result = dict(nominal_coverage=.9, aggregation='Equal-case mean of marginal coordinate coverage, not independent-coordinate significance or joint coverage',
                  datasets=datasets, scope='Primary frozen predictions; deterministic ridge excluded from uncertainty comparisons',
                  limitations=['FEM intervals condition on solver-successful samples; failed draws are retained in the counts.',
                               'SoftVT has one failed bank among twelve cases; coverage is conditional on the eleven scored cases.',
                               'PhysTwin coverage uses observed reconstructed tracks, not trusted material/force labels.',
                               'Isaac parameter truth comes from simulator recipes, not real-world measurements.',
                               'Low coverage is evidence against using these intervals as calibrated confidence for action selection.'])
    (ROOT/'campaign-uncertainty.json').write_text(json.dumps(result,indent=2)+'\n')
    labels={'flow_raw':'Flow raw','mixture_raw':'Mixture raw','flow_pose':'Flow + contact','mixture_pose':'Mixture + contact',
            'flow':'Flow (new)','mdn':'Mixture (new)','25d':'Archived 25D','chart':'Archived chart'}
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none','svg.hashsalt':'gbf-uncertainty-v01'})
    fig,axes=plt.subplots(3,1,figsize=(10,9))
    lines=['PREDICTIVE UNCERTAINTY: PRIMARY CAMPAIGN','',result['aggregation'],'']
    for ax,(dataset,contents) in zip(axes,datasets.items()):
        rows=list(contents['methods'].items())
        for i,(name,row) in enumerate(rows):
            value=100*row['mean_coordinate_coverage'];color='#1769aa' if name.startswith('flow') else '#d87528' if ('mixture'in name or name=='mdn') else '#626e78'
            ax.scatter(value,i,color=color,s=55,zorder=3)
            ax.annotate(f'{value:.1f}%  ({row["scored_cases"]}/{row["expected_cases"]} cases)',(value,i),
                        xytext=(-8,0) if value>70 else (6,0), ha='right' if value>70 else 'left',
                        textcoords='offset points',va='center',fontsize=10)
            lines.append(f'{dataset:9s} {labels[name]:20s} coverage {value:6.2f}%; width {row["mean_width_mm"]:8.3f} mm; cases {row["scored_cases"]}/{row["expected_cases"]}')
        ax.axvline(90,color='#b33c39',linestyle='--',linewidth=1.3,label='Nominal 90%')
        ax.set_yticks(range(len(rows)),[labels[name]for name,_ in rows]);ax.set_ylim(len(rows)-.4,-.6);ax.set_xlim(0,115)
        ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis='x',color='#dce1e6');ax.set_axisbelow(True)
        ax.set_title(dataset,loc='left',fontsize=12);ax.set_xlabel('Measured coordinate coverage (%)')
        for side in ('top','right','left'):ax.spines[side].set_visible(False)
        ax.tick_params(axis='y',length=0)
    fig.suptitle('Nominal 90% intervals: measured coverage',fontsize=17,x=.04,ha='left')
    fig.text(.04,.01,'Dashed red line: nominal 90%. Episode-averaged marginal coverage; coordinates are dependent.\nFEM results condition on successful trajectories. These are predictive intervals, not proof of physical identifiability.',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95),h_pad=2)
    fig.savefig(ROOT/'campaign-uncertainty.svg',metadata={'Date':None});fig.savefig(ROOT/'campaign-uncertainty.png',dpi=150);plt.close(fig)
    lines += ['', 'FEM draw accounting']
    for dataset in ('SoftVT','PhysTwin'):
        for family,row in datasets[dataset]['draws'].items():lines.append(f'{dataset} {family}: {row}')
    lines += ['', 'Limitations'] + result['limitations']
    (ROOT/'campaign-uncertainty.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
