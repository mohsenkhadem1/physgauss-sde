"""Reporting-only fixed figure design; never part of prediction or selection.

Run only after root confirms full reserved scoring. An explicit complete-score
SHA is mandatory. No method/case selection depends on observed performance.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=('initial_only','fixed_views','fixed_all','random_probes','visual_uncertainty','physical_risk')
LABELS=('Initial','Views','Fixed best','Random probes','Visual uncertainty','Physical risk')
COLORS=('#a8afb8','#6d98bd','#307a5e','#d9a54a','#9672b2','#d35c4d')
METRICS=(('aligned_shape_rmse_mm','shape_m','Shape error','mm'),
         ('reaction_force_rmse_N','force_N','Reaction force error','N'),
         ('reaction_torque_rmse_Nm','torque_Nm','Reaction torque error','N m'))
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_data(report):
    data=[]
    for track in ('V','VF'):
        for method,label,color in zip(METHODS,LABELS,COLORS):
            matches=[r for r in report['summary'] if r['initial_camera']=='all' and r['track']==track and r['policy']==method]
            if len(matches)!=1:raise ValueError('Exactly one complete summary required per fixed figure method')
            row=matches[0]
            if row['worlds']!=12 or row['count']!=48:raise ValueError('Figures require all twelve worlds and four repeats')
            for metric,modality,title,unit in METRICS:
                error=row['metrics'][metric];coverage=row['proper_scores'][modality+'_central90_coordinate_coverage']
                width=row['proper_scores'][modality+'_mean_interval_width'];crps=row['proper_scores'][modality+'_crps_mean']
                if not np.isfinite([error,coverage,width,crps]).all() or min(error,width,crps)<0 or not 0<=coverage<=1:raise ValueError('Finite nonnegative full-cohort statistics required')
                data.append(dict(track=track,policy=method,label=label,color=color,metric=metric,modality=modality,
                                 error=error,error_unit=unit,marginal90_coverage=coverage,mean_interval_width=width,crps_mean=crps,
                                 probabilistic_unit='m' if modality=='shape_m' else unit,worlds=12,noise_repeats=4))
    return data


def render(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.4),constrained_layout=True)
    for ti,track in enumerate(('V','VF')):
        for mi,(metric,modality,title,unit) in enumerate(METRICS):
            ax=axes[ti,mi];rows=[r for r in data if r['track']==track and r['metric']==metric]
            ax.bar(np.arange(6),[r['error'] for r in rows],color=COLORS,width=.72)
            ax.set_xticks(range(6),LABELS,rotation=35,ha='right');ax.set_ylabel(f'{title} ({unit})')
            ax.set_title(('Vision only' if track=='V' else 'Vision + reactions')+' · '+('Shape','Force','Torque')[mi])
            ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Initial and two-query predictions: 12 worlds, four noise repeats each',fontsize=15)
    for ext in ('png','svg'):fig.savefig(out/f'response-errors.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.4),constrained_layout=True)
    markers=('o','s','D','v','P','*')
    for ti,track in enumerate(('V','VF')):
        for mi,(metric,modality,title,unit) in enumerate(METRICS):
            ax=axes[ti,mi];rows=[r for r in data if r['track']==track and r['metric']==metric]
            for r,marker in zip(rows,markers):
                ax.scatter(r['error'],100*r['marginal90_coverage'],color=r['color'],marker=marker,s=90 if marker=='*' else 55,label=r['label'])
            ax.axhline(90,color='#555555',ls='--',lw=1,label='Nominal 90%')
            ax.set_ylim(-3,103);ax.set_xlim(left=0);ax.set_xlabel(f'{title} ({unit})');ax.set_ylabel('Marginal 90% interval coverage (%)')
            ax.set_title(('Vision only' if track=='V' else 'Vision + reactions')+' · '+('Shape','Force','Torque')[mi]);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=7,fontsize=9)
    fig.suptitle('Accuracy and uncertainty: lower error does not guarantee reliable intervals',fontsize=15)
    for ext in ('png','svg'):fig.savefig(out/f'accuracy-coverage.{ext}',dpi=180)
    plt.close(fig)
    notes=('All 12 worlds retained; equal repeat and world weighting. Initial uses no added queries; all other displayed methods use two. '
           'Fixed best is selected on development data, not these outcomes. Random probes averages six pair errors. '
           'Coverage is coordinate-marginal; its nominal line is not an assertion of calibration. '
           'Finite-ensemble exploratory study with unresolved observation/posterior numerical convergence. '
           'Scatter shape error is pair-aligned prediction error; shape intervals concern canonical nonrigid coordinates. '
           'Panels have separate physical units and axis scales. No uncertainty bars are invented from correlated coordinates.')
    (out/'FIGURE-NOTES.txt').write_text(notes+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--scores-ready-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sha(a.scores)!=a.scores_ready_sha256:raise ValueError('Root-confirmed complete score SHA required')
    report=json.loads(a.scores.read_text());data=plot_data(report)
    render(data,a.output)
    numeric=dict(scope='Fixed full-cohort reporting only; no outcome-dependent method or case selection',
                 input_scores_sha256=a.scores_ready_sha256,protocol_sha256=report['protocol_sha256'],method_freeze_sha256=report['method_freeze_sha256'],
                 source_sha256=sha(__file__),data=data,paired_world_bootstrap=report['paired_world_bootstrap'])
    (a.output/'plot-data.json').write_text(json.dumps(numeric,indent=2)+'\n')
    if sha(a.scores)!=a.scores_ready_sha256:raise ValueError('Scores changed during rendering')
    artifacts={path.name:sha(path) for path in a.output.iterdir() if path.is_file()}
    (a.output/'figure-build.json').write_text(json.dumps(dict(input_scores_sha256=a.scores_ready_sha256,source_sha256=sha(__file__),artifact_sha256=artifacts,
                                                           matplotlib_version=matplotlib.__version__,visual_review='pending'),indent=2)+'\n')
if __name__=='__main__':main()
