"""Publication figures from committed evaluator summaries, never predictor input."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.benchmark_improvement_v14.common import REPORT,read


def main(dataset):
    label='softvt' if dataset=='SoftVTBench' else 'phystwin'
    report=read(REPORT/(label+'-summary.json'));baseline=report['primary_baseline']
    groups=['all_historical_exposed']
    if dataset=='SoftVTBench':groups.append('prospective_softvt_confirmation')
    fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),5),squeeze=False)
    for ax,group in zip(axes[0],groups):
        result=report['groups'][group][baseline];pairs=result['per_case']
        x=np.array([p['baseline_mm'] for p in pairs]);y=np.array([p['proposal_mm'] for p in pairs])
        complete=np.array([p['complete_pair'] for p in pairs]);limit=max(x.max(),y.max())*1.06
        ax.plot([0,limit],[0,limit],color='.6',lw=1,ls='--',label='Same error')
        ax.scatter(x[complete],y[complete],c=np.where(y[complete]<x[complete],'#117a65','#b03a2e'),s=32,alpha=.8)
        if (~complete).any():ax.scatter(x[~complete],y[~complete],marker='x',c='#7d3c98',label='Partial frame coverage')
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Original model error (mm)',ylabel='New model error (mm)')
        title='Historical episodes — exposed' if group=='all_historical_exposed' else 'Prospective episodes — frozen method'
        ax.set_title(title+'\n'+str(len(pairs))+' paired episodes',fontsize=12)
        mean_label='Full-suffix means' if complete.all() else 'Available-frame means'
        ax.text(.04,.96,f"{mean_label}: {x.mean():.2f} → {y.mean():.2f} mm\n{result['wins']} better / {result['losses']} worse",
                transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='.85',alpha=.95))
        ax.set_aspect('equal');ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
        if (~complete).any():ax.legend(loc='lower right',fontsize=8)
    metric='Mean per-frame coordinate RMSE' if dataset=='SoftVTBench' else 'Manual-track mean distance'
    fig.suptitle(dataset+' prediction benchmark · '+metric,fontsize=14)
    fig.text(.5,.025,'Below the diagonal is better. Repeated objects; episode counts do not imply independent-object generalization.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    output=REPORT/'figures';output.mkdir(exist_ok=True)
    fig.savefig(output/(label+'-comparison.png'),dpi=180)
    fig.savefig(output/(label+'-comparison.svg'))
    plt.close(fig)


if __name__=='__main__':main(sys.argv[1])
