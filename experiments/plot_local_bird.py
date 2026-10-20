"""Plot observed prefix/full convergence diagnostics without extrapolation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root,output):
    root=Path(root);rows=[]
    for path in root.glob('preview-*/preview.json'):
        d=json.loads(path.read_text());rows.append((d['preview_draws_per_chain'],d,False))
    for path in root.glob('diagnostics-*/convergence.json'):
        d=json.loads(path.read_text());rows.append((d['draws_per_chain'],d,True))
    rows.sort(key=lambda row:row[0])
    if not rows:raise ValueError('No recorded convergence diagnostics')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    x=[n for n,_,_ in rows]
    axes[0].plot(x,[d['worst_rhat'] for _,d,_ in rows],color='#226789',lw=1.5)
    axes[0].axhline(1.01,color='#a13f35',ls='--',lw=1,label='Required <1.01')
    axes[0].set_ylabel('Worst rank/folded split Rhat');axes[0].legend(fontsize=8)
    for key,label,color in [('minimum_bulk_ess','Bulk','#226789'),('minimum_tail_ess','Tail','#b96626')]:
        axes[1].plot(x,[d[key] for _,d,_ in rows],label=label,color=color,marker='o',markersize=4)
    axes[1].axhline(400,color='#a13f35',ls='--',lw=1,label='Required ≥400')
    axes[1].set_ylabel('Minimum ESS across all monitors');axes[1].set_yscale('log');axes[1].legend(fontsize=8)
    axes[2].plot(x,[d['failed_quantities'] for _,d,_ in rows],color='#226789',lw=1.5)
    axes[2].set_ylabel('Failed scalar diagnostics');axes[2].set_ylim(bottom=0)
    for n,d,full in rows:
        face='#2d8d60' if full and d.get('all_monitored_quantities_pass') else ('#226789' if full else 'white')
        axes[0].scatter(n,d['worst_rhat'],s=38,edgecolor='#226789',facecolor=face,zorder=3)
        axes[2].scatter(n,d['failed_quantities'],s=38,edgecolor='#226789',facecolor=face,zorder=3)
    for ax in axes:ax.set_xlabel('Retained draws per chain');ax.grid(alpha=.2)
    fig.suptitle('Bird: observed-only diagnostics across all 733 monitors',fontsize=12)
    fig.supxlabel('Open markers: prefix audits. Full gates require at least 2,000 draws per chain; no predictive calibration claim.',fontsize=8)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/bird-local-production')
    p.add_argument('--output',required=True)
    args=p.parse_args();plot(args.run,args.output)
