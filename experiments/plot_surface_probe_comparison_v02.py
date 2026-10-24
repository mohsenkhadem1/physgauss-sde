"""Plot complete scalar study results; no raw data or partial-case filtering."""
import argparse, json
from pathlib import Path
import numpy as np


def main(args):
    result=json.loads(Path(args.summary).read_text())
    if not result['all_comparison_inputs_qualified'] or result['objects']!=16:
        raise ValueError('All16 physical and numerical inputs required; no selective plot')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=('information','uniform_random','fixed_0','fixed_1','fixed_2')
    labels=('Information heuristic','Uniform random','Fixed at 0.09 m','Fixed at 0.18 m','Fixed at 0.27 m')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
    for row,(law,title) in enumerate((('matching_linear','Matching linear model'),('NH80_stress','Nonlinear reference data'))):
        for col,(family,budget) in enumerate((('normal','Bending: 0.08 N'),('paired_axial','Axial: 0.18 N per face'))):
            ax=axes[row,col];report=result['laws'][law]['families'][family]
            differences=report['paired_baseline_minus_proposed_RMS_m']
            means=np.array([differences[p]['mean'] for p in policies])*1e6
            intervals=np.array([differences[p]['CI95'] for p in policies])*1e6
            for j,(mean,CI) in enumerate(zip(means,intervals)):
                color='#207a8a' if CI[0]>0 else '#666666'
                # Draw endpoints directly: a percentile CI need not surround
                # the observed bootstrap point estimate in a small sample.
                ax.plot(CI,[j,j],color=color,lw=2)
                ax.scatter([mean],[j],color=color,s=32,zorder=3)
            ax.axvline(0,color='#a84437',ls='--',lw=1)
            ax.set_yticks(np.arange(5),labels);ax.set_ylim(4.6,-.6)
            ax.set_title(title+' · '+budget,fontsize=11)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
            ax.set_xlabel('Baseline error − uncertainty-policy error (µm)')
    fig.suptitle('Held-force prediction: uncertainty policy versus simple probes',fontsize=15)
    fig.text(.5,.025,'Positive favors uncertainty policy. 95% paired object-bootstrap intervals; 16 objects; exploratory, no multiplicity adjustment.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=170);plt.close(fig)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',default='runs/operator-dev/surface-p2-study-v04/summary.json')
    p.add_argument('--output',default='runs/operator-dev/surface-static-media-v03/probe-comparison.png')
    main(p.parse_args())
