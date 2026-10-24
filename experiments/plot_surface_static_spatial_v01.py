"""Plot published scalar spatial diagnostics; media remains local."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/operator-dev/surface-static-media-v01/spatial-validation.png'));args=parser.parse_args()
    names=['clamped-surface-qualification-v01','clamped-surface-p2-qualification-v01']
    rows=[json.loads((Path('reports')/name/'summary.json').read_text())['per_case']for name in names]
    assert len(rows[0])==len(rows[1])==10
    assert [(r['case_index'],r['load'])for r in rows[0]]==[(r['case_index'],r['load'])for r in rows[1]]
    x=np.arange(10);labels=[f"C{r['case_index']} {'bend'if r['load']=='normal'else'axial'}"for r in rows[0]]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for group,label,marker in zip(rows,['Linear tetrahedra (T4)','Quadratic tetrahedra (T10)'],['o','s']):
        axes[0].plot(x,[r['spatial']['linear']['medium_fine_RMS_m']*1e6 for r in group],marker=marker,label=label)
    axes[0].axhline(10,color='black',linestyle='--',label='10 µm difference: middle-mesh 20 µm budget')
    axes[0].set(title='Difference between matching sizes: 1,125 → 7,533 unknowns',ylabel='Volume coordinate RMS difference (µm)',yscale='log')
    for law,marker in [('linear','s'),('nonlinear','^')]:
        axes[1].plot(x,[r['spatial'][law]['conditional_fine_indicator_m']*1e6 for r in rows[1]],marker=marker,label='Quadratic '+law)
    axes[1].axhline(20,color='black',linestyle='--',label='20 µm conditional fine-mesh budget')
    axes[1].set(title='Quadratic fine-mesh extrapolation indicator',ylabel='Conditional error indicator (µm)',ylim=(0,22))
    for ax in axes:
        ax.set_xticks(x,labels,rotation=40,ha='right');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same physical loads and supports; all ten cases retained',fontsize=14)
    fig.supxlabel('Conditional extrapolation assumes continuing observed order; dynamics and inference remain unqualified.',fontsize=10)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=140);print(str(args.output))


if __name__=='__main__':main()
