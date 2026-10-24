"""Plot scalar development diagnostics; generated media remains local."""
import json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parents[1]
    reports=[json.loads((root/f'reports/clamped-surface-p2-qualification-v0{v}/terminal-scalar-audit.json').read_text())for v in (2,3)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,load,title in zip(axes,('normal','paired_axial'),('Bending force reduced: 0.10 → 0.08 N','Axial force retained: 0.18 N per face')):
        for report,label,color,offset in zip(reports,('Previous test','Revised test'),('#bd5545','#207a8a'),(-.12,.12)):
            rows=sorted([r for r in report['per_case']if r['load']==load],key=lambda r:r['case_index'])
            values=np.array([r['spatial']['nonlinear']['medium_indicator_m']*1e6 for r in rows])
            ax.bar(np.arange(5)+offset,values,width=.24,label=label,color=color)
        ax.axhline(20,color='#333333',ls='--',lw=1.5,label='20 µm accuracy limit')
        ax.set_xticks(np.arange(5),['Corner 1','Corner 2','Corner 3','Corner 4','Center'])
        ax.set_title(title,fontsize=12);ax.set_xlabel('Five declared material/support cases')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Conservative target-mesh indicator (µm)');axes[0].set_ylim(0,25)
    axes[1].legend(frameon=False,loc='upper right',fontsize=9)
    fig.suptitle('Revised block experiment passes all central mesh checks',fontsize=15)
    fig.text(.5,.015,'Conditional spatial indicators from nonlinear static development tests; no calibration or probing claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93))
    out=root/'runs/operator-dev/surface-static-media-v01/central-repair.png';out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160);plt.close(fig);print(out)


if __name__=='__main__':main()
