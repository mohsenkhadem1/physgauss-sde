"""Case-level effect sizes; display failures rather than silently discarding them."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/physgauss-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('reports/gaussian-boundary-flow-v02');data=json.loads((root/'compactexpansion-results.json').read_text())
    if not data['all_required_comparisons_evaluable']:raise ValueError('Wait for complete frozen comparison')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,6.8),layout='constrained')
    for col,dataset in enumerate(('SoftVTBench','PhysTwin')):
        rows=[r for r in data['cases']if r['dataset']==dataset]
        names=[r['case'].replace('-demo0','').replace('single_','').replace('_',' ').replace('-',' ')for r in rows]
        for idx,baseline in enumerate(('nearest_bank_pose_contact','prefix_velocity')):
            ax=axes[idx,col];diff=[]
            for row in rows:
                a,b=[row['methods'][k]for k in ('flow_pose_contact',baseline)]
                if not a['complete']or not b['complete']:raise ValueError('Failed case must be displayed explicitly before plotting')
                diff.append(a['coordinate_rmse_mm']-b['coordinate_rmse_mm'])
            y=np.arange(len(rows));colors=['#2878a0'if v<0 else'#bd4f42'for v in diff]
            ax.barh(y,diff,color=colors,height=.53);ax.axvline(0,color='#40444b',lw=1)
            ax.set_yticks(y,names);ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            span=max(abs(np.array(diff)).max(),.1);ax.set_xlim(min(0,min(diff))-.3*span,max(0,max(diff))+.3*span)
            for j,v in enumerate(diff):ax.text(v+(.03*span if v>=0 else-.03*span),j,f'{v:+.3f}',va='center',ha='left'if v>=0 else'right',fontsize=9)
            comparison='same-contact nearest bank'if idx==0 else'velocity extrapolation'
            ax.set_title(dataset+' · versus '+comparison,fontweight='bold',fontsize=11)
            ax.set_xlabel('Flow + transport/contact error minus baseline (mm)')
    fig.suptitle('Small gains over retrieval; real-video forecasts still trail velocity',fontsize=15,fontweight='bold')
    fig.supxlabel('Negative favors flow; positive favors baseline. Four exposed cases per dataset.\nMechanics and retrieval receive future tool commands; velocity uses only the observed prefix.',fontsize=9)
    fig.savefig(root/'expansion-effect-sizes.png',dpi=180);fig.savefig(root/'expansion-effect-sizes.svg');plt.close(fig)

if __name__=='__main__':main()
