"""Transparent first-pass plots: retain both wins and losses."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/real-transfer-v19'

def main():
    data=json.loads((REPORT/'exploratory-01-scores.json').read_text())
    assert data['complete']
    methods=['official_phystwin','position_ridge','neural','analytic']
    labels=['Released PhysTwin','Position regression','Ours: neural FEM adapter','Matched analytic FEM']
    colors=['#ef8c37','#8795a8','#009c9a','#737481']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),gridspec_kw={'width_ratios':[1,1,1.35]})
    for ax,segment,title in zip(axes[:2],['full_suffix','input_range_extrapolation'],['All future frames','Tool-position extrapolation']):
        values=[data['summary'][segment][m]['manual_track_mean_mm'] for m in methods]
        ax.barh(np.arange(4),values,color=colors,height=.6)
        ax.set_yticks(np.arange(4),labels if ax is axes[0] else ['']*4)
        ax.invert_yaxis();ax.set_xlim(0,53);ax.set_xlabel('Manual tracking error (mm) ↓');ax.set_title(title,weight='bold')
        for i,value in enumerate(values):ax.text(value+.6,i,f'{value:.1f}',va='center',fontsize=10)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    gains=[];names=[]
    for row in data['rows']:
        old=row['results']['analytic']['full_suffix']['manual_track_mean_mm']
        new=row['results']['neural']['full_suffix']['manual_track_mean_mm']
        gains.append(100*(old-new)/old);names.append(row['case'].replace('_',' '))
    ax=axes[2];ax.barh(np.arange(8),gains,color=['#009c9a' if g>=0 else '#ef8c37' for g in gains],height=.62)
    ax.set_yticks(np.arange(8),names,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='#333',lw=.7)
    ax.set_xlabel('Improvement over analytic FEM (%) →');ax.set_title('Every sequence retained',weight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('PhysTwin real-data first pass: improved analytic control, still behind the native baseline',weight='bold',fontsize=13)
    fig.text(.02,.015,'8 historically exposed sequences / 3 objects. 248/248 forecast frames complete. Extrapolation: 164 frames; not unseen-object OOD.',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.06,1,.93));out=REPORT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'phystwin-first-pass.png',dpi=180);fig.savefig(out/'phystwin-first-pass.pdf');plt.close(fig)

if __name__=='__main__':main()
