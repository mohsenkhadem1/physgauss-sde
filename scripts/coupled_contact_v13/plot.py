"""Static full-accounting scientific plots; display only, no model selection."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.coupled_contact_v13.campaign import REPORT

def main():
    d=json.loads((REPORT/'reserved-results.json').read_text());rows=d['cases']
    colors={'slide':'#337b96','edge_drop':'#b06146','curved':'#59875a','moving_sphere':'#856aab'}
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,label in zip(axes,['coordinate_rmse_m','shape_rmse_m'],['World-position coordinate RMSE (mm)','Rigid-aligned shape RMSE (mm)']):
        values=[]
        for scenario,color in colors.items():
            pairs=[(r['results']['rigid'][metric]*1000,r['results']['fem'][metric]*1000) for r in rows if r['scenario']==scenario and r['results'].get('rigid',{}).get('complete') and r['results'].get('fem',{}).get('complete')]
            if pairs:
                x,y=np.array(pairs).T;ax.scatter(x,y,c=color,label=scenario.replace('_',' '),s=38,edgecolor='white',linewidth=.4);values.extend(x);values.extend(y)
        limit=max(values,default=1)*1.08;ax.plot([0,limit],[0,limit],'--',color='gray',lw=1);ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Rigid + same contact',ylabel='Inferred FEM',title=label);ax.set_aspect('equal');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('Reserved cross-interaction cases: 8 objects, 4 interactions each')
    note='Below diagonal favors inferred FEM. All cases retained; frozen reference failed spatial-consistency audit.' if (REPORT/'mesh-development-diagnostic.json').exists() else 'Below diagonal favors inferred FEM. All declared cases and reference qualifications remain in the report.'
    fig.text(.5,.015,note,ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(REPORT/'reserved-comparison.png',dpi=180);plt.close(fig)
    p=json.loads((REPORT/'protocol.json').read_text());objects={o['id']:o for o in p['objects']};fig,axes=plt.subplots(1,2,figsize=(9,4))
    for oid,s in d['selections'].items():
        o=objects[oid];m=s['fem']
        if m['success']:axes[0].scatter(o['young']/1000,m['scale']/1000,c='#337b96');axes[1].scatter(o['friction'],m['friction'],c='#337b96')
    for ax,limit,label in zip(axes,[18,.8],['Young modulus (kPa)','Friction coefficient']):
        ax.plot([0,limit],[0,limit],'--',c='gray');ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Reference parameter',ylabel='Position-fitted parameter',title=label);ax.grid(alpha=.15)
    fig.suptitle('Parameter estimates are discrete fits, not calibrated uncertainty')
    fig.tight_layout();fig.savefig(REPORT/'reserved-parameter-estimates.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
