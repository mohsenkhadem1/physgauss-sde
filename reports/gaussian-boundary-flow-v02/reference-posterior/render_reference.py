"""Summarize original and separately frozen supplemental reference evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.gaussian_boundary_flow_v02.reference_posterior import OUT,REPORT,LOW,HIGH


def main():
    initial=json.loads((REPORT/'results.json').read_text());supp=json.loads((REPORT/'world2-fixed-spacing.json').read_text())
    lower=np.array([float(LOW[1])+.8,float(LOW[3])]);width=np.array([float(HIGH[1]-LOW[1]),float(HIGH[3]-LOW[3])])
    figures=[];summary=[]
    for original in initial['worlds']:
        i=original['world_index'];result=supp if i==2 else original
        record={'world':i,'original_protocol_converged':original['converged'],'accepted_after_supplement':result['converged'],
                'correlation':result.get('posterior_correlation'),'truth_in_conditional_90_intervals':result.get('truth_in_conditional_90_intervals')}
        if result['converged']:
            p=result['posterior'];q=np.array(p['quantiles_q']);truth=np.array(result['truth_q'])
            record['support_stiffness_quantiles_N_per_m3']=np.exp(lower[0]+width[0]*q[0]).tolist()
            record['support_x_quantiles_m']=(lower[1]+width[1]*q[1]).tolist()
            record['truth_physical']=[float(np.exp(lower[0]+width[0]*truth[0])),float(lower[1]+width[1]*truth[1])]
            directory=OUT/'world-2-fixed-spacing' if i==2 else OUT/f'world-{i}'
            path=directory/f"quadrature-L{p['extent']:g}-n{p['n']}.npz"
            data=np.load(path);points=data['q'].reshape(p['n'],p['n'],2);ll=data['log_density'].reshape(p['n'],p['n'])
            figures.append((record,points,np.exp(ll-np.nanmax(ll))))
        summary.append(record)
    output={'scope':'Two-parameter conditional reference, four other true parameters fixed; not an unconditional learned posterior',
            'initial_converged':sum(w['converged'] for w in initial['worlds']),
            'original_failure_preserved':not initial['worlds'][2]['converged'],
            'supplemental_world2_converged':supp['converged'],'worlds':summary,
            'total_numerical_runtime_seconds':initial['elapsed_seconds']+supp['elapsed_seconds'],
            'calibration_claim':False,'global_mode_exclusion_proved':False}
    (REPORT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,(record,points,density) in zip(axes.flat,figures):
        K=np.exp(lower[0]+width[0]*points[...,0])/1000;x=(lower[1]+width[1]*points[...,1])*1000
        ax.contourf(K,x,density,levels=np.linspace(.001,1,25),cmap='Blues')
        ax.contour(K,x,density,levels=[.05,.32,.61],colors='#334155',linewidths=.6)
        truth=record['truth_physical'];ax.plot(truth[0]/1000,truth[1]*1000,'*',color='#f59e0b',markeredgecolor='#92400e',markersize=12,label='True parameters')
        ax.set(xlabel='Support stiffness (kN/m³)',ylabel='Support x-location (mm)',title=f"World {record['world']} | correlation {record['correlation']:+.3f}"+(' | refined diagnostic' if record['world']==2 else ''))
        ax.ticklabel_format(useOffset=False);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Conditional support posterior: strong coupled uncertainty\nOther four parameters fixed to truth; contours show relative density',fontsize=13)
    fig.savefig(REPORT/'posterior-geometry.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
