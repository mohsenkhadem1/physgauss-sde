"""Compact report with all cases and unfavorable results retained."""
import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .development_run import ROOT,BASE,write

def main():
    report=ROOT/'reports/real-transfer-v20';read=lambda name:json.loads((report/name).read_text())
    development=read('development-scores.json');refined=read('refined-scores.json');transfer=read('refined-frozen-transfer-scores.json')
    render=read('render-scores.json')
    if not refined['complete'] or not render['complete'] or not transfer['complete']:raise ValueError('Refusing an incomplete headline comparison')
    methods=['neural','analytic','official_phystwin'];colors=['#147d92','#81909e','#d97732']
    labels=['Neural FEM (240 nodes)','Analytic FEM (240 nodes)','PhysTwin']
    appearance={}
    for method in methods:
        per_case=[]
        for case in render['cases']:
            rows=[r for r in case['rows'] if r['method']==method and r['frame']>0]
            per_case.append(dict(case=case['case'],**{k:float(np.mean([r[k] for r in rows])) for k in ('psnr_db','ssim','lpips','silhouette_iou')}))
        appearance[method]=dict(per_case=per_case,**{k:float(np.mean([r[k] for r in per_case])) for k in ('psnr_db','ssim','lpips','silhouette_iou')})
    old=json.loads((ROOT/'reports/real-transfer-v19/exploratory-01-scores.json').read_text())['summary']['full_suffix']['neural']['manual_track_mean_mm']
    summary=dict(standard_prediction=refined['summary'],rendering=appearance,
        frozen_transfer=transfer['summary'],old_neural_tracking_mm=old,
        standard_tracking_improvement_percent=100*(1-refined['summary']['neural']['tracking_mm']/old),
        registered_pilot_tracking_gain_over_native_percent=100*(1-transfer['summary']['initial_geometry_gate_passed']['neural']['tracking_mm']/transfer['summary']['initial_geometry_gate_passed']['native_frozen']['tracking_mm']),
        limitations=['All eight sequences previously exposed; not a fresh sealed benchmark.',
                     'Rendering uses common released appearance and common skinning; three sampled forecast frames, three initial appearance cameras.',
                     'Only two of five transfer pairs pass initial geometry gate, and only one of those two favors our tracking.',
                     'Transfer baseline preserves native source object springs but adapts new controller springs; not an official author result.',
                     'No measured force accuracy, calibrated uncertainty or active-probing improvement demonstrated.'])
    transfer_images=report/'transfer-render-summary.json'
    if transfer_images.exists():
        images=json.loads(transfer_images.read_text())
        row=next(r for r in transfer['rows'] if r['target']=='single_push_sloth')
        ours=row['results']['neural']['whole_interaction'];native=row['results']['native_frozen']['whole_interaction']
        summary['illustrative_transfer_case']=dict(source='single_lift_sloth',target='single_push_sloth',
            tracking_gain_percent=100*(1-ours['tracking_mm']/native['tracking_mm']),neural=ours,native_frozen=native,
            rendering=next(r for r in images['rows'] if r['target']=='single_push_sloth')['metrics'],
            scope='One illustrative recorded interaction-transfer diagnostic, not proof of calibrated physics. Initial shape differs between recordings.')
    write(report/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.5),layout='constrained')
    ax=axes[0,0];keys=list(development['aggregate']);names=['Control','Longer fit','240 nodes','Narrow grip','Wide grip','No floor']
    vals=[development['aggregate'][k]['manual_track_mean_mm'] for k in keys]
    bars=ax.barh(names,vals,color=['#147d92' if k=='mesh240' else '#c1cbd3' for k in keys]);ax.invert_yaxis()
    ax.bar_label(bars,fmt='%.1f',padding=4);ax.set_xlim(0,max(vals)*1.16);ax.set_xlabel('Development tracking error (mm), lower is better')
    ax.set_title('A  Mesh refinement helped; extra optimization did not',loc='left',fontweight='bold')
    ax=axes[0,1];values=[old]+[refined['summary'][m]['tracking_mm'] for m in methods]
    bars=ax.bar(['Our old\n120 nodes','Our refined\n240 nodes','Matched\nanalytic','Released\nPhysTwin'],values,color=['#a2ced4',*colors])
    ax.bar_label(bars,fmt='%.1f',padding=3);ax.set_ylim(0,max(values)*1.18);ax.set_ylabel('Future tracking error (mm)')
    ax.set_title('B  Standard forecast: all eight exposed sequences',loc='left',fontweight='bold')
    ax=axes[1,0];rows=transfer['rows'];x=np.arange(len(rows));width=.24
    for j,(method,color,label) in enumerate(zip(['neural','analytic','native_frozen'],colors,labels)):
        values=[r['results'][method]['whole_interaction']['tracking_mm'] for r in rows]
        bars=ax.bar(x+(j-1)*width,values,width,color=color,label=label.replace('PhysTwin','Adapted frozen PhysTwin'))
        for bar,row in zip(bars,rows):
            if not row['registration']['initial_geometry_gate']:bar.set_alpha(.4);bar.set_hatch('///')
    names=['Sloth\n2-hand lift','Sloth\nstretch','Sloth\npush *','Zebra\n2-hand lift *','Zebra\nstretch']
    ax.set_xticks(x,names);ax.set_ylabel('Transferred tracking error (mm)');ax.legend(fontsize=8)
    ax.set_title('C  Frozen material, new recorded interactions',loc='left',fontweight='bold')
    ax.text(0,-.2,'* Initial geometry gate passed; other pairs remain diagnostics.',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1]
    for method,color,label in zip(methods,colors,labels):
        x=appearance[method]['lpips'];y=refined['summary'][method]['tracking_mm']
        ax.scatter(x,y,s=130,color=color,label=label,zorder=3)
        offset=(6,-16) if method=='neural' else (6,7)
        ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
    ax.margins(x=.5,y=.3);ax.grid(alpha=.18);ax.set_xlabel('Forecast image LPIPS (lower is better)');ax.set_ylabel('Future tracking error (mm)')
    ax.set_title('D  Appearance and physical prediction measured separately',loc='left',fontweight='bold')
    fig.suptitle('PhysTwin audit: development, forecasting, transfer and rendering',fontsize=16,fontweight='bold')
    figures=report/'figures';figures.mkdir(exist_ok=True)
    fig.savefig(figures/'benchmark-audit.png',dpi=190);fig.savefig(figures/'benchmark-audit.svg');plt.close(fig)
    svg=figures/'benchmark-audit.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rendering'},indent=2))

if __name__=='__main__':main()
