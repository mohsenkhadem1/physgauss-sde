"""Aggregate the paired oracle interventions and show fixed example panels."""
import json,hashlib
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .campaign import ROOT,BASE,CASES

def main():
    report=ROOT/'reports/render-diagnosis-v21';documents=[json.loads((BASE/c/'scores.json').read_text()) for c in CASES]
    methods=sorted({r['method'] for r in documents[0]['rows']});summary={};per_case=[]
    keys=['psnr_db','ssim','lpips','silhouette_iou','rgb_mse','silhouette_mismatch_sse_fraction']
    for d in documents:
        if len(d['rows'])!=84:raise ValueError('Incomplete diagnostic')
        row=dict(case=d['case'],methods={},support=d['support'])
        for method in methods:
            selected=[r for r in d['rows'] if r['method']==method and r['frame']>0]
            row['methods'][method]={k:float(np.mean([r[k] for r in selected])) for k in keys}
        row['initial_appearance_psnr_db']=float(np.mean([r['psnr_db'] for r in d['rows'] if r['method']=='oracle_rigid' and r['frame']==0]))
        per_case.append(row)
    for method in methods:summary[method]={k:float(np.mean([r['methods'][method][k] for r in per_case])) for k in keys}
    ours=summary['neural_rigid'];oracle=summary['oracle_rigid']
    result=dict(complete=True,cases=len(documents),images_scored=sum(len(d['rows']) for d in documents),summary=summary,per_case=per_case,
        initial_appearance_psnr_db=float(np.mean([r['initial_appearance_psnr_db'] for r in per_case])),
        oracle_intervention=dict(psnr_gain_db=oracle['psnr_db']-ours['psnr_db'],
            lpips_reduction_percent=100*(1-oracle['lpips']/ours['lpips']),rgb_mse_reduction_percent=100*(1-oracle['rgb_mse']/ours['rgb_mse']),
            cases_improving_psnr_over_neural=sum(r['methods']['oracle_rigid']['psnr_db']>r['methods']['neural_rigid']['psnr_db'] for r in per_case)),
        mean_oracle_affine_clipped_anchor_fraction=float(np.mean([np.mean([r['clipped_anchor_fraction'] for r in d['transport'] if r['frame']>0 and r['method']=='oracle_affine_cov']) for d in documents])),
        conclusions=['Motion/shape substitution produces a large paired image improvement; appearance weights are unchanged.',
                     'The tested local affine/covariance transport does not improve mean quality; this does not rule out better deformation mappings.',
                     'Recorded tracks are incomplete and noisy: this oracle proxy is not dense ground truth or an achievable upper bound.',
                     'Diagnostic uses reference visibility for matched anchors; it is not a new deployable forecasting result.',
                     'Legacy training used outgoing motion-validity flags for positions. Corrected incoming flags are used here; old fits need separate rebenchmarking before strict temporal-isolation claims.'],
        report_sha256={c:hashlib.sha256((BASE/c/'scores.json').read_bytes()).hexdigest() for c in CASES})
    (report/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.5),layout='constrained')
    labels=['Our motion','PhysTwin motion','Recorded motion','Recorded +\naffine footprints']
    selected=['neural_rigid','native_rigid','oracle_rigid','oracle_affine_cov'];colors=['#247c91','#df8233','#419b6d','#a3b8a8']
    for ax,key,title in zip(axes[:2],['psnr_db','lpips'],['Image PSNR ↑','Perceptual error LPIPS ↓']):
        values=[summary[m][key] for m in selected];bars=ax.bar(np.arange(4),values,color=colors)
        ax.set_xticks(np.arange(4),labels,rotation=15,ha='right');ax.bar_label(bars,fmt='%.2f' if key=='psnr_db' else '%.3f',padding=3)
        ax.set_ylim(0,max(values)*1.18);ax.set_title(title,fontweight='bold');ax.grid(axis='y',alpha=.15)
    ax=axes[2];x=np.arange(len(CASES));gains=[r['methods']['oracle_rigid']['psnr_db']-r['methods']['neural_rigid']['psnr_db'] for r in per_case]
    ax.barh(x,gains,color='#419b6d');ax.set_yticks(x,[c.replace('_',' ') for c in CASES],fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('PSNR gain from recorded motion (dB)');ax.set_title('Improvement in every sequence',fontweight='bold')
    fig.suptitle('Rendering bottleneck: controlled motion and Gaussian-transport interventions',fontweight='bold',fontsize=14)
    figures=report/'figures';figures.mkdir(exist_ok=True);fig.savefig(figures/'oracle-diagnosis.png',dpi=180);fig.savefig(figures/'oracle-diagnosis.svg');plt.close(fig)
    svg=figures/'oracle-diagnosis.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    # Fixed examples chosen in the protocol before scores: last frame, camera 0.
    examples=['single_lift_sloth','double_stretch_zebra'];fig,axes=plt.subplots(2,4,figsize=(12,6.5),layout='constrained')
    columns=['Recorded image','Our predicted motion','Recorded-motion proxy','Proxy + affine footprints']
    for row,case in enumerate(examples):
        document=next(d for d in documents if d['case']==case);frame=max(document['frames']);camera=0
        local=ROOT/'runs/real-transfer-v20/render-assets/data/different_types'/case
        labels=json.loads((local/'mask/mask_info_0.json').read_text());oid=next(k for k,v in labels.items() if v!='hand')
        actual=cv2.cvtColor(cv2.imread(str(local/f'color/0/{frame}.png')),cv2.COLOR_BGR2RGB)
        mask=cv2.imread(str(local/f'mask/0/{oid}/{frame}.png'),0)>0;actual[~mask]=0
        yy,xx=np.where(mask);x0=max(0,int(xx.min())-35);x1=min(mask.shape[1],int(xx.max())+36)
        y0=max(0,int(yy.min())-35);y1=min(mask.shape[0],int(yy.max())+36)
        images=[actual]+[cv2.cvtColor(cv2.imread(str(BASE/case/f'{m}-camera0-frame{frame}.png')),cv2.COLOR_BGR2RGB) for m in ('neural_rigid','oracle_rigid','oracle_affine_cov')]
        for col,image in enumerate(images):
            axes[row,col].imshow(image[y0:y1,x0:x1]);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(columns[col],fontsize=10)
            if col==0:axes[row,col].set_ylabel(case.replace('_',' ')+'\nframe '+str(frame),fontsize=9)
    fig.suptitle('Same Gaussian appearance; only motion/transport changes',fontweight='bold')
    fig.savefig(figures/'oracle-visual-comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summary=summary,intervention=result['oracle_intervention']),indent=2))

if __name__=='__main__':main()
