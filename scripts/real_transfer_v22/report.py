"""Equal-case corrected-mask comparison and novel-camera inventory."""
import json,hashlib
from pathlib import Path
import numpy as np
from .campaign import ROOT,LABEL,CASES,write

def main():
    report=ROOT/'reports/real-transfer-v22'
    geometry=json.loads((report/'scores.json').read_text());assert geometry['complete']
    old=json.loads((ROOT/'reports/real-transfer-v20/refined-scores.json').read_text())
    rows=[];videos=[]
    for case in CASES:
        render=ROOT/'runs/real-transfer-v20/render-audit'/LABEL/case/'scores.json'
        d=json.loads(render.read_text());assert len(d['rows'])==36
        scores={}
        for method in ('neural','analytic','official_phystwin'):
            selected=[r for r in d['rows'] if r['method']==method and r['frame']>0];assert len(selected)==9
            scores[method]={k:float(np.mean([r[k] for r in selected])) for k in ('psnr_db','ssim','lpips','silhouette_iou')}
        rows.append(dict(case=case,render=scores,render_sha256=hashlib.sha256(render.read_bytes()).hexdigest()))
        manifest=ROOT/'runs/real-transfer-v22/novel-views'/case/'manifest.json'
        m=json.loads(manifest.read_text());videos.append(dict(case=case,movie=m['movie'],frame=m['frame'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    means={m:{k:float(np.mean([r['render'][m][k] for r in rows])) for k in rows[0]['render'][m]} for m in rows[0]['render']}
    transfer=json.loads((ROOT/'reports/real-transfer-v20/v22-corrected-frozen-transfer-scores.json').read_text())
    assert transfer['complete']
    write(report/'transfer-scores.json',transfer)
    projections=json.loads((report/'projected-landmarks.json').read_text())
    result=dict(projected_landmark_summary=projections['summary'],transfer_summary=transfer['summary'],complete=True,cases=8,geometry=geometry['summary'],previous_geometry=old['summary'],render=means,per_case=rows,novel_views=videos,
        interpretation='Correction-only comparison using the frozen v20 recipe; all eight sequences historically exposed. Novel-camera orbits have no RGB reference and cannot establish superiority.',
        mask_scope='Incoming transition fixes one-frame look-ahead in adapter masks. Offline upstream tracking has not been certified causal.')
    write(report/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    names=['Ours v20','Ours corrected','Analytic corrected','PhysTwin'];colors=['#9aaeb4','#217d91','#999999','#df8233']
    values=[old['summary']['neural']['tracking_mm']]+[geometry['summary'][m]['tracking_mm'] for m in ('neural','analytic','official_phystwin')]
    bars=axes[0].bar(names,values,color=colors);axes[0].bar_label(bars,fmt='%.2f',padding=3);axes[0].set_ylabel('Manual 3D tracking error (mm), lower is better');axes[0].set_ylim(0,max(values)*1.15)
    labels=['Ours corrected','Analytic corrected','PhysTwin'];values=[means[m]['psnr_db'] for m in ('neural','analytic','official_phystwin')]
    bars=axes[1].bar(labels,values,color=colors[1:]);axes[1].bar_label(bars,fmt='%.2f',padding=3);axes[1].set_ylabel('Recorded-camera PSNR (dB), higher is better');axes[1].set_ylim(0,max(values)*1.15)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('Mask correction: same recipe, eight PhysTwin sequences',fontweight='bold')
    fig.savefig(report/'comparison.png',dpi=160);fig.savefig(report/'comparison.svg');plt.close(fig)
    p=report/'comparison.svg';p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
    print(json.dumps(dict(geometry=result['geometry'],render=means),indent=2))

if __name__=='__main__':main()
