"""Report all held-out frames and explicitly selected success/failure close-ups."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/pidg-comparison-v01');a=p.parse_args();root=Path(a.root);selection=json.loads((root/'selection.json').read_text());base,ours=map(Path,selection['test_runs'])
    b=json.loads((base/'test/metrics.json').read_text());o=json.loads((ours/'test/metrics.json').read_text());assert [r['frame'] for r in b['rows']]==[r['frame'] for r in o['rows']]
    rows=[{'frame':br['frame'],'foreground_psnr_gain_db':orr['foreground_psnr_db']-br['foreground_psnr_db'],'psnr_gain_db':orr['psnr_db']-br['psnr_db'],'lpips_reduction':br['lpips_alex']-orr['lpips_alex']} for br,orr in zip(b['rows'],o['rows'])]
    positive=sorted([r for r in rows if r['foreground_psnr_gain_db']>0 and r['psnr_gain_db']>0],key=lambda r:r['foreground_psnr_gain_db'],reverse=True)
    worst=min(rows,key=lambda r:r['foreground_psnr_gain_db']);representative=sorted(rows,key=lambda r:r['foreground_psnr_gain_db'])[len(rows)//2]
    chosen=positive[:2] if positive else [representative]
    out=Path('output/figures')/('pidg-comparison' if root.name=='pidg-comparison-v01' else root.name);out.mkdir(parents=True,exist_ok=True)
    def figure(selected,name,title):
        fig,axes=plt.subplots(len(selected)*2,3,figsize=(12,4.5*len(selected)),squeeze=False,gridspec_kw={'hspace':.13,'wspace':.05});record=[]
        for i,r in enumerate(selected):
            bb=np.load(base/'test'/f'{r["frame"]}.npz');oo=np.load(ours/'test'/f'{r["frame"]}.npz');assert np.array_equal(bb['truth'],oo['truth']);gt=bb['truth'];bp=bb['prediction'];op=oo['prediction'];mask=bb['mask'];h,w=mask.shape
            high_b=base/'test-render-640'/f'{r["frame"]}.npz';high_o=ours/'test-render-640'/f'{r["frame"]}.npz'
            if high_b.exists() and high_o.exists():
                hb=np.load(high_b);ho=np.load(high_o);assert np.array_equal(hb['truth'],ho['truth']);gt,bp,op,mask=hb['truth'],hb['prediction'],ho['prediction'],hb['mask'];h,w=mask.shape
            diff=((bp-gt)**2-(op-gt)**2).mean(-1);size=w//8;stride=max(4,w//80);candidates=[]
            for y in range(0,h-size+1,stride):
                for x in range(0,w-size+1,stride):
                    if mask[y:y+size,x:x+size].mean()>.25:candidates.append((diff[y:y+size,x:x+size].mean(),x,y))
            value,x,y=max(candidates) if name!='failure' else min(candidates)
            images=[gt,bp,op]
            for j,(im,label) in enumerate(zip(images,['Ground truth','PIDG reproduction','PIDG + our material prior'])):
                ax=axes[2*i,j];ax.imshow(im);ax.add_patch(Rectangle((x,y),size,size,fill=False,ec='#28dfc0',lw=1.8));ax.axis('off');ax.set_title(label,fontsize=12,weight='bold')
                ax=axes[2*i+1,j];ax.imshow(im[y:y+size,x:x+size],interpolation='nearest');ax.axis('off')
                if j:
                    rmse=float(np.sqrt(((im[y:y+size,x:x+size]-gt[y:y+size,x:x+size])**2).mean()));ax.set_title(f'Matched crop RMSE: {rmse:.4f}',fontsize=10)
            axes[2*i,0].text(.025,.92,f'Frame {r["frame"]}',transform=axes[2*i,0].transAxes,fontsize=9,color='white')
            record.append(dict(r,crop_xywh=[x,y,size,size],render_width=int(gt.shape[1]),crop_rmse={name:float(np.sqrt(((im[y:y+size,x:x+size]-gt[y:y+size,x:x+size])**2).mean())) for name,im in [('pidg',bp),('hybrid',op)]}))
        fig.subplots_adjust(top=.93,bottom=.055)
        fig.suptitle(title,fontsize=16,weight='bold',y=.995);fig.text(.5,.005,'Same held-out frames, cameras and crops. Fit at 320px; rendered at 640px when available. Selected improvements; all-frame scores retained.',ha='center',fontsize=8)
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig);return record
    successes=figure(chosen,'selected-comparison','Held-out PIDG balls: selected local improvements' if positive else 'Held-out PIDG balls: representative comparison')
    failures=figure([worst],'failure','Held-out PIDG balls: least favorable frame')
    fig,axes=plt.subplots(len(successes),3,figsize=(10,3.2*len(successes)),squeeze=False)
    for i,rec in enumerate(successes):
        sub='test-render-640' if rec['render_width']==640 else 'test'
        bb=np.load(base/sub/f"{rec['frame']}.npz");oo=np.load(ours/sub/f"{rec['frame']}.npz");x,y,w,h=rec['crop_xywh'];gt=bb['truth'][y:y+h,x:x+w]
        axes[i,0].imshow(gt,interpolation='nearest');axes[i,0].set_title(f'GT crop | frame {rec["frame"]}')
        for j,(prediction,label) in enumerate([(bb['prediction'],'PIDG error'),(oo['prediction'],'Material-prior error')],1):
            error=np.sqrt(((prediction[y:y+h,x:x+w]-gt)**2).mean(-1));im=axes[i,j].imshow(error,cmap='magma',vmin=0,vmax=.05,interpolation='nearest');axes[i,j].set_title(label)
        for ax in axes[i]:ax.axis('off')
    fig.subplots_adjust(right=.88,hspace=.22,wspace=.05);cb=fig.colorbar(im,cax=fig.add_axes([.91,.2,.015,.6]),extend='max');cb.set_label('Pixel RGB RMSE (0–1); shared scale')
    fig.suptitle('Where reconstruction errors occur | matched selected crops',fontsize=13,weight='bold');fig.savefig(out/'error-maps.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True)
    frame_ids=[int(r['frame']) for r in rows]
    for ax,key,label in zip(axes,['foreground_psnr_gain_db','lpips_reduction'],['Foreground PSNR gain (dB)','LPIPS reduction']):
        values=np.array([r[key] for r in rows]);ax.bar(frame_ids,values,width=2.8,color=np.where(values>=0,'#008c88','#ba5149'));ax.axhline(0,color='#273f50',lw=.7);ax.axhline(values.mean(),color='#577689',ls='--',lw=1,label=f'Mean {values.mean():+.4f}');ax.set_ylabel(label);ax.legend(frameon=False,fontsize=9);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[-1].set_xlabel('Held-out source frame');fig.suptitle('All 38 held-out frames | positive values favor the material-prior integration',fontsize=12,weight='bold');fig.tight_layout();fig.savefig(out/'all-frame-gains.png',dpi=180);plt.close(fig)
    report={'scope':'New integration ablation: frozen shared material law inside PIDG reconstruction. Not the complete coupled mechanics/force-inference method.','dataset':'PIDG balls_reaction; public synthetic RGB sequence','training_frames':101,'validation_frames':11,'test_frames':len(rows),'baseline':b['metrics'],'hybrid':o['metrics'],'selection':selection,'foreground_wins':sum(r['foreground_psnr_gain_db']>0 for r in rows),'whole_image_wins':sum(r['psnr_gain_db']>0 for r in rows),'per_frame':rows,'selected_successes':successes,'least_favorable':failures,'physics_limit':'No ground-truth trajectories, forces, material parameters or physical timescale supplied here; Jacobian diagnostics do not establish physical accuracy.','protocol_limit':'Single-seed, 320x180 reconstruction pilot with runtime fixes shared across methods; not published native leaderboard numbers.'}
    if (base/'test-render-640/metrics.json').exists() and (ours/'test-render-640/metrics.json').exists():
        report['secondary_640px_rendering']={'selection_used':False,'baseline':json.loads((base/'test-render-640/metrics.json').read_text())['metrics'],'hybrid':json.loads((ours/'test-render-640/metrics.json').read_text())['metrics']}
    destination=Path('reports')/root.name;destination.mkdir(parents=True,exist_ok=True);(destination/'results.json').write_text(json.dumps(report,indent=2));(out/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['baseline','hybrid','foreground_wins','whole_image_wins']}),flush=True)
if __name__=='__main__':main()
